"""Unit tests for SQLite cache."""
import pytest
import tempfile
import os

from src.harness.contracts import LLMResponse, ValidatorResult
from src.harness.cache import (
    CacheEntry,
    CacheManager,
    SQLiteCache,
    RELIABILITY_THRESHOLD,
    create_cache_manager,
)
from src.harness.orchestrator import OrchestrationResult


def test_sqlite_cache_creation():
    """SQLiteCache can be created and initializes DB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = SQLiteCache(db_path)

        assert cache.db_path.exists()
        stats = cache.get_stats()
        assert stats["total_entries"] == 0


def test_sqlite_cache_key_deterministic():
    """Cache key is deterministic for same inputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = SQLiteCache(db_path)

        key1 = cache._compute_key("prompt", "model", "provider", "valhash")
        key2 = cache._compute_key("prompt", "model", "provider", "valhash")
        key3 = cache._compute_key("different", "model", "provider", "valhash")

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 32  # SHA256 truncated to 32 chars


def test_sqlite_cache_put_and_get():
    """Cache can store and retrieve responses above threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = SQLiteCache(db_path)

        response = LLMResponse(
            content='{"result": "ok"}',
            model="test-model",
            provider="test",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=100.0,
        )

        # Put with high reliability
        cached = cache.put(
            prompt="test prompt",
            model="test-model",
            provider="test",
            validator_set_hash="abc123",
            response=response,
            reliability_score=0.9,
        )
        assert cached is True

        # Get should return entry
        entry = cache.get("test prompt", "test-model", "test", "abc123")
        assert entry is not None
        assert entry.response.content == '{"result": "ok"}'
        assert entry.reliability_score == 0.9
        assert entry.access_count == 1  # Incremented on get


def test_sqlite_cache_rejects_low_reliability():
    """Cache rejects responses below reliability threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = SQLiteCache(db_path)

        response = LLMResponse(
            content="bad response",
            model="test-model",
            provider="test",
        )

        cached = cache.put(
            prompt="test",
            model="test-model",
            provider="test",
            validator_set_hash="abc123",
            response=response,
            reliability_score=0.5,  # Below 0.80 threshold
        )
        assert cached is False

        # Should not be retrievable
        entry = cache.get("test", "test-model", "test", "abc123")
        assert entry is None


def test_sqlite_cache_invalidates_by_validator_hash():
    """Cache invalidation by validator set hash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = SQLiteCache(db_path)

        response = LLMResponse(content="ok", model="m1", provider="p1")

        # Add entries with different validator hashes
        cache.put("p1", "m1", "p1", "hash1", response, 0.9)
        cache.put("p2", "m1", "p1", "hash1", response, 0.9)
        cache.put("p3", "m1", "p1", "hash2", response, 0.9)

        # Invalidate hash1
        deleted = cache.invalidate_by_validator_hash("hash1")
        assert deleted == 2

        # hash2 should still exist
        entry = cache.get("p3", "m1", "p1", "hash2")
        assert entry is not None

        # hash1 entries gone
        assert cache.get("p1", "m1", "p1", "hash1") is None
        assert cache.get("p2", "m1", "p1", "hash1") is None


def test_sqlite_cache_invalidates_by_model():
    """Cache invalidation by model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = SQLiteCache(db_path)

        response = LLMResponse(content="ok", model="m1", provider="p1")

        cache.put("p1", "model1", "provider1", "hash1", response, 0.9)
        cache.put("p2", "model2", "provider1", "hash1", response, 0.9)

        # Invalidate model1
        deleted = cache.invalidate_by_model("model1", "provider1")
        assert deleted == 1

        assert cache.get("p1", "model1", "provider1", "hash1") is None
        assert cache.get("p2", "model2", "provider1", "hash1") is not None


def test_sqlite_cache_stats():
    """Cache statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = SQLiteCache(db_path)

        response = LLMResponse(content="ok", model="m1", provider="p1")

        cache.put("p1", "m1", "p1", "h1", response, 0.9)
        cache.put("p2", "m1", "p1", "h1", response, 0.7)  # Below threshold, rejected
        cache.put("p3", "m1", "p1", "h2", response, 0.95)

        stats = cache.get_stats()

        # Only responses meeting threshold (>= 0.80) are inserted into cache
        assert stats["total_entries"] == 2
        assert stats["entries_above_threshold"] == 2
        assert stats["average_reliability"] > 0.8


def test_sqlite_cache_clear():
    """Cache clear removes all entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = SQLiteCache(db_path)

        response = LLMResponse(content="ok", model="m1", provider="p1")
        cache.put("p1", "m1", "p1", "h1", response, 0.9)

        deleted = cache.clear()
        assert deleted == 1

        stats = cache.get_stats()
        assert stats["total_entries"] == 0


def test_cache_manager_try_get_cached():
    """CacheManager try_get_cached."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        manager = CacheManager(SQLiteCache(db_path))

        response = LLMResponse(content="cached", model="m1", provider="p1")
        manager.cache.put("prompt", "m1", "p1", "hash", response, 0.9)

        result = manager.try_get_cached("prompt", "m1", "p1", "hash")
        assert result is not None
        assert result.content == "cached"

        # Different validator hash - should miss
        result = manager.try_get_cached("prompt", "m1", "p1", "different")
        assert result is None


def test_cache_manager_try_cache_result():
    """CacheManager try_cache_result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        manager = CacheManager(SQLiteCache(db_path))

        result = OrchestrationResult(
            success=True,
            final_output='{"status": "ok"}',
            final_validator_results=[],
            final_semantic_result=None,
            final_critic_feedback=None,
            retry_attempts=[],
            total_attempts=1,
            total_latency_ms=100.0,
            reliability_score=0.9,
        )

        cached = manager.try_cache_result(
            prompt="test",
            model="m1",
            provider="p1",
            validator_set_hash="hash",
            result=result,
        )
        assert cached is True

        # Verify cached
        entry = manager.cache.get("test", "m1", "p1", "hash")
        assert entry is not None
        assert entry.response.content == '{"status": "ok"}'


def test_cache_manager_try_cache_result_low_reliability():
    """CacheManager rejects low reliability results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        manager = CacheManager(SQLiteCache(db_path))

        result = OrchestrationResult(
            success=True,
            final_output="ok",
            final_validator_results=[],
            final_semantic_result=None,
            final_critic_feedback=None,
            retry_attempts=[],
            total_attempts=1,
            total_latency_ms=100.0,
            reliability_score=0.5,  # Below threshold
        )

        cached = manager.try_cache_result(
            prompt="test",
            model="m1",
            provider="p1",
            validator_set_hash="hash",
            result=result,
        )
        assert cached is False


def test_cache_manager_invalidate_task():
    """CacheManager invalidate_task."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        manager = CacheManager(SQLiteCache(db_path))

        response = LLMResponse(content="ok", model="m1", provider="p1")
        manager.cache.put("p1", "m1", "p1", "hash1", response, 0.9)
        manager.cache.put("p2", "m1", "p1", "hash2", response, 0.9)

        deleted = manager.invalidate_task("hash1")
        assert deleted == 1
        assert manager.cache.get("p1", "m1", "p1", "hash1") is None
        assert manager.cache.get("p2", "m1", "p1", "hash2") is not None


def test_cache_manager_stats():
    """CacheManager get_stats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        manager = CacheManager(SQLiteCache(db_path))

        stats = manager.get_stats()
        assert "total_entries" in stats
        assert "entries_above_threshold" in stats
        assert "average_reliability" in stats
        assert "size_bytes" in stats
        assert "size_mb" in stats