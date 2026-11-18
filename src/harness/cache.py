"""Selective SQLite Cache for Agentic Harness v1.2.

Implements high-reliability caching (Reliability >= 0.80) with deterministic
SHA256 keys incorporating provider, model, prompt, and active validator set.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .contracts import LLMResponse, ValidatorResult
from .orchestrator import OrchestrationResult


DEFAULT_DB_PATH = ".cache/agentic_harness.db"
RELIABILITY_THRESHOLD = 0.80


@dataclass
class CacheEntry:
    """Represents a cached response."""
    key: str
    prompt: str
    model: str
    provider: str
    validator_set_hash: str
    response: LLMResponse
    reliability_score: float
    created_at: float
    access_count: int = 0
    last_accessed: float = 0.0


class SQLiteCache:
    """SQLite-based selective cache for high-reliability LLM responses.

    Only caches responses with composite Reliability Score >= 0.80.
    Keys deterministically incorporate SHA256 of provider + model + prompt + validator_set_hash.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self):
        """Context manager that yields a connection and guarantees it is closed."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    validator_set_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    reliability_score REAL NOT NULL,
                    created_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_validator_hash
                ON cache_entries(validator_set_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reliability
                ON cache_entries(reliability_score)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_provider
                ON cache_entries(model, provider)
            """)
            conn.commit()

    def close(self) -> None:
        """Close the cache (no-op for SQLite with per-operation connections)."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _compute_key(
        self,
        prompt: str,
        model: str,
        provider: str,
        validator_set_hash: str,
    ) -> str:
        """Compute deterministic cache key."""
        content = f"{provider}|{model}|{prompt}|{validator_set_hash}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _response_to_json(self, response: LLMResponse) -> str:
        """Serialize LLMResponse to JSON."""
        return json.dumps({
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "latency_ms": response.latency_ms,
            "raw_response": response.raw_response,
        })

    def _json_to_response(self, json_str: str) -> LLMResponse:
        """Deserialize LLMResponse from JSON."""
        data = json.loads(json_str)
        return LLMResponse(**data)

    def get(
        self,
        prompt: str,
        model: str,
        provider: str,
        validator_set_hash: str,
    ) -> Optional[CacheEntry]:
        """Retrieve cached response if exists and meets reliability threshold."""
        key = self._compute_key(prompt, model, provider, validator_set_hash)

        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM cache_entries WHERE key = ? AND reliability_score >= ?",
                (key, RELIABILITY_THRESHOLD),
            )
            row = cursor.fetchone()

        if row:
            # Update access count first
            self._update_access(key)
            # Then fetch updated row
            with self._connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM cache_entries WHERE key = ?",
                    (key,),
                )
                row = cursor.fetchone()
            
            entry = CacheEntry(
                key=row["key"],
                prompt=row["prompt"],
                model=row["model"],
                provider=row["provider"],
                validator_set_hash=row["validator_set_hash"],
                response=self._json_to_response(row["response_json"]),
                reliability_score=row["reliability_score"],
                created_at=row["created_at"],
                access_count=row["access_count"],
                last_accessed=row["last_accessed"],
            )
            return entry

        return None

    def _update_access(self, key: str) -> None:
        """Update access count and last accessed time."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE cache_entries SET access_count = access_count + 1, last_accessed = ? WHERE key = ?",
                (time.time(), key),
            )
            conn.commit()

    def put(
        self,
        prompt: str,
        model: str,
        provider: str,
        validator_set_hash: str,
        response: LLMResponse,
        reliability_score: float,
    ) -> bool:
        """Store response in cache if reliability meets threshold.

        Returns True if cached, False if reliability too low.
        """
        if reliability_score < RELIABILITY_THRESHOLD:
            return False

        key = self._compute_key(prompt, model, provider, validator_set_hash)
        now = time.time()

        with self._connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cache_entries
                   (key, prompt, model, provider, validator_set_hash, response_json,
                    reliability_score, created_at, access_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                (key, prompt, model, provider, validator_set_hash,
                 self._response_to_json(response), reliability_score, now, now),
            )
            conn.commit()

        return True

    def invalidate_by_validator_hash(self, validator_set_hash: str) -> int:
        """Invalidate all cache entries for a specific validator set.

        Returns number of entries deleted.
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM cache_entries WHERE validator_set_hash = ?",
                (validator_set_hash,),
            )
            conn.commit()
            return cursor.rowcount

    def invalidate_by_model(self, model: str, provider: Optional[str] = None) -> int:
        """Invalidate cache entries for a specific model/provider."""
        with self._connection() as conn:
            if provider:
                cursor = conn.execute(
                    "DELETE FROM cache_entries WHERE model = ? AND provider = ?",
                    (model, provider),
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM cache_entries WHERE model = ?",
                    (model,),
                )
            conn.commit()
            return cursor.rowcount

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row

            # Total entries
            total = conn.execute("SELECT COUNT(*) as count FROM cache_entries").fetchone()["count"]

            # Entries above threshold
            above_threshold = conn.execute(
                "SELECT COUNT(*) as count FROM cache_entries WHERE reliability_score >= ?",
                (RELIABILITY_THRESHOLD,),
            ).fetchone()["count"]

            # Average reliability
            avg_rel = conn.execute(
                "SELECT AVG(reliability_score) as avg FROM cache_entries"
            ).fetchone()["avg"]

            # Total size (approximate)
            size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0

            return {
                "total_entries": total,
                "entries_above_threshold": above_threshold,
                "average_reliability": round(avg_rel, 4) if avg_rel else 0.0,
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "db_path": str(self.db_path),
            }

    def clear(self) -> int:
        """Clear all cache entries. Returns number deleted."""
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM cache_entries")
            conn.commit()
            return cursor.rowcount


class CacheManager:
    """High-level cache manager integrating with orchestration pipeline."""

    def __init__(
        self,
        cache: Optional[SQLiteCache] = None,
        reliability_threshold: float = RELIABILITY_THRESHOLD,
    ):
        self.cache = cache or SQLiteCache()
        self.reliability_threshold = reliability_threshold

    def try_get_cached(
        self,
        prompt: str,
        model: str,
        provider: str,
        validator_set_hash: str,
    ) -> Optional[LLMResponse]:
        """Try to get cached response. Returns None if not found or below threshold."""
        entry = self.cache.get(prompt, model, provider, validator_set_hash)
        if entry:
            return entry.response
        return None

    def try_cache_result(
        self,
        prompt: str,
        model: str,
        provider: str,
        validator_set_hash: str,
        result: OrchestrationResult,
    ) -> bool:
        """Attempt to cache orchestration result if reliability meets threshold."""
        if result.reliability_score >= self.reliability_threshold and result.success:
            return self.cache.put(
                prompt=prompt,
                model=model,
                provider=provider,
                validator_set_hash=validator_set_hash,
                response=LLMResponse(
                    content=result.final_output,
                    model=model,
                    provider=provider,
                    prompt_tokens=0,  # Not tracked in final output
                    completion_tokens=0,
                    latency_ms=result.total_latency_ms,
                ),
                reliability_score=result.reliability_score,
            )
        return False

    def invalidate_task(self, validator_set_hash: str) -> int:
        """Invalidate cache for a specific task (validator set)."""
        return self.cache.invalidate_by_validator_hash(validator_set_hash)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


def create_cache_manager(db_path: str = DEFAULT_DB_PATH) -> CacheManager:
    """Factory function to create a CacheManager with default settings."""
    return CacheManager(SQLiteCache(db_path))
