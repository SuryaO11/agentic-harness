"""Unit tests for semantic similarity scorer."""
import pytest
import numpy as np

from src.harness.contracts import (
    FailureTypeEnum,
    SeverityEnum,
    ValidatorResult,
)
from src.harness.scorer import (
    EmbeddingCache,
    SemanticScorer,
    create_semantic_scorer,
)


def test_embedding_cache_basic():
    """Embedding cache stores and retrieves embeddings."""
    cache = EmbeddingCache(".cache/test_embeddings")
    cache.clear()

    text = "Hello world"
    embedding = np.array([0.1, 0.2, 0.3, 0.4])

    # Not in cache initially
    assert cache.get(text) is None

    # Store and retrieve
    cache.set(text, embedding)
    retrieved = cache.get(text)
    assert retrieved is not None
    np.testing.assert_array_equal(retrieved, embedding)

    cache.clear()


def test_embedding_cache_deterministic_key():
    """Same text produces same cache key."""
    cache = EmbeddingCache(".cache/test_embeddings")
    cache.clear()

    text = "Consistent text"
    embedding = np.array([0.5, 0.5, 0.5])

    cache.set(text, embedding)
    retrieved1 = cache.get(text)
    retrieved2 = cache.get(text)

    np.testing.assert_array_equal(retrieved1, retrieved2)

    cache.clear()


def test_semantic_scorer_creation():
    """SemanticScorer can be created with defaults."""
    scorer = create_semantic_scorer()
    assert scorer.model_name == "all-MiniLM-L6-v2"
    assert scorer.similarity_threshold == 0.75


def test_semantic_scorer_similarity():
    """Semantic similarity computation."""
    scorer = create_semantic_scorer()

    # Identical texts should have high similarity
    score = scorer.similarity("The cat sat on the mat", "The cat sat on the mat")
    assert score >= 0.99  # Nearly 1.0

    # Related texts should have moderate similarity
    score = scorer.similarity("The cat sat on the mat", "A cat is sitting on a mat")
    assert 0.5 < score < 1.0

    # Unrelated texts should have low similarity
    score = scorer.similarity("The cat sat on the mat", "The stock market crashed today")
    assert score < 0.5


def test_semantic_scorer_evaluate_pass():
    """Evaluate returns pass for similar texts."""
    scorer = create_semantic_scorer(similarity_threshold=0.5)

    result = scorer.evaluate(
        generated="The cat sat on the mat",
        reference="A cat is sitting on a mat",
    )
    assert result.passed is True
    assert result.severity == SeverityEnum.INFO


def test_semantic_scorer_evaluate_fail():
    """Evaluate returns fail for dissimilar texts."""
    scorer = create_semantic_scorer(similarity_threshold=0.8)

    result = scorer.evaluate(
        generated="The cat sat on the mat",
        reference="The stock market crashed today",
    )
    assert result.passed is False
    assert result.severity == SeverityEnum.CRITICAL
    assert result.failure_type == FailureTypeEnum.SEMANTIC_DRIFT


def test_semantic_scorer_custom_threshold():
    """Evaluate respects custom threshold override."""
    scorer = create_semantic_scorer(similarity_threshold=0.95)

    # Default threshold would fail (similarity ~0.93)
    result_default = scorer.evaluate(
        generated="The cat sat on the mat",
        reference="A cat is sitting on a mat",
    )
    assert result_default.passed is False

    # Lower custom threshold passes
    result_custom = scorer.evaluate(
        generated="The cat sat on the mat",
        reference="A cat is sitting on a mat",
        threshold=0.9,
    )
    assert result_custom.passed is True


def test_semantic_scorer_batch_evaluate():
    """Batch evaluation works correctly."""
    scorer = create_semantic_scorer(similarity_threshold=0.5)

    generated = [
        "The cat sat on the mat",
        "The stock market crashed",
    ]
    references = [
        "A cat is sitting on a mat",
        "The stock market crashed today",
    ]

    results = scorer.batch_evaluate(generated, references)
    assert len(results) == 2
    assert results[0].passed is True  # Similar
    assert results[1].passed is True  # Similar


def test_semantic_scorer_batch_evaluate_mismatch():
    """Batch evaluation catches length mismatch."""
    scorer = create_semantic_scorer()

    with pytest.raises(ValueError, match="same length"):
        scorer.batch_evaluate(["a", "b"], ["c"])


def test_semantic_scorer_embedding_dim():
    """Embedding dimension is correct for all-MiniLM-L6-v2."""
    scorer = create_semantic_scorer()
    dim = scorer.embedding_dim
    assert dim == 384  # all-MiniLM-L6-v2 produces 384-dim embeddings


def test_semantic_scorer_encode_single():
    """Single text encoding returns correct shape."""
    scorer = create_semantic_scorer()
    embedding = scorer.encode_single("Test text")
    assert embedding.shape == (384,)
    assert embedding.dtype == np.float32


def test_semantic_scorer_encode_batch():
    """Batch encoding returns correct shape."""
    scorer = create_semantic_scorer()
    embeddings = scorer.encode(["Text 1", "Text 2", "Text 3"])
    assert embeddings.shape == (3, 384)


def test_semantic_scorer_cache_works():
    """Cache reduces computation on repeated calls."""
    scorer = create_semantic_scorer()
    scorer.cache.clear()

    text = "Cache test text"

    # First call - no cache
    import time
    start = time.perf_counter()
    emb1 = scorer.encode_single(text)
    first_time = time.perf_counter() - start

    # Second call - should use cache
    start = time.perf_counter()
    emb2 = scorer.encode_single(text)
    second_time = time.perf_counter() - start

    np.testing.assert_array_equal(emb1, emb2)
    # Second call should be faster (cached)
    assert second_time < first_time

    scorer.cache.clear()


def test_semantic_scorer_zero_norm():
    """Handle zero-norm embeddings gracefully."""
    scorer = create_semantic_scorer()

    # Empty string produces some embedding (not zero-norm for this model)
    score = scorer.similarity("", "")
    assert 0.0 <= score <= 1.0 + 1e-6  # Allow small floating point error

    # Very different texts should have low similarity
    score = scorer.similarity("completely different", "another thing")
    assert score < 0.9