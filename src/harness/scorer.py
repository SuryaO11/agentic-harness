"""Semantic similarity scoring for Agentic Harness v1.2.

Uses Sentence Transformers (all-MiniLM-L6-v2) for local embedding generation
and cosine similarity scoring to detect semantic drift.
"""
import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .contracts import FailureTypeEnum, SeverityEnum, ValidatorResult


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = ".cache/embeddings"
DEFAULT_SIMILARITY_THRESHOLD = 0.75


class EmbeddingCache:
    """Local file-based cache for sentence embeddings.

    Stores embeddings as pickled numpy arrays keyed by content hash.
    """

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, np.ndarray] = {}

    def _get_cache_key(self, text: str) -> str:
        """Generate deterministic cache key from text content."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def _get_cache_path(self, key: str) -> Path:
        """Get filesystem path for cache key."""
        return self.cache_dir / f"{key}.pkl"

    def get(self, text: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache (memory first, then disk)."""
        key = self._get_cache_key(text)

        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Check disk cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    embedding = pickle.load(f)
                self._memory_cache[key] = embedding
                return embedding
            except Exception:
                # Corrupted cache file - ignore and recompute
                pass

        return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in both memory and disk cache."""
        key = self._get_cache_key(text)
        self._memory_cache[key] = embedding

        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(embedding, f)
        except Exception:
            # Non-fatal: cache write failed, but memory cache works
            pass

    def clear(self) -> None:
        """Clear both memory and disk cache."""
        self._memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception:
                pass


class SemanticScorer:
    """Semantic similarity scorer using Sentence Transformers.

    Computes cosine similarity between generated output and reference/ground truth
    to detect semantic drift independently of syntactic validation.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: str = DEFAULT_CACHE_DIR,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.cache = EmbeddingCache(cache_dir)

        # Load model (will download on first use if not cached)
        self._model: Optional[SentenceTransformer] = None
        self._device = device
        self._load_time_ms: float = 0.0

    def _load_model(self) -> SentenceTransformer:
        """Lazy load the Sentence Transformer model."""
        if self._model is None:
            start = time.perf_counter()
            self._model = SentenceTransformer(self.model_name, device=self._device)
            self._load_time_ms = (time.perf_counter() - start) * 1000
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings, using cache when available.

        Args:
            texts: List of texts to encode.

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim).
        """
        model = self._load_model()

        # Check cache for each text
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Encode uncached texts
        if uncached_texts:
            new_embeddings = model.encode(uncached_texts, convert_to_numpy=True, show_progress_bar=False)
            for idx, emb in zip(uncached_indices, new_embeddings):
                embeddings[idx] = emb
                self.cache.set(uncached_texts[uncached_indices.index(idx)], emb)

        return np.array(embeddings)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to embedding."""
        return self.encode([text])[0]

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text1: First text (e.g., generated output).
            text2: Second text (e.g., ground truth/reference).

        Returns:
            Cosine similarity score in [0.0, 1.0].
        """
        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def evaluate(
        self,
        generated: str,
        reference: str,
        threshold: Optional[float] = None,
    ) -> ValidatorResult:
        """Evaluate semantic similarity against reference.

        Args:
            generated: LLM-generated output.
            reference: Ground truth or expected intent.
            threshold: Optional custom threshold (uses default if not provided).

        Returns:
            ValidatorResult with similarity score and pass/fail.
        """
        active_threshold = threshold if threshold is not None else self.similarity_threshold
        score = self.similarity(generated, reference)

        if score >= active_threshold:
            return ValidatorResult(
                passed=True,
                reason=f"Semantic similarity {score:.4f} >= threshold {active_threshold}",
                severity=SeverityEnum.INFO,
            )
        else:
            return ValidatorResult(
                passed=False,
                reason=f"Semantic similarity {score:.4f} below threshold {active_threshold} (semantic drift detected)",
                severity=SeverityEnum.CRITICAL,
                failure_type=FailureTypeEnum.SEMANTIC_DRIFT,
            )

    def batch_evaluate(
        self,
        generated_list: List[str],
        reference_list: List[str],
        threshold: Optional[float] = None,
    ) -> List[ValidatorResult]:
        """Evaluate multiple generated-reference pairs.

        Args:
            generated_list: List of generated outputs.
            reference_list: List of corresponding references.
            threshold: Optional custom threshold.

        Returns:
            List of ValidatorResults.
        """
        if len(generated_list) != len(reference_list):
            raise ValueError("generated_list and reference_list must have same length")

        # Encode all at once for efficiency
        gen_embeddings = self.encode(generated_list)
        ref_embeddings = self.encode(reference_list)

        results = []
        active_threshold = threshold if threshold is not None else self.similarity_threshold

        for gen_emb, ref_emb in zip(gen_embeddings, ref_embeddings):
            dot_product = np.dot(gen_emb, ref_emb)
            norm_gen = np.linalg.norm(gen_emb)
            norm_ref = np.linalg.norm(ref_emb)

            if norm_gen == 0 or norm_ref == 0:
                score = 0.0
            else:
                score = float(dot_product / (norm_gen * norm_ref))

            if score >= active_threshold:
                results.append(ValidatorResult(
                    passed=True,
                    reason=f"Semantic similarity {score:.4f} >= threshold {active_threshold}",
                    severity=SeverityEnum.INFO,
                ))
            else:
                results.append(ValidatorResult(
                    passed=False,
                    reason=f"Semantic similarity {score:.4f} below threshold {active_threshold}",
                    severity=SeverityEnum.CRITICAL,
                    failure_type=FailureTypeEnum.SEMANTIC_DRIFT,
                ))

        return results

    @property
    def model_load_time_ms(self) -> float:
        """Time taken to load model (0 if not loaded yet)."""
        return self._load_time_ms

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of embeddings produced by the model."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()


def create_semantic_scorer(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: str = DEFAULT_CACHE_DIR,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> SemanticScorer:
    """Factory function to create a SemanticScorer with default settings."""
    return SemanticScorer(
        model_name=model_name,
        cache_dir=cache_dir,
        similarity_threshold=similarity_threshold,
    )