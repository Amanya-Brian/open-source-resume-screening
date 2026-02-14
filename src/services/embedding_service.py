"""Embedding service for semantic similarity computations."""

import hashlib
import logging
from functools import lru_cache
from typing import Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and comparing text embeddings.

    Uses sentence-transformers for semantic similarity computations.
    """

    _instance: Optional["EmbeddingService"] = None

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the embedding service.

        Args:
            settings: Application settings. If None, uses default.
        """
        self.settings = settings or get_settings()
        self._model: Optional[SentenceTransformer] = None
        self._cache: dict[str, np.ndarray] = {}
        self._cache_enabled = self.settings.embedding_cache_enabled

    @classmethod
    def get_instance(cls, settings: Optional[Settings] = None) -> "EmbeddingService":
        """Get singleton instance of EmbeddingService.

        Args:
            settings: Application settings

        Returns:
            EmbeddingService instance
        """
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        """Get or initialize the sentence transformer model.

        Returns:
            Loaded SentenceTransformer model
        """
        if self._model is None:
            logger.info(f"Loading embedding model: {self.settings.embedding_model_name}")
            self._model = SentenceTransformer(self.settings.embedding_model_name)
            logger.info("Embedding model loaded successfully")
        return self._model

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text.

        Args:
            text: Input text

        Returns:
            Hash key for caching
        """
        return hashlib.md5(text.encode()).hexdigest()

    def encode(
        self,
        text: Union[str, list[str]],
        normalize: bool = True,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Encode text into embeddings.

        Args:
            text: Single text or list of texts to encode
            normalize: Whether to L2-normalize embeddings
            use_cache: Whether to use embedding cache

        Returns:
            Numpy array of embeddings (1D for single text, 2D for list)
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        # Check cache for all texts
        embeddings = []
        texts_to_encode = []
        text_indices = []

        for i, t in enumerate(texts):
            if self._cache_enabled and use_cache:
                cache_key = self._get_cache_key(t)
                if cache_key in self._cache:
                    embeddings.append((i, self._cache[cache_key]))
                    continue

            texts_to_encode.append(t)
            text_indices.append(i)

        # Encode texts not in cache
        if texts_to_encode:
            new_embeddings = self.model.encode(
                texts_to_encode,
                batch_size=self.settings.embedding_batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Handle single text case
            if len(texts_to_encode) == 1:
                new_embeddings = [new_embeddings]

            # Add to results and cache
            for idx, emb in zip(text_indices, new_embeddings):
                embeddings.append((idx, emb))
                if self._cache_enabled and use_cache:
                    cache_key = self._get_cache_key(texts[idx])
                    self._cache[cache_key] = emb

        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        result = np.array([e[1] for e in embeddings])

        return result[0] if is_single else result

    def encode_batch(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode a batch of texts with optimized batching.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding (uses default if None)
            normalize: Whether to L2-normalize embeddings

        Returns:
            2D numpy array of embeddings
        """
        if not texts:
            return np.array([])

        batch_size = batch_size or self.settings.embedding_batch_size

        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )

    def cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1 for normalized vectors)
        """
        # Handle batch comparisons
        if embedding1.ndim == 1 and embedding2.ndim == 1:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return float(dot_product / (norm1 * norm2 + 1e-8))

        # For normalized embeddings, dot product equals cosine similarity
        return float(np.dot(embedding1, embedding2))

    def similarity_matrix(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> np.ndarray:
        """Compute similarity matrix between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings (n x d)
            embeddings2: Second set of embeddings (m x d)

        Returns:
            Similarity matrix of shape (n x m)
        """
        # Normalize if needed
        norms1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)

        embeddings1_norm = embeddings1 / (norms1 + 1e-8)
        embeddings2_norm = embeddings2 / (norms2 + 1e-8)

        return np.dot(embeddings1_norm, embeddings2_norm.T)

    def semantic_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """Compute semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        return self.cosine_similarity(emb1, emb2)

    def find_most_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, str, float]]:
        """Find most similar texts to a query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return

        Returns:
            List of (index, text, similarity_score) tuples
        """
        if not candidates:
            return []

        query_emb = self.encode(query)
        candidate_embs = self.encode(candidates)

        similarities = np.dot(candidate_embs, query_emb)

        # Get top-k indices
        if len(candidates) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [
            (int(i), candidates[i], float(similarities[i]))
            for i in top_indices
        ]

    def compute_skill_match_score(
        self,
        resume_skills: list[str],
        required_skills: list[str],
    ) -> tuple[float, list[str], list[str]]:
        """Compute skill match score between resume and requirements.

        Args:
            resume_skills: Skills from resume
            required_skills: Required skills from job

        Returns:
            Tuple of (score, matching_skills, missing_skills)
        """
        if not required_skills:
            return 1.0, resume_skills, []

        if not resume_skills:
            return 0.0, [], required_skills

        # Encode all skills
        resume_embs = self.encode(resume_skills)
        required_embs = self.encode(required_skills)

        # Compute similarity matrix
        sim_matrix = self.similarity_matrix(resume_embs, required_embs)

        # For each required skill, find best match
        matching_skills = []
        missing_skills = []

        threshold = 0.7  # Similarity threshold for match

        for i, req_skill in enumerate(required_skills):
            best_match_idx = np.argmax(sim_matrix[:, i])
            best_similarity = sim_matrix[best_match_idx, i]

            if best_similarity >= threshold:
                matching_skills.append(resume_skills[best_match_idx])
            else:
                missing_skills.append(req_skill)

        score = len(matching_skills) / len(required_skills)
        return score, list(set(matching_skills)), missing_skills

    def compute_text_relevance(
        self,
        source_text: str,
        target_text: str,
        chunk_size: int = 500,
    ) -> float:
        """Compute relevance score between two texts.

        For long texts, chunks the source and finds maximum relevance.

        Args:
            source_text: Source text (e.g., resume)
            target_text: Target text (e.g., job description)
            chunk_size: Maximum chunk size for long texts

        Returns:
            Relevance score (0-1)
        """
        # For short texts, compute directly
        if len(source_text) <= chunk_size:
            return self.semantic_similarity(source_text, target_text)

        # Chunk source text
        words = source_text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Compute similarity for each chunk
        chunk_embs = self.encode(chunks)
        target_emb = self.encode(target_text)

        similarities = np.dot(chunk_embs, target_emb)

        # Return max similarity (best matching section)
        return float(np.max(similarities))

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    def get_cache_size(self) -> int:
        """Get number of cached embeddings.

        Returns:
            Number of cached embeddings
        """
        return len(self._cache)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings.

        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
