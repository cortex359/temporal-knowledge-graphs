"""Embedding cache for avoiding redundant API calls."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from temporal_kg_rag.config.settings import get_settings
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """File-based cache for embeddings."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize embedding cache.

        Args:
            cache_dir: Directory for cache files (defaults to settings)
        """
        settings = get_settings()
        self.cache_dir = cache_dir or settings.cache_dir
        self.enabled = settings.enable_embedding_cache

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Embedding cache initialized at: {self.cache_dir}")
        else:
            logger.info("Embedding cache is disabled")

    def _get_cache_key(self, text: str, model: str, dimensions: int) -> str:
        """
        Generate cache key for text.

        Args:
            text: Text to cache
            model: Model name
            dimensions: Embedding dimensions

        Returns:
            Cache key (hash)
        """
        # Create a unique key based on text, model, and dimensions
        content = f"{text}|{model}|{dimensions}"
        key = hashlib.sha256(content.encode()).hexdigest()
        return key

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        # Use subdirectories to avoid too many files in one directory
        subdir = cache_key[:2]
        cache_path = self.cache_dir / subdir / f"{cache_key}.pkl"
        return cache_path

    def get(
        self,
        text: str,
        model: str,
        dimensions: int,
    ) -> Optional[List[float]]:
        """
        Get embedding from cache.

        Args:
            text: Text to look up
            model: Model name
            dimensions: Embedding dimensions

        Returns:
            Cached embedding vector or None if not found
        """
        if not self.enabled:
            return None

        try:
            cache_key = self._get_cache_key(text, model, dimensions)
            cache_path = self._get_cache_path(cache_key)

            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)

                # Verify the cached data matches
                if (cached_data.get("model") == model and
                    cached_data.get("dimensions") == dimensions):
                    logger.debug(f"Cache hit for text ({len(text)} chars)")
                    return cached_data["embedding"]

        except Exception as e:
            logger.warning(f"Error reading from cache: {e}")

        return None

    def set(
        self,
        text: str,
        embedding: List[float],
        model: str,
        dimensions: int,
    ) -> None:
        """
        Store embedding in cache.

        Args:
            text: Text that was embedded
            embedding: Embedding vector
            model: Model name
            dimensions: Embedding dimensions
        """
        if not self.enabled:
            return

        try:
            cache_key = self._get_cache_key(text, model, dimensions)
            cache_path = self._get_cache_path(cache_key)

            # Create subdirectory if needed
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Store embedding with metadata
            cached_data = {
                "embedding": embedding,
                "model": model,
                "dimensions": dimensions,
                "text_length": len(text),
            }

            with open(cache_path, "wb") as f:
                pickle.dump(cached_data, f)

            logger.debug(f"Cached embedding for text ({len(text)} chars)")

        except Exception as e:
            logger.warning(f"Error writing to cache: {e}")

    def get_batch(
        self,
        texts: List[str],
        model: str,
        dimensions: int,
    ) -> Dict[int, List[float]]:
        """
        Get multiple embeddings from cache.

        Args:
            texts: List of texts to look up
            model: Model name
            dimensions: Embedding dimensions

        Returns:
            Dictionary mapping text index to cached embedding
        """
        cached = {}

        for i, text in enumerate(texts):
            embedding = self.get(text, model, dimensions)
            if embedding is not None:
                cached[i] = embedding

        if cached:
            logger.info(f"Cache hit for {len(cached)}/{len(texts)} texts")

        return cached

    def set_batch(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        model: str,
        dimensions: int,
    ) -> None:
        """
        Store multiple embeddings in cache.

        Args:
            texts: List of texts
            embeddings: List of embedding vectors
            model: Model name
            dimensions: Embedding dimensions
        """
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have same length")

        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding, model, dimensions)

        logger.info(f"Cached {len(texts)} embeddings")

    def clear(self) -> int:
        """
        Clear all cached embeddings.

        Returns:
            Number of files deleted
        """
        if not self.enabled or not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.rglob("*.pkl"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Error deleting cache file {cache_file}: {e}")

        logger.info(f"Cleared {count} cached embeddings")
        return count

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if not self.enabled or not self.cache_dir.exists():
            return {"total_files": 0, "total_size_bytes": 0}

        total_files = 0
        total_size = 0

        for cache_file in self.cache_dir.rglob("*.pkl"):
            total_files += 1
            total_size += cache_file.stat().st_size

        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }


# Global cache instance
_cache: Optional[EmbeddingCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Get the global embedding cache instance."""
    global _cache
    if _cache is None:
        _cache = EmbeddingCache()
    return _cache
