"""Embedding generation using LiteLLM API."""

import asyncio
from typing import List, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from temporal_kg_rag.config.settings import get_settings
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using LiteLLM API with batching and retry logic."""

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize embedding generator.

        Args:
            api_base: LiteLLM API base URL (defaults to settings)
            api_key: LiteLLM API key (defaults to settings)
            model: Embedding model name (defaults to settings)
            dimensions: Embedding dimensions (defaults to settings)
            batch_size: Batch size for processing (defaults to settings)
        """
        settings = get_settings()

        self.api_base = (api_base or settings.litellm_api_base).rstrip("/")
        self.api_key = api_key or settings.litellm_api_key
        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dimensions
        self.batch_size = batch_size or settings.embedding_batch_size

        # HTTP client with timeout
        self.client = httpx.Client(timeout=60.0)

        logger.info(
            f"EmbeddingGenerator initialized: model={self.model}, "
            f"dimensions={self.dimensions}, batch_size={self.batch_size}, "
            f"api_base={self.api_base}"
        )

    def __del__(self):
        """Close HTTP client on cleanup."""
        if hasattr(self, "client"):
            self.client.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            httpx.HTTPError,
            httpx.TimeoutException,
        )),
        reraise=True,
    )
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            httpx.HTTPError: If API call fails
        """
        try:
            # Clean text
            text = text.replace("\n", " ").strip()

            if not text:
                logger.warning("Empty text provided for embedding")
                return [0.0] * self.dimensions

            # Call LiteLLM embeddings endpoint
            response = self.client.post(
                f"{self.api_base}/embeddings",
                json={
                    "model": self.model,
                    "input": text,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )

            response.raise_for_status()
            data = response.json()

            # Extract embedding from response
            embedding = data["data"][0]["embedding"]

            # Validate dimensions
            if len(embedding) != self.dimensions:
                logger.warning(
                    f"Expected {self.dimensions} dimensions, got {len(embedding)}. "
                    f"Updating configured dimensions."
                )
                self.dimensions = len(embedding)

            logger.debug(f"Generated embedding for text ({len(text)} chars)")

            return embedding

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error generating embedding: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            httpx.HTTPError,
            httpx.TimeoutException,
        )),
        reraise=True,
    )
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a single API call.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            httpx.HTTPError: If API call fails
        """
        if not texts:
            return []

        try:
            # Clean texts
            cleaned_texts = [text.replace("\n", " ").strip() for text in texts]

            # Filter out empty texts but keep track of indices
            text_indices = []
            valid_texts = []
            for i, text in enumerate(cleaned_texts):
                if text:
                    text_indices.append(i)
                    valid_texts.append(text)

            if not valid_texts:
                logger.warning("All texts are empty")
                return [[0.0] * self.dimensions] * len(texts)

            # Call LiteLLM embeddings endpoint with batch
            response = self.client.post(
                f"{self.api_base}/embeddings",
                json={
                    "model": self.model,
                    "input": valid_texts,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )

            response.raise_for_status()
            data = response.json()

            # Map embeddings back to original indices
            embeddings = [[0.0] * self.dimensions] * len(texts)
            for i, embedding_data in enumerate(data["data"]):
                original_index = text_indices[i]
                embedding = embedding_data["embedding"]

                # Validate dimensions on first embedding
                if i == 0 and len(embedding) != self.dimensions:
                    logger.warning(
                        f"Expected {self.dimensions} dimensions, got {len(embedding)}. "
                        f"Updating configured dimensions."
                    )
                    self.dimensions = len(embedding)

                embeddings[original_index] = embedding

            logger.info(
                f"Generated {len(valid_texts)} embeddings in batch "
                f"(skipped {len(texts) - len(valid_texts)} empty texts)"
            )

            return embeddings

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error generating embeddings batch: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error generating embeddings batch: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with automatic batching.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        logger.info(
            f"Generating embeddings for {len(texts)} texts in {total_batches} batches"
        )

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            logger.debug(f"Processing batch {batch_num}/{total_batches}")

            embeddings = self.generate_embeddings_batch(batch)
            all_embeddings.extend(embeddings)

        logger.info(f"Successfully generated {len(all_embeddings)} embeddings")

        return all_embeddings

    def get_embedding_dimensions(self) -> int:
        """Get the dimensionality of embeddings."""
        return self.dimensions


# Global generator instance
_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get the global embedding generator instance."""
    global _generator
    if _generator is None:
        _generator = EmbeddingGenerator()
    return _generator


def reset_embedding_generator():
    """Reset the global embedding generator instance."""
    global _generator
    _generator = None


def generate_embedding(text: str) -> List[float]:
    """
    Convenience function to generate a single embedding.

    Args:
        text: Text to embed

    Returns:
        Embedding vector
    """
    generator = get_embedding_generator()
    return generator.generate_embedding(text)


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Convenience function to generate multiple embeddings.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    generator = get_embedding_generator()
    return generator.generate_embeddings(texts)
