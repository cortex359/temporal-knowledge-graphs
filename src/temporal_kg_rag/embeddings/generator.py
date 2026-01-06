"""Embedding generation using OpenAI API."""

import asyncio
from typing import List, Optional

import openai
from openai import OpenAI
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
    """Generate embeddings using OpenAI API with batching and retry logic."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize embedding generator.

        Args:
            api_key: OpenAI API key (defaults to settings)
            model: Embedding model name (defaults to settings)
            dimensions: Embedding dimensions (defaults to settings)
            batch_size: Batch size for processing (defaults to settings)
        """
        settings = get_settings()

        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_embedding_model
        self.dimensions = dimensions or settings.openai_embedding_dimensions
        self.batch_size = batch_size or settings.embedding_batch_size

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        logger.info(
            f"EmbeddingGenerator initialized: model={self.model}, "
            f"dimensions={self.dimensions}, batch_size={self.batch_size}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.APITimeoutError,
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
            openai.APIError: If API call fails
        """
        try:
            # Clean text
            text = text.replace("\n", " ").strip()

            if not text:
                logger.warning("Empty text provided for embedding")
                return [0.0] * self.dimensions

            response = self.client.embeddings.create(
                input=[text],
                model=self.model,
                dimensions=self.dimensions,
            )

            embedding = response.data[0].embedding

            logger.debug(f"Generated embedding for text ({len(text)} chars)")

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.APITimeoutError,
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
            openai.APIError: If API call fails
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

            # Generate embeddings
            response = self.client.embeddings.create(
                input=valid_texts,
                model=self.model,
                dimensions=self.dimensions,
            )

            # Map embeddings back to original indices
            embeddings = [[0.0] * self.dimensions] * len(texts)
            for i, embedding_data in enumerate(response.data):
                original_index = text_indices[i]
                embeddings[original_index] = embedding_data.embedding

            logger.info(
                f"Generated {len(valid_texts)} embeddings in batch "
                f"(skipped {len(texts) - len(valid_texts)} empty texts)"
            )

            return embeddings

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

    def estimate_cost(self, num_texts: int, avg_tokens_per_text: int = 500) -> float:
        """
        Estimate cost for generating embeddings.

        Args:
            num_texts: Number of texts
            avg_tokens_per_text: Average tokens per text

        Returns:
            Estimated cost in USD

        Note:
            Pricing as of 2024: text-embedding-3-small is $0.02 per 1M tokens
        """
        total_tokens = num_texts * avg_tokens_per_text

        # Pricing per 1M tokens
        if "text-embedding-3-small" in self.model:
            price_per_million = 0.02
        elif "text-embedding-3-large" in self.model:
            price_per_million = 0.13
        elif "text-embedding-ada-002" in self.model:
            price_per_million = 0.10
        else:
            # Default estimate
            price_per_million = 0.10

        estimated_cost = (total_tokens / 1_000_000) * price_per_million

        return estimated_cost


# Global generator instance
_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get the global embedding generator instance."""
    global _generator
    if _generator is None:
        _generator = EmbeddingGenerator()
    return _generator


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
