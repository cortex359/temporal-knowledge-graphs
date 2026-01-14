"""Text chunking strategies for document processing."""

import re
from typing import List, Optional

import tiktoken

from temporal_kg_rag.config.settings import get_settings
from temporal_kg_rag.models.chunk import Chunk
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import nltk, but don't fail if it's not available
try:
    import nltk
    # Try to use punkt tokenizer
    try:
        nltk.data.find("tokenizers/punkt_tab")
        NLTK_AVAILABLE = True
    except LookupError:
        logger.warning("NLTK punkt tokenizer not found, will use simple sentence splitter")
        NLTK_AVAILABLE = False
except ImportError:
    logger.warning("NLTK not available, will use simple sentence splitter")
    NLTK_AVAILABLE = False


class Chunker:
    """Text chunking with configurable size and overlap."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        encoding_name: str = "cl100k_base",
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Maximum chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            encoding_name: Tiktoken encoding name
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

        logger.info(
            f"Chunker initialized: size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, encoding={encoding_name}"
        )

    def chunk_text(
        self,
        text: str,
        document_id: Optional[str] = None,
        strategy: str = "semantic",
    ) -> List[Chunk]:
        """
        Chunk text into segments.

        Args:
            text: Text to chunk
            document_id: Optional document ID to associate with chunks
            strategy: Chunking strategy ('semantic' or 'fixed')

        Returns:
            List of Chunk objects
        """
        if strategy == "semantic":
            return self._chunk_semantic(text, document_id)
        elif strategy == "fixed":
            return self._chunk_fixed(text, document_id)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _chunk_semantic(self, text: str, document_id: Optional[str]) -> List[Chunk]:
        """
        Chunk text semantically, preserving sentence boundaries.

        Args:
            text: Text to chunk
            document_id: Optional document ID

        Returns:
            List of Chunk objects
        """
        # Split text into sentences
        if NLTK_AVAILABLE:
            sentences = nltk.sent_tokenize(text)
        else:
            # Simple sentence splitter as fallback
            sentences = self._simple_sentence_split(text)

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))

            # If single sentence exceeds chunk size, split it
            if sentence_tokens > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(self._create_chunk(
                        chunk_text, chunk_index, document_id
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0

                # Split long sentence
                long_chunks = self._split_long_text(sentence, document_id, chunk_index)
                chunks.extend(long_chunks)
                chunk_index += len(long_chunks)
                continue

            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk)
                chunks.append(self._create_chunk(
                    chunk_text, chunk_index, document_id
                ))
                chunk_index += 1

                # Calculate overlap
                overlap_chunk = []
                overlap_tokens = 0

                # Add sentences from the end until we reach overlap size
                for sent in reversed(current_chunk):
                    sent_tokens = len(self.encoding.encode(sent))
                    if overlap_tokens + sent_tokens <= self.chunk_overlap:
                        overlap_chunk.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break

                # Start new chunk with overlap
                current_chunk = overlap_chunk + [sentence]
                current_tokens = overlap_tokens + sentence_tokens
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk(
                chunk_text, chunk_index, document_id
            ))

        logger.info(
            f"Created {len(chunks)} chunks using semantic strategy "
            f"(avg size: {sum(c.token_count for c in chunks) / len(chunks):.0f} tokens)"
        )

        return chunks

    def _simple_sentence_split(self, text: str) -> List[str]:
        """
        Simple sentence splitter as fallback when NLTK is not available.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Split on common sentence endings
        # This is simpler than NLTK but works reasonably well
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _chunk_fixed(self, text: str, document_id: Optional[str]) -> List[Chunk]:
        """
        Chunk text with fixed token size.

        Args:
            text: Text to chunk
            document_id: Optional document ID

        Returns:
            List of Chunk objects
        """
        tokens = self.encoding.encode(text)
        chunks = []
        chunk_index = 0

        start = 0
        while start < len(tokens):
            # Get chunk of tokens
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)

            # Create chunk
            chunks.append(self._create_chunk(
                chunk_text, chunk_index, document_id
            ))
            chunk_index += 1

            # Move start position (with overlap)
            start += self.chunk_size - self.chunk_overlap

        logger.info(
            f"Created {len(chunks)} chunks using fixed strategy "
            f"(avg size: {sum(c.token_count for c in chunks) / len(chunks):.0f} tokens)"
        )

        return chunks

    def _split_long_text(
        self,
        text: str,
        document_id: Optional[str],
        start_index: int,
    ) -> List[Chunk]:
        """
        Split text that exceeds chunk size.

        Args:
            text: Text to split
            document_id: Optional document ID
            start_index: Starting chunk index

        Returns:
            List of Chunk objects
        """
        tokens = self.encoding.encode(text)
        chunks = []
        chunk_index = start_index

        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append(self._create_chunk(
                chunk_text, chunk_index, document_id
            ))
            chunk_index += 1
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _create_chunk(
        self,
        text: str,
        chunk_index: int,
        document_id: Optional[str],
    ) -> Chunk:
        """
        Create a Chunk object.

        Args:
            text: Chunk text
            chunk_index: Chunk index in document
            document_id: Optional document ID

        Returns:
            Chunk object
        """
        token_count = len(self.encoding.encode(text))

        return Chunk(
            text=text,
            chunk_index=chunk_index,
            token_count=token_count,
            document_id=document_id,
        )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def estimate_chunks(self, text: str) -> int:
        """
        Estimate number of chunks that will be created.

        Args:
            text: Text to analyze

        Returns:
            Estimated number of chunks
        """
        total_tokens = self.count_tokens(text)
        effective_chunk_size = self.chunk_size - self.chunk_overlap

        if effective_chunk_size <= 0:
            raise ValueError("Overlap must be less than chunk size")

        # Estimate number of chunks
        num_chunks = max(1, (total_tokens + effective_chunk_size - 1) // effective_chunk_size)

        return num_chunks


def chunk_text(
    text: str,
    document_id: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    strategy: str = "semantic",
) -> List[Chunk]:
    """
    Convenience function to chunk text.

    Args:
        text: Text to chunk
        document_id: Optional document ID
        chunk_size: Optional chunk size (defaults to settings)
        chunk_overlap: Optional overlap size (defaults to settings)
        strategy: Chunking strategy ('semantic' or 'fixed')

    Returns:
        List of Chunk objects
    """
    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_text(text, document_id, strategy)
