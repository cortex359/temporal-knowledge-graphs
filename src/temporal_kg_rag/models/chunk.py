"""Chunk data model."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Chunk model representing a text segment with embedding."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique chunk ID")
    text: str = Field(..., description="Chunk text content")
    embedding: Optional[List[float]] = Field(
        None,
        description="Embedding vector (1536 dimensions for OpenAI)",
    )
    chunk_index: int = Field(..., description="Position in document (0-indexed)")
    token_count: int = Field(..., description="Number of tokens in chunk")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Chunk creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Chunk last update timestamp",
    )
    version: int = Field(default=1, description="Chunk version number")
    is_current: bool = Field(
        default=True,
        description="Whether this is the current version (not superseded)",
    )
    superseded_at: Optional[datetime] = Field(
        None,
        description="Timestamp when this chunk was superseded by a new version",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (start_char, end_char, etc.)",
    )

    # Document reference
    document_id: Optional[str] = Field(None, description="Parent document ID")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_neo4j_dict(self, include_embedding: bool = True) -> Dict[str, Any]:
        """
        Convert chunk to Neo4j-compatible dictionary.

        Args:
            include_embedding: Whether to include the embedding vector

        Returns:
            Dictionary suitable for Neo4j storage
        """
        data = {
            "id": self.id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "is_current": self.is_current,
            "superseded_at": self.superseded_at,
            "metadata": self.metadata,
        }

        if include_embedding and self.embedding:
            data["embedding"] = self.embedding

        return data

    @classmethod
    def from_neo4j_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create Chunk from Neo4j query result."""
        return cls(**data)

    def supersede(self) -> None:
        """Mark this chunk as superseded by a new version."""
        self.is_current = False
        self.superseded_at = datetime.now()

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp to current time."""
        self.updated_at = datetime.now()

    def increment_version(self) -> None:
        """Increment the version number and update timestamp."""
        self.version += 1
        self.update_timestamp()

    def has_embedding(self) -> bool:
        """Check if chunk has an embedding."""
        return self.embedding is not None and len(self.embedding) > 0

    def get_embedding_dimensions(self) -> int:
        """Get the dimensionality of the embedding."""
        return len(self.embedding) if self.embedding else 0
