"""Document data model."""

import json
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document model representing a source document in the knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique document ID")
    title: str = Field(..., description="Document title")
    source: str = Field(..., description="Document source (file path or URL)")
    content_type: str = Field(
        default="text",
        description="Content type (pdf, markdown, html, text, etc.)",
    )
    file_path: Optional[str] = Field(None, description="Original file path")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Document creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Document last update timestamp",
    )
    version: int = Field(default=1, description="Document version number")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (author, date, tags, etc.)",
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_neo4j_dict(self) -> Dict[str, Any]:
        """
        Convert document to Neo4j-compatible dictionary.

        Neo4j doesn't support nested dictionaries as properties,
        so we convert metadata to JSON string and flatten nested dicts.
        """
        # Flatten metadata - convert nested dicts to individual properties
        result = {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "content_type": self.content_type,
            "file_path": self.file_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }

        # Add metadata as a JSON string if it exists and is not empty
        if self.metadata:
            # Flatten nested metadata into top-level properties with prefix
            for key, value in self.metadata.items():
                # Only add primitive types or convert complex types to string
                if isinstance(value, (str, int, float, bool)):
                    result[f"meta_{key}"] = value
                elif value is not None:
                    result[f"meta_{key}"] = str(value)

            # Also store the full metadata as JSON for complete preservation
            result["metadata_json"] = json.dumps(self.metadata)

        return result

    @classmethod
    def from_neo4j_dict(cls, data: Dict[str, Any]) -> "Document":
        """
        Create Document from Neo4j query result.

        Reconstructs metadata from JSON string if available.
        """
        # Extract the metadata JSON if it exists
        metadata = {}
        if "metadata_json" in data:
            try:
                metadata = json.loads(data["metadata_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Remove metadata-related fields from data
        clean_data = {k: v for k, v in data.items() if not k.startswith("meta_") and k != "metadata_json"}

        # Add reconstructed metadata
        clean_data["metadata"] = metadata

        return cls(**clean_data)

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp to current time."""
        self.updated_at = datetime.now()

    def increment_version(self) -> None:
        """Increment the version number and update timestamp."""
        self.version += 1
        self.update_timestamp()
