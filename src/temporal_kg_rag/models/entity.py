"""Entity data model."""

import json
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Entity model representing a named entity extracted from documents."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique entity ID")
    name: str = Field(..., description="Entity name")
    type: str = Field(
        ...,
        description="Entity type (PERSON, ORG, LOCATION, CONCEPT, etc.)",
    )
    first_seen: datetime = Field(
        default_factory=datetime.now,
        description="First time this entity was encountered",
    )
    last_seen: datetime = Field(
        default_factory=datetime.now,
        description="Last time this entity was mentioned",
    )
    mention_count: int = Field(
        default=1,
        description="Number of times this entity has been mentioned",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (aliases, description, etc.)",
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert entity to Neo4j-compatible dictionary."""
        result = {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "mention_count": self.mention_count,
        }

        # Flatten metadata into meta_* properties to avoid nested dict issues
        if self.metadata:
            for key, value in self.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    result[f"meta_{key}"] = value
                elif value is not None:
                    result[f"meta_{key}"] = str(value)

            # Store full metadata as JSON string
            result["metadata_json"] = json.dumps(self.metadata)

        return result

    @classmethod
    def from_neo4j_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create Entity from Neo4j query result."""
        # Reconstruct metadata from JSON if present
        metadata = {}
        if "metadata_json" in data:
            try:
                metadata = json.loads(data["metadata_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Remove flattened meta_* properties and metadata_json from data
        clean_data = {
            k: v
            for k, v in data.items()
            if not k.startswith("meta_") and k != "metadata_json"
        }
        clean_data["metadata"] = metadata

        return cls(**clean_data)

    def update_last_seen(self) -> None:
        """Update the last_seen timestamp to current time."""
        self.last_seen = datetime.now()

    def increment_mention_count(self, count: int = 1) -> None:
        """
        Increment the mention count.

        Args:
            count: Number of mentions to add
        """
        self.mention_count += count
        self.update_last_seen()


class EntityMention(BaseModel):
    """Entity mention in a specific chunk."""

    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    chunk_id: str = Field(..., description="Chunk ID where entity is mentioned")
    position: int = Field(..., description="Character position in chunk")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the mention",
    )
    context: Optional[str] = Field(
        None,
        description="Surrounding context of the mention",
    )
    valid_from: datetime = Field(
        default_factory=datetime.now,
        description="When this mention relationship became valid",
    )
    valid_to: Optional[datetime] = Field(
        None,
        description="When this mention relationship ceased to be valid",
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_neo4j_relationship_dict(self) -> Dict[str, Any]:
        """Convert mention to Neo4j relationship properties."""
        return {
            "position": self.position,
            "confidence": self.confidence,
            "context": self.context,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        }


class EntityRelationship(BaseModel):
    """Relationship between two entities."""

    source_entity_id: str = Field(..., description="Source entity ID")
    target_entity_id: str = Field(..., description="Target entity ID")
    relationship_type: str = Field(
        ...,
        description="Type of relationship (works_for, located_in, etc.)",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the relationship",
    )
    valid_from: datetime = Field(
        default_factory=datetime.now,
        description="When this relationship became valid",
    )
    valid_to: Optional[datetime] = Field(
        None,
        description="When this relationship ceased to be valid",
    )
    source_chunks: list[str] = Field(
        default_factory=list,
        description="Chunk IDs where this relationship was observed",
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_neo4j_relationship_dict(self) -> Dict[str, Any]:
        """Convert relationship to Neo4j relationship properties."""
        return {
            "relationship_type": self.relationship_type,
            "confidence": self.confidence,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "source_chunks": self.source_chunks,
        }
