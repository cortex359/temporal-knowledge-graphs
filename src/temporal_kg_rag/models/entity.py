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
    """
    Temporal quadruple representing a relationship between two entities.

    Stored as: (relationship, timestamp, source, target, description)

    The relationship description is free-form and comprehensive, following these guidelines:
    1. The nature of the relationship (e.g., familial, professional, causal)
    2. The impact or significance of the relationship on both entities
    3. Any historical or contextual information relevant to the relationship
    4. How the relationship evolved over time (if applicable)
    5. Any notable events or actions that resulted from this relationship
    """

    # Core temporal quadruple fields
    source_entity_id: str = Field(..., description="Source entity ID")
    source_entity_name: str = Field(default="", description="Source entity name")
    target_entity_id: str = Field(..., description="Target entity ID")
    target_entity_name: str = Field(default="", description="Target entity name")

    # Free-form relationship label (short identifier)
    relationship: str = Field(
        ...,
        description="Short relationship label (e.g., 'founded', 'collaborated with', 'acquired')",
    )

    # Comprehensive relationship description following the guidelines
    description: str = Field(
        ...,
        description=(
            "Detailed multi-sentence description covering: nature of relationship, "
            "impact/significance, historical context, evolution over time, notable events"
        ),
    )

    # Temporal information (the timestamp in the quadruple)
    timestamp: Optional[datetime] = Field(
        None,
        description="Primary timestamp when this relationship was established or is most relevant",
    )
    valid_from: datetime = Field(
        default_factory=datetime.now,
        description="When this relationship became valid (start of validity period)",
    )
    valid_to: Optional[datetime] = Field(
        None,
        description="When this relationship ceased to be valid (end of validity period)",
    )

    # Metadata
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the relationship extraction",
    )
    source_chunks: list[str] = Field(
        default_factory=list,
        description="Chunk IDs where this relationship was observed",
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional relationship properties extracted from context",
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_neo4j_relationship_dict(self) -> Dict[str, Any]:
        """Convert relationship to Neo4j relationship properties."""
        result = {
            "relationship": self.relationship,
            "description": self.description,
            "timestamp": self.timestamp,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "confidence": self.confidence,
            "source_chunks": self.source_chunks,
        }
        # Add extra properties
        for key, value in self.properties.items():
            if isinstance(value, (str, int, float, bool)):
                result[f"prop_{key}"] = value
        return result

    def to_quadruple(self) -> tuple:
        """
        Return as temporal quadruple: (relationship, timestamp, source, target, description).
        """
        return (
            self.relationship,
            self.timestamp,
            self.source_entity_name,
            self.target_entity_name,
            self.description,
        )

    @classmethod
    def from_quadruple(
        cls,
        relationship: str,
        timestamp: Optional[datetime],
        source_id: str,
        source_name: str,
        target_id: str,
        target_name: str,
        description: str,
        **kwargs,
    ) -> "EntityRelationship":
        """Create EntityRelationship from quadruple components."""
        return cls(
            relationship=relationship,
            timestamp=timestamp,
            source_entity_id=source_id,
            source_entity_name=source_name,
            target_entity_id=target_id,
            target_entity_name=target_name,
            description=description,
            valid_from=timestamp or datetime.now(),
            **kwargs,
        )
