"""Temporal query data models."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TemporalQueryType(str, Enum):
    """Type of temporal query."""

    POINT_IN_TIME = "point_in_time"  # Query as of a specific date
    TIME_RANGE = "time_range"  # Query within a time range
    LATEST = "latest"  # Query for latest/current information
    HISTORY = "history"  # Query for historical changes


class TemporalFilter(BaseModel):
    """Temporal filter for queries."""

    query_type: TemporalQueryType = Field(
        default=TemporalQueryType.LATEST,
        description="Type of temporal query",
    )
    point_in_time: Optional[datetime] = Field(
        None,
        description="Specific point in time for POINT_IN_TIME queries",
    )
    start_time: Optional[datetime] = Field(
        None,
        description="Start of time range for TIME_RANGE queries",
    )
    end_time: Optional[datetime] = Field(
        None,
        description="End of time range for TIME_RANGE queries",
    )
    include_superseded: bool = Field(
        default=False,
        description="Include superseded/historical versions",
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_cypher_where_clause(self, node_var: str = "node") -> str:
        """
        Generate Cypher WHERE clause for this temporal filter.

        Args:
            node_var: Variable name for the node in the Cypher query

        Returns:
            Cypher WHERE clause string
        """
        clauses = []

        if self.query_type == TemporalQueryType.LATEST and not self.include_superseded:
            clauses.append(f"{node_var}.is_current = true")

        elif self.query_type == TemporalQueryType.POINT_IN_TIME and self.point_in_time:
            clauses.append(f"{node_var}.created_at <= $point_in_time")
            clauses.append(
                f"({node_var}.superseded_at IS NULL OR {node_var}.superseded_at > $point_in_time)"
            )

        elif self.query_type == TemporalQueryType.TIME_RANGE:
            if self.start_time:
                clauses.append(f"{node_var}.created_at >= $start_time")
            if self.end_time:
                clauses.append(f"{node_var}.created_at <= $end_time")

        elif self.query_type == TemporalQueryType.HISTORY:
            # Include all versions, no filters
            pass

        return " AND ".join(clauses) if clauses else "true"

    def to_cypher_parameters(self) -> dict:
        """Get parameters for Cypher query."""
        params = {}

        if self.point_in_time:
            params["point_in_time"] = self.point_in_time

        if self.start_time:
            params["start_time"] = self.start_time

        if self.end_time:
            params["end_time"] = self.end_time

        return params

    @classmethod
    def create_latest(cls) -> "TemporalFilter":
        """Create a filter for latest information."""
        return cls(query_type=TemporalQueryType.LATEST)

    @classmethod
    def create_point_in_time(cls, timestamp: datetime) -> "TemporalFilter":
        """Create a filter for a specific point in time."""
        return cls(
            query_type=TemporalQueryType.POINT_IN_TIME,
            point_in_time=timestamp,
        )

    @classmethod
    def create_time_range(
        cls,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> "TemporalFilter":
        """Create a filter for a time range."""
        return cls(
            query_type=TemporalQueryType.TIME_RANGE,
            start_time=start,
            end_time=end,
        )

    @classmethod
    def create_history(cls) -> "TemporalFilter":
        """Create a filter for historical queries (all versions)."""
        return cls(
            query_type=TemporalQueryType.HISTORY,
            include_superseded=True,
        )


class TemporalContext(BaseModel):
    """Temporal context extracted from a user query."""

    has_temporal_reference: bool = Field(
        default=False,
        description="Whether the query contains temporal references",
    )
    temporal_filter: Optional[TemporalFilter] = Field(
        None,
        description="Extracted temporal filter",
    )
    temporal_keywords: list[str] = Field(
        default_factory=list,
        description="Temporal keywords found in query",
    )
    original_temporal_phrase: Optional[str] = Field(
        None,
        description="Original temporal phrase from query",
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}
