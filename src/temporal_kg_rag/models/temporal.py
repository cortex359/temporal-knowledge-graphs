"""
Temporal query data models with bi-temporal support.

This module implements a bi-temporal model following Zep/Graphiti architecture:
- Event time (valid_from/valid_to): When the fact was actually true
- Transaction time (created_at/expired_at): When the fact was recorded/invalidated

This enables:
- Point-in-time queries ("What was true on date X?")
- Transaction-time queries ("What did we know on date Y?")
- Bi-temporal queries ("What did we believe was true about X at time Y?")
"""

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
    BITEMPORAL = "bitemporal"  # Query with both event and transaction time


class TemporalFilter(BaseModel):
    """
    Temporal filter for queries with bi-temporal support.

    Supports two temporal dimensions:
    - Event time: When the fact was actually true (valid_from/valid_to)
    - Transaction time: When the fact was recorded (created_at/expired_at)
    """

    query_type: TemporalQueryType = Field(
        default=TemporalQueryType.LATEST,
        description="Type of temporal query",
    )
    point_in_time: Optional[datetime] = Field(
        None,
        description="Specific point in time for POINT_IN_TIME queries (event time)",
    )
    start_time: Optional[datetime] = Field(
        None,
        description="Start of time range for TIME_RANGE queries (event time)",
    )
    end_time: Optional[datetime] = Field(
        None,
        description="End of time range for TIME_RANGE queries (event time)",
    )
    include_superseded: bool = Field(
        default=False,
        description="Include superseded/historical versions",
    )

    # Bi-temporal fields (transaction time dimension)
    transaction_time: Optional[datetime] = Field(
        None,
        description="Transaction time for bi-temporal queries (what we knew at this time)",
    )
    transaction_start: Optional[datetime] = Field(
        None,
        description="Transaction time range start for bi-temporal queries",
    )
    transaction_end: Optional[datetime] = Field(
        None,
        description="Transaction time range end for bi-temporal queries",
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_cypher_where_clause(self, node_var: str = "node", rel_var: Optional[str] = None) -> str:
        """
        Generate Cypher WHERE clause for this temporal filter.

        Supports bi-temporal filtering on both nodes and relationships.

        Args:
            node_var: Variable name for the node in the Cypher query
            rel_var: Optional variable name for relationship (for bi-temporal)

        Returns:
            Cypher WHERE clause string
        """
        clauses = []

        if self.query_type == TemporalQueryType.LATEST and not self.include_superseded:
            clauses.append(f"{node_var}.is_current = true")

        elif self.query_type == TemporalQueryType.POINT_IN_TIME and self.point_in_time:
            # Event time: when was this fact true?
            clauses.append(f"{node_var}.created_at <= $point_in_time")
            # Handle case where superseded_at property might not exist in DB
            # Use coalesce pattern: check is_current OR superseded_at > point_in_time
            clauses.append(
                f"({node_var}.is_current = true OR {node_var}.superseded_at > $point_in_time)"
            )
            # Note: Relationship temporal filtering is disabled as most relationships
            # don't have valid_from/valid_to properties in the current schema.

        elif self.query_type == TemporalQueryType.TIME_RANGE:
            if self.start_time:
                clauses.append(f"{node_var}.created_at >= $start_time")
            if self.end_time:
                clauses.append(f"{node_var}.created_at <= $end_time")
            # Note: Relationship temporal filtering is disabled as most relationships
            # don't have valid_from/valid_to properties in the current schema.
            # Enable when bi-temporal relationships are fully implemented.

        elif self.query_type == TemporalQueryType.BITEMPORAL:
            # Bi-temporal query: filter by both event time and transaction time
            if self.point_in_time:
                # Event time filter - use created_at as proxy for valid_from
                clauses.append(f"{node_var}.created_at <= $point_in_time")
                clauses.append(
                    f"({node_var}.is_current = true OR {node_var}.superseded_at > $point_in_time)"
                )
            if self.transaction_time:
                # Transaction time filter: what did we know at this time?
                clauses.append(f"{node_var}.created_at <= $transaction_time")
            # Note: Relationship bi-temporal filtering is disabled as most relationships
            # don't have valid_from/valid_to properties in the current schema.

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

        # Bi-temporal parameters
        if self.transaction_time:
            params["transaction_time"] = self.transaction_time

        if self.transaction_start:
            params["transaction_start"] = self.transaction_start

        if self.transaction_end:
            params["transaction_end"] = self.transaction_end

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

    @classmethod
    def create_bitemporal(
        cls,
        event_time: Optional[datetime] = None,
        transaction_time: Optional[datetime] = None,
    ) -> "TemporalFilter":
        """
        Create a bi-temporal filter.

        Args:
            event_time: When was the fact true? (valid time)
            transaction_time: What did we know at this time? (transaction time)

        Returns:
            Bi-temporal filter
        """
        return cls(
            query_type=TemporalQueryType.BITEMPORAL,
            point_in_time=event_time,
            transaction_time=transaction_time,
        )

    @classmethod
    def create_as_of_transaction(cls, transaction_time: datetime) -> "TemporalFilter":
        """
        Create a filter for "what did we know at time X?"

        This returns the state of knowledge as it existed at transaction_time,
        regardless of when the facts were actually true.

        Args:
            transaction_time: The point in transaction time to query

        Returns:
            Transaction-time filter
        """
        return cls(
            query_type=TemporalQueryType.BITEMPORAL,
            transaction_time=transaction_time,
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
