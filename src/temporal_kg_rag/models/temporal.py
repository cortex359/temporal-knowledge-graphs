"""
Temporal query data models for fiscal period-based filtering.

This module implements temporal filtering based on document content time
(fiscal year/quarter), NOT system timestamps like created_at.

Key principle: The temporal relevance comes from WHEN THE CONTENT IS ABOUT
(e.g., "Q1 2021 Earnings Call"), not when it was ingested into the system.

This enables:
- Fiscal period queries ("What happened in Q1 2021?")
- Year range queries ("Between 2020 and 2023")
- Latest fiscal period queries ("Most recent quarter")
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class TemporalQueryType(str, Enum):
    """Type of temporal query."""

    POINT_IN_TIME = "point_in_time"  # Query for a specific fiscal period
    TIME_RANGE = "time_range"  # Query within a fiscal year/quarter range
    LATEST = "latest"  # Query for latest/current information
    HISTORY = "history"  # Query for all historical data


class TemporalFilter(BaseModel):
    """
    Temporal filter for queries based on fiscal periods.

    Filters are based on content time (fiscal_year, fiscal_quarter),
    NOT system timestamps (created_at).
    """

    query_type: TemporalQueryType = Field(
        default=TemporalQueryType.LATEST,
        description="Type of temporal query",
    )

    # Fiscal period fields (primary temporal dimension)
    fiscal_year: Optional[int] = Field(
        None,
        description="Fiscal year to filter by (e.g., 2021)",
    )
    fiscal_quarter: Optional[str] = Field(
        None,
        description="Fiscal quarter to filter by (e.g., 'Q1', 'Q2', 'Q3', 'Q4')",
    )

    # For range queries
    start_year: Optional[int] = Field(
        None,
        description="Start year for TIME_RANGE queries",
    )
    end_year: Optional[int] = Field(
        None,
        description="End year for TIME_RANGE queries",
    )
    start_quarter: Optional[str] = Field(
        None,
        description="Start quarter for TIME_RANGE queries",
    )
    end_quarter: Optional[str] = Field(
        None,
        description="End quarter for TIME_RANGE queries",
    )

    # Legacy fields for backward compatibility (converted to fiscal period)
    point_in_time: Optional[datetime] = Field(
        None,
        description="Specific point in time (will be converted to fiscal period)",
    )
    start_time: Optional[datetime] = Field(
        None,
        description="Start of time range (will be converted to fiscal period)",
    )
    end_time: Optional[datetime] = Field(
        None,
        description="End of time range (will be converted to fiscal period)",
    )

    include_superseded: bool = Field(
        default=False,
        description="Include superseded/historical versions",
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    def _datetime_to_fiscal(self, dt: datetime) -> tuple[int, str]:
        """Convert datetime to fiscal year and quarter."""
        year = dt.year
        month = dt.month

        if month <= 3:
            quarter = "Q1"
        elif month <= 6:
            quarter = "Q2"
        elif month <= 9:
            quarter = "Q3"
        else:
            quarter = "Q4"

        return year, quarter

    def _get_effective_fiscal_period(self) -> tuple[Optional[int], Optional[str]]:
        """Get effective fiscal year/quarter, converting from datetime if needed."""
        if self.fiscal_year is not None:
            return self.fiscal_year, self.fiscal_quarter

        if self.point_in_time:
            return self._datetime_to_fiscal(self.point_in_time)

        return None, None

    def _get_effective_range(self) -> tuple[Optional[int], Optional[str], Optional[int], Optional[str]]:
        """Get effective fiscal range, converting from datetime if needed."""
        start_year = self.start_year
        start_quarter = self.start_quarter
        end_year = self.end_year
        end_quarter = self.end_quarter

        if start_year is None and self.start_time:
            start_year, start_quarter = self._datetime_to_fiscal(self.start_time)

        if end_year is None and self.end_time:
            end_year, end_quarter = self._datetime_to_fiscal(self.end_time)

        return start_year, start_quarter, end_year, end_quarter

    def to_cypher_where_clause(self, node_var: str = "node", rel_var: Optional[str] = None) -> str:
        """
        Generate Cypher WHERE clause for this temporal filter.

        Filters based on fiscal_year and fiscal_quarter properties,
        NOT on system timestamps like created_at.

        Args:
            node_var: Variable name for the node in the Cypher query
            rel_var: Optional variable name for relationship (unused)

        Returns:
            Cypher WHERE clause string
        """
        clauses = []

        if self.query_type == TemporalQueryType.LATEST and not self.include_superseded:
            clauses.append(f"{node_var}.is_current = true")

        elif self.query_type == TemporalQueryType.POINT_IN_TIME:
            # Filter by specific fiscal period
            fiscal_year, fiscal_quarter = self._get_effective_fiscal_period()

            if fiscal_year:
                clauses.append(f"{node_var}.fiscal_year = $fiscal_year")
            if fiscal_quarter:
                clauses.append(f"{node_var}.fiscal_quarter = $fiscal_quarter")

            if not self.include_superseded:
                clauses.append(f"{node_var}.is_current = true")

        elif self.query_type == TemporalQueryType.TIME_RANGE:
            # Filter by fiscal year/quarter range
            start_year, start_quarter, end_year, end_quarter = self._get_effective_range()

            if start_year is not None:
                clauses.append(f"{node_var}.fiscal_year >= $start_year")
            if end_year is not None:
                clauses.append(f"{node_var}.fiscal_year <= $end_year")

            # Quarter filtering for single-year ranges
            if start_year is not None and end_year is not None and start_year == end_year:
                if start_quarter:
                    clauses.append(f"{node_var}.fiscal_quarter >= $start_quarter")
                if end_quarter:
                    clauses.append(f"{node_var}.fiscal_quarter <= $end_quarter")

            if not self.include_superseded:
                clauses.append(f"{node_var}.is_current = true")

        elif self.query_type == TemporalQueryType.HISTORY:
            # Include all versions, no fiscal period filters
            pass

        return " AND ".join(clauses) if clauses else "true"

    def to_cypher_parameters(self) -> dict:
        """Get parameters for Cypher query."""
        params = {}

        # Fiscal period parameters
        fiscal_year, fiscal_quarter = self._get_effective_fiscal_period()
        if fiscal_year is not None:
            params["fiscal_year"] = fiscal_year
        if fiscal_quarter:
            params["fiscal_quarter"] = fiscal_quarter

        # Range parameters
        start_year, start_quarter, end_year, end_quarter = self._get_effective_range()
        if start_year is not None:
            params["start_year"] = start_year
        if end_year is not None:
            params["end_year"] = end_year
        if start_quarter:
            params["start_quarter"] = start_quarter
        if end_quarter:
            params["end_quarter"] = end_quarter

        return params

    @classmethod
    def create_latest(cls) -> "TemporalFilter":
        """Create a filter for latest/current information."""
        return cls(query_type=TemporalQueryType.LATEST)

    @classmethod
    def create_fiscal_period(
        cls,
        year: int,
        quarter: Optional[str] = None,
    ) -> "TemporalFilter":
        """
        Create a filter for a specific fiscal period.

        Args:
            year: Fiscal year (e.g., 2021)
            quarter: Optional fiscal quarter (e.g., 'Q1', 'Q2', 'Q3', 'Q4')

        Returns:
            Fiscal period filter
        """
        # Normalize quarter to uppercase
        if quarter:
            quarter = quarter.upper()
            if not quarter.startswith("Q"):
                quarter = f"Q{quarter}"

        return cls(
            query_type=TemporalQueryType.POINT_IN_TIME,
            fiscal_year=year,
            fiscal_quarter=quarter,
        )

    @classmethod
    def create_fiscal_range(
        cls,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        start_quarter: Optional[str] = None,
        end_quarter: Optional[str] = None,
    ) -> "TemporalFilter":
        """
        Create a filter for a fiscal year/quarter range.

        Args:
            start_year: Start fiscal year
            end_year: End fiscal year
            start_quarter: Optional start quarter
            end_quarter: Optional end quarter

        Returns:
            Fiscal range filter
        """
        # Normalize quarters
        if start_quarter:
            start_quarter = start_quarter.upper()
            if not start_quarter.startswith("Q"):
                start_quarter = f"Q{start_quarter}"
        if end_quarter:
            end_quarter = end_quarter.upper()
            if not end_quarter.startswith("Q"):
                end_quarter = f"Q{end_quarter}"

        return cls(
            query_type=TemporalQueryType.TIME_RANGE,
            start_year=start_year,
            end_year=end_year,
            start_quarter=start_quarter,
            end_quarter=end_quarter,
        )

    @classmethod
    def create_point_in_time(cls, timestamp: datetime) -> "TemporalFilter":
        """
        Create a filter for a specific point in time.

        The datetime will be converted to the corresponding fiscal period.

        Args:
            timestamp: Point in time (will be converted to fiscal period)

        Returns:
            Fiscal period filter
        """
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
        """
        Create a filter for a time range.

        The datetimes will be converted to fiscal periods.

        Args:
            start: Start time (will be converted to fiscal period)
            end: End time (will be converted to fiscal period)

        Returns:
            Fiscal range filter
        """
        return cls(
            query_type=TemporalQueryType.TIME_RANGE,
            start_time=start,
            end_time=end,
        )

    @classmethod
    def create_history(cls) -> "TemporalFilter":
        """Create a filter for historical queries (all fiscal periods)."""
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
