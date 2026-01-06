"""Temporal query builder for constructing time-aware Cypher queries."""

from datetime import datetime
from typing import Dict, Optional

from temporal_kg_rag.models.temporal import TemporalFilter, TemporalQueryType
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalQueryBuilder:
    """Build temporal Cypher query clauses."""

    def __init__(self):
        """Initialize query builder."""
        pass

    def build_where_clause(
        self,
        temporal_filter: TemporalFilter,
        node_var: str = "node",
    ) -> str:
        """
        Build WHERE clause for temporal filtering.

        Args:
            temporal_filter: Temporal filter configuration
            node_var: Variable name for the node in Cypher

        Returns:
            WHERE clause string
        """
        return temporal_filter.to_cypher_where_clause(node_var)

    def build_parameters(self, temporal_filter: TemporalFilter) -> Dict:
        """
        Build query parameters for temporal filter.

        Args:
            temporal_filter: Temporal filter configuration

        Returns:
            Parameters dictionary
        """
        return temporal_filter.to_cypher_parameters()

    def add_temporal_filter_to_query(
        self,
        base_query: str,
        temporal_filter: Optional[TemporalFilter],
        node_var: str = "node",
    ) -> tuple[str, Dict]:
        """
        Add temporal filtering to an existing query.

        Args:
            base_query: Base Cypher query
            temporal_filter: Optional temporal filter
            node_var: Variable name for the node

        Returns:
            Tuple of (modified query, parameters)
        """
        if temporal_filter is None:
            return base_query, {}

        where_clause = self.build_where_clause(temporal_filter, node_var)
        parameters = self.build_parameters(temporal_filter)

        # Add WHERE clause if not present
        if "WHERE" in base_query.upper():
            modified_query = base_query.replace(
                "WHERE",
                f"WHERE {where_clause} AND",
                1,
            )
        else:
            # Add WHERE before RETURN
            if "RETURN" in base_query.upper():
                modified_query = base_query.replace(
                    "RETURN",
                    f"WHERE {where_clause}\nRETURN",
                    1,
                )
            else:
                modified_query = f"{base_query}\nWHERE {where_clause}"

        return modified_query, parameters


# Global builder instance
_builder: Optional[TemporalQueryBuilder] = None


def get_temporal_query_builder() -> TemporalQueryBuilder:
    """Get the global temporal query builder instance."""
    global _builder
    if _builder is None:
        _builder = TemporalQueryBuilder()
    return _builder
