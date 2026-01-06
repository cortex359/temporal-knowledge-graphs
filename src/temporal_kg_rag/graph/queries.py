"""Complex Cypher queries for graph traversal and analysis."""

from typing import Any, Dict, List, Optional

from temporal_kg_rag.graph.neo4j_client import Neo4jClient, get_neo4j_client
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class GraphQueries:
    """Complex graph queries and traversals."""

    def __init__(self, client: Optional[Neo4jClient] = None):
        """
        Initialize graph queries.

        Args:
            client: Optional Neo4j client
        """
        self.client = client or get_neo4j_client()

    def find_related_entities(
        self,
        entity_id: str,
        max_depth: int = 2,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Find entities related to a given entity through graph traversal.

        Args:
            entity_id: Starting entity ID
            max_depth: Maximum traversal depth
            limit: Maximum number of results

        Returns:
            List of related entities with relationship information
        """
        query = f"""
        MATCH path = (start:Entity {{id: $entity_id}})-[*1..{max_depth}]-(related:Entity)
        WHERE start <> related
        WITH related, min(length(path)) as distance, count(*) as connection_strength
        RETURN
            related.id as entity_id,
            related.name as entity_name,
            related.type as entity_type,
            distance,
            connection_strength
        ORDER BY distance ASC, connection_strength DESC
        LIMIT $limit
        """

        return self.client.execute_read_transaction(query, {
            "entity_id": entity_id,
            "limit": limit,
        })

    def get_entity_cooccurrences(
        self,
        entity_id: str,
        min_cooccurrences: int = 2,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find entities that co-occur with the given entity in chunks.

        Args:
            entity_id: Entity ID
            min_cooccurrences: Minimum number of co-occurrences
            limit: Maximum number of results

        Returns:
            List of co-occurring entities
        """
        query = """
        MATCH (e1:Entity {id: $entity_id})<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(e2:Entity)
        WHERE e1 <> e2 AND c.is_current = true
        WITH e2, count(DISTINCT c) as cooccurrence_count
        WHERE cooccurrence_count >= $min_cooccurrences
        RETURN
            e2.id as entity_id,
            e2.name as entity_name,
            e2.type as entity_type,
            cooccurrence_count
        ORDER BY cooccurrence_count DESC
        LIMIT $limit
        """

        return self.client.execute_read_transaction(query, {
            "entity_id": entity_id,
            "min_cooccurrences": min_cooccurrences,
            "limit": limit,
        })

    def get_document_similarity_graph(
        self,
        document_id: str,
        similarity_threshold: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find documents similar to a given document based on shared entities.

        Args:
            document_id: Document ID
            similarity_threshold: Minimum number of shared entities

        Returns:
            List of similar documents
        """
        query = """
        MATCH (d1:Document {id: $document_id})-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(e:Entity)
        WITH d1, collect(DISTINCT e) as entities1
        MATCH (d2:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(e2:Entity)
        WHERE d1 <> d2 AND e2 IN entities1
        WITH d2, count(DISTINCT e2) as shared_entities
        WHERE shared_entities >= $similarity_threshold
        RETURN
            d2.id as document_id,
            d2.title as document_title,
            d2.created_at as document_date,
            shared_entities
        ORDER BY shared_entities DESC
        """

        return self.client.execute_read_transaction(query, {
            "document_id": document_id,
            "similarity_threshold": similarity_threshold,
        })


# Global queries instance
_queries: Optional[GraphQueries] = None


def get_graph_queries() -> GraphQueries:
    """Get the global graph queries instance."""
    global _queries
    if _queries is None:
        _queries = GraphQueries()
    return _queries
