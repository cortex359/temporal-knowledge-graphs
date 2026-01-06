"""Graph-based search using entity relationships and traversal."""

from typing import Dict, List, Optional, Set

from temporal_kg_rag.config.settings import get_settings
from temporal_kg_rag.graph.neo4j_client import Neo4jClient, get_neo4j_client
from temporal_kg_rag.models.temporal import TemporalFilter
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class GraphSearch:
    """Graph-based search using entity relationships."""

    def __init__(self, client: Optional[Neo4jClient] = None):
        """
        Initialize graph search.

        Args:
            client: Optional Neo4j client
        """
        self.client = client or get_neo4j_client()
        self.settings = get_settings()

    def search_by_entities(
        self,
        entity_names: List[str],
        top_k: int = 10,
        temporal_filter: Optional[TemporalFilter] = None,
        match_all: bool = False,
    ) -> List[Dict]:
        """
        Search for chunks that mention specific entities.

        Args:
            entity_names: List of entity names to search for
            top_k: Number of results to return
            temporal_filter: Optional temporal filter
            match_all: If True, only return chunks mentioning all entities

        Returns:
            List of chunks mentioning the entities
        """
        logger.info(
            f"Graph search by entities: {entity_names} "
            f"(match_all={match_all}, top_k={top_k})"
        )

        if match_all:
            # Chunks must mention ALL entities
            query = """
            MATCH (e:Entity)
            WHERE e.name IN $entity_names
            WITH collect(e) as entities
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE e IN entities AND c.is_current = true
            """

            # Add temporal filtering
            if temporal_filter:
                temporal_clause = temporal_filter.to_cypher_where_clause("c")
                if temporal_clause and temporal_clause != "true":
                    query += f"AND {temporal_clause}\n"

            query += """
            WITH c, collect(DISTINCT e.name) as mentioned_entities
            WHERE size(mentioned_entities) = size($entity_names)
            MATCH (c)<-[:HAS_CHUNK]-(doc:Document)
            RETURN
                c.id AS chunk_id,
                c.text AS text,
                c.chunk_index AS chunk_index,
                c.created_at AS created_at,
                1.0 AS score,
                doc.id AS document_id,
                doc.title AS document_title,
                mentioned_entities as entities
            ORDER BY c.created_at DESC
            LIMIT $top_k
            """
        else:
            # Chunks can mention ANY of the entities
            query = """
            MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)
            WHERE e.name IN $entity_names
            AND c.is_current = true
            """

            # Add temporal filtering
            if temporal_filter:
                temporal_clause = temporal_filter.to_cypher_where_clause("c")
                if temporal_clause and temporal_clause != "true":
                    query += f"AND {temporal_clause}\n"

            query += """
            WITH c, count(DISTINCT e) as entity_match_count
            MATCH (c)<-[:HAS_CHUNK]-(doc:Document)
            OPTIONAL MATCH (c)-[:MENTIONS]->(all_e:Entity)
            WITH c, doc, entity_match_count, collect(DISTINCT all_e.name) as entities
            RETURN
                c.id AS chunk_id,
                c.text AS text,
                c.chunk_index AS chunk_index,
                c.created_at AS created_at,
                toFloat(entity_match_count) / size($entity_names) AS score,
                doc.id AS document_id,
                doc.title AS document_title,
                entities
            ORDER BY score DESC, c.created_at DESC
            LIMIT $top_k
            """

        parameters = {
            "entity_names": entity_names,
            "top_k": top_k,
        }

        if temporal_filter:
            parameters.update(temporal_filter.to_cypher_parameters())

        results = self.client.execute_read_transaction(query, parameters)

        logger.info(f"Found {len(results)} chunks mentioning entities")

        return results

    def search_by_entity_type(
        self,
        entity_type: str,
        top_k: int = 10,
        temporal_filter: Optional[TemporalFilter] = None,
    ) -> List[Dict]:
        """
        Search for chunks mentioning entities of a specific type.

        Args:
            entity_type: Type of entity (PERSON, ORG, LOCATION, etc.)
            top_k: Number of results to return
            temporal_filter: Optional temporal filter

        Returns:
            List of chunks
        """
        logger.info(f"Graph search by entity type: {entity_type}")

        query = """
        MATCH (e:Entity {type: $entity_type})<-[:MENTIONS]-(c:Chunk)
        WHERE c.is_current = true
        """

        if temporal_filter:
            temporal_clause = temporal_filter.to_cypher_where_clause("c")
            if temporal_clause and temporal_clause != "true":
                query += f"AND {temporal_clause}\n"

        query += """
        WITH c, count(DISTINCT e) as entity_count
        MATCH (c)<-[:HAS_CHUNK]-(doc:Document)
        OPTIONAL MATCH (c)-[:MENTIONS]->(all_e:Entity)
        WHERE all_e.type = $entity_type
        RETURN
            c.id AS chunk_id,
            c.text AS text,
            c.chunk_index AS chunk_index,
            c.created_at AS created_at,
            toFloat(entity_count) AS score,
            doc.id AS document_id,
            doc.title AS document_title,
            collect(DISTINCT all_e.name) as entities
        ORDER BY score DESC
        LIMIT $top_k
        """

        parameters = {
            "entity_type": entity_type,
            "top_k": top_k,
        }

        if temporal_filter:
            parameters.update(temporal_filter.to_cypher_parameters())

        results = self.client.execute_read_transaction(query, parameters)

        logger.info(f"Found {len(results)} chunks with {entity_type} entities")

        return results

    def search_by_related_entities(
        self,
        entity_name: str,
        max_depth: int = 2,
        top_k: int = 10,
        temporal_filter: Optional[TemporalFilter] = None,
    ) -> List[Dict]:
        """
        Search for chunks mentioning entities related to a given entity.

        Args:
            entity_name: Starting entity name
            max_depth: Maximum traversal depth
            top_k: Number of results to return
            temporal_filter: Optional temporal filter

        Returns:
            List of chunks
        """
        logger.info(
            f"Graph search by related entities: {entity_name} (depth={max_depth})"
        )

        max_depth = min(max_depth, self.settings.max_traversal_depth)

        query = f"""
        MATCH (start:Entity {{name: $entity_name}})
        MATCH path = (start)-[*1..{max_depth}]-(related:Entity)
        WHERE start <> related
        WITH related, min(length(path)) as distance
        MATCH (c:Chunk)-[:MENTIONS]->(related)
        WHERE c.is_current = true
        """

        if temporal_filter:
            temporal_clause = temporal_filter.to_cypher_where_clause("c")
            if temporal_clause and temporal_clause != "true":
                query += f"AND {temporal_clause}\n"

        query += """
        WITH c, collect(DISTINCT {name: related.name, distance: distance}) as related_entities
        MATCH (c)<-[:HAS_CHUNK]-(doc:Document)
        RETURN
            c.id AS chunk_id,
            c.text AS text,
            c.chunk_index AS chunk_index,
            c.created_at AS created_at,
            1.0 / (1.0 + reduce(sum=0.0, r IN related_entities | sum + r.distance)) AS score,
            doc.id AS document_id,
            doc.title AS document_title,
            [r IN related_entities | r.name] as related_entity_names
        ORDER BY score DESC
        LIMIT $top_k
        """

        parameters = {
            "entity_name": entity_name,
            "top_k": top_k,
        }

        if temporal_filter:
            parameters.update(temporal_filter.to_cypher_parameters())

        results = self.client.execute_read_transaction(query, parameters)

        logger.info(f"Found {len(results)} chunks with related entities")

        return results

    def search_by_cooccurrence(
        self,
        entity_name: str,
        min_cooccurrences: int = 2,
        top_k: int = 10,
        temporal_filter: Optional[TemporalFilter] = None,
    ) -> List[Dict]:
        """
        Search based on entity co-occurrence patterns.

        Args:
            entity_name: Entity name
            min_cooccurrences: Minimum number of co-occurrences
            top_k: Number of results to return
            temporal_filter: Optional temporal filter

        Returns:
            List of chunks with co-occurring entities
        """
        logger.info(f"Graph search by co-occurrence: {entity_name}")

        # First, find entities that co-occur with the target entity
        cooccurrence_query = """
        MATCH (e1:Entity {name: $entity_name})<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(e2:Entity)
        WHERE e1 <> e2 AND c.is_current = true
        WITH e2, count(DISTINCT c) as cooccurrence_count
        WHERE cooccurrence_count >= $min_cooccurrences
        RETURN e2.name as entity_name, cooccurrence_count
        ORDER BY cooccurrence_count DESC
        LIMIT 10
        """

        cooccurring_entities = self.client.execute_read_transaction(
            cooccurrence_query,
            {
                "entity_name": entity_name,
                "min_cooccurrences": min_cooccurrences,
            },
        )

        if not cooccurring_entities:
            logger.info("No co-occurring entities found")
            return []

        # Extract entity names
        entity_names = [e["entity_name"] for e in cooccurring_entities]
        logger.info(f"Found {len(entity_names)} co-occurring entities")

        # Search for chunks mentioning these entities
        return self.search_by_entities(
            entity_names=[entity_name] + entity_names,
            top_k=top_k,
            temporal_filter=temporal_filter,
            match_all=False,
        )

    def extract_entities_from_query(self, query: str) -> List[str]:
        """
        Extract potential entity names from query text.

        This is a simple implementation that looks for capitalized words.
        Could be enhanced with NER.

        Args:
            query: Query text

        Returns:
            List of potential entity names
        """
        import re

        # Find sequences of capitalized words
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities = re.findall(pattern, query)

        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)

        return unique_entities

    def search(
        self,
        query: str,
        top_k: int = 10,
        temporal_filter: Optional[TemporalFilter] = None,
        use_entity_extraction: bool = True,
    ) -> List[Dict]:
        """
        Main graph search method that tries multiple strategies.

        Args:
            query: Search query
            top_k: Number of results to return
            temporal_filter: Optional temporal filter
            use_entity_extraction: Whether to extract entities from query

        Returns:
            List of search results
        """
        logger.info(f"Graph search: '{query[:50]}...'")

        results = []

        if use_entity_extraction:
            # Extract entity names from query
            entity_names = self.extract_entities_from_query(query)

            if entity_names:
                logger.info(f"Extracted entities from query: {entity_names}")
                results = self.search_by_entities(
                    entity_names=entity_names,
                    top_k=top_k,
                    temporal_filter=temporal_filter,
                    match_all=False,
                )

        if not results:
            logger.info("No results from entity extraction, falling back to text search")
            # Fallback: full-text search on chunks
            results = self._fulltext_search(query, top_k, temporal_filter)

        return results

    def _fulltext_search(
        self,
        query: str,
        top_k: int,
        temporal_filter: Optional[TemporalFilter],
    ) -> List[Dict]:
        """Fallback full-text search."""
        cypher_query = """
        CALL db.index.fulltext.queryNodes('chunk_text', $query)
        YIELD node, score
        WHERE node.is_current = true
        """

        if temporal_filter:
            temporal_clause = temporal_filter.to_cypher_where_clause("node")
            if temporal_clause and temporal_clause != "true":
                cypher_query += f"AND {temporal_clause}\n"

        cypher_query += """
        MATCH (node)<-[:HAS_CHUNK]-(doc:Document)
        RETURN
            node.id AS chunk_id,
            node.text AS text,
            node.chunk_index AS chunk_index,
            node.created_at AS created_at,
            score,
            doc.id AS document_id,
            doc.title AS document_title
        ORDER BY score DESC
        LIMIT $top_k
        """

        parameters = {
            "query": query,
            "top_k": top_k,
        }

        if temporal_filter:
            parameters.update(temporal_filter.to_cypher_parameters())

        return self.client.execute_read_transaction(cypher_query, parameters)


# Global search instance
_graph_search: Optional[GraphSearch] = None


def get_graph_search() -> GraphSearch:
    """Get the global graph search instance."""
    global _graph_search
    if _graph_search is None:
        _graph_search = GraphSearch()
    return _graph_search
