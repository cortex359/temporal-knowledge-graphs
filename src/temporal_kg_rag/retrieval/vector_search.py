"""Vector similarity search using Neo4j vector index."""

from typing import Dict, List, Optional

from temporal_kg_rag.config.settings import get_settings
from temporal_kg_rag.embeddings.generator import EmbeddingGenerator, get_embedding_generator
from temporal_kg_rag.graph.neo4j_client import Neo4jClient, get_neo4j_client
from temporal_kg_rag.models.temporal import TemporalFilter
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class VectorSearch:
    """Vector similarity search using Neo4j vector index."""

    def __init__(
        self,
        client: Optional[Neo4jClient] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ):
        """
        Initialize vector search.

        Args:
            client: Optional Neo4j client
            embedding_generator: Optional embedding generator
        """
        self.client = client or get_neo4j_client()
        self.embedding_generator = embedding_generator or get_embedding_generator()
        self.settings = get_settings()

    def search(
        self,
        query: str,
        top_k: int = 10,
        temporal_filter: Optional[TemporalFilter] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Search for chunks similar to the query.

        Args:
            query: Search query text
            top_k: Number of results to return
            temporal_filter: Optional temporal filter
            similarity_threshold: Minimum similarity score

        Returns:
            List of search results with chunks and scores
        """
        logger.info(f"Vector search: '{query[:50]}...' (top_k={top_k})")

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)

        # Build the vector search query
        cypher_query = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $embedding)
        YIELD node, score
        """

        # Add temporal filtering
        where_clauses = ["node.is_current = true"]

        if temporal_filter:
            temporal_clause = temporal_filter.to_cypher_where_clause("node")
            if temporal_clause and temporal_clause != "true":
                where_clauses.append(temporal_clause)

        # Add similarity threshold
        threshold = similarity_threshold or self.settings.vector_similarity_threshold
        where_clauses.append(f"score >= {threshold}")

        cypher_query += f"WHERE {' AND '.join(where_clauses)}\n"

        # Get document information
        cypher_query += """
        MATCH (node)<-[:HAS_CHUNK]-(doc:Document)
        OPTIONAL MATCH (node)-[:MENTIONS]->(e:Entity)
        WITH node, score, doc, collect(DISTINCT {
            id: e.id,
            name: e.name,
            type: e.type
        }) as entities
        RETURN
            node.id AS chunk_id,
            node.text AS text,
            node.chunk_index AS chunk_index,
            node.token_count AS token_count,
            node.created_at AS created_at,
            node.version AS version,
            score,
            doc.id AS document_id,
            doc.title AS document_title,
            doc.source AS document_source,
            doc.created_at AS document_date,
            entities
        ORDER BY score DESC
        LIMIT $top_k
        """

        # Prepare parameters
        parameters = {
            "embedding": query_embedding,
            "top_k": top_k,
        }

        if temporal_filter:
            parameters.update(temporal_filter.to_cypher_parameters())

        # Execute query
        results = self.client.execute_read_transaction(cypher_query, parameters)

        logger.info(f"Found {len(results)} results")

        # Log score distribution
        if results:
            scores = [r["score"] for r in results]
            logger.debug(
                f"Score range: {min(scores):.3f} - {max(scores):.3f}, "
                f"avg: {sum(scores)/len(scores):.3f}"
            )

        return results

    def search_with_context(
        self,
        query: str,
        top_k: int = 10,
        temporal_filter: Optional[TemporalFilter] = None,
        context_window: int = 1,
    ) -> List[Dict]:
        """
        Search for chunks and include neighboring chunks for context.

        Args:
            query: Search query text
            top_k: Number of results to return
            temporal_filter: Optional temporal filter
            context_window: Number of chunks before/after to include

        Returns:
            List of search results with context chunks
        """
        logger.info(f"Vector search with context (window={context_window})")

        # First, get the main results
        results = self.search(query, top_k, temporal_filter)

        # For each result, get neighboring chunks
        for result in results:
            document_id = result["document_id"]
            chunk_index = result["chunk_index"]

            # Get neighboring chunks
            neighbor_query = """
            MATCH (doc:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
            WHERE c.is_current = true
            AND c.chunk_index >= $start_index
            AND c.chunk_index <= $end_index
            AND c.chunk_index <> $current_index
            RETURN
                c.chunk_index as chunk_index,
                c.text as text
            ORDER BY c.chunk_index
            """

            neighbors = self.client.execute_read_transaction(neighbor_query, {
                "document_id": document_id,
                "start_index": max(0, chunk_index - context_window),
                "end_index": chunk_index + context_window,
                "current_index": chunk_index,
            })

            result["context_chunks"] = neighbors

        logger.info(f"Added context to {len(results)} results")

        return results

    def find_similar_chunks(
        self,
        chunk_id: str,
        top_k: int = 5,
        exclude_same_document: bool = True,
    ) -> List[Dict]:
        """
        Find chunks similar to a given chunk.

        Args:
            chunk_id: ID of the reference chunk
            top_k: Number of similar chunks to return
            exclude_same_document: Whether to exclude chunks from same document

        Returns:
            List of similar chunks
        """
        logger.info(f"Finding similar chunks to {chunk_id}")

        # Get the embedding of the reference chunk
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        RETURN c.embedding as embedding, c.id as id
        """

        result = self.client.execute_read_transaction(query, {"chunk_id": chunk_id})

        if not result or not result[0].get("embedding"):
            logger.warning(f"Chunk {chunk_id} not found or has no embedding")
            return []

        embedding = result[0]["embedding"]

        # Search for similar chunks
        similar_query = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k_plus_one, $embedding)
        YIELD node, score
        WHERE node.is_current = true
        AND node.id <> $chunk_id
        """

        if exclude_same_document:
            similar_query += """
            MATCH (node)<-[:HAS_CHUNK]-(doc:Document)
            MATCH (ref:Chunk {id: $chunk_id})<-[:HAS_CHUNK]-(ref_doc:Document)
            WHERE doc.id <> ref_doc.id
            """

        similar_query += """
        MATCH (node)<-[:HAS_CHUNK]-(doc:Document)
        RETURN
            node.id AS chunk_id,
            node.text AS text,
            node.chunk_index AS chunk_index,
            score,
            doc.id AS document_id,
            doc.title AS document_title
        ORDER BY score DESC
        LIMIT $top_k
        """

        parameters = {
            "embedding": embedding,
            "chunk_id": chunk_id,
            "top_k": top_k,
            "top_k_plus_one": top_k + 1,  # Query extra to account for self
        }

        similar_chunks = self.client.execute_read_transaction(similar_query, parameters)

        logger.info(f"Found {len(similar_chunks)} similar chunks")

        return similar_chunks


# Global search instance
_vector_search: Optional[VectorSearch] = None


def get_vector_search() -> VectorSearch:
    """Get the global vector search instance."""
    global _vector_search
    if _vector_search is None:
        _vector_search = VectorSearch()
    return _vector_search


def search(query: str, top_k: int = 10) -> List[Dict]:
    """
    Convenience function for vector search.

    Args:
        query: Search query text
        top_k: Number of results to return

    Returns:
        List of search results
    """
    vs = get_vector_search()
    return vs.search(query, top_k)
