"""Context expansion for enriching retrieved chunks with graph information."""

from typing import Dict, List, Optional, Set

from temporal_kg_rag.graph.neo4j_client import Neo4jClient, get_neo4j_client
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class ContextExpander:
    """Expand retrieved chunks with additional context from the knowledge graph."""

    def __init__(self, client: Optional[Neo4jClient] = None):
        """
        Initialize context expander.

        Args:
            client: Optional Neo4j client
        """
        self.client = client or get_neo4j_client()

    def expand_results(
        self,
        results: List[Dict],
        include_neighboring_chunks: bool = True,
        include_entities: bool = True,
        include_related_chunks: bool = True,
        include_document_context: bool = True,
        neighboring_chunk_window: int = 1,
    ) -> List[Dict]:
        """
        Expand search results with additional context.

        Args:
            results: List of search results
            include_neighboring_chunks: Add chunks before/after
            include_entities: Add entity information
            include_related_chunks: Add semantically related chunks
            include_document_context: Add document metadata
            neighboring_chunk_window: Number of chunks before/after

        Returns:
            Results with expanded context
        """
        logger.info(f"Expanding context for {len(results)} results")

        for result in results:
            chunk_id = result["chunk_id"]
            document_id = result.get("document_id")

            # Initialize context dictionary
            result["expanded_context"] = {}

            # Get neighboring chunks
            if include_neighboring_chunks and document_id:
                neighbors = self.get_neighboring_chunks(
                    document_id=document_id,
                    chunk_index=result.get("chunk_index"),
                    window=neighboring_chunk_window,
                )
                result["expanded_context"]["neighboring_chunks"] = neighbors

            # Get entity information
            if include_entities:
                entities = self.get_chunk_entities_with_details(chunk_id)
                result["expanded_context"]["entities"] = entities

            # Get related chunks
            if include_related_chunks:
                related = self.get_related_chunks(chunk_id, limit=3)
                result["expanded_context"]["related_chunks"] = related

            # Get document context
            if include_document_context and document_id:
                doc_context = self.get_document_context(document_id)
                result["expanded_context"]["document"] = doc_context

        logger.info("Context expansion complete")

        return results

    def get_neighboring_chunks(
        self,
        document_id: str,
        chunk_index: int,
        window: int = 1,
    ) -> Dict:
        """
        Get chunks before and after a given chunk.

        Args:
            document_id: Document ID
            chunk_index: Current chunk index
            window: Number of chunks before/after

        Returns:
            Dictionary with before and after chunks
        """
        query = """
        MATCH (doc:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.is_current = true
        AND c.chunk_index >= $start_index
        AND c.chunk_index <= $end_index
        AND c.chunk_index <> $current_index
        RETURN
            c.chunk_index as chunk_index,
            c.text as text,
            CASE
                WHEN c.chunk_index < $current_index THEN 'before'
                ELSE 'after'
            END as position
        ORDER BY c.chunk_index
        """

        results = self.client.execute_read_transaction(query, {
            "document_id": document_id,
            "start_index": max(0, chunk_index - window),
            "end_index": chunk_index + window,
            "current_index": chunk_index,
        })

        neighbors = {"before": [], "after": []}

        for result in results:
            position = result["position"]
            neighbors[position].append({
                "chunk_index": result["chunk_index"],
                "text": result["text"],
            })

        return neighbors

    def get_chunk_entities_with_details(self, chunk_id: str) -> List[Dict]:
        """
        Get detailed entity information for a chunk.

        Args:
            chunk_id: Chunk ID

        Returns:
            List of entities with details
        """
        query = """
        MATCH (c:Chunk {id: $chunk_id})-[m:MENTIONS]->(e:Entity)
        RETURN
            e.id as entity_id,
            e.name as entity_name,
            e.type as entity_type,
            e.mention_count as mention_count,
            m.confidence as confidence,
            m.context as mention_context
        ORDER BY m.confidence DESC, e.mention_count DESC
        """

        results = self.client.execute_read_transaction(query, {"chunk_id": chunk_id})

        return results

    def get_related_chunks(
        self,
        chunk_id: str,
        limit: int = 3,
    ) -> List[Dict]:
        """
        Get chunks related through shared entities.

        Args:
            chunk_id: Chunk ID
            limit: Maximum number of related chunks

        Returns:
            List of related chunks
        """
        query = """
        MATCH (c1:Chunk {id: $chunk_id})-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(c2:Chunk)
        WHERE c1 <> c2 AND c2.is_current = true
        WITH c2, count(DISTINCT e) as shared_entities
        MATCH (c2)<-[:HAS_CHUNK]-(doc:Document)
        RETURN
            c2.id as chunk_id,
            c2.text as text,
            c2.chunk_index as chunk_index,
            shared_entities,
            doc.title as document_title
        ORDER BY shared_entities DESC
        LIMIT $limit
        """

        results = self.client.execute_read_transaction(query, {
            "chunk_id": chunk_id,
            "limit": limit,
        })

        return results

    def get_document_context(self, document_id: str) -> Dict:
        """
        Get document-level context.

        Args:
            document_id: Document ID

        Returns:
            Document context dictionary
        """
        query = """
        MATCH (doc:Document {id: $document_id})
        OPTIONAL MATCH (doc)-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.is_current = true
        WITH doc, count(c) as chunk_count
        OPTIONAL MATCH (doc)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(e:Entity)
        WITH doc, chunk_count, collect(DISTINCT e.name) as entities
        RETURN
            doc.title as title,
            doc.source as source,
            doc.content_type as content_type,
            doc.created_at as created_at,
            doc.metadata as metadata,
            chunk_count,
            entities[..10] as top_entities
        """

        results = self.client.execute_read_transaction(query, {"document_id": document_id})

        return results[0] if results else {}

    def expand_with_entity_graph(
        self,
        results: List[Dict],
        max_depth: int = 1,
    ) -> List[Dict]:
        """
        Expand results with entity graph neighborhoods.

        Args:
            results: List of search results
            max_depth: Maximum graph traversal depth

        Returns:
            Results with entity graph context
        """
        logger.info(f"Expanding with entity graph (depth={max_depth})")

        for result in results:
            chunk_id = result["chunk_id"]

            # Get entity graph neighborhood
            query = f"""
            MATCH (c:Chunk {{id: $chunk_id}})-[:MENTIONS]->(e1:Entity)
            OPTIONAL MATCH path = (e1)-[*1..{max_depth}]-(e2:Entity)
            WHERE e1 <> e2
            WITH e1, e2, path,
                 CASE WHEN e2 IS NULL THEN 0 ELSE length(path) END as distance
            RETURN
                e1.name as entity_name,
                e1.type as entity_type,
                collect(DISTINCT {{
                    name: e2.name,
                    type: e2.type,
                    distance: distance
                }}) as related_entities
            """

            entity_graph = self.client.execute_read_transaction(query, {"chunk_id": chunk_id})

            if "expanded_context" not in result:
                result["expanded_context"] = {}

            result["expanded_context"]["entity_graph"] = entity_graph

        logger.info("Entity graph expansion complete")

        return results

    def build_context_summary(self, expanded_results: List[Dict]) -> str:
        """
        Build a text summary of expanded context for RAG.

        Args:
            expanded_results: Results with expanded context

        Returns:
            Context summary text
        """
        summary_parts = []

        for i, result in enumerate(expanded_results, 1):
            chunk_text = result.get("text", "")
            doc_title = result.get("document_title", "Unknown")

            # Add main chunk
            summary_parts.append(f"[Result {i} from '{doc_title}']")
            summary_parts.append(chunk_text)

            # Add entity information
            if "expanded_context" in result and "entities" in result["expanded_context"]:
                entities = result["expanded_context"]["entities"]
                if entities:
                    entity_names = [e["entity_name"] for e in entities[:5]]
                    summary_parts.append(f"Key entities: {', '.join(entity_names)}")

            # Add neighboring context
            if "expanded_context" in result and "neighboring_chunks" in result["expanded_context"]:
                neighbors = result["expanded_context"]["neighboring_chunks"]

                if neighbors.get("before"):
                    summary_parts.append("[Previous context]")
                    for nb in neighbors["before"][-1:]:  # Last before chunk
                        summary_parts.append(nb["text"][:200] + "...")

                if neighbors.get("after"):
                    summary_parts.append("[Following context]")
                    for nb in neighbors["after"][:1]:  # First after chunk
                        summary_parts.append(nb["text"][:200] + "...")

            summary_parts.append("")  # Empty line between results

        return "\n".join(summary_parts)


# Global context expander instance
_context_expander: Optional[ContextExpander] = None


def get_context_expander() -> ContextExpander:
    """Get the global context expander instance."""
    global _context_expander
    if _context_expander is None:
        _context_expander = ContextExpander()
    return _context_expander


def expand_results(results: List[Dict]) -> List[Dict]:
    """
    Convenience function to expand results.

    Args:
        results: Search results

    Returns:
        Expanded results
    """
    expander = get_context_expander()
    return expander.expand_results(results)
