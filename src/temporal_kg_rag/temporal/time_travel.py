"""Time travel queries for point-in-time data retrieval."""

from datetime import datetime
from typing import List, Optional

from temporal_kg_rag.graph.neo4j_client import Neo4jClient, get_neo4j_client
from temporal_kg_rag.models.chunk import Chunk
from temporal_kg_rag.models.document import Document
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class TimeTraveler:
    """Query historical states of the knowledge graph."""

    def __init__(self, client: Optional[Neo4jClient] = None):
        """
        Initialize time traveler.

        Args:
            client: Optional Neo4j client
        """
        self.client = client or get_neo4j_client()

    def get_documents_at_time(self, timestamp: datetime) -> List[Document]:
        """
        Get documents that existed at a specific point in time.

        Args:
            timestamp: Point in time

        Returns:
            List of documents
        """
        query = """
        MATCH (d:Document)
        WHERE d.created_at <= $timestamp
        RETURN d
        ORDER BY d.created_at DESC
        """

        results = self.client.execute_read_transaction(
            query,
            {"timestamp": timestamp},
        )

        documents = [Document.from_neo4j_dict(r["d"]) for r in results]

        logger.info(
            f"Found {len(documents)} documents at {timestamp.isoformat()}"
        )

        return documents

    def get_chunks_at_time(
        self,
        timestamp: datetime,
        document_id: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Get chunks that were current at a specific point in time.

        Args:
            timestamp: Point in time
            document_id: Optional filter by document

        Returns:
            List of chunks
        """
        query = """
        MATCH (c:Chunk)
        WHERE c.created_at <= $timestamp
        AND (c.superseded_at IS NULL OR c.superseded_at > $timestamp)
        """

        parameters = {"timestamp": timestamp}

        if document_id:
            query += """
            MATCH (c)<-[:HAS_CHUNK]-(d:Document {id: $document_id})
            """
            parameters["document_id"] = document_id

        query += """
        RETURN c
        ORDER BY c.created_at DESC
        """

        results = self.client.execute_read_transaction(query, parameters)

        chunks = [Chunk.from_neo4j_dict(r["c"]) for r in results]

        logger.info(
            f"Found {len(chunks)} chunks at {timestamp.isoformat()}"
        )

        return chunks

    def compare_time_periods(
        self,
        time1: datetime,
        time2: datetime,
    ) -> dict:
        """
        Compare the knowledge graph state between two time periods.

        Args:
            time1: First point in time
            time2: Second point in time

        Returns:
            Dictionary with comparison statistics
        """
        # Get documents at each time
        docs1 = self.get_documents_at_time(time1)
        docs2 = self.get_documents_at_time(time2)

        # Get chunks at each time
        chunks1 = self.get_chunks_at_time(time1)
        chunks2 = self.get_chunks_at_time(time2)

        comparison = {
            "time1": time1.isoformat(),
            "time2": time2.isoformat(),
            "documents_at_time1": len(docs1),
            "documents_at_time2": len(docs2),
            "documents_added": len(docs2) - len(docs1),
            "chunks_at_time1": len(chunks1),
            "chunks_at_time2": len(chunks2),
            "chunks_added": len(chunks2) - len(chunks1),
        }

        logger.info(f"Time period comparison: {comparison}")

        return comparison


# Global time traveler instance
_time_traveler: Optional[TimeTraveler] = None


def get_time_traveler() -> TimeTraveler:
    """Get the global time traveler instance."""
    global _time_traveler
    if _time_traveler is None:
        _time_traveler = TimeTraveler()
    return _time_traveler
