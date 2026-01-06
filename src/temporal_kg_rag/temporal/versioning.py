"""Version management for chunks and documents."""

from datetime import datetime
from typing import List, Optional

from temporal_kg_rag.graph.neo4j_client import Neo4jClient, get_neo4j_client
from temporal_kg_rag.models.chunk import Chunk
from temporal_kg_rag.models.document import Document
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class VersionManager:
    """Manage versions of chunks and documents."""

    def __init__(self, client: Optional[Neo4jClient] = None):
        """
        Initialize version manager.

        Args:
            client: Optional Neo4j client (creates one if not provided)
        """
        self.client = client or get_neo4j_client()

    def create_chunk_version(
        self,
        old_chunk: Chunk,
        new_chunk: Chunk,
        reason: str = "update",
    ) -> None:
        """
        Create a new version of a chunk and link it to the old version.

        Args:
            old_chunk: Previous version of the chunk
            new_chunk: New version of the chunk
            reason: Reason for supersession (update, correction, etc.)
        """
        # Mark old chunk as superseded
        old_chunk.supersede()

        # Increment version on new chunk
        new_chunk.version = old_chunk.version + 1
        new_chunk.is_current = True

        logger.info(
            f"Creating chunk version {new_chunk.version} "
            f"(superseding version {old_chunk.version})"
        )

        # Create SUPERSEDES relationship in Neo4j
        query = """
        MATCH (old:Chunk {id: $old_chunk_id})
        MATCH (new:Chunk {id: $new_chunk_id})
        SET old.is_current = false,
            old.superseded_at = $superseded_at
        MERGE (new)-[r:SUPERSEDES {
            superseded_at: $superseded_at,
            reason: $reason
        }]->(old)
        RETURN new, old, r
        """

        parameters = {
            "old_chunk_id": old_chunk.id,
            "new_chunk_id": new_chunk.id,
            "superseded_at": datetime.now(),
            "reason": reason,
        }

        self.client.execute_write_transaction(query, parameters)

        logger.info(f"Chunk version created successfully")

    def get_chunk_history(self, chunk_id: str) -> List[Chunk]:
        """
        Get version history of a chunk.

        Args:
            chunk_id: Chunk ID

        Returns:
            List of chunk versions ordered from newest to oldest
        """
        query = """
        MATCH path = (current:Chunk {id: $chunk_id})-[:SUPERSEDES*0..]->(older:Chunk)
        WITH nodes(path) as versions
        UNWIND versions as chunk
        RETURN chunk
        ORDER BY chunk.version DESC
        """

        results = self.client.execute_read_transaction(query, {"chunk_id": chunk_id})

        chunks = [Chunk.from_neo4j_dict(result["chunk"]) for result in results]

        logger.info(f"Retrieved {len(chunks)} versions for chunk {chunk_id}")

        return chunks

    def get_current_version(self, chunk_id: str) -> Optional[Chunk]:
        """
        Get the current (most recent) version of a chunk.

        Args:
            chunk_id: Any version of the chunk

        Returns:
            Current version of the chunk or None if not found
        """
        query = """
        MATCH path = (current:Chunk)-[:SUPERSEDES*0..]->(target:Chunk {id: $chunk_id})
        WHERE current.is_current = true
        RETURN current
        LIMIT 1
        """

        results = self.client.execute_read_transaction(query, {"chunk_id": chunk_id})

        if results:
            return Chunk.from_neo4j_dict(results[0]["current"])

        return None

    def get_version_at_time(
        self,
        chunk_id: str,
        timestamp: datetime,
    ) -> Optional[Chunk]:
        """
        Get the version of a chunk that was current at a specific time.

        Args:
            chunk_id: Any version of the chunk
            timestamp: Point in time

        Returns:
            Chunk version that was current at that time or None
        """
        query = """
        MATCH path = (current:Chunk)-[:SUPERSEDES*0..]->(target:Chunk {id: $chunk_id})
        MATCH (version:Chunk) WHERE version IN nodes(path)
        AND version.created_at <= $timestamp
        AND (version.superseded_at IS NULL OR version.superseded_at > $timestamp)
        RETURN version
        ORDER BY version.version DESC
        LIMIT 1
        """

        parameters = {
            "chunk_id": chunk_id,
            "timestamp": timestamp,
        }

        results = self.client.execute_read_transaction(query, parameters)

        if results:
            return Chunk.from_neo4j_dict(results[0]["version"])

        return None

    def get_chunks_modified_in_range(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Chunk]:
        """
        Get chunks that were modified within a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of chunks
        """
        query = """
        MATCH (c:Chunk)
        WHERE c.created_at >= $start_time AND c.created_at <= $end_time
        OR (c.superseded_at >= $start_time AND c.superseded_at <= $end_time)
        RETURN c
        ORDER BY c.created_at DESC
        """

        parameters = {
            "start_time": start_time,
            "end_time": end_time,
        }

        results = self.client.execute_read_transaction(query, parameters)

        chunks = [Chunk.from_neo4j_dict(result["c"]) for result in results]

        logger.info(
            f"Found {len(chunks)} chunks modified between "
            f"{start_time.isoformat()} and {end_time.isoformat()}"
        )

        return chunks

    def count_versions(self, chunk_id: str) -> int:
        """
        Count number of versions of a chunk.

        Args:
            chunk_id: Chunk ID

        Returns:
            Number of versions
        """
        query = """
        MATCH path = (current:Chunk)-[:SUPERSEDES*0..]->(target:Chunk {id: $chunk_id})
        RETURN length(path) + 1 as version_count
        """

        results = self.client.execute_read_transaction(query, {"chunk_id": chunk_id})

        if results:
            return results[0]["version_count"]

        return 0

    def delete_old_versions(
        self,
        keep_versions: int = 5,
        dry_run: bool = True,
    ) -> int:
        """
        Delete old versions beyond a threshold (for cleanup).

        Args:
            keep_versions: Number of versions to keep
            dry_run: If True, only count what would be deleted

        Returns:
            Number of versions deleted (or would be deleted in dry_run)
        """
        query = """
        MATCH path = (current:Chunk)-[:SUPERSEDES*]->(old:Chunk)
        WHERE current.is_current = true
        AND length(path) > $keep_versions
        WITH old
        """

        if dry_run:
            query += "RETURN count(old) as count"
        else:
            query += "DETACH DELETE old RETURN count(*) as count"

        parameters = {"keep_versions": keep_versions}

        results = self.client.execute_write_transaction(query, parameters)

        count = results[0]["count"] if results else 0

        if dry_run:
            logger.info(f"Would delete {count} old versions (keep_versions={keep_versions})")
        else:
            logger.info(f"Deleted {count} old versions (keep_versions={keep_versions})")

        return count


# Global version manager instance
_version_manager: Optional[VersionManager] = None


def get_version_manager() -> VersionManager:
    """Get the global version manager instance."""
    global _version_manager
    if _version_manager is None:
        _version_manager = VersionManager()
    return _version_manager
