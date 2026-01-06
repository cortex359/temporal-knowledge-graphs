"""Neo4j database client with connection pooling and query execution."""

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from neo4j import GraphDatabase, Driver, Session, Transaction, Result
from neo4j.exceptions import ServiceUnavailable, TransientError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from temporal_kg_rag.config.settings import get_settings
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jClient:
    """Neo4j database client with connection pooling and retry logic."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50,
        connection_acquisition_timeout: int = 60,
    ):
        """
        Initialize Neo4j client.

        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            max_connection_lifetime: Maximum connection lifetime in seconds
            max_connection_pool_size: Maximum connection pool size
            connection_acquisition_timeout: Connection acquisition timeout in seconds
        """
        settings = get_settings()

        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database

        self._driver: Optional[Driver] = None
        self._max_connection_lifetime = max_connection_lifetime
        self._max_connection_pool_size = max_connection_pool_size
        self._connection_acquisition_timeout = connection_acquisition_timeout

        logger.info(f"Neo4j client initialized for {self.uri}")

    @property
    def driver(self) -> Driver:
        """Get or create Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=self._max_connection_lifetime,
                max_connection_pool_size=self._max_connection_pool_size,
                connection_acquisition_timeout=self._connection_acquisition_timeout,
            )
            logger.info("Neo4j driver created")
        return self._driver

    def close(self) -> None:
        """Close the Neo4j driver and release resources."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")

    def __enter__(self) -> "Neo4jClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def verify_connectivity(self) -> bool:
        """
        Verify connectivity to Neo4j database.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                if record and record["test"] == 1:
                    logger.info("Neo4j connectivity verified")
                    return True
                return False
        except Exception as e:
            logger.error(f"Neo4j connectivity verification failed: {e}")
            return False

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Get a Neo4j session context manager.

        Yields:
            Neo4j session
        """
        session = self.driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ServiceUnavailable, TransientError)),
        reraise=True,
    )
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.

        Args:
            query: Cypher query string
            parameters: Query parameters
            timeout: Query timeout in seconds

        Returns:
            List of result records as dictionaries

        Raises:
            ServiceUnavailable: If Neo4j service is not available
            TransientError: If a transient error occurs
        """
        parameters = parameters or {}
        timeout = timeout or get_settings().query_timeout_seconds

        logger.debug(f"Executing query: {query[:100]}...")

        try:
            with self.session() as session:
                result = session.run(query, parameters, timeout=timeout)
                records = [dict(record) for record in result]
                logger.debug(f"Query returned {len(records)} records")
                return records
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ServiceUnavailable, TransientError)),
        reraise=True,
    )
    def execute_write_transaction(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a write transaction.

        Args:
            query: Cypher query string
            parameters: Query parameters
            timeout: Query timeout in seconds

        Returns:
            List of result records as dictionaries
        """
        parameters = parameters or {}
        timeout = timeout or get_settings().query_timeout_seconds

        logger.debug(f"Executing write transaction: {query[:100]}...")

        def _execute_query(tx: Transaction) -> List[Dict[str, Any]]:
            result = tx.run(query, parameters, timeout=timeout)
            return [dict(record) for record in result]

        try:
            with self.session() as session:
                records = session.execute_write(_execute_query)
                logger.debug(f"Write transaction returned {len(records)} records")
                return records
        except Exception as e:
            logger.error(f"Write transaction failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ServiceUnavailable, TransientError)),
        reraise=True,
    )
    def execute_read_transaction(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a read transaction.

        Args:
            query: Cypher query string
            parameters: Query parameters
            timeout: Query timeout in seconds

        Returns:
            List of result records as dictionaries
        """
        parameters = parameters or {}
        timeout = timeout or get_settings().query_timeout_seconds

        logger.debug(f"Executing read transaction: {query[:100]}...")

        def _execute_query(tx: Transaction) -> List[Dict[str, Any]]:
            result = tx.run(query, parameters, timeout=timeout)
            return [dict(record) for record in result]

        try:
            with self.session() as session:
                records = session.execute_read(_execute_query)
                logger.debug(f"Read transaction returned {len(records)} records")
                return records
        except Exception as e:
            logger.error(f"Read transaction failed: {e}")
            raise

    def vector_similarity_search(
        self,
        embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using Neo4j vector index.

        Args:
            embedding: Query embedding vector
            top_k: Number of top results to return
            filters: Optional filters to apply (e.g., temporal constraints)

        Returns:
            List of similar chunks with scores
        """
        # Build the query with optional filters
        filter_clause = "WHERE node.is_current = true"
        if filters:
            if "start_time" in filters and "end_time" in filters:
                filter_clause += """
                AND node.created_at >= $start_time
                AND node.created_at <= $end_time
                """
            elif "point_in_time" in filters:
                filter_clause += """
                AND node.created_at <= $point_in_time
                AND (node.superseded_at IS NULL OR node.superseded_at > $point_in_time)
                """

        query = f"""
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $embedding)
        YIELD node, score
        {filter_clause}
        MATCH (node)<-[:HAS_CHUNK]-(doc:Document)
        RETURN
            node.id AS chunk_id,
            node.text AS text,
            node.chunk_index AS chunk_index,
            node.created_at AS created_at,
            score,
            doc.id AS document_id,
            doc.title AS document_title,
            doc.created_at AS document_date
        ORDER BY score DESC
        LIMIT $top_k
        """

        parameters = {
            "embedding": embedding,
            "top_k": top_k,
        }
        if filters:
            parameters.update(filters)

        return self.execute_read_transaction(query, parameters)

    def get_database_stats(self) -> Dict[str, int]:
        """
        Get database statistics.

        Returns:
            Dictionary with node and relationship counts
        """
        query = """
        MATCH (d:Document) WITH count(d) AS doc_count
        MATCH (c:Chunk) WITH doc_count, count(c) AS chunk_count
        MATCH (e:Entity) WITH doc_count, chunk_count, count(e) AS entity_count
        MATCH ()-[r]->() WITH doc_count, chunk_count, entity_count, count(r) AS rel_count
        RETURN doc_count, chunk_count, entity_count, rel_count
        """

        result = self.execute_read_transaction(query)
        if result:
            return {
                "documents": result[0]["doc_count"],
                "chunks": result[0]["chunk_count"],
                "entities": result[0]["entity_count"],
                "relationships": result[0]["rel_count"],
            }
        return {"documents": 0, "chunks": 0, "entities": 0, "relationships": 0}


# Global client instance
_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    """Get the global Neo4j client instance."""
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client


def close_neo4j_client() -> None:
    """Close the global Neo4j client instance."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
