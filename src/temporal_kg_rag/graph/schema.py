"""Neo4j schema initialization and management."""

from typing import List, Dict, Any

from temporal_kg_rag.graph.neo4j_client import Neo4jClient, get_neo4j_client
from temporal_kg_rag.config.settings import get_settings
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


def create_constraints(client: Neo4jClient) -> None:
    """
    Create unique constraints for node IDs.

    Args:
        client: Neo4j client instance
    """
    constraints = [
        # Document ID unique constraint
        """
        CREATE CONSTRAINT document_id_unique IF NOT EXISTS
        FOR (d:Document) REQUIRE d.id IS UNIQUE
        """,
        # Chunk ID unique constraint
        """
        CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
        FOR (c:Chunk) REQUIRE c.id IS UNIQUE
        """,
        # Entity ID unique constraint
        """
        CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
        FOR (e:Entity) REQUIRE e.id IS UNIQUE
        """,
        # Topic ID unique constraint
        """
        CREATE CONSTRAINT topic_id_unique IF NOT EXISTS
        FOR (t:Topic) REQUIRE t.id IS UNIQUE
        """,
        # TimeSnapshot ID unique constraint
        """
        CREATE CONSTRAINT timesnapshot_id_unique IF NOT EXISTS
        FOR (ts:TimeSnapshot) REQUIRE ts.id IS UNIQUE
        """,
    ]

    for constraint in constraints:
        try:
            client.execute_query(constraint.strip())
            logger.info(f"Constraint created successfully")
        except Exception as e:
            logger.warning(f"Constraint creation failed (may already exist): {e}")


def create_indexes(client: Neo4jClient) -> None:
    """
    Create performance indexes.

    Args:
        client: Neo4j client instance
    """
    indexes = [
        # Document created_at index
        """
        CREATE INDEX document_created_at IF NOT EXISTS
        FOR (d:Document) ON (d.created_at)
        """,
        # Document title index
        """
        CREATE INDEX document_title IF NOT EXISTS
        FOR (d:Document) ON (d.title)
        """,
        # Chunk created_at index
        """
        CREATE INDEX chunk_created_at IF NOT EXISTS
        FOR (c:Chunk) ON (c.created_at)
        """,
        # Chunk is_current index
        """
        CREATE INDEX chunk_is_current IF NOT EXISTS
        FOR (c:Chunk) ON (c.is_current)
        """,
        # Chunk version index
        """
        CREATE INDEX chunk_version IF NOT EXISTS
        FOR (c:Chunk) ON (c.version)
        """,
        # Entity name index
        """
        CREATE INDEX entity_name IF NOT EXISTS
        FOR (e:Entity) ON (e.name)
        """,
        # Entity type index
        """
        CREATE INDEX entity_type IF NOT EXISTS
        FOR (e:Entity) ON (e.type)
        """,
        # Entity first_seen index
        """
        CREATE INDEX entity_first_seen IF NOT EXISTS
        FOR (e:Entity) ON (e.first_seen)
        """,
        # TimeSnapshot timestamp index
        """
        CREATE INDEX timesnapshot_timestamp IF NOT EXISTS
        FOR (ts:TimeSnapshot) ON (ts.timestamp)
        """,
    ]

    for index in indexes:
        try:
            client.execute_query(index.strip())
            logger.info(f"Index created successfully")
        except Exception as e:
            logger.warning(f"Index creation failed (may already exist): {e}")


def create_fulltext_indexes(client: Neo4jClient) -> None:
    """
    Create full-text search indexes.

    Args:
        client: Neo4j client instance
    """
    fulltext_indexes = [
        # Document full-text index
        """
        CREATE FULLTEXT INDEX document_text IF NOT EXISTS
        FOR (d:Document)
        ON EACH [d.title]
        """,
        # Chunk full-text index
        """
        CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS
        FOR (c:Chunk)
        ON EACH [c.text]
        """,
        # Entity full-text index
        """
        CREATE FULLTEXT INDEX entity_text IF NOT EXISTS
        FOR (e:Entity)
        ON EACH [e.name]
        """,
    ]

    for index in fulltext_indexes:
        try:
            client.execute_query(index.strip())
            logger.info(f"Full-text index created successfully")
        except Exception as e:
            logger.warning(f"Full-text index creation failed (may already exist): {e}")


def create_vector_index(client: Neo4jClient) -> None:
    """
    Create vector index for embeddings.

    Args:
        client: Neo4j client instance
    """
    settings = get_settings()
    dimensions = settings.openai_embedding_dimensions

    # Check if vector index already exists
    check_query = """
    SHOW INDEXES
    YIELD name, type
    WHERE name = 'chunk_embeddings' AND type = 'VECTOR'
    RETURN count(*) as count
    """

    try:
        result = client.execute_query(check_query)
        if result and result[0].get("count", 0) > 0:
            logger.info("Vector index 'chunk_embeddings' already exists")
            return
    except Exception as e:
        logger.warning(f"Could not check for existing vector index: {e}")

    # Create vector index
    vector_index_query = f"""
    CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
    FOR (c:Chunk)
    ON c.embedding
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """

    try:
        client.execute_query(vector_index_query)
        logger.info(f"Vector index created successfully (dimensions: {dimensions})")
    except Exception as e:
        logger.error(f"Vector index creation failed: {e}")
        raise


def drop_all_constraints_and_indexes(client: Neo4jClient) -> None:
    """
    Drop all constraints and indexes (use with caution!).

    Args:
        client: Neo4j client instance
    """
    logger.warning("Dropping all constraints and indexes...")

    # Get all constraints
    constraints_query = "SHOW CONSTRAINTS YIELD name RETURN name"
    try:
        constraints = client.execute_query(constraints_query)
        for constraint in constraints:
            drop_query = f"DROP CONSTRAINT {constraint['name']} IF EXISTS"
            client.execute_query(drop_query)
            logger.info(f"Dropped constraint: {constraint['name']}")
    except Exception as e:
        logger.error(f"Error dropping constraints: {e}")

    # Get all indexes
    indexes_query = "SHOW INDEXES YIELD name RETURN name"
    try:
        indexes = client.execute_query(indexes_query)
        for index in indexes:
            drop_query = f"DROP INDEX {index['name']} IF EXISTS"
            client.execute_query(drop_query)
            logger.info(f"Dropped index: {index['name']}")
    except Exception as e:
        logger.error(f"Error dropping indexes: {e}")


def get_schema_info(client: Neo4jClient) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get information about current schema.

    Args:
        client: Neo4j client instance

    Returns:
        Dictionary with constraints and indexes information
    """
    # Get constraints
    constraints_query = """
    SHOW CONSTRAINTS
    YIELD name, type, entityType, labelsOrTypes, properties
    RETURN name, type, entityType, labelsOrTypes, properties
    """
    constraints = client.execute_query(constraints_query)

    # Get indexes
    indexes_query = """
    SHOW INDEXES
    YIELD name, type, entityType, labelsOrTypes, properties, state
    RETURN name, type, entityType, labelsOrTypes, properties, state
    """
    indexes = client.execute_query(indexes_query)

    return {
        "constraints": constraints,
        "indexes": indexes,
    }


def init_schema(client: Optional[Neo4jClient] = None, force: bool = False) -> None:
    """
    Initialize the complete Neo4j schema.

    Args:
        client: Optional Neo4j client instance (creates one if not provided)
        force: If True, drop existing schema before creating new one
    """
    if client is None:
        client = get_neo4j_client()

    logger.info("Initializing Neo4j schema...")

    # Verify connectivity
    if not client.verify_connectivity():
        raise ConnectionError("Cannot connect to Neo4j database")

    # Drop existing schema if force is True
    if force:
        logger.warning("Force mode enabled: dropping existing schema")
        drop_all_constraints_and_indexes(client)

    # Create schema components
    logger.info("Creating constraints...")
    create_constraints(client)

    logger.info("Creating indexes...")
    create_indexes(client)

    logger.info("Creating full-text indexes...")
    create_fulltext_indexes(client)

    logger.info("Creating vector index...")
    create_vector_index(client)

    # Get and log schema info
    schema_info = get_schema_info(client)
    logger.info(f"Schema initialized successfully:")
    logger.info(f"  - Constraints: {len(schema_info['constraints'])}")
    logger.info(f"  - Indexes: {len(schema_info['indexes'])}")

    # Wait for indexes to come online
    logger.info("Waiting for indexes to populate...")
    wait_for_indexes(client)

    logger.info("Schema initialization complete!")


def wait_for_indexes(client: Neo4jClient, timeout: int = 300) -> None:
    """
    Wait for indexes to come online.

    Args:
        client: Neo4j client instance
        timeout: Maximum time to wait in seconds
    """
    import time

    start_time = time.time()
    while time.time() - start_time < timeout:
        query = """
        SHOW INDEXES
        YIELD state
        WHERE state <> 'ONLINE'
        RETURN count(*) as pending
        """
        result = client.execute_query(query)
        pending = result[0].get("pending", 0) if result else 0

        if pending == 0:
            logger.info("All indexes are online")
            return

        logger.info(f"Waiting for {pending} indexes to come online...")
        time.sleep(5)

    logger.warning(f"Timeout waiting for indexes (waited {timeout}s)")


def verify_schema(client: Optional[Neo4jClient] = None) -> bool:
    """
    Verify that the schema is properly set up.

    Args:
        client: Optional Neo4j client instance

    Returns:
        True if schema is valid, False otherwise
    """
    if client is None:
        client = get_neo4j_client()

    try:
        schema_info = get_schema_info(client)

        # Check for required constraints
        required_constraints = [
            "document_id_unique",
            "chunk_id_unique",
            "entity_id_unique",
        ]
        constraint_names = [c["name"] for c in schema_info["constraints"]]

        for req_constraint in required_constraints:
            if req_constraint not in constraint_names:
                logger.error(f"Missing required constraint: {req_constraint}")
                return False

        # Check for vector index
        vector_indexes = [
            idx for idx in schema_info["indexes"]
            if idx["name"] == "chunk_embeddings" and idx["type"] == "VECTOR"
        ]
        if not vector_indexes:
            logger.error("Missing required vector index: chunk_embeddings")
            return False

        if vector_indexes[0].get("state") != "ONLINE":
            logger.warning(
                f"Vector index is not online: {vector_indexes[0].get('state')}"
            )

        logger.info("Schema verification successful")
        return True

    except Exception as e:
        logger.error(f"Schema verification failed: {e}")
        return False
