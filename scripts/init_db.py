#!/usr/bin/env python3
"""
Initialize Neo4j database schema.

This script creates all necessary constraints, indexes, and the vector index
required for the temporal knowledge graph RAG system.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_kg_rag.graph.neo4j_client import Neo4jClient, close_neo4j_client
from temporal_kg_rag.graph.schema import init_schema, verify_schema, get_schema_info
from temporal_kg_rag.utils.logger import setup_logging, get_logger
from temporal_kg_rag.config.settings import get_settings

logger = get_logger(__name__)


def main():
    """Main entry point for database initialization."""
    parser = argparse.ArgumentParser(
        description="Initialize Neo4j database schema for temporal KG RAG system"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop existing schema before creating new one (CAUTION: destructive!)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the schema without making changes",
    )
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="Show current schema information",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)

    # Get settings
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("Neo4j Database Initialization")
    logger.info("=" * 60)
    logger.info(f"Neo4j URI: {settings.neo4j_uri}")
    logger.info(f"Neo4j Database: {settings.neo4j_database}")
    logger.info(f"Embedding Dimensions: {settings.openai_embedding_dimensions}")

    # Create Neo4j client
    client = None
    try:
        logger.info("\nConnecting to Neo4j...")
        client = Neo4jClient()

        # Verify connectivity
        if not client.verify_connectivity():
            logger.error("Failed to connect to Neo4j database")
            logger.error("Please check your connection settings and ensure Neo4j is running")
            return 1

        logger.info("✓ Successfully connected to Neo4j")

        # Show schema if requested
        if args.show_schema:
            logger.info("\nCurrent Schema:")
            logger.info("-" * 60)
            schema_info = get_schema_info(client)

            logger.info(f"\nConstraints ({len(schema_info['constraints'])}):")
            for constraint in schema_info['constraints']:
                logger.info(f"  - {constraint['name']} ({constraint['type']})")

            logger.info(f"\nIndexes ({len(schema_info['indexes'])}):")
            for index in schema_info['indexes']:
                state = index.get('state', 'UNKNOWN')
                logger.info(f"  - {index['name']} ({index['type']}) - {state}")

            return 0

        # Verify only if requested
        if args.verify_only:
            logger.info("\nVerifying schema...")
            if verify_schema(client):
                logger.info("✓ Schema verification successful")
                return 0
            else:
                logger.error("✗ Schema verification failed")
                return 1

        # Initialize schema
        if args.force:
            logger.warning("\n" + "!" * 60)
            logger.warning("WARNING: Force mode enabled!")
            logger.warning("This will drop ALL existing constraints and indexes.")
            logger.warning("!" * 60)

            response = input("\nAre you sure you want to continue? (yes/no): ")
            if response.lower() != "yes":
                logger.info("Aborted by user")
                return 0

        logger.info("\nInitializing schema...")
        init_schema(client, force=args.force)

        logger.info("\nVerifying schema...")
        if verify_schema(client):
            logger.info("✓ Schema verification successful")
        else:
            logger.warning("⚠ Schema verification completed with warnings")

        # Show database stats
        logger.info("\nDatabase Statistics:")
        logger.info("-" * 60)
        stats = client.get_database_stats()
        for key, value in stats.items():
            logger.info(f"  {key.capitalize()}: {value:,}")

        logger.info("\n" + "=" * 60)
        logger.info("Database initialization complete!")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 1

    except Exception as e:
        logger.error(f"\nError during database initialization: {e}", exc_info=True)
        return 1

    finally:
        if client:
            client.close()
        close_neo4j_client()


if __name__ == "__main__":
    sys.exit(main())
