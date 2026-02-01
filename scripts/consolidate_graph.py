#!/usr/bin/env python3
"""Script to consolidate the temporal knowledge graph by merging duplicate entities."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_kg_rag.graph.consolidation import consolidate_knowledge_graph
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main consolidation entry point."""
    parser = argparse.ArgumentParser(
        description="Consolidate temporal knowledge graph by merging duplicate entities"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of entities to process at once (default: 50)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for merging entities (default: 0.8)",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Knowledge Graph Consolidation")
    logger.info("=" * 70)
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Similarity threshold: {args.similarity_threshold}")
    logger.info("")

    try:
        stats = consolidate_knowledge_graph(
            batch_size=args.batch_size,
            similarity_threshold=args.similarity_threshold,
        )

        logger.info("")
        logger.info("=" * 70)
        logger.info("Consolidation Complete!")
        logger.info("=" * 70)
        logger.info(f"Entities before: {stats['total_entities_before']}")
        logger.info(f"Entities after: {stats['total_entities_after']}")
        logger.info(f"Entities merged: {stats['entities_merged']}")
        logger.info(f"Merge operations: {stats['merge_operations']}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Consolidation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
