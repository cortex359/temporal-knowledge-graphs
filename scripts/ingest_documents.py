#!/usr/bin/env python3
"""
CLI script for ingesting documents into the temporal knowledge graph.

Usage:
    python scripts/ingest_documents.py --path document.pdf
    python scripts/ingest_documents.py --path docs/ --pattern "*.md"
    python scripts/ingest_documents.py --path file.txt --title "My Document" --metadata '{"author": "John Doe"}'
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_kg_rag.ingestion.pipeline import IngestionPipeline, get_ingestion_pipeline
from temporal_kg_rag.ingestion.document_loader import DocumentLoader
from temporal_kg_rag.utils.logger import setup_logging, get_logger
from temporal_kg_rag.config.settings import get_settings

logger = get_logger(__name__)


def parse_metadata(metadata_str: Optional[str]) -> Dict:
    """Parse metadata JSON string."""
    if not metadata_str:
        return {}

    try:
        metadata = json.loads(metadata_str)
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a JSON object")
        return metadata
    except json.JSONDecodeError as e:
        logger.error(f"Invalid metadata JSON: {e}")
        raise


def collect_files(
    path: Path,
    pattern: Optional[str] = None,
    recursive: bool = False,
) -> List[Path]:
    """
    Collect files to ingest.

    Args:
        path: File or directory path
        pattern: Optional file pattern (e.g., "*.pdf")
        recursive: Whether to search recursively

    Returns:
        List of file paths
    """
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")

    # Collect files from directory
    loader = DocumentLoader()
    files = []

    if recursive:
        glob_pattern = f"**/{pattern}" if pattern else "**/*"
    else:
        glob_pattern = pattern or "*"

    for file_path in path.glob(glob_pattern):
        if file_path.is_file() and loader.is_supported(str(file_path)):
            files.append(file_path)

    return files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into temporal knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single PDF
  %(prog)s --path document.pdf

  # Ingest all markdown files in a directory
  %(prog)s --path docs/ --pattern "*.md"

  # Ingest with metadata
  %(prog)s --path report.pdf --title "Q4 Report" --metadata '{"author": "Jane", "year": 2024}'

  # Ingest directory recursively
  %(prog)s --path docs/ --recursive --pattern "*.txt"

  # Skip entity extraction (faster)
  %(prog)s --path document.pdf --no-entities

  # Skip embeddings (for testing)
  %(prog)s --path document.pdf --no-embeddings
        """,
    )

    parser.add_argument(
        "--path",
        required=True,
        type=str,
        help="Path to file or directory",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="File pattern (e.g., '*.pdf', '*.md')",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search directories recursively",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Document title (for single file only)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        help="Metadata as JSON string",
    )
    parser.add_argument(
        "--no-entities",
        action="store_true",
        help="Skip entity extraction",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding generation",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show database statistics after ingestion",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)

    logger.info("=" * 70)
    logger.info("Document Ingestion Tool")
    logger.info("=" * 70)

    try:
        # Parse metadata
        metadata = parse_metadata(args.metadata)

        # Collect files
        path = Path(args.path).resolve()
        logger.info(f"Scanning path: {path}")

        files = collect_files(path, args.pattern, args.recursive)

        if not files:
            logger.warning("No files found to ingest")
            return 0

        logger.info(f"Found {len(files)} file(s) to ingest")

        # Show supported formats
        loader = DocumentLoader()
        logger.info(f"Supported formats: {', '.join(loader.get_supported_formats())}")
        logger.info("")

        # Initialize pipeline
        logger.info("Initializing ingestion pipeline...")
        pipeline = get_ingestion_pipeline()

        # Ingest files
        if len(files) == 1:
            # Single file
            document = pipeline.ingest_document(
                str(files[0]),
                title=args.title,
                metadata=metadata,
                extract_entities=not args.no_entities,
                generate_embeddings=not args.no_embeddings,
            )
            logger.info(f"\n✓ Successfully ingested: {document.title}")

        else:
            # Multiple files
            if args.title:
                logger.warning("--title ignored for multiple files")

            documents = pipeline.ingest_documents_batch(
                [str(f) for f in files],
                extract_entities=not args.no_entities,
                generate_embeddings=not args.no_embeddings,
            )

            logger.info(
                f"\n✓ Successfully ingested {len(documents)}/{len(files)} documents"
            )

        # Show statistics
        if args.show_stats:
            logger.info("\n" + "=" * 70)
            logger.info("Database Statistics")
            logger.info("=" * 70)

            stats = pipeline.get_statistics()

            logger.info(f"Documents: {stats['documents']:,}")
            logger.info(f"Chunks: {stats['chunks']:,}")
            logger.info(f"Entities: {stats['entities']:,}")
            logger.info(f"Relationships: {stats['relationships']:,}")

            if "cache" in stats:
                cache = stats["cache"]
                logger.info(f"\nEmbedding Cache:")
                logger.info(f"  Cached embeddings: {cache['total_files']:,}")
                logger.info(f"  Cache size: {cache['total_size_mb']:.2f} MB")

        logger.info("\n" + "=" * 70)
        logger.info("Ingestion Complete!")
        logger.info("=" * 70)

        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 1

    except Exception as e:
        logger.error(f"\nIngestion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
