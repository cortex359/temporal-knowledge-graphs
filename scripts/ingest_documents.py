#!/usr/bin/env python3
"""
CLI script for ingesting documents into the temporal knowledge graph.

Usage:
    python scripts/ingest_documents.py --path document.pdf
    python scripts/ingest_documents.py --path docs/ --pattern "*.md"
    python scripts/ingest_documents.py --path file.txt --title "My Document" --metadata '{"author": "John Doe"}'

    # ECT-QA dataset ingestion
    python scripts/ingest_documents.py --ectqa data/ectqa.jsonl
    python scripts/ingest_documents.py --ectqa data/ectqa.jsonl --limit 100 --sector consumer_discretionary
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
from temporal_kg_rag.ingestion.ectqa_loader import ECTQALoader, get_ectqa_loader
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

  # Skip relation extraction (keep entities but skip relations)
  %(prog)s --path document.pdf --no-relations

  # Skip embeddings (for testing)
  %(prog)s --path document.pdf --no-embeddings

  # ECT-QA dataset ingestion
  %(prog)s --ectqa data/ectqa.jsonl
  %(prog)s --ectqa data/ectqa.jsonl --limit 100
  %(prog)s --ectqa data/ectqa.jsonl --sector consumer_discretionary --year 2020
  %(prog)s --ectqa data/ectqa.jsonl --stock-code AAPL --quarter q1
        """,
    )

    parser.add_argument(
        "--path",
        type=str,
        help="Path to file or directory (required unless using --ectqa)",
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
        help="Skip entity extraction (also skips relation extraction)",
    )
    parser.add_argument(
        "--no-relations",
        action="store_true",
        help="Skip semantic relation extraction between entities",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding generation",
    )

    # ECT-QA specific arguments
    parser.add_argument(
        "--ectqa",
        type=str,
        help="Path to ECT-QA JSONL file (alternative to --path)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of documents to ingest (for ECT-QA)",
    )
    parser.add_argument(
        "--sector",
        type=str,
        help="Filter ECT-QA by sector (e.g., consumer_discretionary, technology)",
    )
    parser.add_argument(
        "--year",
        type=str,
        help="Filter ECT-QA by year (e.g., 2020)",
    )
    parser.add_argument(
        "--quarter",
        type=str,
        help="Filter ECT-QA by quarter (e.g., q1, q2, q3, q4)",
    )
    parser.add_argument(
        "--stock-code",
        type=str,
        help="Filter ECT-QA by company stock code (e.g., AAPL)",
    )
    parser.add_argument(
        "--ectqa-stats",
        action="store_true",
        help="Show ECT-QA dataset statistics and exit",
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
        # Handle ECT-QA mode
        if args.ectqa:
            return ingest_ectqa(args)

        # Handle ECT-QA stats only
        if args.ectqa_stats and not args.ectqa:
            logger.error("--ectqa-stats requires --ectqa <file>")
            return 1

        # Regular file ingestion requires --path
        if not args.path:
            logger.error("Either --path or --ectqa is required")
            return 1

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
                extract_relations=not args.no_relations,
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
                extract_relations=not args.no_relations,
                generate_embeddings=not args.no_embeddings,
            )

            logger.info(
                f"\n✓ Successfully ingested {len(documents)}/{len(files)} documents"
            )

        # Show statistics
        if args.show_stats:
            show_database_stats(pipeline)

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


def ingest_ectqa(args) -> int:
    """
    Handle ECT-QA dataset ingestion.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    ectqa_path = Path(args.ectqa).resolve()

    if not ectqa_path.exists():
        logger.error(f"ECT-QA file not found: {ectqa_path}")
        return 1

    # Initialize ECT-QA loader
    ectqa_loader = get_ectqa_loader()

    # Show stats only if requested
    if args.ectqa_stats:
        logger.info("ECT-QA Dataset Statistics")
        logger.info("=" * 70)

        stats = ectqa_loader.get_dataset_stats(str(ectqa_path))

        logger.info(f"Total records: {stats['total_records']:,}")
        logger.info(f"Unique companies: {stats['unique_companies']:,}")
        logger.info(f"Total tokens: {stats['total_tokens']:,}")

        logger.info("\nBy Sector:")
        for sector, count in sorted(stats['sectors'].items(), key=lambda x: -x[1]):
            logger.info(f"  {sector}: {count:,}")

        logger.info("\nBy Year:")
        for year, count in sorted(stats['years'].items()):
            logger.info(f"  {year}: {count:,}")

        logger.info("\nBy Quarter:")
        for quarter, count in sorted(stats['quarters'].items()):
            logger.info(f"  {quarter}: {count:,}")

        return 0

    # Initialize pipeline
    logger.info("Initializing ingestion pipeline...")
    pipeline = get_ingestion_pipeline()

    # Build filter description
    filters = []
    if args.sector:
        filters.append(f"sector={args.sector}")
    if args.year:
        filters.append(f"year={args.year}")
    if args.quarter:
        filters.append(f"quarter={args.quarter}")
    if args.stock_code:
        filters.append(f"stock_code={args.stock_code}")
    if args.limit:
        filters.append(f"limit={args.limit}")

    filter_desc = ", ".join(filters) if filters else "none"
    logger.info(f"Loading ECT-QA documents from: {ectqa_path}")
    logger.info(f"Filters: {filter_desc}")
    logger.info("")

    # Load and ingest documents
    success_count = 0
    fail_count = 0
    total_count = 0

    for document, text in ectqa_loader.load_jsonl(
        str(ectqa_path),
        use_cleaned_content=True,
        limit=args.limit,
        filter_sector=args.sector,
        filter_year=args.year,
        filter_quarter=args.quarter,
        filter_stock_code=args.stock_code,
    ):
        total_count += 1

        try:
            logger.info(f"\n[{total_count}] Processing: {document.title}")

            # Ingest via pipeline (but we already have the document and text)
            # We need to manually run the pipeline steps
            pipeline.graph_ops.create_document(document)

            # Chunk text
            chunks = pipeline.chunker.chunk_text(text, document.id, strategy="semantic")
            logger.info(f"  Created {len(chunks)} chunks")

            # Generate embeddings
            if not args.no_embeddings:
                pipeline._generate_embeddings(chunks)

            # Store chunks
            pipeline.graph_ops.create_chunks_batch(chunks, document.id)

            # Extract entities
            entities = {}
            mentions = []
            if not args.no_entities:
                entities, mentions = pipeline._extract_entities(chunks)
                pipeline.graph_ops.create_entities_and_mentions_batch(entities, mentions)

            # Extract relations
            if not args.no_entities and not args.no_relations and len(entities) >= 2:
                relationships = pipeline._extract_relations(chunks, entities)
                if relationships:
                    pipeline.graph_ops.create_entity_relationships_batch(relationships)
                    logger.info(f"  Extracted {len(relationships)} relations")

            success_count += 1
            logger.info(f"  ✓ Successfully ingested")

        except Exception as e:
            fail_count += 1
            logger.error(f"  ✗ Failed to ingest: {e}")
            continue

    logger.info("\n" + "=" * 70)
    logger.info("ECT-QA Ingestion Summary")
    logger.info("=" * 70)
    logger.info(f"Total processed: {total_count}")
    logger.info(f"Successfully ingested: {success_count}")
    logger.info(f"Failed: {fail_count}")

    # Show statistics
    if args.show_stats:
        show_database_stats(pipeline)

    logger.info("\n" + "=" * 70)
    logger.info("ECT-QA Ingestion Complete!")
    logger.info("=" * 70)

    return 0 if fail_count == 0 else 1


def show_database_stats(pipeline: IngestionPipeline) -> None:
    """Show database statistics."""
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


if __name__ == "__main__":
    sys.exit(main())
