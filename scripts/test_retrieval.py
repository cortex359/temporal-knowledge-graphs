#!/usr/bin/env python3
"""
Test script for the retrieval system.

This script demonstrates all retrieval capabilities:
- Vector search
- Graph search
- Hybrid search with RRF
- Temporal queries
- Context expansion
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_kg_rag.retrieval.vector_search import get_vector_search
from temporal_kg_rag.retrieval.graph_search import get_graph_search
from temporal_kg_rag.retrieval.hybrid_search import get_hybrid_search
from temporal_kg_rag.retrieval.temporal_retrieval import get_temporal_retrieval
from temporal_kg_rag.retrieval.context_expansion import get_context_expander
from temporal_kg_rag.models.temporal import TemporalFilter
from temporal_kg_rag.utils.logger import setup_logging, get_logger
from temporal_kg_rag.config.settings import get_settings

logger = get_logger(__name__)


def print_results(results: list, title: str, max_results: int = 3):
    """Print search results nicely."""
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)

    if not results:
        print("No results found")
        return

    for i, result in enumerate(results[:max_results], 1):
        print(f"\n[Result {i}]")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Document: {result.get('document_title', 'Unknown')}")

        score = result.get('rrf_score') or result.get('hybrid_score') or result.get('score', 0)
        print(f"Score: {score:.4f}")

        if 'created_at' in result:
            print(f"Created: {result['created_at']}")

        text = result.get('text', '')
        print(f"\nText preview: {text[:200]}...")

        if 'entities' in result and result['entities']:
            entity_list = result['entities']
            if isinstance(entity_list[0], dict):
                entity_names = [e.get('name', str(e)) for e in entity_list[:5] if e.get('name')]
            else:
                entity_names = [e for e in entity_list[:5] if e]
            if entity_names:
                print(f"Entities: {', '.join(entity_names)}")

    print(f"\nTotal results: {len(results)}")


def test_vector_search(query: str, top_k: int = 5):
    """Test vector similarity search."""
    logger.info(f"Testing vector search: {query}")

    vs = get_vector_search()
    results = vs.search(query, top_k=top_k)

    print_results(results, f"Vector Search Results for: '{query}'", max_results=top_k)


def test_graph_search(query: str, top_k: int = 5):
    """Test graph-based search."""
    logger.info(f"Testing graph search: {query}")

    gs = get_graph_search()
    results = gs.search(query, top_k=top_k)

    print_results(results, f"Graph Search Results for: '{query}'", max_results=top_k)


def test_hybrid_search(query: str, top_k: int = 5):
    """Test hybrid search with RRF."""
    logger.info(f"Testing hybrid search: {query}")

    hs = get_hybrid_search()
    results = hs.search(query, top_k=top_k)

    print_results(results, f"Hybrid Search Results for: '{query}'", max_results=top_k)


def test_hybrid_comparison(query: str):
    """Compare all three search methods."""
    logger.info(f"Testing hybrid comparison: {query}")

    hs = get_hybrid_search()
    explanation = hs.explain_results(query, top_k=5)

    print("\n" + "=" * 70)
    print(f"Search Comparison for: '{query}'")
    print("=" * 70)

    print("\nOverlap Analysis:")
    print(f"  Vector-only results: {explanation['results']['vector_only']}")
    print(f"  Graph-only results: {explanation['results']['graph_only']}")
    print(f"  Results in both: {explanation['results']['both']}")
    print(f"  Total hybrid results: {explanation['results']['hybrid_total']}")

    print_results(explanation['hybrid_results'], "Top Hybrid Results", max_results=3)


def test_temporal_search(query: str):
    """Test temporal search with auto-detection."""
    logger.info(f"Testing temporal search: {query}")

    tr = get_temporal_retrieval()
    result_dict = tr.search_with_temporal_context(query, top_k=5)

    print("\n" + "=" * 70)
    print(f"Temporal Search for: '{query}'")
    print("=" * 70)

    if result_dict['temporal_context']:
        tc = result_dict['temporal_context']
        print(f"\nDetected temporal context:")
        print(f"  Has temporal reference: {tc.has_temporal_reference}")
        print(f"  Keywords found: {tc.temporal_keywords}")
        if tc.temporal_filter:
            print(f"  Query type: {tc.temporal_filter.query_type}")

    print_results(result_dict['results'], "Temporal Results", max_results=3)


def test_point_in_time_search(query: str, year: int):
    """Test point-in-time search."""
    logger.info(f"Testing point-in-time search: {query} at year {year}")

    tr = get_temporal_retrieval()
    timestamp = datetime(year, 12, 31)
    results = tr.search_at_time(query, timestamp, top_k=5)

    print_results(
        results,
        f"Point-in-Time Search (as of {year}): '{query}'",
        max_results=3
    )


def test_context_expansion(query: str):
    """Test context expansion."""
    logger.info(f"Testing context expansion: {query}")

    # First get some results
    hs = get_hybrid_search()
    results = hs.search(query, top_k=3)

    if not results:
        print("No results to expand")
        return

    # Expand them
    ce = get_context_expander()
    expanded = ce.expand_results(results, neighboring_chunk_window=1)

    print("\n" + "=" * 70)
    print(f"Context Expansion for: '{query}'")
    print("=" * 70)

    for i, result in enumerate(expanded[:2], 1):
        print(f"\n[Expanded Result {i}]")
        print(f"Document: {result.get('document_title', 'Unknown')}")
        print(f"\nMain text: {result['text'][:150]}...")

        if 'expanded_context' in result:
            ctx = result['expanded_context']

            if 'entities' in ctx:
                entities = ctx['entities'][:3]
                print(f"\nKey entities:")
                for e in entities:
                    print(f"  - {e['entity_name']} ({e['entity_type']})")

            if 'neighboring_chunks' in ctx:
                neighbors = ctx['neighboring_chunks']
                if neighbors.get('before'):
                    print(f"\nPrevious chunk:")
                    print(f"  {neighbors['before'][-1]['text'][:100]}...")
                if neighbors.get('after'):
                    print(f"\nNext chunk:")
                    print(f"  {neighbors['after'][0]['text'][:100]}...")

            if 'related_chunks' in ctx:
                related = ctx['related_chunks'][:2]
                if related:
                    print(f"\nRelated chunks:")
                    for r in related:
                        print(f"  - From '{r['document_title']}' (shared entities: {r['shared_entities']})")


def test_entity_search(entity_name: str):
    """Test entity-based search."""
    logger.info(f"Testing entity search: {entity_name}")

    gs = get_graph_search()
    results = gs.search_by_entities([entity_name], top_k=5)

    print_results(results, f"Entity Search for: '{entity_name}'", max_results=3)


def run_demo():
    """Run a comprehensive demonstration of all features."""
    print("\n" + "=" * 70)
    print("TEMPORAL KNOWLEDGE GRAPH - RETRIEVAL SYSTEM DEMO")
    print("=" * 70)

    # Define test queries
    test_queries = [
        ("artificial intelligence", "Basic hybrid search"),
        ("OpenAI GPT-4", "Specific entity search"),
        ("What is quantum computing?", "Question-based search"),
        ("AI in 2023", "Temporal search with year"),
        ("climate change research", "Multi-word topic"),
    ]

    for query, description in test_queries:
        print(f"\n\n{'#' * 70}")
        print(f"# {description}")
        print(f"# Query: '{query}'")
        print(f"{'#' * 70}")

        # Run hybrid search
        test_hybrid_search(query, top_k=3)

        # Show temporal context if relevant
        if any(word in query.lower() for word in ["in", "at", "during", "2023", "2024"]):
            test_temporal_search(query)

        input("\nPress Enter to continue to next demo...")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test retrieval system capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full demonstration
  %(prog)s --demo

  # Test different search methods
  %(prog)s --query "artificial intelligence" --method hybrid
  %(prog)s --query "OpenAI" --method graph
  %(prog)s --query "machine learning" --method vector

  # Test temporal search
  %(prog)s --query "AI developments" --temporal
  %(prog)s --query "quantum computing" --year 2023

  # Test context expansion
  %(prog)s --query "climate change" --expand-context

  # Compare search methods
  %(prog)s --query "neural networks" --compare
        """,
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run full demonstration",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Search query",
    )
    parser.add_argument(
        "--method",
        choices=["vector", "graph", "hybrid"],
        default="hybrid",
        help="Search method",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of results",
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Use temporal search with auto-detection",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Point-in-time search for specific year",
    )
    parser.add_argument(
        "--expand-context",
        action="store_true",
        help="Expand results with context",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all search methods",
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="Search by entity name",
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

    try:
        if args.demo:
            run_demo()
            return 0

        if not args.query and not args.entity:
            parser.print_help()
            print("\nError: Please provide --query or --entity, or use --demo")
            return 1

        # Entity search
        if args.entity:
            test_entity_search(args.entity)
            return 0

        # Regular search with various options
        query = args.query

        if args.compare:
            test_hybrid_comparison(query)
        elif args.temporal:
            test_temporal_search(query)
        elif args.year:
            test_point_in_time_search(query, args.year)
        elif args.expand_context:
            test_context_expansion(query)
        else:
            # Standard search
            if args.method == "vector":
                test_vector_search(query, args.top_k)
            elif args.method == "graph":
                test_graph_search(query, args.top_k)
            else:
                test_hybrid_search(query, args.top_k)

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
