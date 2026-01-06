#!/usr/bin/env python3
"""Test script for RAG system with LangChain and LangGraph implementations."""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_kg_rag.rag.chain import get_rag_chain
from temporal_kg_rag.rag.graph import get_rag_graph
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_section_header(title: str):
    """Print a section header."""
    print_separator()
    print(f"  {title}")
    print_separator()


def format_sources(sources: List[Dict]) -> str:
    """Format sources for display."""
    if not sources:
        return "No sources"

    formatted = []
    for i, source in enumerate(sources[:5], 1):  # Show top 5 sources
        doc_title = source.get("document_title", "Unknown")
        created_at = source.get("created_at", "Unknown")
        score = source.get("score", 0)
        formatted.append(f"  {i}. \"{doc_title}\" ({created_at}) [score: {score:.3f}]")

    if len(sources) > 5:
        formatted.append(f"  ... and {len(sources) - 5} more sources")

    return "\n".join(formatted)


def print_rag_result(result: Dict, show_context: bool = False):
    """Print RAG result in a formatted way."""
    print("\n📝 Answer:")
    print("-" * 80)
    print(result.get("answer", "No answer"))
    print("-" * 80)

    print("\n📚 Sources:")
    print(format_sources(result.get("sources", [])))

    if "metadata" in result:
        metadata = result["metadata"]
        print("\n📊 Metadata:")
        print(f"  - Query type: {metadata.get('query_type', 'N/A')}")
        print(f"  - Temporal detected: {metadata.get('temporal_detected', 'N/A')}")
        print(f"  - Number of results: {metadata.get('num_results', 'N/A')}")
        print(f"  - Verified: {metadata.get('verified', 'N/A')}")
        if metadata.get("verification_notes"):
            print(f"  - Verification notes: {metadata['verification_notes']}")
        if metadata.get("entities_detected"):
            entities = metadata["entities_detected"]
            if entities:
                print(f"  - Entities detected: {', '.join(entities[:5])}")

    if show_context and "context" in result:
        print("\n📄 Context (first 500 chars):")
        print(result["context"][:500] + "...")


def test_langchain_basic(query: str, top_k: int = 5):
    """Test basic LangChain RAG chain."""
    print_section_header("Test: LangChain RAG Chain - Basic Query")
    print(f"Query: {query}")
    print(f"Top K: {top_k}\n")

    chain = get_rag_chain()

    start_time = time.time()
    result = chain.query(query, top_k=top_k)
    elapsed = time.time() - start_time

    print_rag_result(result)
    print(f"\n⏱️  Time elapsed: {elapsed:.2f}s")

    return result


def test_langchain_temporal(query: str, top_k: int = 5):
    """Test LangChain RAG with temporal query."""
    print_section_header("Test: LangChain RAG Chain - Temporal Query")
    print(f"Query: {query}")
    print(f"Top K: {top_k}\n")

    chain = get_rag_chain()

    start_time = time.time()
    result = chain.query(
        query,
        top_k=top_k,
        use_temporal_detection=True,
        expand_context=True,
    )
    elapsed = time.time() - start_time

    print_rag_result(result)

    if result.get("temporal_context"):
        print("\n🕐 Temporal Context:")
        tc = result["temporal_context"]
        if tc.has_temporal_reference:
            print(f"  - Has temporal reference: Yes")
            if tc.temporal_filter:
                tf = tc.temporal_filter
                print(f"  - Query type: {tf.query_type}")
                if tf.point_in_time:
                    print(f"  - Point in time: {tf.point_in_time}")
                if tf.start_time:
                    print(f"  - Start time: {tf.start_time}")
                if tf.end_time:
                    print(f"  - End time: {tf.end_time}")
        else:
            print(f"  - Has temporal reference: No")

    print(f"\n⏱️  Time elapsed: {elapsed:.2f}s")

    return result


def test_langchain_with_history(query: str, history: List[Dict], top_k: int = 5):
    """Test LangChain RAG with conversation history."""
    print_section_header("Test: LangChain RAG Chain - With Conversation History")
    print(f"Query: {query}")
    print(f"History turns: {len(history)}")
    print(f"Top K: {top_k}\n")

    chain = get_rag_chain()

    start_time = time.time()
    result = chain.query_with_history(query, history, top_k=top_k)
    elapsed = time.time() - start_time

    print_rag_result(result)
    print(f"\n⏱️  Time elapsed: {elapsed:.2f}s")

    return result


def test_langchain_streaming(query: str, top_k: int = 5):
    """Test LangChain RAG with streaming."""
    print_section_header("Test: LangChain RAG Chain - Streaming Response")
    print(f"Query: {query}")
    print(f"Top K: {top_k}\n")

    chain = get_rag_chain()

    print("📝 Streaming Answer:")
    print("-" * 80)

    start_time = time.time()
    for chunk in chain.stream_query(query, top_k=top_k):
        print(chunk, end="", flush=True)
    elapsed = time.time() - start_time

    print("\n" + "-" * 80)
    print(f"\n⏱️  Time elapsed: {elapsed:.2f}s")


def test_langgraph_basic(query: str, top_k: int = 10):
    """Test LangGraph RAG workflow."""
    print_section_header("Test: LangGraph RAG Workflow - Basic Query")
    print(f"Query: {query}")
    print(f"Top K: {top_k}\n")

    graph = get_rag_graph()

    start_time = time.time()
    result = graph.query(query, top_k=top_k)
    elapsed = time.time() - start_time

    print_rag_result(result)
    print(f"\n⏱️  Time elapsed: {elapsed:.2f}s")

    return result


def test_langgraph_temporal(query: str, top_k: int = 10):
    """Test LangGraph RAG with temporal query."""
    print_section_header("Test: LangGraph RAG Workflow - Temporal Query")
    print(f"Query: {query}")
    print(f"Top K: {top_k}\n")

    graph = get_rag_graph()

    start_time = time.time()
    result = graph.query(query, top_k=top_k)
    elapsed = time.time() - start_time

    print_rag_result(result)
    print(f"\n⏱️  Time elapsed: {elapsed:.2f}s")

    return result


def compare_implementations(query: str, top_k: int = 5):
    """Compare LangChain and LangGraph implementations."""
    print_section_header("Comparison: LangChain vs LangGraph")
    print(f"Query: {query}")
    print(f"Top K: {top_k}\n")

    # Test LangChain
    print("1️⃣  Testing LangChain RAG Chain...")
    chain = get_rag_chain()
    start_time = time.time()
    chain_result = chain.query(query, top_k=top_k)
    chain_time = time.time() - start_time

    print(f"   ✓ Completed in {chain_time:.2f}s")
    print(f"   - Answer length: {len(chain_result.get('answer', ''))} chars")
    print(f"   - Sources: {len(chain_result.get('sources', []))}")

    # Test LangGraph
    print("\n2️⃣  Testing LangGraph RAG Workflow...")
    graph = get_rag_graph()
    start_time = time.time()
    graph_result = graph.query(query, top_k=top_k)
    graph_time = time.time() - start_time

    print(f"   ✓ Completed in {graph_time:.2f}s")
    print(f"   - Answer length: {len(graph_result.get('answer', ''))} chars")
    print(f"   - Sources: {len(graph_result.get('sources', []))}")
    if "metadata" in graph_result:
        md = graph_result["metadata"]
        print(f"   - Query type detected: {md.get('query_type', 'N/A')}")
        print(f"   - Temporal detected: {md.get('temporal_detected', 'N/A')}")
        print(f"   - Verified: {md.get('verified', 'N/A')}")

    # Comparison
    print("\n📊 Comparison:")
    print(f"   - Speed difference: {abs(chain_time - graph_time):.2f}s")
    if chain_time < graph_time:
        print(f"   - LangChain was {((graph_time - chain_time) / chain_time * 100):.1f}% faster")
    else:
        print(f"   - LangGraph was {((chain_time - graph_time) / graph_time * 100):.1f}% faster")

    print("\n" + "=" * 80)
    print("LangChain Answer:")
    print("-" * 80)
    print(chain_result.get("answer", "No answer"))

    print("\n" + "=" * 80)
    print("LangGraph Answer:")
    print("-" * 80)
    print(graph_result.get("answer", "No answer"))
    print("=" * 80)


def run_demo():
    """Run demonstration with example queries."""
    print_section_header("RAG System Demo - Multiple Query Types")

    queries = [
        {
            "query": "What is artificial intelligence?",
            "type": "Factual",
            "method": "langchain",
        },
        {
            "query": "What were the main AI developments in 2023?",
            "type": "Temporal",
            "method": "langgraph",
        },
        {
            "query": "Compare machine learning and deep learning",
            "type": "Comparison",
            "method": "langgraph",
        },
        {
            "query": "How has climate change policy evolved over time?",
            "type": "Evolution",
            "method": "langgraph",
        },
    ]

    for i, query_info in enumerate(queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Demo Query {i}/{len(queries)} - {query_info['type']} Query")
        print(f"{'=' * 80}")
        print(f"Query: {query_info['query']}")
        print(f"Method: {query_info['method']}\n")

        try:
            if query_info["method"] == "langchain":
                result = test_langchain_basic(query_info["query"], top_k=5)
            else:
                result = test_langgraph_basic(query_info["query"], top_k=10)

            print("\n✓ Query completed successfully")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            logger.error(f"Demo query failed: {e}", exc_info=True)

        if i < len(queries):
            print("\n" + "." * 80)
            input("Press Enter to continue to next query...")


def run_conversation_demo():
    """Run a multi-turn conversation demo."""
    print_section_header("Conversation Demo - Multi-turn Dialogue")

    conversation_history = []

    queries = [
        "What is quantum computing?",
        "When was it first developed?",
        "What are the main challenges?",
    ]

    chain = get_rag_chain()

    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Turn {i}/{len(queries)}")
        print(f"{'=' * 80}")
        print(f"User: {query}\n")

        try:
            if i == 1:
                # First query without history
                result = chain.query(query, top_k=5)
            else:
                # Subsequent queries with history
                result = chain.query_with_history(query, conversation_history, top_k=5)

            answer = result.get("answer", "No answer")
            print(f"Assistant: {answer}\n")

            # Add to history
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": answer})

            print("✓ Turn completed")
        except Exception as e:
            print(f"✗ Error: {e}")
            logger.error(f"Conversation turn failed: {e}", exc_info=True)

        if i < len(queries):
            print("\n" + "." * 80)
            input("Press Enter to continue to next turn...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test RAG system with LangChain and LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with example queries
  python scripts/test_rag.py --demo

  # Test LangChain with custom query
  python scripts/test_rag.py --query "What is AI?" --method langchain

  # Test LangGraph with temporal query
  python scripts/test_rag.py --query "AI in 2023" --method langgraph --temporal

  # Compare implementations
  python scripts/test_rag.py --query "machine learning" --compare

  # Test streaming
  python scripts/test_rag.py --query "climate change" --stream

  # Conversation demo
  python scripts/test_rag.py --conversation-demo
        """,
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Query to test",
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["langchain", "langgraph"],
        default="langgraph",
        help="RAG implementation to use (default: langgraph)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieval results (default: 5)",
    )

    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Enable temporal detection and filtering",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare LangChain and LangGraph implementations",
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Test streaming response (LangChain only)",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with example queries",
    )

    parser.add_argument(
        "--conversation-demo",
        action="store_true",
        help="Run multi-turn conversation demo",
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show retrieved context in output",
    )

    args = parser.parse_args()

    try:
        if args.demo:
            run_demo()
        elif args.conversation_demo:
            run_conversation_demo()
        elif args.compare and args.query:
            compare_implementations(args.query, args.top_k)
        elif args.stream and args.query:
            test_langchain_streaming(args.query, args.top_k)
        elif args.query:
            if args.method == "langchain":
                if args.temporal:
                    test_langchain_temporal(args.query, args.top_k)
                else:
                    test_langchain_basic(args.query, args.top_k)
            else:  # langgraph
                if args.temporal:
                    test_langgraph_temporal(args.query, args.top_k)
                else:
                    test_langgraph_basic(args.query, args.top_k)
        else:
            parser.print_help()
            print("\n⚠️  Please provide a query or use --demo/--conversation-demo")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
