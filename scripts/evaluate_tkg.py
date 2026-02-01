#!/usr/bin/env python3
"""Evaluate TKG answers from JSONL file using the RAG system."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_kg_rag.rag.graph import get_rag_graph
from temporal_kg_rag.retrieval.hybrid_search import get_hybrid_search
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file into a list of dictionaries."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
    return records


def save_jsonl(records: List[Dict], file_path: Path) -> None:
    """Save a list of dictionaries to JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_context_chunks(results: List[Dict]) -> str:
    """Format retrieval results into context string matching the expected format."""
    if not results:
        return ""

    context_parts = []
    for i, result in enumerate(results, 1):
        chunk_content = f"""---NEW CHUNK (chunk{i})---
Document Title: {result.get('document_title', 'Unknown')}
Chunk Order Index: {result.get('chunk_index', 0)}
Chunk Content:
{result.get('text', '')}
---END OF CHUNK---"""
        context_parts.append(chunk_content)

    return "\n".join(context_parts)


def count_evidence_types(results: List[Dict]) -> Dict[str, int]:
    """
    Count evidence types from retrieval results.

    Returns dict with keys: entity, relation, community, text_units, total_evidence
    """
    # In our system, we track different types of evidence
    entity_count = 0
    relation_count = 0
    community_count = 0
    text_units = len(results)

    for result in results:
        # Count entities mentioned in each chunk
        entities = result.get("entities", [])
        if entities:
            entity_count += len(entities)

        # Check for relationship data
        relationships = result.get("relationships", [])
        if relationships:
            relation_count += len(relationships)

    return {
        "entity": entity_count,
        "relation": relation_count,
        "community": community_count,  # Not implemented in our system
        "text_units": text_units,
        "total_evidence": text_units,
    }


def evaluate_question(
    question: str,
    rag_graph,
    hybrid_search,
    top_k: int = 10,
) -> tuple[str, Dict[str, int], str]:
    """
    Evaluate a single question using the RAG system.

    Args:
        question: The question to answer
        rag_graph: RAG graph instance
        hybrid_search: Hybrid search instance
        top_k: Number of retrieval results

    Returns:
        Tuple of (predicted_answer, evidence_counts, context)
    """
    try:
        # Run RAG query with debug info
        result = rag_graph.query(question, top_k=top_k, return_debug_info=True)

        predicted_answer = result.get("answer", "")

        # Get retrieval results for evidence counting
        retrieval_results = []
        if "debug" in result:
            retrieval_results = result["debug"].get("all_retrieval_results", [])

        # Count evidence types
        evidence_counts = count_evidence_types(retrieval_results)

        # Format context
        context = format_context_chunks(retrieval_results)

        return predicted_answer, evidence_counts, context

    except Exception as e:
        logger.error(f"Error evaluating question: {e}", exc_info=True)
        return f"Error: {str(e)}", {"entity": 0, "relation": 0, "community": 0, "text_units": 0, "total_evidence": 0}, ""


def evaluate_dataset(
    input_file: Path,
    output_file: Path,
    top_k: int = 10,
    limit: Optional[int] = None,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate all questions in the dataset.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        top_k: Number of retrieval results per question
        limit: Optional limit on number of questions to evaluate
        skip_existing: If True, skip questions that already have predictions

    Returns:
        Statistics dictionary
    """
    logger.info(f"Loading dataset from {input_file}")
    records = load_jsonl(input_file)
    logger.info(f"Loaded {len(records)} records")

    if limit:
        records = records[:limit]
        logger.info(f"Limited to {len(records)} records")

    # Initialize RAG components
    logger.info("Initializing RAG components...")
    rag_graph = get_rag_graph()
    hybrid_search = get_hybrid_search()

    # Statistics
    stats = {
        "total": len(records),
        "evaluated": 0,
        "skipped": 0,
        "errors": 0,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "avg_time_per_question": 0,
    }

    total_time = 0
    evaluated_records = []

    for i, record in enumerate(records):
        question = record.get("question", "")

        # Check if already has prediction and skip_existing is True
        if skip_existing and record.get("predicted_answer"):
            logger.info(f"[{i+1}/{len(records)}] Skipping (already has prediction): {question[:50]}...")
            stats["skipped"] += 1
            evaluated_records.append(record)
            continue

        logger.info(f"[{i+1}/{len(records)}] Evaluating: {question[:50]}...")

        start_time = time.time()

        try:
            predicted_answer, evidence_counts, context = evaluate_question(
                question=question,
                rag_graph=rag_graph,
                hybrid_search=hybrid_search,
                top_k=top_k,
            )

            elapsed = time.time() - start_time
            total_time += elapsed

            # Update record with new prediction
            # Match the format from the input file: [predicted_text, evidence_counts]
            record["predicted_answer"] = [predicted_answer, evidence_counts]
            record["context"] = context

            stats["evaluated"] += 1
            logger.info(f"  Completed in {elapsed:.2f}s, answer length: {len(predicted_answer)}")

        except Exception as e:
            logger.error(f"  Error: {e}")
            record["predicted_answer"] = [f"Error: {str(e)}", {"entity": 0, "relation": 0, "community": 0, "text_units": 0, "total_evidence": 0}]
            record["context"] = ""
            stats["errors"] += 1

        evaluated_records.append(record)

        # Save intermediate results
        if (i + 1) % 5 == 0:
            save_jsonl(evaluated_records, output_file)
            logger.info(f"  Saved intermediate results ({i+1}/{len(records)})")

    # Final save
    save_jsonl(evaluated_records, output_file)

    # Calculate final stats
    stats["end_time"] = datetime.now().isoformat()
    if stats["evaluated"] > 0:
        stats["avg_time_per_question"] = total_time / stats["evaluated"]

    return stats


def print_statistics(stats: Dict[str, Any]) -> None:
    """Print evaluation statistics."""
    print("\n" + "=" * 60)
    print("EVALUATION STATISTICS")
    print("=" * 60)
    print(f"Total records:           {stats['total']}")
    print(f"Evaluated:               {stats['evaluated']}")
    print(f"Skipped:                 {stats['skipped']}")
    print(f"Errors:                  {stats['errors']}")
    print(f"Avg time per question:   {stats['avg_time_per_question']:.2f}s")
    print(f"Start time:              {stats['start_time']}")
    print(f"End time:                {stats['end_time']}")
    print("=" * 60)


def show_comparison(input_file: Path, output_file: Path, num_samples: int = 3) -> None:
    """Show comparison between ground truth and predicted answers."""
    input_records = load_jsonl(input_file)
    output_records = load_jsonl(output_file)

    print("\n" + "=" * 80)
    print("SAMPLE COMPARISONS (Ground Truth vs Predicted)")
    print("=" * 80)

    for i, (inp, out) in enumerate(zip(input_records[:num_samples], output_records[:num_samples])):
        print(f"\n--- Question {i+1} ---")
        print(f"Q: {inp['question'][:100]}...")
        print(f"\nGround Truth: {inp['answer'][:200]}...")

        predicted = out.get("predicted_answer", ["N/A", {}])
        if isinstance(predicted, list) and len(predicted) > 0:
            pred_text = predicted[0] if isinstance(predicted[0], str) else str(predicted[0])
        else:
            pred_text = str(predicted)

        print(f"\nPredicted: {pred_text[:200]}...")
        print(f"\nQuestion Type: {inp.get('question_type', 'N/A')}")
        print(f"Reasoning Type: {inp.get('reasoning_type', 'N/A')}")
        print(f"Num Hops: {inp.get('num_hops', 'N/A')}")
        print("-" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate TKG answers using the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all questions
  uv run python scripts/evaluate_tkg.py --input data/tkg_answers_from_paper_SKX.jsonl --output data/answered.jsonl

  # Evaluate with limit
  uv run python scripts/evaluate_tkg.py --input data/questions.jsonl --output data/answered.jsonl --limit 10

  # Skip already answered questions
  uv run python scripts/evaluate_tkg.py --input data/questions.jsonl --output data/answered.jsonl --skip-existing

  # Show comparison after evaluation
  uv run python scripts/evaluate_tkg.py --input data/questions.jsonl --output data/answered.jsonl --show-comparison
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSONL file with questions",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluated.jsonl"),
        help="Path to output JSONL file (default: data/evaluated.jsonl)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of retrieval results per question (default: 10)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of questions to evaluate",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip questions that already have predictions",
    )

    parser.add_argument(
        "--show-comparison",
        action="store_true",
        help="Show comparison between ground truth and predictions after evaluation",
    )

    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Only show comparison (no evaluation)",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.comparison_only:
        if not args.output.exists():
            print(f"Error: Output file not found for comparison: {args.output}")
            sys.exit(1)
        show_comparison(args.input, args.output)
        return

    try:
        # Run evaluation
        print(f"Starting evaluation...")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Top-K: {args.top_k}")
        if args.limit:
            print(f"Limit: {args.limit}")

        stats = evaluate_dataset(
            input_file=args.input,
            output_file=args.output,
            top_k=args.top_k,
            limit=args.limit,
            skip_existing=args.skip_existing,
        )

        print_statistics(stats)

        if args.show_comparison:
            show_comparison(args.input, args.output)

        print(f"\nResults saved to: {args.output}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
