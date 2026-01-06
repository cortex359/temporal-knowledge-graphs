"""Streamlit app for chunk retrieval with different search strategies."""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_kg_rag.models.temporal import TemporalFilter, TemporalQueryType
from temporal_kg_rag.retrieval.hybrid_search import get_hybrid_search
from temporal_kg_rag.retrieval.graph_search import get_graph_search
from temporal_kg_rag.retrieval.vector_search import get_vector_search
from temporal_kg_rag.retrieval.temporal_retrieval import get_temporal_retrieval
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Chunk Retrieval Interface",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .result-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid #4ecdc4;
    }
    .score-badge {
        background-color: #4ecdc4;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }
    .source-badge {
        background-color: #e8f4f8;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 2px;
        font-size: 0.9em;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "search_metadata" not in st.session_state:
    st.session_state.search_metadata = {}


def highlight_query_terms(text: str, query: str) -> str:
    """Highlight query terms in text."""
    if not query:
        return text

    words = query.lower().split()
    highlighted = text

    for word in words:
        if len(word) > 2:  # Only highlight words longer than 2 chars
            # Simple case-insensitive replacement
            import re

            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted = pattern.sub(
                lambda m: f'<span class="highlight">{m.group()}</span>',
                highlighted,
            )

    return highlighted


def format_result_card(result: Dict, rank: int, query: str = "") -> str:
    """Format a search result as an HTML card."""
    text = result.get("text", "")
    doc_title = result.get("document_title", "Unknown Document")
    created_at = result.get("created_at", "")
    score = result.get("rrf_score") or result.get("score", 0)
    chunk_number = result.get("chunk_number", "?")

    # Highlight query terms
    highlighted_text = highlight_query_terms(text, query)

    # Format entities if available
    entities_html = ""
    if "entities" in result and result["entities"]:
        entities = result["entities"]
        if isinstance(entities[0], dict):
            entity_names = [e.get("name", str(e)) for e in entities[:5]]
        else:
            entity_names = entities[:5] if isinstance(entities, list) else []

        if entity_names:
            entities_html = "<br><strong>Entities:</strong> " + ", ".join(
                [f'<span class="source-badge">{e}</span>' for e in entity_names]
            )

    card_html = f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h4 style="margin: 0;">#{rank} - Chunk {chunk_number}</h4>
            <span class="score-badge">Score: {score:.4f}</span>
        </div>
        <p style="margin: 10px 0;">{highlighted_text[:500]}{"..." if len(text) > 500 else ""}</p>
        <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
            <strong>Source:</strong> {doc_title}<br>
            <strong>Created:</strong> {created_at}
            {entities_html}
        </div>
    </div>
    """
    return card_html


def perform_search(
    query: str,
    method: str,
    top_k: int,
    temporal_filter: Optional[TemporalFilter] = None,
    entity_filter: Optional[List[str]] = None,
) -> tuple:
    """Perform search with specified method."""
    start_time = time.time()
    results = []

    try:
        if method == "Vector Search":
            vs = get_vector_search()
            results = vs.search(query, top_k=top_k, temporal_filter=temporal_filter)

        elif method == "Graph Search":
            gs = get_graph_search()
            if entity_filter:
                results = gs.search_by_entities(
                    entity_filter,
                    top_k=top_k,
                    temporal_filter=temporal_filter,
                )
            else:
                # Extract entities from query
                entities = gs.extract_entities_from_query(query)
                if entities:
                    results = gs.search_by_entities(
                        entities,
                        top_k=top_k,
                        temporal_filter=temporal_filter,
                    )
                else:
                    # Fallback to full-text search
                    results = gs.full_text_search(query, top_k=top_k)

        elif method == "Hybrid Search":
            hs = get_hybrid_search()
            results = hs.search(
                query=query,
                top_k=top_k,
                temporal_filter=temporal_filter,
            )

        elif method == "Temporal Search":
            tr = get_temporal_retrieval()
            search_result = tr.search_with_temporal_context(
                query=query,
                top_k=top_k,
                auto_detect_temporal=True,
            )
            results = search_result.get("results", [])

        elapsed = time.time() - start_time

        metadata = {
            "num_results": len(results),
            "elapsed_time": elapsed,
            "method": method,
        }

        return results, metadata

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return [], {"error": str(e)}


# Sidebar
with st.sidebar:
    st.title("🔍 Search Configuration")

    # Search method
    st.subheader("Search Method")
    search_method = st.selectbox(
        "Select method",
        [
            "Hybrid Search",
            "Vector Search",
            "Graph Search",
            "Temporal Search",
        ],
        help="Hybrid combines vector and graph search. Temporal auto-detects temporal queries.",
    )

    # Number of results
    st.subheader("Results")
    top_k = st.slider("Number of results", min_value=1, max_value=50, value=10)

    # Temporal filtering
    st.subheader("🕐 Temporal Filtering")
    use_temporal = st.checkbox("Enable temporal filter")

    temporal_filter = None
    if use_temporal:
        filter_type = st.radio(
            "Filter Type",
            ["Point in Time", "Time Range", "Latest Only"],
        )

        if filter_type == "Point in Time":
            point_date = st.date_input(
                "Select Date",
                value=datetime.now(),
            )
            temporal_filter = TemporalFilter(
                query_type=TemporalQueryType.POINT_IN_TIME,
                point_in_time=datetime.combine(point_date, datetime.min.time()),
            )

        elif filter_type == "Time Range":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365),
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now(),
                )
            temporal_filter = TemporalFilter(
                query_type=TemporalQueryType.TIME_RANGE,
                start_time=datetime.combine(start_date, datetime.min.time()),
                end_time=datetime.combine(end_date, datetime.max.time()),
            )

        elif filter_type == "Latest Only":
            temporal_filter = TemporalFilter(
                query_type=TemporalQueryType.LATEST,
            )

    # Entity filtering (for graph search)
    st.subheader("🏷️ Entity Filtering")
    use_entity_filter = st.checkbox("Filter by entities")

    entity_filter = None
    if use_entity_filter:
        entity_input = st.text_input(
            "Entity names (comma-separated)",
            placeholder="e.g., OpenAI, GPT-4",
        )
        if entity_input:
            entity_filter = [e.strip() for e in entity_input.split(",")]

    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        if search_method == "Hybrid Search":
            alpha = st.slider(
                "Vector/Graph Balance",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="0 = Graph only, 1 = Vector only",
            )

        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Minimum similarity score for results",
        )

# Main content
st.title("🔍 Chunk Retrieval Interface")
st.markdown("Search and retrieve text chunks using different strategies")

# Search input
search_query = st.text_input(
    "Enter your search query",
    placeholder="e.g., What is artificial intelligence?",
    key="search_input",
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)
with col3:
    compare_button = st.button("📊 Compare Methods", use_container_width=True)

# Clear results
if clear_button:
    st.session_state.search_results = None
    st.session_state.search_metadata = {}
    st.rerun()

# Compare methods
if compare_button and search_query:
    st.subheader("🔬 Method Comparison")

    methods = ["Vector Search", "Graph Search", "Hybrid Search"]
    comparison_results = {}

    with st.spinner("Comparing methods..."):
        for method in methods:
            results, metadata = perform_search(
                search_query,
                method,
                top_k=5,
                temporal_filter=temporal_filter,
            )
            comparison_results[method] = {
                "results": results,
                "metadata": metadata,
            }

    # Display comparison
    cols = st.columns(len(methods))
    for i, method in enumerate(methods):
        with cols[i]:
            data = comparison_results[method]
            metadata = data["metadata"]

            st.markdown(f"### {method}")
            st.markdown(
                f"""
                <div class="metric-box">
                    <h2>{metadata.get('num_results', 0)}</h2>
                    <p>Results</p>
                    <p style="font-size: 0.8em; color: #666;">
                        {metadata.get('elapsed_time', 0):.3f}s
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if data["results"]:
                with st.expander("View Results"):
                    for j, result in enumerate(data["results"][:3], 1):
                        st.markdown(f"**#{j}** - {result.get('text', '')[:100]}...")
                        st.caption(f"Score: {result.get('score', 0):.4f}")

# Perform search
if search_button and search_query:
    with st.spinner(f"Searching with {search_method}..."):
        results, metadata = perform_search(
            search_query,
            search_method,
            top_k,
            temporal_filter,
            entity_filter,
        )

        st.session_state.search_results = results
        st.session_state.search_metadata = metadata

# Display results
if st.session_state.search_results is not None:
    results = st.session_state.search_results
    metadata = st.session_state.search_metadata

    # Check for errors
    if "error" in metadata:
        st.error(f"❌ Search failed: {metadata['error']}")
        st.stop()

    # Results summary
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Results Found", metadata.get("num_results", 0))
    with col2:
        st.metric("⏱️ Time", f"{metadata.get('elapsed_time', 0):.3f}s")
    with col3:
        st.metric("🔍 Method", metadata.get("method", "Unknown"))
    with col4:
        if results:
            avg_score = sum(r.get("rrf_score", r.get("score", 0)) for r in results) / len(
                results
            )
            st.metric("📈 Avg Score", f"{avg_score:.4f}")

    st.markdown("---")

    # Display results
    if results:
        st.subheader(f"Search Results ({len(results)})")

        # Sorting options
        col1, col2 = st.columns([3, 1])
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                ["Relevance", "Date (Newest)", "Date (Oldest)"],
            )

        # Sort results
        if sort_by == "Date (Newest)":
            results = sorted(
                results,
                key=lambda x: x.get("created_at", ""),
                reverse=True,
            )
        elif sort_by == "Date (Oldest)":
            results = sorted(results, key=lambda x: x.get("created_at", ""))

        # Display each result
        for i, result in enumerate(results, 1):
            result_html = format_result_card(result, i, search_query)
            st.markdown(result_html, unsafe_allow_html=True)

            # Expandable details
            with st.expander("📋 Full Details"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Chunk ID:**", result.get("chunk_id", "N/A"))
                    st.write("**Document ID:**", result.get("document_id", "N/A"))
                    st.write("**Chunk Number:**", result.get("chunk_number", "N/A"))

                with col2:
                    st.write("**Score:**", result.get("rrf_score", result.get("score", 0)))
                    st.write("**Created:**", result.get("created_at", "N/A"))
                    st.write("**Is Current:**", result.get("is_current", True))

                st.write("**Full Text:**")
                st.text_area(
                    "Text",
                    value=result.get("text", ""),
                    height=150,
                    key=f"text_{i}",
                    disabled=True,
                )

        # Export results
        st.markdown("---")
        if st.button("📥 Export Results"):
            import json

            export_data = {
                "query": search_query,
                "method": search_method,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata,
                "results": [
                    {
                        "rank": i,
                        "chunk_id": r.get("chunk_id"),
                        "text": r.get("text"),
                        "score": r.get("rrf_score", r.get("score", 0)),
                        "document_title": r.get("document_title"),
                        "created_at": str(r.get("created_at", "")),
                    }
                    for i, r in enumerate(results, 1)
                ],
            }

            st.download_button(
                "Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

    else:
        st.info("No results found for your query. Try adjusting your search terms or filters.")

# Examples
with st.expander("💡 Example Queries"):
    st.markdown(
        """
    **Factual queries:**
    - What is machine learning?
    - Explain neural networks
    - Who created GPT-4?

    **Temporal queries:**
    - AI developments in 2023
    - Climate policy changes in the last year
    - Recent advances in quantum computing

    **Entity-based queries:**
    - Information about OpenAI
    - Documents mentioning Tesla
    - Chunks related to ChatGPT
    """
    )

# Footer
st.markdown("---")
st.markdown(
    "**Chunk Retrieval Interface** | "
    "Vector • Graph • Hybrid • Temporal Search"
)
