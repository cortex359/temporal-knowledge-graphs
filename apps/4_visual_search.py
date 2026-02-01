"""
Visual Search Interface for Temporal Knowledge Graph.

This app provides an interactive visualization of the search process in a
temporal knowledge graph, showing:
- Query understanding and entity extraction
- Automatic temporal context detection via LLM
- Vector search results with similarity visualization
- Graph traversal paths and entity relationships
- Temporal timeline of results
- Combined hybrid search scoring
"""

import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_kg_rag.models.temporal import TemporalFilter, TemporalQueryType, TemporalContext
from temporal_kg_rag.retrieval.hybrid_search import get_hybrid_search
from temporal_kg_rag.retrieval.graph_search import get_graph_search
from temporal_kg_rag.retrieval.vector_search import get_vector_search
from temporal_kg_rag.retrieval.temporal_retrieval import get_temporal_retrieval
from temporal_kg_rag.graph.operations import get_graph_operations
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Visual Knowledge Graph Search",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for visual styling
st.markdown("""
<style>
/* Main container */
.main-container {
    padding: 0 1rem;
}

/* Search box styling */
.search-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
}

.search-container h1 {
    color: white;
    text-align: center;
    margin-bottom: 1rem;
}

/* Step cards */
.step-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    border-left: 5px solid #667eea;
}

.step-card.active {
    border-left-color: #10b981;
    animation: pulse 1s ease-in-out;
}

.step-card.completed {
    border-left-color: #10b981;
    opacity: 0.9;
}

.step-card.temporal {
    border-left-color: #8b5cf6;
    background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
}

.step-number {
    display: inline-block;
    width: 30px;
    height: 30px;
    background: #667eea;
    color: white;
    border-radius: 50%;
    text-align: center;
    line-height: 30px;
    font-weight: bold;
    margin-right: 10px;
}

.step-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #1f2937;
}

/* Temporal badge */
.temporal-badge {
    display: inline-block;
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 600;
    margin: 0.5rem 0;
}

.temporal-info {
    background: #f5f3ff;
    border: 1px solid #c4b5fd;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}

/* Result cards */
.result-card {
    background: #f8fafc;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    border: 1px solid #e2e8f0;
    transition: all 0.3s ease;
}

.result-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transform: translateY(-2px);
}

/* Score bar */
.score-bar {
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
    margin: 0.5rem 0;
}

.score-fill {
    height: 100%;
    background: linear-gradient(90deg, #10b981 0%, #3b82f6 100%);
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* Entity tags */
.entity-tag {
    display: inline-block;
    padding: 4px 10px;
    margin: 2px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
}

.entity-PERSON { background: #fef3c7; color: #92400e; }
.entity-ORG { background: #dbeafe; color: #1e40af; }
.entity-LOCATION { background: #d1fae5; color: #065f46; }
.entity-DATE { background: #ede9fe; color: #5b21b6; }
.entity-CONCEPT { background: #fce7f3; color: #9d174d; }
.entity-default { background: #f3f4f6; color: #374151; }

/* Timeline */
.timeline-container {
    position: relative;
    padding: 1rem 0;
}

.timeline-line {
    position: absolute;
    left: 50%;
    top: 0;
    bottom: 0;
    width: 2px;
    background: #e2e8f0;
}

.timeline-item {
    position: relative;
    margin: 1rem 0;
    padding: 1rem;
    width: 45%;
}

.timeline-item.left {
    margin-right: 55%;
    text-align: right;
}

.timeline-item.right {
    margin-left: 55%;
}

.timeline-dot {
    position: absolute;
    width: 16px;
    height: 16px;
    background: #667eea;
    border-radius: 50%;
    border: 3px solid white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

/* Metrics */
.metric-card {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #667eea;
}

.metric-label {
    font-size: 0.9rem;
    color: #6b7280;
}

/* Relationship card */
.relationship-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.relationship-arrow {
    color: #667eea;
    font-weight: bold;
    margin: 0 0.5rem;
}

/* Animation */
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(102, 126, 234, 0); }
    100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0); }
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.5s ease-out;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "search_state" not in st.session_state:
    st.session_state.search_state = {
        "query": "",
        "step": 0,
        "results": None,
        "vector_results": None,
        "graph_results": None,
        "entities_extracted": [],
        "temporal_context": None,
        "search_time": 0,
        "graph_data": None,
    }


def get_entity_color(entity_type: Optional[str]) -> str:
    """Get color for entity type."""
    if not entity_type:
        return "#6b7280"

    colors = {
        "PERSON": "#fbbf24",
        "ORG": "#3b82f6",
        "ORGANIZATION": "#3b82f6",
        "LOCATION": "#10b981",
        "GPE": "#10b981",
        "DATE": "#8b5cf6",
        "TIME": "#8b5cf6",
        "CONCEPT": "#ec4899",
        "PRODUCT": "#f97316",
        "EVENT": "#06b6d4",
        "TECHNOLOGY": "#6366f1",
    }
    return colors.get(entity_type.upper(), "#6b7280")


def render_entity_tag(name: Optional[str], entity_type: Optional[str]) -> str:
    """Render an entity as a colored tag."""
    if not name:
        return ""
    color = get_entity_color(entity_type)
    return f'<span class="entity-tag" style="background: {color}20; color: {color}; border: 1px solid {color}40;">{name}</span>'


def extract_query_entities(query: str) -> List[Dict]:
    """Extract entities from query using graph search."""
    try:
        gs = get_graph_search()
        entity_names = gs.extract_entities_from_query(query)

        # Get entity details from graph
        graph_ops = get_graph_operations()
        entities = []

        for name in entity_names:
            # Try to find in graph
            result = graph_ops.client.execute_read_transaction(
                "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($name) RETURN e LIMIT 1",
                {"name": name}
            )
            if result:
                entity_data = result[0]["e"]
                entities.append({
                    "name": entity_data.get("name", name),
                    "type": entity_data.get("type", "CONCEPT"),
                    "id": entity_data.get("id", ""),
                })
            else:
                entities.append({
                    "name": name,
                    "type": "CONCEPT",
                    "id": "",
                })

        return entities
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return []


def extract_temporal_context(query: str) -> Optional[TemporalContext]:
    """Extract temporal context from query using LLM."""
    try:
        tr = get_temporal_retrieval()
        context = tr.parse_temporal_context(query)
        return context
    except Exception as e:
        logger.error(f"Temporal context extraction failed: {e}")
        return None


def perform_visual_search(
    query: str,
    top_k: int = 10,
    manual_temporal_filter: Optional[TemporalFilter] = None,
    auto_temporal: bool = True
) -> Dict:
    """Perform search and collect visualization data."""
    search_data = {
        "query": query,
        "steps": [],
        "vector_results": [],
        "graph_results": [],
        "hybrid_results": [],
        "entities": [],
        "relationships": [],
        "temporal_context": None,
        "temporal_filter": None,
        "timings": {},
    }

    total_start = time.time()

    # Step 0: Temporal Context Detection (if auto-detection enabled)
    temporal_filter = manual_temporal_filter
    if auto_temporal and manual_temporal_filter is None:
        step_start = time.time()
        temporal_context = extract_temporal_context(query)
        search_data["temporal_context"] = temporal_context
        search_data["timings"]["temporal_detection"] = time.time() - step_start

        if temporal_context and temporal_context.has_temporal_reference:
            temporal_filter = temporal_context.temporal_filter
            search_data["temporal_filter"] = temporal_filter

            # Describe what was detected
            temporal_desc = "Detected: "
            if temporal_context.original_temporal_phrase:
                temporal_desc += f'"{temporal_context.original_temporal_phrase}"'
            if temporal_filter:
                temporal_desc += f" -> {temporal_filter.query_type.value}"
                if temporal_filter.point_in_time:
                    temporal_desc += f" ({temporal_filter.point_in_time.strftime('%Y-%m-%d')})"
                elif temporal_filter.start_time and temporal_filter.end_time:
                    temporal_desc += f" ({temporal_filter.start_time.strftime('%Y-%m-%d')} to {temporal_filter.end_time.strftime('%Y-%m-%d')})"

            search_data["steps"].append({
                "name": "Temporal Detection",
                "description": temporal_desc,
                "is_temporal": True,
                "data": {
                    "has_temporal": True,
                    "type": temporal_filter.query_type.value if temporal_filter else None,
                    "phrase": temporal_context.original_temporal_phrase,
                },
                "time": search_data["timings"]["temporal_detection"],
            })
        else:
            search_data["steps"].append({
                "name": "Temporal Detection",
                "description": "No temporal reference detected in query",
                "is_temporal": True,
                "data": {"has_temporal": False},
                "time": search_data["timings"]["temporal_detection"],
            })
    elif manual_temporal_filter:
        search_data["temporal_filter"] = manual_temporal_filter
        search_data["steps"].append({
            "name": "Temporal Filter",
            "description": f"Manual filter applied: {manual_temporal_filter.query_type.value}",
            "is_temporal": True,
            "data": {"manual": True, "type": manual_temporal_filter.query_type.value},
            "time": 0,
        })

    # Step 1: Query Understanding (Entity Extraction)
    step_start = time.time()
    entities = extract_query_entities(query)
    search_data["entities"] = entities
    search_data["timings"]["entity_extraction"] = time.time() - step_start
    search_data["steps"].append({
        "name": "Entity Extraction",
        "description": f"Extracted {len(entities)} entities from query",
        "data": entities,
        "time": search_data["timings"]["entity_extraction"],
    })

    # Step 2: Vector Search
    step_start = time.time()
    try:
        vs = get_vector_search()
        vector_results = vs.search(query, top_k=top_k, temporal_filter=temporal_filter)
        search_data["vector_results"] = vector_results
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        vector_results = []
    search_data["timings"]["vector_search"] = time.time() - step_start
    search_data["steps"].append({
        "name": "Vector Search",
        "description": f"Found {len(vector_results)} chunks via semantic similarity",
        "data": vector_results,
        "time": search_data["timings"]["vector_search"],
    })

    # Step 3: Graph Search
    step_start = time.time()
    try:
        gs = get_graph_search()
        if entities:
            entity_names = [e["name"] for e in entities]
            graph_results = gs.search_by_entities(entity_names, top_k=top_k, temporal_filter=temporal_filter)
        else:
            graph_results = []
        search_data["graph_results"] = graph_results
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        graph_results = []
    search_data["timings"]["graph_search"] = time.time() - step_start
    search_data["steps"].append({
        "name": "Graph Traversal",
        "description": f"Found {len(graph_results)} chunks via entity relationships",
        "data": graph_results,
        "time": search_data["timings"]["graph_search"],
    })

    # Step 4: Get Entity Relationships
    step_start = time.time()
    relationships = []
    if entities:
        try:
            graph_ops = get_graph_operations()
            for entity in entities[:5]:  # Limit to first 5 entities
                if entity.get("id"):
                    rels = graph_ops.get_entity_relationships(entity["id"], direction="both")
                    for rel in rels[:3]:  # Limit relationships per entity
                        relationships.append(rel)
        except Exception as e:
            logger.error(f"Relationship fetch failed: {e}")
    search_data["relationships"] = relationships
    search_data["timings"]["relationship_fetch"] = time.time() - step_start

    # Step 5: Hybrid Fusion
    step_start = time.time()
    try:
        hs = get_hybrid_search()
        hybrid_results = hs.search(query=query, top_k=top_k, temporal_filter=temporal_filter)
        search_data["hybrid_results"] = hybrid_results
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        hybrid_results = []
    search_data["timings"]["hybrid_fusion"] = time.time() - step_start
    search_data["steps"].append({
        "name": "Hybrid Fusion (RRF)",
        "description": f"Combined {len(hybrid_results)} results using Reciprocal Rank Fusion",
        "data": hybrid_results,
        "time": search_data["timings"]["hybrid_fusion"],
    })

    search_data["total_time"] = time.time() - total_start

    return search_data


def build_search_graph(search_data: Dict) -> tuple:
    """Build graph visualization from search data."""
    nodes = []
    edges = []

    # Add query node
    nodes.append(Node(
        id="query",
        label=search_data["query"][:30] + "..." if len(search_data["query"]) > 30 else search_data["query"],
        size=35,
        color="#667eea",
        font={"color": "white"},
    ))

    # Add entity nodes
    for i, entity in enumerate(search_data.get("entities", [])):
        node_id = f"entity_{i}"
        nodes.append(Node(
            id=node_id,
            label=entity["name"],
            size=25,
            color=get_entity_color(entity["type"]),
            title=f"Type: {entity['type']}",
        ))
        edges.append(Edge(
            source="query",
            target=node_id,
            label="mentions",
            color="#94a3b8",
        ))

    # Add result chunk nodes (top 5)
    for i, result in enumerate(search_data.get("hybrid_results", [])[:5]):
        node_id = f"chunk_{i}"
        score = result.get("rrf_score", result.get("score", 0))
        nodes.append(Node(
            id=node_id,
            label=f"Chunk {i+1}",
            size=20 + score * 20,
            color="#10b981",
            title=result.get("text", "")[:200],
        ))

        # Connect to entities that appear in this chunk
        chunk_entities = result.get("entities", [])
        for j, entity in enumerate(search_data.get("entities", [])):
            entity_names_lower = [(e.get("name") or "").lower() if isinstance(e, dict) else str(e or "").lower()
                                  for e in chunk_entities if e]
            entity_name = entity.get("name") or ""
            if entity_name and entity_name.lower() in entity_names_lower:
                edges.append(Edge(
                    source=f"entity_{j}",
                    target=node_id,
                    color="#d1d5db",
                ))

    # Add relationship edges between entities
    for rel in search_data.get("relationships", [])[:10]:
        source_name = rel.get("source") or ""
        target_name = rel.get("target") or ""

        # Find matching entity nodes
        source_idx = None
        target_idx = None
        for i, entity in enumerate(search_data.get("entities", [])):
            entity_name = entity.get("name") or ""
            if entity_name and source_name and entity_name.lower() == source_name.lower():
                source_idx = i
            if entity_name and target_name and entity_name.lower() == target_name.lower():
                target_idx = i

        if source_idx is not None and target_idx is not None:
            edges.append(Edge(
                source=f"entity_{source_idx}",
                target=f"entity_{target_idx}",
                label=rel.get("relationship", "related")[:15],
                color="#f59e0b",
            ))

    return nodes, edges


def render_timeline(results: List[Dict]) -> None:
    """Render temporal timeline of results."""
    # Group results by date
    dated_results = []
    for r in results:
        created_at = r.get("created_at")
        if created_at:
            if isinstance(created_at, str):
                try:
                    date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except:
                    date = datetime.now()
            else:
                date = created_at
            dated_results.append((date, r))

    if not dated_results:
        st.info("No temporal data available for results")
        return

    # Sort by date
    dated_results.sort(key=lambda x: x[0])

    # Create timeline visualization
    st.markdown("### Temporal Distribution")

    # Simple bar chart of results over time
    date_counts = {}
    for date, r in dated_results:
        month_key = date.strftime("%Y-%m")
        date_counts[month_key] = date_counts.get(month_key, 0) + 1

    if date_counts:
        import pandas as pd
        df = pd.DataFrame(list(date_counts.items()), columns=["Month", "Results"])
        df = df.sort_values("Month")
        st.bar_chart(df.set_index("Month"))

    # Timeline cards
    st.markdown("### Timeline View")
    for i, (date, result) in enumerate(dated_results[:10]):
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"**{date.strftime('%Y-%m-%d')}**")
        with col2:
            with st.container():
                st.markdown(f"""
                <div class="result-card">
                    <strong>{result.get('document_title', 'Unknown')}</strong>
                    <p style="margin: 0.5rem 0; color: #6b7280; font-size: 0.9rem;">
                        {result.get('text', '')[:150]}...
                    </p>
                </div>
                """, unsafe_allow_html=True)


def render_search_steps(search_data: Dict) -> None:
    """Render the search process steps."""
    st.markdown("## Search Process Visualization")

    for i, step in enumerate(search_data.get("steps", [])):
        is_temporal = step.get("is_temporal", False)
        card_class = "temporal" if is_temporal else "completed"
        step_time = step.get('time') or 0.0
        step_name = step.get('name') or f"Step {i+1}"
        step_desc = step.get('description') or ""

        st.markdown(f"""
        <div class="step-card {card_class} fade-in" style="animation-delay: {i * 0.1}s;">
            <span class="step-number" style="background: {'#8b5cf6' if is_temporal else '#667eea'};">{i + 1}</span>
            <span class="step-title">{step_name}</span>
            <span style="float: right; color: #6b7280;">{step_time:.3f}s</span>
            <p style="margin: 0.5rem 0 0 40px; color: #6b7280;">{step_desc}</p>
        </div>
        """, unsafe_allow_html=True)


def render_temporal_context(search_data: Dict) -> None:
    """Render detected temporal context information."""
    temporal_context = search_data.get("temporal_context")
    temporal_filter = search_data.get("temporal_filter")

    if not temporal_context or not temporal_context.has_temporal_reference:
        return

    st.markdown("""
    <div class="temporal-info">
        <h4 style="margin: 0 0 0.5rem 0; color: #5b21b6;">
            <span style="margin-right: 8px;">&#128337;</span> Temporal Context Detected
        </h4>
    """, unsafe_allow_html=True)

    # Show detected phrase
    if temporal_context.original_temporal_phrase:
        st.markdown(f"""
        <p><strong>Detected phrase:</strong> "{temporal_context.original_temporal_phrase}"</p>
        """, unsafe_allow_html=True)

    # Show filter type and parameters
    if temporal_filter:
        filter_type = temporal_filter.query_type.value
        st.markdown(f'<span class="temporal-badge">{filter_type.upper()}</span>', unsafe_allow_html=True)

        if temporal_filter.point_in_time:
            st.markdown(f"**Point in time:** {temporal_filter.point_in_time.strftime('%Y-%m-%d')}")
        elif temporal_filter.start_time and temporal_filter.end_time:
            st.markdown(f"**Time range:** {temporal_filter.start_time.strftime('%Y-%m-%d')} to {temporal_filter.end_time.strftime('%Y-%m-%d')}")

    st.markdown("</div>", unsafe_allow_html=True)


# ============== MAIN APP ==============

# Header
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 2.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        Visual Knowledge Graph Search
    </h1>
    <p style="color: #6b7280; font-size: 1.1rem;">
        Explore how your query traverses the temporal knowledge graph
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("### Search Settings")

    top_k = st.slider("Number of results", 5, 30, 10)

    st.markdown("### Temporal Settings")

    auto_temporal = st.checkbox(
        "Auto-detect temporal context",
        value=True,
        help="Automatically detect time references in your query using LLM"
    )

    use_manual_temporal = st.checkbox("Use manual temporal filter", value=False)

    manual_temporal_filter = None
    if use_manual_temporal:
        filter_type = st.selectbox(
            "Filter type",
            ["Time Range", "Point in Time", "Latest Only"]
        )

        if filter_type == "Time Range":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("From", datetime.now() - timedelta(days=365))
            with col2:
                end_date = st.date_input("To", datetime.now())
            manual_temporal_filter = TemporalFilter(
                query_type=TemporalQueryType.TIME_RANGE,
                start_time=datetime.combine(start_date, datetime.min.time()),
                end_time=datetime.combine(end_date, datetime.max.time()),
            )
        elif filter_type == "Point in Time":
            point_date = st.date_input("Date", datetime.now())
            manual_temporal_filter = TemporalFilter(
                query_type=TemporalQueryType.POINT_IN_TIME,
                point_in_time=datetime.combine(point_date, datetime.min.time()),
            )
        else:
            manual_temporal_filter = TemporalFilter(query_type=TemporalQueryType.LATEST)

    st.markdown("### Visualization")
    show_graph = st.checkbox("Show Knowledge Graph", value=True)
    show_timeline = st.checkbox("Show Timeline", value=True)
    show_steps = st.checkbox("Show Search Steps", value=True)

# Search input
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "Search Query",
        placeholder="Enter your question (temporal references like 'in Q1 2021' or 'latest' are auto-detected)...",
        label_visibility="collapsed",
    )
with col2:
    search_clicked = st.button("Search", type="primary", use_container_width=True)

# Example queries with temporal references
with st.expander("Example Queries with Temporal References"):
    st.markdown("""
    **Automatic temporal detection examples:**
    - `What were the Q1 2021 earnings results?` - Detects Q1 2021
    - `Show me the latest financial reports` - Detects "latest"
    - `How did revenue change between 2020 and 2023?` - Detects time range
    - `What happened on January 15, 2024?` - Detects specific date
    - `Recent developments in AI technology` - Detects "recent"

    **Queries without temporal context:**
    - `What is the company's business model?`
    - `Explain the risk factors`
    """)

# Perform search
if search_clicked and query:
    with st.spinner("Analyzing query and searching knowledge graph..."):
        search_data = perform_visual_search(
            query,
            top_k=top_k,
            manual_temporal_filter=manual_temporal_filter if use_manual_temporal else None,
            auto_temporal=auto_temporal and not use_manual_temporal
        )
        st.session_state.search_state["results"] = search_data

# Display results
if st.session_state.search_state.get("results"):
    search_data = st.session_state.search_state["results"]

    # Show temporal context if detected
    render_temporal_context(search_data)

    # Metrics row
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(search_data.get('hybrid_results', []))}</div>
            <div class="metric-label">Results</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(search_data.get('entities', []))}</div>
            <div class="metric-label">Entities</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(search_data.get('relationships', []))}</div>
            <div class="metric-label">Relations</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{search_data.get('total_time', 0):.2f}s</div>
            <div class="metric-label">Total Time</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        temporal_status = "Yes" if search_data.get("temporal_filter") else "No"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {'#8b5cf6' if search_data.get('temporal_filter') else '#6b7280'};">{temporal_status}</div>
            <div class="metric-label">Temporal Filter</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Results", "Knowledge Graph", "Timeline", "Search Process"])

    with tab1:
        st.markdown("### Search Results")

        # Extracted entities
        if search_data.get("entities"):
            st.markdown("**Extracted Entities:**")
            entity_html = " ".join([
                render_entity_tag(e["name"], e["type"])
                for e in search_data["entities"]
            ])
            st.markdown(entity_html, unsafe_allow_html=True)
            st.markdown("")

        # Results
        for i, result in enumerate(search_data.get("hybrid_results", [])):
            score = result.get("rrf_score", result.get("score", 0))
            doc_title = result.get("document_title", "Unknown Document")
            text = result.get("text", "")
            created_at = result.get("created_at", "")

            with st.container():
                st.markdown(f"""
                <div class="result-card fade-in" style="animation-delay: {i * 0.05}s;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="font-size: 1.1rem;">#{i+1} {doc_title}</strong>
                        <span style="color: #667eea; font-weight: 600;">Score: {score:.4f}</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill" style="width: {min(score * 100, 100)}%;"></div>
                    </div>
                    <p style="margin: 0.5rem 0; color: #374151;">{text[:300]}{'...' if len(text) > 300 else ''}</p>
                    <div style="display: flex; justify-content: space-between; color: #6b7280; font-size: 0.85rem;">
                        <span>Created: {created_at}</span>
                        <span>Chunk: {result.get('chunk_number', 'N/A')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Entity tags for this result
                chunk_entities = result.get("entities", [])
                if chunk_entities:
                    entity_tags = []
                    for e in chunk_entities[:5]:
                        if isinstance(e, dict):
                            entity_tags.append(render_entity_tag(e.get("name", ""), e.get("type", "CONCEPT")))
                        elif e:
                            entity_tags.append(render_entity_tag(str(e), "CONCEPT"))
                    if entity_tags:
                        st.markdown(" ".join(entity_tags), unsafe_allow_html=True)

                st.markdown("")

    with tab2:
        if show_graph:
            st.markdown("### Knowledge Graph Visualization")
            st.markdown("*Shows query, extracted entities, and top result chunks*")

            nodes, edges = build_search_graph(search_data)

            if nodes:
                config = Config(
                    width=800,
                    height=500,
                    directed=True,
                    physics=True,
                    hierarchical=False,
                    nodeHighlightBehavior=True,
                    highlightColor="#f59e0b",
                    collapsible=False,
                    node={"labelProperty": "label"},
                    link={"labelProperty": "label", "renderLabel": True},
                )

                agraph(nodes=nodes, edges=edges, config=config)
            else:
                st.info("No graph data to display")

            # Relationships table
            if search_data.get("relationships"):
                st.markdown("### Entity Relationships")
                for rel in search_data["relationships"][:10]:
                    st.markdown(f"""
                    <div class="relationship-card">
                        <strong>{rel.get('source', 'Unknown')}</strong>
                        <span class="relationship-arrow">-> {rel.get('relationship', 'related')} -></span>
                        <strong>{rel.get('target', 'Unknown')}</strong>
                        <p style="margin: 0.5rem 0 0 0; color: #6b7280; font-size: 0.85rem;">
                            {rel.get('description', '')[:150]}{'...' if len(rel.get('description', '')) > 150 else ''}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Enable 'Show Knowledge Graph' in the sidebar")

    with tab3:
        if show_timeline:
            render_timeline(search_data.get("hybrid_results", []))
        else:
            st.info("Enable 'Show Timeline' in the sidebar")

    with tab4:
        if show_steps:
            render_search_steps(search_data)

            # Timing breakdown
            st.markdown("### Timing Breakdown")
            timings = search_data.get("timings", {})
            if timings:
                import pandas as pd
                timing_df = pd.DataFrame([
                    {"Step": k.replace("_", " ").title(), "Time (s)": v}
                    for k, v in timings.items()
                ])
                st.bar_chart(timing_df.set_index("Step"))
        else:
            st.info("Enable 'Show Search Steps' in the sidebar")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6b7280;'>"
    "Visual Knowledge Graph Search | Temporal Knowledge Graph RAG System"
    "</div>",
    unsafe_allow_html=True,
)
