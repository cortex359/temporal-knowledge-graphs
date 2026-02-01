"""Streamlit app for temporal knowledge graph visualization and exploration."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_kg_rag.graph.neo4j_client import get_neo4j_client
from temporal_kg_rag.graph.operations import get_graph_operations
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Temporal Knowledge Graph Explorer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .entity-badge {
        background-color: #e8f4f8;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 2px;
        display: inline-block;
    }
    .temporal-badge {
        background-color: #fff4e6;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 2px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Initialize session state
if "neo4j_client" not in st.session_state:
    st.session_state.neo4j_client = None
if "graph_ops" not in st.session_state:
    st.session_state.graph_ops = None
if "selected_node" not in st.session_state:
    st.session_state.selected_node = None
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None
if "full_graph_data" not in st.session_state:
    st.session_state.full_graph_data = None


def init_connections():
    """Initialize Neo4j connections."""
    try:
        client = get_neo4j_client()
        if client.verify_connectivity():
            st.session_state.neo4j_client = client
            st.session_state.graph_ops = get_graph_operations()
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        return False


def get_database_stats() -> Dict:
    """Get database statistics."""
    try:
        client = st.session_state.neo4j_client
        stats = client.get_database_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {}


def get_entity_types() -> List[str]:
    """Get all entity types in the database."""
    try:
        client = st.session_state.neo4j_client
        query = """
        MATCH (e:Entity)
        RETURN DISTINCT e.type AS entity_type
        ORDER BY entity_type
        """
        result = client.execute_query(query)
        return [record["entity_type"] for record in result if record["entity_type"]]
    except Exception as e:
        logger.error(f"Failed to get entity types: {e}")
        return []


def search_entities(
    search_term: str,
    entity_type: Optional[str] = None,
    limit: int = 20,
) -> List[Dict]:
    """Search for entities by name."""
    try:
        client = st.session_state.neo4j_client

        type_filter = ""
        if entity_type and entity_type != "All":
            type_filter = "AND e.type = $entity_type"

        query = f"""
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($search_term)
        {type_filter}
        RETURN e.id AS id, e.name AS name, e.type AS type,
               e.first_seen AS first_seen, e.last_seen AS last_seen
        ORDER BY e.name
        LIMIT $limit
        """

        params = {"search_term": search_term, "limit": limit}
        if entity_type and entity_type != "All":
            params["entity_type"] = entity_type

        result = client.execute_query(query, params)
        return result
    except Exception as e:
        logger.error(f"Failed to search entities: {e}")
        return []


def get_documents(limit: int = 50, offset: int = 0) -> List[Dict]:
    """Get documents from database."""
    try:
        client = st.session_state.neo4j_client
        query = """
        MATCH (d:Document)
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        WITH d, count(c) AS chunk_count
        RETURN d.id AS id, d.title AS title, d.source AS source,
               d.created_at AS created_at, d.updated_at AS updated_at,
               chunk_count
        ORDER BY d.created_at DESC
        SKIP $offset
        LIMIT $limit
        """
        result = client.execute_query(query, {"limit": limit, "offset": offset})
        return result
    except Exception as e:
        logger.error(f"Failed to get documents: {e}")
        return []


def build_entity_neighborhood_graph(
    entity_name: str,
    max_depth: int = 2,
    max_entities: int = 20,
) -> Optional[Dict]:
    """Build graph data for entity neighborhood."""
    try:
        client = st.session_state.neo4j_client

        query = """
        MATCH (e:Entity {name: $entity_name})
        OPTIONAL MATCH path = (e)-[r:RELATES_TO*1..2]-(other:Entity)
        WITH e, other, r, path
        LIMIT $max_entities
        RETURN e, collect(DISTINCT other) AS neighbors, collect(DISTINCT r) AS relationships
        """

        result = client.execute_query(
            query,
            {"entity_name": entity_name, "max_entities": max_entities},
        )

        if not result:
            return None

        nodes = []
        edges = []
        seen_nodes = set()

        # Add central entity
        central = result[0]["e"]
        central_id = central["id"]
        nodes.append(
            Node(
                id=central_id,
                label=central["name"],
                title=f"{central['name']} ({central.get('type', 'Unknown')})",
                color="#ff6b6b",
                size=30,
            )
        )
        seen_nodes.add(central_id)

        # Add neighbors
        for neighbor in result[0].get("neighbors", []):
            if neighbor and neighbor["id"] not in seen_nodes:
                nodes.append(
                    Node(
                        id=neighbor["id"],
                        label=neighbor["name"][:20],
                        title=f"{neighbor['name']} ({neighbor.get('type', 'Unknown')})",
                        color="#4ecdc4",
                        size=20,
                    )
                )
                seen_nodes.add(neighbor["id"])
                edges.append(
                    Edge(
                        source=central_id,
                        target=neighbor["id"],
                        label="relates_to",
                    )
                )

        return {"nodes": nodes, "edges": edges}

    except Exception as e:
        logger.error(f"Failed to build entity graph: {e}")
        return None


def build_document_graph(
    document_id: str,
    include_entities: bool = True,
) -> Optional[Dict]:
    """Build graph data for document and its chunks."""
    try:
        client = st.session_state.neo4j_client

        # Get document and chunks
        query = """
        MATCH (d:Document {id: $document_id})
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.is_current = true
        """

        if include_entities:
            query += """
            OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
            RETURN d, collect(DISTINCT c) AS chunks, collect(DISTINCT e) AS entities
            """
        else:
            query += """
            RETURN d, collect(DISTINCT c) AS chunks, [] AS entities
            """

        result = client.execute_query(query, {"document_id": document_id})

        if not result:
            return None

        nodes = []
        edges = []

        # Add document node
        doc = result[0]["d"]
        doc_id = doc["id"]
        nodes.append(
            Node(
                id=doc_id,
                label=doc["title"][:30],
                title=f"Document: {doc['title']}",
                color="#8c7ae6",
                size=35,
                shape="box",
            )
        )

        # Add chunk nodes
        for chunk in result[0].get("chunks", []):
            if chunk:
                chunk_id = chunk["id"]
                nodes.append(
                    Node(
                        id=chunk_id,
                        label=f"Chunk {chunk.get('chunk_number', '?')}",
                        title=chunk["text"][:100],
                        color="#95afc0",
                        size=15,
                    )
                )
                edges.append(
                    Edge(
                        source=doc_id,
                        target=chunk_id,
                        label="HAS_CHUNK",
                    )
                )

        # Add entity nodes
        if include_entities:
            seen_entities = set()
            for entity in result[0].get("entities", []):
                if entity and entity["id"] not in seen_entities:
                    entity_id = entity["id"]
                    nodes.append(
                        Node(
                            id=entity_id,
                            label=entity["name"][:20],
                            title=f"{entity['name']} ({entity.get('type', 'Unknown')})",
                            color="#4ecdc4",
                            size=12,
                        )
                    )
                    seen_entities.add(entity_id)

                    # Connect to document (simplified)
                    edges.append(
                        Edge(
                            source=doc_id,
                            target=entity_id,
                            label="MENTIONS",
                            color="#dfe6e9",
                        )
                    )

        return {"nodes": nodes, "edges": edges}

    except Exception as e:
        logger.error(f"Failed to build document graph: {e}")
        return None


def export_graph_data(graph_data: Dict) -> str:
    """Export graph data as JSON."""
    try:
        # Convert nodes and edges to serializable format
        export_data = {
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "title": node.title,
                }
                for node in graph_data["nodes"]
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge.label,
                }
                for edge in graph_data["edges"]
            ],
        }
        return json.dumps(export_data, indent=2)
    except Exception as e:
        logger.error(f"Failed to export graph data: {e}")
        return "{}"


# Entity type colors for visualization
ENTITY_TYPE_COLORS = {
    "PERSON": "#e74c3c",      # Red
    "ORG": "#3498db",         # Blue
    "LOCATION": "#2ecc71",    # Green
    "DATE": "#f39c12",        # Orange
    "EVENT": "#9b59b6",       # Purple
    "CONCEPT": "#1abc9c",     # Teal
    "PRODUCT": "#e91e63",     # Pink
    "TECHNOLOGY": "#00bcd4",  # Cyan
    "DEFAULT": "#95a5a6",     # Gray
}


def get_full_knowledge_graph(
    max_entities: int = 100,
    max_relationships: int = 200,
    entity_types: Optional[List[str]] = None,
) -> Optional[Dict]:
    """
    Build the full knowledge graph with all entities and relationships.

    Args:
        max_entities: Maximum number of entities to include
        max_relationships: Maximum number of relationships to include
        entity_types: Optional filter for specific entity types

    Returns:
        Dictionary with nodes and edges for visualization
    """
    try:
        client = st.session_state.neo4j_client

        # Build type filter if specified
        type_filter = ""
        if entity_types and "All" not in entity_types:
            type_filter = "WHERE e.type IN $entity_types"

        # Query all entities
        entity_query = f"""
        MATCH (e:Entity)
        {type_filter}
        RETURN e.id AS id, e.name AS name, e.type AS type,
               e.mention_count AS mention_count,
               e.first_seen AS first_seen, e.last_seen AS last_seen
        ORDER BY e.mention_count DESC
        LIMIT $max_entities
        """

        params = {"max_entities": max_entities}
        if entity_types and "All" not in entity_types:
            params["entity_types"] = entity_types

        entities = client.execute_query(entity_query, params)

        if not entities:
            return None

        entity_ids = [e["id"] for e in entities]

        # Query relationships between these entities via co-occurrence in chunks
        relationship_query = """
        MATCH (e1:Entity)<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(e2:Entity)
        WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids
        AND e1.id < e2.id
        WITH e1, e2, count(c) AS co_occurrence_count
        WHERE co_occurrence_count >= 1
        RETURN e1.id AS source, e2.id AS target,
               co_occurrence_count,
               'CO_OCCURS' AS relationship_type
        ORDER BY co_occurrence_count DESC
        LIMIT $max_relationships
        """

        relationships = client.execute_query(
            relationship_query,
            {"entity_ids": entity_ids, "max_relationships": max_relationships}
        )

        # Also get explicit RELATES_TO relationships if they exist
        explicit_rel_query = """
        MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
        WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids
        RETURN e1.id AS source, e2.id AS target,
               1 AS co_occurrence_count,
               type(r) AS relationship_type
        LIMIT $max_relationships
        """

        explicit_rels = client.execute_query(
            explicit_rel_query,
            {"entity_ids": entity_ids, "max_relationships": max_relationships}
        )

        # Combine relationships
        all_relationships = relationships + explicit_rels

        # Build nodes
        nodes = []
        for entity in entities:
            entity_type = entity.get("type", "DEFAULT")
            color = ENTITY_TYPE_COLORS.get(entity_type, ENTITY_TYPE_COLORS["DEFAULT"])

            # Size based on mention count
            mention_count = entity.get("mention_count", 1) or 1
            size = min(10 + mention_count * 2, 40)

            # Build tooltip
            tooltip_parts = [
                f"<b>{entity['name']}</b>",
                f"Type: {entity_type}",
                f"Mentions: {mention_count}",
            ]
            if entity.get("first_seen"):
                tooltip_parts.append(f"First seen: {entity['first_seen']}")
            if entity.get("last_seen"):
                tooltip_parts.append(f"Last seen: {entity['last_seen']}")

            nodes.append(
                Node(
                    id=entity["id"],
                    label=entity["name"][:25] + ("..." if len(entity["name"]) > 25 else ""),
                    title="<br>".join(tooltip_parts),
                    color=color,
                    size=size,
                )
            )

        # Build edges
        edges = []
        seen_edges = set()

        for rel in all_relationships:
            edge_key = (rel["source"], rel["target"])
            if edge_key not in seen_edges:
                # Edge width based on co-occurrence count
                count = rel.get("co_occurrence_count", 1)
                width = min(1 + count * 0.5, 5)

                edges.append(
                    Edge(
                        source=rel["source"],
                        target=rel["target"],
                        label=rel.get("relationship_type", ""),
                        width=width,
                        title=f"Co-occurrences: {count}" if rel.get("relationship_type") == "CO_OCCURS" else "",
                    )
                )
                seen_edges.add(edge_key)

        return {"nodes": nodes, "edges": edges}

    except Exception as e:
        logger.error(f"Failed to build full knowledge graph: {e}")
        return None


def get_document_entity_graph(
    max_documents: int = 20,
    max_entities_per_doc: int = 10,
) -> Optional[Dict]:
    """
    Build a graph showing documents and their connected entities.

    Args:
        max_documents: Maximum number of documents to include
        max_entities_per_doc: Maximum entities per document

    Returns:
        Dictionary with nodes and edges for visualization
    """
    try:
        client = st.session_state.neo4j_client

        query = """
        MATCH (d:Document)
        WITH d
        ORDER BY d.created_at DESC
        LIMIT $max_documents

        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
        WHERE c.is_current = true

        WITH d, e, count(c) AS mention_count
        ORDER BY mention_count DESC

        WITH d, collect({entity: e, mentions: mention_count})[0..$max_entities_per_doc] AS top_entities

        RETURN d.id AS doc_id, d.title AS doc_title, d.created_at AS created_at,
               [te IN top_entities WHERE te.entity IS NOT NULL |
                {id: te.entity.id, name: te.entity.name, type: te.entity.type, mentions: te.mentions}
               ] AS entities
        """

        results = client.execute_query(
            query,
            {"max_documents": max_documents, "max_entities_per_doc": max_entities_per_doc}
        )

        if not results:
            return None

        nodes = []
        edges = []
        seen_entities = set()

        for doc in results:
            # Add document node
            doc_id = doc["doc_id"]
            nodes.append(
                Node(
                    id=doc_id,
                    label=doc["doc_title"][:30] + ("..." if len(doc["doc_title"]) > 30 else ""),
                    title=f"<b>Document:</b> {doc['doc_title']}",
                    color="#8e44ad",
                    size=30,
                    shape="box",
                )
            )

            # Add entity nodes and edges
            for entity_data in doc.get("entities", []):
                if entity_data:
                    entity_id = entity_data["id"]

                    # Add entity node if not already added
                    if entity_id not in seen_entities:
                        entity_type = entity_data.get("type", "DEFAULT")
                        color = ENTITY_TYPE_COLORS.get(entity_type, ENTITY_TYPE_COLORS["DEFAULT"])

                        nodes.append(
                            Node(
                                id=entity_id,
                                label=entity_data["name"][:20],
                                title=f"<b>{entity_data['name']}</b><br>Type: {entity_type}",
                                color=color,
                                size=15,
                            )
                        )
                        seen_entities.add(entity_id)

                    # Add edge from document to entity
                    edges.append(
                        Edge(
                            source=doc_id,
                            target=entity_id,
                            label="MENTIONS",
                            color="#bdc3c7",
                            title=f"Mentions: {entity_data.get('mentions', 1)}",
                        )
                    )

        return {"nodes": nodes, "edges": edges}

    except Exception as e:
        logger.error(f"Failed to build document-entity graph: {e}")
        return None


# Sidebar
with st.sidebar:
    st.title("🕸️ Graph Explorer")

    # Connection status
    st.subheader("Connection Status")
    if st.button("🔄 Connect to Neo4j"):
        with st.spinner("Connecting..."):
            if init_connections():
                st.success("✅ Connected to Neo4j")
            else:
                st.error("❌ Failed to connect to Neo4j")

    if st.session_state.neo4j_client:
        st.success("✅ Connected")

        # Database statistics
        st.subheader("📊 Database Statistics")
        stats = get_database_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats.get("documents", 0))
            st.metric("Chunks", stats.get("chunks", 0))
        with col2:
            st.metric("Entities", stats.get("entities", 0))
            st.metric("Relationships", stats.get("relationships", 0))

        # Temporal filter
        st.subheader("🕐 Temporal Filter")
        use_temporal_filter = st.checkbox("Enable temporal filtering")

        if use_temporal_filter:
            filter_type = st.radio(
                "Filter Type",
                ["Point in Time", "Date Range"],
            )

            if filter_type == "Point in Time":
                point_date = st.date_input(
                    "Select Date",
                    value=datetime.now(),
                )
            else:
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

    else:
        st.warning("Not connected to Neo4j")

# Main content
st.title("🕸️ Temporal Knowledge Graph Explorer")

if not st.session_state.neo4j_client:
    st.info("👈 Please connect to Neo4j using the sidebar")
    st.stop()

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 Full Graph",
    "📈 Entity Explorer",
    "📄 Document Explorer",
    "🔍 Custom Query"
])

# Tab 1: Full Knowledge Graph
with tab1:
    st.header("Full Knowledge Graph")
    st.markdown(
        "Visualize the complete knowledge graph with all entities and their relationships."
    )

    # Graph options
    col1, col2, col3 = st.columns(3)

    with col1:
        graph_mode = st.selectbox(
            "Graph Mode",
            ["Entity Network", "Document-Entity Graph"],
            help="Choose visualization mode"
        )

    with col2:
        if graph_mode == "Entity Network":
            max_entities = st.slider(
                "Max Entities",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Maximum number of entities to display"
            )
        else:
            max_documents = st.slider(
                "Max Documents",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                help="Maximum number of documents to display"
            )

    with col3:
        if graph_mode == "Entity Network":
            max_relationships = st.slider(
                "Max Relationships",
                min_value=50,
                max_value=1000,
                value=200,
                step=50,
                help="Maximum number of relationships to display"
            )
        else:
            max_entities_per_doc = st.slider(
                "Entities per Document",
                min_value=3,
                max_value=20,
                value=10,
                step=1,
                help="Maximum entities shown per document"
            )

    # Entity type filter (only for Entity Network mode)
    if graph_mode == "Entity Network":
        available_types = ["All"] + get_entity_types()
        selected_types = st.multiselect(
            "Filter by Entity Types",
            available_types,
            default=["All"],
            help="Select specific entity types to display"
        )

    # Visualization settings
    st.subheader("Visualization Settings")
    col1, col2, col3 = st.columns(3)

    with col1:
        physics_enabled = st.checkbox("Enable Physics", value=True, help="Enable physics simulation for node positioning")
    with col2:
        hierarchical = st.checkbox("Hierarchical Layout", value=False, help="Use hierarchical layout instead of force-directed")
    with col3:
        show_labels = st.checkbox("Show Edge Labels", value=False, help="Display relationship labels on edges")

    # Build and display graph
    if st.button("🔄 Load Full Graph", type="primary"):
        with st.spinner("Building knowledge graph... This may take a moment for large graphs."):
            if graph_mode == "Entity Network":
                # Handle "All" selection
                type_filter = None if "All" in selected_types else selected_types

                full_graph = get_full_knowledge_graph(
                    max_entities=max_entities,
                    max_relationships=max_relationships,
                    entity_types=type_filter,
                )
            else:
                full_graph = get_document_entity_graph(
                    max_documents=max_documents,
                    max_entities_per_doc=max_entities_per_doc,
                )

            if full_graph:
                st.session_state.full_graph_data = full_graph
                st.success(
                    f"Loaded {len(full_graph['nodes'])} nodes and "
                    f"{len(full_graph['edges'])} edges"
                )
            else:
                st.warning("No data found in the knowledge graph")

    # Display the graph
    if "full_graph_data" in st.session_state and st.session_state.full_graph_data:
        graph_data = st.session_state.full_graph_data

        # Show legend for entity types
        if graph_mode == "Entity Network":
            st.subheader("Entity Type Legend")
            legend_cols = st.columns(len(ENTITY_TYPE_COLORS) - 1)  # Exclude DEFAULT
            for i, (type_name, color) in enumerate([
                (k, v) for k, v in ENTITY_TYPE_COLORS.items() if k != "DEFAULT"
            ]):
                with legend_cols[i % len(legend_cols)]:
                    st.markdown(
                        f'<span style="color:{color}">●</span> {type_name}',
                        unsafe_allow_html=True
                    )

        # Configure the graph
        config = Config(
            width="100%",
            height=700,
            directed=graph_mode == "Document-Entity Graph",
            physics=physics_enabled,
            hierarchical=hierarchical,
            nodeHighlightBehavior=True,
            highlightColor="#f1c40f",
            collapsible=False,
            node={"labelProperty": "label"},
            link={"labelProperty": "label" if show_labels else ""},
        )

        # Render the graph
        selected_node = agraph(
            nodes=graph_data["nodes"],
            edges=graph_data["edges"],
            config=config,
        )

        # Show selected node info
        if selected_node:
            st.info(f"Selected: {selected_node}")

        # Export options
        col1, col2 = st.columns(2)
        with col1:
            export_data = export_graph_data(graph_data)
            st.download_button(
                "📥 Export as JSON",
                data=export_data,
                file_name="full_knowledge_graph.json",
                mime="application/json",
            )
        with col2:
            st.metric("Total Nodes", len(graph_data["nodes"]))
            st.metric("Total Edges", len(graph_data["edges"]))

# Tab 2: Entity Explorer
with tab2:
    st.header("Entity Explorer")

    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input(
            "Search entities",
            placeholder="Enter entity name...",
        )
    with col2:
        entity_types = ["All"] + get_entity_types()
        entity_type_filter = st.selectbox("Entity Type", entity_types)

    if search_term:
        entities = search_entities(search_term, entity_type_filter)

        if entities:
            st.subheader(f"Found {len(entities)} entities")

            for entity in entities:
                with st.expander(f"🏷️ {entity['name']} ({entity.get('type', 'Unknown')})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**ID:** {entity['id']}")
                        st.write(f"**Type:** {entity.get('type', 'Unknown')}")
                    with col2:
                        if entity.get("first_seen"):
                            st.write(f"**First Seen:** {entity['first_seen']}")
                        if entity.get("last_seen"):
                            st.write(f"**Last Seen:** {entity['last_seen']}")

                    if st.button(f"Visualize Graph", key=f"viz_{entity['id']}"):
                        with st.spinner("Building graph..."):
                            graph_data = build_entity_neighborhood_graph(entity["name"])
                            if graph_data:
                                st.session_state.graph_data = graph_data

            # Visualize graph if available
            if st.session_state.graph_data:
                st.subheader("Entity Neighborhood Graph")

                config = Config(
                    width=800,
                    height=600,
                    directed=False,
                    physics=True,
                    hierarchical=False,
                )

                agraph(
                    nodes=st.session_state.graph_data["nodes"],
                    edges=st.session_state.graph_data["edges"],
                    config=config,
                )

                # Export button
                if st.button("📥 Export Graph Data"):
                    export_data = export_graph_data(st.session_state.graph_data)
                    st.download_button(
                        "Download JSON",
                        data=export_data,
                        file_name="graph_data.json",
                        mime="application/json",
                    )
        else:
            st.info("No entities found")

# Tab 3: Document Explorer
with tab3:
    st.header("Document Explorer")

    documents = get_documents(limit=50)

    if documents:
        st.subheader(f"Total documents: {len(documents)}")

        for doc in documents:
            with st.expander(f"📄 {doc['title']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**ID:** {doc['id']}")
                    st.write(f"**Source:** {doc.get('source', 'N/A')}")
                    st.write(f"**Chunks:** {doc.get('chunk_count', 0)}")
                with col2:
                    if doc.get("created_at"):
                        st.write(f"**Created:** {doc['created_at']}")
                    if doc.get("updated_at"):
                        st.write(f"**Updated:** {doc['updated_at']}")

                col1, col2 = st.columns(2)
                with col1:
                    include_entities = st.checkbox(
                        "Include entities",
                        value=True,
                        key=f"entities_{doc['id']}",
                    )
                with col2:
                    if st.button("Visualize", key=f"doc_viz_{doc['id']}"):
                        with st.spinner("Building graph..."):
                            graph_data = build_document_graph(
                                doc["id"],
                                include_entities=include_entities,
                            )
                            if graph_data:
                                st.session_state.graph_data = graph_data

        # Visualize document graph if available
        if st.session_state.graph_data:
            st.subheader("Document Graph")

            config = Config(
                width=800,
                height=600,
                directed=True,
                physics=True,
                hierarchical=True,
            )

            agraph(
                nodes=st.session_state.graph_data["nodes"],
                edges=st.session_state.graph_data["edges"],
                config=config,
            )

            # Export button
            if st.button("📥 Export Document Graph"):
                export_data = export_graph_data(st.session_state.graph_data)
                st.download_button(
                    "Download JSON",
                    data=export_data,
                    file_name="document_graph.json",
                    mime="application/json",
                )
    else:
        st.info("No documents found in the database")

# Tab 4: Custom Query
with tab4:
    st.header("Custom Cypher Query")

    st.warning("⚠️ Use with caution. Only READ queries are recommended.")

    query = st.text_area(
        "Enter Cypher query",
        height=150,
        placeholder="MATCH (n) RETURN n LIMIT 10",
    )

    if st.button("Execute Query"):
        if query.strip():
            try:
                client = st.session_state.neo4j_client
                result = client.execute_query(query)

                st.success(f"✅ Query executed. Returned {len(result)} records")

                if result:
                    st.dataframe(result)
            except Exception as e:
                st.error(f"❌ Query failed: {e}")
        else:
            st.warning("Please enter a query")

# Footer
st.markdown("---")
st.markdown(
    "**Temporal Knowledge Graph Explorer** | "
    "Built with Streamlit • Neo4j • LangChain"
)
