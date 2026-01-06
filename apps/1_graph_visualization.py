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
tab1, tab2, tab3 = st.tabs(["📈 Entity Explorer", "📄 Document Explorer", "🔍 Custom Query"])

# Tab 1: Entity Explorer
with tab1:
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

# Tab 2: Document Explorer
with tab2:
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

# Tab 3: Custom Query
with tab3:
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
