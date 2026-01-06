"""Streamlit app for RAG chatbot with temporal knowledge graph."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temporal_kg_rag.rag.chain import get_rag_chain
from temporal_kg_rag.rag.graph import get_rag_graph
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Temporal Knowledge Graph Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .source-card {
        background-color: #fff;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        font-size: 0.9em;
    }
    .metadata-badge {
        background-color: #e0e0e0;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        margin: 2px;
        display: inline-block;
    }
    .temporal-indicator {
        background-color: #fff3e0;
        border-left: 3px solid #ff9800;
        padding: 8px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 0.9em;
    }
    .entity-tag {
        background-color: #e1f5fe;
        color: #01579b;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.85em;
        margin: 2px;
        display: inline-block;
    }
    .warning-box {
        background-color: #fff9c4;
        border-left: 3px solid #fbc02d;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_implementation" not in st.session_state:
    st.session_state.rag_implementation = "LangGraph"
if "use_streaming" not in st.session_state:
    st.session_state.use_streaming = False
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


def format_sources(sources: List[Dict]) -> str:
    """Format sources for display."""
    if not sources:
        return "No sources available"

    source_html = ""
    for i, source in enumerate(sources[:5], 1):
        doc_title = source.get("document_title", "Unknown")
        created_at = source.get("created_at", "Unknown date")
        score = source.get("score", 0)
        chunk_id = source.get("chunk_id", "")

        source_html += f"""
        <div class="source-card">
            <strong>#{i} {doc_title}</strong><br>
            <small>Date: {created_at} | Score: {score:.4f}</small><br>
            <small style="color: #666;">Chunk ID: {chunk_id}</small>
        </div>
        """

    if len(sources) > 5:
        source_html += f"<p><em>... and {len(sources) - 5} more sources</em></p>"

    return source_html


def format_metadata(metadata: Dict) -> str:
    """Format metadata for display."""
    badges = []

    if "query_type" in metadata:
        badges.append(f'<span class="metadata-badge">Type: {metadata["query_type"]}</span>')

    if "temporal_detected" in metadata:
        temporal = "Yes" if metadata["temporal_detected"] else "No"
        badges.append(f'<span class="metadata-badge">Temporal: {temporal}</span>')

    if "num_results" in metadata:
        badges.append(f'<span class="metadata-badge">Results: {metadata["num_results"]}</span>')

    if "verified" in metadata:
        verified = "✓" if metadata["verified"] else "✗"
        badges.append(f'<span class="metadata-badge">Verified: {verified}</span>')

    return " ".join(badges)


def format_entities(entities: List[str]) -> str:
    """Format entities as tags."""
    if not entities:
        return ""

    entity_tags = [f'<span class="entity-tag">{e}</span>' for e in entities[:10]]
    html = "".join(entity_tags)

    if len(entities) > 10:
        html += f" <em>+{len(entities) - 10} more</em>"

    return html


def query_rag(
    question: str,
    implementation: str,
    use_history: bool = True,
    streaming: bool = False,
) -> Dict:
    """Query the RAG system."""
    try:
        if implementation == "LangChain":
            chain = get_rag_chain()

            if use_history and st.session_state.conversation_history:
                result = chain.query_with_history(
                    question,
                    st.session_state.conversation_history,
                    top_k=5,
                )
            else:
                result = chain.query(
                    question,
                    top_k=5,
                    use_temporal_detection=True,
                    expand_context=True,
                )

        else:  # LangGraph
            graph = get_rag_graph()
            result = graph.query(question, top_k=10)

        return result

    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "metadata": {"error": str(e)},
        }


def stream_rag_response(question: str):
    """Stream RAG response."""
    try:
        chain = get_rag_chain()

        response_placeholder = st.empty()
        full_response = ""

        for chunk in chain.stream_query(question, top_k=5):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

        return {
            "answer": full_response,
            "sources": [],
            "metadata": {},
        }

    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "metadata": {"error": str(e)},
        }


# Sidebar
with st.sidebar:
    st.title("💬 Chatbot Settings")

    # RAG implementation
    st.subheader("🤖 RAG Implementation")
    rag_impl = st.radio(
        "Select implementation",
        ["LangGraph", "LangChain"],
        help="LangGraph uses multi-node workflow. LangChain is simpler.",
    )
    st.session_state.rag_implementation = rag_impl

    # Streaming (only for LangChain)
    if rag_impl == "LangChain":
        st.session_state.use_streaming = st.checkbox(
            "Enable streaming",
            help="Stream the response token by token",
        )

    # Conversation settings
    st.subheader("💭 Conversation")
    use_history = st.checkbox(
        "Use conversation history",
        value=True,
        help="Include previous messages for context",
    )

    max_history = st.slider(
        "Max history turns",
        min_value=0,
        max_value=10,
        value=3,
        help="Number of previous turns to include",
    )

    # Display mode
    st.subheader("📊 Display Options")
    show_sources = st.checkbox("Show sources", value=True)
    show_metadata = st.checkbox("Show metadata", value=True)
    show_entities = st.checkbox("Show detected entities", value=True)

    # Actions
    st.subheader("🔧 Actions")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()

    if st.button("📥 Export Conversation", use_container_width=True):
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "implementation": st.session_state.rag_implementation,
            "messages": st.session_state.messages,
        }

        st.download_button(
            "Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    # System status
    st.subheader("📡 System Status")
    try:
        if rag_impl == "LangGraph":
            graph = get_rag_graph()
            st.success("✅ LangGraph Ready")
        else:
            chain = get_rag_chain()
            st.success("✅ LangChain Ready")
    except Exception as e:
        st.error(f"❌ System Error: {str(e)}")

    # Statistics
    if st.session_state.messages:
        st.subheader("📈 Statistics")
        user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
        st.metric("Total Messages", len(st.session_state.messages))
        st.metric("User Messages", user_msgs)
        st.metric("Bot Responses", len(st.session_state.messages) - user_msgs)

# Main content
st.title("💬 Temporal Knowledge Graph Chatbot")
st.markdown(
    "Ask questions about your documents. The chatbot uses temporal awareness "
    "to provide accurate, time-contextualized answers with source citations."
)

# Example queries
with st.expander("💡 Example Questions"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **General Questions:**
        - What is artificial intelligence?
        - Explain quantum computing
        - How does machine learning work?

        **Temporal Questions:**
        - What were the main AI developments in 2023?
        - How has climate policy evolved?
        - Recent advances in renewable energy
        """
        )

    with col2:
        st.markdown(
            """
        **Comparison Questions:**
        - Compare GPT-3 and GPT-4
        - Difference between solar and wind energy
        - Machine learning vs deep learning

        **Follow-up Questions:**
        - Tell me more about that
        - When was this developed?
        - What are the challenges?
        """
        )

    # Quick query buttons
    st.markdown("**Quick Queries:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("What is AI?"):
            st.session_state.quick_query = "What is artificial intelligence?"
    with col2:
        if st.button("AI in 2023"):
            st.session_state.quick_query = "What were the main AI developments in 2023?"
    with col3:
        if st.button("Compare ML & DL"):
            st.session_state.quick_query = "Compare machine learning and deep learning"

st.markdown("---")

# Chat interface
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)

        else:  # assistant
            with st.chat_message("assistant"):
                st.markdown(content)

                # Display sources
                if show_sources and "sources" in message and message["sources"]:
                    with st.expander("📚 Sources"):
                        st.markdown(
                            format_sources(message["sources"]),
                            unsafe_allow_html=True,
                        )

                # Display metadata
                if show_metadata and "metadata" in message:
                    metadata = message["metadata"]

                    # Verification warnings
                    if not metadata.get("verified", True):
                        st.markdown(
                            '<div class="warning-box">⚠️ '
                            + metadata.get(
                                "verification_notes",
                                "Answer may need verification",
                            )
                            + "</div>",
                            unsafe_allow_html=True,
                        )

                    # Temporal context
                    if metadata.get("temporal_detected"):
                        st.markdown(
                            '<div class="temporal-indicator">🕐 Temporal query detected</div>',
                            unsafe_allow_html=True,
                        )

                    # Metadata badges
                    with st.expander("ℹ️ Metadata"):
                        st.markdown(
                            format_metadata(metadata),
                            unsafe_allow_html=True,
                        )

                # Display entities
                if show_entities and "entities" in message and message["entities"]:
                    with st.expander("🏷️ Detected Entities"):
                        st.markdown(
                            format_entities(message["entities"]),
                            unsafe_allow_html=True,
                        )

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Update conversation history for LangChain
    st.session_state.conversation_history.append(
        {"role": "user", "content": prompt}
    )

    # Keep only last N turns
    if len(st.session_state.conversation_history) > max_history * 2:
        st.session_state.conversation_history = st.session_state.conversation_history[
            -max_history * 2 :
        ]

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.time()

            # Query RAG system
            if st.session_state.use_streaming and rag_impl == "LangChain":
                result = stream_rag_response(prompt)
            else:
                result = query_rag(
                    prompt,
                    rag_impl,
                    use_history=use_history,
                )

            elapsed = time.time() - start_time

        # Display response
        answer = result.get("answer", "No answer generated")
        st.markdown(answer)

        # Display sources
        if show_sources and result.get("sources"):
            with st.expander("📚 Sources"):
                st.markdown(
                    format_sources(result["sources"]),
                    unsafe_allow_html=True,
                )

        # Display metadata
        if show_metadata and result.get("metadata"):
            metadata = result["metadata"]

            # Verification warnings
            if not metadata.get("verified", True):
                st.markdown(
                    '<div class="warning-box">⚠️ '
                    + metadata.get(
                        "verification_notes",
                        "Answer may need verification",
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )

            # Temporal context
            if metadata.get("temporal_detected"):
                st.markdown(
                    '<div class="temporal-indicator">🕐 Temporal query detected</div>',
                    unsafe_allow_html=True,
                )

            # Metadata badges
            with st.expander("ℹ️ Metadata"):
                badges_html = format_metadata(metadata)
                st.markdown(badges_html, unsafe_allow_html=True)
                st.caption(f"Response time: {elapsed:.2f}s")

        # Display entities
        if show_entities and metadata.get("entities_detected"):
            entities = metadata["entities_detected"]
            if entities:
                with st.expander("🏷️ Detected Entities"):
                    st.markdown(
                        format_entities(entities),
                        unsafe_allow_html=True,
                    )

    # Add assistant message to history
    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": result.get("sources", []),
        "metadata": result.get("metadata", {}),
        "entities": result.get("metadata", {}).get("entities_detected", []),
    }

    st.session_state.messages.append(assistant_message)

    # Update conversation history
    st.session_state.conversation_history.append(
        {"role": "assistant", "content": answer}
    )

    st.rerun()

# Handle quick queries
if "quick_query" in st.session_state:
    prompt = st.session_state.quick_query
    del st.session_state.quick_query

    # Add to messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_history.append({"role": "user", "content": prompt})

    # Generate response
    result = query_rag(prompt, rag_impl, use_history=use_history)
    answer = result.get("answer", "No answer generated")

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": result.get("sources", []),
        "metadata": result.get("metadata", {}),
        "entities": result.get("metadata", {}).get("entities_detected", []),
    }

    st.session_state.messages.append(assistant_message)
    st.session_state.conversation_history.append({"role": "assistant", "content": answer})

    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    f"**Temporal Knowledge Graph Chatbot** | "
    f"Implementation: {st.session_state.rag_implementation} | "
    f"Powered by LangChain • LangGraph • Neo4j"
)
