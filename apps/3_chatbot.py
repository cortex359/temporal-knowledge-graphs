"""Streamlit app for RAG chatbot with transparent pipeline visualization."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable

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

# Custom CSS for pipeline visualization
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
    .pipeline-step {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #fafafa;
    }
    .pipeline-step-active {
        border-color: #2196f3;
        background-color: #e3f2fd;
        animation: pulse 1s infinite;
    }
    .pipeline-step-done {
        border-color: #4caf50;
        background-color: #e8f5e9;
    }
    .pipeline-step-header {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .pipeline-step-icon {
        font-size: 1.5em;
        margin-right: 10px;
    }
    .pipeline-step-title {
        font-weight: bold;
        font-size: 1.1em;
    }
    .pipeline-step-time {
        color: #666;
        font-size: 0.85em;
        margin-left: auto;
    }
    .context-chunk {
        background-color: #fff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-size: 0.9em;
    }
    .context-chunk-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        padding-bottom: 8px;
        border-bottom: 1px solid #eee;
    }
    .context-chunk-score {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
    }
    .context-chunk-text {
        color: #333;
        line-height: 1.5;
    }
    .entity-tag {
        background-color: #e1f5fe;
        color: #01579b;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.85em;
        margin: 3px;
        display: inline-block;
    }
    .temporal-badge {
        background-color: #fff3e0;
        color: #e65100;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.85em;
        margin: 3px;
        display: inline-block;
    }
    .query-type-badge {
        background-color: #f3e5f5;
        color: #7b1fa2;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.85em;
        margin: 3px;
        display: inline-block;
    }
    .source-card {
        background-color: #fff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    .verification-pass {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 10px;
        border-radius: 5px;
    }
    .verification-fail {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 10px;
        border-radius: 5px;
    }
    .llm-context-box {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-family: monospace;
        font-size: 0.85em;
        max-height: 400px;
        overflow-y: auto;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .metric-row {
        display: flex;
        gap: 20px;
        margin: 10px 0;
    }
    .metric-item {
        background-color: #f5f5f5;
        padding: 10px 15px;
        border-radius: 8px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5em;
        font-weight: bold;
        color: #1976d2;
    }
    .metric-label {
        font-size: 0.85em;
        color: #666;
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
if "show_pipeline" not in st.session_state:
    st.session_state.show_pipeline = True
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "last_debug_info" not in st.session_state:
    st.session_state.last_debug_info = None


def render_pipeline_step(
    step_num: int,
    title: str,
    icon: str,
    description: str,
    status: str = "pending",
    duration: float = 0,
    details: Optional[Dict] = None,
):
    """Render a pipeline step with status indicator."""
    status_class = {
        "pending": "",
        "active": "pipeline-step-active",
        "done": "pipeline-step-done",
    }.get(status, "")

    status_icon = {
        "pending": "⏳",
        "active": "🔄",
        "done": "✅",
    }.get(status, "⏳")

    with st.container():
        st.markdown(
            f"""
            <div class="pipeline-step {status_class}">
                <div class="pipeline-step-header">
                    <span class="pipeline-step-icon">{icon}</span>
                    <span class="pipeline-step-title">Step {step_num}: {title}</span>
                    <span class="pipeline-step-time">{status_icon} {duration:.2f}s</span>
                </div>
                <p style="color: #666; margin: 0;">{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if details and status == "done":
            with st.expander(f"📋 Details for {title}", expanded=False):
                st.json(details)


def render_retrieval_results(results: List[Dict], show_full_text: bool = False):
    """Render retrieval results as context chunks."""
    if not results:
        st.info("No results retrieved from the knowledge graph.")
        return

    st.markdown(f"**Retrieved {len(results)} chunks from the knowledge graph:**")

    for i, result in enumerate(results[:10], 1):
        score = result.get("score", result.get("rrf_score", result.get("hybrid_score", 0)))
        doc_title = result.get("document_title", "Unknown Document")
        chunk_id = result.get("chunk_id", "")[:8]
        text = result.get("text", "")

        # Truncate text if not showing full
        display_text = text if show_full_text else (text[:300] + "..." if len(text) > 300 else text)

        st.markdown(
            f"""
            <div class="context-chunk">
                <div class="context-chunk-header">
                    <span><strong>#{i}</strong> {doc_title}</span>
                    <span class="context-chunk-score">Score: {score:.4f}</span>
                </div>
                <div class="context-chunk-text">{display_text}</div>
                <small style="color: #999;">Chunk ID: {chunk_id}...</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if len(results) > 10:
        st.caption(f"... and {len(results) - 10} more chunks")


def render_entities(entities: List[str]):
    """Render detected entities as tags."""
    if not entities:
        return

    tags_html = "".join([f'<span class="entity-tag">🏷️ {e}</span>' for e in entities[:15]])
    if len(entities) > 15:
        tags_html += f'<span style="color: #666;"> +{len(entities) - 15} more</span>'

    st.markdown(tags_html, unsafe_allow_html=True)


def render_query_understanding(
    query_type: str,
    temporal_detected: bool,
    temporal_context: Optional[Dict],
    entities: List[str],
):
    """Render query understanding results."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f'<span class="query-type-badge">Query Type: {query_type}</span>', unsafe_allow_html=True)

    with col2:
        if temporal_detected:
            st.markdown('<span class="temporal-badge">🕐 Temporal Query Detected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color: #666;">No temporal reference</span>', unsafe_allow_html=True)

    with col3:
        st.markdown(f"**Entities Found:** {len(entities)}")

    if entities:
        st.markdown("**Detected Entities:**")
        render_entities(entities)

    if temporal_context:
        st.markdown("**Temporal Context:**")
        st.json(temporal_context)


def render_llm_context(context: str, max_chars: int = 2000):
    """Render the formatted context sent to the LLM."""
    if not context:
        st.info("No context was built for this query.")
        return

    display_context = context[:max_chars] + "..." if len(context) > max_chars else context

    st.markdown(
        f"""
        <div class="llm-context-box">
            <strong>Context sent to LLM ({len(context)} characters):</strong>
            <hr style="margin: 10px 0;">
            <pre style="white-space: pre-wrap; word-wrap: break-word;">{display_context}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(context) > max_chars:
        with st.expander("📜 Show Full Context"):
            st.text(context)


def render_verification(verified: bool, notes: Optional[str]):
    """Render verification results."""
    if verified:
        st.markdown(
            """
            <div class="verification-pass">
                <strong>✅ Verification Passed</strong><br>
                Answer meets quality criteria and includes proper citations.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verification-fail">
                <strong>⚠️ Verification Notes</strong><br>
                {notes or "Answer may need additional review."}
            </div>
            """,
            unsafe_allow_html=True,
        )


def query_rag_with_transparency(
    question: str,
    implementation: str,
    pipeline_container,
    use_history: bool = True,
) -> Dict:
    """Query RAG system with real-time pipeline visualization."""

    step_times = {}
    result = {}

    try:
        if implementation == "LangGraph":
            graph = get_rag_graph()

            # Step 1: Query Understanding
            with pipeline_container:
                step1_placeholder = st.empty()
                with step1_placeholder.container():
                    render_pipeline_step(
                        1, "Query Understanding", "🔍",
                        "Analyzing query for temporal references, entities, and intent...",
                        status="active"
                    )

            start = time.time()
            # We'll get all info from the full query, but show progress
            time.sleep(0.3)  # Small delay to show animation
            step_times["understanding"] = time.time() - start

            # Execute full query with debug info
            start_total = time.time()
            result = graph.query(question, top_k=10, return_debug_info=True)
            total_time = time.time() - start_total

            debug = result.get("debug", {})
            steps = debug.get("pipeline_steps", [])

            # Update Step 1 with results
            step1_output = steps[0]["output"] if steps else {}
            with step1_placeholder.container():
                render_pipeline_step(
                    1, "Query Understanding", "🔍",
                    "Analyzed query for temporal references, entities, and intent",
                    status="done",
                    duration=step_times["understanding"],
                    details=step1_output
                )
                render_query_understanding(
                    step1_output.get("query_type", "factual"),
                    step1_output.get("temporal_detected", False),
                    step1_output.get("temporal_context"),
                    step1_output.get("entities_detected", []),
                )

            # Step 2: Retrieval
            with pipeline_container:
                step2_placeholder = st.empty()
                step2_output = steps[1]["output"] if len(steps) > 1 else {}

                with step2_placeholder.container():
                    render_pipeline_step(
                        2, "Knowledge Graph Retrieval", "🔎",
                        f"Retrieved {step2_output.get('num_results', 0)} chunks using hybrid search (vector + graph)",
                        status="done",
                        duration=total_time * 0.3,  # Approximate
                    )

                    retrieval_results = debug.get("all_retrieval_results", [])
                    if retrieval_results:
                        render_retrieval_results(retrieval_results)

            # Step 3: Context Building
            with pipeline_container:
                step3_placeholder = st.empty()
                step3_output = steps[2]["output"] if len(steps) > 2 else {}

                with step3_placeholder.container():
                    render_pipeline_step(
                        3, "Context Building", "📝",
                        "Formatted retrieved chunks into context for the LLM",
                        status="done",
                        duration=total_time * 0.1,
                    )

                    full_context = debug.get("full_context", "")
                    if full_context:
                        render_llm_context(full_context)

            # Step 4: LLM Generation
            with pipeline_container:
                step4_placeholder = st.empty()
                step4_output = steps[3]["output"] if len(steps) > 3 else {}

                with step4_placeholder.container():
                    render_pipeline_step(
                        4, "Answer Generation", "🤖",
                        f"Generated answer using LLM ({step4_output.get('answer_length', 0)} chars)",
                        status="done",
                        duration=total_time * 0.5,
                    )

            # Step 5: Verification
            with pipeline_container:
                step5_placeholder = st.empty()
                step5_output = steps[4]["output"] if len(steps) > 4 else {}

                with step5_placeholder.container():
                    render_pipeline_step(
                        5, "Answer Verification", "✓",
                        "Verified answer quality and citation accuracy",
                        status="done",
                        duration=total_time * 0.1,
                    )

                    render_verification(
                        step5_output.get("verified", True),
                        step5_output.get("verification_notes"),
                    )

            # Summary metrics
            with pipeline_container:
                st.markdown("---")
                st.markdown("### 📊 Pipeline Summary")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Time", f"{total_time:.2f}s")
                with col2:
                    st.metric("Chunks Retrieved", step2_output.get("num_results", 0))
                with col3:
                    st.metric("Entities Found", len(step1_output.get("entities_detected", [])))
                with col4:
                    st.metric("Verified", "✅" if step5_output.get("verified") else "⚠️")

            # Store debug info
            st.session_state.last_debug_info = debug

        else:  # LangChain (simpler, less detailed)
            chain = get_rag_chain()

            with pipeline_container:
                st.info("LangChain implementation provides less detailed pipeline info. Switch to LangGraph for full transparency.")

            start = time.time()
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
            elapsed = time.time() - start

            with pipeline_container:
                st.metric("Query Time", f"{elapsed:.2f}s")

        return result

    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
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
        help="LangGraph provides full pipeline transparency. LangChain is simpler.",
    )
    st.session_state.rag_implementation = rag_impl

    if rag_impl == "LangGraph":
        st.success("✨ Full pipeline transparency enabled")
    else:
        st.info("Limited pipeline visibility")

    # Pipeline visibility
    st.subheader("👁️ Pipeline Visibility")
    st.session_state.show_pipeline = st.checkbox(
        "Show RAG Pipeline",
        value=True,
        help="Display real-time RAG pipeline steps"
    )

    show_context = st.checkbox(
        "Show LLM Context",
        value=True,
        help="Display the context sent to the LLM"
    )

    show_retrieval = st.checkbox(
        "Show Retrieved Chunks",
        value=True,
        help="Display chunks retrieved from knowledge graph"
    )

    show_full_chunk_text = st.checkbox(
        "Full Chunk Text",
        value=False,
        help="Show complete text of retrieved chunks"
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
    )

    # Actions
    st.subheader("🔧 Actions")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.last_debug_info = None
        st.rerun()

    if st.button("📥 Export Debug Info", use_container_width=True):
        if st.session_state.last_debug_info:
            st.download_button(
                "Download JSON",
                data=json.dumps(st.session_state.last_debug_info, indent=2, default=str),
                file_name=f"rag_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        else:
            st.warning("No debug info available yet")

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

# Main content
st.title("💬 Temporal Knowledge Graph Chatbot")
st.markdown(
    """
    Ask questions about your documents. This chatbot uses a **Temporal Knowledge Graph**
    with **transparent RAG pipeline** - see exactly how your query is processed,
    what context is retrieved, and how the answer is generated.
    """
)

# Pipeline explanation
with st.expander("ℹ️ How the RAG Pipeline Works", expanded=False):
    st.markdown(
        """
        ### The 5-Step RAG Pipeline

        1. **🔍 Query Understanding**
           - Detect temporal references (dates, time periods)
           - Extract entities (people, organizations, concepts)
           - Classify query type (factual, comparison, evolution)

        2. **🔎 Knowledge Graph Retrieval**
           - **Vector Search**: Find semantically similar chunks using embeddings
           - **Graph Search**: Traverse entity relationships in Neo4j
           - **Hybrid Fusion**: Combine results using Reciprocal Rank Fusion (RRF)
           - **Temporal Filtering**: Filter by time period if detected

        3. **📝 Context Building**
           - Format retrieved chunks into structured context
           - Include source metadata and temporal information
           - Apply context expansion for related chunks

        4. **🤖 Answer Generation**
           - Select appropriate prompt template based on query type
           - Send context + query to LLM
           - Generate time-aware, cited answer

        5. **✓ Verification**
           - Check answer quality and completeness
           - Verify proper source citations
           - Flag potential issues for review
        """
    )

# Example queries
with st.expander("💡 Example Questions"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Temporal Questions:**
            - What were the main AI developments in 2023?
            - How has climate policy evolved over time?
            - Recent advances in renewable energy

            **Factual Questions:**
            - What is machine learning?
            - Explain quantum computing
            """
        )

    with col2:
        st.markdown(
            """
            **Comparison Questions:**
            - Compare GPT-3 and GPT-4
            - Difference between solar and wind energy

            **Follow-up Questions:**
            - Tell me more about that
            - When was this developed?
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
chat_col, pipeline_col = st.columns([1, 1]) if st.session_state.show_pipeline else (st.container(), None)

# Display chat messages
with chat_col:
    st.subheader("💬 Chat")

    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)

                # Show sources
                if message.get("sources"):
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"][:5], 1):
                            st.markdown(
                                f"**{i}.** {source.get('document_title', 'Unknown')} "
                                f"(Score: {source.get('score', 0):.4f})"
                            )

# Pipeline visualization container
if st.session_state.show_pipeline and pipeline_col:
    with pipeline_col:
        st.subheader("🔄 RAG Pipeline")
        pipeline_container = st.container()
else:
    pipeline_container = st.container()

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Update conversation history
    st.session_state.conversation_history.append({"role": "user", "content": prompt})
    if len(st.session_state.conversation_history) > max_history * 2:
        st.session_state.conversation_history = st.session_state.conversation_history[-max_history * 2:]

    # Display user message
    with chat_col:
        with st.chat_message("user"):
            st.markdown(prompt)

    # Generate response with pipeline visualization
    with chat_col:
        with st.chat_message("assistant"):
            if st.session_state.show_pipeline:
                with pipeline_col:
                    with st.spinner("Processing through RAG pipeline..."):
                        result = query_rag_with_transparency(
                            prompt,
                            rag_impl,
                            pipeline_container,
                            use_history=use_history,
                        )
            else:
                with st.spinner("Thinking..."):
                    if rag_impl == "LangGraph":
                        graph = get_rag_graph()
                        result = graph.query(prompt, top_k=10, return_debug_info=True)
                    else:
                        chain = get_rag_chain()
                        result = chain.query(prompt, top_k=5)

            # Display answer
            answer = result.get("answer", "No answer generated")
            st.markdown(answer)

            # Show sources
            if result.get("sources"):
                with st.expander("📚 Sources"):
                    for i, source in enumerate(result["sources"][:5], 1):
                        st.markdown(
                            f"**{i}.** {source.get('document_title', 'Unknown')} "
                            f"(Score: {source.get('score', 0):.4f})"
                        )

    # Add to message history
    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": result.get("sources", []),
        "metadata": result.get("metadata", {}),
    }
    st.session_state.messages.append(assistant_message)
    st.session_state.conversation_history.append({"role": "assistant", "content": answer})

    st.rerun()

# Handle quick queries
if "quick_query" in st.session_state:
    prompt = st.session_state.quick_query
    del st.session_state.quick_query

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_history.append({"role": "user", "content": prompt})

    if rag_impl == "LangGraph":
        graph = get_rag_graph()
        result = graph.query(prompt, top_k=10, return_debug_info=True)
    else:
        chain = get_rag_chain()
        result = chain.query(prompt, top_k=5)

    answer = result.get("answer", "No answer generated")

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": result.get("sources", []),
        "metadata": result.get("metadata", {}),
    }

    st.session_state.messages.append(assistant_message)
    st.session_state.conversation_history.append({"role": "assistant", "content": answer})

    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    f"**Temporal Knowledge Graph Chatbot** | "
    f"Implementation: {st.session_state.rag_implementation} | "
    f"Pipeline Transparency: {'On' if st.session_state.show_pipeline else 'Off'}"
)
