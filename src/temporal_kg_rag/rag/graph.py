"""LangGraph workflow for RAG with multiple processing nodes."""

from typing import Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from temporal_kg_rag.config.settings import get_settings
from temporal_kg_rag.rag.context_builder import ContextBuilder, get_context_builder
from temporal_kg_rag.rag.prompts import PromptTemplates
from temporal_kg_rag.retrieval.hybrid_search import HybridSearch, get_hybrid_search
from temporal_kg_rag.retrieval.temporal_retrieval import TemporalRetrieval, get_temporal_retrieval
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


# Define state structure for the graph
class RAGState(TypedDict):
    """State passed between nodes in the RAG workflow."""

    # Input
    query: str
    top_k: Optional[int]

    # Query understanding
    temporal_detected: bool
    temporal_context: Optional[Dict]
    entities_detected: List[str]
    query_type: str

    # Retrieval
    retrieval_results: List[Dict]
    num_results: int

    # Context building
    formatted_context: str
    context_metadata: Dict

    # Generation
    answer: str
    sources: List[Dict]

    # Verification
    verified: bool
    verification_notes: Optional[str]

    # Metadata
    error: Optional[str]


class RAGGraph:
    """LangGraph-based RAG workflow with multiple nodes."""

    def __init__(
        self,
        hybrid_search: Optional[HybridSearch] = None,
        temporal_retrieval: Optional[TemporalRetrieval] = None,
        context_builder: Optional[ContextBuilder] = None,
        llm: Optional[ChatOpenAI] = None,
    ):
        """
        Initialize RAG graph.

        Args:
            hybrid_search: Optional hybrid search instance
            temporal_retrieval: Optional temporal retrieval instance
            context_builder: Optional context builder instance
            llm: Optional LLM instance
        """
        self.hybrid_search = hybrid_search or get_hybrid_search()
        self.temporal_retrieval = temporal_retrieval or get_temporal_retrieval()
        self.context_builder = context_builder or get_context_builder()
        self.prompt_templates = PromptTemplates()

        # Initialize LLM
        settings = get_settings()
        if llm is None:
            self.llm = ChatOpenAI(
                base_url=settings.litellm_api_base,
                api_key=settings.litellm_api_key,
                model=settings.default_llm_model,
                temperature=0.7,
            )
        else:
            self.llm = llm

        # Build the workflow
        self.workflow = self._build_workflow()

        logger.info("RAG graph initialized")

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("understand_query", self._understand_query_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("build_context", self._build_context_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("verify", self._verify_node)

        # Define edges
        workflow.set_entry_point("understand_query")
        workflow.add_edge("understand_query", "retrieve")
        workflow.add_edge("retrieve", "build_context")
        workflow.add_edge("build_context", "generate")
        workflow.add_edge("generate", "verify")
        workflow.add_edge("verify", END)

        return workflow.compile()

    def _understand_query_node(self, state: RAGState) -> RAGState:
        """
        Node 1: Understand the query and extract temporal/entity context.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        logger.info("Node: Understanding query")

        query = state["query"]

        # Parse temporal context
        temporal_context_obj = self.temporal_retrieval.parse_temporal_context(query)

        state["temporal_detected"] = temporal_context_obj.has_temporal_reference
        state["temporal_context"] = (
            temporal_context_obj.dict() if temporal_context_obj.has_temporal_reference else None
        )

        # Extract entities (simple implementation)
        from temporal_kg_rag.retrieval.graph_search import get_graph_search

        gs = get_graph_search()
        entities = gs.extract_entities_from_query(query)
        state["entities_detected"] = entities

        # Classify query type (simple heuristic)
        query_lower = query.lower()
        if any(word in query_lower for word in ["compare", "difference", "versus"]):
            state["query_type"] = "comparison"
        elif any(word in query_lower for word in ["history", "evolution", "changed"]):
            state["query_type"] = "evolution"
        elif any(word in query_lower for word in ["what", "explain", "describe"]):
            state["query_type"] = "exploratory"
        else:
            state["query_type"] = "factual"

        logger.info(
            f"Query understanding: type={state['query_type']}, "
            f"temporal={state['temporal_detected']}, "
            f"entities={len(state['entities_detected'])}"
        )

        return state

    def _retrieve_node(self, state: RAGState) -> RAGState:
        """
        Node 2: Retrieve relevant information.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        logger.info("Node: Retrieving information")

        query = state["query"]
        top_k = state.get("top_k", 10)

        # Use temporal retrieval if temporal context detected
        if state["temporal_detected"] and state["temporal_context"]:
            from temporal_kg_rag.models.temporal import TemporalFilter

            # Recreate temporal filter from dict
            temporal_filter_dict = state["temporal_context"].get("temporal_filter")
            if temporal_filter_dict:
                temporal_filter = TemporalFilter(**temporal_filter_dict)
            else:
                temporal_filter = None

            results = self.hybrid_search.search(
                query=query,
                top_k=top_k,
                temporal_filter=temporal_filter,
            )
        else:
            results = self.hybrid_search.search(query=query, top_k=top_k)

        state["retrieval_results"] = results
        state["num_results"] = len(results)

        logger.info(f"Retrieved {len(results)} results")

        return state

    def _build_context_node(self, state: RAGState) -> RAGState:
        """
        Node 3: Build context from retrieval results.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        logger.info("Node: Building context")

        if not state["retrieval_results"]:
            state["formatted_context"] = "No relevant information found."
            state["context_metadata"] = {"num_sources": 0}
            return state

        # Build context
        from temporal_kg_rag.models.temporal import TemporalContext

        temporal_context = None
        if state["temporal_context"]:
            temporal_context = TemporalContext(**state["temporal_context"])

        context_dict = self.context_builder.build_context(
            query=state["query"],
            results=state["retrieval_results"],
            temporal_context=temporal_context,
            expand_context=True,
        )

        state["formatted_context"] = context_dict["formatted_context"]
        state["context_metadata"] = context_dict["metadata"]

        logger.info(f"Context built: {len(state['formatted_context'])} characters")

        return state

    def _generate_node(self, state: RAGState) -> RAGState:
        """
        Node 4: Generate answer using LLM.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        logger.info("Node: Generating answer")

        if state["formatted_context"] == "No relevant information found.":
            state["answer"] = "I couldn't find any relevant information to answer your question."
            state["sources"] = []
            return state

        # Select prompt template based on query type
        prompt_template = None
        if state["query_type"] == "comparison":
            prompt_template = self.prompt_templates.COMPARISON_QUERY_PROMPT
        elif state["query_type"] == "evolution":
            prompt_template = self.prompt_templates.EVOLUTION_QUERY_PROMPT
        elif state["query_type"] == "exploratory":
            prompt_template = self.prompt_templates.EXPLORATORY_QUERY_PROMPT
        elif state["temporal_detected"]:
            prompt_template = self.prompt_templates.TEMPORAL_QUERY_PROMPT
        else:
            prompt_template = self.prompt_templates.FACTUAL_QUERY_PROMPT

        # Build prompt
        user_prompt = prompt_template.format(
            query=state["query"],
            context=state["formatted_context"],
            temporal_context=state.get("temporal_context", "Current information"),
            entity_relationships="",  # Could add this
        )

        messages = [
            SystemMessage(content=self.prompt_templates.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        # Generate answer
        try:
            response = self.llm.invoke(messages)
            state["answer"] = response.content
            state["sources"] = state["context_metadata"].get("sources", [])
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            state["answer"] = f"Error generating answer: {str(e)}"
            state["error"] = str(e)
            state["sources"] = []

        logger.info("Answer generated")

        return state

    def _verify_node(self, state: RAGState) -> RAGState:
        """
        Node 5: Verify answer quality and citations.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        logger.info("Node: Verifying answer")

        # Simple verification checks
        verification_passed = True
        notes = []

        # Check 1: Answer is not empty
        if not state["answer"] or len(state["answer"]) < 10:
            verification_passed = False
            notes.append("Answer is too short or empty")

        # Check 2: Answer references sources (simple check)
        answer_lower = state["answer"].lower()
        has_citation = any(
            marker in answer_lower
            for marker in ["source", "according to", "from", "document", "["]
        )

        if not has_citation and state["sources"]:
            notes.append("Answer may lack proper source citations")

        # Check 3: Temporal consistency
        if state["temporal_detected"]:
            # Check if answer mentions time period
            if not any(
                word in answer_lower
                for word in ["year", "time", "period", "date", "recent", "current"]
            ):
                notes.append("Answer may lack temporal context despite temporal query")

        state["verified"] = verification_passed
        state["verification_notes"] = "; ".join(notes) if notes else None

        logger.info(f"Verification: {'passed' if verification_passed else 'failed'}")

        return state

    def query(
        self,
        question: str,
        top_k: int = 10,
    ) -> Dict:
        """
        Execute RAG workflow.

        Args:
            question: User question
            top_k: Number of retrieval results

        Returns:
            Result dictionary
        """
        logger.info(f"Executing RAG workflow for: '{question[:50]}...'")

        # Initialize state
        initial_state = {
            "query": question,
            "top_k": top_k,
            "temporal_detected": False,
            "temporal_context": None,
            "entities_detected": [],
            "query_type": "factual",
            "retrieval_results": [],
            "num_results": 0,
            "formatted_context": "",
            "context_metadata": {},
            "answer": "",
            "sources": [],
            "verified": False,
            "verification_notes": None,
            "error": None,
        }

        # Execute workflow
        try:
            final_state = self.workflow.invoke(initial_state)
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            return {
                "answer": f"Error executing RAG workflow: {str(e)}",
                "sources": [],
                "error": str(e),
            }

        # Format response
        return {
            "answer": final_state["answer"],
            "sources": final_state["sources"],
            "metadata": {
                "query_type": final_state["query_type"],
                "temporal_detected": final_state["temporal_detected"],
                "num_results": final_state["num_results"],
                "verified": final_state["verified"],
                "verification_notes": final_state["verification_notes"],
                "entities_detected": final_state["entities_detected"],
            },
        }


# Global RAG graph instance
_rag_graph: Optional[RAGGraph] = None


def get_rag_graph() -> RAGGraph:
    """Get the global RAG graph instance."""
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = RAGGraph()
    return _rag_graph


def create_rag_graph(
    hybrid_search: Optional[HybridSearch] = None,
    temporal_retrieval: Optional[TemporalRetrieval] = None,
    context_builder: Optional[ContextBuilder] = None,
    llm: Optional[ChatOpenAI] = None,
) -> RAGGraph:
    """
    Create a new RAG graph instance.

    Args:
        hybrid_search: Optional hybrid search
        temporal_retrieval: Optional temporal retrieval
        context_builder: Optional context builder
        llm: Optional LLM

    Returns:
        RAG graph instance
    """
    return RAGGraph(
        hybrid_search=hybrid_search,
        temporal_retrieval=temporal_retrieval,
        context_builder=context_builder,
        llm=llm,
    )
