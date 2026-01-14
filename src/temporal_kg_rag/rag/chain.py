"""LangChain RAG chain implementation."""

from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from temporal_kg_rag.config.settings import get_settings
from temporal_kg_rag.rag.context_builder import ContextBuilder, get_context_builder
from temporal_kg_rag.retrieval.hybrid_search import HybridSearch, get_hybrid_search
from temporal_kg_rag.retrieval.temporal_retrieval import TemporalRetrieval, get_temporal_retrieval
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class RAGChain:
    """Simple RAG chain using LangChain."""

    def __init__(
        self,
        hybrid_search: Optional[HybridSearch] = None,
        temporal_retrieval: Optional[TemporalRetrieval] = None,
        context_builder: Optional[ContextBuilder] = None,
        llm: Optional[ChatOpenAI] = None,
    ):
        """
        Initialize RAG chain.

        Args:
            hybrid_search: Optional hybrid search instance
            temporal_retrieval: Optional temporal retrieval instance
            context_builder: Optional context builder instance
            llm: Optional LangChain LLM instance
        """
        self.hybrid_search = hybrid_search or get_hybrid_search()
        self.temporal_retrieval = temporal_retrieval or get_temporal_retrieval()
        self.context_builder = context_builder or get_context_builder()

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

        logger.info("RAG chain initialized")

    def query(
        self,
        question: str,
        top_k: int = 5,
        use_temporal_detection: bool = True,
        expand_context: bool = True,
    ) -> Dict:
        """
        Query the RAG system.

        Args:
            question: User question
            top_k: Number of retrieval results
            use_temporal_detection: Whether to detect temporal context
            expand_context: Whether to expand with graph information

        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"RAG query: '{question[:50]}...'")

        # Step 1: Retrieval with temporal awareness
        if use_temporal_detection:
            retrieval_result = self.temporal_retrieval.search_with_temporal_context(
                query=question,
                top_k=top_k,
                auto_detect_temporal=True,
            )
            results = retrieval_result["results"]
            temporal_context = retrieval_result.get("temporal_context")
        else:
            results = self.hybrid_search.search(question, top_k=top_k)
            temporal_context = None

        if not results:
            logger.warning("No results found for query")
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "metadata": {"num_sources": 0},
            }

        logger.info(f"Retrieved {len(results)} results")

        # Step 2: Build context
        context_dict = self.context_builder.build_context(
            query=question,
            results=results,
            temporal_context=temporal_context,
            expand_context=expand_context,
        )

        # Step 3: Build prompt
        prompt_dict = self.context_builder.build_prompt(context_dict)

        # Step 4: Generate answer
        logger.info("Generating answer with LLM")
        messages = [
            SystemMessage(content=prompt_dict["system"]),
            HumanMessage(content=prompt_dict["user"]),
        ]

        try:
            response = self.llm.invoke(messages)
            answer = response.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": context_dict["metadata"]["sources"],
                "metadata": context_dict["metadata"],
                "error": str(e),
            }

        logger.info("Answer generated successfully")

        # Step 5: Format response
        return {
            "answer": answer,
            "sources": context_dict["metadata"]["sources"],
            "context": context_dict["formatted_context"],
            "temporal_context": temporal_context,
            "metadata": {
                **context_dict["metadata"],
                "query": question,
                "temporal_detection_used": use_temporal_detection,
                "context_expanded": expand_context,
            },
        }

    def query_with_history(
        self,
        question: str,
        conversation_history: List[Dict],
        top_k: int = 5,
    ) -> Dict:
        """
        Query with conversation history.

        Args:
            question: User question
            conversation_history: Previous conversation turns
            top_k: Number of retrieval results

        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"RAG query with history ({len(conversation_history)} turns)")

        # Retrieve results
        retrieval_result = self.temporal_retrieval.search_with_temporal_context(
            query=question,
            top_k=top_k,
        )
        results = retrieval_result["results"]
        temporal_context = retrieval_result.get("temporal_context")

        if not results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
            }

        # Build context with history
        context_dict = self.context_builder.build_context_with_history(
            query=question,
            results=results,
            conversation_history=conversation_history,
            temporal_context=temporal_context,
        )

        # Build prompt
        prompt_dict = self.context_builder.build_prompt(context_dict)

        # Add conversation history to messages
        messages = [SystemMessage(content=prompt_dict["system"])]

        # Add history
        for turn in conversation_history[-3:]:  # Last 3 turns
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(SystemMessage(content=turn["content"]))

        # Add current query
        messages.append(HumanMessage(content=prompt_dict["user"]))

        # Generate answer
        try:
            response = self.llm.invoke(messages)
            answer = response.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": context_dict["metadata"]["sources"],
                "error": str(e),
            }

        return {
            "answer": answer,
            "sources": context_dict["metadata"]["sources"],
            "metadata": context_dict["metadata"],
        }

    def stream_query(
        self,
        question: str,
        top_k: int = 5,
    ):
        """
        Stream answer generation.

        Args:
            question: User question
            top_k: Number of retrieval results

        Yields:
            Answer chunks
        """
        logger.info(f"Streaming RAG query: '{question[:50]}...'")

        # Retrieve and build context
        retrieval_result = self.temporal_retrieval.search_with_temporal_context(
            query=question,
            top_k=top_k,
        )
        results = retrieval_result["results"]

        if not results:
            yield "I couldn't find any relevant information to answer your question."
            return

        context_dict = self.context_builder.build_context(
            query=question,
            results=results,
            temporal_context=retrieval_result.get("temporal_context"),
        )

        prompt_dict = self.context_builder.build_prompt(context_dict)

        # Build messages
        messages = [
            SystemMessage(content=prompt_dict["system"]),
            HumanMessage(content=prompt_dict["user"]),
        ]

        # Stream response
        try:
            for chunk in self.llm.stream(messages):
                if hasattr(chunk, "content"):
                    yield chunk.content
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"\n\nError: {str(e)}"


# Global RAG chain instance
_rag_chain: Optional[RAGChain] = None


def get_rag_chain() -> RAGChain:
    """Get the global RAG chain instance."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain


def query(question: str, top_k: int = 5) -> Dict:
    """
    Convenience function to query RAG system.

    Args:
        question: User question
        top_k: Number of results

    Returns:
        Answer dictionary
    """
    chain = get_rag_chain()
    return chain.query(question, top_k)
