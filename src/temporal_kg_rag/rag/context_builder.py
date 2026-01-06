"""Context builder for assembling retrieval results into LLM context."""

from typing import Dict, List, Optional

from temporal_kg_rag.models.temporal import TemporalContext
from temporal_kg_rag.rag.prompts import PromptTemplates
from temporal_kg_rag.retrieval.context_expansion import ContextExpander, get_context_expander
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class ContextBuilder:
    """Build rich context from retrieval results for RAG."""

    def __init__(self, context_expander: Optional[ContextExpander] = None):
        """
        Initialize context builder.

        Args:
            context_expander: Optional context expander instance
        """
        self.context_expander = context_expander or get_context_expander()
        self.prompt_templates = PromptTemplates()

    def build_context(
        self,
        query: str,
        results: List[Dict],
        temporal_context: Optional[TemporalContext] = None,
        expand_context: bool = True,
        max_context_length: int = 4000,
    ) -> Dict:
        """
        Build comprehensive context from retrieval results.

        Args:
            query: User query
            results: Retrieval results
            temporal_context: Optional temporal context
            expand_context: Whether to expand with graph information
            max_context_length: Maximum context length in characters

        Returns:
            Dictionary with formatted context and metadata
        """
        logger.info(f"Building context for query: '{query[:50]}...'")

        # Expand results with additional context if requested
        if expand_context and results:
            logger.info("Expanding context with graph information")
            results = self.context_expander.expand_results(
                results,
                include_neighboring_chunks=True,
                include_entities=True,
                include_related_chunks=False,  # Avoid too much context
                neighboring_chunk_window=1,
            )

        # Format main context
        formatted_context = self.prompt_templates.format_context(
            results,
            include_metadata=True,
        )

        # Truncate if too long
        if len(formatted_context) > max_context_length:
            logger.warning(
                f"Context too long ({len(formatted_context)} chars), "
                f"truncating to {max_context_length}"
            )
            formatted_context = formatted_context[:max_context_length] + "\n[Context truncated...]"

        # Format entity relationships
        entity_relationships = self.prompt_templates.format_entity_relationships(results)

        # Format temporal context
        temporal_context_str = None
        if temporal_context and temporal_context.has_temporal_reference:
            if temporal_context.temporal_filter:
                temporal_context_str = self.prompt_templates.format_temporal_context(
                    temporal_context.temporal_filter.dict()
                )

        # Build metadata
        metadata = {
            "num_results": len(results),
            "context_length": len(formatted_context),
            "has_temporal_context": temporal_context is not None and temporal_context.has_temporal_reference,
            "sources": self._extract_sources(results),
        }

        logger.info(
            f"Context built: {metadata['num_results']} results, "
            f"{metadata['context_length']} characters"
        )

        return {
            "query": query,
            "formatted_context": formatted_context,
            "entity_relationships": entity_relationships,
            "temporal_context": temporal_context_str,
            "temporal_context_obj": temporal_context,
            "results": results,
            "metadata": metadata,
        }

    def build_prompt(
        self,
        context_dict: Dict,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """
        Build complete prompt from context dictionary.

        Args:
            context_dict: Context dictionary from build_context
            system_prompt: Optional custom system prompt

        Returns:
            Dictionary with system and user prompts
        """
        # Use custom system prompt or default
        system = system_prompt or self.prompt_templates.SYSTEM_PROMPT

        # Select appropriate prompt template
        prompt_template = self.prompt_templates.select_prompt_template(
            query=context_dict["query"],
            has_temporal_context=context_dict["metadata"]["has_temporal_context"],
        )

        # Build user prompt
        user_prompt = self.prompt_templates.build_prompt(
            query=context_dict["query"],
            context=context_dict["formatted_context"],
            temporal_context=context_dict.get("temporal_context"),
            entity_relationships=context_dict.get("entity_relationships"),
            prompt_template=prompt_template,
        )

        return {
            "system": system,
            "user": user_prompt,
            "metadata": context_dict["metadata"],
        }

    def build_context_with_history(
        self,
        query: str,
        results: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
        temporal_context: Optional[TemporalContext] = None,
    ) -> Dict:
        """
        Build context including conversation history.

        Args:
            query: User query
            results: Retrieval results
            conversation_history: Previous conversation turns
            temporal_context: Optional temporal context

        Returns:
            Context dictionary with history
        """
        # Build base context
        context_dict = self.build_context(query, results, temporal_context)

        # Add conversation history
        if conversation_history:
            history_str = self._format_conversation_history(conversation_history)
            context_dict["conversation_history"] = history_str
            context_dict["metadata"]["has_history"] = True
            context_dict["metadata"]["history_length"] = len(conversation_history)
        else:
            context_dict["metadata"]["has_history"] = False

        return context_dict

    def build_summary_context(
        self,
        results: List[Dict],
        max_chunks: int = 5,
    ) -> str:
        """
        Build a summary context suitable for RAG.

        Args:
            results: Retrieval results
            max_chunks: Maximum number of chunks to include

        Returns:
            Summary context string
        """
        return self.context_expander.build_context_summary(results[:max_chunks])

    def _extract_sources(self, results: List[Dict]) -> List[Dict]:
        """Extract source information from results."""
        sources = []

        for result in results:
            source = {
                "document_id": result.get("document_id"),
                "document_title": result.get("document_title"),
                "chunk_id": result.get("chunk_id"),
                "created_at": str(result.get("created_at", "")),
                "score": result.get("rrf_score") or result.get("score", 0),
            }
            sources.append(source)

        return sources

    def _format_conversation_history(
        self,
        history: List[Dict],
        max_turns: int = 3,
    ) -> str:
        """Format conversation history for context."""
        history_parts = []

        # Take last N turns
        recent_history = history[-max_turns:] if len(history) > max_turns else history

        for turn in recent_history:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")

            if role == "user":
                history_parts.append(f"User: {content}")
            elif role == "assistant":
                history_parts.append(f"Assistant: {content}")

        return "\n".join(history_parts)

    def add_expanded_context_to_results(
        self,
        results: List[Dict],
        include_neighbors: bool = True,
        include_entities: bool = True,
    ) -> List[Dict]:
        """
        Add expanded context to results in place.

        Args:
            results: Retrieval results
            include_neighbors: Include neighboring chunks
            include_entities: Include entity information

        Returns:
            Results with expanded context
        """
        return self.context_expander.expand_results(
            results,
            include_neighboring_chunks=include_neighbors,
            include_entities=include_entities,
            include_related_chunks=False,
            include_document_context=True,
        )

    def format_sources_for_response(self, results: List[Dict]) -> str:
        """
        Format sources for inclusion in response.

        Args:
            results: Retrieval results

        Returns:
            Formatted source citations
        """
        return self.prompt_templates.format_sources_for_citation(results)


# Global context builder instance
_context_builder: Optional[ContextBuilder] = None


def get_context_builder() -> ContextBuilder:
    """Get the global context builder instance."""
    global _context_builder
    if _context_builder is None:
        _context_builder = ContextBuilder()
    return _context_builder
