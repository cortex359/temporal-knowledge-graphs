"""Prompt templates for RAG system."""

from typing import Dict, List, Optional
from datetime import datetime

from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class PromptTemplates:
    """Prompt templates for different query types and scenarios."""

    # System prompt for the RAG assistant
    SYSTEM_PROMPT = """You are a knowledgeable AI assistant with access to a temporal knowledge graph.
Your role is to provide accurate, well-sourced answers based on the retrieved information.

Key guidelines:
- Always cite your sources using the provided document titles and dates
- If information is from a specific time period, clearly state this in your answer
- If the retrieved context doesn't contain enough information to answer fully, say so
- Be precise about temporal aspects - distinguish between current and historical information
- When information conflicts, acknowledge this and explain the different perspectives
- Use the entity relationships in the knowledge graph to provide richer context

Format your citations like this: [Source: "Document Title" (Date)]"""

    # Query understanding prompt
    QUERY_UNDERSTANDING_PROMPT = """Analyze the following user query and extract:
1. The main intent and information need
2. Any temporal references (dates, time periods, "current", "historical", etc.)
3. Key entities or concepts mentioned
4. The type of query (factual, comparison, temporal evolution, exploratory, etc.)

User Query: {query}

Provide your analysis in a structured format."""

    # Factual query prompt
    FACTUAL_QUERY_PROMPT = """Based on the following context from our knowledge graph, answer the user's question accurately and concisely.

User Question: {query}

Retrieved Context:
{context}

Instructions:
- Provide a direct, factual answer
- Cite specific sources for your claims
- If the context contains multiple perspectives, present them fairly
- If information is insufficient, clearly state what's missing

Answer:"""

    # Temporal query prompt
    TEMPORAL_QUERY_PROMPT = """Based on the following context from our temporal knowledge graph, answer the user's question about information at a specific point in time or time period.

User Question: {query}

Temporal Context: {temporal_context}

Retrieved Information:
{context}

Instructions:
- Clearly indicate the time period your answer applies to
- If information has changed over time, explain the evolution
- Cite sources with their dates
- Distinguish between information that was current at that time vs. retrospective analysis

Answer:"""

    # Comparison query prompt
    COMPARISON_QUERY_PROMPT = """Based on the following context from our knowledge graph, compare and contrast the requested topics.

User Question: {query}

Retrieved Context:
{context}

Instructions:
- Organize your answer to clearly show similarities and differences
- Use specific examples from the sources
- Acknowledge any limitations in the comparison
- Cite sources for each point

Answer:"""

    # Exploratory query prompt
    EXPLORATORY_QUERY_PROMPT = """Based on the following context from our knowledge graph, provide a comprehensive overview of the topic.

User Question: {query}

Retrieved Context:
{context}

Entity Relationships:
{entity_relationships}

Instructions:
- Provide a well-structured overview covering key aspects
- Use the entity relationships to show connections between concepts
- Include relevant details and examples
- Cite sources throughout your explanation
- Suggest related topics the user might want to explore

Answer:"""

    # Evolution/history query prompt
    EVOLUTION_QUERY_PROMPT = """Based on the following context from our temporal knowledge graph, describe how the topic has evolved over time.

User Question: {query}

Temporal Information Across Time Periods:
{temporal_context}

Instructions:
- Organize your answer chronologically
- Highlight key changes and developments
- Explain the context and causes of changes when available
- Note which information is most current
- Cite sources with dates for each time period

Answer:"""

    # Source verification prompt
    VERIFICATION_PROMPT = """Review the following answer and verify that:
1. All factual claims are supported by the provided sources
2. Citations are accurate and properly formatted
3. Temporal claims are consistent with the source dates
4. No unsupported inferences or hallucinations are present

Answer to verify:
{answer}

Sources used:
{sources}

Provide your verification assessment:"""

    @staticmethod
    def format_context(results: List[Dict], include_metadata: bool = True) -> str:
        """
        Format retrieval results into a context string.

        Args:
            results: List of retrieval results
            include_metadata: Whether to include metadata

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, result in enumerate(results, 1):
            doc_title = result.get("document_title", "Unknown Source")
            text = result.get("text", "")
            created_at = result.get("created_at")

            # Format timestamp
            date_str = ""
            if created_at:
                if isinstance(created_at, datetime):
                    date_str = created_at.strftime("%Y-%m-%d")
                else:
                    date_str = str(created_at)[:10]

            context_parts.append(f"[Source {i}: \"{doc_title}\" ({date_str})]")
            context_parts.append(text)

            # Add entity information if available
            if include_metadata and "entities" in result:
                entities = result["entities"]
                if entities:
                    if isinstance(entities[0], dict):
                        entity_names = [e.get("name", str(e)) for e in entities[:5] if e.get("name")]
                    else:
                        entity_names = [e for e in (entities[:5] if isinstance(entities, list) else []) if e]
                    if entity_names:
                        context_parts.append(f"Key entities: {', '.join(entity_names)}")

            context_parts.append("")  # Empty line between sources

        return "\n".join(context_parts)

    @staticmethod
    def format_entity_relationships(results: List[Dict]) -> str:
        """
        Format entity relationships from retrieval results.

        Args:
            results: List of retrieval results

        Returns:
            Formatted entity relationship string
        """
        entities = set()
        relationships = []

        for result in results:
            if "entities" in result:
                result_entities = result["entities"]
                if isinstance(result_entities, list):
                    for entity in result_entities:
                        if isinstance(entity, dict):
                            name = entity.get("name")
                            if name:
                                entities.add(str(name))
                        elif entity is not None:
                            entity_str = str(entity).strip()
                            if entity_str:
                                entities.add(entity_str)

        # Filter out any empty strings or None values that might have slipped through
        entities = {e for e in entities if e and isinstance(e, str)}

        if not entities:
            return "No entity relationships found."

        return f"Entities mentioned: {', '.join(sorted(entities))}"

    @staticmethod
    def format_temporal_context(temporal_filter: Optional[Dict]) -> str:
        """
        Format temporal context information.

        Args:
            temporal_filter: Temporal filter information

        Returns:
            Formatted temporal context string
        """
        if not temporal_filter:
            return "Current information (no temporal filter applied)"

        filter_type = temporal_filter.get("query_type", "unknown")

        if filter_type == "point_in_time":
            timestamp = temporal_filter.get("point_in_time", "unknown")
            return f"Information as of: {timestamp}"
        elif filter_type == "time_range":
            start = temporal_filter.get("start_time", "unknown")
            end = temporal_filter.get("end_time", "unknown")
            return f"Information from {start} to {end}"
        elif filter_type == "latest":
            return "Most recent/current information"
        else:
            return f"Temporal filter: {filter_type}"

    @staticmethod
    def select_prompt_template(
        query: str,
        has_temporal_context: bool = False,
        is_comparison: bool = False,
        is_exploratory: bool = False,
        is_evolution: bool = False,
    ) -> str:
        """
        Select appropriate prompt template based on query characteristics.

        Args:
            query: User query
            has_temporal_context: Whether query has temporal context
            is_comparison: Whether query is asking for comparison
            is_exploratory: Whether query is exploratory
            is_evolution: Whether query asks about evolution/history

        Returns:
            Selected prompt template
        """
        query_lower = query.lower()

        # Check for evolution/history queries
        if is_evolution or any(word in query_lower for word in [
            "evolution", "history", "changed over time", "timeline", "how has"
        ]):
            return PromptTemplates.EVOLUTION_QUERY_PROMPT

        # Check for comparison queries
        if is_comparison or any(word in query_lower for word in [
            "compare", "difference", "versus", "vs", "contrast"
        ]):
            return PromptTemplates.COMPARISON_QUERY_PROMPT

        # Check for temporal queries
        if has_temporal_context:
            return PromptTemplates.TEMPORAL_QUERY_PROMPT

        # Check for exploratory queries
        if is_exploratory or any(word in query_lower for word in [
            "tell me about", "explain", "what is", "overview", "describe"
        ]):
            return PromptTemplates.EXPLORATORY_QUERY_PROMPT

        # Default to factual query
        return PromptTemplates.FACTUAL_QUERY_PROMPT

    @staticmethod
    def build_prompt(
        query: str,
        context: str,
        temporal_context: Optional[str] = None,
        entity_relationships: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ) -> str:
        """
        Build complete prompt from template and components.

        Args:
            query: User query
            context: Formatted context
            temporal_context: Optional temporal context
            entity_relationships: Optional entity relationships
            prompt_template: Optional specific template to use

        Returns:
            Complete prompt
        """
        # Select template if not provided
        if prompt_template is None:
            prompt_template = PromptTemplates.select_prompt_template(
                query,
                has_temporal_context=temporal_context is not None,
            )

        # Format the prompt
        prompt = prompt_template.format(
            query=query,
            context=context,
            temporal_context=temporal_context or "Current information",
            entity_relationships=entity_relationships or "No entity information",
        )

        return prompt

    @staticmethod
    def format_sources_for_citation(results: List[Dict]) -> str:
        """
        Format sources for citation verification.

        Args:
            results: List of retrieval results

        Returns:
            Formatted source list
        """
        source_parts = []

        for i, result in enumerate(results, 1):
            doc_title = result.get("document_title", "Unknown")
            doc_id = result.get("document_id", "")
            chunk_id = result.get("chunk_id", "")
            created_at = result.get("created_at", "")

            source_parts.append(
                f"{i}. \"{doc_title}\" (Date: {created_at})\n"
                f"   Document ID: {doc_id}\n"
                f"   Chunk ID: {chunk_id}"
            )

        return "\n".join(source_parts)


# Convenience function
def get_prompt_templates() -> PromptTemplates:
    """Get prompt templates instance."""
    return PromptTemplates()
