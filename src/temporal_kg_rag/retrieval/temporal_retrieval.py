"""Temporal retrieval with time-aware search capabilities."""

import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx

from temporal_kg_rag.config.settings import get_settings
from temporal_kg_rag.graph.neo4j_client import Neo4jClient, get_neo4j_client
from temporal_kg_rag.models.temporal import TemporalContext, TemporalFilter, TemporalQueryType
from temporal_kg_rag.retrieval.hybrid_search import HybridSearch, get_hybrid_search
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalRetrieval:
    """Temporal-aware retrieval with time travel and version queries."""

    def __init__(
        self,
        client: Optional[Neo4jClient] = None,
        hybrid_search: Optional[HybridSearch] = None,
    ):
        """
        Initialize temporal retrieval.

        Args:
            client: Optional Neo4j client
            hybrid_search: Optional hybrid search instance
        """
        self.client = client or get_neo4j_client()
        self.hybrid_search = hybrid_search or get_hybrid_search()

    def parse_temporal_context(self, query: str) -> TemporalContext:
        """
        Parse temporal context from query using LLM for better date recognition.

        Args:
            query: Query text

        Returns:
            Temporal context
        """
        # Try LLM-based temporal parsing first
        try:
            settings = get_settings()
            llm_result = self._parse_temporal_with_llm(query, settings)

            if llm_result:
                logger.info(f"LLM parsed temporal context: {llm_result}")
                return llm_result

        except Exception as e:
            logger.warning(f"LLM temporal parsing failed: {e}, falling back to keyword-based parsing")

        # Fallback to keyword-based parsing
        return self._parse_temporal_fallback(query)

    def _parse_temporal_with_llm(self, query: str, settings) -> Optional[TemporalContext]:
        """
        Use LLM to parse temporal context from query.

        Args:
            query: Query text
            settings: Application settings

        Returns:
            Temporal context or None if no temporal reference found
        """
        prompt = f"""Analyze this query and extract temporal information.

Query: {query}

Return a JSON object with the following structure:
{{
    "has_temporal_reference": true/false,
    "temporal_type": "latest" | "point_in_time" | "time_range" | "history" | null,
    "dates": [
        {{"date": "YYYY-MM-DD", "type": "start" | "end" | "point"}}
    ],
    "temporal_phrase": "extracted phrase or null"
}}

Examples:
- "What happened on 21. Januar 2024?" -> {{"has_temporal_reference": true, "temporal_type": "point_in_time", "dates": [{{"date": "2024-01-21", "type": "point"}}], "temporal_phrase": "on 21. Januar 2024"}}
- "What is the latest AI development?" -> {{"has_temporal_reference": true, "temporal_type": "latest", "dates": [], "temporal_phrase": "latest"}}
- "Between 2020 and 2023" -> {{"has_temporal_reference": true, "temporal_type": "time_range", "dates": [{{"date": "2020-01-01", "type": "start"}}, {{"date": "2023-12-31", "type": "end"}}], "temporal_phrase": "Between 2020 and 2023"}}

Return only valid JSON, no additional text."""

        client = httpx.Client(timeout=30.0)
        response = client.post(
            f"{settings.litellm_api_base}/chat/completions",
            json={
                "model": settings.entity_extraction_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 500,
            },
            headers={"Authorization": f"Bearer {settings.litellm_api_key}"}
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"].strip()

        # Extract JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            return None

        data = json.loads(json_match.group())

        if not data.get("has_temporal_reference", False):
            return TemporalContext(has_temporal_reference=False)

        # Create temporal filter based on parsed data
        temporal_filter = None
        temporal_type = data.get("temporal_type")
        dates = data.get("dates", [])

        if temporal_type == "latest":
            temporal_filter = TemporalFilter.create_latest()

        elif temporal_type == "history":
            temporal_filter = TemporalFilter.create_history()

        elif temporal_type == "point_in_time" and dates:
            date_str = dates[0]["date"]
            timestamp = datetime.fromisoformat(date_str)
            temporal_filter = TemporalFilter.create_point_in_time(timestamp)

        elif temporal_type == "time_range" and len(dates) >= 2:
            start_date = datetime.fromisoformat(dates[0]["date"])
            end_date = datetime.fromisoformat(dates[1]["date"])
            temporal_filter = TemporalFilter.create_time_range(start_date, end_date)

        context = TemporalContext(
            has_temporal_reference=True,
            temporal_filter=temporal_filter,
            temporal_keywords=[data.get("temporal_phrase", "")],
            original_temporal_phrase=data.get("temporal_phrase"),
        )

        return context

    def _parse_temporal_fallback(self, query: str) -> TemporalContext:
        """
        Fallback keyword-based temporal parsing.

        Args:
            query: Query text

        Returns:
            Temporal context
        """
        query_lower = query.lower()

        # Temporal keywords
        keywords = {
            "latest": ["latest", "current", "now", "recent", "today"],
            "point_in_time": ["in", "at", "during", "on"],
            "time_range": ["between", "from", "to", "since", "until", "before", "after"],
            "history": ["history", "evolution", "changed", "over time", "timeline"],
        }

        # Check for temporal keywords
        found_keywords = []
        has_temporal_reference = False

        for category, words in keywords.items():
            for word in words:
                if word in query_lower:
                    found_keywords.append(word)
                    has_temporal_reference = True
                    break

        if not has_temporal_reference:
            return TemporalContext(has_temporal_reference=False)

        # Try to extract years (simple patterns)
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, query)

        # Create temporal filter based on detected patterns
        temporal_filter = None

        if "latest" in query_lower or "current" in query_lower or "now" in query_lower:
            temporal_filter = TemporalFilter.create_latest()

        elif "history" in query_lower or "evolution" in query_lower or "over time" in query_lower:
            temporal_filter = TemporalFilter.create_history()

        elif years:
            # Found a year, create point-in-time or range query
            if len(years) == 1:
                year = int(years[0])
                timestamp = datetime(year, 12, 31, 23, 59, 59)
                temporal_filter = TemporalFilter.create_point_in_time(timestamp)

            elif len(years) >= 2:
                start_year = int(min(years))
                end_year = int(max(years))
                start_time = datetime(start_year, 1, 1)
                end_time = datetime(end_year, 12, 31, 23, 59, 59)
                temporal_filter = TemporalFilter.create_time_range(start_time, end_time)

        context = TemporalContext(
            has_temporal_reference=True,
            temporal_filter=temporal_filter,
            temporal_keywords=found_keywords,
            original_temporal_phrase=None,
        )

        logger.info(f"Fallback parsed temporal context: {context.temporal_keywords}")

        return context

    def search_with_temporal_context(
        self,
        query: str,
        top_k: int = 10,
        auto_detect_temporal: bool = True,
        temporal_filter: Optional[TemporalFilter] = None,
    ) -> Dict:
        """
        Search with automatic temporal context detection.

        Args:
            query: Search query
            top_k: Number of results
            auto_detect_temporal: Whether to auto-detect temporal context
            temporal_filter: Optional manual temporal filter (overrides auto-detection)

        Returns:
            Dictionary with results and temporal context
        """
        logger.info(f"Temporal search: '{query[:50]}...'")

        # Parse temporal context if auto-detection is enabled
        temporal_context = None
        if auto_detect_temporal and temporal_filter is None:
            temporal_context = self.parse_temporal_context(query)
            if temporal_context.has_temporal_reference:
                temporal_filter = temporal_context.temporal_filter

        # Execute search with temporal filter
        results = self.hybrid_search.search(
            query=query,
            top_k=top_k,
            temporal_filter=temporal_filter,
        )

        return {
            "results": results,
            "temporal_context": temporal_context,
            "temporal_filter_applied": temporal_filter is not None,
            "query": query,
        }

    def search_at_time(
        self,
        query: str,
        timestamp: datetime,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Search for information as it existed at a specific point in time.

        Args:
            query: Search query
            timestamp: Point in time
            top_k: Number of results

        Returns:
            List of search results
        """
        logger.info(f"Point-in-time search at {timestamp.isoformat()}")

        temporal_filter = TemporalFilter.create_point_in_time(timestamp)

        results = self.hybrid_search.search(
            query=query,
            top_k=top_k,
            temporal_filter=temporal_filter,
        )

        # Add temporal context to results
        for result in results:
            result["query_timestamp"] = timestamp.isoformat()
            result["temporal_query_type"] = "point_in_time"

        return results

    def search_time_range(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Search for information within a time range.

        Args:
            query: Search query
            start_time: Start of time range
            end_time: End of time range
            top_k: Number of results

        Returns:
            List of search results
        """
        logger.info(
            f"Time range search: {start_time.isoformat()} to {end_time.isoformat()}"
        )

        temporal_filter = TemporalFilter.create_time_range(start_time, end_time)

        results = self.hybrid_search.search(
            query=query,
            top_k=top_k,
            temporal_filter=temporal_filter,
        )

        # Add temporal context to results
        for result in results:
            result["query_start_time"] = start_time.isoformat()
            result["query_end_time"] = end_time.isoformat()
            result["temporal_query_type"] = "time_range"

        return results

    def compare_over_time(
        self,
        query: str,
        timestamps: List[datetime],
        top_k: int = 5,
    ) -> Dict:
        """
        Compare search results across multiple points in time.

        Args:
            query: Search query
            timestamps: List of timestamps to compare
            top_k: Number of results per timestamp

        Returns:
            Dictionary with comparative results
        """
        logger.info(f"Comparing '{query}' across {len(timestamps)} time points")

        comparative_results = {}

        for timestamp in timestamps:
            results = self.search_at_time(query, timestamp, top_k)
            comparative_results[timestamp.isoformat()] = results

        # Analyze changes over time
        all_chunk_ids = set()
        for results in comparative_results.values():
            all_chunk_ids.update(r["chunk_id"] for r in results)

        analysis = {
            "query": query,
            "timestamps": [t.isoformat() for t in timestamps],
            "results_by_time": comparative_results,
            "unique_chunks_across_time": len(all_chunk_ids),
            "temporal_evolution": self._analyze_temporal_evolution(comparative_results),
        }

        return analysis

    def _analyze_temporal_evolution(
        self,
        results_by_time: Dict[str, List[Dict]],
    ) -> Dict:
        """Analyze how results evolve over time."""
        timestamps = sorted(results_by_time.keys())

        evolution = {
            "new_chunks_per_period": [],
            "disappeared_chunks_per_period": [],
            "persistent_chunks": set(),
        }

        prev_chunk_ids = set()

        for i, timestamp in enumerate(timestamps):
            current_chunk_ids = {r["chunk_id"] for r in results_by_time[timestamp]}

            if i == 0:
                # First time period
                evolution["new_chunks_per_period"].append(len(current_chunk_ids))
                evolution["disappeared_chunks_per_period"].append(0)
            else:
                # Compare with previous period
                new_chunks = current_chunk_ids - prev_chunk_ids
                disappeared_chunks = prev_chunk_ids - current_chunk_ids

                evolution["new_chunks_per_period"].append(len(new_chunks))
                evolution["disappeared_chunks_per_period"].append(len(disappeared_chunks))

                # Track persistent chunks
                if i == 1:
                    evolution["persistent_chunks"] = prev_chunk_ids & current_chunk_ids
                else:
                    evolution["persistent_chunks"] &= current_chunk_ids

            prev_chunk_ids = current_chunk_ids

        evolution["persistent_chunks"] = list(evolution["persistent_chunks"])

        return evolution


# Global temporal retrieval instance
_temporal_retrieval: Optional[TemporalRetrieval] = None


def get_temporal_retrieval() -> TemporalRetrieval:
    """Get the global temporal retrieval instance."""
    global _temporal_retrieval
    if _temporal_retrieval is None:
        _temporal_retrieval = TemporalRetrieval()
    return _temporal_retrieval
