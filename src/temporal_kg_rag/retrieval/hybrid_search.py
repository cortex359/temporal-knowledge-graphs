"""Hybrid search combining vector and graph search with Reciprocal Rank Fusion."""

from collections import defaultdict
from typing import Dict, List, Optional

from temporal_kg_rag.config.settings import get_settings
from temporal_kg_rag.models.temporal import TemporalFilter
from temporal_kg_rag.retrieval.graph_search import GraphSearch, get_graph_search
from temporal_kg_rag.retrieval.vector_search import VectorSearch, get_vector_search
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class HybridSearch:
    """Hybrid search combining vector and graph-based retrieval."""

    def __init__(
        self,
        vector_search: Optional[VectorSearch] = None,
        graph_search: Optional[GraphSearch] = None,
    ):
        """
        Initialize hybrid search.

        Args:
            vector_search: Optional vector search instance
            graph_search: Optional graph search instance
        """
        self.vector_search = vector_search or get_vector_search()
        self.graph_search = graph_search or get_graph_search()
        self.settings = get_settings()

    def search(
        self,
        query: str,
        top_k: int = 10,
        temporal_filter: Optional[TemporalFilter] = None,
        alpha: Optional[float] = None,
        rrf_k: Optional[int] = None,
        retrieval_multiplier: int = 2,
    ) -> List[Dict]:
        """
        Hybrid search using both vector and graph methods.

        Args:
            query: Search query
            top_k: Number of results to return
            temporal_filter: Optional temporal filter
            alpha: Weight for vector vs graph (0=all graph, 1=all vector)
            rrf_k: K parameter for Reciprocal Rank Fusion
            retrieval_multiplier: Retrieve this many times top_k from each method

        Returns:
            List of search results with combined scores
        """
        logger.info(
            f"Hybrid search: '{query[:50]}...' "
            f"(top_k={top_k}, alpha={alpha or self.settings.hybrid_search_alpha})"
        )

        # Retrieve more results from each method to have better fusion
        retrieval_k = top_k * retrieval_multiplier

        # Execute both searches in parallel
        logger.info(f"Executing vector search (k={retrieval_k})...")
        vector_results = self.vector_search.search(
            query=query,
            top_k=retrieval_k,
            temporal_filter=temporal_filter,
        )

        logger.info(f"Executing graph search (k={retrieval_k})...")
        graph_results = self.graph_search.search(
            query=query,
            top_k=retrieval_k,
            temporal_filter=temporal_filter,
        )

        logger.info(
            f"Retrieved {len(vector_results)} vector results, "
            f"{len(graph_results)} graph results"
        )

        # Use alpha-weighted combination or RRF
        alpha_val = alpha if alpha is not None else self.settings.hybrid_search_alpha

        if alpha_val == 1.0:
            # Pure vector search
            return vector_results[:top_k]
        elif alpha_val == 0.0:
            # Pure graph search
            return graph_results[:top_k]
        else:
            # Combine using RRF or weighted scores
            k_param = rrf_k or self.settings.hybrid_search_k

            combined_results = self.reciprocal_rank_fusion(
                [vector_results, graph_results],
                k=k_param,
                weights=[alpha_val, 1.0 - alpha_val],
            )

            return combined_results[:top_k]

    def reciprocal_rank_fusion(
        self,
        result_lists: List[List[Dict]],
        k: int = 60,
        weights: Optional[List[float]] = None,
    ) -> List[Dict]:
        """
        Combine multiple result lists using Reciprocal Rank Fusion.

        RRF formula: RRF(d) = Σ weight_i * 1 / (k + rank_i(d))

        Args:
            result_lists: List of result lists from different search methods
            k: K parameter for RRF (typically 60)
            weights: Optional weights for each result list

        Returns:
            Combined and re-ranked results
        """
        if weights is None:
            weights = [1.0] * len(result_lists)

        if len(weights) != len(result_lists):
            raise ValueError("Number of weights must match number of result lists")

        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        chunk_data = {}

        for weight, results in zip(weights, result_lists):
            for rank, result in enumerate(results, start=1):
                chunk_id = result["chunk_id"]

                # RRF formula
                rrf_scores[chunk_id] += weight / (k + rank)

                # Store chunk data (from first occurrence)
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = result

        # Sort by RRF score
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True,
        )

        # Build result list
        combined_results = []
        for chunk_id in sorted_chunk_ids:
            result = chunk_data[chunk_id].copy()
            result["rrf_score"] = rrf_scores[chunk_id]
            result["hybrid_score"] = rrf_scores[chunk_id]  # Alias for compatibility
            combined_results.append(result)

        logger.info(f"RRF fusion produced {len(combined_results)} unique results")

        return combined_results

    def search_with_reranking(
        self,
        query: str,
        top_k: int = 10,
        temporal_filter: Optional[TemporalFilter] = None,
        retrieval_k: int = 50,
    ) -> List[Dict]:
        """
        Search with a two-stage retrieval and reranking approach.

        Args:
            query: Search query
            top_k: Final number of results to return
            temporal_filter: Optional temporal filter
            retrieval_k: Number of candidates to retrieve before reranking

        Returns:
            Reranked search results
        """
        logger.info(f"Hybrid search with reranking (retrieval_k={retrieval_k})")

        # Stage 1: Retrieve candidates
        candidates = self.search(
            query=query,
            top_k=retrieval_k,
            temporal_filter=temporal_filter,
        )

        # Stage 2: Rerank candidates
        # For now, we'll just use the hybrid scores
        # This could be enhanced with a reranking model
        reranked = sorted(
            candidates,
            key=lambda x: x.get("rrf_score", x.get("score", 0)),
            reverse=True,
        )

        logger.info(f"Reranking complete, returning top {top_k}")

        return reranked[:top_k]

    def explain_results(
        self,
        query: str,
        top_k: int = 5,
        temporal_filter: Optional[TemporalFilter] = None,
    ) -> Dict:
        """
        Run hybrid search and provide explanation of results.

        Args:
            query: Search query
            top_k: Number of results
            temporal_filter: Optional temporal filter

        Returns:
            Dictionary with results and explanations
        """
        logger.info("Running explainable hybrid search")

        # Get results from each method separately
        vector_results = self.vector_search.search(query, top_k, temporal_filter)
        graph_results = self.graph_search.search(query, top_k, temporal_filter)
        hybrid_results = self.search(query, top_k, temporal_filter)

        # Analyze overlap
        vector_ids = {r["chunk_id"] for r in vector_results}
        graph_ids = {r["chunk_id"] for r in graph_results}
        hybrid_ids = {r["chunk_id"] for r in hybrid_results}

        overlap_vector_graph = vector_ids & graph_ids
        unique_to_vector = vector_ids - graph_ids
        unique_to_graph = graph_ids - vector_ids

        explanation = {
            "query": query,
            "results": {
                "vector_only": len(unique_to_vector),
                "graph_only": len(unique_to_graph),
                "both": len(overlap_vector_graph),
                "hybrid_total": len(hybrid_ids),
            },
            "vector_results": vector_results,
            "graph_results": graph_results,
            "hybrid_results": hybrid_results,
            "analysis": {
                "vector_graph_overlap": list(overlap_vector_graph),
                "unique_to_vector": list(unique_to_vector),
                "unique_to_graph": list(unique_to_graph),
            },
        }

        logger.info(
            f"Explanation: {len(overlap_vector_graph)} overlap, "
            f"{len(unique_to_vector)} vector-only, "
            f"{len(unique_to_graph)} graph-only"
        )

        return explanation


# Global hybrid search instance
_hybrid_search: Optional[HybridSearch] = None


def get_hybrid_search() -> HybridSearch:
    """Get the global hybrid search instance."""
    global _hybrid_search
    if _hybrid_search is None:
        _hybrid_search = HybridSearch()
    return _hybrid_search


def search(
    query: str,
    top_k: int = 10,
    temporal_filter: Optional[TemporalFilter] = None,
) -> List[Dict]:
    """
    Convenience function for hybrid search.

    Args:
        query: Search query
        top_k: Number of results
        temporal_filter: Optional temporal filter

    Returns:
        List of search results
    """
    hs = get_hybrid_search()
    return hs.search(query, top_k, temporal_filter)
