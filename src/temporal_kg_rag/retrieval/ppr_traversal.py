"""
Personalized PageRank (PPR) based graph traversal for temporal knowledge graphs.

This module implements state-of-the-art PPR-based traversal with temporal awareness:
- Seeded PPR from query entities
- Temporal decay for older facts
- Bi-temporal filtering support

Based on research from:
- HippoRAG (NeurIPS 2024): Neurobiologically inspired long-term memory
- STAR-RAG (2024): Temporal retrieval via graph summarization
- EvePPR (2023): Fully dynamic personalized PageRank
"""

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from temporal_kg_rag.config.settings import get_settings
from temporal_kg_rag.graph.neo4j_client import Neo4jClient, get_neo4j_client
from temporal_kg_rag.models.temporal import TemporalFilter
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class PPRTraversal:
    """
    Personalized PageRank traversal for temporal knowledge graphs.

    Implements seeded PPR with temporal decay, allowing the retrieval
    to prioritize time-relevant information while still leveraging
    the graph structure for relevance scoring.
    """

    def __init__(
        self,
        client: Optional[Neo4jClient] = None,
        damping_factor: Optional[float] = None,
        max_iterations: Optional[int] = None,
        convergence_threshold: Optional[float] = None,
        temporal_decay: Optional[float] = None,
    ):
        """
        Initialize PPR traversal.

        Args:
            client: Optional Neo4j client
            damping_factor: Probability of following edges (1 - teleport probability)
            max_iterations: Maximum iterations for PPR convergence
            convergence_threshold: Convergence threshold for PPR
            temporal_decay: Decay factor for older facts (per year)
        """
        self.client = client or get_neo4j_client()
        self.settings = get_settings()

        self.damping_factor = (
            damping_factor
            if damping_factor is not None
            else self.settings.ppr_damping_factor
        )
        self.max_iterations = (
            max_iterations
            if max_iterations is not None
            else self.settings.ppr_max_iterations
        )
        self.convergence_threshold = (
            convergence_threshold
            if convergence_threshold is not None
            else self.settings.ppr_convergence_threshold
        )
        self.temporal_decay = (
            temporal_decay
            if temporal_decay is not None
            else self.settings.ppr_temporal_decay
        )

        logger.info(
            f"PPRTraversal initialized: damping={self.damping_factor}, "
            f"max_iter={self.max_iterations}, temporal_decay={self.temporal_decay}"
        )

    def search(
        self,
        seed_entities: List[str],
        top_k: int = 10,
        temporal_filter: Optional[TemporalFilter] = None,
        max_depth: Optional[int] = None,
    ) -> List[Dict]:
        """
        Perform PPR-based search starting from seed entities.

        Args:
            seed_entities: List of entity names to use as seeds
            top_k: Number of results to return
            temporal_filter: Optional temporal filter
            max_depth: Maximum traversal depth (default: settings.max_traversal_depth)

        Returns:
            List of chunks ranked by PPR score with temporal decay
        """
        if not seed_entities:
            logger.warning("No seed entities provided for PPR search")
            return []

        max_depth = max_depth or self.settings.max_traversal_depth

        logger.info(
            f"PPR search with {len(seed_entities)} seeds, "
            f"max_depth={max_depth}, top_k={top_k}"
        )

        # Step 1: Get seed entity IDs
        seed_nodes = self._get_seed_nodes(seed_entities, temporal_filter)

        if not seed_nodes:
            logger.warning("No valid seed nodes found in graph")
            return []

        logger.info(f"Found {len(seed_nodes)} seed nodes")

        # Step 2: Run PPR from seed nodes
        ppr_scores = self._compute_ppr(seed_nodes, max_depth, temporal_filter)

        if not ppr_scores:
            logger.warning("PPR computation returned no scores")
            return []

        logger.info(f"PPR computed scores for {len(ppr_scores)} entities")

        # Step 3: Get chunks connected to high-scoring entities
        results = self._get_chunks_by_entity_scores(
            ppr_scores, top_k, temporal_filter
        )

        logger.info(f"PPR search returned {len(results)} chunks")

        return results

    def _get_seed_nodes(
        self,
        entity_names: List[str],
        temporal_filter: Optional[TemporalFilter],
    ) -> Dict[str, float]:
        """
        Get seed node IDs for the given entity names.

        Args:
            entity_names: List of entity names
            temporal_filter: Optional temporal filter

        Returns:
            Dictionary of entity_id -> initial PPR score
        """
        # Query to find entities by name (fuzzy matching)
        query = """
        UNWIND $entity_names as name
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower(name)
           OR toLower(name) CONTAINS toLower(e.name)
        RETURN DISTINCT e.id as entity_id, e.name as entity_name
        """

        results = self.client.execute_read_transaction(
            query, {"entity_names": entity_names}
        )

        if not results:
            return {}

        # Distribute initial PPR score evenly among seeds
        seed_count = len(results)
        initial_score = 1.0 / seed_count

        return {r["entity_id"]: initial_score for r in results}

    def _compute_ppr(
        self,
        seed_nodes: Dict[str, float],
        max_depth: int,
        temporal_filter: Optional[TemporalFilter],
    ) -> Dict[str, float]:
        """
        Compute Personalized PageRank scores from seed nodes.

        Uses an approximation suitable for Neo4j execution:
        iterative propagation with temporal decay.

        Args:
            seed_nodes: Dictionary of seed entity_id -> initial score
            max_depth: Maximum traversal depth
            temporal_filter: Optional temporal filter

        Returns:
            Dictionary of entity_id -> PPR score
        """
        # For small graphs, we can compute PPR in Python
        # For large graphs, we use Neo4j's GDS library or approximation

        # Get subgraph around seed nodes
        subgraph = self._get_subgraph(seed_nodes, max_depth, temporal_filter)

        if not subgraph["nodes"]:
            return seed_nodes

        # Build adjacency structure
        adjacency: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        out_degree: Dict[str, int] = defaultdict(int)

        for edge in subgraph["edges"]:
            source = edge["source"]
            target = edge["target"]
            weight = edge.get("weight", 1.0) or 1.0

            adjacency[source].append((target, weight))
            out_degree[source] += 1

        # Initialize scores
        all_nodes = set(subgraph["nodes"])
        scores = {node: 0.0 for node in all_nodes}
        for seed_id, seed_score in seed_nodes.items():
            if seed_id in scores:
                scores[seed_id] = seed_score

        # Power iteration for PPR
        teleport_prob = 1.0 - self.damping_factor

        for iteration in range(self.max_iterations):
            new_scores = {node: 0.0 for node in all_nodes}

            # Propagate scores along edges
            for node in all_nodes:
                if out_degree[node] > 0:
                    share = self.damping_factor * scores[node] / out_degree[node]
                    for neighbor, weight in adjacency[node]:
                        if neighbor in new_scores:
                            new_scores[neighbor] += share * weight

            # Add teleport probability (back to seeds)
            for seed_id, seed_score in seed_nodes.items():
                if seed_id in new_scores:
                    new_scores[seed_id] += teleport_prob * seed_score

            # Normalize
            total = sum(new_scores.values())
            if total > 0:
                new_scores = {k: v / total for k, v in new_scores.items()}

            # Check convergence
            diff = sum(abs(new_scores[n] - scores[n]) for n in all_nodes)
            scores = new_scores

            if diff < self.convergence_threshold:
                logger.debug(f"PPR converged after {iteration + 1} iterations")
                break

        return scores

    def _get_subgraph(
        self,
        seed_nodes: Dict[str, float],
        max_depth: int,
        temporal_filter: Optional[TemporalFilter],
    ) -> Dict:
        """
        Get the subgraph around seed nodes for PPR computation.

        Args:
            seed_nodes: Seed node IDs
            max_depth: Maximum traversal depth
            temporal_filter: Optional temporal filter

        Returns:
            Dictionary with 'nodes' and 'edges' lists
        """
        seed_ids = list(seed_nodes.keys())

        # Build temporal clause for relationships
        rel_temporal_clause = ""
        if temporal_filter:
            clauses = []
            if temporal_filter.point_in_time:
                clauses.append(
                    "(r.valid_from IS NULL OR r.valid_from <= $point_in_time)"
                )
                clauses.append(
                    "(r.valid_to IS NULL OR r.valid_to > $point_in_time)"
                )
            if clauses:
                rel_temporal_clause = "WHERE " + " AND ".join(clauses)

        # Query to get subgraph
        # Use CALL (seed) { ... } syntax for Neo4j 5.x compatibility
        query = f"""
        MATCH (seed:Entity)
        WHERE seed.id IN $seed_ids

        CALL (seed) {{
            MATCH path = (seed)-[r*1..{max_depth}]-(neighbor:Entity)
            {rel_temporal_clause}
            RETURN neighbor, relationships(path) as rels
        }}

        WITH collect(DISTINCT neighbor) + collect(DISTINCT seed) as all_nodes
        UNWIND all_nodes as n
        WITH collect(DISTINCT n.id) as node_ids

        MATCH (e1:Entity)-[r]-(e2:Entity)
        WHERE e1.id IN node_ids AND e2.id IN node_ids

        RETURN
            node_ids as nodes,
            collect(DISTINCT {{
                source: e1.id,
                target: e2.id,
                type: type(r),
                weight: 1.0
            }}) as edges
        """

        params = {"seed_ids": seed_ids}
        if temporal_filter:
            params.update(temporal_filter.to_cypher_parameters())

        results = self.client.execute_read_transaction(query, params)

        if not results:
            return {"nodes": list(seed_nodes.keys()), "edges": []}

        return {
            "nodes": results[0].get("nodes", []),
            "edges": results[0].get("edges", []),
        }

    def _get_chunks_by_entity_scores(
        self,
        entity_scores: Dict[str, float],
        top_k: int,
        temporal_filter: Optional[TemporalFilter],
    ) -> List[Dict]:
        """
        Get chunks connected to entities, scored by PPR.

        Args:
            entity_scores: Dictionary of entity_id -> PPR score
            top_k: Number of results to return
            temporal_filter: Optional temporal filter

        Returns:
            List of chunk results
        """
        # Get top entities by PPR score
        top_entities = sorted(
            entity_scores.items(), key=lambda x: x[1], reverse=True
        )[:50]  # Consider top 50 entities for chunk retrieval

        entity_ids = [e[0] for e in top_entities]
        entity_score_map = dict(top_entities)

        # Build temporal clause
        temporal_clause = ""
        if temporal_filter:
            clause = temporal_filter.to_cypher_where_clause("c", "r")
            if clause and clause != "true":
                temporal_clause = f"AND {clause}"

        query = f"""
        MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity)
        WHERE e.id IN $entity_ids
        AND c.is_current = true
        {temporal_clause}

        WITH c, collect({{
            entity_id: e.id,
            entity_name: e.name,
            confidence: COALESCE(r.confidence, 1.0)
        }}) as mentioned_entities

        MATCH (c)<-[:HAS_CHUNK]-(doc:Document)

        RETURN
            c.id as chunk_id,
            c.text as text,
            c.chunk_index as chunk_index,
            c.created_at as created_at,
            doc.id as document_id,
            doc.title as document_title,
            mentioned_entities
        """

        params = {"entity_ids": entity_ids}
        if temporal_filter:
            params.update(temporal_filter.to_cypher_parameters())

        results = self.client.execute_read_transaction(query, params)

        # Score chunks by aggregated PPR scores of their entities
        chunk_scores: Dict[str, Dict] = {}

        for result in results:
            chunk_id = result["chunk_id"]

            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {
                    **result,
                    "ppr_score": 0.0,
                    "entity_contributions": [],
                }

            # Aggregate PPR scores from connected entities
            for entity in result["mentioned_entities"]:
                entity_id = entity["entity_id"]
                if entity_id in entity_score_map:
                    ppr_contribution = entity_score_map[entity_id]
                    confidence = entity.get("confidence", 1.0) or 1.0

                    chunk_scores[chunk_id]["ppr_score"] += ppr_contribution * confidence
                    chunk_scores[chunk_id]["entity_contributions"].append({
                        "entity_name": entity["entity_name"],
                        "ppr_score": ppr_contribution,
                    })

        # Apply temporal decay to chunk scores based on created_at
        if temporal_filter and temporal_filter.point_in_time:
            query_time = temporal_filter.point_in_time
            for chunk_id, chunk_data in chunk_scores.items():
                if chunk_data.get("created_at"):
                    age_years = (query_time - chunk_data["created_at"]).days / 365.0
                    decay = self.temporal_decay ** max(0, age_years)
                    chunk_data["ppr_score"] *= decay
                    chunk_data["temporal_decay_applied"] = decay

        # Sort by PPR score and return top_k
        sorted_chunks = sorted(
            chunk_scores.values(),
            key=lambda x: x["ppr_score"],
            reverse=True,
        )[:top_k]

        # Add score alias for compatibility with other search methods
        for chunk in sorted_chunks:
            chunk["score"] = chunk["ppr_score"]

        return sorted_chunks

    def search_with_query_entities(
        self,
        query: str,
        extracted_entities: List[str],
        top_k: int = 10,
        temporal_filter: Optional[TemporalFilter] = None,
    ) -> List[Dict]:
        """
        Convenience method that takes extracted entities directly.

        Args:
            query: Original query text (for logging)
            extracted_entities: Pre-extracted entity names
            top_k: Number of results
            temporal_filter: Optional temporal filter

        Returns:
            List of chunk results
        """
        logger.info(
            f"PPR search for query: '{query[:50]}...' "
            f"with entities: {extracted_entities}"
        )

        return self.search(
            seed_entities=extracted_entities,
            top_k=top_k,
            temporal_filter=temporal_filter,
        )


# Global PPR traversal instance
_ppr_traversal: Optional[PPRTraversal] = None


def get_ppr_traversal() -> PPRTraversal:
    """Get the global PPR traversal instance."""
    global _ppr_traversal
    if _ppr_traversal is None:
        _ppr_traversal = PPRTraversal()
    return _ppr_traversal


def ppr_search(
    seed_entities: List[str],
    top_k: int = 10,
    temporal_filter: Optional[TemporalFilter] = None,
) -> List[Dict]:
    """
    Convenience function for PPR-based search.

    Args:
        seed_entities: Entity names to use as seeds
        top_k: Number of results
        temporal_filter: Optional temporal filter

    Returns:
        List of search results
    """
    ppr = get_ppr_traversal()
    return ppr.search(seed_entities, top_k, temporal_filter)
