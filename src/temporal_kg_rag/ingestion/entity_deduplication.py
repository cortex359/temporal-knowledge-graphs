"""
Embedding-based entity deduplication with three-stage pipeline.

This module implements state-of-the-art entity deduplication using:
1. Embedding-based blocking (fast, batch) - reduces candidate pairs by ~90%
2. String similarity filtering (medium) - further filters obvious non-matches
3. LLM validation (expensive, selective) - only for ambiguous cases

Based on research from:
- Zep/Graphiti (2025): Three-stage entity resolution
- EAGER: Embedding-Assisted Entity Resolution for Knowledge Graphs
- Pre-trained Embeddings for Entity Resolution (VLDB)
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import httpx
import numpy as np

from temporal_kg_rag.config.settings import get_settings
from temporal_kg_rag.embeddings.generator import get_embedding_generator
from temporal_kg_rag.models.entity import Entity
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CandidatePair:
    """A candidate pair of entities for deduplication."""

    entity1_id: str
    entity1_name: str
    entity2_id: str
    entity2_name: str
    entity_type: str
    embedding_similarity: float
    string_similarity: float = 0.0
    llm_confidence: Optional[float] = None
    is_duplicate: Optional[bool] = None


class EntityDeduplicator:
    """
    Three-stage entity deduplication pipeline.

    Stage 1: Embedding-based blocking
        - Batch embed all entity names
        - Use cosine similarity to find candidate pairs
        - Reduces search space by ~90%

    Stage 2: String similarity filtering
        - Apply Jaro-Winkler similarity
        - Filter out obvious non-matches

    Stage 3: LLM validation (optional)
        - Only for ambiguous cases where embedding + string similarity disagree
        - Batch LLM calls for efficiency
    """

    def __init__(
        self,
        embedding_threshold: Optional[float] = None,
        string_threshold: Optional[float] = None,
        llm_threshold: Optional[float] = None,
        use_llm_validation: Optional[bool] = None,
    ):
        """
        Initialize entity deduplicator.

        Args:
            embedding_threshold: Cosine similarity threshold for blocking
            string_threshold: Jaro-Winkler threshold for filtering
            llm_threshold: Confidence threshold for LLM validation
            use_llm_validation: Whether to use LLM for final validation
        """
        self.settings = get_settings()
        self.embedding_generator = get_embedding_generator()

        self.embedding_threshold = (
            embedding_threshold
            if embedding_threshold is not None
            else self.settings.dedup_embedding_threshold
        )
        self.string_threshold = (
            string_threshold
            if string_threshold is not None
            else self.settings.dedup_string_threshold
        )
        self.llm_threshold = (
            llm_threshold
            if llm_threshold is not None
            else self.settings.dedup_llm_threshold
        )
        self.use_llm_validation = (
            use_llm_validation
            if use_llm_validation is not None
            else self.settings.dedup_use_llm_validation
        )

        self.client = httpx.Client(timeout=60.0)

        # Cache for entity embeddings
        self._embedding_cache: Dict[str, List[float]] = {}

        logger.info(
            f"EntityDeduplicator initialized: "
            f"embedding_threshold={self.embedding_threshold}, "
            f"string_threshold={self.string_threshold}, "
            f"llm_validation={self.use_llm_validation}"
        )

    def are_entities_similar(
        self,
        entity1_name: str,
        entity1_type: str,
        entity2_name: str,
        entity2_type: str,
    ) -> bool:
        """
        Check if two entities are semantically similar.

        This is the main entry point for single-pair comparison.
        Uses a fast path for obvious matches/non-matches.

        Args:
            entity1_name: Name of first entity
            entity1_type: Type of first entity
            entity2_name: Name of second entity
            entity2_type: Type of second entity

        Returns:
            True if entities are similar, False otherwise
        """
        # Fast path: exact match
        if entity1_name.lower().strip() == entity2_name.lower().strip():
            return True

        # Fast path: different types
        if entity1_type != entity2_type:
            return False

        # Check string similarity first (fast)
        string_sim = self._jaro_winkler_similarity(entity1_name, entity2_name)
        if string_sim >= 0.95:
            return True
        if string_sim < 0.5:
            return False

        # Check embedding similarity
        emb1 = self._get_entity_embedding(entity1_name)
        emb2 = self._get_entity_embedding(entity2_name)
        embedding_sim = self._cosine_similarity(emb1, emb2)

        if embedding_sim >= self.embedding_threshold and string_sim >= self.string_threshold:
            return True

        # Ambiguous case - use LLM if enabled
        if self.use_llm_validation and embedding_sim >= 0.7:
            return self._llm_validate_pair(
                entity1_name, entity1_type, entity2_name, entity2_type
            )

        return False

    def find_duplicates(
        self,
        entities: List[Dict],
        entity_type: Optional[str] = None,
    ) -> List[List[str]]:
        """
        Find groups of duplicate entities using the three-stage pipeline.

        Args:
            entities: List of entity dictionaries with 'id', 'name', 'type'
            entity_type: Optional filter for specific entity type

        Returns:
            List of groups where each group is a list of duplicate entity IDs
        """
        if len(entities) <= 1:
            return []

        # Filter by type if specified
        if entity_type:
            entities = [e for e in entities if e.get("type") == entity_type]

        if len(entities) <= 1:
            return []

        logger.info(f"Finding duplicates among {len(entities)} entities...")

        # Stage 1: Embedding-based blocking
        logger.info("Stage 1: Embedding-based blocking...")
        candidate_pairs = self._embedding_blocking(entities)
        logger.info(f"Found {len(candidate_pairs)} candidate pairs from embeddings")

        if not candidate_pairs:
            return []

        # Stage 2: String similarity filtering
        logger.info("Stage 2: String similarity filtering...")
        filtered_pairs = self._string_similarity_filter(candidate_pairs)
        logger.info(f"Filtered to {len(filtered_pairs)} pairs after string similarity")

        if not filtered_pairs:
            return []

        # Stage 3: LLM validation (optional, only for ambiguous cases)
        if self.use_llm_validation:
            logger.info("Stage 3: LLM validation for ambiguous cases...")
            validated_pairs = self._llm_validate_batch(filtered_pairs)
        else:
            # Without LLM, accept pairs with high combined similarity
            validated_pairs = [
                p for p in filtered_pairs
                if p.embedding_similarity >= self.embedding_threshold
                and p.string_similarity >= self.string_threshold
            ]

        logger.info(f"Final validated pairs: {len(validated_pairs)}")

        # Cluster duplicates using Union-Find
        duplicate_groups = self._cluster_duplicates(validated_pairs, entities)

        logger.info(f"Found {len(duplicate_groups)} duplicate groups")

        return duplicate_groups

    def _embedding_blocking(self, entities: List[Dict]) -> List[CandidatePair]:
        """
        Stage 1: Find candidate pairs using embedding similarity.

        Args:
            entities: List of entity dictionaries

        Returns:
            List of candidate pairs with embedding similarity
        """
        # Batch generate embeddings for all entities
        entity_names = [e["name"] for e in entities]
        embeddings = self._batch_get_embeddings(entity_names)

        # Build embedding matrix for efficient similarity computation
        embedding_matrix = np.array(embeddings)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embedding_matrix / norms

        # Compute similarity matrix
        similarity_matrix = np.dot(normalized, normalized.T)

        # Find pairs above threshold
        candidate_pairs = []
        max_candidates = self.settings.dedup_max_candidates_per_entity

        for i in range(len(entities)):
            # Get similarities for this entity
            similarities = similarity_matrix[i]

            # Find top candidates (excluding self)
            indices = np.argsort(similarities)[::-1]

            candidates_for_entity = 0
            for j in indices:
                if i >= j:  # Skip self and already processed pairs
                    continue

                sim = similarities[j]
                if sim < self.embedding_threshold * 0.8:  # Allow some slack for blocking
                    break

                # Only consider same type
                if entities[i].get("type") != entities[j].get("type"):
                    continue

                candidate_pairs.append(
                    CandidatePair(
                        entity1_id=entities[i]["id"],
                        entity1_name=entities[i]["name"],
                        entity2_id=entities[j]["id"],
                        entity2_name=entities[j]["name"],
                        entity_type=entities[i].get("type", "UNKNOWN"),
                        embedding_similarity=float(sim),
                    )
                )

                candidates_for_entity += 1
                if candidates_for_entity >= max_candidates:
                    break

        return candidate_pairs

    def _string_similarity_filter(
        self, pairs: List[CandidatePair]
    ) -> List[CandidatePair]:
        """
        Stage 2: Filter pairs using string similarity.

        Args:
            pairs: Candidate pairs from embedding blocking

        Returns:
            Filtered pairs with string similarity scores
        """
        filtered = []

        for pair in pairs:
            string_sim = self._jaro_winkler_similarity(
                pair.entity1_name, pair.entity2_name
            )
            pair.string_similarity = string_sim

            # Accept if both similarities are high enough
            # or if embedding is very high (> 0.95)
            if (
                string_sim >= self.string_threshold * 0.8
                or pair.embedding_similarity >= 0.95
            ):
                filtered.append(pair)

        return filtered

    def _llm_validate_batch(self, pairs: List[CandidatePair]) -> List[CandidatePair]:
        """
        Stage 3: Validate ambiguous pairs using LLM.

        Only validates pairs where embedding and string similarity disagree
        or are in the ambiguous range.

        Args:
            pairs: Candidate pairs to validate

        Returns:
            Validated pairs marked as duplicates or not
        """
        # Separate clear duplicates from ambiguous cases
        clear_duplicates = []
        ambiguous_pairs = []

        for pair in pairs:
            # High confidence duplicate
            if (
                pair.embedding_similarity >= 0.95
                and pair.string_similarity >= 0.9
            ):
                pair.is_duplicate = True
                pair.llm_confidence = 1.0
                clear_duplicates.append(pair)
            # High confidence non-duplicate
            elif (
                pair.embedding_similarity < 0.7
                and pair.string_similarity < 0.6
            ):
                pair.is_duplicate = False
                pair.llm_confidence = 0.0
            # Ambiguous - need LLM
            else:
                ambiguous_pairs.append(pair)

        if not ambiguous_pairs:
            return clear_duplicates

        logger.info(f"Validating {len(ambiguous_pairs)} ambiguous pairs with LLM...")

        # Batch LLM validation
        batch_size = 5  # Process 5 pairs per LLM call
        for i in range(0, len(ambiguous_pairs), batch_size):
            batch = ambiguous_pairs[i : i + batch_size]
            self._llm_validate_pairs_batch(batch)

        # Filter to confirmed duplicates
        validated = clear_duplicates + [
            p for p in ambiguous_pairs
            if p.is_duplicate and p.llm_confidence >= self.llm_threshold
        ]

        return validated

    def _llm_validate_pairs_batch(self, pairs: List[CandidatePair]) -> None:
        """
        Validate multiple pairs in a single LLM call.

        Args:
            pairs: Pairs to validate (mutated in place)
        """
        if not pairs:
            return

        # Build prompt for batch validation
        pairs_text = "\n".join(
            f"{i+1}. \"{p.entity1_name}\" vs \"{p.entity2_name}\" (Type: {p.entity_type})"
            for i, p in enumerate(pairs)
        )

        prompt = f"""Determine if these entity pairs refer to the same real-world entity.
Consider variations in spelling, abbreviations, acronyms, and common aliases.

Entity pairs:
{pairs_text}

Return a JSON array with one object per pair:
[
    {{"pair": 1, "same": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}},
    ...
]

Examples of same entities:
- "OpenAI" and "Open AI" -> same (spacing)
- "GPT-4" and "GPT 4" -> same (punctuation)
- "MIT" and "Massachusetts Institute of Technology" -> same (acronym)
- "IBM" and "International Business Machines" -> same (acronym)
- "Google" and "Alphabet" -> NOT same (different entities, parent company)

Return ONLY valid JSON."""

        try:
            response = self.client.post(
                f"{self.settings.litellm_api_base}/chat/completions",
                json={
                    "model": self.settings.entity_extraction_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 500,
                },
                headers={"Authorization": f"Bearer {self.settings.litellm_api_key}"},
            )
            response.raise_for_status()

            content = response.json()["choices"][0]["message"]["content"].strip()

            # Parse JSON response
            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                results = json.loads(json_match.group())

                for result in results:
                    pair_idx = result.get("pair", 0) - 1
                    if 0 <= pair_idx < len(pairs):
                        pairs[pair_idx].is_duplicate = result.get("same", False)
                        pairs[pair_idx].llm_confidence = result.get("confidence", 0.0)

        except Exception as e:
            logger.error(f"LLM batch validation failed: {e}")
            # Fall back to combined similarity threshold
            for pair in pairs:
                combined_score = (
                    pair.embedding_similarity * 0.6 + pair.string_similarity * 0.4
                )
                pair.is_duplicate = combined_score >= 0.8
                pair.llm_confidence = combined_score

    def _llm_validate_pair(
        self,
        entity1_name: str,
        entity1_type: str,
        entity2_name: str,
        entity2_type: str,
    ) -> bool:
        """
        Validate a single pair using LLM (legacy interface).

        Args:
            entity1_name: Name of first entity
            entity1_type: Type of first entity
            entity2_name: Name of second entity
            entity2_type: Type of second entity

        Returns:
            True if entities are the same, False otherwise
        """
        pair = CandidatePair(
            entity1_id="temp1",
            entity1_name=entity1_name,
            entity2_id="temp2",
            entity2_name=entity2_name,
            entity_type=entity1_type,
            embedding_similarity=0.8,
        )

        self._llm_validate_pairs_batch([pair])

        return pair.is_duplicate or False

    def _cluster_duplicates(
        self, pairs: List[CandidatePair], entities: List[Dict]
    ) -> List[List[str]]:
        """
        Cluster duplicate pairs into groups using Union-Find.

        Args:
            pairs: Validated duplicate pairs
            entities: Original entity list

        Returns:
            List of duplicate groups (each group is a list of entity IDs)
        """
        # Build Union-Find structure
        parent = {e["id"]: e["id"] for e in entities}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union duplicate pairs
        for pair in pairs:
            if pair.is_duplicate:
                union(pair.entity1_id, pair.entity2_id)

        # Group entities by their root
        groups: Dict[str, List[str]] = {}
        for e in entities:
            root = find(e["id"])
            if root not in groups:
                groups[root] = []
            groups[root].append(e["id"])

        # Return only groups with duplicates (size > 1)
        return [g for g in groups.values() if len(g) > 1]

    def _get_entity_embedding(self, name: str) -> List[float]:
        """Get embedding for an entity name, using cache."""
        cache_key = name.lower().strip()
        if cache_key not in self._embedding_cache:
            self._embedding_cache[cache_key] = self.embedding_generator.generate_embedding(name)
        return self._embedding_cache[cache_key]

    def _batch_get_embeddings(self, names: List[str]) -> List[List[float]]:
        """Get embeddings for multiple entity names, using cache."""
        # Find which names need embeddings
        cache_keys = [n.lower().strip() for n in names]
        missing_indices = [
            i for i, key in enumerate(cache_keys) if key not in self._embedding_cache
        ]

        # Generate missing embeddings in batch
        if missing_indices:
            missing_names = [names[i] for i in missing_indices]
            new_embeddings = self.embedding_generator.generate_embeddings(missing_names)

            for i, idx in enumerate(missing_indices):
                self._embedding_cache[cache_keys[idx]] = new_embeddings[i]

        # Return all embeddings in order
        return [self._embedding_cache[key] for key in cache_keys]

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def _jaro_winkler_similarity(s1: str, s2: str) -> float:
        """
        Compute Jaro-Winkler similarity between two strings.

        Returns a value between 0 and 1, where 1 means identical strings.
        """
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()

        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)

        if len1 == 0 or len2 == 0:
            return 0.0

        # Jaro distance
        match_distance = max(len1, len2) // 2 - 1
        if match_distance < 0:
            match_distance = 0

        s1_matches = [False] * len1
        s2_matches = [False] * len2

        matches = 0
        transpositions = 0

        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)

            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        jaro = (
            matches / len1 + matches / len2 + (matches - transpositions / 2) / matches
        ) / 3

        # Winkler modification
        prefix = 0
        for i in range(min(len1, len2, 4)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break

        return jaro + prefix * 0.1 * (1 - jaro)

    def deduplicate_entities(
        self,
        entities: Dict[str, Entity],
    ) -> Dict[str, Entity]:
        """
        Deduplicate entities by finding and merging similar ones.

        This is the legacy interface for backward compatibility.

        Args:
            entities: Dictionary of entity_id -> Entity

        Returns:
            Deduplicated dictionary of entities
        """
        if len(entities) <= 1:
            return entities

        logger.info(f"Deduplicating {len(entities)} entities...")

        # Convert to list format for processing
        entity_list = [
            {"id": eid, "name": e.name, "type": e.type}
            for eid, e in entities.items()
        ]

        # Find duplicate groups
        duplicate_groups = self.find_duplicates(entity_list)

        if not duplicate_groups:
            logger.info("No duplicates found")
            return entities

        # Merge duplicates
        merged_entities: Dict[str, Entity] = dict(entities)
        entity_mapping: Dict[str, str] = {}  # old_id -> canonical_id

        for group in duplicate_groups:
            # First entity in group becomes canonical
            canonical_id = group[0]
            canonical_entity = merged_entities[canonical_id]

            for dup_id in group[1:]:
                dup_entity = merged_entities[dup_id]

                # Merge into canonical
                canonical_entity.mention_count += dup_entity.mention_count
                canonical_entity.last_seen = max(
                    canonical_entity.last_seen, dup_entity.last_seen
                )
                canonical_entity.first_seen = min(
                    canonical_entity.first_seen, dup_entity.first_seen
                )

                # Track mapping and remove duplicate
                entity_mapping[dup_id] = canonical_id
                del merged_entities[dup_id]

        logger.info(
            f"Deduplication complete: {len(entities)} -> {len(merged_entities)} entities "
            f"({len(entities) - len(merged_entities)} duplicates merged)"
        )

        return merged_entities

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")


def get_entity_deduplicator() -> EntityDeduplicator:
    """Get the global entity deduplicator instance."""
    return EntityDeduplicator()
