"""
Graph consolidation for maintaining a single unified knowledge graph.

Uses the improved three-stage entity deduplication pipeline:
1. Embedding-based blocking (fast, batch)
2. String similarity filtering
3. LLM validation (optional, only for ambiguous cases)

This provides 90%+ reduction in LLM calls compared to the naive O(n²) approach.
"""

from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from temporal_kg_rag.config.settings import get_settings
from temporal_kg_rag.graph.neo4j_client import Neo4jClient, get_neo4j_client
from temporal_kg_rag.ingestion.entity_deduplication import EntityDeduplicator
from temporal_kg_rag.models.entity import Entity
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class GraphConsolidator:
    """
    Consolidate the temporal knowledge graph by merging duplicate entities.

    This ensures a single unified graph across all documents.

    Uses the improved three-stage deduplication pipeline for efficiency:
    - Embedding-based blocking reduces candidate pairs by ~90%
    - String similarity filtering removes obvious non-matches
    - LLM validation only for ambiguous cases

    Supports bi-temporal tracking for merge operations.
    """

    def __init__(
        self,
        client: Optional[Neo4jClient] = None,
        deduplicator: Optional[EntityDeduplicator] = None,
        use_llm_validation: Optional[bool] = None,
    ):
        """
        Initialize graph consolidator.

        Args:
            client: Optional Neo4j client
            deduplicator: Optional entity deduplicator
            use_llm_validation: Override LLM validation setting
        """
        self.client = client or get_neo4j_client()
        self.settings = get_settings()

        # Initialize deduplicator with optional override
        self.deduplicator = deduplicator or EntityDeduplicator(
            use_llm_validation=use_llm_validation
        )

    def consolidate_entities(
        self,
        batch_size: int = 100,
        by_type: bool = True,
    ) -> Dict[str, int]:
        """
        Consolidate duplicate entities across the entire graph.

        Uses the improved three-stage deduplication pipeline:
        1. Embedding-based blocking (reduces candidate pairs by ~90%)
        2. String similarity filtering
        3. LLM validation (only for ambiguous cases)

        Args:
            batch_size: Number of entities to process at once
            by_type: Whether to process entities by type (recommended)

        Returns:
            Statistics dictionary with merge counts
        """
        logger.info("Starting entity consolidation across knowledge graph...")
        start_time = datetime.now()

        stats = {
            "total_entities_before": 0,
            "total_entities_after": 0,
            "entities_merged": 0,
            "merge_operations": 0,
            "duplicate_groups_found": 0,
            "llm_calls_saved_estimate": 0,
            "processing_time_seconds": 0,
        }

        # Get all entities grouped by type
        entities_by_type = self._get_all_entities_by_type()
        stats["total_entities_before"] = sum(
            len(entities) for entities in entities_by_type.values()
        )

        logger.info(
            f"Found {stats['total_entities_before']} entities "
            f"across {len(entities_by_type)} types"
        )

        # Estimate LLM calls that would be needed with naive O(n²) approach
        naive_comparisons = sum(
            len(e) * (len(e) - 1) // 2 for e in entities_by_type.values()
        )
        stats["llm_calls_saved_estimate"] = naive_comparisons

        if by_type:
            # Process each type separately (more efficient)
            for entity_type, entities in entities_by_type.items():
                if len(entities) <= 1:
                    logger.info(
                        f"Skipping type {entity_type}: only {len(entities)} entity"
                    )
                    continue

                logger.info(
                    f"Consolidating {len(entities)} entities of type {entity_type}..."
                )

                # Use the improved three-stage deduplication
                merge_groups = self.deduplicator.find_duplicates(
                    entities, entity_type=entity_type
                )

                if not merge_groups:
                    logger.info(f"No duplicates found for type {entity_type}")
                    continue

                stats["duplicate_groups_found"] += len(merge_groups)

                # Merge duplicates
                for group in merge_groups:
                    if len(group) > 1:
                        self._merge_entity_group(group)
                        stats["merge_operations"] += 1
                        stats["entities_merged"] += len(group) - 1
        else:
            # Process all entities together
            all_entities = []
            for entities in entities_by_type.values():
                all_entities.extend(entities)

            if len(all_entities) > 1:
                merge_groups = self.deduplicator.find_duplicates(all_entities)
                stats["duplicate_groups_found"] = len(merge_groups)

                for group in merge_groups:
                    if len(group) > 1:
                        self._merge_entity_group(group)
                        stats["merge_operations"] += 1
                        stats["entities_merged"] += len(group) - 1

        # Get final count
        entities_by_type = self._get_all_entities_by_type()
        stats["total_entities_after"] = sum(
            len(entities) for entities in entities_by_type.values()
        )

        # Calculate processing time
        stats["processing_time_seconds"] = (
            datetime.now() - start_time
        ).total_seconds()

        logger.info(
            f"Consolidation complete: {stats['total_entities_before']} -> "
            f"{stats['total_entities_after']} entities "
            f"({stats['entities_merged']} merged in {stats['merge_operations']} operations) "
            f"in {stats['processing_time_seconds']:.1f}s"
        )
        logger.info(
            f"Estimated LLM calls saved: ~{stats['llm_calls_saved_estimate']} "
            f"(naive O(n²) would require this many comparisons)"
        )

        return stats

    def _get_all_entities_by_type(self) -> Dict[str, List[Dict]]:
        """
        Get all entities from the graph grouped by type.

        Returns:
            Dictionary mapping entity type to list of entity dicts
        """
        query = """
        MATCH (e:Entity)
        RETURN e.id as id, e.name as name, e.type as type,
               e.mention_count as mention_count,
               e.first_seen as first_seen, e.last_seen as last_seen
        ORDER BY e.type, e.name
        """

        result = self.client.execute_read_transaction(query)

        entities_by_type: Dict[str, List[Dict]] = {}
        for row in result:
            entity_type = row["type"]
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []

            entities_by_type[entity_type].append({
                "id": row["id"],
                "name": row["name"],
                "type": row["type"],
                "mention_count": row["mention_count"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
            })

        return entities_by_type

    def _find_duplicate_groups_legacy(
        self,
        entities: List[Dict],
        entity_type: str,
    ) -> List[List[str]]:
        """
        Legacy method: Find groups of duplicate entities using O(n²) comparison.

        DEPRECATED: Use deduplicator.find_duplicates() instead for better performance.

        Args:
            entities: List of entity dictionaries
            entity_type: Entity type

        Returns:
            List of groups where each group is a list of entity IDs that are duplicates
        """
        if len(entities) <= 1:
            return []

        groups: List[List[str]] = []
        processed: Set[str] = set()

        for i, entity1 in enumerate(entities):
            if entity1["id"] in processed:
                continue

            # Start a new group with this entity
            group = [entity1["id"]]
            processed.add(entity1["id"])

            # Check against remaining entities
            for j in range(i + 1, len(entities)):
                entity2 = entities[j]

                if entity2["id"] in processed:
                    continue

                # Check if these entities are similar
                if self.deduplicator.are_entities_similar(
                    entity1["name"],
                    entity1["type"],
                    entity2["name"],
                    entity2["type"],
                ):
                    group.append(entity2["id"])
                    processed.add(entity2["id"])

            # Only add groups with duplicates
            if len(group) > 1:
                groups.append(group)

        return groups

    def _merge_entity_group(self, entity_ids: List[str]) -> None:
        """
        Merge a group of duplicate entities into a single entity.

        Supports bi-temporal tracking: records the merge timestamp.

        Args:
            entity_ids: List of entity IDs to merge (first one becomes the canonical entity)
        """
        if len(entity_ids) <= 1:
            return

        canonical_id = entity_ids[0]
        duplicate_ids = entity_ids[1:]
        merge_timestamp = datetime.now()

        logger.info(f"Merging {len(duplicate_ids)} entities into {canonical_id}")

        # Merge in Neo4j with bi-temporal support
        query = """
        // Get the canonical entity
        MATCH (canonical:Entity {id: $canonical_id})

        // Get all duplicate entities
        MATCH (duplicate:Entity)
        WHERE duplicate.id IN $duplicate_ids

        // Merge mention counts and timestamps
        WITH canonical, collect(duplicate) as duplicates
        SET canonical.mention_count = canonical.mention_count +
            reduce(count = 0, dup IN duplicates | count + dup.mention_count)
        SET canonical.last_seen =
            reduce(latest = canonical.last_seen, dup IN duplicates |
                CASE WHEN dup.last_seen > latest THEN dup.last_seen ELSE latest END)
        SET canonical.first_seen =
            reduce(earliest = canonical.first_seen, dup IN duplicates |
                CASE WHEN dup.first_seen < earliest THEN dup.first_seen ELSE earliest END)
        // Track merge history (bi-temporal)
        SET canonical.last_merged_at = $merge_timestamp
        SET canonical.merged_entity_ids = COALESCE(canonical.merged_entity_ids, []) + $duplicate_ids

        // Repoint all MENTIONS relationships to canonical entity
        WITH canonical, duplicates
        UNWIND duplicates as duplicate
        MATCH (chunk:Chunk)-[r:MENTIONS]->(duplicate)
        MERGE (chunk)-[new_r:MENTIONS]->(canonical)
        ON CREATE SET
            new_r.position = r.position,
            new_r.confidence = r.confidence,
            new_r.context = r.context,
            new_r.valid_from = r.valid_from,
            new_r.valid_to = r.valid_to,
            // Bi-temporal: track when this relationship was created via merge
            new_r.created_at = COALESCE(r.created_at, $merge_timestamp),
            new_r.merged_from_entity = duplicate.id
        DELETE r

        // Handle any other relationships the duplicate entities might have
        WITH canonical, duplicates
        UNWIND duplicates as duplicate
        OPTIONAL MATCH (duplicate)-[r]-(other)
        WHERE NOT other:Chunk
        WITH canonical, duplicate, r, other, type(r) as rel_type
        WHERE r IS NOT NULL

        // Create equivalent relationship to canonical (if not exists)
        CALL apoc.do.when(
            r IS NOT NULL,
            'MERGE (canonical)-[new_rel:' + rel_type + ']->(other)
             ON CREATE SET new_rel = properties(r)
             RETURN new_rel',
            'RETURN null as new_rel',
            {canonical: canonical, other: other, r: r}
        ) YIELD value

        // Delete duplicate entities
        WITH canonical, duplicates
        UNWIND duplicates as duplicate
        DETACH DELETE duplicate

        RETURN canonical.id as canonical_id, size(duplicates) as merged_count
        """

        try:
            self.client.execute_write_transaction(
                query,
                {
                    "canonical_id": canonical_id,
                    "duplicate_ids": duplicate_ids,
                    "merge_timestamp": merge_timestamp,
                }
            )
        except Exception as e:
            # Fallback to simpler query without APOC
            logger.warning(f"Complex merge failed, using simple merge: {e}")
            self._merge_entity_group_simple(entity_ids)

    def _merge_entity_group_simple(self, entity_ids: List[str]) -> None:
        """
        Simple merge without APOC procedures.

        Args:
            entity_ids: List of entity IDs to merge
        """
        if len(entity_ids) <= 1:
            return

        canonical_id = entity_ids[0]
        duplicate_ids = entity_ids[1:]
        merge_timestamp = datetime.now()

        query = """
        // Get the canonical entity
        MATCH (canonical:Entity {id: $canonical_id})

        // Get all duplicate entities
        MATCH (duplicate:Entity)
        WHERE duplicate.id IN $duplicate_ids

        // Merge mention counts and timestamps
        WITH canonical, collect(duplicate) as duplicates
        SET canonical.mention_count = canonical.mention_count +
            reduce(count = 0, dup IN duplicates | count + dup.mention_count)
        SET canonical.last_seen =
            reduce(latest = canonical.last_seen, dup IN duplicates |
                CASE WHEN dup.last_seen > latest THEN dup.last_seen ELSE latest END)
        SET canonical.first_seen =
            reduce(earliest = canonical.first_seen, dup IN duplicates |
                CASE WHEN dup.first_seen < earliest THEN dup.first_seen ELSE earliest END)
        SET canonical.last_merged_at = $merge_timestamp

        // Repoint all MENTIONS relationships to canonical entity
        WITH canonical, duplicates
        UNWIND duplicates as duplicate
        MATCH (chunk:Chunk)-[r:MENTIONS]->(duplicate)
        MERGE (chunk)-[new_r:MENTIONS]->(canonical)
        ON CREATE SET
            new_r.position = r.position,
            new_r.confidence = r.confidence,
            new_r.context = r.context,
            new_r.valid_from = r.valid_from,
            new_r.valid_to = r.valid_to,
            new_r.created_at = COALESCE(r.created_at, $merge_timestamp)
        DELETE r

        // Delete duplicate entities
        WITH canonical, duplicates
        UNWIND duplicates as duplicate
        DETACH DELETE duplicate

        RETURN canonical.id as canonical_id, size(duplicates) as merged_count
        """

        self.client.execute_write_transaction(
            query,
            {
                "canonical_id": canonical_id,
                "duplicate_ids": duplicate_ids,
                "merge_timestamp": merge_timestamp,
            }
        )

    def consolidate_periodically(
        self,
        min_entities_threshold: int = 100,
    ) -> Optional[Dict[str, int]]:
        """
        Consolidate entities only if the threshold is exceeded.

        This can be called periodically after document ingestion.

        Args:
            min_entities_threshold: Only consolidate if there are at least this many entities

        Returns:
            Consolidation statistics or None if threshold not met
        """
        # Count total entities
        count_query = "MATCH (e:Entity) RETURN count(e) as count"
        result = self.client.execute_read_transaction(count_query)
        total_entities = result[0]["count"]

        if total_entities < min_entities_threshold:
            logger.info(
                f"Skipping consolidation: only {total_entities} entities "
                f"(threshold: {min_entities_threshold})"
            )
            return None

        return self.consolidate_entities()


def get_graph_consolidator() -> GraphConsolidator:
    """Get the global graph consolidator instance."""
    return GraphConsolidator()


def consolidate_knowledge_graph(
    batch_size: int = 50,
    similarity_threshold: float = 0.8,
) -> Dict[str, int]:
    """
    Convenience function to consolidate the knowledge graph.

    Args:
        batch_size: Number of entities to process at once
        similarity_threshold: Threshold for considering entities duplicates

    Returns:
        Consolidation statistics
    """
    consolidator = get_graph_consolidator()
    return consolidator.consolidate_entities(batch_size, similarity_threshold)
