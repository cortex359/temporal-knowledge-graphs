"""
Relation extraction using LLM to identify semantic relationships between entities.

Extracts relationships as temporal quadruples:
    (relationship, timestamp, source, target, description)

Relationship descriptions are free-form and comprehensive, following these guidelines:
1. The nature of the relationship (e.g., familial, professional, causal)
2. The impact or significance of the relationship on both entities
3. Any historical or contextual information relevant to the relationship
4. How the relationship evolved over time (if applicable)
5. Any notable events or actions that resulted from this relationship
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import httpx

from temporal_kg_rag.config.settings import get_settings
from temporal_kg_rag.models.entity import Entity, EntityRelationship
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)

# Relationship extraction guidelines for the LLM prompt
RELATIONSHIP_GUIDELINES = """
Relationship Guidelines:
1. Make sure relationship descriptions are detailed and comprehensive.
   Use multiple complete sentences for each point below:
   a). The nature of the relationship (e.g., familial, professional, causal)
   b). The impact or significance of the relationship on both entities
   c). Any historical or contextual information relevant to the relationship
   d). How the relationship evolved over time (if applicable)
   e). Any notable events or actions that resulted from this relationship
"""


class RelationExtractor:
    """Extract semantic relationships between entities using LLM as temporal quadruples."""

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize relation extractor.

        Args:
            api_base: LiteLLM API base URL
            api_key: LiteLLM API key
            model: LLM model name
        """
        settings = get_settings()
        self.api_base = (api_base or settings.litellm_api_base).rstrip("/")
        self.api_key = api_key or settings.litellm_api_key
        # Use relation_extraction_model if set, otherwise fall back to entity_extraction_model
        self.model = model or settings.relation_extraction_model or settings.entity_extraction_model

        self.client = httpx.Client(timeout=120.0)

        logger.info(f"RelationExtractor initialized: model={self.model}")

    def __del__(self):
        """Close HTTP client on cleanup."""
        if hasattr(self, "client"):
            self.client.close()

    def extract_relations(
        self,
        text: str,
        entities: List[Entity],
        chunk_id: Optional[str] = None,
    ) -> List[EntityRelationship]:
        """
        Extract semantic relationships between entities from text as temporal quadruples.

        Args:
            text: Text containing the entities
            entities: List of entities found in the text
            chunk_id: Optional chunk ID for tracking source

        Returns:
            List of EntityRelationship objects (temporal quadruples)
        """
        if not text.strip() or len(entities) < 2:
            return []

        try:
            # Call LLM to extract relations
            raw_relations = self._call_llm_for_relations(text, entities)

            # Build entity lookup by name for ID resolution
            entity_lookup = {}
            for entity in entities:
                entity_lookup[entity.name.lower()] = entity
                # Also add without common suffixes/prefixes
                simplified = self._simplify_entity_name(entity.name)
                if simplified:
                    entity_lookup[simplified.lower()] = entity

            # Convert to EntityRelationship objects
            relationships = []
            for rel_data in raw_relations:
                relationship = self._parse_relation(rel_data, entity_lookup, chunk_id)
                if relationship:
                    relationships.append(relationship)

            logger.info(
                f"Extracted {len(relationships)} relationships from "
                f"{len(entities)} entities"
            )

            return relationships

        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            return []

    def _call_llm_for_relations(
        self,
        text: str,
        entities: List[Entity],
    ) -> List[Dict]:
        """
        Call LLM to extract relations between entities as temporal quadruples.

        Args:
            text: Source text
            entities: Entities found in the text

        Returns:
            List of relation dictionaries
        """
        # Format entities for prompt
        entity_list = "\n".join([
            f"- {e.name} ({e.type})" for e in entities
        ])

        prompt = f"""Analyze the text and extract all meaningful relationships between the listed entities.

ENTITIES FOUND:
{entity_list}

TEXT:
{text[:4000]}

{RELATIONSHIP_GUIDELINES}

Extract relationships as temporal quadruples. For each relationship:
1. "relationship": A short, descriptive label for the relationship (e.g., "founded", "collaborated with", "acquired", "invested in", "led research at")
2. "timestamp": The date/time when this relationship was established or is most relevant (ISO format YYYY-MM-DD, or null if unknown)
3. "source": The source entity name (exactly as listed above)
4. "target": The target entity name (exactly as listed above)
5. "description": A comprehensive, multi-paragraph description following ALL the guidelines above

The description MUST include multiple complete sentences covering:
- The nature of the relationship (professional, familial, causal, organizational, etc.)
- The impact and significance for both entities involved
- Historical and contextual background information
- How the relationship evolved or changed over time
- Notable events, outcomes, or consequences of this relationship

Format as JSON array:
[
  {{
    "relationship": "short relationship label",
    "timestamp": "YYYY-MM-DD or null",
    "source": "Source Entity Name",
    "target": "Target Entity Name",
    "description": "Detailed multi-sentence description covering all guideline points..."
  }}
]

IMPORTANT RULES:
- Only extract relationships explicitly stated or strongly implied in the text
- The relationship label should be concise (1-4 words)
- The description should be comprehensive (3-6 sentences minimum)
- Include temporal information when available
- Do NOT invent relationships not supported by the text
- Use entity names EXACTLY as they appear in the entity list

Return ONLY valid JSON array, no extra text.

JSON:"""

        try:
            response = self.client.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert at knowledge graph construction and relationship extraction. "
                                "Your task is to identify and describe relationships between entities with rich, "
                                "comprehensive descriptions that capture the full context and significance of each relationship. "
                                "Always follow the relationship guidelines precisely."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 4000,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )

            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"].strip()

            # Parse JSON from response
            relations = self._parse_json_response(content)

            return relations

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error in relation extraction: "
                f"{e.response.status_code} - {e.response.text}"
            )
            return []
        except Exception as e:
            logger.error(f"Error calling LLM for relations: {e}")
            return []

    def _parse_json_response(self, content: str) -> List[Dict]:
        """Parse JSON response with error recovery."""
        # Remove markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            relations = json.loads(content)
            if isinstance(relations, list):
                return relations
            return []
        except json.JSONDecodeError:
            # Try to recover
            try:
                start = content.find('[')
                end = content.rfind(']')
                if start != -1 and end > start:
                    json_str = content[start:end + 1]
                    # Fix common issues
                    json_str = re.sub(r',\s*]', ']', json_str)
                    json_str = re.sub(r',\s*}', '}', json_str)
                    return json.loads(json_str)
            except Exception:
                pass

            logger.warning(f"Failed to parse relation JSON: {content[:200]}")
            return []

    def _parse_relation(
        self,
        rel_data: Dict,
        entity_lookup: Dict[str, Entity],
        chunk_id: Optional[str],
    ) -> Optional[EntityRelationship]:
        """
        Parse a relation dictionary into an EntityRelationship (temporal quadruple).

        Args:
            rel_data: Raw relation data from LLM
            entity_lookup: Mapping of entity names to Entity objects
            chunk_id: Source chunk ID

        Returns:
            EntityRelationship or None if parsing fails
        """
        try:
            source_name = rel_data.get("source", "").strip()
            target_name = rel_data.get("target", "").strip()
            relationship = rel_data.get("relationship", "").strip()
            description = rel_data.get("description", "").strip()

            if not source_name or not target_name or not relationship:
                return None

            # Validate description meets guidelines (should be substantial)
            if len(description) < 50:
                logger.debug(
                    f"Description too short for relationship '{relationship}': {description[:50]}"
                )
                # Still create the relationship but log the issue

            # Look up entities
            source_entity = self._find_entity(source_name, entity_lookup)
            target_entity = self._find_entity(target_name, entity_lookup)

            if not source_entity or not target_entity:
                logger.debug(
                    f"Could not resolve entities: {source_name} -> {target_name}"
                )
                return None

            # Parse timestamp
            timestamp = self._parse_date(rel_data.get("timestamp"))

            # Create relationship as temporal quadruple
            entity_relationship = EntityRelationship(
                source_entity_id=source_entity.id,
                source_entity_name=source_entity.name,
                target_entity_id=target_entity.id,
                target_entity_name=target_entity.name,
                relationship=relationship,
                description=description,
                timestamp=timestamp,
                valid_from=timestamp or datetime.now(),
                confidence=0.85,  # LLM extractions get 0.85 confidence
                source_chunks=[chunk_id] if chunk_id else [],
            )

            return entity_relationship

        except Exception as e:
            logger.debug(f"Failed to parse relation: {e}")
            return None

    def _find_entity(
        self,
        name: str,
        entity_lookup: Dict[str, Entity],
    ) -> Optional[Entity]:
        """Find an entity by name with fuzzy matching."""
        name_lower = name.lower().strip()

        # Exact match
        if name_lower in entity_lookup:
            return entity_lookup[name_lower]

        # Simplified match
        simplified = self._simplify_entity_name(name)
        if simplified and simplified.lower() in entity_lookup:
            return entity_lookup[simplified.lower()]

        # Partial match (entity name contains or is contained in search name)
        for key, entity in entity_lookup.items():
            if name_lower in key or key in name_lower:
                return entity

        return None

    @staticmethod
    def _simplify_entity_name(name: str) -> str:
        """Simplify entity name by removing common prefixes/suffixes."""
        name = name.strip()
        # Remove common suffixes
        suffixes = [" Inc.", " Inc", " Corp.", " Corp", " LLC", " Ltd.", " Ltd",
                    " GmbH", " AG", " Co.", " Company", " Corporation"]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]

        # Remove common prefixes
        prefixes = ["The ", "Dr. ", "Mr. ", "Mrs. ", "Ms. ", "Prof. "]
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]

        return name.strip()

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
        """Parse a date string into datetime."""
        if not date_str or str(date_str).lower() in ["null", "none", "", "unknown"]:
            return None

        try:
            # Try various formats
            for fmt in ["%Y-%m-%d", "%Y-%m", "%Y", "%d/%m/%Y", "%m/%d/%Y"]:
                try:
                    return datetime.strptime(str(date_str), fmt)
                except ValueError:
                    continue
        except Exception:
            pass

        return None

    def extract_relations_from_chunks(
        self,
        chunks_with_entities: List[Tuple["Chunk", List[Entity]]],
    ) -> List[EntityRelationship]:
        """
        Extract relations from multiple chunks.

        Args:
            chunks_with_entities: List of (chunk, entities) tuples

        Returns:
            List of all extracted relationships (temporal quadruples)
        """
        from temporal_kg_rag.models.chunk import Chunk

        all_relationships = []

        for chunk, entities in chunks_with_entities:
            if len(entities) >= 2:
                relationships = self.extract_relations(
                    text=chunk.text,
                    entities=entities,
                    chunk_id=chunk.id,
                )
                all_relationships.extend(relationships)

        # Deduplicate relationships
        deduplicated = self._deduplicate_relationships(all_relationships)

        logger.info(
            f"Extracted {len(deduplicated)} unique relationships from "
            f"{len(chunks_with_entities)} chunks"
        )

        return deduplicated

    def _deduplicate_relationships(
        self,
        relationships: List[EntityRelationship],
    ) -> List[EntityRelationship]:
        """
        Deduplicate relationships, merging source chunks and keeping best description.

        Args:
            relationships: List of relationships (may have duplicates)

        Returns:
            Deduplicated list with merged source chunks
        """
        unique = {}

        for rel in relationships:
            # Create a key based on source, target, and relationship label
            key = (
                rel.source_entity_id,
                rel.target_entity_id,
                rel.relationship.lower(),
            )

            if key in unique:
                # Merge source chunks
                existing = unique[key]
                for chunk_id in rel.source_chunks:
                    if chunk_id not in existing.source_chunks:
                        existing.source_chunks.append(chunk_id)
                # Keep higher confidence
                existing.confidence = max(existing.confidence, rel.confidence)
                # Keep longer/more detailed description
                if len(rel.description) > len(existing.description):
                    existing.description = rel.description
            else:
                unique[key] = rel

        return list(unique.values())


# Global extractor instance
_relation_extractor: Optional[RelationExtractor] = None


def get_relation_extractor() -> RelationExtractor:
    """Get the global relation extractor instance."""
    global _relation_extractor
    if _relation_extractor is None:
        _relation_extractor = RelationExtractor()
    return _relation_extractor


def extract_relations(
    text: str,
    entities: List[Entity],
    chunk_id: Optional[str] = None,
) -> List[EntityRelationship]:
    """
    Convenience function to extract relations as temporal quadruples.

    Args:
        text: Source text
        entities: Entities in the text
        chunk_id: Optional chunk ID

    Returns:
        List of extracted relationships (temporal quadruples)
    """
    extractor = get_relation_extractor()
    return extractor.extract_relations(text, entities, chunk_id)
