"""Entity extraction using LLM via LiteLLM."""

import json
from typing import Dict, List, Optional, Set, Tuple

import httpx

from temporal_kg_rag.config.settings import get_settings
from temporal_kg_rag.models.entity import Entity, EntityMention
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class EntityExtractor:
    """Extract named entities from text using LLM."""

    # Supported entity types
    ENTITY_TYPES = [
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "FACILITY",
        "DATE",
        "TIME",
        "MONEY",
        "PERCENT",
        "PRODUCT",
        "EVENT",
        "WORK",
        "LAW",
        "LANGUAGE",
        "GROUP",
    ]

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize entity extractor.

        Args:
            api_base: LiteLLM API base URL (defaults to settings)
            api_key: LiteLLM API key (defaults to settings)
            model: LLM model name (defaults to settings)
        """
        settings = get_settings()
        self.api_base = (api_base or settings.litellm_api_base).rstrip("/")
        self.api_key = api_key or settings.litellm_api_key
        self.model = model or settings.default_llm_model

        # HTTP client with timeout
        self.client = httpx.Client(timeout=60.0)

        logger.info(
            f"EntityExtractor initialized with LLM: model={self.model}, "
            f"api_base={self.api_base}"
        )

    def __del__(self):
        """Close HTTP client on cleanup."""
        if hasattr(self, "client"):
            self.client.close()

    def extract_entities(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        min_confidence: float = 0.5,
    ) -> Tuple[List[Entity], List[EntityMention]]:
        """
        Extract entities from text using LLM.

        Args:
            text: Text to analyze
            chunk_id: Optional chunk ID for entity mentions
            min_confidence: Minimum confidence threshold (not used with LLM)

        Returns:
            Tuple of (entities, entity mentions)
        """
        if not text.strip():
            return [], []

        try:
            # Call LLM to extract entities
            extracted_entities = self._call_llm_for_entities(text)

            # Track entities and mentions
            entity_dict: Dict[Tuple[str, str], Entity] = {}  # (name, type) -> Entity
            mentions: List[EntityMention] = []

            # Process extracted entities
            for ent_data in extracted_entities:
                entity_name = ent_data.get("name", "").strip()
                entity_type = ent_data.get("type", "OTHER").upper()
                context = ent_data.get("context", "")

                if not entity_name:
                    continue

                # Normalize entity type
                if entity_type not in self.ENTITY_TYPES:
                    entity_type = "OTHER"

                # Create or update entity
                entity_key = (entity_name, entity_type)

                if entity_key not in entity_dict:
                    entity = Entity(
                        name=entity_name,
                        type=entity_type,
                        mention_count=0,
                    )
                    entity_dict[entity_key] = entity
                else:
                    entity = entity_dict[entity_key]

                entity.mention_count += 1

                # Create entity mention
                if chunk_id:
                    # Find position of entity in text (simple approach)
                    position = text.lower().find(entity_name.lower())
                    if position == -1:
                        position = 0

                    mention = EntityMention(
                        entity_id=entity.id,
                        entity_name=entity_name,
                        entity_type=entity_type,
                        chunk_id=chunk_id,
                        position=position,
                        confidence=0.8,  # LLM extractions get 0.8 confidence
                        context=context or text[:200],  # First 200 chars as context
                    )
                    mentions.append(mention)

            entities = list(entity_dict.values())

            logger.info(
                f"Extracted {len(entities)} unique entities "
                f"({len(mentions)} mentions) from text"
            )

            return entities, mentions

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            # Return empty results on error rather than failing
            return [], []

    def _call_llm_for_entities(self, text: str) -> List[Dict]:
        """
        Call LLM to extract entities from text.

        Args:
            text: Text to analyze

        Returns:
            List of entity dictionaries with name, type, and context
        """
        # Prepare prompt for entity extraction
        # Keep context brief to avoid JSON issues
        prompt = f"""Extract named entities from the text below. Return ONLY a valid JSON array.

Entity types: {', '.join(self.ENTITY_TYPES)}

Text:
{text[:2000]}

Format (keep context under 50 chars):
[
  {{"name": "Entity Name", "type": "ENTITY_TYPE", "context": "brief context"}},
  ...
]

IMPORTANT: Return ONLY the JSON array, no extra text. Keep all context fields under 50 characters.

JSON:"""

        try:
            # Call LiteLLM chat completions endpoint
            response = self.client.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert at named entity recognition. Extract entities accurately and return them in valid JSON format.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2000,  # Increased to avoid truncation
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )

            response.raise_for_status()
            data = response.json()

            # Extract content from response
            content = data["choices"][0]["message"]["content"].strip()

            # Try to parse JSON from response
            # Sometimes LLM might include markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Parse JSON
            entities = json.loads(content)

            if not isinstance(entities, list):
                logger.warning("LLM returned non-list response, wrapping in list")
                entities = [entities] if isinstance(entities, dict) else []

            return entities

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error calling LLM for entity extraction: "
                f"{e.response.status_code} - {e.response.text}"
            )
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response content (first 500 chars): {content[:500]}")

            # Try to extract JSON array manually as fallback
            try:
                # Find first [ and last ]
                start = content.find('[')
                end = content.rfind(']')

                # If no closing bracket, try to complete the JSON
                if start != -1:
                    if end == -1 or end < start:
                        # Find last complete object
                        last_brace = content.rfind('}')
                        if last_brace > start:
                            json_str = content[start:last_brace+1] + ']'
                        else:
                            json_str = '[]'
                    else:
                        json_str = content[start:end+1]

                    # Try to fix common JSON issues
                    json_str = json_str.replace("'", '"')  # Single quotes to double
                    json_str = json_str.replace(',]', ']')  # Trailing comma
                    json_str = json_str.replace(',}', '}')  # Trailing comma

                    entities = json.loads(json_str)
                    logger.info(f"Successfully recovered {len(entities)} entities from malformed JSON")
                    return entities
            except Exception as recovery_error:
                logger.debug(f"JSON recovery also failed: {recovery_error}")
                pass

            return []
        except Exception as e:
            logger.error(f"Error calling LLM for entity extraction: {e}")
            return []

    def extract_entities_from_chunks(
        self,
        chunks: List["Chunk"],
        use_deduplication: bool = True,
    ) -> Tuple[Dict[str, Entity], List[EntityMention]]:
        """
        Extract entities from a list of chunks with LLM-based deduplication.

        Args:
            chunks: List of Chunk objects
            use_deduplication: Whether to use LLM-based entity deduplication

        Returns:
            Tuple of (entity_dict, mentions_list) where entity_dict maps entity IDs to Entity objects
        """
        from temporal_kg_rag.models.chunk import Chunk

        all_entities = {}
        all_mentions = []

        for chunk in chunks:
            entities, mentions = self.extract_entities(chunk.text, chunk.id)

            # Merge entities
            for entity in entities:
                if entity.id in all_entities:
                    # Update existing entity
                    existing = all_entities[entity.id]
                    existing.mention_count += entity.mention_count
                    existing.last_seen = max(existing.last_seen, entity.last_seen)
                else:
                    all_entities[entity.id] = entity

            # Collect mentions
            all_mentions.extend(mentions)

        # Apply LLM-based deduplication
        if use_deduplication and len(all_entities) > 1:
            from temporal_kg_rag.ingestion.entity_deduplication import get_entity_deduplicator

            deduplicator = get_entity_deduplicator()
            all_entities = deduplicator.deduplicate_entities(all_entities)

        return all_entities, all_mentions

    def extract_entities_batch(
        self,
        texts: List[str],
        chunk_ids: Optional[List[str]] = None,
    ) -> List[Tuple[List[Entity], List[EntityMention]]]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of texts to analyze
            chunk_ids: Optional list of chunk IDs

        Returns:
            List of (entities, mentions) tuples
        """
        if chunk_ids is None:
            chunk_ids = [None] * len(texts)

        results = []
        for text, chunk_id in zip(texts, chunk_ids):
            entities, mentions = self.extract_entities(text, chunk_id)
            results.append((entities, mentions))

        return results

    def _normalize_entity_name(self, name: str) -> str:
        """
        Normalize entity name.

        Args:
            name: Raw entity name

        Returns:
            Normalized entity name
        """
        # Remove extra whitespace
        name = " ".join(name.split())

        # Remove trailing punctuation
        name = name.rstrip(".,;:!?")

        return name.strip()

    def get_entity_types(self) -> List[str]:
        """Get list of supported entity types."""
        return self.ENTITY_TYPES.copy()


# Global extractor instance
_extractor: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    """Get the global entity extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor


def reset_entity_extractor():
    """Reset the global entity extractor instance."""
    global _extractor
    if _extractor is not None and hasattr(_extractor, "client"):
        _extractor.client.close()
    _extractor = None


def extract_entities(text: str) -> Tuple[List[Entity], List[EntityMention]]:
    """
    Convenience function to extract entities from text.

    Args:
        text: Text to analyze

    Returns:
        Tuple of (entities, mentions)
    """
    extractor = get_entity_extractor()
    return extractor.extract_entities(text)
