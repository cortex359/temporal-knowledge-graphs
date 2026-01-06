"""Entity extraction using spaCy NER."""

from typing import Dict, List, Optional, Set, Tuple

import spacy
from spacy.language import Language

from temporal_kg_rag.config.settings import get_settings
from temporal_kg_rag.models.entity import Entity, EntityMention
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class EntityExtractor:
    """Extract named entities from text using spaCy."""

    # Map spaCy entity types to our simplified types
    ENTITY_TYPE_MAPPING = {
        "PERSON": "PERSON",
        "PER": "PERSON",
        "ORG": "ORGANIZATION",
        "ORGANIZATION": "ORGANIZATION",
        "GPE": "LOCATION",  # Geopolitical entity
        "LOC": "LOCATION",
        "LOCATION": "LOCATION",
        "FAC": "FACILITY",
        "FACILITY": "FACILITY",
        "DATE": "DATE",
        "TIME": "TIME",
        "MONEY": "MONEY",
        "PERCENT": "PERCENT",
        "PRODUCT": "PRODUCT",
        "EVENT": "EVENT",
        "WORK_OF_ART": "WORK",
        "LAW": "LAW",
        "LANGUAGE": "LANGUAGE",
        "NORP": "GROUP",  # Nationalities, religious/political groups
    }

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize entity extractor.

        Args:
            model_name: spaCy model name (defaults to settings)
        """
        settings = get_settings()
        self.model_name = model_name or settings.spacy_model

        try:
            self.nlp: Language = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            logger.error(
                f"spaCy model '{self.model_name}' not found. "
                f"Please install it with: python -m spacy download {self.model_name}"
            )
            raise

    def extract_entities(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        min_confidence: float = 0.5,
    ) -> Tuple[List[Entity], List[EntityMention]]:
        """
        Extract entities from text.

        Args:
            text: Text to analyze
            chunk_id: Optional chunk ID for entity mentions
            min_confidence: Minimum confidence threshold

        Returns:
            Tuple of (entities, entity mentions)
        """
        if not text.strip():
            return [], []

        # Process text with spaCy
        doc = self.nlp(text)

        # Track entities and mentions
        entity_dict: Dict[Tuple[str, str], Entity] = {}  # (name, type) -> Entity
        mentions: List[EntityMention] = []

        # Extract entities
        for ent in doc.ents:
            # Map entity type
            entity_type = self.ENTITY_TYPE_MAPPING.get(ent.label_, "OTHER")

            # Normalize entity name
            entity_name = self._normalize_entity_name(ent.text)

            if not entity_name:
                continue

            # Create or update entity
            entity_key = (entity_name, entity_type)

            if entity_key not in entity_dict:
                entity = Entity(
                    name=entity_name,
                    type=entity_type,
                    mention_count=1,
                    metadata={
                        "original_label": ent.label_,
                    }
                )
                entity_dict[entity_key] = entity
            else:
                entity_dict[entity_key].increment_mention_count()

            # Create entity mention if chunk_id is provided
            if chunk_id:
                # Get surrounding context (10 words before and after)
                context = self._get_context(text, ent.start_char, ent.end_char)

                mention = EntityMention(
                    entity_id=entity_dict[entity_key].id,
                    entity_name=entity_name,
                    entity_type=entity_type,
                    chunk_id=chunk_id,
                    position=ent.start_char,
                    confidence=1.0,  # spaCy doesn't provide confidence scores
                    context=context,
                )
                mentions.append(mention)

        entities = list(entity_dict.values())

        logger.info(
            f"Extracted {len(entities)} unique entities "
            f"({len(mentions)} total mentions) from text ({len(text)} chars)"
        )

        # Log entity type distribution
        type_counts = {}
        for entity in entities:
            type_counts[entity.type] = type_counts.get(entity.type, 0) + 1

        logger.debug(f"Entity types: {type_counts}")

        return entities, mentions

    def extract_entities_from_chunks(
        self,
        chunks: List,  # List of Chunk objects
    ) -> Tuple[Dict[str, Entity], List[EntityMention]]:
        """
        Extract entities from multiple chunks.

        Args:
            chunks: List of Chunk objects

        Returns:
            Tuple of (entity dict by ID, all mentions)
        """
        all_entity_dict: Dict[str, Entity] = {}
        all_mentions: List[EntityMention] = []

        # Track entity by (name, type) for deduplication
        entity_lookup: Dict[Tuple[str, str], str] = {}  # (name, type) -> entity_id

        for chunk in chunks:
            entities, mentions = self.extract_entities(
                chunk.text,
                chunk_id=chunk.id,
            )

            # Merge entities
            for entity in entities:
                entity_key = (entity.name, entity.type)

                if entity_key in entity_lookup:
                    # Entity already exists, update it
                    existing_id = entity_lookup[entity_key]
                    existing_entity = all_entity_dict[existing_id]
                    existing_entity.increment_mention_count(entity.mention_count)
                    existing_entity.update_last_seen()

                    # Update mention entity_ids to point to the existing entity
                    for mention in mentions:
                        if mention.entity_id == entity.id:
                            mention.entity_id = existing_id
                else:
                    # New entity
                    entity_lookup[entity_key] = entity.id
                    all_entity_dict[entity.id] = entity

            all_mentions.extend(mentions)

        logger.info(
            f"Extracted {len(all_entity_dict)} unique entities from {len(chunks)} chunks"
        )

        return all_entity_dict, all_mentions

    def _normalize_entity_name(self, name: str) -> str:
        """
        Normalize entity name.

        Args:
            name: Raw entity name

        Returns:
            Normalized name
        """
        # Strip whitespace
        name = name.strip()

        # Remove extra whitespace
        name = " ".join(name.split())

        # Remove leading/trailing punctuation
        name = name.strip(".,;:!?")

        return name

    def _get_context(
        self,
        text: str,
        start: int,
        end: int,
        window: int = 50,
    ) -> str:
        """
        Get surrounding context for an entity mention.

        Args:
            text: Full text
            start: Entity start position
            end: Entity end position
            window: Number of characters before and after

        Returns:
            Context string
        """
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)

        context = text[context_start:context_end]

        # Add ellipsis if truncated
        if context_start > 0:
            context = "..." + context
        if context_end < len(text):
            context = context + "..."

        return context

    def get_entity_types(self) -> List[str]:
        """Get list of supported entity types."""
        return sorted(set(self.ENTITY_TYPE_MAPPING.values()))


# Global extractor instance
_extractor: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    """Get the global entity extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor


def extract_entities(
    text: str,
    chunk_id: Optional[str] = None,
) -> Tuple[List[Entity], List[EntityMention]]:
    """
    Convenience function to extract entities.

    Args:
        text: Text to analyze
        chunk_id: Optional chunk ID

    Returns:
        Tuple of (entities, entity mentions)
    """
    extractor = get_entity_extractor()
    return extractor.extract_entities(text, chunk_id)
