"""Document ingestion pipeline orchestration."""

from pathlib import Path
from typing import Dict, List, Optional

from temporal_kg_rag.embeddings.cache import EmbeddingCache, get_embedding_cache
from temporal_kg_rag.embeddings.generator import EmbeddingGenerator, get_embedding_generator
from temporal_kg_rag.graph.neo4j_client import Neo4jClient, get_neo4j_client
from temporal_kg_rag.graph.operations import GraphOperations, get_graph_operations
from temporal_kg_rag.ingestion.chunker import Chunker, chunk_text
from temporal_kg_rag.ingestion.document_loader import DocumentLoader, load_document
from temporal_kg_rag.ingestion.entity_extractor import EntityExtractor, get_entity_extractor
from temporal_kg_rag.models.chunk import Chunk
from temporal_kg_rag.models.document import Document
from temporal_kg_rag.models.entity import Entity, EntityMention
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """Orchestrate the complete document ingestion workflow."""

    def __init__(
        self,
        client: Optional[Neo4jClient] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        embedding_cache: Optional[EmbeddingCache] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        graph_operations: Optional[GraphOperations] = None,
        chunker: Optional[Chunker] = None,
    ):
        """
        Initialize ingestion pipeline.

        Args:
            client: Optional Neo4j client
            embedding_generator: Optional embedding generator
            embedding_cache: Optional embedding cache
            entity_extractor: Optional entity extractor
            graph_operations: Optional graph operations
            chunker: Optional chunker
        """
        self.client = client or get_neo4j_client()
        self.embedding_generator = embedding_generator or get_embedding_generator()
        self.embedding_cache = embedding_cache or get_embedding_cache()
        self.entity_extractor = entity_extractor or get_entity_extractor()
        self.graph_ops = graph_operations or get_graph_operations()
        self.chunker = chunker or Chunker()

        logger.info("Ingestion pipeline initialized")

    def ingest_document(
        self,
        file_path: str,
        title: Optional[str] = None,
        metadata: Optional[Dict] = None,
        extract_entities: bool = True,
        generate_embeddings: bool = True,
    ) -> Document:
        """
        Ingest a document into the knowledge graph.

        This is the main entry point for document ingestion. It orchestrates:
        1. Document loading
        2. Text chunking
        3. Embedding generation
        4. Entity extraction
        5. Graph storage

        Args:
            file_path: Path to the document file
            title: Optional document title
            metadata: Optional metadata dictionary
            extract_entities: Whether to extract entities
            generate_embeddings: Whether to generate embeddings

        Returns:
            Created Document object

        Raises:
            Exception: If ingestion fails
        """
        logger.info("=" * 60)
        logger.info(f"Starting document ingestion: {file_path}")
        logger.info("=" * 60)

        try:
            # Step 1: Load document
            logger.info("Step 1/6: Loading document...")
            document, text = self._load_document(file_path, title, metadata)

            # Step 2: Create document in graph
            logger.info("Step 2/6: Creating document node...")
            self.graph_ops.create_document(document)

            # Step 3: Chunk text
            logger.info("Step 3/6: Chunking text...")
            chunks = self._chunk_text(text, document.id)

            # Step 4: Generate embeddings
            if generate_embeddings:
                logger.info("Step 4/6: Generating embeddings...")
                self._generate_embeddings(chunks)
            else:
                logger.info("Step 4/6: Skipping embedding generation")

            # Step 5: Store chunks
            logger.info("Step 5/6: Storing chunks...")
            self.graph_ops.create_chunks_batch(chunks, document.id)

            # Step 6: Extract entities
            if extract_entities:
                logger.info("Step 6/6: Extracting entities...")
                entities, mentions = self._extract_entities(chunks)
                self.graph_ops.create_entities_and_mentions_batch(entities, mentions)
            else:
                logger.info("Step 6/6: Skipping entity extraction")

            logger.info("=" * 60)
            logger.info("Document ingestion completed successfully!")
            logger.info(f"  Document ID: {document.id}")
            logger.info(f"  Chunks created: {len(chunks)}")
            if extract_entities:
                logger.info(f"  Entities extracted: {len(entities)}")
                logger.info(f"  Entity mentions: {len(mentions)}")
            logger.info("=" * 60)

            return document

        except Exception as e:
            logger.error(f"Document ingestion failed: {e}", exc_info=True)
            raise

    def ingest_documents_batch(
        self,
        file_paths: List[str],
        extract_entities: bool = True,
        generate_embeddings: bool = True,
    ) -> List[Document]:
        """
        Ingest multiple documents.

        Args:
            file_paths: List of file paths
            extract_entities: Whether to extract entities
            generate_embeddings: Whether to generate embeddings

        Returns:
            List of created Document objects
        """
        logger.info(f"Starting batch ingestion of {len(file_paths)} documents")

        documents = []
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"\nProcessing document {i}/{len(file_paths)}")

            try:
                document = self.ingest_document(
                    file_path,
                    extract_entities=extract_entities,
                    generate_embeddings=generate_embeddings,
                )
                documents.append(document)
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
                # Continue with next document

        logger.info(
            f"\nBatch ingestion complete: "
            f"{len(documents)}/{len(file_paths)} documents processed successfully"
        )

        return documents

    def _load_document(
        self,
        file_path: str,
        title: Optional[str],
        metadata: Optional[Dict],
    ) -> tuple[Document, str]:
        """Load document from file."""
        loader = DocumentLoader()
        document, text = loader.load(file_path, title, metadata)

        logger.info(f"  Loaded: {document.title}")
        logger.info(f"  Type: {document.content_type}")
        logger.info(f"  Size: {len(text)} characters, {len(text.split())} words")

        return document, text

    def _chunk_text(self, text: str, document_id: str) -> List[Chunk]:
        """Chunk text into segments."""
        chunks = self.chunker.chunk_text(text, document_id, strategy="semantic")

        total_tokens = sum(chunk.token_count for chunk in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0

        logger.info(f"  Created {len(chunks)} chunks")
        logger.info(f"  Total tokens: {total_tokens:,}")
        logger.info(f"  Average tokens per chunk: {avg_tokens:.0f}")

        return chunks

    def _generate_embeddings(self, chunks: List[Chunk]) -> None:
        """Generate embeddings for chunks."""
        # Extract texts
        texts = [chunk.text for chunk in chunks]

        # Check cache
        cached_embeddings = self.embedding_cache.get_batch(
            texts,
            self.embedding_generator.model,
            self.embedding_generator.dimensions,
        )

        # Determine which texts need new embeddings
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            if i in cached_embeddings:
                chunks[i].embedding = cached_embeddings[i]
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        cache_hit_rate = len(cached_embeddings) / len(texts) * 100 if texts else 0
        logger.info(f"  Cache hit rate: {cache_hit_rate:.1f}%")

        # Generate new embeddings
        if texts_to_embed:
            logger.info(f"  Generating {len(texts_to_embed)} new embeddings...")

            # Estimate cost
            avg_tokens = sum(chunk.token_count for chunk in chunks) / len(chunks)
            estimated_cost = self.embedding_generator.estimate_cost(
                len(texts_to_embed),
                int(avg_tokens),
            )
            logger.info(f"  Estimated cost: ${estimated_cost:.4f}")

            # Generate embeddings
            new_embeddings = self.embedding_generator.generate_embeddings(texts_to_embed)

            # Assign embeddings to chunks and cache
            for i, embedding in zip(indices_to_embed, new_embeddings):
                chunks[i].embedding = embedding

            # Cache new embeddings
            self.embedding_cache.set_batch(
                texts_to_embed,
                new_embeddings,
                self.embedding_generator.model,
                self.embedding_generator.dimensions,
            )

            logger.info(f"  ✓ Embeddings generated and cached")
        else:
            logger.info(f"  ✓ All embeddings retrieved from cache")

    def _extract_entities(
        self,
        chunks: List[Chunk],
    ) -> tuple[Dict[str, Entity], List[EntityMention]]:
        """Extract entities from chunks."""
        entities, mentions = self.entity_extractor.extract_entities_from_chunks(chunks)

        # Log entity type distribution
        type_counts = {}
        for entity in entities.values():
            type_counts[entity.type] = type_counts.get(entity.type, 0) + 1

        logger.info(f"  Extracted {len(entities)} unique entities:")
        for entity_type, count in sorted(type_counts.items()):
            logger.info(f"    {entity_type}: {count}")
        logger.info(f"  Total mentions: {len(mentions)}")

        return entities, mentions

    def get_statistics(self) -> Dict:
        """
        Get ingestion statistics.

        Returns:
            Dictionary with statistics
        """
        stats = self.client.get_database_stats()

        cache_stats = self.embedding_cache.get_stats()
        stats["cache"] = cache_stats

        return stats


# Global pipeline instance
_pipeline: Optional[IngestionPipeline] = None


def get_ingestion_pipeline() -> IngestionPipeline:
    """Get the global ingestion pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


def ingest_document(
    file_path: str,
    title: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Document:
    """
    Convenience function to ingest a document.

    Args:
        file_path: Path to the document file
        title: Optional document title
        metadata: Optional metadata dictionary

    Returns:
        Created Document object
    """
    pipeline = get_ingestion_pipeline()
    return pipeline.ingest_document(file_path, title, metadata)
