"""Graph database CRUD operations."""

from datetime import datetime
from typing import Dict, List, Optional

from temporal_kg_rag.graph.neo4j_client import Neo4jClient, get_neo4j_client
from temporal_kg_rag.models.chunk import Chunk
from temporal_kg_rag.models.document import Document
from temporal_kg_rag.models.entity import Entity, EntityMention, EntityRelationship
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class GraphOperations:
    """CRUD operations for the temporal knowledge graph."""

    def __init__(self, client: Optional[Neo4jClient] = None):
        """
        Initialize graph operations.

        Args:
            client: Optional Neo4j client
        """
        self.client = client or get_neo4j_client()

    # ===== Document Operations =====

    def create_document(self, document: Document) -> Document:
        """
        Create a document node in the graph.

        Args:
            document: Document to create

        Returns:
            Created document
        """
        # Get document properties (already flattened by to_neo4j_dict)
        doc_props = document.to_neo4j_dict()

        query = """
        CREATE (d:Document)
        SET d = $props
        RETURN d
        """

        result = self.client.execute_write_transaction(
            query,
            {"props": doc_props},
        )

        logger.info(f"Created document: {document.title} (ID: {document.id})")

        return document

    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        query = """
        MATCH (d:Document {id: $document_id})
        RETURN d
        """

        result = self.client.execute_read_transaction(
            query,
            {"document_id": document_id},
        )

        if result:
            return Document.from_neo4j_dict(result[0]["d"])

        return None

    # ===== Chunk Operations =====

    def create_chunk(self, chunk: Chunk) -> Chunk:
        """
        Create a chunk node in the graph.

        Args:
            chunk: Chunk to create

        Returns:
            Created chunk
        """
        query = """
        CREATE (c:Chunk)
        SET c = $props
        RETURN c
        """

        result = self.client.execute_write_transaction(
            query,
            {"props": chunk.to_neo4j_dict()},
        )

        logger.debug(f"Created chunk: {chunk.id} (index: {chunk.chunk_index})")

        return chunk

    def link_chunk_to_document(self, chunk_id: str, document_id: str) -> None:
        """
        Create HAS_CHUNK relationship between document and chunk.

        Args:
            chunk_id: Chunk ID
            document_id: Document ID
        """
        query = """
        MATCH (d:Document {id: $document_id})
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (d)-[r:HAS_CHUNK {created_at: $created_at}]->(c)
        RETURN r
        """

        self.client.execute_write_transaction(query, {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "created_at": datetime.now(),
        })

        logger.debug(f"Linked chunk {chunk_id} to document {document_id}")

    def get_document_chunks(
        self,
        document_id: str,
        current_only: bool = True,
    ) -> List[Chunk]:
        """
        Get all chunks for a document.

        Args:
            document_id: Document ID
            current_only: If True, only return current versions

        Returns:
            List of chunks
        """
        query = """
        MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        """

        if current_only:
            query += "WHERE c.is_current = true\n"

        query += """
        RETURN c
        ORDER BY c.chunk_index
        """

        results = self.client.execute_read_transaction(
            query,
            {"document_id": document_id},
        )

        chunks = [Chunk.from_neo4j_dict(r["c"]) for r in results]

        return chunks

    # ===== Entity Operations =====

    def create_or_update_entity(self, entity: Entity) -> Entity:
        """
        Create entity or update if it already exists.

        Args:
            entity: Entity to create/update

        Returns:
            Created or updated entity
        """
        query = """
        MERGE (e:Entity {name: $name, type: $type})
        ON CREATE SET
            e.id = $id,
            e.first_seen = $first_seen,
            e.last_seen = $last_seen,
            e.mention_count = $mention_count,
            e.metadata = $metadata
        ON MATCH SET
            e.last_seen = $last_seen,
            e.mention_count = e.mention_count + $mention_count,
            e.metadata = $metadata
        RETURN e
        """

        result = self.client.execute_write_transaction(
            query,
            entity.to_neo4j_dict(),
        )

        logger.debug(f"Created/updated entity: {entity.name} ({entity.type})")

        return entity

    def create_entity_mention(self, mention: EntityMention) -> None:
        """
        Create MENTIONS relationship between chunk and entity.

        Args:
            mention: Entity mention information
        """
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MATCH (e:Entity {id: $entity_id})
        MERGE (c)-[m:MENTIONS {
            position: $position,
            confidence: $confidence,
            context: $context,
            valid_from: $valid_from,
            valid_to: $valid_to
        }]->(e)
        RETURN m
        """

        params = {
            "chunk_id": mention.chunk_id,
            "entity_id": mention.entity_id,
            **mention.to_neo4j_relationship_dict(),
        }

        self.client.execute_write_transaction(query, params)

        logger.debug(
            f"Created mention: {mention.entity_name} in chunk {mention.chunk_id}"
        )

    def get_chunk_entities(self, chunk_id: str) -> List[Entity]:
        """
        Get all entities mentioned in a chunk.

        Args:
            chunk_id: Chunk ID

        Returns:
            List of entities
        """
        query = """
        MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS]->(e:Entity)
        RETURN DISTINCT e
        """

        results = self.client.execute_read_transaction(
            query,
            {"chunk_id": chunk_id},
        )

        entities = [Entity.from_neo4j_dict(r["e"]) for r in results]

        return entities

    def get_entity_chunks(
        self,
        entity_id: str,
        current_only: bool = True,
    ) -> List[Chunk]:
        """
        Get all chunks that mention an entity.

        Args:
            entity_id: Entity ID
            current_only: If True, only return current chunk versions

        Returns:
            List of chunks
        """
        query = """
        MATCH (e:Entity {id: $entity_id})<-[:MENTIONS]-(c:Chunk)
        """

        if current_only:
            query += "WHERE c.is_current = true\n"

        query += """
        RETURN c
        ORDER BY c.created_at DESC
        """

        results = self.client.execute_read_transaction(
            query,
            {"entity_id": entity_id},
        )

        chunks = [Chunk.from_neo4j_dict(r["c"]) for r in results]

        return chunks

    # ===== Batch Operations =====

    def create_chunks_batch(
        self,
        chunks: List[Chunk],
        document_id: str,
    ) -> None:
        """
        Create multiple chunks and link them to a document.

        Args:
            chunks: List of chunks to create
            document_id: Document ID to link to
        """
        # Create chunks
        query = """
        UNWIND $chunks AS chunk_data
        CREATE (c:Chunk)
        SET c = chunk_data
        WITH c
        MATCH (d:Document {id: $document_id})
        MERGE (d)-[:HAS_CHUNK {created_at: datetime()}]->(c)
        """

        chunks_data = [chunk.to_neo4j_dict() for chunk in chunks]

        self.client.execute_write_transaction(query, {
            "chunks": chunks_data,
            "document_id": document_id,
        })

        logger.info(f"Created {len(chunks)} chunks for document {document_id}")

    def create_entities_and_mentions_batch(
        self,
        entities: Dict[str, Entity],
        mentions: List[EntityMention],
    ) -> None:
        """
        Create multiple entities and their mentions.

        Args:
            entities: Dictionary of entity ID to Entity
            mentions: List of entity mentions
        """
        # Create/update entities
        entity_query = """
        UNWIND $entities AS entity_data
        MERGE (e:Entity {name: entity_data.name, type: entity_data.type})
        ON CREATE SET
            e.id = entity_data.id,
            e.first_seen = entity_data.first_seen,
            e.last_seen = entity_data.last_seen,
            e.mention_count = entity_data.mention_count,
            e.metadata = entity_data.metadata
        ON MATCH SET
            e.last_seen = entity_data.last_seen,
            e.mention_count = e.mention_count + entity_data.mention_count
        """

        entities_data = [entity.to_neo4j_dict() for entity in entities.values()]

        self.client.execute_write_transaction(entity_query, {
            "entities": entities_data,
        })

        logger.info(f"Created/updated {len(entities)} entities")

        # Create mentions
        if mentions:
            mention_query = """
            UNWIND $mentions AS mention_data
            MATCH (c:Chunk {id: mention_data.chunk_id})
            MATCH (e:Entity {id: mention_data.entity_id})
            MERGE (c)-[m:MENTIONS]->(e)
            SET m.position = mention_data.position,
                m.confidence = mention_data.confidence,
                m.context = mention_data.context,
                m.valid_from = mention_data.valid_from
            SET m.valid_to = CASE WHEN mention_data.valid_to IS NOT NULL
                                 THEN mention_data.valid_to
                                 ELSE null END
            """

            mentions_data = []
            for mention in mentions:
                mention_dict = mention.to_neo4j_relationship_dict()
                mention_dict["chunk_id"] = mention.chunk_id
                mention_dict["entity_id"] = mention.entity_id
                mentions_data.append(mention_dict)

            self.client.execute_write_transaction(mention_query, {
                "mentions": mentions_data,
            })

            logger.info(f"Created {len(mentions)} entity mentions")

    def create_entity_relationship(self, relationship: EntityRelationship) -> None:
        """
        Create a semantic relationship between two entities as a temporal quadruple.

        Temporal quadruple: (relationship, timestamp, source, target, description)

        Args:
            relationship: EntityRelationship to create
        """
        query = """
        MATCH (source:Entity {id: $source_id})
        MATCH (target:Entity {id: $target_id})
        MERGE (source)-[r:RELATES_TO {
            relationship: $relationship,
            timestamp: $timestamp,
            description: $description,
            confidence: $confidence,
            valid_from: $valid_from,
            source_chunks: $source_chunks
        }]->(target)
        SET r.valid_to = $valid_to
        RETURN r
        """

        params = {
            "source_id": relationship.source_entity_id,
            "target_id": relationship.target_entity_id,
            "relationship": relationship.relationship,
            "timestamp": relationship.timestamp,
            "description": relationship.description,
            "confidence": relationship.confidence,
            "valid_from": relationship.valid_from,
            "valid_to": relationship.valid_to,
            "source_chunks": relationship.source_chunks,
        }

        self.client.execute_write_transaction(query, params)

        logger.debug(
            f"Created relationship: {relationship.source_entity_name} "
            f"-[{relationship.relationship}]-> "
            f"{relationship.target_entity_name}"
        )

    def create_entity_relationships_batch(
        self,
        relationships: List[EntityRelationship],
    ) -> None:
        """
        Create multiple entity relationships in batch as temporal quadruples.

        Temporal quadruple: (relationship, timestamp, source, target, description)

        Args:
            relationships: List of EntityRelationship objects
        """
        if not relationships:
            return

        query = """
        UNWIND $relationships AS rel
        MATCH (source:Entity {id: rel.source_id})
        MATCH (target:Entity {id: rel.target_id})
        MERGE (source)-[r:RELATES_TO {relationship: rel.relationship}]->(target)
        SET r.timestamp = rel.timestamp,
            r.description = rel.description,
            r.confidence = rel.confidence,
            r.valid_from = rel.valid_from,
            r.valid_to = rel.valid_to,
            r.source_chunks = COALESCE(r.source_chunks, []) + rel.source_chunks,
            r.created_at = COALESCE(r.created_at, datetime())
        """

        rel_data = []
        for rel in relationships:
            rel_data.append({
                "source_id": rel.source_entity_id,
                "target_id": rel.target_entity_id,
                "relationship": rel.relationship,
                "timestamp": rel.timestamp,
                "description": rel.description,
                "confidence": rel.confidence,
                "valid_from": rel.valid_from,
                "valid_to": rel.valid_to,
                "source_chunks": rel.source_chunks,
            })

        self.client.execute_write_transaction(query, {"relationships": rel_data})

        logger.info(f"Created {len(relationships)} entity relationships")

    def get_entity_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        relationship_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get relationships for an entity as temporal quadruples.

        Args:
            entity_id: Entity ID
            direction: "outgoing", "incoming", or "both"
            relationship_filter: Optional filter string to match relationship labels

        Returns:
            List of relationship dictionaries (temporal quadruples)
        """
        rel_filter = ""
        if relationship_filter:
            rel_filter = f"AND toLower(r.relationship) CONTAINS toLower('{relationship_filter}')"

        if direction == "outgoing":
            query = f"""
            MATCH (e:Entity {{id: $entity_id}})-[r:RELATES_TO]->(other:Entity)
            WHERE 1=1 {rel_filter}
            RETURN e.name as source, other.name as target, other.id as target_id,
                   r.relationship as relationship, r.timestamp as timestamp,
                   r.description as description, r.confidence as confidence,
                   r.valid_from as valid_from, r.valid_to as valid_to
            """
        elif direction == "incoming":
            query = f"""
            MATCH (other:Entity)-[r:RELATES_TO]->(e:Entity {{id: $entity_id}})
            WHERE 1=1 {rel_filter}
            RETURN other.name as source, other.id as source_id, e.name as target,
                   r.relationship as relationship, r.timestamp as timestamp,
                   r.description as description, r.confidence as confidence,
                   r.valid_from as valid_from, r.valid_to as valid_to
            """
        else:  # both
            query = f"""
            MATCH (e:Entity {{id: $entity_id}})-[r:RELATES_TO]-(other:Entity)
            WHERE 1=1 {rel_filter}
            RETURN e.name as source, other.name as target, other.id as other_id,
                   r.relationship as relationship, r.timestamp as timestamp,
                   r.description as description, r.confidence as confidence,
                   r.valid_from as valid_from, r.valid_to as valid_to,
                   CASE WHEN startNode(r) = e THEN 'outgoing' ELSE 'incoming' END as direction
            """

        results = self.client.execute_read_transaction(
            query, {"entity_id": entity_id}
        )

        return results

    def get_relationship_path(
        self,
        source_entity_id: str,
        target_entity_id: str,
        max_hops: int = 3,
    ) -> List[Dict]:
        """
        Find relationship paths between two entities.

        Args:
            source_entity_id: Source entity ID
            target_entity_id: Target entity ID
            max_hops: Maximum path length

        Returns:
            List of paths with relationships (temporal quadruples)
        """
        query = f"""
        MATCH path = shortestPath(
            (source:Entity {{id: $source_id}})-[r:RELATES_TO*1..{max_hops}]-(target:Entity {{id: $target_id}})
        )
        RETURN [node in nodes(path) | node.name] as entities,
               [rel in relationships(path) | {{
                   relationship: rel.relationship,
                   timestamp: rel.timestamp,
                   description: rel.description
               }}] as relationships,
               length(path) as path_length
        """

        results = self.client.execute_read_transaction(query, {
            "source_id": source_entity_id,
            "target_id": target_entity_id,
        })

        return results

    # ===== Query Operations =====

    def search_documents(self, search_text: str, limit: int = 10) -> List[Document]:
        """
        Search documents by title or content.

        Args:
            search_text: Text to search for
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        query = """
        CALL db.index.fulltext.queryNodes('document_text', $search_text)
        YIELD node, score
        RETURN node
        ORDER BY score DESC
        LIMIT $limit
        """

        results = self.client.execute_read_transaction(query, {
            "search_text": search_text,
            "limit": limit,
        })

        documents = [Document.from_neo4j_dict(r["node"]) for r in results]

        return documents


# Global operations instance
_operations: Optional[GraphOperations] = None


def get_graph_operations() -> GraphOperations:
    """Get the global graph operations instance."""
    global _operations
    if _operations is None:
        _operations = GraphOperations()
    return _operations
