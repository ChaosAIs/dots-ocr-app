"""
GraphRAG Indexer - Entity extraction and graph building during document indexing.

This module integrates with the existing indexer to add GraphRAG capabilities:
- Extract entities and relationships from document chunks
- Store entities in PostgreSQL, Qdrant, and Neo4j
- Build knowledge graph for graph-based retrieval
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple

from .base import Entity, Relationship
from .entity_extractor import EntityExtractor
from .utils import entity_to_dict, relationship_to_dict

logger = logging.getLogger(__name__)

# Feature flag
GRAPH_RAG_ENABLED = os.getenv("GRAPH_RAG_ENABLED", "false").lower() == "true"


class GraphRAGIndexer:
    """
    Handles entity extraction and graph building during document indexing.

    This class is designed to be called after the regular chunking and
    vector indexing is complete.
    """

    def __init__(
        self,
        workspace_id: str = "default",
        entity_extractor: EntityExtractor = None,
    ):
        """
        Initialize the GraphRAG indexer.

        Args:
            workspace_id: Workspace ID for multi-tenant isolation
            entity_extractor: Optional custom entity extractor
        """
        self.workspace_id = workspace_id
        self.entity_extractor = entity_extractor or EntityExtractor()
        self._storage_initialized = False
        self._kv_storage = None
        self._entity_vector_store = None
        self._edge_vector_store = None
        self._graph_storage = None

    async def _init_storage(self):
        """Initialize storage backends lazily."""
        if self._storage_initialized:
            return

        try:
            from ..storage import (
                PostgresKVStorage,
                QdrantEntityVectorStore,
                QdrantEdgeVectorStore,
                Neo4jStorage,
            )

            self._kv_storage = {
                "entities": PostgresKVStorage("graphrag_entities", self.workspace_id),
                "hyperedges": PostgresKVStorage("graphrag_hyperedges", self.workspace_id),
                "chunks": PostgresKVStorage("graphrag_chunks", self.workspace_id),
            }
            self._entity_vector_store = QdrantEntityVectorStore(self.workspace_id)
            self._edge_vector_store = QdrantEdgeVectorStore(self.workspace_id)
            self._graph_storage = Neo4jStorage(self.workspace_id)

            self._storage_initialized = True
            logger.info("GraphRAG storage initialized")

        except Exception as e:
            logger.error(f"Failed to initialize GraphRAG storage: {e}")
            raise

    async def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        source_name: str,
        use_gleaning: bool = True,
    ) -> Tuple[int, int]:
        """
        Extract entities from chunks and store in graph.

        Args:
            chunks: List of chunk dicts with 'id', 'content', 'metadata'
            source_name: Source document name
            use_gleaning: Whether to use gleaning loop for extraction

        Returns:
            Tuple of (num_entities, num_relationships)
        """
        if not GRAPH_RAG_ENABLED:
            logger.debug("GraphRAG is disabled, skipping entity extraction")
            return 0, 0

        await self._init_storage()

        logger.info(f"Starting GraphRAG indexing for {source_name}: {len(chunks)} chunks")

        # Extract entities and relationships
        entities, relationships = await self.entity_extractor.extract_batch(
            chunks, use_gleaning=use_gleaning
        )

        if not entities and not relationships:
            logger.info(f"No entities or relationships extracted from {source_name}")
            return 0, 0

        # Store entities
        await self._store_entities(entities)

        # Store relationships
        await self._store_relationships(relationships, entities)

        logger.info(
            f"GraphRAG indexing complete for {source_name}: "
            f"{len(entities)} entities, {len(relationships)} relationships"
        )

        return len(entities), len(relationships)

    async def _store_entities(self, entities: List[Entity]) -> None:
        """Store entities in all storage backends."""
        if not entities:
            return

        # Prepare data for storage
        entity_data = {e.id: entity_to_dict(e) for e in entities}

        # Store in PostgreSQL KV
        await self._kv_storage["entities"].upsert(entity_data)

        # Store in Qdrant for vector search
        await self._entity_vector_store.upsert(entity_data)

        # Store in Neo4j graph
        for entity in entities:
            await self._graph_storage.upsert_node(
                entity.id,
                {
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "description": entity.description,
                    "key_score": entity.key_score,
                    "source_chunk_id": entity.source_chunk_id,
                },
            )

        logger.debug(f"Stored {len(entities)} entities")

    async def _store_relationships(
        self, relationships: List[Relationship], entities: List[Entity]
    ) -> None:
        """Store relationships in all storage backends."""
        if not relationships:
            return

        # Build entity name to ID mapping
        entity_name_to_id = {e.name.lower(): e.id for e in entities}

        # Prepare data for storage
        rel_data = {}
        for rel in relationships:
            # Get entity names from metadata
            src_name = rel.metadata.get("src_name", "").lower()
            tgt_name = rel.metadata.get("tgt_name", "").lower()

            # Update entity IDs if we have the actual entities
            if src_name in entity_name_to_id:
                rel.src_entity_id = entity_name_to_id[src_name]
            if tgt_name in entity_name_to_id:
                rel.tgt_entity_id = entity_name_to_id[tgt_name]

            rel_data[rel.id] = relationship_to_dict(rel)

        # Store in PostgreSQL KV
        await self._kv_storage["hyperedges"].upsert(rel_data)

        # Store in Qdrant for vector search
        edge_vector_data = {}
        for rel in relationships:
            edge_vector_data[rel.id] = {
                "src_name": rel.metadata.get("src_name", ""),
                "tgt_name": rel.metadata.get("tgt_name", ""),
                "description": rel.description,
                "keywords": rel.keywords,
                "weight": rel.weight,
            }
        await self._edge_vector_store.upsert(edge_vector_data)

        # Store in Neo4j graph
        for rel in relationships:
            await self._graph_storage.upsert_edge(
                rel.src_entity_id,
                rel.tgt_entity_id,
                {
                    "description": rel.description,
                    "keywords": rel.keywords,
                    "weight": rel.weight,
                    "source_chunk_id": rel.source_chunk_id,
                },
            )

        logger.debug(f"Stored {len(relationships)} relationships")

    async def delete_by_source(self, source_name: str) -> None:
        """
        Delete all GraphRAG data for a source document.

        Args:
            source_name: Source document name
        """
        if not GRAPH_RAG_ENABLED:
            return

        await self._init_storage()

        logger.info(f"Deleting GraphRAG data for source: {source_name}")

        # Get all entities for this source
        # Note: This requires querying by source_chunk_id pattern
        # For now, we'll use workspace-level deletion
        # TODO: Implement source-level deletion

        logger.warning(
            f"Source-level deletion not yet implemented for GraphRAG. "
            f"Use workspace-level deletion instead."
        )

    async def delete_workspace(self) -> None:
        """Delete all GraphRAG data for the current workspace."""
        if not GRAPH_RAG_ENABLED:
            return

        await self._init_storage()

        logger.info(f"Deleting all GraphRAG data for workspace: {self.workspace_id}")

        # Delete from all storage backends
        await self._kv_storage["entities"].drop()
        await self._kv_storage["hyperedges"].drop()
        await self._kv_storage["chunks"].drop()
        await self._graph_storage.drop()

        # Delete from Qdrant
        from ..storage import delete_entities_by_workspace, delete_edges_by_workspace
        delete_entities_by_workspace(self.workspace_id)
        delete_edges_by_workspace(self.workspace_id)

        logger.info(f"Deleted all GraphRAG data for workspace: {self.workspace_id}")


def index_chunks_sync(
    chunks: List[Dict[str, Any]],
    source_name: str,
    workspace_id: str = "default",
    use_gleaning: bool = True,
) -> Tuple[int, int]:
    """
    Synchronous wrapper for GraphRAG chunk indexing.

    This can be called from the existing synchronous indexer.

    Args:
        chunks: List of chunk dicts with 'id', 'content', 'metadata'
        source_name: Source document name
        workspace_id: Workspace ID
        use_gleaning: Whether to use gleaning loop

    Returns:
        Tuple of (num_entities, num_relationships)
    """
    if not GRAPH_RAG_ENABLED:
        return 0, 0

    indexer = GraphRAGIndexer(workspace_id=workspace_id)

    # Run async function in event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        indexer.index_chunks(chunks, source_name, use_gleaning)
    )

