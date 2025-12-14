"""
GraphRAG Indexer - Entity extraction and graph building during document indexing.

This module integrates with the existing indexer to add GraphRAG capabilities:
- Extract entities and relationships from document chunks using LLM
- Store entities and relationships in Neo4j graph database
- Build knowledge graph for graph-based retrieval

Key Features:
- Chunk-by-chunk processing: Save to Neo4j after each chunk is processed
- Document-level deduplication: Same entity in multi-page PDF stored once
- Incremental saving: No data loss if processing fails mid-way
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Tuple

from .entity_extractor import EntityExtractor

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
        self._graph_storage = None

    async def _init_storage(self):
        """Initialize Neo4j storage backend lazily."""
        if self._storage_initialized:
            return

        try:
            from ..storage import Neo4jStorage

            self._graph_storage = Neo4jStorage(self.workspace_id)
            self._storage_initialized = True
            logger.info("[GraphRAG] Neo4j storage initialized")

        except Exception as e:
            logger.error(f"[GraphRAG] Failed to initialize Neo4j storage: {e}")
            raise

    async def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        source_name: str,
        use_gleaning: bool = True,
    ) -> Tuple[int, int]:
        """
        Extract entities from chunks and store in Neo4j graph.

        Processes each chunk individually and saves to Neo4j immediately after
        extraction. This ensures:
        1. No data loss if processing fails mid-way
        2. Document-level deduplication across multi-page PDFs
        3. Same entity in different chunks is merged in Neo4j

        Args:
            chunks: List of chunk dicts with 'id', 'page_content', 'metadata'
            source_name: Source document name (used for deduplication)
            use_gleaning: Whether to use gleaning loop for extraction

        Returns:
            Tuple of (num_entities, num_relationships)
        """
        if not GRAPH_RAG_ENABLED:
            logger.debug("[GraphRAG] GraphRAG is disabled, skipping entity extraction")
            return 0, 0

        await self._init_storage()

        logger.info(f"[GraphRAG] Starting indexing for {source_name}: {len(chunks)} chunks")

        total_entities = 0
        total_relationships = 0

        # Track entity types for relationship resolution
        entity_type_map: Dict[str, str] = {}  # name.lower() -> entity_type

        # Process each chunk and save to Neo4j immediately
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("id", f"chunk_{i}")
            content = chunk.get("page_content", "")

            if not content.strip():
                logger.debug(f"[GraphRAG] Skipping empty chunk {chunk_id}")
                continue

            logger.info(f"[GraphRAG] Processing chunk {i + 1}/{len(chunks)}: {chunk_id}")

            try:
                # Extract entities and relationships from this chunk
                if use_gleaning:
                    entities, relationships = await self.entity_extractor.extract_with_gleaning(
                        content, chunk_id
                    )
                else:
                    entities, relationships = await self.entity_extractor.extract_simple(
                        content, chunk_id
                    )

                if not entities and not relationships:
                    logger.debug(f"[GraphRAG] No entities/relationships in chunk {chunk_id}")
                    continue

                # Save entities to Neo4j immediately (with deduplication)
                for entity in entities:
                    await self._graph_storage.upsert_entity_for_document(
                        name=entity.name,
                        entity_type=entity.entity_type,
                        description=entity.description,
                        source_doc=source_name,
                        source_chunk_id=chunk_id,
                        key_score=entity.key_score,
                    )
                    # Track entity type for relationship resolution
                    entity_type_map[entity.name.lower().strip()] = entity.entity_type

                total_entities += len(entities)

                # Save relationships to Neo4j immediately (with deduplication)
                for rel in relationships:
                    src_name = rel.metadata.get("src_name", "")
                    tgt_name = rel.metadata.get("tgt_name", "")

                    # Get entity types from our map or use 'unknown'
                    src_type = entity_type_map.get(src_name.lower().strip(), "unknown")
                    tgt_type = entity_type_map.get(tgt_name.lower().strip(), "unknown")

                    await self._graph_storage.upsert_relationship_for_document(
                        src_name=src_name,
                        tgt_name=tgt_name,
                        src_entity_type=src_type,
                        tgt_entity_type=tgt_type,
                        description=rel.description,
                        keywords=rel.keywords,
                        source_doc=source_name,
                        source_chunk_id=chunk_id,
                        weight=rel.weight,
                    )

                total_relationships += len(relationships)

                logger.info(
                    f"[GraphRAG] Chunk {chunk_id}: saved {len(entities)} entities, "
                    f"{len(relationships)} relationships to Neo4j"
                )

            except Exception as e:
                logger.error(f"[GraphRAG] Failed to process chunk {chunk_id}: {e}", exc_info=True)
                continue

        logger.info(
            f"[GraphRAG] Indexing complete for {source_name}: "
            f"{total_entities} entities, {total_relationships} relationships"
        )

        return total_entities, total_relationships

    async def delete_by_source(self, source_name: str) -> None:
        """
        Delete all GraphRAG data for a source document from Neo4j.

        Args:
            source_name: Source document name
        """
        if not GRAPH_RAG_ENABLED:
            return

        await self._init_storage()

        logger.info(f"[GraphRAG] Deleting data for document: {source_name}")

        # Delete from Neo4j graph using document-based deletion
        try:
            count = await self._graph_storage.delete_by_document(source_name)
            logger.info(f"[GraphRAG] Deleted {count} entities for document: {source_name}")
        except Exception as e:
            logger.warning(f"[GraphRAG] Error deleting from Neo4j: {e}")

    async def delete_workspace(self) -> None:
        """Delete all GraphRAG data for the current workspace."""
        if not GRAPH_RAG_ENABLED:
            return

        await self._init_storage()

        logger.info(f"[GraphRAG] Deleting all data for workspace: {self.workspace_id}")

        await self._graph_storage.drop()

        logger.info(f"[GraphRAG] Deleted all data for workspace: {self.workspace_id}")


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
        chunks: List of chunk dicts with 'id', 'page_content', 'metadata'
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


def delete_graphrag_by_source_sync(
    source_name: str,
    workspace_id: str = "default",
) -> None:
    """
    Synchronous wrapper for GraphRAG source deletion.

    This can be called from the document delete endpoint.
    Works correctly even when called from within an existing async event loop.

    Args:
        source_name: Source document name
        workspace_id: Workspace ID
    """
    if not GRAPH_RAG_ENABLED:
        return

    indexer = GraphRAGIndexer(workspace_id=workspace_id)

    # Check if we're already in an event loop
    try:
        _loop = asyncio.get_running_loop()  # noqa: F841 - only used to check if loop exists
        # We're in an existing event loop - run in a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                indexer.delete_by_source(source_name)
            )
            future.result()  # Wait for completion
    except RuntimeError:
        # No running event loop - use asyncio.run directly
        asyncio.run(indexer.delete_by_source(source_name))

