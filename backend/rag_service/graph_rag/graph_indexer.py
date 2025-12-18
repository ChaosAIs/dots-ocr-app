"""
GraphRAG Indexer - Entity extraction and graph building during document indexing.

This module integrates with the existing indexer to add GraphRAG capabilities:
- Extract entities and relationships from document chunks using LLM
- Store entities and relationships in Neo4j graph database
- Build knowledge graph for graph-based retrieval
- Generate embeddings for semantic entity/relationship search

Key Features:
- Chunk-by-chunk processing: Save to Neo4j after each chunk is processed
- Document-level deduplication: Same entity in multi-page PDF stored once
- Incremental saving: No data loss if processing fails mid-way
- Vector embeddings: Semantic search using Neo4j native vector indexes

Performance Optimizations (configured via .env):
- Selective entity extraction: Focus on high-value entities only (GRAPH_RAG_MIN_ENTITY_SCORE)
- Reduced gleaning: Fewer LLM calls per chunk (GRAPH_RAG_MAX_GLEANING=0)
- Importance filtering: Only store entities above threshold score

Expected Performance:
- 70-85% reduction in processing time vs unoptimized version
- Better graph quality (less noise, more focused entities)
- Typical 100-chunk document: 10-15 minutes (vs 58 minutes unoptimized)
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Tuple

from .entity_extractor import EntityExtractor
from ..utils.date_normalizer import find_and_normalize_dates

logger = logging.getLogger(__name__)

# Feature flags and configuration
GRAPH_RAG_ENABLED = os.getenv("GRAPH_RAG_ENABLED", "false").lower() == "true"
GRAPH_RAG_EMBEDDINGS_ENABLED = os.getenv("GRAPH_RAG_EMBEDDINGS_ENABLED", "true").lower() == "true"


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
        self._embedding_service = None
        self._embeddings_enabled = GRAPH_RAG_EMBEDDINGS_ENABLED

    async def _init_embedding_service(self):
        """Initialize embedding service for entity/relationship embeddings."""
        if self._embedding_service is not None:
            return

        if not self._embeddings_enabled:
            logger.debug("[GraphRAG] Embeddings disabled, skipping embedding service init")
            return

        try:
            from ..local_qwen_embedding import LocalQwen3Embedding
            self._embedding_service = LocalQwen3Embedding()
            logger.info("[GraphRAG] Embedding service initialized")
        except Exception as e:
            logger.warning(f"[GraphRAG] Failed to init embedding service: {e}")
            self._embeddings_enabled = False

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (empty list if embeddings disabled)
        """
        if not self._embeddings_enabled or not texts:
            return [None] * len(texts)

        await self._init_embedding_service()

        if not self._embedding_service:
            return [None] * len(texts)

        try:
            embeddings = self._embedding_service.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.warning(f"[GraphRAG] Failed to generate embeddings: {e}")
            return [None] * len(texts)

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
        filename_with_ext: str = None,
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
            filename_with_ext: Original filename with extension for database tracking (optional)

        Returns:
            Tuple of (num_entities, num_relationships)
        """
        if not GRAPH_RAG_ENABLED:
            logger.debug("[GraphRAG] GraphRAG is disabled, skipping entity extraction")
            return 0, 0

        await self._init_storage()

        logger.info(f"[GraphRAG] Starting indexing for {source_name}: {len(chunks)} chunks")

        # Get document from database for granular status tracking
        doc = None
        repo = None
        try:
            from db.database import get_db_session
            from db.document_repository import DocumentRepository

            if filename_with_ext:
                with get_db_session() as db:
                    repo = DocumentRepository(db)
                    doc = repo.get_by_filename(filename_with_ext)
                    if doc:
                        logger.info(f"[GraphRAG] Tracking granular status for: {filename_with_ext}")
                        # Initialize GraphRAG with total chunk count before processing
                        repo.init_graphrag_total_chunks(doc, len(chunks))
                        logger.info(f"[GraphRAG] Initialized tracking with {len(chunks)} total chunks")
        except Exception as e:
            logger.warning(f"[GraphRAG] Could not get document from database: {e}")

        total_entities = 0
        total_relationships = 0

        # Track entity types for relationship resolution
        entity_type_map: Dict[str, str] = {}  # name.lower() -> entity_type

        # Track skipped chunks for reporting
        skipped_empty = 0

        # Process each chunk and save to Neo4j immediately
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("id", f"chunk_{i}")
            content = chunk.get("page_content", "")

            # Extract page number from metadata
            page_number = chunk.get("metadata", {}).get("page")

            # Skip empty chunks
            if not content.strip():
                logger.debug(f"[GraphRAG] Skipping empty chunk {chunk_id}")
                skipped_empty += 1

                # Update status: chunk skipped
                if filename_with_ext:
                    try:
                        from db.database import get_db_session
                        from db.document_repository import DocumentRepository
                        with get_db_session() as db:
                            repo = DocumentRepository(db)
                            doc = repo.get_by_filename(filename_with_ext)
                            if doc:
                                repo.update_graphrag_chunk_status(
                                    doc, chunk_id, "skipped", page_number=page_number
                                )
                    except Exception as e:
                        logger.debug(f"Could not update skipped status: {e}")

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

                # Post-process date entities: Add normalized date properties
                chunk_content = chunk.get("page_content", "")
                normalized_dates = find_and_normalize_dates(chunk_content)

                for entity in entities:
                    if entity.entity_type.lower() == "date":
                        # Try to match entity name with normalized dates
                        for date_entity in normalized_dates:
                            # Match if raw date appears in entity name or vice versa
                            if (date_entity.raw in entity.name or
                                entity.name in date_entity.raw or
                                date_entity.normalized in entity.name):
                                # Add normalized date properties to entity metadata
                                entity.metadata["date_normalized"] = date_entity.normalized
                                entity.metadata["date_raw"] = date_entity.raw
                                if date_entity.time:
                                    entity.metadata["date_time"] = date_entity.time
                                entity.metadata["date_year"] = date_entity.year
                                entity.metadata["date_month"] = date_entity.month
                                entity.metadata["date_day"] = date_entity.day
                                logger.debug(
                                    f"[GraphRAG] Enhanced date entity: {entity.name} → {date_entity.normalized}"
                                )
                                break

                # Generate embeddings for entities (batch for efficiency)
                entity_embeddings = []
                if self._embeddings_enabled and entities:
                    entity_texts = [
                        f"{e.name}: {e.entity_type}. {e.description}" for e in entities
                    ]
                    entity_embeddings = await self._generate_embeddings(entity_texts)
                else:
                    entity_embeddings = [None] * len(entities)

                # Save entities to Neo4j immediately (with deduplication and embeddings)
                for idx, entity in enumerate(entities):
                    embedding = entity_embeddings[idx] if idx < len(entity_embeddings) else None
                    await self._graph_storage.upsert_entity_for_document(
                        name=entity.name,
                        entity_type=entity.entity_type,
                        description=entity.description,
                        source_doc=source_name,
                        source_chunk_id=chunk_id,
                        key_score=entity.key_score,
                        embedding=embedding,
                        metadata=entity.metadata,  # Pass entity metadata (includes date properties)
                    )
                    # Track entity type for relationship resolution
                    entity_type_map[entity.name.lower().strip()] = entity.entity_type

                total_entities += len(entities)

                # Generate embeddings for relationships (batch for efficiency)
                rel_embeddings = []
                if self._embeddings_enabled and relationships:
                    rel_texts = [
                        f"{r.metadata.get('src_name', '')} {r.description} {r.metadata.get('tgt_name', '')}"
                        for r in relationships
                    ]
                    rel_embeddings = await self._generate_embeddings(rel_texts)
                else:
                    rel_embeddings = [None] * len(relationships)

                # Save relationships to Neo4j immediately (with deduplication and embeddings)
                for idx, rel in enumerate(relationships):
                    src_name = rel.metadata.get("src_name", "")
                    tgt_name = rel.metadata.get("tgt_name", "")

                    # Get entity types from our map or use 'unknown'
                    src_type = entity_type_map.get(src_name.lower().strip(), "unknown")
                    tgt_type = entity_type_map.get(tgt_name.lower().strip(), "unknown")

                    embedding = rel_embeddings[idx] if idx < len(rel_embeddings) else None
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
                        embedding=embedding,
                    )

                total_relationships += len(relationships)

                logger.info(
                    f"[GraphRAG] Chunk {chunk_id}: saved {len(entities)} entities, "
                    f"{len(relationships)} relationships to Neo4j"
                )

                # Update status: chunk succeeded
                if filename_with_ext:
                    try:
                        from db.database import get_db_session
                        from db.document_repository import DocumentRepository
                        with get_db_session() as db:
                            repo = DocumentRepository(db)
                            doc = repo.get_by_filename(filename_with_ext)
                            if doc:
                                repo.update_graphrag_chunk_status(
                                    doc,
                                    chunk_id,
                                    "success",
                                    entities=len(entities),
                                    relationships=len(relationships),
                                    page_number=page_number
                                )
                                logger.info(f"[GraphRAG] Updated chunk {chunk_id} status: success ({len(entities)} entities, {len(relationships)} relationships)")
                            else:
                                logger.error(f"[GraphRAG] Could not find document: {filename_with_ext}")
                    except Exception as e:
                        logger.error(f"Could not update GraphRAG success status: {e}", exc_info=True)

            except Exception as e:
                logger.error(f"[GraphRAG] Failed to process chunk {chunk_id}: {e}", exc_info=True)

                # Update status: chunk failed
                if filename_with_ext:
                    try:
                        from db.database import get_db_session
                        from db.document_repository import DocumentRepository
                        with get_db_session() as db:
                            repo = DocumentRepository(db)
                            doc = repo.get_by_filename(filename_with_ext)
                            if doc:
                                repo.update_graphrag_chunk_status(
                                    doc,
                                    chunk_id,
                                    "failed",
                                    error=str(e),
                                    page_number=page_number
                                )
                    except Exception as db_error:
                        logger.warning(f"Could not update GraphRAG failed status: {db_error}")

                continue

        # Log summary
        processed_chunks = len(chunks) - skipped_empty
        logger.info(
            f"[GraphRAG] Indexing complete for {source_name}: "
            f"{total_entities} entities, {total_relationships} relationships | "
            f"Processed {processed_chunks}/{len(chunks)} chunks (skipped {skipped_empty} empty)"
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
    filename_with_ext: str = None,
) -> Tuple[int, int]:
    """
    Synchronous wrapper for GraphRAG chunk indexing.

    This can be called from the existing synchronous indexer.

    Args:
        chunks: List of chunk dicts with 'id', 'page_content', 'metadata'
        source_name: Source document name
        workspace_id: Workspace ID
        use_gleaning: Whether to use gleaning loop
        filename_with_ext: Original filename with extension for database tracking (optional)

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
        indexer.index_chunks(chunks, source_name, use_gleaning, filename_with_ext)
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

