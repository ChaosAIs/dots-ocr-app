"""
Unified Retriever - Abstracted retrieval interface for iterative reasoning.

This module provides a unified retrieval interface that works with or without GraphRAG,
allowing the iterative reasoning engine to use the same logic regardless of backend.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from .vectorstore import PARENT_CHUNK_AUTO_INCLUDE

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Unified result from retrieval operations.

    Contains entities, relationships, and chunks regardless of backend.
    When GraphRAG is disabled, entities and relationships will be empty.

    Parent chunks (if parent/child chunk design is enabled) are automatically
    included when child chunks are retrieved.
    """
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    parent_chunks: List[Dict[str, Any]] = field(default_factory=list)  # Parent context chunks
    sources: Set[str] = field(default_factory=set)
    query_used: str = ""
    retrieval_mode: str = "vector"  # "vector", "graph", "hybrid"

    def is_empty(self) -> bool:
        """Check if no results were retrieved."""
        return not self.entities and not self.relationships and not self.chunks


class UnifiedRetriever:
    """
    Unified retrieval interface that abstracts GraphRAG and vector-only search.

    Provides consistent interface for iterative reasoning regardless of whether
    GraphRAG is enabled or disabled.
    """

    def __init__(
        self,
        graphrag_enabled: bool = False,
        source_names: Optional[List[str]] = None,
        workspace_id: str = "default",
        accessible_doc_ids: Optional[Set] = None
    ):
        """
        Initialize the unified retriever.

        Args:
            graphrag_enabled: Whether to use GraphRAG (Neo4j + Qdrant) or vector-only (Qdrant)
            source_names: Optional list of source document names to filter results
            workspace_id: Workspace ID for multi-tenant isolation
            accessible_doc_ids: Optional set of document IDs the user can access (for access control)
        """
        self.graphrag_enabled = graphrag_enabled
        self.source_names = source_names or []
        self.workspace_id = workspace_id
        self.accessible_doc_ids = accessible_doc_ids
        self._graphrag = None
        self._initialized = False

        # Log access control info
        if accessible_doc_ids is not None:
            logger.info(f"[UnifiedRetriever] Access control enabled: {len(accessible_doc_ids)} accessible documents")

    async def _init_graphrag(self):
        """Lazily initialize GraphRAG if enabled."""
        if self._initialized:
            return

        logger.info(f"[UnifiedRetriever] Initializing - graphrag_enabled={self.graphrag_enabled}")

        if self.graphrag_enabled:
            try:
                from .graph_rag import GraphRAG, GRAPH_RAG_QUERY_ENABLED
                logger.info(f"[UnifiedRetriever] GraphRAG module imported, GRAPH_RAG_QUERY_ENABLED={GRAPH_RAG_QUERY_ENABLED}")
                self._graphrag = GraphRAG(workspace_id=self.workspace_id)
                logger.info("[UnifiedRetriever] GraphRAG backend initialized successfully")
            except ImportError as e:
                logger.warning(f"[UnifiedRetriever] Failed to import GraphRAG: {e}")
                import traceback
                logger.warning(f"[UnifiedRetriever] Import traceback: {traceback.format_exc()}")
                self.graphrag_enabled = False
                self._graphrag = None
            except Exception as e:
                logger.warning(f"[UnifiedRetriever] Failed to init GraphRAG: {e}, falling back to vector-only")
                import traceback
                logger.warning(f"[UnifiedRetriever] Init traceback: {traceback.format_exc()}")
                self.graphrag_enabled = False
                self._graphrag = None
        else:
            logger.info("[UnifiedRetriever] GraphRAG disabled by caller, using vector-only")

        self._initialized = True

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        mode: str = "hybrid"
    ) -> RetrievalResult:
        """
        Retrieve relevant content for the given query.

        Args:
            query: Search query string
            top_k: Maximum number of results to retrieve
            mode: Retrieval mode ("local", "global", "hybrid") - only used when GraphRAG enabled

        Returns:
            RetrievalResult with entities, relationships, and chunks
        """
        await self._init_graphrag()

        # Debug: Log retrieval decision
        logger.info(f"[UnifiedRetriever] Retrieve decision debug:")
        logger.info(f"[UnifiedRetriever]   - graphrag_enabled: {self.graphrag_enabled}")
        logger.info(f"[UnifiedRetriever]   - _graphrag instance: {self._graphrag is not None}")
        logger.info(f"[UnifiedRetriever]   - Will use: {'graph_retrieval' if (self.graphrag_enabled and self._graphrag) else 'vector_only'}")

        if self.graphrag_enabled and self._graphrag:
            return await self._graph_retrieval(query, top_k, mode)
        else:
            return await self._vector_only_retrieval(query, top_k)

    async def _vector_only_retrieval(
        self,
        query: str,
        top_k: int = 20
    ) -> RetrievalResult:
        """
        Vector-only retrieval using unified_vector_search.

        This is used when GraphRAG is disabled. Returns chunks only,
        with empty entities and relationships.

        Now uses unified_vector_search() for consistent behavior with legacy mode.

        Args:
            query: Search query string
            top_k: Maximum number of chunks to retrieve

        Returns:
            RetrievalResult with chunks only (no entities/relationships)
        """
        logger.info(f"[UnifiedRetriever] Vector-only retrieval for: '{query[:50]}...'")

        try:
            from .vectorstore import unified_vector_search

            # Determine document IDs to search
            doc_id_list = []
            if self.accessible_doc_ids is not None and len(self.accessible_doc_ids) > 0:
                doc_id_list = [str(doc_id) for doc_id in self.accessible_doc_ids]
                logger.info(f"[UnifiedRetriever] Access control: filtering to {len(doc_id_list)} accessible document IDs")

            # Use unified vector search (same as legacy path)
            search_result = unified_vector_search(
                query=query,
                document_ids=doc_id_list,
                k=top_k,
                fetch_k=50,
                lambda_mult=0.5,
                per_document_selection=True,
                include_parent_chunks=None  # Uses PARENT_CHUNK_AUTO_INCLUDE env variable
            )

            # Check for blocked/empty results
            if search_result.retrieval_stats.get("blocked"):
                logger.warning("[UnifiedRetriever] Search blocked - no document access")
                return RetrievalResult(query_used=query, retrieval_mode="vector")

            # Log summary
            stats = search_result.retrieval_stats
            chunk_types = stats.get("chunk_types", {})
            logger.info(
                f"[UnifiedRetriever] Vector search results: {len(search_result.chunks)} chunks "
                f"(parents={chunk_types.get('parent', 0)}, children={chunk_types.get('child', 0)}, "
                f"atomic={chunk_types.get('atomic', 0)}) + {len(search_result.parent_chunks)} context chunks "
                f"from {len(search_result.sources)} sources"
            )

            return RetrievalResult(
                entities=[],
                relationships=[],
                chunks=search_result.chunks,
                parent_chunks=search_result.parent_chunks,
                sources=search_result.sources,
                query_used=query,
                retrieval_mode="vector"
            )

        except Exception as e:
            logger.error(f"[UnifiedRetriever] Vector retrieval failed: {e}")
            return RetrievalResult(query_used=query, retrieval_mode="vector")

    async def _graph_retrieval(
        self,
        query: str,
        top_k: int = 20,
        mode: str = "hybrid"
    ) -> RetrievalResult:
        """
        Graph-enhanced retrieval using GraphRAG (Neo4j + Qdrant).

        This uses the existing GraphRAG infrastructure which already combines
        graph search with vector search internally.

        When PARENT_CHUNK_AUTO_INCLUDE is enabled, automatically fetches
        parent chunks for any child chunks in the results.

        Args:
            query: Search query string
            top_k: Maximum number of results
            mode: Retrieval mode ("local", "global", "hybrid")

        Returns:
            RetrievalResult with entities, relationships, and chunks
        """
        logger.info(f"[UnifiedRetriever] Graph retrieval ({mode}) for: '{query[:50]}...'")

        try:
            from .graph_rag.base import QueryParam, QueryMode
            from .vectorstore import get_parent_chunks_for_children

            # Create query params (single-step, no iteration here - iteration handled by engine)
            params = QueryParam()
            params.max_steps = 1  # Single retrieval step
            params.top_k = top_k
            if self.source_names:
                params.source_names = self.source_names

            # Add accessible document IDs for access control filtering
            if self.accessible_doc_ids is not None and len(self.accessible_doc_ids) > 0:
                params.accessible_doc_ids = self.accessible_doc_ids
                logger.info(f"[UnifiedRetriever] Graph retrieval: filtering to {len(self.accessible_doc_ids)} accessible document IDs")

            # Determine retrieval mode
            if mode == "local":
                context = await self._graphrag._local_retrieval(query, params)
            elif mode == "global":
                context = await self._graphrag._global_retrieval(query, params)
            else:  # hybrid (default)
                context = await self._graphrag._hybrid_retrieval(query, params)

            # Extract sources from chunks and track child chunks
            sources = set()
            child_chunk_ids = []

            logger.debug(f"[UnifiedRetriever] Processing {len(context.chunks)} chunks from graph retrieval")

            for i, chunk in enumerate(context.chunks):
                metadata = chunk.get("metadata", {})
                source = metadata.get("source", "")
                chunk_id = chunk.get("chunk_id", "")
                chunk_type = metadata.get("chunk_type", "unknown")
                is_parent_chunk = metadata.get("is_parent_chunk", False)
                parent_chunk_id = metadata.get("parent_chunk_id")

                if source:
                    sources.add(source)

                # Detailed logging for each chunk
                logger.debug(
                    f"[UnifiedRetriever] Graph chunk {i+1}/{len(context.chunks)}: id={chunk_id}, "
                    f"type={chunk_type}, is_parent={is_parent_chunk}, parent_id={parent_chunk_id}"
                )

                # Track child chunks that have a parent
                if parent_chunk_id:
                    child_chunk_ids.append(chunk_id)
                    logger.debug(f"[UnifiedRetriever] → Identified as CHILD chunk, will fetch parent: {parent_chunk_id}")

            # Summary of chunk types found
            parent_count = sum(1 for c in context.chunks if c.get("metadata", {}).get("is_parent_chunk"))
            child_count = len(child_chunk_ids)
            atomic_count = len(context.chunks) - parent_count - child_count

            logger.info(
                f"[UnifiedRetriever] Graph retrieval found {len(context.entities)} entities, "
                f"{len(context.relationships)} relationships, {len(context.chunks)} chunks "
                f"(parents={parent_count}, children={child_count}, atomic={atomic_count})"
            )

            # Auto-include parent chunks for child matches
            parent_chunks = []
            if PARENT_CHUNK_AUTO_INCLUDE and child_chunk_ids:
                logger.info(f"[UnifiedRetriever] PARENT_CHUNK_AUTO_INCLUDE enabled, fetching parents for {len(child_chunk_ids)} child chunks")
                logger.debug(f"[UnifiedRetriever] Child chunk IDs: {child_chunk_ids}")

                parent_docs = get_parent_chunks_for_children(child_chunk_ids, self.source_names)

                # Add parent chunks (deduplicated from regular chunks)
                seen_ids = {c.get("chunk_id", "") for c in context.chunks}
                already_present_count = 0
                for doc in parent_docs:
                    parent_id = doc.metadata.get("chunk_id", "")
                    if parent_id not in seen_ids:
                        parent_chunk = {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata,
                            "chunk_id": parent_id,
                            "is_parent": True,
                            "is_context_expansion": True,
                        }
                        parent_chunks.append(parent_chunk)
                        seen_ids.add(parent_id)
                        sources.add(doc.metadata.get("source", "Unknown"))
                        content_preview = doc.page_content[:80].replace('\n', ' ') if doc.page_content else ""
                        logger.debug(f"[UnifiedRetriever] + Added parent chunk: {parent_id}, content='{content_preview}...'")
                    else:
                        already_present_count += 1
                        logger.debug(f"[UnifiedRetriever] - Parent chunk {parent_id} already in results, skipping")

                if parent_chunks:
                    logger.info(
                        f"[UnifiedRetriever] ✓ Added {len(parent_chunks)} parent chunks for context expansion "
                        f"({already_present_count} already present, skipped)"
                    )
            elif not PARENT_CHUNK_AUTO_INCLUDE:
                logger.debug("[UnifiedRetriever] PARENT_CHUNK_AUTO_INCLUDE is disabled, skipping parent fetch")
            elif not child_chunk_ids:
                logger.debug("[UnifiedRetriever] No child chunks found in graph results, no parent fetch needed")

            return RetrievalResult(
                entities=context.entities,
                relationships=context.relationships,
                chunks=context.chunks,
                parent_chunks=parent_chunks,
                sources=sources,
                query_used=query,
                retrieval_mode=mode
            )

        except Exception as e:
            logger.error(f"[UnifiedRetriever] Graph retrieval failed: {e}, falling back to vector-only")
            return await self._vector_only_retrieval(query, top_k)

    def merge_results(
        self,
        *results: RetrievalResult,
        deduplicate: bool = True
    ) -> RetrievalResult:
        """
        Merge multiple retrieval results, optionally deduplicating.

        Args:
            *results: Variable number of RetrievalResult objects to merge
            deduplicate: Whether to deduplicate chunks by chunk_id

        Returns:
            Merged RetrievalResult
        """
        all_entities = []
        all_relationships = []
        all_chunks = []
        all_parent_chunks = []
        all_sources = set()
        seen_entity_ids = set()
        seen_chunk_ids = set()
        seen_parent_chunk_ids = set()

        for result in results:
            # Merge entities (deduplicate by id)
            for entity in result.entities:
                entity_id = entity.get("id")
                if entity_id and entity_id not in seen_entity_ids:
                    all_entities.append(entity)
                    seen_entity_ids.add(entity_id)
                elif not entity_id:
                    all_entities.append(entity)

            # Merge relationships (no dedup, may have duplicates)
            all_relationships.extend(result.relationships)

            # Merge chunks (deduplicate by chunk_id if enabled)
            for chunk in result.chunks:
                chunk_id = chunk.get("chunk_id", "")
                if deduplicate and chunk_id:
                    if chunk_id not in seen_chunk_ids:
                        all_chunks.append(chunk)
                        seen_chunk_ids.add(chunk_id)
                else:
                    all_chunks.append(chunk)

            # Merge parent chunks (deduplicate by chunk_id)
            for parent_chunk in result.parent_chunks:
                parent_id = parent_chunk.get("chunk_id", "")
                if parent_id and parent_id not in seen_parent_chunk_ids:
                    all_parent_chunks.append(parent_chunk)
                    seen_parent_chunk_ids.add(parent_id)

            # Merge sources
            all_sources.update(result.sources)

        return RetrievalResult(
            entities=all_entities,
            relationships=all_relationships,
            chunks=all_chunks,
            parent_chunks=all_parent_chunks,
            sources=all_sources,
            query_used=results[-1].query_used if results else "",
            retrieval_mode="merged"
        )
