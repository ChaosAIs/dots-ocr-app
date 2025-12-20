"""
Unified Retriever - Abstracted retrieval interface for iterative reasoning.

This module provides a unified retrieval interface that works with or without GraphRAG,
allowing the iterative reasoning engine to use the same logic regardless of backend.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Unified result from retrieval operations.

    Contains entities, relationships, and chunks regardless of backend.
    When GraphRAG is disabled, entities and relationships will be empty.
    """
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
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
        workspace_id: str = "default"
    ):
        """
        Initialize the unified retriever.

        Args:
            graphrag_enabled: Whether to use GraphRAG (Neo4j + Qdrant) or vector-only (Qdrant)
            source_names: Optional list of source document names to filter results
            workspace_id: Workspace ID for multi-tenant isolation
        """
        self.graphrag_enabled = graphrag_enabled
        self.source_names = source_names or []
        self.workspace_id = workspace_id
        self._graphrag = None
        self._initialized = False

    async def _init_graphrag(self):
        """Lazily initialize GraphRAG if enabled."""
        if self._initialized:
            return

        if self.graphrag_enabled:
            try:
                from .graph_rag import GraphRAG
                self._graphrag = GraphRAG(workspace_id=self.workspace_id)
                logger.info("[UnifiedRetriever] GraphRAG backend initialized")
            except Exception as e:
                logger.warning(f"[UnifiedRetriever] Failed to init GraphRAG: {e}, falling back to vector-only")
                self.graphrag_enabled = False
                self._graphrag = None

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
        Vector-only retrieval using Qdrant.

        This is used when GraphRAG is disabled. Returns chunks only,
        with empty entities and relationships.

        Args:
            query: Search query string
            top_k: Maximum number of chunks to retrieve

        Returns:
            RetrievalResult with chunks only (no entities/relationships)
        """
        logger.info(f"[UnifiedRetriever] Vector-only retrieval for: '{query[:50]}...'")

        try:
            from .vectorstore import get_vectorstore

            vectorstore = get_vectorstore()

            # Build search kwargs with optional source filtering
            search_kwargs = {"k": top_k}

            # Add source filter if specified
            if self.source_names and len(self.source_names) > 0:
                from qdrant_client import models
                search_kwargs["filter"] = models.Filter(
                    should=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=source_name),
                        )
                        for source_name in self.source_names
                    ]
                )
                logger.debug(f"[UnifiedRetriever] Filtering to sources: {self.source_names}")

            # Perform vector search
            docs = vectorstore.similarity_search(query, **search_kwargs)

            # Convert to unified chunk format
            chunks = []
            sources = set()
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                sources.add(source)
                chunks.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "_score": 0,  # Qdrant similarity_search doesn't return scores directly
                })

            logger.info(f"[UnifiedRetriever] Vector search found {len(chunks)} chunks from {len(sources)} sources")

            return RetrievalResult(
                entities=[],
                relationships=[],
                chunks=chunks,
                sources=sources,
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

            # Create query params (single-step, no iteration here - iteration handled by engine)
            params = QueryParam()
            params.max_steps = 1  # Single retrieval step
            params.top_k = top_k
            if self.source_names:
                params.source_names = self.source_names

            # Determine retrieval mode
            if mode == "local":
                context = await self._graphrag._local_retrieval(query, params)
            elif mode == "global":
                context = await self._graphrag._global_retrieval(query, params)
            else:  # hybrid (default)
                context = await self._graphrag._hybrid_retrieval(query, params)

            # Extract sources from chunks
            sources = set()
            for chunk in context.chunks:
                source = chunk.get("metadata", {}).get("source", "")
                if source:
                    sources.add(source)

            logger.info(
                f"[UnifiedRetriever] Graph retrieval found {len(context.entities)} entities, "
                f"{len(context.relationships)} relationships, {len(context.chunks)} chunks"
            )

            return RetrievalResult(
                entities=context.entities,
                relationships=context.relationships,
                chunks=context.chunks,
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
        all_sources = set()
        seen_entity_ids = set()
        seen_chunk_ids = set()

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

            # Merge sources
            all_sources.update(result.sources)

        return RetrievalResult(
            entities=all_entities,
            relationships=all_relationships,
            chunks=all_chunks,
            sources=all_sources,
            query_used=results[-1].query_used if results else "",
            retrieval_mode="merged"
        )
