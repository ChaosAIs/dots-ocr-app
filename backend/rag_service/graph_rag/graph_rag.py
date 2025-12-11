"""
GraphRAG Orchestrator - Main entry point for graph-based retrieval.

This module orchestrates the GraphRAG query pipeline:
1. Query mode detection (LOCAL, GLOBAL, HYBRID, NAIVE)
2. Entity/relationship retrieval based on mode
3. Context building from graph and vector stores
4. Response generation with graph-enhanced context
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .base import QueryMode, QueryParam
from .query_mode_detector import QueryModeDetector
from .utils import extract_entity_names_from_query

logger = logging.getLogger(__name__)

# Feature flag
GRAPH_RAG_ENABLED = os.getenv("GRAPH_RAG_ENABLED", "false").lower() == "true"
DEFAULT_MODE = os.getenv("GRAPH_RAG_DEFAULT_MODE", "auto")


@dataclass
class GraphRAGContext:
    """Context retrieved from GraphRAG."""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    mode: QueryMode
    enhanced_query: str


class GraphRAG:
    """
    Main GraphRAG orchestrator.

    Handles query processing with graph-enhanced retrieval:
    - Detects query mode (or uses specified mode)
    - Retrieves relevant entities and relationships
    - Builds context from graph traversal
    - Returns enhanced context for LLM response generation
    """

    def __init__(
        self,
        workspace_id: str = "default",
        query_mode_detector: QueryModeDetector = None,
    ):
        """
        Initialize GraphRAG.

        Args:
            workspace_id: Workspace ID for multi-tenant isolation
            query_mode_detector: Optional custom query mode detector
        """
        self.workspace_id = workspace_id
        self.query_mode_detector = query_mode_detector or QueryModeDetector()
        self._storage_initialized = False
        self._entity_vector_store = None
        self._edge_vector_store = None
        self._graph_storage = None
        self._kv_storage = None

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

            self._entity_vector_store = QdrantEntityVectorStore(self.workspace_id)
            self._edge_vector_store = QdrantEdgeVectorStore(self.workspace_id)
            self._graph_storage = Neo4jStorage(self.workspace_id)
            self._kv_storage = {
                "entities": PostgresKVStorage("graphrag_entities", self.workspace_id),
                "hyperedges": PostgresKVStorage("graphrag_hyperedges", self.workspace_id),
                "chunks": PostgresKVStorage("graphrag_chunks", self.workspace_id),
            }

            self._storage_initialized = True
            logger.info("GraphRAG storage initialized")

        except Exception as e:
            logger.error(f"Failed to initialize GraphRAG storage: {e}")
            raise

    async def query(
        self,
        query: str,
        mode: str = None,
        params: QueryParam = None,
    ) -> GraphRAGContext:
        """
        Process a query and retrieve graph-enhanced context.

        Args:
            query: User query string
            mode: Optional mode override ("local", "global", "hybrid", "naive", "auto")
            params: Optional query parameters

        Returns:
            GraphRAGContext with entities, relationships, and chunks
        """
        if not GRAPH_RAG_ENABLED:
            logger.debug("GraphRAG is disabled")
            return GraphRAGContext(
                entities=[],
                relationships=[],
                chunks=[],
                mode=QueryMode.NAIVE,
                enhanced_query=query,
            )

        await self._init_storage()

        params = params or QueryParam()

        # Determine query mode
        if mode and mode != "auto":
            query_mode = QueryMode(mode.lower())
            enhanced_query = query
        else:
            query_mode, enhanced_query = await self.query_mode_detector.detect_mode(query)

        logger.info(f"Processing query with mode: {query_mode.value}")

        # Retrieve based on mode
        if query_mode == QueryMode.LOCAL:
            context = await self._local_retrieval(enhanced_query, params)
        elif query_mode == QueryMode.GLOBAL:
            context = await self._global_retrieval(enhanced_query, params)
        elif query_mode == QueryMode.HYBRID:
            context = await self._hybrid_retrieval(enhanced_query, params)
        else:  # NAIVE
            context = await self._naive_retrieval(enhanced_query, params)

        context.mode = query_mode
        context.enhanced_query = enhanced_query

        logger.info(
            f"Retrieved {len(context.entities)} entities, "
            f"{len(context.relationships)} relationships, "
            f"{len(context.chunks)} chunks"
        )

        return context

    async def _local_retrieval(
        self, query: str, params: QueryParam
    ) -> GraphRAGContext:
        """
        LOCAL mode: Entity-focused retrieval.

        1. Find entities matching the query
        2. Get entity descriptions and related chunks
        """
        # Search for entities by vector similarity
        entities = await self._entity_vector_store.query(query, top_k=params.top_k)

        # Get related chunks for top entities
        chunks = []
        for entity in entities[:10]:  # Limit to top 10 entities
            chunk_id = entity.get("source_chunk_id")
            if chunk_id:
                chunk = await self._kv_storage["chunks"].get_by_id(chunk_id)
                if chunk and chunk not in chunks:
                    chunks.append(chunk)

        return GraphRAGContext(
            entities=entities,
            relationships=[],
            chunks=chunks,
            mode=QueryMode.LOCAL,
            enhanced_query=query,
        )

    async def _global_retrieval(
        self, query: str, params: QueryParam
    ) -> GraphRAGContext:
        """
        GLOBAL mode: Relationship-focused retrieval.

        1. Find relationships matching the query
        2. Get connected entities
        3. Traverse graph for related context
        """
        # Search for relationships by vector similarity
        relationships = await self._edge_vector_store.query(query, top_k=params.top_k)

        # Get entities connected by these relationships
        entity_ids = set()
        for rel in relationships:
            if rel.get("src_entity_id"):
                entity_ids.add(rel["src_entity_id"])
            if rel.get("tgt_entity_id"):
                entity_ids.add(rel["tgt_entity_id"])

        # Fetch entity details
        entities = []
        for entity_id in list(entity_ids)[:20]:  # Limit
            entity = await self._kv_storage["entities"].get_by_id(entity_id)
            if entity:
                entities.append(entity)

        # Get chunks from entities
        chunks = []
        for entity in entities[:10]:
            chunk_id = entity.get("source_chunk_id")
            if chunk_id:
                chunk = await self._kv_storage["chunks"].get_by_id(chunk_id)
                if chunk and chunk not in chunks:
                    chunks.append(chunk)

        return GraphRAGContext(
            entities=entities,
            relationships=relationships,
            chunks=chunks,
            mode=QueryMode.GLOBAL,
            enhanced_query=query,
        )

    async def _hybrid_retrieval(
        self, query: str, params: QueryParam
    ) -> GraphRAGContext:
        """
        HYBRID mode: Combined entity and relationship retrieval.

        1. Find entities matching the query
        2. Find relationships matching the query
        3. Traverse graph from matched entities
        4. Combine all context
        """
        # Get both entities and relationships
        entities = await self._entity_vector_store.query(query, top_k=params.top_k)
        relationships = await self._edge_vector_store.query(query, top_k=params.top_k)

        # Expand entities via graph traversal
        expanded_entities = list(entities)
        for entity in entities[:5]:  # Limit traversal
            entity_id = entity.get("id")
            if entity_id:
                # Get 1-hop neighbors
                edges = await self._graph_storage.get_node_edges(entity_id)
                for src_id, tgt_id, edge_data in edges:
                    neighbor_id = tgt_id if src_id == entity_id else src_id
                    neighbor = await self._kv_storage["entities"].get_by_id(neighbor_id)
                    if neighbor and neighbor not in expanded_entities:
                        expanded_entities.append(neighbor)

        # Get chunks
        chunks = []
        seen_chunk_ids = set()
        for entity in expanded_entities[:15]:
            chunk_id = entity.get("source_chunk_id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                chunk = await self._kv_storage["chunks"].get_by_id(chunk_id)
                if chunk:
                    chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)

        return GraphRAGContext(
            entities=expanded_entities[:params.top_k],
            relationships=relationships,
            chunks=chunks,
            mode=QueryMode.HYBRID,
            enhanced_query=query,
        )

    async def _naive_retrieval(
        self, query: str, params: QueryParam
    ) -> GraphRAGContext:
        """
        NAIVE mode: Simple vector search without graph.

        Falls back to basic vector similarity search.
        """
        # Just return empty graph context - let the regular RAG handle it
        return GraphRAGContext(
            entities=[],
            relationships=[],
            chunks=[],
            mode=QueryMode.NAIVE,
            enhanced_query=query,
        )

    def format_context(self, context: GraphRAGContext) -> str:
        """
        Format GraphRAG context for LLM consumption.

        Args:
            context: GraphRAGContext from query()

        Returns:
            Formatted string with entities, relationships, and chunks
        """
        parts = []

        if context.entities:
            parts.append("## Relevant Entities\n")
            for entity in context.entities[:10]:
                name = entity.get("name", "Unknown")
                entity_type = entity.get("entity_type", "")
                description = entity.get("description", "")
                parts.append(f"- **{name}** ({entity_type}): {description}\n")

        if context.relationships:
            parts.append("\n## Relationships\n")
            for rel in context.relationships[:10]:
                src = rel.get("src_name", "?")
                tgt = rel.get("tgt_name", "?")
                desc = rel.get("description", "related to")
                parts.append(f"- {src} → {tgt}: {desc}\n")

        if context.chunks:
            parts.append("\n## Related Content\n")
            for chunk in context.chunks[:5]:
                content = chunk.get("content", "")[:500]
                parts.append(f"{content}\n---\n")

        return "".join(parts) if parts else ""

