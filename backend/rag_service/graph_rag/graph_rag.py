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
        self._graph_storage = None

    async def _init_storage(self):
        """Initialize Neo4j storage backend lazily."""
        if self._storage_initialized:
            return

        try:
            from ..storage import Neo4jStorage

            self._graph_storage = Neo4jStorage(self.workspace_id)

            self._storage_initialized = True
            logger.info("[GraphRAG Query] Neo4j storage initialized")

        except Exception as e:
            logger.error(f"[GraphRAG Query] Failed to initialize Neo4j storage: {e}")
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
        LOCAL mode: Entity-focused retrieval from Neo4j.

        1. Extract entity names from query
        2. Find matching entities in Neo4j graph
        3. Get entity descriptions
        """
        # Extract potential entity names from query
        entity_names = extract_entity_names_from_query(query)
        logger.debug(f"[GraphRAG Query] LOCAL mode - entity names from query: {entity_names}")

        # Search for entities in Neo4j by name matching
        entities = []
        for name in entity_names[:10]:  # Limit to top 10 names
            try:
                found = await self._graph_storage.get_nodes_by_name(name, limit=5)
                for entity in found:
                    if entity not in entities:
                        entities.append(entity)
            except Exception as e:
                logger.warning(f"[GraphRAG Query] Error finding entity '{name}': {e}")

        logger.debug(f"[GraphRAG Query] LOCAL mode - found {len(entities)} entities")

        return GraphRAGContext(
            entities=entities[:params.top_k],
            relationships=[],
            chunks=[],
            mode=QueryMode.LOCAL,
            enhanced_query=query,
        )

    async def _global_retrieval(
        self, query: str, params: QueryParam
    ) -> GraphRAGContext:
        """
        GLOBAL mode: Relationship-focused retrieval from Neo4j.

        1. Extract entity names from query
        2. Find matching entities and their relationships
        3. Get connected entities via graph traversal
        """
        # Extract potential entity names from query
        entity_names = extract_entity_names_from_query(query)
        logger.debug(f"[GraphRAG Query] GLOBAL mode - entity names from query: {entity_names}")

        entities = []
        relationships = []

        # Find entities and their relationships
        for name in entity_names[:5]:  # Limit traversal
            try:
                found = await self._graph_storage.get_nodes_by_name(name, limit=3)
                for entity in found:
                    if entity not in entities:
                        entities.append(entity)
                    # Get edges for this entity
                    entity_id = entity.get("id")
                    if entity_id:
                        edges = await self._graph_storage.get_node_edges(entity_id)
                        for src_id, tgt_id, edge_data in edges:
                            rel = {
                                "src_entity_id": src_id,
                                "tgt_entity_id": tgt_id,
                                "description": edge_data.get("description", ""),
                                "keywords": edge_data.get("keywords", ""),
                            }
                            if rel not in relationships:
                                relationships.append(rel)
            except Exception as e:
                logger.warning(f"[GraphRAG Query] Error finding entity '{name}': {e}")

        logger.debug(
            f"[GraphRAG Query] GLOBAL mode - found {len(entities)} entities, "
            f"{len(relationships)} relationships"
        )

        return GraphRAGContext(
            entities=entities[:params.top_k],
            relationships=relationships[:params.top_k],
            chunks=[],
            mode=QueryMode.GLOBAL,
            enhanced_query=query,
        )

    async def _hybrid_retrieval(
        self, query: str, params: QueryParam
    ) -> GraphRAGContext:
        """
        HYBRID mode: Combined entity and relationship retrieval from Neo4j.

        1. Find entities matching the query
        2. Get their relationships
        3. Traverse graph from matched entities to get neighbors
        """
        # Extract potential entity names from query
        entity_names = extract_entity_names_from_query(query)
        logger.debug(f"[GraphRAG Query] HYBRID mode - entity names from query: {entity_names}")

        entities = []
        relationships = []
        seen_entity_ids = set()

        # Find entities and expand via graph traversal
        for name in entity_names[:5]:  # Limit initial search
            try:
                found = await self._graph_storage.get_nodes_by_name(name, limit=3)
                for entity in found:
                    entity_id = entity.get("id")
                    if entity_id and entity_id not in seen_entity_ids:
                        entities.append(entity)
                        seen_entity_ids.add(entity_id)

                        # Get 1-hop neighbors
                        edges = await self._graph_storage.get_node_edges(entity_id)
                        for src_id, tgt_id, edge_data in edges:
                            # Add relationship
                            rel = {
                                "src_entity_id": src_id,
                                "tgt_entity_id": tgt_id,
                                "description": edge_data.get("description", ""),
                                "keywords": edge_data.get("keywords", ""),
                            }
                            if rel not in relationships:
                                relationships.append(rel)

                            # Add neighbor entity
                            neighbor_id = tgt_id if src_id == entity_id else src_id
                            if neighbor_id not in seen_entity_ids:
                                neighbor = await self._graph_storage.get_node(neighbor_id)
                                if neighbor:
                                    entities.append(neighbor)
                                    seen_entity_ids.add(neighbor_id)
            except Exception as e:
                logger.warning(f"[GraphRAG Query] Error in hybrid retrieval for '{name}': {e}")

        logger.debug(
            f"[GraphRAG Query] HYBRID mode - found {len(entities)} entities, "
            f"{len(relationships)} relationships"
        )

        return GraphRAGContext(
            entities=entities[:params.top_k],
            relationships=relationships[:params.top_k],
            chunks=[],
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

