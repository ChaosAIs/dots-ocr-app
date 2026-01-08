"""
Bridge to existing vector search service.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class VectorServiceBridge:
    """Bridge to existing vector search (Qdrant) service."""

    def __init__(self):
        self._vector_service = None
        self._qdrant_client = None

    @property
    def vector_service(self):
        """Lazy load vector service."""
        if self._vector_service is None:
            try:
                from backend.rag_service.vector_store import VectorStoreService
                self._vector_service = VectorStoreService()
            except ImportError:
                logger.warning("VectorStoreService not available")
        return self._vector_service

    @property
    def qdrant_client(self):
        """Lazy load Qdrant client."""
        if self._qdrant_client is None:
            try:
                from backend.analytics_service.query_cache_manager import QueryCacheManager
                cache_manager = QueryCacheManager()
                self._qdrant_client = cache_manager.qdrant_client
            except ImportError:
                logger.warning("Qdrant client not available")
        return self._qdrant_client

    def semantic_search(
        self,
        query: str,
        workspace_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform semantic search using vector embeddings.

        Args:
            query: Search query
            workspace_id: Workspace context
            document_ids: Optional document ID filter
            top_k: Number of results
            filters: Additional filters

        Returns:
            Search results with scores
        """
        if self.vector_service:
            try:
                # Build filters
                search_filters = filters or {}
                if document_ids:
                    search_filters["document_id"] = {"$in": document_ids}

                results = self.vector_service.search(
                    query=query,
                    workspace_id=workspace_id,
                    top_k=top_k,
                    filters=search_filters
                )

                return {
                    "success": True,
                    "documents": results.get("documents", []),
                    "scores": results.get("scores", []),
                    "total_found": len(results.get("documents", []))
                }

            except Exception as e:
                logger.error(f"Vector search failed: {e}")

        # Fallback: Return empty results
        return {
            "success": False,
            "documents": [],
            "scores": [],
            "total_found": 0,
            "error": "Vector service not available"
        }

    def get_document_embeddings(
        self,
        document_id: str,
        workspace_id: str
    ) -> Optional[List[float]]:
        """Get embeddings for a document.

        Args:
            document_id: Document ID
            workspace_id: Workspace context

        Returns:
            Embedding vector or None
        """
        if self.vector_service:
            try:
                return self.vector_service.get_embeddings(
                    document_id=document_id,
                    workspace_id=workspace_id
                )
            except Exception as e:
                logger.error(f"Error getting embeddings: {e}")

        return None

    def similarity_search_by_vector(
        self,
        vector: List[float],
        workspace_id: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search by vector similarity.

        Args:
            vector: Query vector
            workspace_id: Workspace context
            top_k: Number of results
            filters: Additional filters

        Returns:
            Search results
        """
        if self.vector_service:
            try:
                return self.vector_service.search_by_vector(
                    vector=vector,
                    workspace_id=workspace_id,
                    top_k=top_k,
                    filters=filters
                )
            except Exception as e:
                logger.error(f"Vector similarity search failed: {e}")

        return {
            "success": False,
            "documents": [],
            "error": "Vector service not available"
        }
