"""
Qdrant Vector Storage adapters for GraphRAG.

Provides vector storage for:
- Entity embeddings (name + description)
- Hyperedge embeddings (keywords + source + target + description)
"""

import os
import logging
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from ..graph_rag.base import BaseVectorStorage
from ..local_qwen_embedding import LocalQwen3Embedding

logger = logging.getLogger(__name__)

# Collection names
ENTITY_COLLECTION = "entities"
EDGE_COLLECTION = "hyperedges"

# Singleton instances
_client: QdrantClient = None
_embeddings: LocalQwen3Embedding = None


def _get_qdrant_client() -> QdrantClient:
    """Get or create the Qdrant client instance."""
    global _client
    if _client is None:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        _client = QdrantClient(host=host, port=port, check_compatibility=False)
        logger.info(f"Connected to Qdrant at {host}:{port}")
    return _client


def _get_embeddings() -> LocalQwen3Embedding:
    """Get or create the embeddings instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = LocalQwen3Embedding()
    return _embeddings


def _ensure_collection_exists(client: QdrantClient, collection_name: str, dim: int = 2560):
    """Ensure a collection exists, create if not."""
    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if collection_name not in names:
        logger.info(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        logger.info(f"Collection {collection_name} created")


class QdrantEntityVectorStore(BaseVectorStorage):
    """Qdrant vector store for entity embeddings."""

    def __init__(self, workspace_id: str = "default"):
        super().__init__(workspace_id)
        self.collection_name = ENTITY_COLLECTION
        self._client = None
        self._embeddings = None

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = _get_qdrant_client()
            _ensure_collection_exists(self._client, self.collection_name)
        return self._client

    def _get_embeddings(self) -> LocalQwen3Embedding:
        if self._embeddings is None:
            self._embeddings = _get_embeddings()
        return self._embeddings

    async def query(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Query for similar entities."""
        client = self._get_client()
        embeddings = self._get_embeddings()

        # Generate query embedding
        query_vector = embeddings.embed_query(query)

        # Search with workspace filter
        results = client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="workspace_id",
                        match=models.MatchValue(value=self.workspace_id),
                    )
                ]
            ),
            limit=top_k,
            with_payload=True,
        )

        return [
            {"id": hit.id, "score": hit.score, **hit.payload}
            for hit in results
        ]

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Upsert entity vectors."""
        if not data:
            return

        client = self._get_client()
        embeddings = self._get_embeddings()

        points = []
        for entity_id, entity_data in data.items():
            # Build text for embedding: name + description
            text_parts = [entity_data.get("name", "")]
            if entity_data.get("description"):
                text_parts.append(entity_data["description"])
            text = " ".join(text_parts)

            # Generate embedding
            vector = embeddings.embed_query(text)

            # Build payload
            payload = {
                "workspace_id": self.workspace_id,
                "entity_id": entity_id,
                **{k: v for k, v in entity_data.items() if k != "embedding"},
            }

            points.append(models.PointStruct(
                id=entity_id,
                vector=vector,
                payload=payload,
            ))

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=self.collection_name, points=batch)

        logger.info(f"Upserted {len(points)} entities to {self.collection_name}")

    async def delete(self, ids: List[str]) -> None:
        """Delete entity vectors by IDs."""
        if not ids:
            return

        client = self._get_client()
        client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=ids),
        )


class QdrantEdgeVectorStore(BaseVectorStorage):
    """Qdrant vector store for relationship/hyperedge embeddings."""

    def __init__(self, workspace_id: str = "default"):
        super().__init__(workspace_id)
        self.collection_name = EDGE_COLLECTION
        self._client = None
        self._embeddings = None

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = _get_qdrant_client()
            _ensure_collection_exists(self._client, self.collection_name)
        return self._client

    def _get_embeddings(self) -> LocalQwen3Embedding:
        if self._embeddings is None:
            self._embeddings = _get_embeddings()
        return self._embeddings

    async def query(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Query for similar edges/relationships."""
        client = self._get_client()
        embeddings = self._get_embeddings()

        # Generate query embedding
        query_vector = embeddings.embed_query(query)

        # Search with workspace filter
        results = client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="workspace_id",
                        match=models.MatchValue(value=self.workspace_id),
                    )
                ]
            ),
            limit=top_k,
            with_payload=True,
        )

        return [
            {"id": hit.id, "score": hit.score, **hit.payload}
            for hit in results
        ]

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Upsert edge vectors."""
        if not data:
            return

        client = self._get_client()
        embeddings = self._get_embeddings()

        points = []
        for edge_id, edge_data in data.items():
            # Build text for embedding: keywords + src + tgt + description
            text_parts = []
            if edge_data.get("keywords"):
                text_parts.append(edge_data["keywords"])
            if edge_data.get("src_name"):
                text_parts.append(edge_data["src_name"])
            if edge_data.get("tgt_name"):
                text_parts.append(edge_data["tgt_name"])
            if edge_data.get("description"):
                text_parts.append(edge_data["description"])
            text = " ".join(text_parts)

            if not text.strip():
                continue

            # Generate embedding
            vector = embeddings.embed_query(text)

            # Build payload
            payload = {
                "workspace_id": self.workspace_id,
                "edge_id": edge_id,
                **{k: v for k, v in edge_data.items() if k != "embedding"},
            }

            points.append(models.PointStruct(
                id=edge_id,
                vector=vector,
                payload=payload,
            ))

        if not points:
            return

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=self.collection_name, points=batch)

        logger.info(f"Upserted {len(points)} edges to {self.collection_name}")

    async def delete(self, ids: List[str]) -> None:
        """Delete edge vectors by IDs."""
        if not ids:
            return

        client = self._get_client()
        client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=ids),
        )


def delete_entities_by_workspace(workspace_id: str) -> None:
    """Delete all entity vectors for a workspace."""
    client = _get_qdrant_client()
    _ensure_collection_exists(client, ENTITY_COLLECTION)
    client.delete(
        collection_name=ENTITY_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="workspace_id",
                        match=models.MatchValue(value=workspace_id),
                    )
                ]
            )
        ),
    )
    logger.info(f"Deleted entities for workspace: {workspace_id}")


def delete_edges_by_workspace(workspace_id: str) -> None:
    """Delete all edge vectors for a workspace."""
    client = _get_qdrant_client()
    _ensure_collection_exists(client, EDGE_COLLECTION)
    client.delete(
        collection_name=EDGE_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="workspace_id",
                        match=models.MatchValue(value=workspace_id),
                    )
                ]
            )
        ),
    )
    logger.info(f"Deleted edges for workspace: {workspace_id}")


def delete_entities_by_source(workspace_id: str, source_name: str) -> None:
    """
    Delete entity vectors for a specific source document.

    Args:
        workspace_id: Workspace ID
        source_name: Source document name
    """
    client = _get_qdrant_client()
    _ensure_collection_exists(client, ENTITY_COLLECTION)

    # Delete where source_chunk_id starts with source_name
    # Qdrant doesn't support LIKE, so we use text match with prefix
    # We need to scroll and delete matching points
    try:
        # Scroll to find matching points
        offset = None
        deleted_count = 0

        while True:
            points, offset = client.scroll(
                collection_name=ENTITY_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="workspace_id",
                            match=models.MatchValue(value=workspace_id),
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=True,
            )

            if not points:
                break

            # Filter points by source_chunk_id prefix
            ids_to_delete = []
            for point in points:
                source_chunk_id = point.payload.get("source_chunk_id", "")
                if (source_chunk_id.startswith(f"{source_name}_") or
                        source_chunk_id == source_name):
                    ids_to_delete.append(point.id)

            if ids_to_delete:
                client.delete(
                    collection_name=ENTITY_COLLECTION,
                    points_selector=models.PointIdsList(points=ids_to_delete),
                )
                deleted_count += len(ids_to_delete)

            if offset is None:
                break

        if deleted_count > 0:
            logger.info(
                f"Deleted {deleted_count} entities for source: {source_name}"
            )
    except Exception as e:
        logger.error(f"Error deleting entities by source: {e}")


def delete_edges_by_source(workspace_id: str, source_name: str) -> None:
    """
    Delete edge vectors for a specific source document.

    Args:
        workspace_id: Workspace ID
        source_name: Source document name
    """
    client = _get_qdrant_client()
    _ensure_collection_exists(client, EDGE_COLLECTION)

    try:
        # Scroll to find matching points
        offset = None
        deleted_count = 0

        while True:
            points, offset = client.scroll(
                collection_name=EDGE_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="workspace_id",
                            match=models.MatchValue(value=workspace_id),
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=True,
            )

            if not points:
                break

            # Filter points by source_chunk_id prefix
            ids_to_delete = []
            for point in points:
                source_chunk_id = point.payload.get("source_chunk_id", "")
                if (source_chunk_id.startswith(f"{source_name}_") or
                        source_chunk_id == source_name):
                    ids_to_delete.append(point.id)

            if ids_to_delete:
                client.delete(
                    collection_name=EDGE_COLLECTION,
                    points_selector=models.PointIdsList(points=ids_to_delete),
                )
                deleted_count += len(ids_to_delete)

            if offset is None:
                break

        if deleted_count > 0:
            logger.info(
                f"Deleted {deleted_count} edges for source: {source_name}"
            )
    except Exception as e:
        logger.error(f"Error deleting edges by source: {e}")

