"""
Query Cache Manager - Qdrant-based Semantic Query Cache

Manages semantic query caching using Qdrant with permission-based validation.

Features:
- Semantic similarity search for questions
- Permission-based validation (users only get cached answers from documents they can access)
- TTL-based expiration
- Confidence scoring with negative feedback tracking
- Background async storage (non-blocking)
- Workspace isolation
"""

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http import models

from .query_cache_config import QueryCacheConfig, get_query_cache_config

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry."""
    id: str
    question: str
    normalized_question: str
    answer: str
    source_document_ids: List[str]
    intent: str
    confidence_score: float
    hit_count: int
    negative_feedback_count: int
    created_at: float
    last_accessed_at: float
    expires_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheSearchResult:
    """Result from cache search."""
    cache_hit: bool
    entry: Optional[CacheEntry] = None
    similarity_score: float = 0.0
    permission_granted: bool = False
    search_time_ms: float = 0.0
    candidates_checked: int = 0


class QueryCacheManager:
    """
    Manages semantic query caching using Qdrant with permission-based validation.

    Key Features:
    - Semantic similarity search for question matching
    - Permission validation: user must have READ access to ALL source documents
    - TTL-based expiration
    - Confidence scoring with negative feedback tracking
    - Workspace isolation via collection naming
    """

    def __init__(
        self,
        qdrant_client: Optional[QdrantClient] = None,
        embedding_service: Any = None,
        config: Optional[QueryCacheConfig] = None
    ):
        """
        Initialize the Query Cache Manager.

        Args:
            qdrant_client: Qdrant client instance (optional, will create if not provided)
            embedding_service: Embedding service with embed_query() method
            config: Cache configuration
        """
        self._client = qdrant_client
        self._embeddings = embedding_service
        self.config = config or get_query_cache_config()
        self._initialized_collections: Set[str] = set()

    def _get_client(self) -> QdrantClient:
        """Get or create the Qdrant client."""
        if self._client is None:
            from rag_service.vectorstore import get_qdrant_client
            self._client = get_qdrant_client()
        return self._client

    def _get_embeddings(self):
        """Get or create the embeddings service."""
        if self._embeddings is None:
            from rag_service.vectorstore import get_embeddings
            self._embeddings = get_embeddings()
        return self._embeddings

    def _get_collection_name(self, workspace_id: str) -> str:
        """Generate collection name for workspace."""
        return f"{self.config.collection_prefix}{workspace_id}"

    def _normalize_question(self, question: str) -> str:
        """Normalize question for consistent matching."""
        normalized = question.lower().strip()
        normalized = " ".join(normalized.split())
        return normalized

    def _hash_string(self, text: str) -> str:
        """Generate SHA256 hash of text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _ensure_collection_exists(self, workspace_id: str) -> bool:
        """
        Ensure the cache collection exists for a workspace.

        Args:
            workspace_id: The workspace ID

        Returns:
            True if collection exists or was created successfully
        """
        collection_name = self._get_collection_name(workspace_id)

        if collection_name in self._initialized_collections:
            return True

        client = self._get_client()

        try:
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]

            if collection_name not in collection_names:
                logger.info(f"[QueryCache] Creating cache collection: {collection_name}")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.config.embedding_dimension,
                        distance=models.Distance.COSINE,
                    ),
                )

                # Create indexes for efficient filtering
                self._create_indexes(collection_name)

                logger.info(f"[QueryCache] Collection {collection_name} created successfully")

            self._initialized_collections.add(collection_name)
            return True

        except Exception as e:
            logger.error(f"[QueryCache] Error ensuring collection exists: {e}")
            return False

    def _create_indexes(self, collection_name: str):
        """Create payload indexes for efficient filtering."""
        client = self._get_client()

        indexes = [
            ("question_hash", models.PayloadSchemaType.KEYWORD),
            ("source_document_ids", models.PayloadSchemaType.KEYWORD),
            ("intent", models.PayloadSchemaType.KEYWORD),
            ("expires_at", models.PayloadSchemaType.FLOAT),
            ("is_valid", models.PayloadSchemaType.BOOL),
        ]

        for field_name, field_type in indexes:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
            except Exception as e:
                # Index might already exist
                logger.debug(f"[QueryCache] Index creation note for {field_name}: {e}")

    def search_cache(
        self,
        question: str,
        workspace_id: str,
        user_accessible_doc_ids: List[str],
        intent: Optional[str] = None
    ) -> CacheSearchResult:
        """
        Search for a cached answer with permission validation.

        The search process:
        1. Generate embedding for the question
        2. Find semantically similar cached questions
        3. For each candidate, check if user has access to ALL source documents
        4. Return first candidate that passes permission check

        Args:
            question: The question to search for
            workspace_id: Workspace ID for collection isolation
            user_accessible_doc_ids: List of document IDs the user can access
            intent: Optional intent filter

        Returns:
            CacheSearchResult with cache hit status and entry if found
        """
        start_time = time.time()

        if not self.config.cache_enabled:
            return CacheSearchResult(
                cache_hit=False,
                search_time_ms=(time.time() - start_time) * 1000
            )

        # Ensure collection exists
        if not self._ensure_collection_exists(workspace_id):
            return CacheSearchResult(
                cache_hit=False,
                search_time_ms=(time.time() - start_time) * 1000
            )

        collection_name = self._get_collection_name(workspace_id)
        client = self._get_client()
        embeddings = self._get_embeddings()

        try:
            # Generate question embedding
            normalized_question = self._normalize_question(question)
            question_embedding = embeddings.embed_query(normalized_question)

            # Get similarity threshold
            threshold = self.config.get_similarity_threshold(intent or "default")

            # Build filter for valid, non-expired entries
            now = time.time()
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="is_valid",
                        match=models.MatchValue(value=True)
                    ),
                    models.FieldCondition(
                        key="expires_at",
                        range=models.Range(gt=now)
                    )
                ]
            )

            # Add intent filter if specified
            if intent:
                query_filter.must.append(
                    models.FieldCondition(
                        key="intent",
                        match=models.MatchValue(value=intent)
                    )
                )

            # Search for similar questions using query_points (qdrant-client >= 1.7.0)
            results = client.query_points(
                collection_name=collection_name,
                query=question_embedding,
                query_filter=query_filter,
                limit=self.config.search_limit,
                score_threshold=threshold,
                with_payload=True
            ).points

            logger.info(
                f"[QueryCache] Search found {len(results)} candidates "
                f"(threshold: {threshold}, workspace: {workspace_id})"
            )

            # Check each candidate with permission validation
            user_doc_set = set(user_accessible_doc_ids)
            candidates_checked = 0

            for result in results:
                candidates_checked += 1
                payload = result.payload or {}
                source_doc_ids = payload.get("source_document_ids", [])

                # Permission check: user must have access to ALL source documents
                required_docs = set(source_doc_ids)

                if required_docs.issubset(user_doc_set):
                    # Permission granted - return this cache entry
                    entry = self._payload_to_cache_entry(result.id, payload)

                    # Update access stats (async/background in production)
                    self._update_access_stats(collection_name, result.id)

                    logger.info(
                        f"[QueryCache] CACHE HIT - similarity: {result.score:.3f}, "
                        f"checked {candidates_checked} candidates"
                    )

                    return CacheSearchResult(
                        cache_hit=True,
                        entry=entry,
                        similarity_score=result.score,
                        permission_granted=True,
                        search_time_ms=(time.time() - start_time) * 1000,
                        candidates_checked=candidates_checked
                    )
                else:
                    # Permission denied - try next candidate
                    missing_docs = required_docs - user_doc_set
                    logger.debug(
                        f"[QueryCache] Candidate rejected - user missing access to "
                        f"{len(missing_docs)} documents"
                    )

            # No valid cache hit found
            logger.info(
                f"[QueryCache] CACHE MISS - checked {candidates_checked} candidates, "
                f"no permission-valid match"
            )

            return CacheSearchResult(
                cache_hit=False,
                search_time_ms=(time.time() - start_time) * 1000,
                candidates_checked=candidates_checked
            )

        except Exception as e:
            logger.error(f"[QueryCache] Search error: {e}")
            return CacheSearchResult(
                cache_hit=False,
                search_time_ms=(time.time() - start_time) * 1000
            )

    def store_cache_entry(
        self,
        question: str,
        answer: str,
        workspace_id: str,
        source_document_ids: List[str],
        intent: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Store a new cache entry.

        This is designed to be called in a non-blocking manner (background task).

        Args:
            question: The question (should be enhanced/self-contained)
            answer: The generated answer
            workspace_id: Workspace ID
            source_document_ids: List of document IDs used to generate the answer
            intent: Query intent type
            metadata: Additional metadata to store

        Returns:
            Cache entry ID if successful, None otherwise
        """
        if not self.config.cache_enabled:
            return None

        # Check answer length limit
        if len(answer) > self.config.max_answer_length:
            logger.warning(
                f"[QueryCache] Answer too long ({len(answer)} chars), "
                f"not caching"
            )
            return None

        # Ensure collection exists
        if not self._ensure_collection_exists(workspace_id):
            return None

        collection_name = self._get_collection_name(workspace_id)
        client = self._get_client()
        embeddings = self._get_embeddings()

        try:
            # Normalize and hash the question
            normalized_question = self._normalize_question(question)
            question_hash = self._hash_string(normalized_question)

            # Generate embedding
            question_embedding = embeddings.embed_query(normalized_question)

            # Calculate expiration
            ttl = self.config.get_ttl(intent)
            now = time.time()
            expires_at = now + ttl

            # Generate entry ID
            entry_id = str(uuid.uuid4())

            # Build payload
            payload = {
                "question": question,
                "normalized_question": normalized_question,
                "question_hash": question_hash,
                "answer": answer,
                "source_document_ids": source_document_ids,
                "intent": intent,
                "confidence_score": self.config.initial_confidence_score,
                "hit_count": 0,
                "negative_feedback_count": 0,
                "created_at": now,
                "last_accessed_at": now,
                "expires_at": expires_at,
                "is_valid": True,
                "metadata": metadata or {}
            }

            # Upsert to Qdrant
            client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=entry_id,
                        vector=question_embedding,
                        payload=payload
                    )
                ]
            )

            logger.info(
                f"[QueryCache] Stored cache entry: {entry_id[:8]}... "
                f"(intent: {intent}, TTL: {ttl}s, docs: {len(source_document_ids)})"
            )

            return entry_id

        except Exception as e:
            logger.error(f"[QueryCache] Error storing cache entry: {e}")
            return None

    def invalidate_by_question(
        self,
        question: str,
        workspace_id: str
    ) -> int:
        """
        Invalidate cache entries matching a question.

        Args:
            question: The question to invalidate
            workspace_id: Workspace ID

        Returns:
            Number of entries invalidated
        """
        collection_name = self._get_collection_name(workspace_id)

        # Check if collection exists - if not, nothing to invalidate
        if collection_name not in self._initialized_collections:
            try:
                client = self._get_client()
                collections = client.get_collections().collections
                collection_names = [c.name for c in collections]
                if collection_name not in collection_names:
                    logger.debug(f"[QueryCache] Collection {collection_name} doesn't exist - nothing to invalidate")
                    return 0
            except Exception:
                return 0

        client = self._get_client()

        try:
            normalized_question = self._normalize_question(question)
            question_hash = self._hash_string(normalized_question)

            # Find and update matching entries
            result = client.set_payload(
                collection_name=collection_name,
                payload={"is_valid": False},
                points=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="question_hash",
                                match=models.MatchValue(value=question_hash)
                            )
                        ]
                    )
                )
            )

            logger.info(f"[QueryCache] Invalidated entries for question hash: {question_hash[:16]}...")
            return 1  # Qdrant doesn't return count for set_payload

        except Exception as e:
            logger.error(f"[QueryCache] Error invalidating by question: {e}")
            return 0

    def invalidate_by_document(
        self,
        document_id: str,
        workspace_id: str
    ) -> int:
        """
        Invalidate all cache entries that used a specific document.

        Call this when a document is updated or deleted.

        Args:
            document_id: The document ID
            workspace_id: Workspace ID

        Returns:
            Number of entries invalidated
        """
        collection_name = self._get_collection_name(workspace_id)

        # Check if collection exists - if not, nothing to invalidate
        if collection_name not in self._initialized_collections:
            try:
                client = self._get_client()
                collections = client.get_collections().collections
                collection_names = [c.name for c in collections]
                if collection_name not in collection_names:
                    logger.debug(f"[QueryCache] Collection {collection_name} doesn't exist - nothing to invalidate")
                    return 0
            except Exception:
                return 0

        client = self._get_client()

        try:
            # Find entries that use this document
            result = client.set_payload(
                collection_name=collection_name,
                payload={"is_valid": False},
                points=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_document_ids",
                                match=models.MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )

            logger.info(f"[QueryCache] Invalidated entries for document: {document_id}")
            return 1

        except Exception as e:
            logger.error(f"[QueryCache] Error invalidating by document: {e}")
            return 0

    def record_negative_feedback(
        self,
        entry_id: str,
        workspace_id: str
    ) -> bool:
        """
        Record negative feedback for a cache entry.

        Decreases confidence score and may invalidate if threshold exceeded.

        Args:
            entry_id: Cache entry ID
            workspace_id: Workspace ID

        Returns:
            True if successful
        """
        collection_name = self._get_collection_name(workspace_id)
        client = self._get_client()

        try:
            # Get current entry
            points = client.retrieve(
                collection_name=collection_name,
                ids=[entry_id],
                with_payload=True
            )

            if not points:
                return False

            payload = points[0].payload or {}
            current_confidence = payload.get("confidence_score", 1.0)
            current_negative = payload.get("negative_feedback_count", 0)

            # Update scores
            new_confidence = max(
                0.0,
                current_confidence - self.config.confidence_decay_on_negative
            )
            new_negative = current_negative + 1

            # Check if should invalidate
            is_valid = (
                new_confidence >= self.config.min_confidence_score and
                new_negative < self.config.max_negative_feedback_count
            )

            # Update entry
            client.set_payload(
                collection_name=collection_name,
                payload={
                    "confidence_score": new_confidence,
                    "negative_feedback_count": new_negative,
                    "is_valid": is_valid
                },
                points=[entry_id]
            )

            logger.info(
                f"[QueryCache] Recorded negative feedback for {entry_id[:8]}...: "
                f"confidence {current_confidence:.2f} → {new_confidence:.2f}, "
                f"is_valid: {is_valid}"
            )

            return True

        except Exception as e:
            logger.error(f"[QueryCache] Error recording negative feedback: {e}")
            return False

    def cleanup_expired(self, workspace_id: str) -> int:
        """
        Clean up expired cache entries.

        Args:
            workspace_id: Workspace ID

        Returns:
            Number of entries deleted
        """
        collection_name = self._get_collection_name(workspace_id)
        client = self._get_client()

        try:
            now = time.time()

            result = client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        should=[
                            models.FieldCondition(
                                key="expires_at",
                                range=models.Range(lt=now)
                            ),
                            models.FieldCondition(
                                key="is_valid",
                                match=models.MatchValue(value=False)
                            )
                        ]
                    )
                )
            )

            logger.info(f"[QueryCache] Cleaned up expired entries in {collection_name}")
            return 1

        except Exception as e:
            logger.error(f"[QueryCache] Error cleaning up expired entries: {e}")
            return 0

    def get_cache_stats(self, workspace_id: str) -> Dict[str, Any]:
        """
        Get cache statistics for a workspace.

        Args:
            workspace_id: Workspace ID

        Returns:
            Dictionary with cache statistics
        """
        collection_name = self._get_collection_name(workspace_id)
        client = self._get_client()

        try:
            info = client.get_collection(collection_name)
            return {
                "collection_name": collection_name,
                "total_entries": info.points_count,
                "status": info.status.value if hasattr(info.status, 'value') else str(info.status),
            }
        except Exception as e:
            return {
                "collection_name": collection_name,
                "error": str(e)
            }

    def _payload_to_cache_entry(self, entry_id: str, payload: Dict[str, Any]) -> CacheEntry:
        """Convert Qdrant payload to CacheEntry."""
        return CacheEntry(
            id=str(entry_id),
            question=payload.get("question", ""),
            normalized_question=payload.get("normalized_question", ""),
            answer=payload.get("answer", ""),
            source_document_ids=payload.get("source_document_ids", []),
            intent=payload.get("intent", "general"),
            confidence_score=payload.get("confidence_score", 1.0),
            hit_count=payload.get("hit_count", 0),
            negative_feedback_count=payload.get("negative_feedback_count", 0),
            created_at=payload.get("created_at", 0),
            last_accessed_at=payload.get("last_accessed_at", 0),
            expires_at=payload.get("expires_at", 0),
            metadata=payload.get("metadata", {})
        )

    def _update_access_stats(self, collection_name: str, entry_id: str):
        """Update access statistics for a cache entry."""
        try:
            client = self._get_client()

            # Get current hit count
            points = client.retrieve(
                collection_name=collection_name,
                ids=[entry_id],
                with_payload=True
            )

            if points:
                current_hits = points[0].payload.get("hit_count", 0)
                client.set_payload(
                    collection_name=collection_name,
                    payload={
                        "hit_count": current_hits + 1,
                        "last_accessed_at": time.time()
                    },
                    points=[entry_id]
                )
        except Exception as e:
            # Non-critical, log and continue
            logger.debug(f"[QueryCache] Failed to update access stats: {e}")


# Singleton instance
_cache_manager: Optional[QueryCacheManager] = None


def get_query_cache_manager() -> QueryCacheManager:
    """Get or create the query cache manager singleton."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = QueryCacheManager()
    return _cache_manager
