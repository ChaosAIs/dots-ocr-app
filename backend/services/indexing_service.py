"""
Indexing service for vector and GraphRAG indexing operations.
"""
import os
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List, Dict
from uuid import UUID

from sqlalchemy.orm import Session

from db.models import Document, IndexStatus
from db.document_repository import DocumentRepository
from db.database import get_db_session

logger = logging.getLogger(__name__)


class IndexingService:
    """
    Service for document indexing operations including:
    - Vector indexing (Qdrant)
    - GraphRAG indexing (Neo4j)
    - Metadata extraction
    - Batch indexing
    """

    def __init__(
        self,
        output_dir: str,
        broadcast_callback: Optional[Callable] = None
    ):
        """
        Initialize indexing service.

        Args:
            output_dir: Base directory for output files
            broadcast_callback: Callback for WebSocket broadcasts
        """
        self.output_dir = output_dir
        self.broadcast_callback = broadcast_callback

        # Batch indexing status
        self._batch_index_status = {
            "status": "idle",
            "total_documents": 0,
            "indexed_documents": 0,
            "current_document": None,
            "started_at": None,
            "completed_at": None,
            "errors": [],
            "message": None,
        }
        self._batch_index_lock = threading.Lock()

        # Import RAG service functions
        try:
            from rag_service.indexer import (
                trigger_embedding_for_document,
                reindex_document,
                index_document_now,
            )
            from rag_service.vectorstore import (
                delete_documents_by_source,
                delete_documents_by_file_path,
                delete_documents_by_document_id,
                get_collection_info,
                clear_collection,
                is_document_indexed,
                delete_document_metadata_embedding,
                delete_metadata_by_source_name
            )
            self._trigger_embedding = trigger_embedding_for_document
            self._reindex_document = reindex_document
            self._index_document_now = index_document_now
            self._delete_by_source = delete_documents_by_source
            self._delete_by_file_path = delete_documents_by_file_path
            self._delete_by_document_id = delete_documents_by_document_id
            self._get_collection_info = get_collection_info
            self._clear_collection = clear_collection
            self._is_document_indexed = is_document_indexed
            self._delete_metadata_embedding = delete_document_metadata_embedding
            self._delete_metadata_by_source = delete_metadata_by_source_name
        except ImportError as e:
            logger.warning(f"RAG service not available: {e}")
            self._trigger_embedding = None
            self._reindex_document = None
            self._index_document_now = None
            self._delete_by_source = None
            self._delete_by_file_path = None
            self._delete_by_document_id = None
            self._get_collection_info = None
            self._clear_collection = None
            self._is_document_indexed = None
            self._delete_metadata_embedding = None
            self._delete_metadata_by_source = None

        # Import GraphRAG functions
        try:
            from rag_service.graph_rag import (
                delete_graphrag_by_source_sync,
                GRAPH_RAG_INDEX_ENABLED,
                GRAPH_RAG_QUERY_ENABLED
            )
            self._delete_graphrag_by_source = delete_graphrag_by_source_sync
            self.graph_rag_index_enabled = GRAPH_RAG_INDEX_ENABLED
            self.graph_rag_query_enabled = GRAPH_RAG_QUERY_ENABLED
        except ImportError:
            self._delete_graphrag_by_source = None
            self.graph_rag_index_enabled = False
            self.graph_rag_query_enabled = False

    def trigger_indexing(
        self,
        source_name: str,
        filename: str,
        conversion_id: str,
        broadcast_callback: Optional[Callable] = None
    ):
        """
        Trigger two-phase indexing for a document.

        Args:
            source_name: Document source name (filename without extension)
            filename: Original filename
            conversion_id: Conversion tracking ID
            broadcast_callback: Callback for progress updates
        """
        if not self._trigger_embedding:
            logger.error("RAG service not available - cannot trigger indexing")
            return

        self._trigger_embedding(
            source_name=source_name,
            output_dir=self.output_dir,
            filename=filename,
            conversion_id=conversion_id,
            broadcast_callback=broadcast_callback or self.broadcast_callback
        )

    def reindex_document(self, filename: str):
        """
        Re-index a document after markdown update.

        Args:
            filename: Document filename
        """
        if not self._reindex_document:
            logger.error("RAG service not available - cannot reindex")
            return

        self._reindex_document(filename, self.output_dir)

    def index_single_document(
        self,
        filename: str,
        doc_dir: str,
        conversion_id: str,
        broadcast_callback: Optional[Callable] = None
    ):
        """
        Index a single document.

        Args:
            filename: Document filename
            doc_dir: Document output directory
            conversion_id: Conversion tracking ID
            broadcast_callback: Callback for progress updates
        """
        if not self._trigger_embedding:
            logger.error("RAG service not available")
            return

        file_name_without_ext = os.path.splitext(filename)[0]

        # Delete existing embeddings first
        if self._delete_by_source:
            try:
                self._delete_by_source(file_name_without_ext)
                logger.info(f"Deleted existing embeddings for: {file_name_without_ext}")
            except Exception as e:
                logger.warning(f"Error deleting existing embeddings: {e}")

        # Trigger indexing
        self._trigger_embedding(
            source_name=file_name_without_ext,
            output_dir=self.output_dir,
            filename=filename,
            conversion_id=conversion_id,
            broadcast_callback=broadcast_callback or self.broadcast_callback
        )

    def delete_document_embeddings(
        self,
        filename: str,
        doc_id: Optional[str] = None,
        output_path: Optional[str] = None
    ):
        """
        Delete all embeddings for a document from Qdrant (vectors and metadata).

        Uses multiple strategies to ensure complete cleanup:
        1. Delete by document_id (most reliable - always present in metadata)
        2. Delete by source name (handles _page_N suffixes)
        3. Delete by file path (if output_path provided)
        4. Delete metadata embedding by document ID
        5. Delete metadata by source name (fallback)

        Args:
            filename: Document filename
            doc_id: Optional document ID for metadata deletion
            output_path: Optional output directory path for file_path based deletion
        """
        file_name_without_ext = os.path.splitext(filename)[0]
        total_deleted = 0

        # Strategy 1: Delete vector embeddings by document_id (MOST RELIABLE)
        # document_id is always stored in metadata during indexing
        if doc_id and self._delete_by_document_id:
            try:
                deleted = self._delete_by_document_id(doc_id)
                total_deleted += deleted if isinstance(deleted, int) else 0
                logger.info(f"[EmbeddingDelete] Deleted {deleted} vector embeddings by document_id: {doc_id}")
            except Exception as e:
                logger.error(f"[EmbeddingDelete] Failed to delete vector embeddings by document_id: {e}")

        # Strategy 2: Delete vector embeddings by source name (fallback)
        if self._delete_by_source:
            try:
                deleted = self._delete_by_source(file_name_without_ext)
                total_deleted += deleted if isinstance(deleted, int) else 0
                if deleted > 0:
                    logger.info(f"[EmbeddingDelete] Deleted {deleted} vector embeddings by source: {file_name_without_ext}")
            except Exception as e:
                logger.error(f"[EmbeddingDelete] Failed to delete vector embeddings by source: {e}")

        # Strategy 3: Delete vector embeddings by file path (if output_path provided)
        if output_path and self._delete_by_file_path:
            try:
                deleted = self._delete_by_file_path(output_path)
                total_deleted += deleted if isinstance(deleted, int) else 0
                if deleted > 0:
                    logger.info(f"[EmbeddingDelete] Deleted {deleted} vector embeddings by file_path: {output_path}")
            except Exception as e:
                logger.warning(f"[EmbeddingDelete] Failed to delete vector embeddings by file_path: {e}")

        # Strategy 4: Delete GraphRAG data
        if self._delete_graphrag_by_source and self.graph_rag_index_enabled:
            try:
                self._delete_graphrag_by_source(file_name_without_ext)
                logger.info(f"[EmbeddingDelete] Deleted GraphRAG data for: {file_name_without_ext}")
            except Exception as e:
                logger.warning(f"[EmbeddingDelete] Failed to delete GraphRAG data: {e}")

        # Strategy 5: Delete metadata embedding by document ID
        if doc_id and self._delete_metadata_embedding:
            try:
                self._delete_metadata_embedding(doc_id)
                logger.info(f"[EmbeddingDelete] Deleted metadata embedding by doc_id: {doc_id}")
            except Exception as e:
                logger.warning(f"[EmbeddingDelete] Failed to delete metadata embedding by doc_id: {e}")

        # Strategy 6: Delete metadata by source name (fallback for orphaned records)
        if self._delete_metadata_by_source:
            try:
                deleted = self._delete_metadata_by_source(file_name_without_ext)
                if deleted > 0:
                    logger.info(f"[EmbeddingDelete] Deleted {deleted} orphaned metadata entries by source: {file_name_without_ext}")
            except Exception as e:
                logger.warning(f"[EmbeddingDelete] Failed to delete metadata by source name: {e}")

        logger.info(f"[EmbeddingDelete] Completed cleanup for '{filename}' (doc_id={doc_id}, total_deleted={total_deleted})")

    def is_document_indexed(self, source_name: str) -> bool:
        """
        Check if a document is indexed.

        Args:
            source_name: Document source name

        Returns:
            True if indexed
        """
        if not self._is_document_indexed:
            return False
        return self._is_document_indexed(source_name)

    def get_collection_info(self) -> dict:
        """Get vector collection info."""
        if not self._get_collection_info:
            return {}
        return self._get_collection_info()

    def get_batch_index_status(self) -> dict:
        """Get current batch indexing status."""
        with self._batch_index_lock:
            return self._batch_index_status.copy()

    def start_batch_indexing(self):
        """
        Start batch indexing of all documents in background.

        Returns:
            True if started, False if already running
        """
        with self._batch_index_lock:
            if self._batch_index_status["status"] == "running":
                return False

        def _batch_index_task():
            try:
                with self._batch_index_lock:
                    self._batch_index_status = {
                        "status": "running",
                        "total_documents": 0,
                        "indexed_documents": 0,
                        "current_index": 0,
                        "current_document": None,
                        "started_at": datetime.now().isoformat(),
                        "completed_at": None,
                        "errors": [],
                        "message": "Initializing batch indexing...",
                    }

                # Clear all existing embeddings
                logger.info("Clearing all vector embeddings for re-indexing...")
                with self._batch_index_lock:
                    self._batch_index_status["message"] = "Clearing existing embeddings..."

                if self._clear_collection:
                    try:
                        self._clear_collection()
                        logger.info("Cleared all vector embeddings")
                    except Exception as e:
                        logger.error(f"Error clearing collection: {e}")

                # Find all document directories
                output_path = Path(self.output_dir)
                doc_dirs = [d for d in output_path.iterdir() if d.is_dir()]

                with self._batch_index_lock:
                    self._batch_index_status["total_documents"] = len(doc_dirs)
                    self._batch_index_status["message"] = f"Found {len(doc_dirs)} documents to index"

                logger.info(f"Found {len(doc_dirs)} documents to index")

                total_chunks = 0
                for i, doc_dir in enumerate(doc_dirs):
                    source_name = doc_dir.name

                    with self._batch_index_lock:
                        self._batch_index_status["current_document"] = source_name
                        self._batch_index_status["current_index"] = i
                        self._batch_index_status["message"] = f"Indexing: {source_name} ({i+1}/{len(doc_dirs)})"

                    try:
                        if self._index_document_now:
                            chunks = self._index_document_now(source_name, self.output_dir)
                            total_chunks += chunks
                            logger.info(f"Indexed {source_name}: {chunks} chunks")

                        with self._batch_index_lock:
                            self._batch_index_status["indexed_documents"] = i + 1

                    except Exception as e:
                        error_msg = f"Error indexing {source_name}: {str(e)}"
                        logger.error(error_msg)
                        with self._batch_index_lock:
                            self._batch_index_status["errors"].append(error_msg)
                            self._batch_index_status["indexed_documents"] = i + 1

                with self._batch_index_lock:
                    self._batch_index_status["status"] = "completed"
                    self._batch_index_status["completed_at"] = datetime.now().isoformat()
                    self._batch_index_status["current_document"] = None
                    self._batch_index_status["message"] = f"Completed: indexed {total_chunks} chunks from {len(doc_dirs)} documents"

                logger.info(f"Batch indexing completed: {total_chunks} chunks from {len(doc_dirs)} documents")

            except Exception as e:
                logger.error(f"Batch indexing error: {str(e)}")
                with self._batch_index_lock:
                    self._batch_index_status["status"] = "error"
                    self._batch_index_status["completed_at"] = datetime.now().isoformat()
                    self._batch_index_status["message"] = f"Error: {str(e)}"

        thread = threading.Thread(target=_batch_index_task, daemon=True)
        thread.start()
        return True

    def reindex_failed_chunks(
        self,
        filename: str,
        phases: List[str],
        db: Session
    ) -> Dict:
        """
        Re-index only failed chunks/pages for specified phases.

        Args:
            filename: Document filename
            phases: List of phases to reindex (vector, graphrag, metadata)
            db: Database session

        Returns:
            Results dict for each phase
        """
        from rag_service.selective_reindexer import (
            reindex_failed_vector_pages,
            reindex_failed_graphrag_chunks,
            reextract_failed_metadata
        )

        file_name_without_ext = os.path.splitext(filename)[0]
        results = {}

        repo = DocumentRepository(db)
        doc = repo.get_by_filename(filename)

        if not doc:
            raise ValueError(f"Document not found: {filename}")

        if "vector" in phases:
            logger.info(f"[Selective Re-index] Starting vector re-indexing for {filename}")
            results["vector"] = reindex_failed_vector_pages(doc, file_name_without_ext, self.output_dir)

        if "graphrag" in phases:
            logger.info(f"[Selective Re-index] Starting GraphRAG re-indexing for {filename}")
            results["graphrag"] = reindex_failed_graphrag_chunks(doc, file_name_without_ext, self.output_dir)

        if "metadata" in phases:
            logger.info(f"[Selective Re-index] Starting metadata re-extraction for {filename}")
            results["metadata"] = reextract_failed_metadata(doc, file_name_without_ext, self.output_dir)

        return results
