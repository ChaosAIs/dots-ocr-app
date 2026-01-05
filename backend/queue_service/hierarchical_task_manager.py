"""
Hierarchical Task Queue Manager

Manages the 3-level task queue hierarchy:
- Document level: documents table with ocr_status, vector_status, graphrag_status
- Page level: task_queue_page table
- Chunk level: task_queue_chunk table

Processing flow:
1. OCR: Document → Pages (OCR each page)
2. Vector Index: Pages → Chunks (index each chunk to Qdrant)
3. GraphRAG Index: Chunks (extract entities to Neo4j)

Priority order for task pickup:
1. Failed tasks (retry first)
2. Pending tasks (new work)
3. Chunks before Pages before Documents (granular first)
"""

import os
import socket
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, List, NamedTuple, Literal, Tuple, Dict, Any, Union
from uuid import UUID
from sqlalchemy import and_, or_, func
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from db.database import get_db_session, create_db_session
from db.models import Document, DONE_STATUSES
from .models import TaskQueuePage, TaskQueueChunk, TaskQueueDocument, TaskStatus
from rag_service.graphrag_skip_config import (
    should_skip_graphrag_for_file,
    should_skip_graphrag_for_document_type,
)

logger = logging.getLogger(__name__)

# Load settings from environment
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "30"))
STALE_TASK_TIMEOUT = int(os.getenv("STALE_TASK_TIMEOUT", "120"))
MAX_TASK_RETRIES = int(os.getenv("MAX_TASK_RETRIES", "3"))

# Phase types
PhaseType = Literal["ocr", "vector", "graphrag"]


class PageTaskData(NamedTuple):
    """Data for a page-level task"""
    id: UUID
    document_id: UUID
    page_number: int
    page_file_path: Optional[str]
    phase: PhaseType
    retry_count: int


class ChunkTaskData(NamedTuple):
    """Data for a chunk-level task"""
    id: UUID
    page_id: UUID
    document_id: UUID
    chunk_id: str
    chunk_index: int
    phase: PhaseType
    retry_count: int


class ReindexTaskData(NamedTuple):
    """Data for a document-level reindex task"""
    id: UUID
    document_id: UUID
    filename: str
    output_path: Optional[str]
    retry_count: int


class HierarchicalTaskQueueManager:
    """
    Manages the hierarchical task queue for OCR, Vector, and GraphRAG processing.

    Key features:
    - 3-level hierarchy: Document → Page → Chunk
    - 3 processing phases: OCR → Vector → GraphRAG
    - Sequential dependencies between phases
    - Priority-based task pickup (failed first, then pending)
    - Heartbeat-based stale detection
    - Status bubble-up from children to parents
    """

    def __init__(
        self,
        stale_timeout_seconds: int = STALE_TASK_TIMEOUT,
        heartbeat_interval_seconds: int = HEARTBEAT_INTERVAL,
        max_retries: int = MAX_TASK_RETRIES
    ):
        """
        Initialize the hierarchical task queue manager.

        Args:
            stale_timeout_seconds: Timeout for stale task detection
            heartbeat_interval_seconds: Interval for worker heartbeat updates
            max_retries: Maximum retry attempts for failed tasks
        """
        self.stale_timeout_seconds = stale_timeout_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.max_retries = max_retries
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.new_task_event = threading.Event()

        logger.info(
            f"HierarchicalTaskQueueManager initialized "
            f"(stale_timeout={stale_timeout_seconds}s, heartbeat={heartbeat_interval_seconds}s, max_retries={max_retries})"
        )

    def _generate_worker_id(self, thread_id: int) -> str:
        """Generate unique worker ID: hostname:pid:thread_id"""
        return f"{self.hostname}:{self.pid}:{thread_id}"

    # =========================================================================
    # Document-level operations
    # =========================================================================

    def create_document_task(
        self,
        document_id: UUID,
        total_pages: int,
        db: Optional[Session] = None,
        force_reset: bool = False
    ) -> bool:
        """
        Create page tasks for a new document or reset existing tasks on re-upload.

        Called when a document is uploaded. Creates task_queue_page records
        for each page with ocr_status = 'pending'.

        Args:
            document_id: Document UUID
            total_pages: Number of pages in the document
            db: Database session
            force_reset: If True, delete existing tasks and re-create them

        Returns:
            True if tasks created successfully
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            # Check if pages already exist
            existing_pages = db.query(TaskQueuePage).filter(
                TaskQueuePage.document_id == document_id
            ).all()

            if existing_pages:
                if force_reset:
                    # Delete existing page tasks (chunks will cascade delete)
                    for page in existing_pages:
                        db.delete(page)
                    db.flush()
                    logger.info(f"Reset existing {len(existing_pages)} page tasks for document {document_id}")
                else:
                    # Check if any pages need processing (not all completed)
                    pending_or_failed = [
                        p for p in existing_pages
                        if p.ocr_status in (TaskStatus.PENDING, TaskStatus.FAILED)
                    ]
                    if pending_or_failed:
                        logger.info(f"Found {len(pending_or_failed)} pending/failed pages for document {document_id}")
                        # Notify workers about existing tasks
                        self.new_task_event.set()
                    else:
                        logger.info(f"All {len(existing_pages)} pages already processed for document {document_id}")
                    return True

            # Create page tasks
            for page_num in range(total_pages):
                page = TaskQueuePage(
                    document_id=document_id,
                    page_number=page_num,
                    ocr_status=TaskStatus.PENDING,
                    vector_status=TaskStatus.PENDING,
                    graphrag_status=TaskStatus.PENDING,
                    max_retries=self.max_retries
                )
                db.add(page)

            # Update document status
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.ocr_status = TaskStatus.PENDING
                doc.vector_status = TaskStatus.PENDING
                doc.graphrag_status = TaskStatus.PENDING
                doc.total_pages = total_pages

            db.commit()
            logger.info(f"Created {total_pages} page tasks for document {document_id}")

            # Notify workers
            self.new_task_event.set()
            return True

        except Exception as e:
            logger.error(f"Error creating document task: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()

    def update_document_status(
        self,
        document_id: UUID,
        phase: PhaseType,
        db: Optional[Session] = None
    ) -> None:
        """
        Update document status based on children (bubble-up logic).

        Rules (UPDATED - skipped counts as done):
        1. If ANY child is "processing" → parent = "processing"
        2. If ANY child is "pending"    → parent = "pending"
        3. If ALL children in DONE_STATUSES (completed/skipped) → parent = "completed"
        4. If ANY child "failed" (no pending/processing) → parent = "failed"

        Args:
            document_id: Document UUID
            phase: Which phase to update ("ocr", "vector", "graphrag")
            db: Database session
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            # Get status counts from pages/chunks
            if phase == "ocr":
                # OCR status comes from pages
                status_col = TaskQueuePage.ocr_status
                counts = db.query(
                    func.count().filter(status_col == TaskStatus.PROCESSING).label("processing"),
                    func.count().filter(status_col == TaskStatus.PENDING).label("pending"),
                    func.count().filter(status_col == TaskStatus.COMPLETED).label("completed"),
                    func.count().filter(status_col == TaskStatus.FAILED).label("failed"),
                    func.count().filter(status_col == TaskStatus.SKIPPED).label("skipped"),
                    func.count().label("total")
                ).filter(TaskQueuePage.document_id == document_id).first()
            else:
                # Vector and GraphRAG status come from chunks
                if phase == "vector":
                    status_col = TaskQueueChunk.vector_status
                else:
                    status_col = TaskQueueChunk.graphrag_status

                counts = db.query(
                    func.count().filter(status_col == TaskStatus.PROCESSING).label("processing"),
                    func.count().filter(status_col == TaskStatus.PENDING).label("pending"),
                    func.count().filter(status_col == TaskStatus.COMPLETED).label("completed"),
                    func.count().filter(status_col == TaskStatus.FAILED).label("failed"),
                    func.count().filter(status_col == TaskStatus.SKIPPED).label("skipped"),
                    func.count().label("total")
                ).filter(TaskQueueChunk.document_id == document_id).first()

            if not counts or counts.total == 0:
                return

            # Calculate done count (completed + skipped both count as done)
            done_count = (counts.completed or 0) + (counts.skipped or 0)

            # Determine new status using bubble-up rules
            if counts.processing > 0:
                new_status = TaskStatus.PROCESSING
            elif counts.pending > 0:
                new_status = TaskStatus.PENDING
            elif done_count == counts.total:
                # All tasks are done (either completed or skipped)
                new_status = TaskStatus.COMPLETED
            elif counts.failed > 0:
                new_status = TaskStatus.FAILED
            else:
                new_status = TaskStatus.PENDING

            # Update document
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                from db.models import ConvertStatus

                if phase == "ocr":
                    doc.ocr_status = new_status
                    if new_status == TaskStatus.COMPLETED:
                        doc.ocr_completed_at = datetime.now(timezone.utc)
                        # Also update convert_status for frontend compatibility
                        doc.convert_status = ConvertStatus.CONVERTED
                    elif new_status == TaskStatus.PROCESSING:
                        doc.convert_status = ConvertStatus.CONVERTING
                    elif new_status == TaskStatus.FAILED:
                        doc.convert_status = ConvertStatus.FAILED
                    elif new_status == TaskStatus.PENDING:
                        doc.convert_status = ConvertStatus.PENDING
                elif phase == "vector":
                    doc.vector_status = new_status
                    if new_status == TaskStatus.COMPLETED:
                        doc.vector_completed_at = datetime.now(timezone.utc)
                else:
                    doc.graphrag_status = new_status
                    if new_status == TaskStatus.COMPLETED:
                        doc.graphrag_completed_at = datetime.now(timezone.utc)

                db.commit()
                logger.debug(f"Updated document {document_id} {phase}_status = {new_status.value}")

        except Exception as e:
            logger.error(f"Error updating document status: {e}")
            db.rollback()
        finally:
            if should_close_db:
                db.close()

    def update_page_status(
        self,
        page_id: UUID,
        phase: PhaseType,
        db: Optional[Session] = None
    ) -> None:
        """
        Update page status based on children chunks (bubble-up logic).

        Rules (UPDATED - skipped counts as done):
        1. If ANY chunk is "processing" → page = "processing"
        2. If ANY chunk is "pending"    → page = "pending"
        3. If ALL chunks in DONE_STATUSES (completed/skipped) → page = "completed"
        4. If ANY chunk "failed" (no pending/processing) → page = "failed"

        Args:
            page_id: Page UUID
            phase: Which phase to update ("vector", "graphrag")
            db: Database session
        """
        if phase == "ocr":
            # OCR status is set directly on pages, not from chunks
            return

        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            # Get status counts from chunks for this page
            if phase == "vector":
                status_col = TaskQueueChunk.vector_status
            else:
                status_col = TaskQueueChunk.graphrag_status

            counts = db.query(
                func.count().filter(status_col == TaskStatus.PROCESSING).label("processing"),
                func.count().filter(status_col == TaskStatus.PENDING).label("pending"),
                func.count().filter(status_col == TaskStatus.COMPLETED).label("completed"),
                func.count().filter(status_col == TaskStatus.FAILED).label("failed"),
                func.count().filter(status_col == TaskStatus.SKIPPED).label("skipped"),
                func.count().label("total")
            ).filter(TaskQueueChunk.page_id == page_id).first()

            if not counts or counts.total == 0:
                return

            # Calculate done count (completed + skipped both count as done)
            done_count = (counts.completed or 0) + (counts.skipped or 0)

            # Determine new status using bubble-up rules
            if counts.processing > 0:
                new_status = TaskStatus.PROCESSING
            elif counts.pending > 0:
                new_status = TaskStatus.PENDING
            elif done_count == counts.total:
                # All tasks are done (either completed or skipped)
                new_status = TaskStatus.COMPLETED
            elif counts.failed > 0:
                new_status = TaskStatus.FAILED
            else:
                new_status = TaskStatus.PENDING

            # Update page
            page = db.query(TaskQueuePage).filter(TaskQueuePage.id == page_id).first()
            if page:
                if phase == "vector":
                    page.vector_status = new_status
                    if new_status == TaskStatus.COMPLETED:
                        page.vector_completed_at = datetime.now(timezone.utc)
                else:
                    page.graphrag_status = new_status
                    if new_status == TaskStatus.COMPLETED:
                        page.graphrag_completed_at = datetime.now(timezone.utc)

                db.commit()
                logger.debug(f"Updated page {page_id} {phase}_status = {new_status.value}")

        except Exception as e:
            logger.error(f"Error updating page status: {e}")
            db.rollback()
        finally:
            if should_close_db:
                db.close()

    def sync_document_indexing_details(
        self,
        document_id: UUID,
        db: Optional[Session] = None
    ) -> None:
        """
        Sync queue status to document's indexing_details JSONB for frontend compatibility.

        The frontend reads indexing_details to display status. This method aggregates
        status from task_queue_page and task_queue_chunk tables into the JSONB format
        expected by the frontend.

        Args:
            document_id: Document UUID
            db: Database session
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                return

            # Initialize indexing_details if needed
            if doc.indexing_details is None:
                doc.indexing_details = {}

            doc.indexing_details["version"] = "2.0"  # Mark as hierarchical queue version

            # Get aggregate stats from queue tables
            now = datetime.now(timezone.utc)

            # === Vector Indexing Stats ===
            vector_stats = db.query(
                func.count().label("total"),
                func.count().filter(TaskQueueChunk.vector_status == TaskStatus.COMPLETED).label("completed"),
                func.count().filter(TaskQueueChunk.vector_status == TaskStatus.PROCESSING).label("processing"),
                func.count().filter(TaskQueueChunk.vector_status == TaskStatus.FAILED).label("failed"),
                func.count().filter(TaskQueueChunk.vector_status == TaskStatus.PENDING).label("pending"),
            ).filter(TaskQueueChunk.document_id == document_id).first()

            if vector_stats and vector_stats.total > 0:
                # Determine overall vector status
                if vector_stats.processing > 0:
                    vector_status = "processing"
                elif vector_stats.completed == vector_stats.total:
                    vector_status = "completed"
                elif vector_stats.failed > 0 and vector_stats.pending == 0 and vector_stats.processing == 0:
                    vector_status = "failed"
                elif vector_stats.completed > 0:
                    vector_status = "partial"
                else:
                    vector_status = "pending"

                doc.indexing_details["vector_indexing"] = {
                    "status": vector_status,
                    "total_chunks": vector_stats.total,
                    "indexed_chunks": vector_stats.completed,
                    "failed_chunks": vector_stats.failed,
                    "pending_chunks": vector_stats.pending,
                    "processing_chunks": vector_stats.processing,
                    "updated_at": now.isoformat(),
                }

            # === GraphRAG Indexing Stats ===
            graphrag_stats = db.query(
                func.count().label("total"),
                func.count().filter(TaskQueueChunk.graphrag_status == TaskStatus.COMPLETED).label("completed"),
                func.count().filter(TaskQueueChunk.graphrag_status == TaskStatus.PROCESSING).label("processing"),
                func.count().filter(TaskQueueChunk.graphrag_status == TaskStatus.FAILED).label("failed"),
                func.count().filter(TaskQueueChunk.graphrag_status == TaskStatus.PENDING).label("pending"),
                func.count().filter(TaskQueueChunk.graphrag_status == TaskStatus.SKIPPED).label("skipped"),
                func.sum(TaskQueueChunk.entities_extracted).label("entities"),
                func.sum(TaskQueueChunk.relationships_extracted).label("relationships"),
            ).filter(TaskQueueChunk.document_id == document_id).first()

            if graphrag_stats and graphrag_stats.total > 0:
                # Calculate done count (completed + skipped)
                done_count = (graphrag_stats.completed or 0) + (graphrag_stats.skipped or 0)

                # Determine overall graphrag status (skipped documents show as "completed")
                if graphrag_stats.processing > 0:
                    graphrag_status = "processing"
                elif done_count == graphrag_stats.total:
                    # All chunks are done (either completed or skipped)
                    graphrag_status = "completed"
                elif graphrag_stats.failed > 0 and graphrag_stats.pending == 0 and graphrag_stats.processing == 0:
                    graphrag_status = "failed"
                elif done_count > 0:
                    graphrag_status = "partial"
                else:
                    graphrag_status = "pending"

                doc.indexing_details["graphrag_indexing"] = {
                    "status": graphrag_status,
                    "total_chunks": graphrag_stats.total,
                    "expected_total_chunks": graphrag_stats.total,
                    "processed_chunks": graphrag_stats.completed or 0,
                    "skipped_chunks": graphrag_stats.skipped or 0,
                    "failed_chunks": graphrag_stats.failed or 0,
                    "pending_chunks": graphrag_stats.pending or 0,
                    "processing_chunks": graphrag_stats.processing or 0,
                    "entities_extracted": graphrag_stats.entities or 0,
                    "relationships_extracted": graphrag_stats.relationships or 0,
                    "updated_at": now.isoformat(),
                }

                # Add skip reason if all chunks were skipped
                if graphrag_stats.skipped == graphrag_stats.total and doc.skip_graphrag_reason:
                    doc.indexing_details["graphrag_indexing"]["skip_reason"] = doc.skip_graphrag_reason

            # === OCR Stats (from pages) ===
            # Note: SKIPPED pages count as "done" (e.g., markdown files that bypass OCR)
            ocr_stats = db.query(
                func.count().label("total"),
                func.count().filter(TaskQueuePage.ocr_status == TaskStatus.COMPLETED).label("completed"),
                func.count().filter(TaskQueuePage.ocr_status == TaskStatus.SKIPPED).label("skipped"),
                func.count().filter(TaskQueuePage.ocr_status == TaskStatus.PROCESSING).label("processing"),
                func.count().filter(TaskQueuePage.ocr_status == TaskStatus.FAILED).label("failed"),
                func.count().filter(TaskQueuePage.ocr_status == TaskStatus.PENDING).label("pending"),
            ).filter(TaskQueuePage.document_id == document_id).first()

            if ocr_stats and ocr_stats.total > 0:
                # Calculate done pages (completed + skipped)
                done_pages = ocr_stats.completed + ocr_stats.skipped

                # Determine overall OCR status
                if ocr_stats.processing > 0:
                    ocr_status = "processing"
                elif done_pages == ocr_stats.total:
                    # All pages are done (completed or skipped)
                    ocr_status = "completed" if ocr_stats.skipped == 0 else "skipped"
                elif ocr_stats.failed > 0 and ocr_stats.pending == 0 and ocr_stats.processing == 0:
                    ocr_status = "failed"
                elif done_pages > 0:
                    ocr_status = "partial"
                else:
                    ocr_status = "pending"

                doc.indexing_details["ocr_processing"] = {
                    "status": ocr_status,
                    "total_pages": ocr_stats.total,
                    "completed_pages": ocr_stats.completed,
                    "skipped_pages": ocr_stats.skipped,
                    "failed_pages": ocr_stats.failed,
                    "pending_pages": ocr_stats.pending,
                    "processing_pages": ocr_stats.processing,
                    "updated_at": now.isoformat(),
                }

                # === Metadata Extraction Stats ===
                # In the hierarchical queue, metadata extraction happens AFTER vector indexing completes,
                # NOT during OCR. Don't overwrite existing metadata_extraction status - preserve it.
                if "metadata_extraction" not in doc.indexing_details:
                    doc.indexing_details["metadata_extraction"] = {
                        "status": "pending",
                        "updated_at": now.isoformat(),
                    }
                else:
                    # Only update timestamp, preserve the actual status set by extraction logic
                    doc.indexing_details["metadata_extraction"]["updated_at"] = now.isoformat()

            # Update overall index_status based on all phases
            # The document is fully indexed only when all phases are complete
            from db.models import IndexStatus

            vector_complete = (
                doc.indexing_details.get("vector_indexing", {}).get("status") == "completed"
            )
            graphrag_complete = (
                doc.indexing_details.get("graphrag_indexing", {}).get("status") == "completed"
            )
            vector_processing = (
                doc.indexing_details.get("vector_indexing", {}).get("status") == "processing"
            )
            graphrag_processing = (
                doc.indexing_details.get("graphrag_indexing", {}).get("status") == "processing"
            )
            vector_failed = (
                doc.indexing_details.get("vector_indexing", {}).get("status") == "failed"
            )
            graphrag_failed = (
                doc.indexing_details.get("graphrag_indexing", {}).get("status") == "failed"
            )

            if vector_complete and graphrag_complete:
                doc.index_status = IndexStatus.INDEXED
            elif vector_processing or graphrag_processing:
                doc.index_status = IndexStatus.INDEXING
            elif vector_failed or graphrag_failed:
                doc.index_status = IndexStatus.FAILED
            elif vector_complete or graphrag_complete:
                doc.index_status = IndexStatus.PARTIAL
            else:
                doc.index_status = IndexStatus.PENDING

            # Mark JSONB field as modified so SQLAlchemy knows to update it
            flag_modified(doc, "indexing_details")

            db.commit()
            logger.debug(f"Synced indexing_details for document {document_id}")

        except Exception as e:
            logger.error(f"Error syncing indexing_details: {e}", exc_info=True)
            db.rollback()
        finally:
            if should_close_db:
                db.close()

    # =========================================================================
    # Page-level operations
    # =========================================================================

    def claim_ocr_page_task(
        self,
        worker_id: str,
        db: Optional[Session] = None
    ) -> Optional[PageTaskData]:
        """
        Claim the next available OCR page task.

        Priority: Failed first (retry), then Pending.

        Args:
            worker_id: Worker identifier
            db: Database session

        Returns:
            PageTaskData if task claimed, None otherwise
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            # Find next OCR task (failed first, then pending)
            page = db.query(TaskQueuePage).filter(
                TaskQueuePage.ocr_status.in_([TaskStatus.PENDING, TaskStatus.FAILED]),
                TaskQueuePage.ocr_retry_count < TaskQueuePage.max_retries
            ).order_by(
                # Failed first (to retry), then pending
                TaskQueuePage.ocr_status.desc(),  # PENDING > FAILED alphabetically, so desc
                TaskQueuePage.ocr_retry_count.asc(),
                TaskQueuePage.created_at.asc()
            ).with_for_update(skip_locked=True).first()

            if not page:
                return None

            # Claim the task
            now = datetime.now(timezone.utc)
            page.ocr_status = TaskStatus.PROCESSING
            page.ocr_worker_id = worker_id
            page.ocr_started_at = now
            page.ocr_last_heartbeat = now

            db.commit()

            logger.info(f"Worker {worker_id} claimed OCR page task: doc={page.document_id}, page={page.page_number}")

            return PageTaskData(
                id=page.id,
                document_id=page.document_id,
                page_number=page.page_number,
                page_file_path=page.page_file_path,
                phase="ocr",
                retry_count=page.ocr_retry_count
            )

        except Exception as e:
            logger.error(f"Error claiming OCR page task: {e}")
            db.rollback()
            return None
        finally:
            if should_close_db:
                db.close()

    def complete_ocr_page_task(
        self,
        page_id: UUID,
        page_file_path: str,
        db: Optional[Session] = None
    ) -> bool:
        """
        Mark OCR page task as completed.

        Args:
            page_id: Page task UUID
            page_file_path: Path to the generated markdown file
            db: Database session

        Returns:
            True if successful
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            page = db.query(TaskQueuePage).filter(TaskQueuePage.id == page_id).first()
            if not page:
                return False

            page.ocr_status = TaskStatus.COMPLETED
            page.ocr_completed_at = datetime.now(timezone.utc)
            page.page_file_path = page_file_path
            page.ocr_error = None

            db.commit()

            # Update parent document status and sync to indexing_details
            self.update_document_status(page.document_id, "ocr", db)
            self.sync_document_indexing_details(page.document_id, db)

            logger.info(f"Completed OCR page task: page={page_id}")
            return True

        except Exception as e:
            logger.error(f"Error completing OCR page task: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()

    def fail_ocr_page_task(
        self,
        page_id: UUID,
        error_message: str,
        db: Optional[Session] = None
    ) -> bool:
        """
        Mark OCR page task as failed.

        Args:
            page_id: Page task UUID
            error_message: Error description
            db: Database session

        Returns:
            True if successful
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            page = db.query(TaskQueuePage).filter(TaskQueuePage.id == page_id).first()
            if not page:
                return False

            page.ocr_retry_count += 1
            page.ocr_error = error_message

            if page.ocr_retry_count >= page.max_retries:
                page.ocr_status = TaskStatus.FAILED
                logger.error(f"OCR page task permanently failed: page={page_id}, error={error_message}")
            else:
                page.ocr_status = TaskStatus.PENDING  # Will be retried
                page.ocr_worker_id = None
                page.ocr_last_heartbeat = None
                logger.warning(f"OCR page task failed, will retry: page={page_id}, retry={page.ocr_retry_count}")

            db.commit()

            # Update parent document status and sync to indexing_details
            self.update_document_status(page.document_id, "ocr", db)
            self.sync_document_indexing_details(page.document_id, db)

            return True

        except Exception as e:
            logger.error(f"Error failing OCR page task: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()

    def update_ocr_heartbeat(
        self,
        page_id: UUID,
        db: Optional[Session] = None
    ) -> bool:
        """Update heartbeat for OCR page task."""
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            page = db.query(TaskQueuePage).filter(TaskQueuePage.id == page_id).first()
            if page:
                page.ocr_last_heartbeat = datetime.now(timezone.utc)
                db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating OCR heartbeat: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()

    # =========================================================================
    # Chunk-level operations
    # =========================================================================

    def create_chunk_tasks(
        self,
        page_id: UUID,
        document_id: UUID,
        chunks: List[Tuple[str, int, Optional[str], Optional[Dict[str, Any]]]],  # List of (chunk_id, chunk_index, content, metadata)
        db: Optional[Session] = None
    ) -> bool:
        """
        Create chunk tasks for a page.

        Called after OCR is complete and content is chunked.
        V3.0: Now stores chunk content and metadata to avoid redundant LLM calls.

        Args:
            page_id: Page task UUID
            document_id: Document UUID
            chunks: List of (chunk_id, chunk_index, content, metadata) tuples
                   - content: The actual chunk text (stored for Stages 2 & 3)
                   - metadata: Chunk metadata as dict (strategy, positions, etc.)
            db: Database session

        Returns:
            True if tasks created successfully
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            for chunk_data in chunks:
                # Handle both old format (id, index) and new format (id, index, content, metadata)
                if len(chunk_data) == 2:
                    chunk_id, chunk_index = chunk_data
                    chunk_content, chunk_metadata = None, None
                else:
                    chunk_id, chunk_index, chunk_content, chunk_metadata = chunk_data

                chunk = TaskQueueChunk(
                    page_id=page_id,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    chunk_content=chunk_content,
                    chunk_metadata=chunk_metadata,
                    vector_status=TaskStatus.PENDING,
                    graphrag_status=TaskStatus.PENDING,
                    max_retries=self.max_retries
                )
                db.add(chunk)

            # Update page chunk count
            page = db.query(TaskQueuePage).filter(TaskQueuePage.id == page_id).first()
            if page:
                page.chunk_count = len(chunks)

            db.commit()
            logger.info(f"Created {len(chunks)} chunk tasks for page {page_id}")

            # Notify workers
            self.new_task_event.set()
            return True

        except Exception as e:
            logger.error(f"Error creating chunk tasks: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()

    def claim_vector_chunk_task(
        self,
        worker_id: str,
        db: Optional[Session] = None
    ) -> Optional[ChunkTaskData]:
        """
        Claim the next available Vector indexing chunk task.

        Only picks up chunks where the parent page's OCR is completed or skipped.
        OCR can be SKIPPED for markdown files that bypass the OCR process.

        Args:
            worker_id: Worker identifier
            db: Database session

        Returns:
            ChunkTaskData if task claimed, None otherwise
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            # Find next Vector chunk task where OCR is completed or skipped
            # OCR can be COMPLETED (normal conversion) or SKIPPED (markdown files that bypass OCR)
            chunk = db.query(TaskQueueChunk).join(
                TaskQueuePage, TaskQueueChunk.page_id == TaskQueuePage.id
            ).filter(
                TaskQueueChunk.vector_status.in_([TaskStatus.PENDING, TaskStatus.FAILED]),
                TaskQueueChunk.vector_retry_count < TaskQueueChunk.max_retries,
                TaskQueuePage.ocr_status.in_(DONE_STATUSES)  # COMPLETED or SKIPPED
            ).order_by(
                TaskQueueChunk.vector_status.desc(),  # Failed first
                TaskQueueChunk.vector_retry_count.asc(),
                TaskQueueChunk.created_at.asc()
            ).with_for_update(skip_locked=True).first()

            if not chunk:
                return None

            # Claim the task
            now = datetime.now(timezone.utc)
            chunk.vector_status = TaskStatus.PROCESSING
            chunk.vector_worker_id = worker_id
            chunk.vector_started_at = now
            chunk.vector_last_heartbeat = now

            db.commit()

            logger.info(f"Worker {worker_id} claimed Vector chunk task: chunk={chunk.chunk_id}")

            return ChunkTaskData(
                id=chunk.id,
                page_id=chunk.page_id,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                phase="vector",
                retry_count=chunk.vector_retry_count
            )

        except Exception as e:
            logger.error(f"Error claiming Vector chunk task: {e}")
            db.rollback()
            return None
        finally:
            if should_close_db:
                db.close()

    def complete_vector_chunk_task(
        self,
        chunk_task_id: UUID,
        db: Optional[Session] = None
    ) -> bool:
        """Mark Vector chunk task as completed."""
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            chunk = db.query(TaskQueueChunk).filter(TaskQueueChunk.id == chunk_task_id).first()
            if not chunk:
                return False

            chunk.vector_status = TaskStatus.COMPLETED
            chunk.vector_completed_at = datetime.now(timezone.utc)
            chunk.vector_error = None

            # Store page_id and document_id before commit
            page_id = chunk.page_id
            document_id = chunk.document_id

            db.commit()

            # Update parent page status (skip for legacy chunks without page_id)
            if page_id is not None:
                self.update_page_status(page_id, "vector", db)

            # Update document status and sync to indexing_details
            self.update_document_status(document_id, "vector", db)
            self.sync_document_indexing_details(document_id, db)

            # Check if all vector chunks for this document are complete
            # If so, trigger metadata extraction in background
            # NOTE: Data extraction is now triggered INSIDE _run_metadata_extraction
            # after document_metadata is populated (ensures eligibility check has the data it needs)
            self._check_and_trigger_metadata_extraction(document_id, db)

            logger.debug(f"Completed Vector chunk task: chunk={chunk_task_id}")
            return True

        except Exception as e:
            logger.error(f"Error completing Vector chunk task: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()

    def fail_vector_chunk_task(
        self,
        chunk_task_id: UUID,
        error_message: str,
        db: Optional[Session] = None
    ) -> bool:
        """Mark Vector chunk task as failed."""
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            chunk = db.query(TaskQueueChunk).filter(TaskQueueChunk.id == chunk_task_id).first()
            if not chunk:
                return False

            chunk.vector_retry_count += 1
            chunk.vector_error = error_message

            if chunk.vector_retry_count >= chunk.max_retries:
                chunk.vector_status = TaskStatus.FAILED
            else:
                chunk.vector_status = TaskStatus.PENDING
                chunk.vector_worker_id = None
                chunk.vector_last_heartbeat = None

            # Store page_id and document_id before commit
            page_id = chunk.page_id
            document_id = chunk.document_id

            db.commit()

            # Update parent page status (skip for legacy chunks without page_id)
            if page_id is not None:
                self.update_page_status(page_id, "vector", db)

            # Update document status and sync to indexing_details
            self.update_document_status(document_id, "vector", db)
            self.sync_document_indexing_details(document_id, db)

            return True

        except Exception as e:
            logger.error(f"Error failing Vector chunk task: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()

    def update_vector_heartbeat(
        self,
        chunk_task_id: UUID,
        db: Optional[Session] = None
    ) -> bool:
        """Update heartbeat for Vector chunk task."""
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            chunk = db.query(TaskQueueChunk).filter(TaskQueueChunk.id == chunk_task_id).first()
            if chunk:
                chunk.vector_last_heartbeat = datetime.now(timezone.utc)
                db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating Vector heartbeat: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()

    def claim_graphrag_chunk_task(
        self,
        worker_id: str,
        db: Optional[Session] = None
    ) -> Optional[ChunkTaskData]:
        """
        Claim the next available GraphRAG indexing chunk task.

        Only picks up chunks where Vector indexing is completed.
        Automatically skips chunks from documents that shouldn't be GraphRAG indexed
        (e.g., spreadsheets, invoices, receipts with tabular data).

        Args:
            worker_id: Worker identifier
            db: Database session

        Returns:
            ChunkTaskData if task claimed, None otherwise
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            while True:  # Loop to handle skipped chunks
                # Find next GraphRAG chunk task where Vector is completed
                chunk = db.query(TaskQueueChunk).filter(
                    TaskQueueChunk.graphrag_status.in_([TaskStatus.PENDING, TaskStatus.FAILED]),
                    TaskQueueChunk.graphrag_retry_count < TaskQueueChunk.max_retries,
                    TaskQueueChunk.vector_status == TaskStatus.COMPLETED  # Dependency check
                ).order_by(
                    TaskQueueChunk.graphrag_status.desc(),  # Failed first
                    TaskQueueChunk.graphrag_retry_count.asc(),
                    TaskQueueChunk.created_at.asc()
                ).with_for_update(skip_locked=True).first()

                if not chunk:
                    return None

                # Check if this chunk should skip GraphRAG
                should_skip, skip_reason = self._should_skip_graphrag(chunk, db)

                if should_skip:
                    # Mark as skipped (not failed) - this counts as "done"
                    chunk.graphrag_status = TaskStatus.SKIPPED
                    chunk.graphrag_skip_reason = skip_reason
                    chunk.graphrag_completed_at = datetime.now(timezone.utc)

                    # Store page_id and document_id before commit
                    page_id = chunk.page_id
                    document_id = chunk.document_id

                    db.commit()

                    logger.info(f"⏭️ Skipped GraphRAG for chunk {chunk.chunk_id}: {skip_reason}")

                    # Update parent page status (skip for legacy chunks without page_id)
                    if page_id is not None:
                        self.update_page_status(page_id, "graphrag", db)

                    # Update document status and sync to indexing_details
                    self.update_document_status(document_id, "graphrag", db)
                    self.sync_document_indexing_details(document_id, db)

                    # V3.0: Check if document is fully indexed and cleanup queue records
                    self._check_and_cleanup_completed_document(document_id, db)

                    # Continue to find next non-skipped chunk
                    continue

                # Claim the task (normal processing)
                now = datetime.now(timezone.utc)
                chunk.graphrag_status = TaskStatus.PROCESSING
                chunk.graphrag_worker_id = worker_id
                chunk.graphrag_started_at = now
                chunk.graphrag_last_heartbeat = now

                db.commit()

                logger.info(f"Worker {worker_id} claimed GraphRAG chunk task: chunk={chunk.chunk_id}")

                return ChunkTaskData(
                    id=chunk.id,
                    page_id=chunk.page_id,
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    phase="graphrag",
                    retry_count=chunk.graphrag_retry_count
                )

        except Exception as e:
            logger.error(f"Error claiming GraphRAG chunk task: {e}")
            db.rollback()
            return None
        finally:
            if should_close_db:
                db.close()

    def _should_skip_graphrag(self, chunk: TaskQueueChunk, db: Session) -> Tuple[bool, Optional[str]]:
        """
        Determine if a chunk should skip GraphRAG indexing.

        Checks in priority order:
        1. Document-level skip flag (set at upload or after metadata extraction)
        2. File extension (spreadsheets, CSVs, etc.)
        3. Document type from metadata (invoices, receipts, etc.)

        Args:
            chunk: The TaskQueueChunk to check
            db: Database session

        Returns:
            Tuple of (should_skip: bool, reason: str or None)
        """
        doc = db.query(Document).filter(Document.id == chunk.document_id).first()
        if not doc:
            return False, None

        # 1. Check document-level skip flag (highest priority)
        if doc.skip_graphrag:
            return True, doc.skip_graphrag_reason or "document_disabled"

        # 2. Check file extension
        should_skip, reason = should_skip_graphrag_for_file(doc.filename)
        if should_skip:
            return True, reason

        # 3. Check document types from metadata
        if doc.document_metadata:
            # Use document_types list (the standard format)
            doc_types = doc.document_metadata.get('document_types', [])

            # Check ALL document types for skip
            # A document should skip GraphRAG if ANY of its types is in the skip list
            for doc_type in doc_types:
                if doc_type:
                    should_skip, reason = should_skip_graphrag_for_document_type(doc_type)
                    if should_skip:
                        logger.info(f"[GraphRAG Skip] Document {doc.filename} skipping due to document_type: {doc_type}")
                        return True, reason

        return False, None

    def complete_graphrag_chunk_task(
        self,
        chunk_task_id: UUID,
        entities_extracted: int = 0,
        relationships_extracted: int = 0,
        db: Optional[Session] = None
    ) -> bool:
        """Mark GraphRAG chunk task as completed."""
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            chunk = db.query(TaskQueueChunk).filter(TaskQueueChunk.id == chunk_task_id).first()
            if not chunk:
                return False

            chunk.graphrag_status = TaskStatus.COMPLETED
            chunk.graphrag_completed_at = datetime.now(timezone.utc)
            chunk.graphrag_error = None
            chunk.entities_extracted = entities_extracted
            chunk.relationships_extracted = relationships_extracted

            # Store page_id and document_id before commit
            page_id = chunk.page_id
            document_id = chunk.document_id

            db.commit()

            # Update parent page status (skip for legacy chunks without page_id)
            if page_id is not None:
                self.update_page_status(page_id, "graphrag", db)

            # Update document status and sync to indexing_details
            self.update_document_status(document_id, "graphrag", db)
            self.sync_document_indexing_details(document_id, db)

            # Note: Data extraction is now triggered after Vector indexing completes
            # (before GraphRAG), see complete_vector_chunk_task

            # V3.0: Check if document is fully indexed and cleanup queue records
            self._check_and_cleanup_completed_document(document_id, db)

            logger.debug(f"Completed GraphRAG chunk task: chunk={chunk_task_id}, entities={entities_extracted}")
            return True

        except Exception as e:
            logger.error(f"Error completing GraphRAG chunk task: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()

    def fail_graphrag_chunk_task(
        self,
        chunk_task_id: UUID,
        error_message: str,
        db: Optional[Session] = None
    ) -> bool:
        """Mark GraphRAG chunk task as failed."""
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            chunk = db.query(TaskQueueChunk).filter(TaskQueueChunk.id == chunk_task_id).first()
            if not chunk:
                return False

            chunk.graphrag_retry_count += 1
            chunk.graphrag_error = error_message

            if chunk.graphrag_retry_count >= chunk.max_retries:
                chunk.graphrag_status = TaskStatus.FAILED
            else:
                chunk.graphrag_status = TaskStatus.PENDING
                chunk.graphrag_worker_id = None
                chunk.graphrag_last_heartbeat = None

            # Store page_id and document_id before commit
            page_id = chunk.page_id
            document_id = chunk.document_id

            db.commit()

            # Update parent page status (skip for legacy chunks without page_id)
            if page_id is not None:
                self.update_page_status(page_id, "graphrag", db)

            # Update document status and sync to indexing_details
            self.update_document_status(document_id, "graphrag", db)
            self.sync_document_indexing_details(document_id, db)

            return True

        except Exception as e:
            logger.error(f"Error failing GraphRAG chunk task: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()

    def update_graphrag_heartbeat(
        self,
        chunk_task_id: UUID,
        db: Optional[Session] = None
    ) -> bool:
        """Update heartbeat for GraphRAG chunk task."""
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            chunk = db.query(TaskQueueChunk).filter(TaskQueueChunk.id == chunk_task_id).first()
            if chunk:
                chunk.graphrag_last_heartbeat = datetime.now(timezone.utc)
                db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating GraphRAG heartbeat: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()

    # =========================================================================
    # Stale task detection
    # =========================================================================

    def release_stale_tasks(self, db: Optional[Session] = None) -> dict:
        """
        Release stale tasks (no heartbeat for > stale_timeout_seconds).

        Returns:
            Dict with counts of released tasks per phase
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        stale_threshold = datetime.now(timezone.utc) - timedelta(seconds=self.stale_timeout_seconds)
        released = {"ocr_pages": 0, "vector_chunks": 0, "graphrag_chunks": 0}

        try:
            # Release stale OCR page tasks
            stale_ocr = db.query(TaskQueuePage).filter(
                TaskQueuePage.ocr_status == TaskStatus.PROCESSING,
                TaskQueuePage.ocr_last_heartbeat < stale_threshold
            ).all()

            for page in stale_ocr:
                logger.warning(f"Releasing stale OCR task: page={page.id}, last_heartbeat={page.ocr_last_heartbeat}")
                page.ocr_status = TaskStatus.PENDING
                page.ocr_worker_id = None
                page.ocr_last_heartbeat = None
                released["ocr_pages"] += 1

            # Release stale Vector chunk tasks
            stale_vector = db.query(TaskQueueChunk).filter(
                TaskQueueChunk.vector_status == TaskStatus.PROCESSING,
                TaskQueueChunk.vector_last_heartbeat < stale_threshold
            ).all()

            for chunk in stale_vector:
                logger.warning(f"Releasing stale Vector task: chunk={chunk.id}, last_heartbeat={chunk.vector_last_heartbeat}")
                chunk.vector_status = TaskStatus.PENDING
                chunk.vector_worker_id = None
                chunk.vector_last_heartbeat = None
                released["vector_chunks"] += 1

            # Release stale GraphRAG chunk tasks
            stale_graphrag = db.query(TaskQueueChunk).filter(
                TaskQueueChunk.graphrag_status == TaskStatus.PROCESSING,
                TaskQueueChunk.graphrag_last_heartbeat < stale_threshold
            ).all()

            for chunk in stale_graphrag:
                logger.warning(f"Releasing stale GraphRAG task: chunk={chunk.id}, last_heartbeat={chunk.graphrag_last_heartbeat}")
                chunk.graphrag_status = TaskStatus.PENDING
                chunk.graphrag_worker_id = None
                chunk.graphrag_last_heartbeat = None
                released["graphrag_chunks"] += 1

            if any(v > 0 for v in released.values()):
                db.commit()
                logger.info(f"Released stale tasks: {released}")

            return released

        except Exception as e:
            logger.error(f"Error releasing stale tasks: {e}")
            db.rollback()
            return released
        finally:
            if should_close_db:
                db.close()

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_queue_stats(self, db: Optional[Session] = None) -> dict:
        """Get queue statistics for all phases."""
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            stats = {}

            # OCR page stats
            for status in TaskStatus:
                count = db.query(TaskQueuePage).filter(
                    TaskQueuePage.ocr_status == status
                ).count()
                stats[f"ocr_{status.value}"] = count

            # Vector chunk stats
            for status in TaskStatus:
                count = db.query(TaskQueueChunk).filter(
                    TaskQueueChunk.vector_status == status
                ).count()
                stats[f"vector_{status.value}"] = count

            # GraphRAG chunk stats
            for status in TaskStatus:
                count = db.query(TaskQueueChunk).filter(
                    TaskQueueChunk.graphrag_status == status
                ).count()
                stats[f"graphrag_{status.value}"] = count

            return stats

        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {}
        finally:
            if should_close_db:
                db.close()

    def _check_and_trigger_metadata_extraction(
        self,
        document_id: UUID,
        db: Optional[Session] = None
    ) -> None:
        """
        Check if all vector chunks for a document are complete.
        If so, trigger metadata extraction in a background thread.

        Metadata is stored in PostgreSQL's document_metadata column (NOT in vector DB).
        It is used by the document router to intelligently filter which documents
        to search based on query analysis (subject matching, document type, etc.).

        Args:
            document_id: Document UUID
            db: Database session
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            # Check if all vector chunks are complete
            vector_stats = db.query(
                func.count().label("total"),
                func.count().filter(TaskQueueChunk.vector_status == TaskStatus.COMPLETED).label("completed"),
            ).filter(TaskQueueChunk.document_id == document_id).first()

            if not vector_stats or vector_stats.total == 0:
                return

            # Not all chunks complete yet
            if vector_stats.completed < vector_stats.total:
                return

            # Check if metadata already extracted
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                return

            # Skip if FULL metadata extraction already done
            # Early routing metadata (classification_source="early_routing") is incomplete:
            # it only has document_types but NOT the full metadata (entities, topics, etc.)
            # and most importantly, it does NOT upsert to the metadata vector collection
            if doc.document_metadata:
                classification_source = doc.document_metadata.get("classification_source", "")
                if classification_source != "early_routing":
                    # Full extraction was done, skip
                    logger.debug(f"Metadata already exists for document {document_id}, skipping extraction")
                    return
                else:
                    # Early routing only - need full extraction for metadata vector embedding
                    logger.info(f"[Metadata] Early routing metadata exists, proceeding with full extraction for doc={document_id}")

            # All vector chunks complete and no metadata yet - trigger extraction
            logger.info(f"[Metadata] All vector chunks complete for doc={document_id}, triggering metadata extraction")

            # Get document info for metadata extraction
            filename = doc.filename
            source_name = filename.rsplit('.', 1)[0] if '.' in filename else filename

            # Run metadata extraction in background thread
            def _extract_metadata_task():
                try:
                    self._run_metadata_extraction(document_id, source_name, filename)
                except Exception as e:
                    logger.error(f"[Metadata] Extraction failed for doc={document_id}: {e}", exc_info=True)

            extraction_thread = threading.Thread(
                target=_extract_metadata_task,
                daemon=True,
                name=f"metadata-{source_name}"
            )
            extraction_thread.start()

        except Exception as e:
            logger.error(f"Error checking metadata extraction: {e}")
        finally:
            if should_close_db:
                db.close()

    def _run_metadata_extraction(
        self,
        document_id: UUID,
        source_name: str,
        filename: str
    ) -> None:
        """
        Run first-time metadata extraction for a document.

        Extracts document metadata (type, subject, topics, entities, summary)
        using HierarchicalMetadataExtractor and stores it in PostgreSQL.
        This is called automatically when all vector chunks complete.

        For re-extraction of failed metadata, use reextract_failed_metadata()
        from selective_reindexer.py instead.

        Args:
            document_id: Document UUID
            source_name: Document source name (filename without extension)
            filename: Original filename with extension
        """
        import asyncio
        from db.document_repository import DocumentRepository

        logger.info(f"[Metadata] Starting extraction for {source_name}")

        db = create_db_session()
        try:
            # Get document
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                logger.warning(f"[Metadata] Document not found: {document_id}")
                return

            # Get chunks from Qdrant
            from rag_service.vectorstore import get_all_chunks_for_source
            chunks = get_all_chunks_for_source(source_name)

            if not chunks:
                logger.warning(f"[Metadata] No chunks found in Qdrant for {source_name}")
                return

            # Convert to extractor format
            chunks_data = [
                {
                    "id": chunk.metadata.get("chunk_id"),
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                }
                for chunk in chunks
            ]

            logger.info(f"[Metadata] Extracting from {len(chunks_data)} chunks for {source_name}")

            # Run extraction using HierarchicalMetadataExtractor directly
            from rag_service.graph_rag.metadata_extractor import HierarchicalMetadataExtractor
            extractor = HierarchicalMetadataExtractor()

            # Run async extraction in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                metadata = loop.run_until_complete(
                    extractor.extract_metadata(
                        chunks=chunks_data,
                        source_name=source_name,
                        batch_size=10,
                        progress_callback=None
                    )
                )
            finally:
                loop.close()

            # Save metadata to database
            repo = DocumentRepository(db)
            document_types = metadata.get('document_types', ['unknown'])
            repo.update_document_metadata(
                doc,
                metadata,
                message=f"Extracted: {document_types} - {metadata.get('subject_name', 'N/A')}"
            )
            repo.update_metadata_extraction_status(doc, "completed")

            # Check if document type should skip GraphRAG and set the flag
            # Use primary (first) document type for skip check
            primary_doc_type = document_types[0] if document_types else ''
            should_skip, skip_reason = should_skip_graphrag_for_document_type(primary_doc_type)
            if should_skip and not doc.skip_graphrag:
                doc.skip_graphrag = True
                doc.skip_graphrag_reason = skip_reason
                db.commit()
                logger.info(f"[Metadata] Set skip_graphrag=True for {source_name}: {skip_reason}")

            # Embed metadata to vector collection for fast document routing
            from rag_service.vectorstore import upsert_document_metadata_embedding
            upsert_document_metadata_embedding(
                document_id=str(document_id),
                source_name=source_name,
                filename=filename,
                metadata=metadata
            )

            logger.info(
                f"[Metadata] ✅ Saved for {source_name}: "
                f"document_types={document_types} | "
                f"subject={metadata.get('subject_name')} | "
                f"confidence={metadata.get('confidence', 0):.2f}"
            )
            logger.debug(f"[Metadata] Extracted document_types for {source_name}: {document_types}")

            # Trigger data extraction AFTER metadata is populated
            # This ensures document_type is available for eligibility checking
            try:
                from extraction_service.task_queue_integration import run_document_extraction
                logger.info(f"[Extraction] Triggering data extraction after metadata for {source_name}")
                run_document_extraction(document_id)
            except ImportError as e:
                logger.debug(f"[Extraction] Extraction service not available: {e}")
            except Exception as e:
                logger.error(f"[Extraction] Failed to trigger extraction for {source_name}: {e}")

        except ImportError as e:
            logger.warning(f"[Metadata] Extractor not available: {e}")
        except Exception as e:
            logger.error(f"[Metadata] Extraction failed for {source_name}: {e}", exc_info=True)

            # Update status to failed
            try:
                repo = DocumentRepository(db)
                doc = db.query(Document).filter(Document.id == document_id).first()
                if doc:
                    repo.update_metadata_extraction_status(doc, "failed", error=str(e))
            except Exception as db_error:
                logger.warning(f"[Metadata] Could not update failed status: {db_error}")
        finally:
            db.close()

    def _check_and_trigger_data_extraction(
        self,
        document_id: UUID,
        db: Optional[Session] = None
    ) -> None:
        """
        Check if all Vector chunks for a document are complete.
        If so, trigger structured data extraction in a background thread.

        Data extraction converts documents (invoices, receipts, statements, etc.)
        into structured JSON format stored in the documents_data table.
        This enables analytics queries on the extracted data.

        Note: Data extraction now runs BEFORE GraphRAG indexing to ensure
        structured data is available early in the pipeline.

        Args:
            document_id: Document UUID
            db: Database session
        """
        try:
            from extraction_service.task_queue_integration import check_and_trigger_data_extraction
            check_and_trigger_data_extraction(document_id, db)
        except ImportError as e:
            logger.debug(f"[Extraction] Extraction service not available: {e}")
        except Exception as e:
            logger.error(f"[Extraction] Error triggering extraction for {document_id}: {e}")

    # =========================================================================
    # V3.0: Queue Cleanup (Hard Delete after completion)
    # =========================================================================

    def cleanup_completed_document_queue(
        self,
        document_id: UUID,
        db: Optional[Session] = None
    ) -> Dict[str, int]:
        """
        Hard delete queue records (pages and chunks) for a fully indexed document.

        V3.0: Called when all processing stages are complete for a document.
        The document table retains status info, so queue records are no longer needed.

        This cleanup:
        - Frees up database storage (especially chunk_content TEXT columns)
        - Keeps queue tables lean for better query performance
        - Relies on CASCADE delete from task_queue_page to task_queue_chunk

        Args:
            document_id: Document UUID
            db: Database session

        Returns:
            Dict with counts of deleted records
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        deleted = {"pages": 0, "chunks": 0}

        try:
            # Count records before deletion for logging
            chunk_count = db.query(TaskQueueChunk).filter(
                TaskQueueChunk.document_id == document_id
            ).count()

            page_count = db.query(TaskQueuePage).filter(
                TaskQueuePage.document_id == document_id
            ).count()

            if page_count == 0 and chunk_count == 0:
                return deleted

            # Delete pages (CASCADE will delete chunks automatically)
            db.query(TaskQueuePage).filter(
                TaskQueuePage.document_id == document_id
            ).delete(synchronize_session=False)

            db.commit()

            deleted["pages"] = page_count
            deleted["chunks"] = chunk_count

            logger.info(f"🧹 Cleaned up queue records for document {document_id}: "
                       f"{page_count} pages, {chunk_count} chunks deleted")

            return deleted

        except Exception as e:
            logger.error(f"Error cleaning up queue for document {document_id}: {e}")
            db.rollback()
            return deleted
        finally:
            if should_close_db:
                db.close()

    def _check_and_cleanup_completed_document(
        self,
        document_id: UUID,
        db: Optional[Session] = None
    ) -> None:
        """
        Check if a document is fully indexed and trigger cleanup if so.

        V3.0: A document is considered fully indexed when:
        - All chunks have vector_status in DONE_STATUSES (completed/skipped)
        - All chunks have graphrag_status in DONE_STATUSES (completed/skipped)

        Args:
            document_id: Document UUID
            db: Database session
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            # Check if all chunks are done (both vector and graphrag)
            chunk_stats = db.query(
                func.count().label("total"),
                func.count().filter(
                    TaskQueueChunk.vector_status.in_(DONE_STATUSES)
                ).label("vector_done"),
                func.count().filter(
                    TaskQueueChunk.graphrag_status.in_(DONE_STATUSES)
                ).label("graphrag_done"),
            ).filter(TaskQueueChunk.document_id == document_id).first()

            if not chunk_stats or chunk_stats.total == 0:
                return

            # Not fully done yet
            if chunk_stats.vector_done < chunk_stats.total:
                return
            if chunk_stats.graphrag_done < chunk_stats.total:
                return

            # All stages complete - cleanup queue records
            logger.info(f"🎉 Document {document_id} fully indexed, cleaning up queue records")
            self.cleanup_completed_document_queue(document_id, db)

        except Exception as e:
            logger.error(f"Error checking cleanup status for document {document_id}: {e}")
        finally:
            if should_close_db:
                db.close()

    # =========================================================================
    # Reindex Task Operations
    # =========================================================================

    def create_reindex_task(
        self,
        document_id: UUID,
        db: Optional[Session] = None
    ) -> bool:
        """
        Create a reindex task for a document.

        This marks a document for reindexing in the queue. The actual reindex
        processing will be done by a background worker, not synchronously.

        Args:
            document_id: Document UUID
            db: Database session

        Returns:
            True if task created/updated successfully
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            # Find or create TaskQueueDocument
            tq_doc = db.query(TaskQueueDocument).filter(
                TaskQueueDocument.document_id == document_id
            ).first()

            if not tq_doc:
                tq_doc = TaskQueueDocument(
                    document_id=document_id,
                    reindex_status=TaskStatus.PENDING,
                    reindex_retry_count=0,
                    reindex_error=None
                )
                db.add(tq_doc)
            else:
                # Reset reindex status
                tq_doc.reindex_status = TaskStatus.PENDING
                tq_doc.reindex_retry_count = 0
                tq_doc.reindex_error = None
                tq_doc.reindex_worker_id = None
                tq_doc.reindex_started_at = None
                tq_doc.reindex_completed_at = None
                tq_doc.reindex_last_heartbeat = None

            db.commit()
            logger.info(f"Created reindex task for document {document_id}")

            # Notify workers
            self.new_task_event.set()
            return True

        except Exception as e:
            logger.error(f"Error creating reindex task: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()

    def claim_reindex_task(
        self,
        worker_id: str,
        db: Optional[Session] = None
    ) -> Optional[ReindexTaskData]:
        """
        Claim the next available reindex task.

        Priority: Failed first (retry), then Pending.

        Args:
            worker_id: Worker identifier
            db: Database session

        Returns:
            ReindexTaskData if task claimed, None otherwise
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            # Find next reindex task (failed first, then pending)
            tq_doc = db.query(TaskQueueDocument).filter(
                TaskQueueDocument.reindex_status.in_([TaskStatus.PENDING, TaskStatus.FAILED]),
                TaskQueueDocument.reindex_retry_count < TaskQueueDocument.max_retries
            ).order_by(
                # Failed first (to retry), then pending
                TaskQueueDocument.reindex_status.desc(),
                TaskQueueDocument.reindex_retry_count.asc(),
                TaskQueueDocument.created_at.asc()
            ).with_for_update(skip_locked=True).first()

            if not tq_doc:
                return None

            # Get the document to retrieve filename and output_path
            doc = db.query(Document).filter(Document.id == tq_doc.document_id).first()
            if not doc:
                logger.error(f"Document not found for reindex task: {tq_doc.document_id}")
                return None

            # Claim the task
            now = datetime.now(timezone.utc)
            tq_doc.reindex_status = TaskStatus.PROCESSING
            tq_doc.reindex_worker_id = worker_id
            tq_doc.reindex_started_at = now
            tq_doc.reindex_last_heartbeat = now

            db.commit()

            logger.info(f"Worker {worker_id} claimed reindex task: doc={tq_doc.document_id}, file={doc.filename}")

            return ReindexTaskData(
                id=tq_doc.id,
                document_id=tq_doc.document_id,
                filename=doc.filename,
                output_path=doc.output_path,
                retry_count=tq_doc.reindex_retry_count
            )

        except Exception as e:
            logger.error(f"Error claiming reindex task: {e}")
            db.rollback()
            return None
        finally:
            if should_close_db:
                db.close()

    def complete_reindex_task(
        self,
        task_id: UUID,
        db: Optional[Session] = None
    ) -> bool:
        """
        Mark reindex task as completed.

        Args:
            task_id: TaskQueueDocument UUID
            db: Database session

        Returns:
            True if successful
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            tq_doc = db.query(TaskQueueDocument).filter(TaskQueueDocument.id == task_id).first()
            if not tq_doc:
                return False

            tq_doc.reindex_status = TaskStatus.COMPLETED
            tq_doc.reindex_completed_at = datetime.now(timezone.utc)
            tq_doc.reindex_error = None

            db.commit()

            logger.info(f"Completed reindex task: doc={tq_doc.document_id}")
            return True

        except Exception as e:
            logger.error(f"Error completing reindex task: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()

    def fail_reindex_task(
        self,
        task_id: UUID,
        error_message: str,
        db: Optional[Session] = None
    ) -> bool:
        """
        Mark reindex task as failed.

        Args:
            task_id: TaskQueueDocument UUID
            error_message: Error description
            db: Database session

        Returns:
            True if successful
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            tq_doc = db.query(TaskQueueDocument).filter(TaskQueueDocument.id == task_id).first()
            if not tq_doc:
                return False

            tq_doc.reindex_retry_count += 1
            tq_doc.reindex_error = error_message

            if tq_doc.reindex_retry_count >= tq_doc.max_retries:
                tq_doc.reindex_status = TaskStatus.FAILED
                logger.error(f"Reindex task permanently failed: doc={tq_doc.document_id}, error={error_message}")
            else:
                tq_doc.reindex_status = TaskStatus.PENDING  # Will be retried
                tq_doc.reindex_worker_id = None
                tq_doc.reindex_last_heartbeat = None
                logger.warning(f"Reindex task failed, will retry: doc={tq_doc.document_id}, retry={tq_doc.reindex_retry_count}")

            db.commit()
            return True

        except Exception as e:
            logger.error(f"Error failing reindex task: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()

    def update_reindex_heartbeat(
        self,
        task_id: UUID,
        db: Optional[Session] = None
    ) -> bool:
        """Update heartbeat for reindex task."""
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            tq_doc = db.query(TaskQueueDocument).filter(TaskQueueDocument.id == task_id).first()
            if tq_doc:
                tq_doc.reindex_last_heartbeat = datetime.now(timezone.utc)
                db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating reindex heartbeat: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()
