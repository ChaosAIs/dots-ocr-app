"""
Hierarchical Worker Pool

Worker pool that processes tasks from the hierarchical task queue.
Supports three processing phases: OCR → Vector Index → GraphRAG Index.
"""

import os
import threading
import time
import logging
from typing import Optional, Callable, Union

from .hierarchical_task_manager import (
    HierarchicalTaskQueueManager,
    PageTaskData,
    ChunkTaskData,
    PhaseType
)

logger = logging.getLogger(__name__)

# Load settings from environment
WORKER_POLL_INTERVAL = int(os.getenv("WORKER_POLL_INTERVAL", "5"))
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "30"))


class HierarchicalWorker(threading.Thread):
    """
    Worker thread that claims and processes tasks from the hierarchical queue.

    Each worker can process all three phases:
    1. OCR: Process PDF pages to generate markdown
    2. Vector: Index chunks to Qdrant
    3. GraphRAG: Extract entities to Neo4j

    Priority order for task pickup:
    1. Failed tasks (retry first)
    2. Pending tasks (new work)
    3. Chunks before Pages (granular first)
    """

    def __init__(
        self,
        worker_id: int,
        task_manager: HierarchicalTaskQueueManager,
        ocr_processor: Callable[[PageTaskData], str],  # Returns page_file_path
        vector_processor: Callable[[ChunkTaskData], bool],  # Returns success
        graphrag_processor: Callable[[ChunkTaskData], tuple],  # Returns (entities, relationships)
        poll_interval: int = WORKER_POLL_INTERVAL,
        heartbeat_interval: int = HEARTBEAT_INTERVAL,
        status_broadcast_callback: Optional[Callable[[dict], None]] = None
    ):
        """
        Initialize hierarchical worker.

        Args:
            worker_id: Unique worker ID
            task_manager: HierarchicalTaskQueueManager instance
            ocr_processor: Function to process OCR tasks (page_data) -> page_file_path
            vector_processor: Function to process Vector tasks (chunk_data) -> success
            graphrag_processor: Function to process GraphRAG tasks (chunk_data) -> (entities, relationships)
            poll_interval: Seconds to wait between queue polls
            heartbeat_interval: Seconds between heartbeat updates
            status_broadcast_callback: Callback to broadcast status updates to frontend via WebSocket
        """
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.task_manager = task_manager
        self.ocr_processor = ocr_processor
        self.vector_processor = vector_processor
        self.graphrag_processor = graphrag_processor
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.status_broadcast_callback = status_broadcast_callback
        self.running = True
        self.current_task: Optional[Union[PageTaskData, ChunkTaskData]] = None
        self.current_phase: Optional[PhaseType] = None

        # Generate unique worker identifier
        self.worker_identifier = task_manager._generate_worker_id(worker_id)
        logger.info(f"HierarchicalWorker {self.worker_identifier} initialized")

    def run(self):
        """Main worker loop"""
        logger.info(f"HierarchicalWorker {self.worker_identifier} started")

        while self.running:
            try:
                task_found = False

                # Priority 1: Try to claim GraphRAG chunk task (most granular, depends on Vector)
                chunk_task = self.task_manager.claim_graphrag_chunk_task(self.worker_identifier)
                if chunk_task:
                    task_found = True
                    self.current_task = chunk_task
                    self.current_phase = "graphrag"
                    self._process_graphrag_chunk(chunk_task)
                    self.current_task = None
                    self.current_phase = None
                    continue

                # Priority 2: Try to claim Vector chunk task (depends on OCR)
                chunk_task = self.task_manager.claim_vector_chunk_task(self.worker_identifier)
                if chunk_task:
                    task_found = True
                    self.current_task = chunk_task
                    self.current_phase = "vector"
                    self._process_vector_chunk(chunk_task)
                    self.current_task = None
                    self.current_phase = None
                    continue

                # Priority 3: Try to claim OCR page task
                page_task = self.task_manager.claim_ocr_page_task(self.worker_identifier)
                if page_task:
                    task_found = True
                    self.current_task = page_task
                    self.current_phase = "ocr"
                    self._process_ocr_page(page_task)
                    self.current_task = None
                    self.current_phase = None
                    continue

                # No tasks available - wait and check for new task event
                if not task_found:
                    if self.task_manager.new_task_event.wait(timeout=self.poll_interval):
                        # New task available - clear event and try again immediately
                        self.task_manager.new_task_event.clear()
                    # else: Timeout - continue polling

            except Exception as e:
                logger.error(f"Error in worker {self.worker_identifier}: {e}", exc_info=True)
                time.sleep(self.poll_interval)

        logger.info(f"HierarchicalWorker {self.worker_identifier} stopped")

    def _broadcast_document_status(self, document_id, phase: str, status: str):
        """
        Broadcast document status update to frontend via WebSocket.

        Args:
            document_id: Document UUID
            phase: Which phase completed ("ocr", "vector", "graphrag")
            status: New status ("completed", "failed", etc.)
        """
        if not self.status_broadcast_callback:
            return

        try:
            # Get updated document info from task manager
            from db.database import create_db_session
            from db.models import Document, IndexStatus

            db = create_db_session()
            try:
                doc = db.query(Document).filter(Document.id == document_id).first()
                if doc:
                    # Broadcast phase-specific status update
                    # Use event_type to match frontend WebSocket handler expectations
                    event_type = f"{phase}_completed" if status == "completed" else f"{phase}_failed"
                    self.status_broadcast_callback({
                        "event_type": event_type,
                        "type": "document_status",
                        "document_id": str(document_id),
                        "filename": doc.filename,
                        "ocr_status": doc.ocr_status.value if doc.ocr_status else None,
                        "vector_status": doc.vector_status.value if doc.vector_status else None,
                        "graphrag_status": doc.graphrag_status.value if doc.graphrag_status else None,
                        "index_status": doc.index_status.value if doc.index_status else None,
                        "phase_completed": phase,
                        "phase_status": status,
                        "indexing_details": doc.indexing_details
                    })
                    logger.debug(f"Broadcast status update: doc={document_id}, {phase}={status}, event_type={event_type}")

                    # Also broadcast indexing_completed if document is now fully indexed
                    # This ensures frontend gets the final status update to refresh the document list
                    if doc.index_status == IndexStatus.INDEXED:
                        logger.info(f"📤 Document {document_id} fully indexed - broadcasting indexing_completed")
                        self.status_broadcast_callback({
                            "event_type": "indexing_completed",
                            "type": "document_status",
                            "document_id": str(document_id),
                            "filename": doc.filename,
                            "index_status": "indexed",
                            "progress": 100,
                            "message": "Indexing completed",
                            "indexing_details": doc.indexing_details
                        })
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error broadcasting document status: {e}")

    def _process_ocr_page(self, task: PageTaskData):
        """Process an OCR page task with heartbeat updates."""
        try:
            logger.info(f"Worker {self.worker_identifier} processing OCR page: doc={task.document_id}, page={task.page_number}")

            # Start heartbeat thread
            heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                args=(task.id, "ocr"),
                daemon=True
            )
            heartbeat_thread.start()

            # Process OCR
            page_file_path = self.ocr_processor(task)

            # Mark as completed
            self.task_manager.complete_ocr_page_task(task.id, page_file_path)
            logger.info(f"Worker {self.worker_identifier} completed OCR page: doc={task.document_id}, page={task.page_number}")

            # Broadcast status update to frontend
            self._broadcast_document_status(task.document_id, "ocr", "completed")

        except Exception as e:
            logger.error(f"Worker {self.worker_identifier} failed OCR page: doc={task.document_id}, page={task.page_number}: {e}", exc_info=True)
            self.task_manager.fail_ocr_page_task(task.id, str(e))
            # Broadcast failure status
            self._broadcast_document_status(task.document_id, "ocr", "failed")

    def _process_vector_chunk(self, task: ChunkTaskData):
        """Process a Vector indexing chunk task with heartbeat updates."""
        try:
            logger.debug(f"Worker {self.worker_identifier} processing Vector chunk: chunk={task.chunk_id}")

            # Start heartbeat thread
            heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                args=(task.id, "vector"),
                daemon=True
            )
            heartbeat_thread.start()

            # Process Vector indexing
            success = self.vector_processor(task)

            if success:
                self.task_manager.complete_vector_chunk_task(task.id)
                logger.debug(f"Worker {self.worker_identifier} completed Vector chunk: chunk={task.chunk_id}")
                # Broadcast status update to frontend
                self._broadcast_document_status(task.document_id, "vector", "completed")
            else:
                self.task_manager.fail_vector_chunk_task(task.id, "Vector indexing returned failure")
                self._broadcast_document_status(task.document_id, "vector", "failed")

        except Exception as e:
            logger.error(f"Worker {self.worker_identifier} failed Vector chunk: chunk={task.chunk_id}: {e}", exc_info=True)
            self.task_manager.fail_vector_chunk_task(task.id, str(e))
            self._broadcast_document_status(task.document_id, "vector", "failed")

    def _process_graphrag_chunk(self, task: ChunkTaskData):
        """Process a GraphRAG indexing chunk task with heartbeat updates."""
        try:
            logger.debug(f"Worker {self.worker_identifier} processing GraphRAG chunk: chunk={task.chunk_id}")

            # Start heartbeat thread
            heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                args=(task.id, "graphrag"),
                daemon=True
            )
            heartbeat_thread.start()

            # Process GraphRAG indexing
            entities, relationships = self.graphrag_processor(task)

            # Mark as completed
            self.task_manager.complete_graphrag_chunk_task(task.id, entities, relationships)
            logger.debug(f"Worker {self.worker_identifier} completed GraphRAG chunk: chunk={task.chunk_id}, entities={entities}")

            # Broadcast status update to frontend
            self._broadcast_document_status(task.document_id, "graphrag", "completed")

        except Exception as e:
            logger.error(f"Worker {self.worker_identifier} failed GraphRAG chunk: chunk={task.chunk_id}: {e}", exc_info=True)
            self.task_manager.fail_graphrag_chunk_task(task.id, str(e))
            self._broadcast_document_status(task.document_id, "graphrag", "failed")

    def _heartbeat_loop(self, task_id, phase: PhaseType):
        """
        Heartbeat loop to update task heartbeat.

        Args:
            task_id: Task ID (page or chunk)
            phase: Which phase is being processed
        """
        while self.current_task and self.current_task.id == task_id:
            try:
                if phase == "ocr":
                    self.task_manager.update_ocr_heartbeat(task_id)
                elif phase == "vector":
                    self.task_manager.update_vector_heartbeat(task_id)
                elif phase == "graphrag":
                    self.task_manager.update_graphrag_heartbeat(task_id)

                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error updating heartbeat: {e}")
                break

    def stop(self):
        """Stop the worker gracefully"""
        self.running = False


class HierarchicalWorkerPool:
    """
    Worker pool that processes tasks from the hierarchical queue.

    Manages multiple HierarchicalWorker threads.
    """

    def __init__(
        self,
        num_workers: int,
        task_manager: HierarchicalTaskQueueManager,
        ocr_processor: Callable,
        vector_processor: Callable,
        graphrag_processor: Callable,
        poll_interval: int = WORKER_POLL_INTERVAL,
        heartbeat_interval: int = HEARTBEAT_INTERVAL,
        status_broadcast_callback: Optional[Callable[[dict], None]] = None
    ):
        """
        Initialize hierarchical worker pool.

        Args:
            num_workers: Number of worker threads
            task_manager: HierarchicalTaskQueueManager instance
            ocr_processor: Function to process OCR tasks
            vector_processor: Function to process Vector tasks
            graphrag_processor: Function to process GraphRAG tasks
            poll_interval: Seconds to wait between queue polls
            heartbeat_interval: Seconds between heartbeat updates
            status_broadcast_callback: Callback to broadcast status updates to frontend via WebSocket
        """
        self.num_workers = num_workers
        self.task_manager = task_manager
        self.ocr_processor = ocr_processor
        self.vector_processor = vector_processor
        self.graphrag_processor = graphrag_processor
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.status_broadcast_callback = status_broadcast_callback
        self.workers = []

        logger.info(f"HierarchicalWorkerPool initializing with {num_workers} workers...")

    def start(self):
        """Start all worker threads"""
        for i in range(self.num_workers):
            worker = HierarchicalWorker(
                worker_id=i,
                task_manager=self.task_manager,
                ocr_processor=self.ocr_processor,
                vector_processor=self.vector_processor,
                graphrag_processor=self.graphrag_processor,
                poll_interval=self.poll_interval,
                heartbeat_interval=self.heartbeat_interval,
                status_broadcast_callback=self.status_broadcast_callback
            )
            worker.start()
            self.workers.append(worker)

        logger.info(f"HierarchicalWorkerPool started with {self.num_workers} workers")

    def stop(self, wait: bool = True):
        """
        Stop all worker threads.

        Args:
            wait: Whether to wait for workers to finish
        """
        logger.info("Stopping HierarchicalWorkerPool...")

        for worker in self.workers:
            worker.stop()

        if wait:
            for worker in self.workers:
                worker.join(timeout=5)

        logger.info("HierarchicalWorkerPool stopped")

    def get_active_workers_count(self) -> int:
        """Get number of active workers (currently processing a task)"""
        return sum(1 for w in self.workers if w.current_task is not None)

    def get_workers_status(self) -> list:
        """Get status of all workers"""
        return [
            {
                "worker_id": w.worker_identifier,
                "is_alive": w.is_alive(),
                "current_phase": w.current_phase,
                "current_task": str(w.current_task.id) if w.current_task else None
            }
            for w in self.workers
        ]
