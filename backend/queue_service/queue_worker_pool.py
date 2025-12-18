"""
Queue-based Worker Pool

Worker pool that claims tasks from the database task queue and processes them.
Supports checkpoint-based resumption for long-running tasks.
"""

import threading
import time
import logging
from typing import Optional, Callable
from uuid import UUID

from db.database import get_db_session
from db.document_repository import DocumentRepository
from db.models import ConvertStatus, IndexStatus
from .task_queue_manager import TaskQueueManager, TaskData
from .models import TaskType, TaskQueue

logger = logging.getLogger(__name__)


class QueueWorker(threading.Thread):
    """
    Worker thread that claims and processes tasks from the database queue.
    
    Each worker:
    1. Claims next available task from queue
    2. Checks document status to determine if resume is needed
    3. Processes task (OCR or indexing) with checkpoint-based resume
    4. Updates heartbeat periodically
    5. Marks task as completed or failed
    """
    
    def __init__(
        self,
        worker_id: int,
        task_queue_manager: TaskQueueManager,
        ocr_processor: Callable,
        indexing_processor: Callable,
        poll_interval: int = 5,
        heartbeat_interval: int = 30
    ):
        """
        Initialize queue worker.
        
        Args:
            worker_id: Unique worker ID
            task_queue_manager: TaskQueueManager instance
            ocr_processor: Function to process OCR tasks (document_id) -> result
            indexing_processor: Function to process indexing tasks (document_id) -> result
            poll_interval: Seconds to wait between queue polls (default: 5)
            heartbeat_interval: Seconds between heartbeat updates (default: 30)
        """
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.task_queue_manager = task_queue_manager
        self.ocr_processor = ocr_processor
        self.indexing_processor = indexing_processor
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.running = True
        self.current_task: Optional[TaskQueue] = None
        self.last_heartbeat = 0
        
        # Generate unique worker identifier
        self.worker_identifier = task_queue_manager._generate_worker_id(worker_id)
        logger.info(f"QueueWorker {self.worker_identifier} initialized")
    
    def run(self):
        """Main worker loop"""
        logger.info(f"QueueWorker {self.worker_identifier} started")
        
        while self.running:
            try:
                # Claim next task
                with get_db_session() as db:
                    task = self.task_queue_manager.claim_next_task(self.worker_identifier, db)
                
                if not task:
                    # No tasks available - wait and check for new task event
                    if self.task_queue_manager.new_task_event.wait(timeout=self.poll_interval):
                        # New task available - clear event and try again immediately
                        self.task_queue_manager.new_task_event.clear()
                        continue
                    else:
                        # Timeout - continue polling
                        continue
                
                # Process the task
                self.current_task = task
                self._process_task(task)
                self.current_task = None
            
            except Exception as e:
                logger.error(f"Error in worker {self.worker_identifier}: {e}", exc_info=True)
                time.sleep(self.poll_interval)
        
        logger.info(f"QueueWorker {self.worker_identifier} stopped")
    
    def _process_task(self, task: TaskData):
        """
        Process a task with heartbeat updates.

        Args:
            task: TaskData instance (detached from SQLAlchemy session)
        """
        try:
            logger.info(f"Worker {self.worker_identifier} processing {task.task_type.value} for document {task.document_id}")

            # Start heartbeat thread
            heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                args=(task.id,),
                daemon=True
            )
            heartbeat_thread.start()

            # Process based on task type
            if task.task_type == TaskType.OCR:
                result = self.ocr_processor(task.document_id)
            elif task.task_type == TaskType.INDEXING:
                result = self.indexing_processor(task.document_id)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            # Mark as completed
            with get_db_session() as db:
                self.task_queue_manager.complete_task(task.id, db)

            logger.info(f"✅ Worker {self.worker_identifier} completed {task.task_type.value} for document {task.document_id}")

        except Exception as e:
            logger.error(f"❌ Worker {self.worker_identifier} failed {task.task_type.value} for document {task.document_id}: {e}", exc_info=True)

            # Mark as failed (will retry if retries available)
            with get_db_session() as db:
                self.task_queue_manager.fail_task(task.id, str(e), db)
    
    def _heartbeat_loop(self, task_id: UUID):
        """
        Heartbeat loop to update task heartbeat.
        
        Args:
            task_id: Task ID
        """
        while self.current_task and self.current_task.id == task_id:
            try:
                with get_db_session() as db:
                    self.task_queue_manager.update_heartbeat(task_id, db)
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error updating heartbeat: {e}")
                break
    
    def stop(self):
        """Stop the worker gracefully"""
        self.running = False


class QueueWorkerPool:
    """
    Worker pool that processes tasks from the database queue.

    Manages multiple QueueWorker threads that claim and process tasks.
    """

    def __init__(
        self,
        num_workers: int,
        task_queue_manager: TaskQueueManager,
        ocr_processor: Callable,
        indexing_processor: Callable,
        poll_interval: int = 5,
        heartbeat_interval: int = 30
    ):
        """
        Initialize queue worker pool.

        Args:
            num_workers: Number of worker threads
            task_queue_manager: TaskQueueManager instance
            ocr_processor: Function to process OCR tasks
            indexing_processor: Function to process indexing tasks
            poll_interval: Seconds to wait between queue polls
            heartbeat_interval: Seconds between heartbeat updates
        """
        self.num_workers = num_workers
        self.task_queue_manager = task_queue_manager
        self.ocr_processor = ocr_processor
        self.indexing_processor = indexing_processor
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.workers = []

        logger.info(f"QueueWorkerPool initializing with {num_workers} workers...")

    def start(self):
        """Start all worker threads"""
        for i in range(self.num_workers):
            worker = QueueWorker(
                worker_id=i,
                task_queue_manager=self.task_queue_manager,
                ocr_processor=self.ocr_processor,
                indexing_processor=self.indexing_processor,
                poll_interval=self.poll_interval,
                heartbeat_interval=self.heartbeat_interval
            )
            worker.start()
            self.workers.append(worker)

        logger.info(f"✅ QueueWorkerPool started with {self.num_workers} workers")

    def stop(self, wait: bool = True):
        """
        Stop all worker threads.

        Args:
            wait: Whether to wait for workers to finish
        """
        logger.info("Stopping QueueWorkerPool...")

        for worker in self.workers:
            worker.stop()

        if wait:
            for worker in self.workers:
                worker.join(timeout=5)

        logger.info("QueueWorkerPool stopped")

    def get_active_workers_count(self) -> int:
        """Get number of active workers"""
        return sum(1 for w in self.workers if w.current_task is not None)

