"""
Task Queue Manager

Manages the task queue for OCR and indexing operations.
Provides methods for enqueueing, claiming, and managing tasks.
"""

import os
import socket
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, List, NamedTuple
from uuid import UUID
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from db.database import get_db_session, create_db_session
from db.models import Document, ConvertStatus, IndexStatus
from .models import TaskQueue, TaskType, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)


class TaskData(NamedTuple):
    """Simple data class for task information (detached from SQLAlchemy session)"""
    id: UUID
    document_id: UUID
    task_type: TaskType
    priority: TaskPriority
    status: TaskStatus
    worker_id: Optional[str]
    retry_count: int
    max_retries: int


class TaskQueueManager:
    """
    Manages the task queue for OCR and indexing coordination.
    
    This class provides methods to:
    - Enqueue new tasks
    - Claim tasks for processing
    - Update task status and heartbeat
    - Detect and release stale tasks
    - Find orphaned documents
    """
    
    def __init__(self, stale_timeout_seconds: int = 300, heartbeat_interval_seconds: int = 30):
        """
        Initialize the task queue manager.
        
        Args:
            stale_timeout_seconds: Timeout for stale task detection (default: 5 minutes)
            heartbeat_interval_seconds: Interval for worker heartbeat updates (default: 30 seconds)
        """
        self.stale_timeout_seconds = stale_timeout_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.new_task_event = threading.Event()  # For instant notification
        logger.info(f"TaskQueueManager initialized (stale_timeout={stale_timeout_seconds}s, heartbeat={heartbeat_interval_seconds}s)")
    
    def _generate_worker_id(self, thread_id: int) -> str:
        """Generate unique worker ID: hostname:pid:thread_id"""
        return f"{self.hostname}:{self.pid}:{thread_id}"
    
    def enqueue_task(
        self,
        document_id: UUID,
        task_type: TaskType,
        priority: TaskPriority = TaskPriority.NORMAL,
        db: Optional[Session] = None
    ) -> Optional[UUID]:
        """
        Enqueue a new task for processing.
        
        Args:
            document_id: Document ID
            task_type: Type of task (OCR or INDEXING)
            priority: Task priority (HIGH, NORMAL, LOW)
            db: Database session (optional, will create if not provided)
        
        Returns:
            Task ID if created, None if task already exists
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()
        
        try:
            # Check if task already exists (PENDING or CLAIMED)
            existing_task = db.query(TaskQueue).filter(
                and_(
                    TaskQueue.document_id == document_id,
                    TaskQueue.task_type == task_type,
                    TaskQueue.status.in_([TaskStatus.PENDING, TaskStatus.CLAIMED])
                )
            ).first()
            
            if existing_task:
                logger.info(f"Task already exists: {task_type.value} for document {document_id} (status={existing_task.status.value})")
                return None
            
            # Create new task
            task = TaskQueue(
                document_id=document_id,
                task_type=task_type,
                priority=priority,
                status=TaskStatus.PENDING
            )
            db.add(task)
            db.commit()
            db.refresh(task)
            
            logger.info(f"✅ Enqueued task: {task_type.value} for document {document_id} (priority={priority.value}, task_id={task.id})")
            
            # Notify workers immediately
            self.new_task_event.set()
            
            return task.id
        
        except Exception as e:
            logger.error(f"Error enqueueing task: {e}")
            db.rollback()
            return None
        finally:
            if should_close_db:
                db.close()
    
    def claim_next_task(self, worker_id: str, db: Optional[Session] = None) -> Optional[TaskData]:
        """
        Claim the next available task for processing.

        Uses SELECT FOR UPDATE SKIP LOCKED for atomic claiming.
        Tasks are ordered by priority (HIGH > NORMAL > LOW) and creation time.

        Args:
            worker_id: Unique worker identifier
            db: Database session (optional)

        Returns:
            Claimed task data or None if no tasks available
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            # Claim next pending task atomically
            task = db.query(TaskQueue).filter(
                TaskQueue.status == TaskStatus.PENDING
            ).order_by(
                TaskQueue.priority.desc(),  # HIGH > NORMAL > LOW
                TaskQueue.created_at.asc()  # FIFO within same priority
            ).with_for_update(skip_locked=True).first()

            if not task:
                return None

            # Update task status
            now = datetime.utcnow()
            task.status = TaskStatus.CLAIMED
            task.worker_id = worker_id
            task.claimed_at = now
            task.last_heartbeat = now
            task.started_at = now

            db.commit()
            db.refresh(task)

            logger.info(f"✅ Worker {worker_id} claimed task: {task.task_type.value} for document {task.document_id} (task_id={task.id})")

            # Extract task data before closing session
            task_data = TaskData(
                id=task.id,
                document_id=task.document_id,
                task_type=task.task_type,
                priority=task.priority,
                status=task.status,
                worker_id=task.worker_id,
                retry_count=task.retry_count,
                max_retries=task.max_retries
            )

            return task_data

        except Exception as e:
            logger.error(f"Error claiming task: {e}")
            db.rollback()
            return None
        finally:
            if should_close_db:
                db.close()

    def update_heartbeat(self, task_id: UUID, db: Optional[Session] = None) -> bool:
        """
        Update task heartbeat to indicate worker is still alive.

        Args:
            task_id: Task ID
            db: Database session (optional)

        Returns:
            True if updated successfully, False otherwise
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            task = db.query(TaskQueue).filter(TaskQueue.id == task_id).first()
            if not task:
                return False

            task.last_heartbeat = datetime.utcnow()
            db.commit()
            return True

        except Exception as e:
            logger.error(f"Error updating heartbeat: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()

    def complete_task(self, task_id: UUID, db: Optional[Session] = None) -> bool:
        """
        Mark task as completed.

        Args:
            task_id: Task ID
            db: Database session (optional)

        Returns:
            True if updated successfully, False otherwise
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            task = db.query(TaskQueue).filter(TaskQueue.id == task_id).first()
            if not task:
                return False

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            db.commit()

            logger.info(f"✅ Task completed: {task.task_type.value} for document {task.document_id} (task_id={task_id})")
            return True

        except Exception as e:
            logger.error(f"Error completing task: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()

    def fail_task(self, task_id: UUID, error_message: str, db: Optional[Session] = None) -> bool:
        """
        Mark task as failed and handle retry logic.

        Args:
            task_id: Task ID
            error_message: Error message
            db: Database session (optional)

        Returns:
            True if updated successfully, False otherwise
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            task = db.query(TaskQueue).filter(TaskQueue.id == task_id).first()
            if not task:
                return False

            task.retry_count += 1
            task.error_message = error_message

            if task.retry_count >= task.max_retries:
                # Max retries reached - mark as FAILED
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.utcnow()
                logger.error(f"❌ Task failed permanently: {task.task_type.value} for document {task.document_id} (retries={task.retry_count}, error={error_message})")
            else:
                # Retry - release back to PENDING
                task.status = TaskStatus.PENDING
                task.worker_id = None
                task.claimed_at = None
                task.last_heartbeat = None
                logger.warning(f"⚠️ Task failed, will retry: {task.task_type.value} for document {task.document_id} (retry={task.retry_count}/{task.max_retries}, error={error_message})")

            db.commit()
            return True

        except Exception as e:
            logger.error(f"Error failing task: {e}")
            db.rollback()
            return False
        finally:
            if should_close_db:
                db.close()

    def release_stale_tasks(self, db: Optional[Session] = None) -> int:
        """
        Release stale tasks (no heartbeat for > stale_timeout_seconds).

        Args:
            db: Database session (optional)

        Returns:
            Number of tasks released
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            stale_threshold = datetime.utcnow() - timedelta(seconds=self.stale_timeout_seconds)

            # Find stale tasks
            stale_tasks = db.query(TaskQueue).filter(
                and_(
                    TaskQueue.status == TaskStatus.CLAIMED,
                    TaskQueue.last_heartbeat < stale_threshold
                )
            ).all()

            count = 0
            for task in stale_tasks:
                logger.warning(f"⚠️ Releasing stale task: {task.task_type.value} for document {task.document_id} (worker={task.worker_id}, last_heartbeat={task.last_heartbeat})")

                # Release back to PENDING
                task.status = TaskStatus.PENDING
                task.worker_id = None
                task.claimed_at = None
                task.last_heartbeat = None
                count += 1

            if count > 0:
                db.commit()
                logger.info(f"Released {count} stale tasks")

            return count

        except Exception as e:
            logger.error(f"Error releasing stale tasks: {e}")
            db.rollback()
            return 0
        finally:
            if should_close_db:
                db.close()

    def find_orphaned_ocr_documents(self, db: Optional[Session] = None) -> List[UUID]:
        """
        Find documents that need OCR but don't have a task in the queue.

        This includes:
        1. Documents with convert_status = PENDING (never started)
        2. Documents with convert_status = CONVERTING (started but worker died/crashed)

        Returns:
            List of document IDs
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            # Find documents with convert_status in (PENDING, CONVERTING) but no active OCR task
            orphaned = db.query(Document.id).filter(
                and_(
                    Document.convert_status.in_([ConvertStatus.PENDING, ConvertStatus.CONVERTING]),
                    ~Document.id.in_(
                        db.query(TaskQueue.document_id).filter(
                            and_(
                                TaskQueue.task_type == TaskType.OCR,
                                TaskQueue.status.in_([TaskStatus.PENDING, TaskStatus.CLAIMED])
                            )
                        )
                    )
                )
            ).all()

            return [doc_id for (doc_id,) in orphaned]

        except Exception as e:
            logger.error(f"Error finding orphaned OCR documents: {e}")
            return []
        finally:
            if should_close_db:
                db.close()

    def find_orphaned_indexing_documents(self, db: Optional[Session] = None) -> List[UUID]:
        """
        Find documents that need indexing but don't have a task in the queue.

        This includes:
        1. Documents with index_status = PENDING (never started)
        2. Documents with index_status = INDEXING (started but worker died/crashed)

        Returns:
            List of document IDs
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            # Find documents with index_status in (PENDING, INDEXING) and convert_status = CONVERTED
            # but no active INDEXING task in the queue
            orphaned = db.query(Document.id).filter(
                and_(
                    Document.index_status.in_([IndexStatus.PENDING, IndexStatus.INDEXING]),
                    Document.convert_status == ConvertStatus.CONVERTED,
                    ~Document.id.in_(
                        db.query(TaskQueue.document_id).filter(
                            and_(
                                TaskQueue.task_type == TaskType.INDEXING,
                                TaskQueue.status.in_([TaskStatus.PENDING, TaskStatus.CLAIMED])
                            )
                        )
                    )
                )
            ).all()

            return [doc_id for (doc_id,) in orphaned]

        except Exception as e:
            logger.error(f"Error finding orphaned indexing documents: {e}")
            return []
        finally:
            if should_close_db:
                db.close()

    def get_queue_stats(self, db: Optional[Session] = None) -> dict:
        """
        Get queue statistics.

        Returns:
            Dictionary with queue statistics
        """
        should_close_db = db is None
        if db is None:
            db = create_db_session()

        try:
            stats = {
                "pending": db.query(TaskQueue).filter(TaskQueue.status == TaskStatus.PENDING).count(),
                "claimed": db.query(TaskQueue).filter(TaskQueue.status == TaskStatus.CLAIMED).count(),
                "completed": db.query(TaskQueue).filter(TaskQueue.status == TaskStatus.COMPLETED).count(),
                "failed": db.query(TaskQueue).filter(TaskQueue.status == TaskStatus.FAILED).count(),
                "ocr_pending": db.query(TaskQueue).filter(
                    and_(TaskQueue.task_type == TaskType.OCR, TaskQueue.status == TaskStatus.PENDING)
                ).count(),
                "indexing_pending": db.query(TaskQueue).filter(
                    and_(TaskQueue.task_type == TaskType.INDEXING, TaskQueue.status == TaskStatus.PENDING)
                ).count(),
            }
            return stats

        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {}
        finally:
            if should_close_db:
                db.close()

