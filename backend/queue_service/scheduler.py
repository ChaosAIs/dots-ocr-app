"""
Task Scheduler

Periodic scheduler for task queue maintenance:
- Release stale tasks (workers that died without updating heartbeat)
- Find and enqueue orphaned documents (documents that need processing but have no task)
- Log queue statistics
"""

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .task_queue_manager import TaskQueueManager
from .models import TaskType, TaskPriority

logger = logging.getLogger(__name__)


class TaskScheduler:
    """
    Periodic scheduler for task queue maintenance.
    
    Runs background jobs to:
    1. Release stale tasks (no heartbeat for > 5 minutes)
    2. Find orphaned documents and enqueue them
    3. Log queue statistics
    """
    
    def __init__(self, task_queue_manager: TaskQueueManager, check_interval_seconds: int = 300):
        """
        Initialize the task scheduler.
        
        Args:
            task_queue_manager: TaskQueueManager instance
            check_interval_seconds: Interval for periodic checks (default: 5 minutes)
        """
        self.task_queue_manager = task_queue_manager
        self.check_interval_seconds = check_interval_seconds
        self.scheduler = BackgroundScheduler()
        logger.info(f"TaskScheduler initialized (check_interval={check_interval_seconds}s)")
    
    def _periodic_maintenance(self):
        """
        Periodic maintenance job.
        
        1. Release stale tasks
        2. Find and enqueue orphaned documents
        3. Log queue statistics
        """
        try:
            logger.info("🔄 Running periodic task queue maintenance...")
            
            # 1. Release stale tasks
            stale_count = self.task_queue_manager.release_stale_tasks()
            if stale_count > 0:
                logger.warning(f"Released {stale_count} stale tasks")
            
            # 2. Find orphaned OCR documents
            orphaned_ocr = self.task_queue_manager.find_orphaned_ocr_documents()
            if orphaned_ocr:
                logger.info(f"Found {len(orphaned_ocr)} orphaned OCR documents, enqueueing...")
                for doc_id in orphaned_ocr:
                    self.task_queue_manager.enqueue_task(
                        document_id=doc_id,
                        task_type=TaskType.OCR,
                        priority=TaskPriority.NORMAL
                    )
            
            # 3. Find orphaned indexing documents
            orphaned_indexing = self.task_queue_manager.find_orphaned_indexing_documents()
            if orphaned_indexing:
                logger.info(f"Found {len(orphaned_indexing)} orphaned indexing documents, enqueueing...")
                for doc_id in orphaned_indexing:
                    self.task_queue_manager.enqueue_task(
                        document_id=doc_id,
                        task_type=TaskType.INDEXING,
                        priority=TaskPriority.NORMAL
                    )
            
            # 4. Log queue statistics
            stats = self.task_queue_manager.get_queue_stats()
            logger.info(f"📊 Queue stats: {stats}")
            
            logger.info("✅ Periodic maintenance completed")
        
        except Exception as e:
            logger.error(f"Error in periodic maintenance: {e}", exc_info=True)
    
    def start(self):
        """Start the scheduler"""
        try:
            # Add periodic maintenance job
            self.scheduler.add_job(
                func=self._periodic_maintenance,
                trigger=IntervalTrigger(seconds=self.check_interval_seconds),
                id="task_queue_maintenance",
                name="Task Queue Maintenance",
                replace_existing=True
            )
            
            self.scheduler.start()
            logger.info(f"✅ TaskScheduler started (interval={self.check_interval_seconds}s)")
            
            # Run maintenance immediately on startup
            self._periodic_maintenance()
        
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}", exc_info=True)
    
    def stop(self):
        """Stop the scheduler"""
        try:
            self.scheduler.shutdown(wait=False)
            logger.info("TaskScheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}", exc_info=True)

