"""
Task Scheduler

Periodic scheduler for hierarchical task queue maintenance:
- Release stale tasks (workers that died without updating heartbeat)
- Log queue statistics
"""

import os
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .hierarchical_task_manager import HierarchicalTaskQueueManager

logger = logging.getLogger(__name__)

# Load settings from environment
TASK_QUEUE_CHECK_INTERVAL = int(os.getenv("TASK_QUEUE_CHECK_INTERVAL", "60"))


class TaskScheduler:
    """
    Periodic scheduler for hierarchical task queue maintenance.

    Runs background jobs to:
    1. Release stale tasks (no heartbeat for > stale_timeout_seconds)
    2. Log queue statistics
    """

    def __init__(
        self,
        task_manager: HierarchicalTaskQueueManager,
        check_interval_seconds: int = TASK_QUEUE_CHECK_INTERVAL
    ):
        """
        Initialize the task scheduler.

        Args:
            task_manager: HierarchicalTaskQueueManager instance
            check_interval_seconds: Interval for periodic checks
        """
        self.task_manager = task_manager
        self.check_interval_seconds = check_interval_seconds
        self.scheduler = BackgroundScheduler()
        logger.info(f"TaskScheduler initialized (check_interval={check_interval_seconds}s)")

    def _periodic_maintenance(self):
        """
        Periodic maintenance job.

        1. Release stale tasks
        2. Log queue statistics
        """
        try:
            logger.debug("Running periodic task queue maintenance...")

            # 1. Release stale tasks
            released = self.task_manager.release_stale_tasks()
            if any(v > 0 for v in released.values()):
                logger.warning(f"Released stale tasks: {released}")

            # 2. Log queue statistics
            stats = self.task_manager.get_queue_stats()
            if stats:
                ocr_pending = stats.get("ocr_pending", 0)
                ocr_processing = stats.get("ocr_processing", 0)
                vector_pending = stats.get("vector_pending", 0)
                vector_processing = stats.get("vector_processing", 0)
                graphrag_pending = stats.get("graphrag_pending", 0)
                graphrag_processing = stats.get("graphrag_processing", 0)

                total_pending = ocr_pending + vector_pending + graphrag_pending
                total_processing = ocr_processing + vector_processing + graphrag_processing

                if total_pending > 0 or total_processing > 0:
                    logger.info(
                        f"Queue stats: "
                        f"OCR(pending={ocr_pending}, processing={ocr_processing}), "
                        f"Vector(pending={vector_pending}, processing={vector_processing}), "
                        f"GraphRAG(pending={graphrag_pending}, processing={graphrag_processing})"
                    )

            logger.debug("Periodic maintenance completed")

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
            logger.info(f"TaskScheduler started (interval={self.check_interval_seconds}s)")

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
