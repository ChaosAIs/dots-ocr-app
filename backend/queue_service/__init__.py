"""
Task Queue Service for OCR and Indexing Coordination

This module provides a central queue system for managing OCR and indexing tasks.
It coordinates worker processes and ensures reliable task execution with:
- Automatic task enqueueing on file upload
- Periodic checks for pending/orphaned tasks
- Heartbeat-based stale task detection
- Retry logic for failed tasks
- Resume from checkpoint for long-running tasks
"""

from .models import TaskQueue, TaskType, TaskPriority, TaskStatus
from .task_queue_manager import TaskQueueManager
from .scheduler import TaskScheduler
from .queue_worker_pool import QueueWorkerPool, QueueWorker

__all__ = [
    "TaskQueue",
    "TaskType",
    "TaskPriority",
    "TaskStatus",
    "TaskQueueManager",
    "TaskScheduler",
    "QueueWorkerPool",
    "QueueWorker",
]

