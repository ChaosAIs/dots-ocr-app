"""
Hierarchical Task Queue Service for OCR and Indexing Coordination

This module provides a hierarchical queue system for managing OCR and indexing tasks
with a 3-level hierarchy: Document → Page → Chunk.

Features:
- Three processing phases: OCR → Vector Index → GraphRAG Index
- Sequential dependencies between phases
- Heartbeat-based stale task detection
- Priority-based task pickup (failed first, then pending)
- Automatic status bubble-up from children to parents
- Retry logic for failed tasks
"""

from .models import (
    TaskStatus,
    TaskQueuePage,
    TaskQueueChunk,
    TaskQueueDocument,
)

from .hierarchical_task_manager import (
    HierarchicalTaskQueueManager,
    PageTaskData,
    ChunkTaskData,
    ReindexTaskData,
)

from .hierarchical_worker_pool import (
    HierarchicalWorkerPool,
    HierarchicalWorker,
)

from .scheduler import TaskScheduler

__all__ = [
    "TaskStatus",
    "TaskQueuePage",
    "TaskQueueChunk",
    "TaskQueueDocument",
    "HierarchicalTaskQueueManager",
    "PageTaskData",
    "ChunkTaskData",
    "ReindexTaskData",
    "HierarchicalWorkerPool",
    "HierarchicalWorker",
    "TaskScheduler",
]
