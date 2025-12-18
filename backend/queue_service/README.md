# Task Queue System

## Overview

The Task Queue System provides automatic, reliable, and resumable processing of OCR and indexing tasks. It replaces manual triggers with a fully automated background processing system.

## Architecture

### Components

1. **TaskQueue** (`models.py`)
   - Database model for task coordination
   - Tracks task status, worker assignment, and retry logic
   - Uses PostgreSQL for ACID guarantees

2. **TaskQueueManager** (`task_queue_manager.py`)
   - Core queue management logic
   - Handles task enqueueing, claiming, completion, and failure
   - Implements heartbeat system for worker liveness detection
   - Detects orphaned documents and stale tasks

3. **TaskScheduler** (`scheduler.py`)
   - Periodic maintenance scheduler (default: every 5 minutes)
   - Releases stale tasks (no heartbeat for > 5 minutes)
   - Finds and enqueues orphaned documents
   - Logs queue statistics

4. **QueueWorkerPool** (`queue_worker_pool.py`)
   - Manages multiple worker threads
   - Workers claim tasks from database queue
   - Processes tasks with checkpoint-based resume
   - Updates heartbeat periodically

### Database Schema

```sql
CREATE TABLE task_queue (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    task_type VARCHAR(20),  -- 'OCR' or 'INDEXING'
    priority VARCHAR(10),   -- 'HIGH', 'NORMAL', 'LOW'
    status VARCHAR(20),     -- 'PENDING', 'CLAIMED', 'COMPLETED', 'FAILED', 'CANCELLED'
    worker_id VARCHAR(100),
    claimed_at TIMESTAMP,
    last_heartbeat TIMESTAMP,
    retry_count INTEGER,
    max_retries INTEGER,
    error_message TEXT,
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### Progress Tracking

Progress is tracked in the `documents` table JSONB fields:

- **`ocr_details`**: Page-level OCR status, embedded images, retry counts
- **`indexing_details`**: Phase-level indexing status (vector, metadata, GraphRAG)

The `task_queue` table is ONLY for worker coordination, NOT for progress tracking.

## Features

### 1. Automatic Processing

- Files are automatically processed on upload (no manual triggers needed)
- OCR tasks are enqueued with HIGH priority
- Indexing tasks are enqueued automatically after OCR completion

### 2. Checkpoint-Based Resume

- **OCR**: Resumes from last successfully converted page
- **Indexing**: Resumes from last completed phase (vector → metadata → GraphRAG)
- Workers can take over incomplete tasks after restart

### 3. Worker Coordination

- Atomic task claiming using `SELECT FOR UPDATE SKIP LOCKED`
- Heartbeat system prevents duplicate work
- Stale task detection (no heartbeat for > 5 minutes)
- Automatic retry with exponential backoff

### 4. Priority System

- **HIGH**: User uploads (processed immediately)
- **NORMAL**: Auto-resume, scheduled tasks
- **LOW**: Background cleanup

### 5. Fault Tolerance

- Tasks survive worker crashes
- Automatic retry on failure (max 3 retries)
- Orphan detection finds documents without tasks
- Stale task release prevents stuck tasks

## Configuration

Add to `backend/.env`:

```bash
# Enable/disable task queue system
TASK_QUEUE_ENABLED=true

# Periodic maintenance interval (seconds)
TASK_QUEUE_CHECK_INTERVAL=300  # 5 minutes

# Worker poll interval (seconds)
WORKER_POLL_INTERVAL=5  # 5 seconds

# Heartbeat interval (seconds)
HEARTBEAT_INTERVAL=30  # 30 seconds

# Stale task timeout (seconds)
STALE_TASK_TIMEOUT=300  # 5 minutes

# Number of concurrent workers
NUM_WORKERS=4
```

## Usage

### Automatic Mode (Recommended)

1. Enable task queue in `.env`:
   ```bash
   TASK_QUEUE_ENABLED=true
   ```

2. Start the backend:
   ```bash
   cd backend
   python main.py
   ```

3. Upload a file - it will be processed automatically!

### Manual Enqueueing (Advanced)

```python
from queue_service import TaskQueueManager, TaskType, TaskPriority

manager = TaskQueueManager()

# Enqueue OCR task
task_id = manager.enqueue_task(
    document_id=doc.id,
    task_type=TaskType.OCR,
    priority=TaskPriority.HIGH
)

# Enqueue indexing task
task_id = manager.enqueue_task(
    document_id=doc.id,
    task_type=TaskType.INDEXING,
    priority=TaskPriority.NORMAL
)
```

## Monitoring

### Queue Statistics

```python
stats = manager.get_queue_stats()
# Returns: {
#     "pending": 5,
#     "claimed": 2,
#     "completed": 100,
#     "failed": 3,
#     "ocr_pending": 3,
#     "indexing_pending": 2
# }
```

### Check Logs

```bash
# Backend logs show queue activity
tail -f backend/logs/app.log | grep -E "✅|❌|🔄"
```

## Testing

Run the test script:

```bash
cd backend
python test_queue_system.py
```

## Migration from Old System

The queue system is backward compatible:

- Old manual trigger endpoints still work (when `TASK_QUEUE_ENABLED=false`)
- Existing documents are automatically detected and enqueued
- No data migration needed (migration 011 adds `task_queue` table)

## Troubleshooting

### Tasks stuck in CLAIMED status

- Check worker logs for errors
- Stale tasks are automatically released after 5 minutes
- Manually release: `UPDATE task_queue SET status='PENDING', worker_id=NULL WHERE status='CLAIMED'`

### Orphaned documents

- Scheduler automatically detects and enqueues orphaned documents every 5 minutes
- Check logs for "Found X orphaned documents"

### High database load

- Increase `WORKER_POLL_INTERVAL` (default: 5 seconds)
- Reduce `NUM_WORKERS` if too many concurrent tasks

## Performance

- **Task pickup latency**: 1-5 seconds (configurable via `WORKER_POLL_INTERVAL`)
- **Concurrent tasks**: Configurable via `NUM_WORKERS` (default: 4)
- **Database queries**: ~1 query per worker per poll interval
- **Heartbeat overhead**: 1 UPDATE per worker per 30 seconds

## Future Enhancements

- PostgreSQL LISTEN/NOTIFY for instant task pickup (<100ms)
- Task priority adjustment based on queue depth
- Worker auto-scaling based on queue size
- Metrics dashboard for queue monitoring

