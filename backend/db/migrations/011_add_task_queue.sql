-- Migration: Add task_queue table for OCR and indexing worker coordination
-- This table is ONLY for worker coordination, NOT for progress tracking
-- Progress is tracked in documents.ocr_details and documents.indexing_details

-- Create task_queue table
CREATE TABLE IF NOT EXISTS task_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    task_type VARCHAR(20) NOT NULL CHECK (task_type IN ('OCR', 'INDEXING')),
    priority VARCHAR(10) NOT NULL DEFAULT 'NORMAL' CHECK (priority IN ('HIGH', 'NORMAL', 'LOW')),
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'CLAIMED', 'COMPLETED', 'FAILED', 'CANCELLED')),
    
    -- Worker coordination (heartbeat system)
    worker_id VARCHAR(100),
    claimed_at TIMESTAMP WITH TIME ZONE,
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    
    -- Retry logic
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    error_message TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_task_queue_status ON task_queue(status);
CREATE INDEX IF NOT EXISTS idx_task_queue_document_id ON task_queue(document_id);
CREATE INDEX IF NOT EXISTS idx_task_queue_task_type ON task_queue(task_type);

-- Composite index for claiming tasks (priority + creation time)
CREATE INDEX IF NOT EXISTS idx_task_queue_claim ON task_queue(status, priority DESC, created_at ASC) 
WHERE status = 'PENDING';

-- Index for stale task detection
CREATE INDEX IF NOT EXISTS idx_task_queue_heartbeat ON task_queue(last_heartbeat) 
WHERE status = 'CLAIMED';

-- Prevent duplicate active tasks for same document+type
CREATE UNIQUE INDEX IF NOT EXISTS idx_task_queue_unique_active ON task_queue(document_id, task_type) 
WHERE status IN ('PENDING', 'CLAIMED');

-- Add table comment
COMMENT ON TABLE task_queue IS 'Task queue for OCR and indexing worker coordination. Progress tracking is in documents.ocr_details and documents.indexing_details.';

-- Add column comments
COMMENT ON COLUMN task_queue.task_type IS 'Type of task: OCR (document conversion) or INDEXING (vector + metadata + GraphRAG)';
COMMENT ON COLUMN task_queue.priority IS 'Task priority: HIGH (user uploads), NORMAL (auto-resume), LOW (background)';
COMMENT ON COLUMN task_queue.status IS 'Task status: PENDING (waiting), CLAIMED (being processed), COMPLETED, FAILED, CANCELLED';
COMMENT ON COLUMN task_queue.worker_id IS 'ID of worker processing this task (hostname:pid:thread_id)';
COMMENT ON COLUMN task_queue.last_heartbeat IS 'Last heartbeat from worker. Tasks with stale heartbeat (>5min) are released back to PENDING';
COMMENT ON COLUMN task_queue.retry_count IS 'Number of times this task has been retried after failure';

