-- Migration: Add reindex status columns to task_queue_document table
-- This migration adds the reindex phase columns to support background reindexing
--
-- The reindex phase allows documents to be queued for re-extraction and re-indexing
-- without blocking the UI. The actual work is done by background workers.

-- Add reindex phase columns to task_queue_document
DO $$
BEGIN
    -- reindex_status: The status of the reindex task (NULL = never requested)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'task_queue_document'
        AND column_name = 'reindex_status'
    ) THEN
        ALTER TABLE task_queue_document
        ADD COLUMN reindex_status task_status NULL DEFAULT NULL;
        RAISE NOTICE 'Added reindex_status column to task_queue_document';
    END IF;

    -- reindex_worker_id: ID of the worker processing this task
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'task_queue_document'
        AND column_name = 'reindex_worker_id'
    ) THEN
        ALTER TABLE task_queue_document
        ADD COLUMN reindex_worker_id VARCHAR(100) NULL;
        RAISE NOTICE 'Added reindex_worker_id column to task_queue_document';
    END IF;

    -- reindex_started_at: When the reindex task started processing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'task_queue_document'
        AND column_name = 'reindex_started_at'
    ) THEN
        ALTER TABLE task_queue_document
        ADD COLUMN reindex_started_at TIMESTAMP WITH TIME ZONE NULL;
        RAISE NOTICE 'Added reindex_started_at column to task_queue_document';
    END IF;

    -- reindex_completed_at: When the reindex task completed
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'task_queue_document'
        AND column_name = 'reindex_completed_at'
    ) THEN
        ALTER TABLE task_queue_document
        ADD COLUMN reindex_completed_at TIMESTAMP WITH TIME ZONE NULL;
        RAISE NOTICE 'Added reindex_completed_at column to task_queue_document';
    END IF;

    -- reindex_error: Error message if reindex failed
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'task_queue_document'
        AND column_name = 'reindex_error'
    ) THEN
        ALTER TABLE task_queue_document
        ADD COLUMN reindex_error TEXT NULL;
        RAISE NOTICE 'Added reindex_error column to task_queue_document';
    END IF;

    -- reindex_retry_count: Number of retry attempts
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'task_queue_document'
        AND column_name = 'reindex_retry_count'
    ) THEN
        ALTER TABLE task_queue_document
        ADD COLUMN reindex_retry_count INTEGER NOT NULL DEFAULT 0;
        RAISE NOTICE 'Added reindex_retry_count column to task_queue_document';
    END IF;

    -- reindex_last_heartbeat: Last heartbeat timestamp for detecting stale tasks
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'task_queue_document'
        AND column_name = 'reindex_last_heartbeat'
    ) THEN
        ALTER TABLE task_queue_document
        ADD COLUMN reindex_last_heartbeat TIMESTAMP WITH TIME ZONE NULL;
        RAISE NOTICE 'Added reindex_last_heartbeat column to task_queue_document';
    END IF;
END$$;

-- Create index for finding pending reindex tasks efficiently
CREATE INDEX IF NOT EXISTS idx_task_queue_document_reindex_pending
ON task_queue_document (reindex_status, reindex_retry_count)
WHERE reindex_status IN ('pending', 'failed');

COMMENT ON COLUMN task_queue_document.reindex_status IS 'Status of the reindex task: NULL (never requested), pending, processing, completed, failed';
COMMENT ON COLUMN task_queue_document.reindex_worker_id IS 'ID of the worker processing this reindex task';
COMMENT ON COLUMN task_queue_document.reindex_started_at IS 'When the reindex task started processing';
COMMENT ON COLUMN task_queue_document.reindex_completed_at IS 'When the reindex task completed';
COMMENT ON COLUMN task_queue_document.reindex_error IS 'Error message if the reindex task failed';
COMMENT ON COLUMN task_queue_document.reindex_retry_count IS 'Number of times the reindex task has been retried';
COMMENT ON COLUMN task_queue_document.reindex_last_heartbeat IS 'Last heartbeat timestamp for detecting stale tasks';
