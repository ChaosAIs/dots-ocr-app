-- Migration: Add chunk content storage for V3.0 chunking optimization
-- Purpose: Store chunk content and metadata in task_queue_chunk table to avoid
--          redundant LLM calls during Vector and GraphRAG indexing stages.
--
-- Background:
--   Current flow calls chunk_markdown_with_summaries() 3 times per page:
--   1. Stage 1 (OCR): Creates chunk tasks
--   2. Stage 2 (Vector): Re-chunks to get content for indexing
--   3. Stage 3 (GraphRAG): Re-chunks again for entity extraction
--
--   With V3.0 LLM-driven chunking, this means 3 LLM calls per page instead of 1.
--   This migration adds columns to store chunk content after Stage 1,
--   so Stages 2 and 3 can read from the database instead of re-chunking.
--
-- Cleanup policy:
--   Queue records (task_queue_page, task_queue_chunk) are HARD DELETED
--   after a document is fully indexed. The document table retains all status info.

-- ============================================================================
-- STEP 1: Add chunk content columns to task_queue_chunk
-- ============================================================================

-- Actual chunk text content (stored after OCR/chunking, cleared after indexing)
ALTER TABLE task_queue_chunk
ADD COLUMN IF NOT EXISTS chunk_content TEXT;

-- Chunk metadata as JSON (strategy info, positions, etc.)
ALTER TABLE task_queue_chunk
ADD COLUMN IF NOT EXISTS chunk_metadata JSONB;

-- ============================================================================
-- STEP 2: Add 'skipped' status to task_status enum if not exists
-- ============================================================================

-- Check if 'skipped' already exists, add if not
DO $$ BEGIN
    ALTER TYPE task_status ADD VALUE IF NOT EXISTS 'skipped';
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- ============================================================================
-- STEP 3: Add graphrag_skip_reason column if not exists
-- ============================================================================

ALTER TABLE task_queue_chunk
ADD COLUMN IF NOT EXISTS graphrag_skip_reason VARCHAR(50);

-- ============================================================================
-- STEP 4: Add comments
-- ============================================================================

COMMENT ON COLUMN task_queue_chunk.chunk_content IS
    'Actual chunk text content. Stored after OCR/chunking in Stage 1. '
    'Used by Vector and GraphRAG indexing stages to avoid re-chunking. '
    'Cleared when queue record is deleted after successful indexing.';

COMMENT ON COLUMN task_queue_chunk.chunk_metadata IS
    'Chunk metadata as JSON including: source, file_path, chunk_index, '
    'total_chunks, start_position, end_position, strategy_used, etc. '
    'Stored alongside chunk_content for complete chunk reconstruction.';

-- ============================================================================
-- STEP 5: Create index for cleanup queries
-- ============================================================================

-- Index for finding completed chunks that can be cleaned up
CREATE INDEX IF NOT EXISTS idx_tqc_cleanup_ready ON task_queue_chunk(document_id, vector_status, graphrag_status)
    WHERE vector_status = 'completed' AND graphrag_status IN ('completed', 'skipped');
