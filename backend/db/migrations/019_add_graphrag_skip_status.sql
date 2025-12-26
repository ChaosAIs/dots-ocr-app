-- Migration: 019_add_graphrag_skip_status.sql
-- Description: Add SKIPPED status to task_status enum and skip tracking columns
-- Date: 2025-12-26

-- ============================================================================
-- Step 1: Add 'skipped' value to task_status enum
-- ============================================================================

-- Check if 'skipped' already exists in the enum before adding
DO $$
BEGIN
    -- Add 'skipped' to task_status enum if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM pg_enum
        WHERE enumlabel = 'skipped'
        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'task_status')
    ) THEN
        ALTER TYPE task_status ADD VALUE 'skipped' AFTER 'failed';
    END IF;
END $$;


-- ============================================================================
-- Step 2: Add skip tracking columns to documents table
-- ============================================================================

-- Add skip_graphrag boolean flag
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS skip_graphrag BOOLEAN DEFAULT FALSE;

-- Add skip_graphrag_reason for tracking why it was skipped
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS skip_graphrag_reason VARCHAR(100) DEFAULT NULL;

-- Add comments for documentation
COMMENT ON COLUMN documents.skip_graphrag IS 'Whether to skip GraphRAG indexing for this document (tabular/structured data)';
COMMENT ON COLUMN documents.skip_graphrag_reason IS 'Reason for skipping GraphRAG: file_type:xxx, document_type:xxx, user_disabled, etc.';


-- ============================================================================
-- Step 3: Add skip reason column to task_queue_chunk table
-- ============================================================================

-- Add graphrag_skip_reason to track per-chunk skip reasons
ALTER TABLE task_queue_chunk
ADD COLUMN IF NOT EXISTS graphrag_skip_reason VARCHAR(50) DEFAULT NULL;

COMMENT ON COLUMN task_queue_chunk.graphrag_skip_reason IS 'Reason this chunk was skipped for GraphRAG indexing';


-- ============================================================================
-- Step 4: Create indexes for efficient querying
-- ============================================================================

-- Index for querying documents that skip GraphRAG
CREATE INDEX IF NOT EXISTS idx_documents_skip_graphrag
ON documents (skip_graphrag) WHERE skip_graphrag = TRUE;

-- Index for querying skipped chunks
CREATE INDEX IF NOT EXISTS idx_task_queue_chunk_graphrag_skipped
ON task_queue_chunk (graphrag_status) WHERE graphrag_status = 'skipped';


-- ============================================================================
-- Step 5: Update existing spreadsheet/CSV documents to skip GraphRAG
-- ============================================================================

-- Set skip_graphrag=TRUE for existing documents with spreadsheet extensions
UPDATE documents
SET
    skip_graphrag = TRUE,
    skip_graphrag_reason = 'file_type:' || LOWER(SUBSTRING(filename FROM '\.([^.]+)$'))
WHERE
    skip_graphrag IS NOT TRUE
    AND LOWER(SUBSTRING(filename FROM '\.([^.]+)$')) IN (
        'xlsx', 'xls', 'xlsm', 'xlsb',  -- Excel
        'csv', 'tsv',                    -- Delimited
        'json', 'xml', 'yaml', 'yml',   -- Structured
        'log', 'sql'                     -- Technical
    );

-- Log how many documents were updated
DO $$
DECLARE
    updated_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO updated_count
    FROM documents
    WHERE skip_graphrag = TRUE;

    RAISE NOTICE 'Updated % documents to skip GraphRAG indexing', updated_count;
END $$;
