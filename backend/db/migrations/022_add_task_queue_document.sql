-- Migration: 022_add_task_queue_document.sql
-- Description: Add document-level task queue table for classification and extraction tracking
-- Also drops deprecated line_items and line_items_storage columns from documents_data

-- ============================================================================
-- STEP 1: Create task_queue_document table
-- ============================================================================

CREATE TABLE IF NOT EXISTS task_queue_document (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Convert status (coordination across pages)
    convert_status task_status NOT NULL DEFAULT 'pending',
    convert_worker_id VARCHAR(100),
    convert_started_at TIMESTAMPTZ,
    convert_completed_at TIMESTAMPTZ,
    convert_error TEXT,
    convert_retry_count INT DEFAULT 0,
    convert_last_heartbeat TIMESTAMPTZ,

    -- Classification & Metadata status (NEW)
    classification_status task_status NOT NULL DEFAULT 'pending',
    classification_worker_id VARCHAR(100),
    classification_started_at TIMESTAMPTZ,
    classification_completed_at TIMESTAMPTZ,
    classification_error TEXT,
    classification_retry_count INT DEFAULT 0,
    classification_last_heartbeat TIMESTAMPTZ,

    -- Data Extraction status (for tabular documents)
    extraction_status task_status NOT NULL DEFAULT 'pending',
    extraction_worker_id VARCHAR(100),
    extraction_started_at TIMESTAMPTZ,
    extraction_completed_at TIMESTAMPTZ,
    extraction_error TEXT,
    extraction_retry_count INT DEFAULT 0,
    extraction_last_heartbeat TIMESTAMPTZ,

    -- Routing decision
    processing_path VARCHAR(20) DEFAULT 'standard',  -- 'standard' or 'tabular'

    -- Common fields
    max_retries INT DEFAULT 3,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(document_id)
);

-- Create indexes for worker polling
CREATE INDEX IF NOT EXISTS idx_tqd_convert_pending
    ON task_queue_document(convert_status, convert_retry_count, created_at)
    WHERE convert_status IN ('pending', 'failed');

CREATE INDEX IF NOT EXISTS idx_tqd_classification_pending
    ON task_queue_document(classification_status, classification_retry_count, created_at)
    WHERE classification_status IN ('pending', 'failed');

CREATE INDEX IF NOT EXISTS idx_tqd_extraction_pending
    ON task_queue_document(extraction_status, extraction_retry_count, created_at)
    WHERE extraction_status IN ('pending', 'failed');

-- Heartbeat monitoring indexes
CREATE INDEX IF NOT EXISTS idx_tqd_convert_heartbeat
    ON task_queue_document(convert_last_heartbeat)
    WHERE convert_status = 'processing';

CREATE INDEX IF NOT EXISTS idx_tqd_classification_heartbeat
    ON task_queue_document(classification_last_heartbeat)
    WHERE classification_status = 'processing';

CREATE INDEX IF NOT EXISTS idx_tqd_extraction_heartbeat
    ON task_queue_document(extraction_last_heartbeat)
    WHERE extraction_status = 'processing';

-- Document lookup index
CREATE INDEX IF NOT EXISTS idx_tqd_document_id ON task_queue_document(document_id);

-- ============================================================================
-- STEP 2: Migrate inline line_items to external storage (if any exist)
-- ============================================================================

-- First, insert any inline line_items into the external table
INSERT INTO documents_data_line_items (documents_data_id, line_number, data)
SELECT
    dd.id as documents_data_id,
    (row_number() OVER (PARTITION BY dd.id ORDER BY ordinality)) - 1 as line_number,
    item as data
FROM documents_data dd,
     jsonb_array_elements(dd.line_items) WITH ORDINALITY as t(item, ordinality)
WHERE dd.line_items_storage = 'inline'
  AND dd.line_items IS NOT NULL
  AND jsonb_array_length(dd.line_items) > 0
  AND NOT EXISTS (
      -- Skip if already migrated (has entries in external table)
      SELECT 1 FROM documents_data_line_items li WHERE li.documents_data_id = dd.id
  );

-- ============================================================================
-- STEP 3: Drop deprecated columns from documents_data
-- ============================================================================

-- Drop line_items column (no longer used - all data in external table)
ALTER TABLE documents_data DROP COLUMN IF EXISTS line_items;

-- Drop line_items_storage column (no longer needed - always external)
ALTER TABLE documents_data DROP COLUMN IF EXISTS line_items_storage;

-- ============================================================================
-- STEP 4: Add comment for documentation
-- ============================================================================

COMMENT ON TABLE task_queue_document IS 'Document-level task queue for classification, extraction, and processing path routing';
COMMENT ON COLUMN task_queue_document.processing_path IS 'Processing path: standard (chunking) or tabular (data extraction)';
COMMENT ON COLUMN task_queue_document.classification_status IS 'Status of document classification and metadata extraction';
COMMENT ON COLUMN task_queue_document.extraction_status IS 'Status of tabular data extraction (only for tabular path)';
