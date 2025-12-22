-- Migration: Hierarchical Task Queue System
-- Replaces the old single-level task_queue with a 3-level hierarchy:
--   documents (file level) → task_queue_page (page level) → task_queue_chunk (chunk level)
-- Each level has 3 status columns: ocr_status, vector_status, graphrag_status

-- ============================================================================
-- STEP 1: Create task_status enum
-- ============================================================================
DO $$ BEGIN
    CREATE TYPE task_status AS ENUM ('pending', 'processing', 'completed', 'failed');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- ============================================================================
-- STEP 2: Add new columns to documents table
-- ============================================================================

-- Add three status columns for each processing phase
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_status task_status DEFAULT 'pending';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS vector_status task_status DEFAULT 'pending';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS graphrag_status task_status DEFAULT 'pending';

-- Add timestamps and error tracking for each phase
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_started_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_completed_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_error TEXT;

ALTER TABLE documents ADD COLUMN IF NOT EXISTS vector_started_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS vector_completed_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS vector_error TEXT;

ALTER TABLE documents ADD COLUMN IF NOT EXISTS graphrag_started_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS graphrag_completed_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS graphrag_error TEXT;

-- Worker tracking at document level
ALTER TABLE documents ADD COLUMN IF NOT EXISTS current_worker_id VARCHAR(100);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS last_heartbeat TIMESTAMP WITH TIME ZONE;

-- ============================================================================
-- STEP 3: Create task_queue_page table
-- ============================================================================
CREATE TABLE IF NOT EXISTS task_queue_page (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Page identification
    page_number INTEGER NOT NULL,
    page_file_path VARCHAR(1000),  -- Path to _nohf.md file after OCR

    -- Phase 1: OCR (PDF page → Markdown)
    ocr_status task_status NOT NULL DEFAULT 'pending',
    ocr_worker_id VARCHAR(100),
    ocr_started_at TIMESTAMP WITH TIME ZONE,
    ocr_completed_at TIMESTAMP WITH TIME ZONE,
    ocr_error TEXT,
    ocr_retry_count INTEGER DEFAULT 0,
    ocr_last_heartbeat TIMESTAMP WITH TIME ZONE,

    -- Phase 2: Vector Index (Markdown → Qdrant)
    vector_status task_status NOT NULL DEFAULT 'pending',
    vector_worker_id VARCHAR(100),
    vector_started_at TIMESTAMP WITH TIME ZONE,
    vector_completed_at TIMESTAMP WITH TIME ZONE,
    vector_error TEXT,
    vector_retry_count INTEGER DEFAULT 0,
    vector_last_heartbeat TIMESTAMP WITH TIME ZONE,

    -- Phase 3: GraphRAG Index (Chunks → Neo4j)
    graphrag_status task_status NOT NULL DEFAULT 'pending',
    graphrag_worker_id VARCHAR(100),
    graphrag_started_at TIMESTAMP WITH TIME ZONE,
    graphrag_completed_at TIMESTAMP WITH TIME ZONE,
    graphrag_error TEXT,
    graphrag_retry_count INTEGER DEFAULT 0,
    graphrag_last_heartbeat TIMESTAMP WITH TIME ZONE,

    -- Common metadata
    chunk_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Ensure unique page per document
    UNIQUE(document_id, page_number)
);

-- Indexes for task_queue_page
CREATE INDEX IF NOT EXISTS idx_tqp_document ON task_queue_page(document_id);
CREATE INDEX IF NOT EXISTS idx_tqp_ocr_pending ON task_queue_page(ocr_status, ocr_retry_count, created_at)
    WHERE ocr_status IN ('pending', 'failed');
CREATE INDEX IF NOT EXISTS idx_tqp_vector_pending ON task_queue_page(vector_status, vector_retry_count, created_at)
    WHERE vector_status IN ('pending', 'failed');
CREATE INDEX IF NOT EXISTS idx_tqp_graphrag_pending ON task_queue_page(graphrag_status, graphrag_retry_count, created_at)
    WHERE graphrag_status IN ('pending', 'failed');
CREATE INDEX IF NOT EXISTS idx_tqp_ocr_heartbeat ON task_queue_page(ocr_last_heartbeat)
    WHERE ocr_status = 'processing';
CREATE INDEX IF NOT EXISTS idx_tqp_vector_heartbeat ON task_queue_page(vector_last_heartbeat)
    WHERE vector_status = 'processing';
CREATE INDEX IF NOT EXISTS idx_tqp_graphrag_heartbeat ON task_queue_page(graphrag_last_heartbeat)
    WHERE graphrag_status = 'processing';

-- ============================================================================
-- STEP 4: Create task_queue_chunk table
-- ============================================================================
CREATE TABLE IF NOT EXISTS task_queue_chunk (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    page_id UUID NOT NULL REFERENCES task_queue_page(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Chunk identification
    chunk_id VARCHAR(64) NOT NULL,     -- External chunk ID (used in Qdrant)
    chunk_index INTEGER NOT NULL,       -- Position within page (0-indexed)
    chunk_content_hash VARCHAR(64),     -- For deduplication

    -- Phase 2: Vector Index (Chunk → Qdrant embedding)
    vector_status task_status NOT NULL DEFAULT 'pending',
    vector_worker_id VARCHAR(100),
    vector_started_at TIMESTAMP WITH TIME ZONE,
    vector_completed_at TIMESTAMP WITH TIME ZONE,
    vector_error TEXT,
    vector_retry_count INTEGER DEFAULT 0,
    vector_last_heartbeat TIMESTAMP WITH TIME ZONE,

    -- Phase 3: GraphRAG Index (Chunk → Neo4j entities)
    graphrag_status task_status NOT NULL DEFAULT 'pending',
    graphrag_worker_id VARCHAR(100),
    graphrag_started_at TIMESTAMP WITH TIME ZONE,
    graphrag_completed_at TIMESTAMP WITH TIME ZONE,
    graphrag_error TEXT,
    graphrag_retry_count INTEGER DEFAULT 0,
    graphrag_last_heartbeat TIMESTAMP WITH TIME ZONE,

    -- Processing results
    entities_extracted INTEGER DEFAULT 0,
    relationships_extracted INTEGER DEFAULT 0,

    -- Common metadata
    max_retries INTEGER DEFAULT 3,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Ensure unique chunk per page
    UNIQUE(page_id, chunk_index)
);

-- Indexes for task_queue_chunk
CREATE INDEX IF NOT EXISTS idx_tqc_page ON task_queue_chunk(page_id);
CREATE INDEX IF NOT EXISTS idx_tqc_document ON task_queue_chunk(document_id);
CREATE INDEX IF NOT EXISTS idx_tqc_chunk_id ON task_queue_chunk(chunk_id);
CREATE INDEX IF NOT EXISTS idx_tqc_vector_pending ON task_queue_chunk(vector_status, vector_retry_count, created_at)
    WHERE vector_status IN ('pending', 'failed');
CREATE INDEX IF NOT EXISTS idx_tqc_graphrag_pending ON task_queue_chunk(graphrag_status, graphrag_retry_count, created_at)
    WHERE graphrag_status IN ('pending', 'failed');
CREATE INDEX IF NOT EXISTS idx_tqc_vector_heartbeat ON task_queue_chunk(vector_last_heartbeat)
    WHERE vector_status = 'processing';
CREATE INDEX IF NOT EXISTS idx_tqc_graphrag_heartbeat ON task_queue_chunk(graphrag_last_heartbeat)
    WHERE graphrag_status = 'processing';

-- ============================================================================
-- STEP 5: Migrate existing data from old system
-- ============================================================================

-- Migrate document statuses from convert_status/index_status to new columns
UPDATE documents SET
    ocr_status = CASE
        WHEN convert_status = 'converted' THEN 'completed'::task_status
        WHEN convert_status = 'converting' THEN 'processing'::task_status
        WHEN convert_status = 'failed' THEN 'failed'::task_status
        WHEN convert_status = 'partial' THEN 'failed'::task_status
        ELSE 'pending'::task_status
    END,
    vector_status = CASE
        WHEN index_status = 'indexed' THEN 'completed'::task_status
        WHEN index_status = 'indexing' THEN 'processing'::task_status
        WHEN index_status = 'failed' THEN 'failed'::task_status
        WHEN index_status = 'partial' THEN 'failed'::task_status
        ELSE 'pending'::task_status
    END,
    graphrag_status = CASE
        WHEN index_status = 'indexed' THEN 'completed'::task_status
        WHEN index_status = 'indexing' THEN 'processing'::task_status
        WHEN index_status = 'failed' THEN 'failed'::task_status
        WHEN index_status = 'partial' THEN 'failed'::task_status
        ELSE 'pending'::task_status
    END
WHERE ocr_status IS NULL OR vector_status IS NULL OR graphrag_status IS NULL;

-- ============================================================================
-- STEP 6: Add comments
-- ============================================================================
COMMENT ON TABLE task_queue_page IS 'Page-level task queue for OCR, Vector indexing, and GraphRAG indexing';
COMMENT ON TABLE task_queue_chunk IS 'Chunk-level task queue for Vector indexing and GraphRAG indexing';

COMMENT ON COLUMN documents.ocr_status IS 'OCR processing status (pending, processing, completed, failed)';
COMMENT ON COLUMN documents.vector_status IS 'Vector indexing status (pending, processing, completed, failed)';
COMMENT ON COLUMN documents.graphrag_status IS 'GraphRAG indexing status (pending, processing, completed, failed)';

COMMENT ON COLUMN task_queue_page.ocr_status IS 'OCR status for this page';
COMMENT ON COLUMN task_queue_page.vector_status IS 'Vector indexing status (only starts after ocr_status = completed)';
COMMENT ON COLUMN task_queue_page.graphrag_status IS 'GraphRAG status (only starts after vector_status = completed)';

COMMENT ON COLUMN task_queue_chunk.vector_status IS 'Vector indexing status for this chunk';
COMMENT ON COLUMN task_queue_chunk.graphrag_status IS 'GraphRAG status (only starts after vector_status = completed)';

-- ============================================================================
-- STEP 7: Drop old task_queue table (optional - can keep for backup)
-- ============================================================================
-- Uncomment to drop the old table after migration is verified:
-- DROP TABLE IF EXISTS task_queue;
