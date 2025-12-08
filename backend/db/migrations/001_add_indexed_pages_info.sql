-- Migration: Add indexed_pages_info column to documents table
-- This column tracks page-level indexing status for partial indexing support
-- Format: {"indexed": [1,2,3], "pending": [4,5], "failed": []}

-- Add the new column
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS indexed_pages_info JSONB DEFAULT '{"indexed": [], "pending": [], "failed": []}'::jsonb;

-- Add 'partial' value to index_status enum if not exists
DO $$ 
BEGIN
    -- Check if 'partial' already exists in the enum
    IF NOT EXISTS (
        SELECT 1 FROM pg_enum 
        WHERE enumlabel = 'partial' 
        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'index_status')
    ) THEN
        ALTER TYPE index_status ADD VALUE 'partial';
    END IF;
END $$;

-- Add index for faster queries on indexed_pages_info
CREATE INDEX IF NOT EXISTS idx_documents_indexed_pages_info 
ON documents USING GIN (indexed_pages_info);

-- Comment on the column
COMMENT ON COLUMN documents.indexed_pages_info IS 'Tracks page-level indexing status: indexed (completed), pending (queued), failed (errors)';

