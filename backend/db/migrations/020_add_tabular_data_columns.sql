-- Migration: Add Tabular Data Processing Columns
-- Version: 020
-- Description: Add columns to support optimized tabular data workflow
--              (CSV, Excel, invoices, receipts, bank statements, etc.)
--
-- This migration adds:
-- 1. is_tabular_data: Flag to identify tabular/dataset-style documents
-- 2. processing_path: Indicator for which processing path to use
-- 3. summary_chunk_ids: Track the 1-3 summary chunks for tabular documents
--
-- These columns enable the optimized tabular data workflow that:
-- - Skips row-level chunking (no embedding of individual data rows)
-- - Preserves document discovery through summary/metadata indexing
-- - Routes data queries through SQL-based analytics after document discovery

-- Add is_tabular_data column
-- Flag to identify tabular/dataset-style documents (CSV, Excel, invoices, receipts, etc.)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'documents' AND column_name = 'is_tabular_data'
    ) THEN
        ALTER TABLE documents ADD COLUMN is_tabular_data BOOLEAN NOT NULL DEFAULT FALSE;
        COMMENT ON COLUMN documents.is_tabular_data IS 'Flag to identify tabular/dataset-style documents (CSV, Excel, invoices, receipts, bank statements, etc.)';
    END IF;
END $$;

-- Add processing_path column
-- Indicates which processing path to use: "standard" (full chunking), "tabular" (summary-only), "hybrid"
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'documents' AND column_name = 'processing_path'
    ) THEN
        ALTER TABLE documents ADD COLUMN processing_path VARCHAR(50) NOT NULL DEFAULT 'standard';
        COMMENT ON COLUMN documents.processing_path IS 'Processing path: standard (full chunking), tabular (summary-only), hybrid';
    END IF;
END $$;

-- Add summary_chunk_ids column
-- Track the 1-3 summary chunk IDs for tabular documents (instead of many row chunks)
-- Example: ["doc-uuid_summary", "doc-uuid_schema", "doc-uuid_context"]
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'documents' AND column_name = 'summary_chunk_ids'
    ) THEN
        ALTER TABLE documents ADD COLUMN summary_chunk_ids TEXT[];
        COMMENT ON COLUMN documents.summary_chunk_ids IS 'Array of summary chunk IDs for tabular documents (1-3 chunks: summary, schema, context)';
    END IF;
END $$;

-- Create index for efficient tabular document queries
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'idx_documents_is_tabular_data'
    ) THEN
        CREATE INDEX idx_documents_is_tabular_data ON documents(is_tabular_data) WHERE is_tabular_data = TRUE;
    END IF;
END $$;

-- Create index for processing_path queries
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'idx_documents_processing_path'
    ) THEN
        CREATE INDEX idx_documents_processing_path ON documents(processing_path);
    END IF;
END $$;

-- Update existing spreadsheet documents to be marked as tabular
-- This identifies documents that should use the optimized pathway
UPDATE documents
SET is_tabular_data = TRUE,
    processing_path = 'tabular'
WHERE (
    -- By file extension
    LOWER(filename) LIKE '%.csv'
    OR LOWER(filename) LIKE '%.tsv'
    OR LOWER(filename) LIKE '%.xlsx'
    OR LOWER(filename) LIKE '%.xls'
    OR LOWER(filename) LIKE '%.xlsm'
    OR LOWER(filename) LIKE '%.xlsb'
    OR LOWER(filename) LIKE '%.ods'
)
AND is_tabular_data = FALSE;

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE 'Migration 020_add_tabular_data_columns completed successfully';
    RAISE NOTICE 'Added columns: is_tabular_data, processing_path, summary_chunk_ids';
    RAISE NOTICE 'Created indexes: idx_documents_is_tabular_data, idx_documents_processing_path';
END $$;
