-- Migration: Add document_metadata column to documents table
-- This column stores extracted document metadata from hierarchical summarization
-- Format: {
--   "extraction_version": "1.0",
--   "extracted_at": "2025-12-15T10:30:00Z",
--   "extraction_method": "hierarchical_summarization",
--   "document_type": "resume|manual|report|article|technical_doc|other",
--   "subject_name": "Felix Yang",
--   "subject_type": "person|organization|product|concept",
--   "title": "Felix Yang - Senior Consultant Resume",
--   "author": "Felix Yang",
--   "summary": "Senior consultant with 20+ years experience...",
--   "topics": ["software development", "architecture", "consulting"],
--   "key_entities": [{"name": "BDO", "type": "organization", "score": 95}],
--   "hierarchical_summary": {"level1_summaries": [...], "meta_summary": "..."},
--   "confidence": 0.85,
--   "processing_stats": {"total_chunks": 20, "llm_calls": 5, ...}
-- }

-- Add the new column
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS document_metadata JSONB DEFAULT NULL;

-- Add GIN index for efficient JSONB queries
CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin 
ON documents USING gin(document_metadata);

-- Add specific indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_documents_metadata_document_type 
ON documents ((document_metadata->>'document_type'));

CREATE INDEX IF NOT EXISTS idx_documents_metadata_subject_name 
ON documents ((document_metadata->>'subject_name'));

-- Comment on the column
COMMENT ON COLUMN documents.document_metadata IS 'Extracted document metadata: type, subject, topics, entities, hierarchical summary from LLM-based analysis';

