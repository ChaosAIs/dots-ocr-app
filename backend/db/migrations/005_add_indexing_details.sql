-- Migration: Add indexing_details column for granular indexing status tracking
-- This enables selective re-indexing by tracking success/failure at chunk and page level
-- for vector indexing, metadata extraction, and GraphRAG indexing

-- Add indexing_details JSONB column to documents table
ALTER TABLE documents ADD COLUMN IF NOT EXISTS indexing_details JSONB DEFAULT NULL;

-- Create GIN index for efficient JSONB queries
CREATE INDEX IF NOT EXISTS idx_documents_indexing_details ON documents USING GIN (indexing_details);

-- Create partial indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_documents_vector_status 
ON documents ((indexing_details->'vector_indexing'->>'status')) 
WHERE indexing_details IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_documents_graphrag_status 
ON documents ((indexing_details->'graphrag_indexing'->>'status')) 
WHERE indexing_details IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_documents_metadata_status 
ON documents ((indexing_details->'metadata_extraction'->>'status')) 
WHERE indexing_details IS NOT NULL;

-- Add comment explaining the schema
COMMENT ON COLUMN documents.indexing_details IS 'Granular indexing status tracking for selective re-indexing. Schema: {
  "version": "1.0",
  "vector_indexing": {
    "status": "completed|partial|failed|pending",
    "total_chunks": 150,
    "indexed_chunks": 145,
    "failed_chunks": 5,
    "started_at": "2025-12-15T10:00:00Z",
    "completed_at": "2025-12-15T10:05:00Z",
    "pages": {
      "page_101": {
        "status": "failed",
        "file_path": "output/doc/doc_page_101_nohf.md",
        "error": "Embedding timeout",
        "failed_at": "2025-12-15T10:26:00Z",
        "retry_count": 2,
        "chunk_count": 15
      }
    },
    "chunks": {
      "chunk_uuid_1": {
        "status": "success|failed",
        "page": 101,
        "file_path": "output/doc/doc_page_101_nohf.md",
        "indexed_at": "2025-12-15T10:25:00Z",
        "error": null
      }
    }
  },
  "metadata_extraction": {
    "status": "completed|failed|pending|skipped",
    "extracted_at": "2025-12-15T10:06:00Z",
    "error": null,
    "retry_count": 0
  },
  "graphrag_indexing": {
    "status": "completed|partial|failed|pending|disabled",
    "total_chunks": 150,
    "processed_chunks": 140,
    "failed_chunks": 10,
    "skipped_chunks": 5,
    "entities_extracted": 234,
    "relationships_extracted": 567,
    "started_at": "2025-12-15T10:07:00Z",
    "completed_at": "2025-12-15T10:25:00Z",
    "chunks": {
      "chunk_uuid_1": {
        "status": "success|failed|skipped",
        "page": 101,
        "processed_at": "2025-12-15T10:15:00Z",
        "entities": 3,
        "relationships": 5,
        "error": null,
        "retry_count": 0
      }
    }
  }
}';

