-- Migration: Add ocr_details column for granular OCR status tracking
-- This enables selective retry of failed pages and embedded images

-- Add ocr_details column to documents table
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_details JSONB DEFAULT NULL;

-- Create GIN index for efficient JSONB queries
CREATE INDEX IF NOT EXISTS idx_documents_ocr_details ON documents USING GIN (ocr_details);

-- Create partial indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_documents_ocr_status 
ON documents ((ocr_details->'page_ocr'->>'status')) 
WHERE ocr_details IS NOT NULL;

-- Add column comment with schema documentation
COMMENT ON COLUMN documents.ocr_details IS 'Granular OCR status tracking for selective re-processing. Schema: {
  "version": "1.0",
  "page_ocr": {
    "status": "completed|partial|failed|pending",
    "total_pages": 150,
    "converted_pages": 145,
    "failed_pages": 5,
    "started_at": "2025-12-15T10:00:00Z",
    "completed_at": "2025-12-15T10:05:00Z",
    "pages": {
      "page_0": {
        "status": "success|failed",
        "page_number": 0,
        "file_path": "output/doc/doc_page_0_nohf.md",
        "converted_at": "2025-12-15T10:01:00Z",
        "error": null,
        "retry_count": 0,
        "embedded_images_count": 3,
        "embedded_images": {
          "image_0": {
            "status": "success|failed|skipped",
            "image_position": 0,
            "ocr_backend": "gemma3|qwen3",
            "converted_at": "2025-12-15T10:01:15Z",
            "error": null,
            "retry_count": 0,
            "image_size_pixels": 250000,
            "was_skipped": false,
            "skip_reason": null
          },
          "image_1": {
            "status": "failed",
            "image_position": 1,
            "ocr_backend": "gemma3",
            "failed_at": "2025-12-15T10:01:30Z",
            "error": "Timeout calling Gemma3 API",
            "retry_count": 2,
            "image_size_pixels": 180000
          }
        }
      },
      "page_101": {
        "status": "failed",
        "page_number": 101,
        "file_path": "output/doc/doc_page_101_nohf.md",
        "error": "OCR inference timeout",
        "failed_at": "2025-12-15T10:26:00Z",
        "retry_count": 2,
        "embedded_images_count": 0
      }
    }
  }
}';

