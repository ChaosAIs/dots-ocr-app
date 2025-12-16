# OCR Status Tracking Implementation

## Overview

This document describes the implementation of granular OCR status tracking for the dots-ocr-app, similar to the existing indexing status tracking system. The implementation enables selective retry of failed OCR operations at both page and embedded image levels.

## Implementation Summary

### 1. Database Schema (Migration 006)

**File**: `backend/db/migrations/006_add_ocr_details.sql`

Added a new JSONB column `ocr_details` to the `documents` table with the following structure:

```json
{
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
            "image_size_pixels": 250000
          }
        }
      }
    }
  }
}
```

**Indexes Created**:

- GIN index on `ocr_details` for efficient JSONB queries
- Partial index on `ocr_details->'page_ocr'->>'status'` for status queries

### 2. Repository Methods

**File**: `backend/db/document_repository.py`

Added the following methods to `DocumentRepository`:

#### Initialization Methods

- `init_ocr_details(doc)` - Initialize OCR tracking structure
- `set_total_pages_for_ocr(doc, total_pages)` - Set total page count

#### Status Update Methods

- `update_page_ocr_status(doc, page_number, page_file_path, status, error, embedded_images_count)` - Track page-level OCR
- `update_embedded_image_ocr_status(doc, page_number, image_position, status, ocr_backend, error, image_size_pixels, skip_reason)` - Track embedded image OCR

#### Query Methods

- `get_failed_pages(doc)` - Get list of failed page numbers
- `get_pages_with_failed_embedded_images(doc)` - Get pages with failed embedded images
- `get_ocr_summary(doc)` - Get comprehensive OCR status summary

### 3. Parser Integration

**File**: `backend/dots_ocr_service/parser.py`

#### Changes Made:

1. **Added Context Tracking**:

   - `self.current_filename` - Track current document being processed
   - `self.current_page_number` - Track current page being processed

2. **Page-Level Tracking**:

   - Modified `parse_pdf()` to initialize OCR tracking
   - Updated task execution to track success/failure for each page
   - Added `_init_ocr_tracking()`, `_track_page_ocr_success()`, `_track_page_ocr_failure()` methods

3. **Embedded Image Tracking**:
   - Modified `_add_image_analysis_to_markdown()` to track each embedded image
   - Set context before processing embedded images
   - Track success, failure, and skipped states
   - Added `_track_embedded_image_ocr_success()`, `_track_embedded_image_ocr_failure()`, `_track_embedded_image_ocr_skipped()` methods

### 4. API Endpoints

**File**: `backend/main.py`

Added two new endpoints:

#### POST /retry-ocr

Retry OCR for failed pages or embedded images only.

**Parameters**:

- `filename` (required) - Document filename
- `retry_type` (optional) - "all", "pages", or "images" (default: "all")
- `page_numbers` (optional) - Comma-separated page numbers to retry

**Response**:

```json
{
  "status": "retry_planned",
  "message": "Retry plan created...",
  "retry_plan": {
    "filename": "document.pdf",
    "retry_type": "all",
    "ocr_summary": {...},
    "pages_to_retry": [5, 10],
    "images_to_retry": {0: [1, 2]}
  }
}
```

#### GET /ocr-status/{filename}

Get detailed OCR status for a document.

**Response**:

```json
{
  "filename": "document.pdf",
  "ocr_summary": {
    "status": "partial",
    "total_pages": 10,
    "converted_pages": 8,
    "failed_pages": 2,
    "failed_page_numbers": [5, 10],
    "pages_with_failed_images": {0: [1]}
  },
  "ocr_details": {...}
}
```

## Key Features

### 1. Multi-Level Tracking

- **Document Level**: Overall OCR status (completed, partial, failed, pending)
- **Page Level**: Individual page OCR status with retry counts
- **Embedded Image Level**: Each embedded image OCR status within pages

### 2. Selective Retry

- Retry only failed pages without re-processing successful ones
- Retry only failed embedded images within pages
- Support for specific page number selection

### 3. Comprehensive Error Tracking

- Error messages stored for debugging
- Retry counts to prevent infinite loops
- Timestamps for audit trail
- Image size tracking for analysis

### 4. Status States

- **success**: OCR completed successfully
- **failed**: OCR failed with error message
- **skipped**: Intentionally skipped (e.g., image too small)
- **pending**: Not yet processed

## Testing

A test script `backend/test_ocr_tracking.py` validates the OCR tracking logic without requiring database connection.

Run with:

```bash
cd backend && python test_ocr_tracking.py
```

## Migration Instructions

1. **Run Database Migration**:

   ```bash
   # Execute migration when database is running
   psql -h localhost -p 6400 -U postgres -d dots_ocr -f backend/db/migrations/006_add_ocr_details.sql
   ```

2. **Restart Backend**:
   The changes are backward compatible. Existing documents will have `ocr_details = NULL` and will work normally.

3. **New Documents**:
   All new OCR conversions will automatically track status at page and embedded image levels.

## Future Enhancements

1. **Actual Retry Implementation**: The `/retry-ocr` endpoint currently only analyzes what needs retry. Implement actual retry logic.
2. **Frontend Integration**: Add UI to display OCR status and trigger retries.
3. **Metrics Dashboard**: Show OCR success rates, common failure reasons, etc.
4. **Auto-Retry**: Automatically retry failed operations with exponential backoff.

## Files Modified

1. `backend/db/migrations/006_add_ocr_details.sql` (new)
2. `backend/db/document_repository.py` (added ~200 lines)
3. `backend/dots_ocr_service/parser.py` (added ~150 lines)
4. `backend/main.py` (added ~150 lines)
5. `backend/test_ocr_tracking.py` (new)

## Backward Compatibility

- ✅ Existing documents continue to work (ocr_details is nullable)
- ✅ No breaking changes to existing APIs
- ✅ Additive changes only
- ✅ Existing conversion flow unchanged

## Usage Examples

### Check OCR Status

```bash
curl http://localhost:8080/ocr-status/document.pdf
```

### Retry All Failed Operations

```bash
curl -X POST http://localhost:8080/retry-ocr \
  -F "filename=document.pdf" \
  -F "retry_type=all"
```

### Retry Only Failed Pages

```bash
curl -X POST http://localhost:8080/retry-ocr \
  -F "filename=document.pdf" \
  -F "retry_type=pages"
```

### Retry Specific Pages

```bash
curl -X POST http://localhost:8080/retry-ocr \
  -F "filename=document.pdf" \
  -F "retry_type=pages" \
  -F "page_numbers=5,10,15"
```

### Retry Only Failed Embedded Images

```bash
curl -X POST http://localhost:8080/retry-ocr \
  -F "filename=document.pdf" \
  -F "retry_type=images"
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     OCR Conversion Flow                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  parse_pdf()     │
                    │  - Initialize    │
                    │    OCR tracking  │
                    └────────┬─────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  For each page:              │
              │  _parse_single_image()       │
              └──────────┬───────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
    ┌─────────┐                   ┌──────────┐
    │ Success │                   │  Failed  │
    └────┬────┘                   └─────┬────┘
         │                               │
         ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│ Track page success  │         │ Track page failure  │
│ - update_page_ocr_  │         │ - update_page_ocr_  │
│   status(success)   │         │   status(failed)    │
└─────────┬───────────┘         └─────────────────────┘
          │
          ▼
┌──────────────────────────────┐
│ _add_image_analysis_to_      │
│ markdown()                    │
│ - Process embedded images    │
└──────────┬───────────────────┘
           │
           ▼
    ┌──────────────────────────────┐
    │  For each embedded image:    │
    └──────────┬───────────────────┘
               │
    ┌──────────┴──────────┬──────────────┐
    │                     │              │
    ▼                     ▼              ▼
┌─────────┐         ┌─────────┐    ┌─────────┐
│ Success │         │ Failed  │    │ Skipped │
└────┬────┘         └────┬────┘    └────┬────┘
     │                   │              │
     ▼                   ▼              ▼
┌─────────────┐    ┌─────────────┐  ┌─────────────┐
│ Track image │    │ Track image │  │ Track image │
│ success     │    │ failure     │  │ skipped     │
└─────────────┘    └─────────────┘  └─────────────┘
```
