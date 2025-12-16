# Unified Index Button Implementation

## Overview

This document describes the implementation of a simplified, unified "Index" button that combines the previous separate "Convert" and "Index" buttons into a single intelligent button with state-aware behavior.

## Goals

1. **Simplify UI**: Reduce button clutter by combining convert + index operations
2. **Intelligent State Management**: Button automatically determines what needs to be done
3. **Granular Status Tracking**: Track OCR, vector indexing, metadata extraction, and GraphRAG separately
4. **Smart Retry**: Only retry failed operations, not successful ones

## Button States

### 1. **"Index" (Primary, Active)**
- **Condition**: First-time upload, no conversion or indexing done
- **Icon**: `pi-database`
- **Color**: Primary (blue)
- **Action**: Trigger convert → vector index → metadata extraction (GraphRAG runs in background)
- **Tooltip**: "Index for RAG Search"

### 2. **"Indexing..." (Loading)**
- **Condition**: Any operation in progress (converting, indexing, processing)
- **Icon**: `pi-spin pi-spinner`
- **Color**: Primary (blue)
- **Action**: Disabled, shows progress
- **Tooltip**: "Indexing"

### 3. **"Retry Index" (Warning, Active)**
- **Condition**: Any phase failed (OCR, vector, metadata, or GraphRAG)
- **Icon**: `pi-refresh`
- **Color**: Warning (orange)
- **Action**: Retry only failed operations
- **Tooltip**: "Retry Indexing"

### 4. **Disabled (Success)**
- **Condition**: All phases complete (OCR + Vector + Metadata + GraphRAG)
- **Icon**: `pi-check`
- **Color**: Success (green)
- **Action**: None
- **Tooltip**: "All indexing complete"

### 5. **Disabled (Partial Success)**
- **Condition**: OCR + Vector + Metadata complete, GraphRAG still processing
- **Icon**: `pi-clock`
- **Color**: Secondary (gray)
- **Action**: None (GraphRAG running in background)
- **Tooltip**: "GraphRAG indexing pending"

## Status Badge Updates

The Status column now shows more granular states:

- **Pending**: No conversion done yet
- **Conversion Failed**: OCR conversion failed
- **Partial (X/Y)**: Partial conversion (some pages failed)
- **Converted**: OCR complete, not indexed
- **Indexing**: Currently indexing (vector or metadata)
- **Indexing GraphRAG**: Vector + Metadata done, GraphRAG processing
- **Partially Indexed**: Vector + Metadata done, GraphRAG pending
- **Indexing Failed**: Any indexing phase failed
- **Fully Indexed**: All phases complete

## Implementation Details

### Frontend Changes

#### 1. **documentList.jsx** (~400 lines modified)

**New Functions:**
- `handleUnifiedIndex(document)` - Main handler for unified button
- `handleConvertForUnifiedIndex(document)` - Helper for conversion step
- `getUnifiedIndexButtonState(rowData)` - Determines button state
- Updated `statusBodyTemplate(rowData)` - Enhanced status logic

**State Machine Logic:**
```javascript
// Check what needs to be done
const needsConversion = !markdown_exists || convert_status === "failed" || convert_status === "partial";
const needsVectorIndex = !indexing_details || indexing_details?.vector_indexing?.status === "failed";
const needsMetadata = !indexing_details || indexing_details?.metadata_extraction?.status === "failed";
const needsGraphRAG = !indexing_details || indexing_details?.graphrag_indexing?.status === "failed";

// Execute only what's needed
if (needsConversion) await handleConvertForUnifiedIndex(document);
if (needsVectorIndex || needsMetadata || needsGraphRAG) await indexDocument(document);
```

#### 2. **Translation Files**

**en.json** - Added keys:
- `RetryIndex`: "Retry Indexing"
- `AllIndexingComplete`: "All indexing complete"
- `GraphRAGPending`: "GraphRAG indexing pending"
- `ConversionFailed`: "Conversion Failed"
- `IndexingFailed`: "Indexing Failed"
- `FullyIndexed`: "Fully Indexed"
- `IndexingGraphRAG`: "Indexing GraphRAG"
- `PartiallyIndexed`: "Partially Indexed"

**fr.json** - Added French translations for all new keys

#### 3. **documentList.scss**

Added new status badge styles:
- `.failed` - Red background for failures
- `.indexing` - Blue background for in-progress
- `.partial-indexed` - Gray background for partial completion

### Backend Changes

#### **main.py** - `/documents` endpoint

Added two new fields to document list response:
```python
"indexing_details": db_info.get("indexing_details"),  # Granular indexing status
"ocr_details": db_info.get("ocr_details"),  # Granular OCR status
```

These fields contain the full JSONB structures for status tracking.

## User Experience Flow

### First-Time Upload
1. User uploads document → Status: "Pending"
2. User clicks "Index" button (primary blue)
3. System converts to markdown → Status: "Indexing"
4. System indexes (vector + metadata) → Status: "Indexing GraphRAG"
5. GraphRAG processes in background → Status: "Partially Indexed"
6. GraphRAG completes → Status: "Fully Indexed", Button: Disabled (green check)

### Retry Failed Operation
1. Document has failed conversion → Status: "Conversion Failed"
2. Button shows "Retry Index" (orange warning)
3. User clicks button
4. System retries only failed operations
5. Updates status based on results

### View Existing Document
1. Document fully indexed → Status: "Fully Indexed"
2. Button disabled with green checkmark
3. User can view markdown or check status logs

## Benefits

1. **Simplified UI**: One button instead of two
2. **Intelligent**: Automatically knows what to do
3. **Efficient**: Only retries failed operations
4. **Transparent**: Clear status indicators
5. **Granular**: Tracks each phase separately
6. **Resilient**: Can recover from partial failures

## Files Modified

1. ✅ `frontend/src/components/documents/documentList.jsx` (~400 lines)
2. ✅ `frontend/public/assets/i18n/translations/en.json` (+8 keys)
3. ✅ `frontend/public/assets/i18n/translations/fr.json` (+8 keys)
4. ✅ `frontend/src/components/documents/documentList.scss` (+15 lines)
5. ✅ `backend/main.py` (+2 fields in /documents endpoint)

## Testing Checklist

- [ ] First-time upload → Index button works
- [ ] Conversion failure → Retry button appears
- [ ] Partial conversion → Retry button works
- [ ] Vector indexing failure → Retry button works
- [ ] Metadata extraction failure → Retry button works
- [ ] GraphRAG failure → Retry button works
- [ ] All complete → Button disabled with checkmark
- [ ] Status badges show correct states
- [ ] Translations work in English and French

## Future Enhancements

1. Add progress percentage for each phase
2. Show detailed phase breakdown in tooltip
3. Add "View Details" button to see granular status
4. Add auto-retry with exponential backoff
5. Add notification when GraphRAG completes

