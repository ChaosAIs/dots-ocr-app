# Auto Status Update Fix for Background Indexing

## Problem

When the backend server starts up, it automatically resumes incomplete indexing tasks in the background (via `AUTO_RESUME_INDEXING` feature). However, the frontend UI does not automatically update to reflect the status changes when indexing completes because:

1. **No WebSocket Connection**: The auto-resume happens during server startup in a background thread, before the frontend has loaded and established WebSocket connections.
2. **No Polling Mechanism**: The frontend only loads documents on:
   - Component mount
   - Manual refresh
   - After specific user actions (convert, index, delete)
3. **Database Updated, UI Not Notified**: The backend updates the database status (INDEXING → INDEXED), but the frontend has no way to know about this change.

## Solution

Added a **polling mechanism** that automatically refreshes the document list when documents are in "INDEXING" or "PENDING" status.

### Implementation

**File**: `frontend/src/components/documents/documentList.jsx`

Added a new `useEffect` hook that:
1. Checks if any documents have `index_status` of "INDEXING" or "PENDING"
2. If yes, polls the backend every 3 seconds to refresh the document list
3. Automatically stops polling when all documents are no longer in indexing state
4. Prevents polling when the component is already loading to avoid race conditions

```javascript
// Poll for document status updates when documents are being indexed in background
// This handles cases where indexing happens during server startup (auto-resume)
useEffect(() => {
  let intervalId = null;

  // Check if any documents are currently being indexed (status = "INDEXING")
  const hasIndexingDocuments = documents.some(
    (doc) => doc.index_status === "INDEXING" || doc.index_status === "PENDING"
  );

  if (hasIndexingDocuments && !loading) {
    // Poll every 3 seconds to check for status updates
    intervalId = setInterval(() => {
      loadDocuments();
    }, 3000);
  }

  return () => {
    if (intervalId) {
      clearInterval(intervalId);
    }
  };
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [documents, loading]);
```

## Benefits

1. **Automatic UI Updates**: The UI now automatically updates when background indexing completes
2. **No User Intervention Required**: Users don't need to manually refresh the page
3. **Efficient Polling**: Only polls when necessary (when documents are being indexed)
4. **Consistent with Existing Pattern**: Uses the same polling pattern as batch indexing (lines 36-48)
5. **Resource Efficient**: Automatically stops polling when no documents are being indexed

## Testing

1. **Scenario 1: Server Restart with Incomplete Indexing**
   - Upload a document and start indexing
   - Stop the backend server before indexing completes
   - Restart the backend server (auto-resume will trigger)
   - Open the frontend
   - **Expected**: Status should automatically update from "INDEXING" to "INDEXED" without manual refresh

2. **Scenario 2: Manual Indexing**
   - Upload a document
   - Click "Index" button
   - **Expected**: Status updates via WebSocket (existing behavior, unchanged)

3. **Scenario 3: No Indexing in Progress**
   - View documents that are already indexed
   - **Expected**: No polling occurs (efficient)

## Related Files

- **Frontend**: `frontend/src/components/documents/documentList.jsx`
- **Backend**: `backend/main.py` (`_resume_incomplete_indexing` function)
- **Backend**: `backend/rag_service/indexer.py` (`trigger_embedding_for_document` function)

## Notes

- The polling interval is set to 3 seconds, which is slightly longer than the batch indexing poll (2 seconds) to reduce server load
- The `loadDocuments` function is not included in the dependency array to avoid infinite loops, following the existing pattern in the codebase
- The eslint warning is suppressed with `// eslint-disable-next-line react-hooks/exhaustive-deps`

