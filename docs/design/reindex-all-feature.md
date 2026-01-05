# Reindex All Feature - Design Document

## Document Information

| Field | Value |
|-------|-------|
| **Feature Name** | Reindex All |
| **Author** | Development Team |
| **Created Date** | 2026-01-04 |
| **Last Updated** | 2026-01-04 |
| **Status** | Draft |
| **Version** | 1.3 |

### Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-04 | Initial design document |
| 1.1 | 2026-01-04 | Added Role-Based Access Control (RBAC) - Admin can reindex all documents, regular users can only reindex documents they own |
| 1.2 | 2026-01-04 | Added Section 11: Existing Permission System Analysis - Documents the current authentication flow, user roles, document ownership model, permission types, UserDocument bridge table, PermissionService, admin bypass, and permission patterns in existing endpoints |
| 1.3 | 2026-01-04 | **MAJOR: Queue-Based Architecture** - Changed from synchronous processing to fully asynchronous queue-based reindexing. The endpoint now creates reindex tasks that are processed by background workers. Added reindex_status to TaskQueueDocument model. |

---

## 1. Overview

### 1.1 Purpose

The "Reindex All" feature allows users to re-run **data extraction** and **indexing** processes for fully indexed documents without repeating the OCR conversion step. This is essential when:

- Extraction schemas are modified (new fields added, field mappings changed)
- LLM prompts are updated for better extraction accuracy
- Indexing logic is enhanced (chunking strategy, embedding model changes)
- GraphRAG entity extraction rules are updated

### 1.2 Scope

**In Scope:**
- Re-extract data to `documents_data` and `documents_data_line_items` tables
- Re-index to Qdrant vector database
- Re-extract entities to Neo4j graph database
- Re-extract document metadata
- Support for single workspace or all workspaces

**Out of Scope:**
- Re-running OCR conversion (uses existing markdown files)
- Re-chunking documents (uses stored chunk content from V3.0 optimization)
- Modifying original uploaded files

### 1.3 User Stories

1. **As an administrator**, I want to reindex all documents across all workspaces when I update the extraction logic, so that all documents reflect the new extraction schema.

2. **As an administrator**, I want to reindex all documents in a specific workspace regardless of ownership, so that I can ensure consistency after schema changes.

3. **As a regular user**, I want to reindex all documents that I own across all my workspaces, so that my documents are updated with the latest extraction logic.

4. **As a regular user**, I want to reindex my documents in a specific workspace, so that I can refresh the extracted data for my files without affecting other users' documents.

---

## 2. User Interface Design

### 2.1 UI Placement

#### 2.1.1 Global Reindex Button (All Workspaces)

**Location:** Workspace sidebar header, beside the "+" (Create Workspace) button

**File:** `frontend/src/components/workspace/WorkspaceSidebar.jsx`

```
┌─────────────────────────────────┐
│ Workspaces           [🔄] [+]  │  ← Reindex button added here
├─────────────────────────────────┤
│ 📁 My Documents ⭐              │
│    0 documents                  │
│ 📁 Receipts        [✏️][⭐][🗑️] │
│    0 documents                  │
│ 📁 Bank statement  [✏️][⭐][🗑️] │
│    0 documents                  │
│ 📁 Invoice         [✏️][⭐][🗑️] │
│    6 documents                  │
└─────────────────────────────────┘
```

**Button Specifications:**
- Icon: `pi pi-sync`
- Style: Text button, rounded
- Size: 1.75rem × 1.75rem
- Tooltip: "Reindex All Workspaces"

#### 2.1.2 Workspace-Specific Reindex Button

**Location:** Document list header, beside the "Upload" button

**File:** `frontend/src/components/documents/documentList.jsx`

```
┌──────────────────────────────────────────────────────────────────────┐
│ 📁 Invoice                                                           │
│ Workspace Documents                          [🔄 Reindex] [📤 Upload] [↻] │
├──────────────────────────────────────────────────────────────────────┤
│ Filename                    │ Size    │ Upload Time      │ Status    │
├─────────────────────────────┼─────────┼──────────────────┼───────────┤
│ Dec-03 - Keyboard Mouse...  │ 9.98 KB │ 1/4/2026, 2:41PM │ ✅ Indexed│
│ Dec-06 - Wired Keyboard...  │ 13.29KB │ 1/4/2026, 2:41PM │ ✅ Indexed│
└──────────────────────────────────────────────────────────────────────┘
```

**Button Specifications:**
- Icon: `pi pi-sync`
- Label: "Reindex"
- Style: Outlined, secondary severity
- Disabled when: No workspace selected
- Tooltip: "Re-extract and reindex all documents in this workspace"

### 2.2 Confirmation Dialog

Before triggering reindex, a confirmation dialog is displayed:

```
┌─────────────────────────────────────────────────────────────┐
│ ⚠️ Confirm Reindex                                     [X] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ This will re-run data extraction and indexing for          │
│ 6 fully indexed documents.                                 │
│                                                             │
│ • Existing extracted data will be replaced                 │
│ • Vector embeddings will be regenerated                    │
│ • Graph relationships will be re-extracted                 │
│ • This may take several minutes                            │
│                                                             │
│ OCR conversion will NOT be re-run.                         │
│                                                             │
│                              [Cancel]  [Confirm Reindex]   │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Progress Indication

After reindex starts, documents transition through status states visible in the UI:

| Status | Tag Color | Description |
|--------|-----------|-------------|
| Fully Indexed | Green | Before reindex starts |
| Indexing | Blue | During reindex process |
| Indexing Metadata | Blue | Metadata extraction phase |
| Indexing GraphRAG | Blue | GraphRAG extraction phase |
| Fully Indexed | Green | After reindex completes |

---

## 3. API Design

### 3.1 Endpoint Specification

#### POST /documents/reindex

**Description:** Initiates reindex process for fully indexed documents.

**Authentication:** Required (Bearer token)

**Request:**

```http
POST /documents/reindex
Content-Type: application/json
Authorization: Bearer <token>

{
  "workspace_id": "uuid-string-or-null"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `workspace_id` | string (UUID) \| null | No | Workspace to reindex. If null, reindexes all workspaces accessible to user. |

**Response (Success - 200):**

```json
{
  "status": "success",
  "message": "Reindex started for 6 documents",
  "queued_documents": 6,
  "skipped_documents": 2,
  "workspace_id": "550e8400-e29b-41d4-a716-446655440000",
  "details": {
    "queued": [
      {"id": "doc-uuid-1", "filename": "invoice-001.pdf"},
      {"id": "doc-uuid-2", "filename": "invoice-002.pdf"}
    ],
    "skipped": [
      {"id": "doc-uuid-3", "filename": "draft.pdf", "reason": "not_fully_indexed"}
    ]
  }
}
```

**Response (No Documents - 200):**

```json
{
  "status": "success",
  "message": "No fully indexed documents to reindex",
  "queued_documents": 0,
  "skipped_documents": 0,
  "workspace_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response (Error - 4xx/5xx):**

```json
{
  "detail": "Error message describing the failure"
}
```

| Status Code | Condition |
|-------------|-----------|
| 200 | Success (even if 0 documents queued) |
| 400 | Invalid workspace_id format |
| 401 | Not authenticated |
| 403 | No permission to access workspace |
| 404 | Workspace not found |
| 500 | Internal server error |

### 3.2 Document Eligibility Criteria

A document is eligible for reindexing if:

```python
eligible = (
    document.convert_status == "converted" AND
    document.index_status == "indexed" AND
    document.deleted_at IS NULL AND
    (
        workspace_id IS NULL OR
        document.workspace_id == workspace_id
    ) AND
    user_has_permission(current_user, document, "write")
)
```

---

## 4. Backend Architecture

### 4.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         API Layer (main.py)                         │
│                    POST /documents/reindex                          │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Reindex Coordinator                            │
│  • Validates permissions                                            │
│  • Queries eligible documents                                       │
│  • Orchestrates cleanup and reset                                   │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐
│  Data Cleanup   │   │ Embedding       │   │ Task Queue Reset        │
│  Service        │   │ Cleanup Service │   │ Service                 │
│                 │   │                 │   │                         │
│ • documents_    │   │ • Qdrant        │   │ • TaskQueueDocument     │
│   data          │   │   vectors       │   │ • TaskQueuePage         │
│ • documents_    │   │ • Neo4j graph   │   │ • TaskQueueChunk        │
│   data_line_    │   │ • Metadata      │   │                         │
│   items         │   │   embeddings    │   │ Reset statuses to       │
└─────────────────┘   └─────────────────┘   │ PENDING                 │
                                            └────────────┬────────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Existing Worker Infrastructure                      │
│                  (HierarchicalWorkerPool)                           │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │Classification│  │ Extraction  │  │   Vector    │  │  GraphRAG  │ │
│  │   Worker    │→ │   Worker    │→ │   Worker    │→ │   Worker   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Reusable Components

| Component | File Location | Role in Reindex |
|-----------|---------------|-----------------|
| `DocumentRepository.get_fully_indexed_documents()` | `backend/db/document_repository.py` | Query eligible documents |
| `IndexingService.delete_document_embeddings()` | `backend/services/indexing_service.py:195-274` | Clear Qdrant + Neo4j data |
| `HierarchicalTaskQueueManager` | `backend/queue_service/hierarchical_task_manager.py` | Manage task states |
| `HierarchicalWorkerPool` | `backend/queue_service/hierarchical_worker_pool.py` | Process PENDING tasks |
| `_check_and_route_tabular_document()` | `backend/services/task_queue_service.py:498-776` | Re-classify document type |
| `_trigger_tabular_extraction()` | `backend/services/task_queue_service.py:778-831` | Trigger data extraction |
| `process_vector_chunk_task()` | `backend/services/task_queue_service.py:833-979` | Vector indexing |
| `process_graphrag_chunk_task()` | `backend/services/task_queue_service.py:981-1093` | GraphRAG indexing |

### 4.3 Database Operations

#### 4.3.1 Phase 1: Data Extraction Cleanup

```sql
-- Delete existing line items
DELETE FROM documents_data_line_items
WHERE document_id = :doc_id;

-- Delete existing document data
DELETE FROM documents_data
WHERE document_id = :doc_id;

-- Reset document extraction flags
UPDATE documents SET
    extraction_status = NULL,
    extraction_eligible = NULL,
    extraction_schema_type = NULL
WHERE id = :doc_id;
```

#### 4.3.2 Phase 2: Task Queue Reset

```sql
-- Reset TaskQueueDocument (classification + extraction)
UPDATE task_queue_document SET
    classification_status = 'pending',
    classification_started_at = NULL,
    classification_completed_at = NULL,
    classification_error = NULL,
    extraction_status = 'pending',
    extraction_started_at = NULL,
    extraction_completed_at = NULL,
    extraction_error = NULL
WHERE document_id = :doc_id;

-- Reset TaskQueuePage (vector + graphrag, keep OCR completed!)
UPDATE task_queue_page SET
    vector_status = 'pending',
    vector_worker_id = NULL,
    vector_started_at = NULL,
    vector_completed_at = NULL,
    vector_error = NULL,
    vector_retry_count = 0,
    graphrag_status = 'pending',
    graphrag_worker_id = NULL,
    graphrag_started_at = NULL,
    graphrag_completed_at = NULL,
    graphrag_error = NULL,
    graphrag_retry_count = 0
    -- NOTE: ocr_status remains 'completed'
WHERE document_id = :doc_id;

-- Reset TaskQueueChunk (keep chunk_content!)
UPDATE task_queue_chunk SET
    vector_status = 'pending',
    vector_worker_id = NULL,
    vector_started_at = NULL,
    vector_completed_at = NULL,
    vector_error = NULL,
    vector_retry_count = 0,
    graphrag_status = 'pending',
    graphrag_worker_id = NULL,
    graphrag_started_at = NULL,
    graphrag_completed_at = NULL,
    graphrag_error = NULL,
    graphrag_retry_count = 0,
    entities_extracted = 0,
    relationships_extracted = 0
    -- NOTE: chunk_content and chunk_metadata are preserved
WHERE document_id = :doc_id;
```

#### 4.3.3 Phase 3: Document Status Update

```sql
-- Update document to indexing state
UPDATE documents SET
    index_status = 'indexing',
    indexing_details = '{
        "vector_indexing": {"status": "pending"},
        "metadata_extraction": {"status": "pending"},
        "graphrag_indexing": {"status": "pending"}
    }'::jsonb,
    updated_at = NOW()
WHERE id = :doc_id;
```

---

## 5. Processing Flow

### 5.1 Sequence Diagram

```
┌──────┐     ┌──────┐     ┌──────────┐     ┌─────────┐     ┌────────┐     ┌────────┐
│ User │     │ API  │     │ Reindex  │     │ Cleanup │     │ Queue  │     │ Worker │
│      │     │      │     │ Service  │     │ Service │     │ Reset  │     │ Pool   │
└──┬───┘     └──┬───┘     └────┬─────┘     └────┬────┘     └───┬────┘     └───┬────┘
   │            │              │                │              │              │
   │ Click      │              │                │              │              │
   │ Reindex    │              │                │              │              │
   │───────────>│              │                │              │              │
   │            │              │                │              │              │
   │            │ POST /docs/  │                │              │              │
   │            │ reindex      │                │              │              │
   │            │─────────────>│                │              │              │
   │            │              │                │              │              │
   │            │              │ Get eligible   │              │              │
   │            │              │ documents      │              │              │
   │            │              │───────────────>│              │              │
   │            │              │<───────────────│              │              │
   │            │              │                │              │              │
   │            │              │ For each document:            │              │
   │            │              │────────────────────────────────────────────>│
   │            │              │                │              │              │
   │            │              │ 1. Delete      │              │              │
   │            │              │    extracted   │              │              │
   │            │              │    data        │              │              │
   │            │              │───────────────>│              │              │
   │            │              │                │              │              │
   │            │              │ 2. Delete      │              │              │
   │            │              │    embeddings  │              │              │
   │            │              │───────────────>│              │              │
   │            │              │                │              │              │
   │            │              │ 3. Reset task  │              │              │
   │            │              │    statuses    │              │              │
   │            │              │───────────────────────────────>│              │
   │            │              │                │              │              │
   │            │              │ 4. Update doc  │              │              │
   │            │              │    status      │              │              │
   │            │              │───────────────────────────────>│              │
   │            │              │                │              │              │
   │            │<─────────────│                │              │              │
   │<───────────│ 200 OK       │                │              │              │
   │            │              │                │              │              │
   │            │              │                │              │ Workers pick │
   │            │              │                │              │ up PENDING   │
   │            │              │                │              │ tasks        │
   │            │              │                │              │<────────────>│
   │            │              │                │              │              │
   │            │              │                │              │ Process:     │
   │            │              │                │              │ 1.Classify   │
   │            │              │                │              │ 2.Extract    │
   │            │              │                │              │ 3.Vector     │
   │            │              │                │              │ 4.GraphRAG   │
   │            │              │                │              │<────────────>│
```

### 5.2 State Transitions

```
                              ┌─────────────────┐
                              │  FULLY INDEXED  │
                              │  (Start State)  │
                              └────────┬────────┘
                                       │
                              User clicks Reindex
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   CLEANUP &     │
                              │     RESET       │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
           ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
           │ Delete         │ │ Delete         │ │ Reset Task     │
           │ documents_data │ │ Qdrant vectors │ │ Queue to       │
           │ tables         │ │ Neo4j graph    │ │ PENDING        │
           └────────┬───────┘ └────────┬───────┘ └────────┬───────┘
                    │                  │                  │
                    └──────────────────┼──────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │    INDEXING     │
                              │  (Processing)   │
                              └────────┬────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
     │ Classification  │────>│   Extraction    │────>│ Vector Indexing │
     │ (Re-detect type)│     │ (Re-extract to  │     │ (Re-embed to    │
     │                 │     │  documents_data)│     │  Qdrant)        │
     └─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                              │
                                                              ▼
                                                    ┌─────────────────┐
                                                    │ GraphRAG Index  │
                                                    │ (Re-extract to  │
                                                    │  Neo4j)         │
                                                    └────────┬────────┘
                                                              │
                                                              ▼
                                                    ┌─────────────────┐
                                                    │  FULLY INDEXED  │
                                                    │  (End State)    │
                                                    └─────────────────┘
```

---

## 6. Data Preservation Strategy

### 6.1 What Gets Preserved

| Data | Preserved | Reason |
|------|-----------|--------|
| Original uploaded file | ✅ Yes | Source file unchanged |
| Converted markdown files | ✅ Yes | OCR not re-run |
| Page images (JPG) | ✅ Yes | Generated during OCR |
| `TaskQueueChunk.chunk_content` | ✅ Yes | V3.0 optimization - no re-chunking |
| `TaskQueuePage.page_file_path` | ✅ Yes | Points to existing markdown |
| `ocr_status = COMPLETED` | ✅ Yes | OCR phase not repeated |

### 6.2 What Gets Reset/Deleted

| Data | Action | Reason |
|------|--------|--------|
| `documents_data` | 🗑️ Delete & Recreate | Re-extract with new logic |
| `documents_data_line_items` | 🗑️ Delete & Recreate | Re-extract with new logic |
| Qdrant vector embeddings | 🗑️ Delete & Recreate | Re-embed with new logic |
| Neo4j graph entities | 🗑️ Delete & Recreate | Re-extract relationships |
| `document_metadata` | 🔄 Reset | Re-extract metadata |
| `vector_status` | 🔄 Reset to PENDING | Trigger re-indexing |
| `graphrag_status` | 🔄 Reset to PENDING | Trigger re-indexing |
| `extraction_status` | 🔄 Reset to PENDING | Trigger re-extraction |
| `classification_status` | 🔄 Reset to PENDING | Re-classify document |

---

## 7. Frontend Implementation

### 7.1 Service Layer

**File:** `frontend/src/services/documentService.js`

```javascript
/**
 * Reindex documents - re-run extraction and indexing
 * @param {string|null} workspaceId - Workspace ID, or null for all workspaces
 * @returns {Promise<Object>} - Reindex response
 */
async reindexDocuments(workspaceId = null) {
  try {
    const response = await http.post(
      `${this.apiDomain}/documents/reindex`,
      { workspace_id: workspaceId }
    );
    return response.data;
  } catch (error) {
    console.error("Error starting reindex:", error);
    throw error;
  }
}

/**
 * Get count of documents eligible for reindex
 * @param {string|null} workspaceId - Workspace ID, or null for all
 * @returns {Promise<number>} - Count of eligible documents
 */
async getReindexEligibleCount(workspaceId = null) {
  try {
    const params = workspaceId ? { workspace_id: workspaceId } : {};
    const response = await http.get(
      `${this.apiDomain}/documents/reindex-eligible-count`,
      { params }
    );
    return response.data.count;
  } catch (error) {
    console.error("Error getting reindex count:", error);
    return 0;
  }
}
```

### 7.2 WorkspaceSidebar Component Changes

**File:** `frontend/src/components/workspace/WorkspaceSidebar.jsx`

```jsx
// Add state for reindex
const [reindexing, setReindexing] = useState(false);

// Handler for global reindex
const handleReindexAll = async () => {
  // Get eligible count first
  const count = await documentService.getReindexEligibleCount(null);

  if (count === 0) {
    messageService.infoToast(t("Workspace.NoDocumentsToReindex"));
    return;
  }

  // Show confirmation dialog
  messageService.confirmDialog(
    t("Workspace.ReindexAllConfirm", { count }),
    async (confirmed) => {
      if (confirmed) {
        try {
          setReindexing(true);
          const response = await documentService.reindexDocuments(null);
          messageService.successToast(
            t("Workspace.ReindexStarted", { count: response.queued_documents })
          );
        } catch (error) {
          messageService.errorToast(t("Workspace.ReindexFailed"));
        } finally {
          setReindexing(false);
        }
      }
    }
  );
};

// In header JSX (beside + button)
<Button
  icon={reindexing ? "pi pi-spin pi-spinner" : "pi pi-sync"}
  text
  rounded
  onClick={handleReindexAll}
  disabled={reindexing}
  tooltip={t("Workspace.ReindexAllTooltip")}
  tooltipOptions={{ position: "bottom" }}
  style={{ width: '1.75rem', height: '1.75rem', padding: 0 }}
/>
```

### 7.3 DocumentList Component Changes

**File:** `frontend/src/components/documents/documentList.jsx`

```jsx
// Add state for workspace reindex
const [reindexingWorkspace, setReindexingWorkspace] = useState(false);

// Handler for workspace reindex
const handleReindexWorkspace = async () => {
  if (!currentWorkspaceId) return;

  const count = await documentService.getReindexEligibleCount(currentWorkspaceId);

  if (count === 0) {
    messageService.infoToast(t("DocumentList.NoDocumentsToReindex"));
    return;
  }

  messageService.confirmDialog(
    t("DocumentList.ReindexConfirm", { count }),
    async (confirmed) => {
      if (confirmed) {
        try {
          setReindexingWorkspace(true);
          const response = await documentService.reindexDocuments(currentWorkspaceId);
          messageService.successToast(
            t("DocumentList.ReindexStarted", { count: response.queued_documents })
          );
          // Refresh document list to show new statuses
          loadDocuments();
        } catch (error) {
          messageService.errorToast(t("DocumentList.ReindexFailed"));
        } finally {
          setReindexingWorkspace(false);
        }
      }
    }
  );
};

// In header JSX (beside Upload button)
<Button
  icon={reindexingWorkspace ? "pi pi-spin pi-spinner" : "pi pi-sync"}
  label={t("DocumentList.Reindex")}
  outlined
  severity="secondary"
  onClick={handleReindexWorkspace}
  disabled={!currentWorkspaceId || reindexingWorkspace}
  tooltip={t("DocumentList.ReindexTooltip")}
  tooltipOptions={{ position: "top" }}
/>
```

---

## 8. Internationalization

### 8.1 Translation Keys

**File:** `frontend/src/locales/en/translation.json`

```json
{
  "Workspace": {
    "ReindexAll": "Reindex All",
    "ReindexAllTooltip": "Re-extract and reindex documents across all workspaces",
    "ReindexAllConfirmAdmin": "This will re-run data extraction and indexing for ALL {{count}} documents across all workspaces. Existing data will be replaced. Continue?",
    "ReindexAllConfirmUser": "This will re-run data extraction and indexing for {{count}} documents that you own across all workspaces. Existing data will be replaced. Continue?",
    "ReindexStarted": "Reindex started for {{count}} documents",
    "ReindexFailed": "Failed to start reindexing",
    "NoDocumentsToReindex": "No fully indexed documents to reindex"
  },
  "DocumentList": {
    "Reindex": "Reindex",
    "ReindexTooltip": "Re-extract and reindex documents in this workspace",
    "ReindexConfirmAdmin": "This will re-run data extraction and indexing for ALL {{count}} documents in this workspace. Existing data will be replaced. Continue?",
    "ReindexConfirmUser": "This will re-run data extraction and indexing for {{count}} documents that you own in this workspace. Existing data will be replaced. Continue?",
    "ReindexStarted": "Reindex started for {{count}} documents",
    "ReindexFailed": "Failed to start reindexing",
    "NoDocumentsToReindex": "No fully indexed documents to reindex"
  }
}
```

**File:** `frontend/src/locales/zh/translation.json`

```json
{
  "Workspace": {
    "ReindexAll": "重新索引全部",
    "ReindexAllTooltip": "重新提取和索引所有工作区的文档",
    "ReindexAllConfirmAdmin": "这将为所有工作区的全部 {{count}} 个文档重新运行数据提取和索引。现有数据将被替换。是否继续？",
    "ReindexAllConfirmUser": "这将为您在所有工作区拥有的 {{count}} 个文档重新运行数据提取和索引。现有数据将被替换。是否继续？",
    "ReindexStarted": "已开始重新索引 {{count}} 个文档",
    "ReindexFailed": "启动重新索引失败",
    "NoDocumentsToReindex": "没有可重新索引的文档"
  },
  "DocumentList": {
    "Reindex": "重新索引",
    "ReindexTooltip": "重新提取和索引此工作区的文档",
    "ReindexConfirmAdmin": "这将为此工作区的全部 {{count}} 个文档重新运行数据提取和索引。现有数据将被替换。是否继续？",
    "ReindexConfirmUser": "这将为您在此工作区拥有的 {{count}} 个文档重新运行数据提取和索引。现有数据将被替换。是否继续？",
    "ReindexStarted": "已开始重新索引 {{count}} 个文档",
    "ReindexFailed": "启动重新索引失败",
    "NoDocumentsToReindex": "没有可重新索引的文档"
  }
}
```

---

## 9. Error Handling

### 9.1 Error Scenarios

| Scenario | Handling |
|----------|----------|
| No documents eligible | Return success with `queued_documents: 0`, show info toast |
| User lacks permission | Return 403, show error toast |
| Workspace not found | Return 404, show error toast |
| Database connection error | Return 500, show error toast, log error |
| Qdrant/Neo4j unavailable | Log warning, continue with available services |
| Worker pool busy | Queue tasks normally, workers process when available |

### 9.2 Partial Failure Handling

If cleanup succeeds but task queue reset fails:

1. Log the error with document ID
2. Include document in `skipped_documents` response
3. Document remains in original state (fully indexed)
4. User can retry later

---

## 10. Performance Considerations

### 10.1 Batch Processing

- Process documents in batches of 10 to avoid long-running transactions
- Use bulk delete operations where possible
- Commit after each document to prevent large rollbacks

### 10.2 Resource Usage

| Resource | Impact | Mitigation |
|----------|--------|------------|
| Database | High during cleanup | Batch operations, indexing |
| Qdrant | Moderate | Existing delete optimizations |
| Neo4j | Moderate | Batch node deletion |
| LLM API | High during extraction | Existing rate limiting |
| Memory | Low | Stream processing |

### 10.3 Estimated Processing Time

| Documents | Estimated Time |
|-----------|----------------|
| 1-10 | 1-5 minutes |
| 10-50 | 5-15 minutes |
| 50-100 | 15-30 minutes |
| 100+ | 30+ minutes |

*Times vary based on document size, complexity, and LLM response times.*

---

## 11. Existing Permission System Analysis

This section documents the current permission system architecture in the codebase, which the Reindex All feature will leverage.

### 11.1 Authentication Flow

**Location:** `backend/auth/dependencies.py` and `backend/auth/jwt_utils.py`

| Dependency | Purpose | Location |
|------------|---------|----------|
| `get_current_user()` | Extracts JWT, validates, returns User object | `auth/dependencies.py:20-65` |
| `get_current_active_user()` | Ensures user status is ACTIVE | `auth/dependencies.py:68-88` |
| `require_admin()` | Checks `user.role == UserRole.ADMIN` | `auth/dependencies.py:91-102` |
| `get_optional_user()` | Returns User if authenticated, None otherwise | `auth/dependencies.py:105-120` |

**JWT Token Structure:**
```python
{
    "sub": str(user_id),      # User UUID as string
    "username": username,      # Username
    "role": role,              # User role ("admin" or "user")
    "exp": expires_at,         # Expiration datetime
    "iat": now,               # Issued at datetime
    "type": "access"          # Token type
}
```

**Token Expiration:**
- Access tokens: 30 minutes (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`)
- Refresh tokens: 7 days (configurable via `REFRESH_TOKEN_EXPIRE_DAYS`)

### 11.2 User Roles & Status

**Location:** `backend/db/models.py:19-28`

```python
class UserRole(str, enum.Enum):
    ADMIN = "admin"
    USER = "user"

class UserStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
```

### 11.3 Document Ownership Model

**Location:** `backend/db/models.py:337-466`

```python
class Document(Base):
    owner_id = Column(PGUUID, ForeignKey("users.id"))  # Document owner (can be NULL)
    visibility = Column(String)  # 'private', 'shared', 'public'
    workspace_id = Column(PGUUID, ForeignKey("workspaces.id"))

    # Relationships
    owner = relationship("User")
    user_permissions = relationship("UserDocument")
```

### 11.4 Permission Types

**Location:** `backend/db/models.py:30-50`

```python
class DocumentPermission(str, enum.Enum):
    READ = "read"           # View document content
    UPDATE = "update"       # Edit document metadata
    DELETE = "delete"       # Delete document
    SHARE = "share"         # Share with other users
    FULL = "full"           # All permissions (owner level)

class PermissionOrigin(str, enum.Enum):
    OWNER = "owner"                 # User created/owns document
    SHARED = "shared"               # Explicitly shared by owner
    ADMIN_GRANTED = "admin_granted" # Admin granted permission
    PUBLIC = "public"               # Public visibility
```

### 11.5 UserDocument Bridge Table

**Location:** `backend/db/models.py:542-614`

The `UserDocument` model stores explicit permission grants between users and documents:

```python
class UserDocument(Base):
    __tablename__ = "user_documents"

    user_id = Column(PGUUID, ForeignKey("users.id"))
    document_id = Column(PGUUID, ForeignKey("documents.id"))

    # PostgreSQL array storing permission enums
    permissions = Column(ARRAY(Enum(DocumentPermission)), default=['read'])

    origin = Column(Enum(PermissionOrigin))  # How permission was granted
    is_owner = Column(Boolean)               # TRUE = document owner

    # Sharing metadata
    shared_by = Column(PGUUID, ForeignKey("users.id"))
    shared_at = Column(DateTime)
    expires_at = Column(DateTime)            # Optional expiration

    def has_permission(self, permission: str) -> bool:
        """Returns TRUE if user is owner, has 'full', or specific permission"""
```

### 11.6 Permission Service

**Location:** `backend/services/permission_service.py`

The `PermissionService` class provides all permission-related operations:

**Permission Check Hierarchy (evaluated in order):**
```
1. Admin Bypass    → user.role == ADMIN         → Allow ALL
2. Ownership       → is_owner == TRUE           → Allow ALL
3. Full Permission → 'full' in permissions      → Allow ALL
4. Specific Match  → required_perm in perms     → Allow specific
5. Default         → Deny
```

**Key Methods:**
```python
class PermissionService:
    # Permission checks
    def check_permission(user, document_id, permission) -> bool
    def has_any_access(user, document_id) -> bool
    def is_owner(user, document_id) -> bool
    def can_read(user, document_id) -> bool
    def can_update(user, document_id) -> bool
    def can_delete(user, document_id) -> bool
    def can_share(user, document_id) -> bool

    # Permission management
    def grant_owner_permission(user_id, document_id)
    def share_document(document_id, owner, target_user_id, permissions)
    def revoke_access(document_id, owner, target_user_id)
    def transfer_ownership(document_id, current_owner, new_owner_id)

    # Queries
    def get_user_accessible_document_ids(user, permission=None) -> Set[UUID]
    def get_document_users(document_id, requesting_user) -> List[UserDocument]
```

### 11.7 Admin Bypass Implementation

**Location:** `backend/db/user_document_repository.py:314-316`

Admins automatically bypass all permission checks:

```python
def check_permission(self, user_id, document_id, required_permission, user_role=None):
    # Admin bypass - return True immediately
    if user_role == UserRole.ADMIN:
        return True
    # ... proceed with normal permission checks
```

### 11.8 Existing Permission Patterns in Endpoints

**Pattern 1: Simple Ownership Check** (most common in codebase)
```python
@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    doc = repo.get_by_id(doc_uuid)

    # Check permission: must be owner OR have delete permission
    if doc.owner_id and doc.owner_id != current_user.id:
        perm_service = PermissionService(db)
        if not perm_service.has_permission(current_user.id, doc.id, 'delete'):
            raise HTTPException(status_code=403, detail="Permission denied")
```

**Pattern 2: Admin-Only Endpoint**
```python
@app.post("/admin/some-action")
async def admin_action(current_user: User = Depends(require_admin)):
    # Only admins can reach here
```

**Pattern 3: Using PermissionService Convenience Methods**
```python
perm_service = PermissionService(db)
if not perm_service.can_delete(current_user, document_id):
    raise HTTPException(status_code=403, detail="Cannot delete document")
```

### 11.9 Permission Flow Diagram

```
API Request with JWT Token
         │
         ▼
┌─────────────────────────────┐
│ get_current_active_user()   │
│ ├─ Extract Bearer token     │
│ ├─ JWTUtils.verify_token()  │
│ ├─ Get User from database   │
│ └─ Check status == ACTIVE   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Endpoint Permission Check   │
└──────────────┬──────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
┌─────────┐         ┌─────────┐
│ ADMIN?  │───YES──>│ ALLOW   │
└────┬────┘         └─────────┘
     │NO
     ▼
┌─────────────────┐
│ Is document     │───YES──>┌─────────┐
│ owner?          │         │ ALLOW   │
└────────┬────────┘         └─────────┘
         │NO
         ▼
┌─────────────────────────┐
│ PermissionService       │
│ .check_permission()     │
│ ├─ Query UserDocument   │
│ ├─ Check expiration     │
│ ├─ Check 'full' perm    │
│ └─ Check specific perm  │
└───────────┬─────────────┘
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
┌─────────┐   ┌─────────┐
│ ALLOW   │   │ DENY    │
│ (found) │   │ (403)   │
└─────────┘   └─────────┘
```

### 11.10 Relevant Source Files

| File | Purpose |
|------|---------|
| `backend/auth/dependencies.py` | Authentication dependencies (get_current_user, require_admin) |
| `backend/auth/jwt_utils.py` | JWT token creation and verification |
| `backend/db/models.py` | User, Document, UserDocument, enums |
| `backend/db/user_document_repository.py` | Permission database operations |
| `backend/services/permission_service.py` | Permission business logic |
| `backend/services/sharing_api.py` | Sharing API endpoints |

---

## 12. Access Control & Security for Reindex Feature

### 12.1 Role-Based Access Control (RBAC)

The system implements role-based access control with two user roles defined in `backend/db/models.py`:

```python
class UserRole(str, enum.Enum):
    ADMIN = "admin"
    USER = "user"
```

#### 12.1.1 Access Control Matrix

| Action | Admin Role | User Role |
|--------|------------|-----------|
| **Global Reindex (All Workspaces)** | ✅ Can reindex ALL documents across ALL workspaces | ✅ Can reindex only documents they OWN across all workspaces |
| **Workspace Reindex** | ✅ Can reindex ALL documents in any workspace | ✅ Can reindex only documents they OWN in the workspace |
| **View Reindex Button (Sidebar)** | ✅ Always visible | ✅ Visible (filters to owned docs) |
| **View Reindex Button (Workspace)** | ✅ Always visible | ✅ Visible (filters to owned docs) |

#### 12.1.2 Authorization Logic

**Backend Authorization (Python):**

```python
from db.models import UserRole

def get_reindexable_documents(
    current_user: User,
    workspace_id: Optional[UUID],
    db: Session
) -> List[Document]:
    """
    Get documents eligible for reindex based on user role.

    - ADMIN: Can reindex ALL fully indexed documents
    - USER: Can only reindex documents they OWN (owner_id == user.id)
    """
    query = db.query(Document).filter(
        Document.convert_status == ConvertStatus.CONVERTED,
        Document.index_status == IndexStatus.INDEXED,
        Document.deleted_at.is_(None)
    )

    # Filter by workspace if specified
    if workspace_id:
        query = query.filter(Document.workspace_id == workspace_id)

    # CRITICAL: Non-admin users can only reindex their own documents
    if current_user.role != UserRole.ADMIN:
        query = query.filter(Document.owner_id == current_user.id)

    return query.all()
```

**API Endpoint Authorization:**

```python
@app.post("/documents/reindex")
async def reindex_documents(
    request: ReindexRequest,
    current_user: User = Depends(get_current_active_user),  # Must be authenticated
    db: Session = Depends(get_db)
):
    """
    Reindex documents based on user role:
    - Admin: Can reindex any document
    - User: Can only reindex documents they own
    """
    # Get documents based on role-filtered query
    documents = get_reindexable_documents(
        current_user=current_user,
        workspace_id=request.workspace_id,
        db=db
    )

    # Process reindex for filtered documents
    # ...
```

#### 12.1.3 Document Eligibility with Access Control

A document is eligible for reindexing if:

```python
# For ADMIN users:
admin_eligible = (
    document.convert_status == "converted" AND
    document.index_status == "indexed" AND
    document.deleted_at IS NULL AND
    (workspace_id IS NULL OR document.workspace_id == workspace_id)
)

# For regular USER:
user_eligible = (
    document.convert_status == "converted" AND
    document.index_status == "indexed" AND
    document.deleted_at IS NULL AND
    document.owner_id == current_user.id AND  # MUST be owner
    (workspace_id IS NULL OR document.workspace_id == workspace_id)
)
```

### 12.2 Frontend Access Control

#### 12.2.1 UI Behavior by Role

**Admin User:**
- Global reindex button shows count of ALL fully indexed documents
- Workspace reindex button shows count of ALL documents in workspace
- Confirmation dialog shows total count for all documents

**Regular User:**
- Global reindex button shows count of documents user OWNS
- Workspace reindex button shows count of owned documents in workspace
- Confirmation dialog clearly indicates "your documents"

#### 12.2.2 Frontend Implementation

**Fetch eligible count with role awareness:**

```javascript
// documentService.js
async getReindexEligibleCount(workspaceId = null) {
  try {
    const params = workspaceId ? { workspace_id: workspaceId } : {};
    // Backend automatically filters based on user role from JWT token
    const response = await http.get(
      `${this.apiDomain}/documents/reindex-eligible-count`,
      { params }
    );
    return response.data;  // { count: N, is_admin: bool }
  } catch (error) {
    console.error("Error getting reindex count:", error);
    return { count: 0, is_admin: false };
  }
}
```

**Confirmation dialog with role context:**

```jsx
// For Admin users:
t("Workspace.ReindexAllConfirmAdmin", { count })
// "This will re-run data extraction and indexing for ALL {count} documents across all workspaces."

// For regular users:
t("Workspace.ReindexAllConfirmUser", { count })
// "This will re-run data extraction and indexing for {count} documents that you own."
```

### 12.3 API Response with Access Info

**Response includes role context:**

```json
{
  "status": "success",
  "message": "Reindex started for 6 documents",
  "queued_documents": 6,
  "skipped_documents": 2,
  "workspace_id": "550e8400-e29b-41d4-a716-446655440000",
  "access_context": {
    "user_role": "user",
    "scope": "owned_documents_only",
    "total_workspace_documents": 15,
    "accessible_documents": 6
  },
  "details": {
    "queued": [...],
    "skipped": [
      {"id": "doc-uuid-3", "filename": "other-user.pdf", "reason": "not_owner"}
    ]
  }
}
```

### 12.4 Security Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         User Clicks "Reindex"                            │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    Authentication Check                                   │
│                    (JWT Token Valid?)                                     │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             ┌─────────────┐             ┌─────────────┐
             │   Invalid   │             │    Valid    │
             │   Token     │             │   Token     │
             └──────┬──────┘             └──────┬──────┘
                    │                           │
                    ▼                           ▼
             ┌─────────────┐       ┌────────────────────────┐
             │ 401 Error   │       │ Extract User Role      │
             │ Unauthorized│       │ from JWT payload       │
             └─────────────┘       └───────────┬────────────┘
                                               │
                                 ┌─────────────┴─────────────┐
                                 │                           │
                                 ▼                           ▼
                          ┌─────────────┐             ┌─────────────┐
                          │ role=ADMIN  │             │ role=USER   │
                          └──────┬──────┘             └──────┬──────┘
                                 │                           │
                                 ▼                           ▼
                    ┌────────────────────────┐  ┌────────────────────────┐
                    │ Query ALL documents    │  │ Query documents WHERE  │
                    │ matching criteria      │  │ owner_id = user.id     │
                    │                        │  │ AND matching criteria  │
                    └───────────┬────────────┘  └───────────┬────────────┘
                                │                           │
                                └─────────────┬─────────────┘
                                              │
                                              ▼
                                ┌────────────────────────┐
                                │ Process Reindex for    │
                                │ Filtered Documents     │
                                └────────────────────────┘
```

### 12.5 Rate Limiting

- Reindex can only be triggered once per workspace while in progress
- Global reindex blocked if any workspace is already reindexing
- Rate limit: Maximum 1 reindex request per user per minute

### 12.6 Audit Logging

All reindex operations are logged with:

```python
audit_log = {
    "action": "reindex_initiated",
    "user_id": current_user.id,
    "user_role": current_user.role.value,
    "workspace_id": workspace_id,
    "document_count": len(documents),
    "scope": "all_documents" if is_admin else "owned_documents",
    "timestamp": datetime.utcnow().isoformat(),
    "ip_address": request.client.host
}
```

Logged events:
- Reindex initiation (with user role and scope)
- Reindex completion/failure
- Per-document status changes (in `document_status_log` table)

---

## 13. Testing Strategy

### 13.1 Unit Tests

- [ ] `get_fully_indexed_documents()` returns correct documents
- [ ] Data cleanup deletes correct records
- [ ] Task queue reset preserves `chunk_content` and `ocr_status`
- [ ] Permission checks work correctly

**Access Control Unit Tests:**
- [ ] Admin user query returns ALL eligible documents
- [ ] Regular user query returns ONLY owned documents
- [ ] Documents without owner_id are excluded for regular users
- [ ] Workspace filter works correctly with role filter

### 13.2 Integration Tests

- [ ] Full reindex flow for single document
- [ ] Workspace-filtered reindex
- [ ] All-workspace reindex
- [ ] Concurrent reindex requests handled
- [ ] Worker pool processes reset tasks

**Access Control Integration Tests:**
- [ ] Admin can reindex documents owned by other users
- [ ] Regular user cannot reindex documents owned by others
- [ ] Admin global reindex processes all documents
- [ ] User global reindex processes only owned documents
- [ ] Mixed ownership workspace: admin sees all, user sees owned only

### 13.3 E2E Tests

- [ ] UI button triggers reindex
- [ ] Confirmation dialog shows correct count
- [ ] Progress updates shown in UI
- [ ] Documents return to "Fully Indexed" state

**Access Control E2E Tests:**
- [ ] Admin sees "ALL X documents" in confirmation dialog
- [ ] Regular user sees "X documents that you own" in confirmation dialog
- [ ] Regular user's reindex count matches their owned documents
- [ ] Admin can reindex from any workspace view
- [ ] Regular user reindex button works correctly with owned docs only

---

## 14. Rollout Plan

### 14.1 Phase 1: Backend Implementation

1. Add `get_fully_indexed_documents()` to repository
2. Implement reindex endpoint in `main.py`
3. Add data cleanup functions
4. Add task queue reset functions
5. Unit tests

### 14.2 Phase 2: Frontend Implementation

1. Add `reindexDocuments()` to document service
2. Add reindex button to WorkspaceSidebar
3. Add reindex button to DocumentList
4. Add confirmation dialogs
5. Add translations

### 14.3 Phase 3: Testing & Documentation

1. Integration testing
2. E2E testing
3. Performance testing with large datasets
4. Update user documentation

---

## 15. Queue-Based Reindex Architecture (v1.3)

### 15.1 Overview

In v1.3, the reindex feature was redesigned to use a **fully asynchronous queue-based architecture**. This change prevents UI blocking when reindexing large numbers of documents.

**Key Changes:**
- Endpoint `/documents/reindex` now only **creates tasks** (fast, non-blocking)
- Actual reindex work is done by **background workers**
- Each document is processed as a **separate task** in the queue
- Workers pick up and process reindex tasks alongside OCR/Vector/GraphRAG tasks

### 15.2 New Data Model

Added `reindex_status` columns to `task_queue_document`:

```sql
-- Migration: 026_add_reindex_status_columns.sql
ALTER TABLE task_queue_document ADD COLUMN reindex_status task_status DEFAULT NULL;
ALTER TABLE task_queue_document ADD COLUMN reindex_worker_id VARCHAR(100);
ALTER TABLE task_queue_document ADD COLUMN reindex_started_at TIMESTAMPTZ;
ALTER TABLE task_queue_document ADD COLUMN reindex_completed_at TIMESTAMPTZ;
ALTER TABLE task_queue_document ADD COLUMN reindex_error TEXT;
ALTER TABLE task_queue_document ADD COLUMN reindex_retry_count INT DEFAULT 0;
ALTER TABLE task_queue_document ADD COLUMN reindex_last_heartbeat TIMESTAMPTZ;
```

**Status Values:**
- `NULL`: Never requested reindex
- `pending`: Queued for reindex, waiting for worker
- `processing`: Worker is processing this document
- `completed`: Reindex finished successfully
- `failed`: Reindex failed (after max retries)

### 15.3 Processing Flow (Queue-Based)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  User Clicks "Reindex" Button                             │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    POST /documents/reindex                                │
│                    (Fast, Non-Blocking)                                   │
│                                                                           │
│  1. Query eligible documents (INDEXED status)                            │
│  2. Apply RBAC filter (admin=all, user=owned only)                       │
│  3. For each document:                                                    │
│     - Create/update TaskQueueDocument with reindex_status='pending'      │
│     - Update document.index_status = 'indexing'                          │
│  4. Notify worker pool (new_task_event.set())                           │
│  5. Return immediately with queued count                                 │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │
                              ▼  Returns immediately
┌──────────────────────────────────────────────────────────────────────────┐
│  Response: { status: "success", documents_processed: 6 }                 │
│  (UI shows "Reindex queued for 6 documents")                             │
└──────────────────────────────────────────────────────────────────────────┘

                    ===== BACKGROUND PROCESSING =====

┌──────────────────────────────────────────────────────────────────────────┐
│                    HierarchicalWorkerPool                                 │
│                    (Background Workers)                                   │
│                                                                           │
│  Workers poll for tasks in priority order:                               │
│    1. Reindex tasks    (reindex_status='pending')                        │
│    2. GraphRAG tasks   (graphrag_status='pending')                       │
│    3. Vector tasks     (vector_status='pending')                         │
│    4. OCR tasks        (ocr_status='pending')                            │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    process_reindex_task()                                 │
│                    (TaskQueueService)                                     │
│                                                                           │
│  For each claimed reindex task:                                          │
│    Step 1: Delete DocumentData and line items                            │
│    Step 2: Clear Qdrant vectors and Neo4j GraphRAG                       │
│    Step 3: Reset TaskQueueDocument extraction_status                     │
│    Step 4: Reset TaskQueuePage vector/graphrag statuses                  │
│    Step 5: Delete existing TaskQueueChunks                               │
│    Step 6: Recreate chunks from markdown files                           │
│    Step 7: Update document status                                        │
│    Step 8: Complete reindex task                                         │
│    Step 9: Notify workers about new chunk tasks                          │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Workers pick up Vector and GraphRAG tasks for recreated chunks          │
│  Document status transitions: INDEXING → INDEXED                         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 15.4 Code Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `ReindexTaskData` | `queue_service/hierarchical_task_manager.py` | Typed container for reindex task data |
| `create_reindex_task()` | `queue_service/hierarchical_task_manager.py` | Create/queue a reindex task |
| `claim_reindex_task()` | `queue_service/hierarchical_task_manager.py` | Worker claims next reindex task |
| `complete_reindex_task()` | `queue_service/hierarchical_task_manager.py` | Mark reindex as completed |
| `fail_reindex_task()` | `queue_service/hierarchical_task_manager.py` | Mark reindex as failed (with retry) |
| `process_reindex_task()` | `services/task_queue_service.py` | Execute the actual reindex work |
| `_process_reindex()` | `queue_service/hierarchical_worker_pool.py` | Worker method to handle reindex |
| `/documents/reindex` | `main.py` | API endpoint (creates tasks only) |

### 15.5 Benefits of Queue-Based Architecture

1. **Non-Blocking UI**: User gets immediate response, documents processed in background
2. **Scalable**: Can process thousands of documents without timeout
3. **Fault-Tolerant**: Individual document failures don't affect others
4. **Resumable**: Retry failed documents automatically (up to max_retries)
5. **Observable**: Track progress via task queue statuses
6. **Resource-Efficient**: Workers process one document at a time

### 15.6 Task Priority

Workers check for tasks in this order (highest priority first):

1. **Reindex tasks** - User-requested, should process quickly
2. **GraphRAG tasks** - Final indexing phase
3. **Vector tasks** - Vector embedding phase
4. **OCR tasks** - Initial conversion phase

This ensures reindex tasks are picked up promptly while not blocking critical indexing work.

---

## 16. Future Enhancements

1. **Selective Reindex**: Allow reindexing specific phases only (extraction-only, indexing-only)
2. **Scheduled Reindex**: Automatic reindex on schema changes
3. **Progress Tracking**: Detailed progress bar for large batch reindex
4. **Reindex History**: View past reindex operations and their results
5. **Dry Run Mode**: Preview which documents would be affected without executing

---

## 16. Appendix

### 16.1 Related Files

| File | Purpose |
|------|---------|
| `backend/main.py` | API endpoints |
| `backend/db/document_repository.py` | Document queries |
| `backend/db/models.py` | Database models |
| `backend/services/indexing_service.py` | Embedding cleanup |
| `backend/services/task_queue_service.py` | Task processing |
| `backend/queue_service/hierarchical_task_manager.py` | Queue management |
| `backend/queue_service/hierarchical_worker_pool.py` | Worker pool |
| `backend/extraction_service/extraction_service.py` | Data extraction |
| `frontend/src/services/documentService.js` | Frontend API calls |
| `frontend/src/components/workspace/WorkspaceSidebar.jsx` | Sidebar UI |
| `frontend/src/components/documents/documentList.jsx` | Document list UI |

### 16.2 Database Schema Reference

```sql
-- documents_data (extracted document-level fields)
CREATE TABLE documents_data (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    field_name VARCHAR,
    field_value TEXT,
    confidence FLOAT,
    ...
);

-- documents_data_line_items (extracted row-level data)
CREATE TABLE documents_data_line_items (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    line_number INTEGER,
    field_data JSONB,
    ...
);

-- task_queue_document
CREATE TABLE task_queue_document (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    convert_status VARCHAR,
    classification_status VARCHAR,
    extraction_status VARCHAR,
    ...
);

-- task_queue_page
CREATE TABLE task_queue_page (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    page_number INTEGER,
    ocr_status VARCHAR,      -- Preserved as 'completed'
    vector_status VARCHAR,   -- Reset to 'pending'
    graphrag_status VARCHAR, -- Reset to 'pending'
    page_file_path VARCHAR,  -- Preserved
    ...
);

-- task_queue_chunk
CREATE TABLE task_queue_chunk (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    page_id UUID,
    chunk_content TEXT,      -- Preserved (V3.0)
    chunk_metadata JSONB,    -- Preserved
    vector_status VARCHAR,   -- Reset to 'pending'
    graphrag_status VARCHAR, -- Reset to 'pending'
    ...
);
```
