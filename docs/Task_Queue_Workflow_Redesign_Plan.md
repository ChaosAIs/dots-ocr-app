# Task Queue Workflow Redesign Plan

## Executive Summary

This document outlines the redesign of the document processing pipeline to:
1. Fix the missing tabular data workflow in the new task queue system
2. Add proper metadata extraction for all document types
3. **Store tabular data schema in `document_metadata` for direct use by dynamic SQL generation**

---

## Problem Statement

### Current Issues

1. **Tabular Routing Bypassed**: When `TASK_QUEUE_ENABLED=true`, CSV/Excel files go through row-level chunking instead of the optimized tabular path
2. **No Metadata for Tabular Documents**: `TabularExtractionService` does not populate `documents.document_metadata`
3. **No Extraction Status in Task Queue**: Data extraction is not tracked as a first-class task
4. **Late Classification**: Document type is determined after chunking, too late to change strategy
5. **Metadata Extraction After Indexing**: Metadata is extracted after vector indexing, missing opportunity to inform chunking
6. **Field Mappings Not in Metadata**: Dynamic SQL generator has to infer field mappings from `documents_data` at query time, which fails when extraction hasn't run

### Root Cause

The old worker pool path had tabular routing in `trigger_embedding_for_document()`:
```
Old Path: Convert → trigger_embedding_for_document() → [ROUTING] → Tabular OR Standard
New Path: Convert → Chunking → Vector → GraphRAG → Metadata → Extraction (NO ROUTING!)
```

---

## Proposed Solution

### New Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DOCUMENT-LEVEL TASKS                               │
│                          (task_queue_document table)                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐                │
│  │ 1. CONVERT  │───>│ 2. CLASSIFY &    │───>│ 3. ROUTING      │                │
│  │             │    │    METADATA      │    │    DECISION     │                │
│  └─────────────┘    └──────────────────┘    └────────┬────────┘                │
│                                                      │                          │
│                           ┌──────────────────────────┴──────────────────────┐  │
│                           │                                                  │  │
│                           ▼                                                  ▼  │
│              ┌────────────────────────┐                    ┌─────────────────┐  │
│              │   TABULAR PATH         │                    │  STANDARD PATH  │  │
│              │                        │                    │                 │  │
│              │  4a. Data Extraction   │                    │  4b. Chunking   │  │
│              │  (to documents_data)   │                    │  (semantic)     │  │
│              │                        │                    │                 │  │
│              │  5a. Summary Chunks    │                    │                 │  │
│              │  (1-3 chunks)          │                    │                 │  │
│              └───────────┬────────────┘                    └────────┬────────┘  │
│                          │                                          │           │
│                          └──────────────────┬───────────────────────┘           │
│                                             │                                   │
│                                             ▼                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CHUNK-LEVEL TASKS                                  │
│                      (task_queue_chunk table - existing)                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────┐                            │
│  │ 6. VECTOR INDEXING  │───>│ 7. GRAPHRAG         │                            │
│  │ (all documents)     │    │ (standard only)     │                            │
│  └─────────────────────┘    └─────────────────────┘                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Workflow Steps

### Step 1: CONVERT (OCR/Document Conversion)

**Trigger**: Document uploaded
**Input**: Raw file (PDF, CSV, Excel, Word, Image)
**Output**: Markdown file(s)
**Status Column**: `convert_status`

| File Type | Conversion Method |
|-----------|-------------------|
| PDF, Images | DotsOCR Parser |
| CSV, Excel, Word | doc_converter_manager |

**No changes needed** - this step works correctly.

---

### Step 2: CLASSIFY & METADATA EXTRACTION (NEW)

**Trigger**: Convert completed
**Input**: Converted markdown content
**Output**: Document classification + metadata
**Status Column**: `classification_status` (NEW)

#### What This Step Does:

1. **Read converted markdown** (full document content)
2. **Detect document type** using `TabularDataDetector` + LLM analysis
3. **Extract metadata** using simplified `HierarchicalMetadataExtractor`
4. **Store results** in `documents.document_metadata`
5. **Embed metadata** to `metadatas` collection in Qdrant

#### Metadata Extracted (Standard Documents):

```json
{
  "document_type": "resume|report|contract|manual|...",
  "subject_name": "Felix Yang Resume",
  "topics": ["software engineering", "python", "machine learning"],
  "key_entities": [
    {"type": "person", "name": "Felix Yang"},
    {"type": "organization", "name": "Google"}
  ],
  "is_tabular": false,
  "summary": "Resume of a software engineer with 10 years experience",
  "confidence": 0.95
}
```

#### Metadata Extracted (Tabular Documents) - WITH DATA SCHEMA:

For tabular documents (CSV, Excel, invoices with line items), the metadata includes a **complete data schema** that the dynamic SQL generator can use directly:

```json
{
  "document_type": "spreadsheet",
  "subject_name": "ProductInventory",
  "topics": ["inventory", "products", "stock management"],
  "is_tabular": true,
  "schema_type": "inventory_report",
  "row_count": 1000,
  "summary": "Product inventory data with 1000 items across multiple categories",
  "confidence": 0.95,

  "tabular_schema": {
    "column_count": 7,
    "columns": [
      {
        "name": "ProductID",
        "display_name": "Product ID",
        "data_type": "string",
        "semantic_type": "identifier",
        "nullable": false,
        "sample_values": ["P000001", "P000002", "P000003"]
      },
      {
        "name": "ProductName",
        "display_name": "Product Name",
        "data_type": "string",
        "semantic_type": "entity",
        "nullable": false,
        "sample_values": ["Nike T-Shirt", "Samsung TV", "Apple iPhone"]
      },
      {
        "name": "Category",
        "display_name": "Category",
        "data_type": "string",
        "semantic_type": "category",
        "nullable": false,
        "unique_values": ["Electronics", "Clothing", "Books", "Sports"],
        "aggregation": "group_by"
      },
      {
        "name": "Brand",
        "display_name": "Brand",
        "data_type": "string",
        "semantic_type": "category",
        "nullable": true,
        "unique_values": ["Nike", "Samsung", "Apple", "Sony"],
        "aggregation": "group_by"
      },
      {
        "name": "Price",
        "display_name": "Price",
        "data_type": "decimal",
        "semantic_type": "amount",
        "nullable": false,
        "min_value": 9.99,
        "max_value": 999.99,
        "aggregation": "sum,avg,min,max"
      },
      {
        "name": "StockQuantity",
        "display_name": "Stock Quantity",
        "data_type": "integer",
        "semantic_type": "quantity",
        "nullable": false,
        "min_value": 0,
        "max_value": 1500,
        "aggregation": "sum,avg,min,max"
      },
      {
        "name": "SKU",
        "display_name": "SKU",
        "data_type": "string",
        "semantic_type": "identifier",
        "nullable": false,
        "pattern": "XXX-XXX-0000"
      }
    ],
    "primary_key": "ProductID",
    "suggested_groupings": ["Category", "Brand"],
    "suggested_metrics": ["Price", "StockQuantity"],
    "date_columns": [],
    "amount_columns": ["Price"],
    "quantity_columns": ["StockQuantity"]
  }
}
```

#### Schema Field Definitions:

| Field | Description | Used By |
|-------|-------------|---------|
| `name` | Original column name from CSV/Excel | SQL query generation |
| `display_name` | Human-readable name | UI display |
| `data_type` | `string`, `integer`, `decimal`, `date`, `boolean` | SQL type casting |
| `semantic_type` | `identifier`, `entity`, `category`, `amount`, `quantity`, `date` | Query understanding |
| `aggregation` | `sum`, `avg`, `min`, `max`, `count`, `group_by` | SQL aggregation functions |
| `sample_values` | Sample data for context | LLM prompt context |
| `unique_values` | Distinct values (for categories) | Filter suggestions |
| `min_value/max_value` | Value range (for numerics) | Validation, range queries |

#### For Tabular Documents (Processing Flags):

- Set `is_tabular_data = true` in `documents` table
- Set `processing_path = 'tabular'`
- Store complete `tabular_schema` in `document_metadata`

---

### Step 3: ROUTING DECISION

**Trigger**: Classification completed
**Input**: `documents.document_metadata`, `is_tabular_data`, `processing_path`
**Output**: Route to TABULAR or STANDARD path

#### Routing Logic:

```
IF is_tabular_data == true AND processing_path == 'tabular':
    → TABULAR PATH (Steps 4a, 5a)
ELSE:
    → STANDARD PATH (Step 4b)
```

#### Documents Routed to TABULAR PATH:

| Extension | Document Type |
|-----------|---------------|
| .csv | Always |
| .xlsx, .xls | Always |
| .pdf | If detected as invoice/receipt/bank statement with line items |

#### Documents Routed to STANDARD PATH (with Embedded Tables):

Many regular documents contain embedded tables within their content:
- Research papers with data tables
- Reports with comparison tables
- Manuals with specification tables
- Contracts with term tables

**These are NOT tabular documents** - they go through the STANDARD PATH with table-aware chunking.

##### Table-Aware Chunking Strategy

The `AdaptiveChunker` handles embedded tables by:

1. **Preserving Table Structure**: Tables are kept intact within chunks, not split across chunks
2. **Table Content Ratio Detection**: Calculates what percentage of content is markdown tables
3. **Hybrid Document Handling**: Documents with < 50% table content are treated as hybrid (standard chunking with table preservation)

```
Document Type Decision:
├── Table Content Ratio < 50%  → STANDARD PATH (hybrid document)
│   └── Tables preserved in chunks, indexed with surrounding context
├── Table Content Ratio 50-85% → Analyzed for skip decision
└── Table Content Ratio > 85%  → TABULAR PATH candidate
```

##### Key Differences:

| Aspect | TABULAR PATH | STANDARD PATH (with tables) |
|--------|--------------|------------------------------|
| **Goal** | Extract structured data for SQL queries | Preserve tables for semantic search |
| **Storage** | `documents_data_line_items` table | Qdrant vector embeddings |
| **Query Type** | SQL aggregation (SUM, GROUP BY) | Semantic similarity search |
| **Table Handling** | Parse into rows, store in PostgreSQL | Keep as markdown in chunks |
| **Use Case** | "Total sales by category" | "What does the comparison table show?" |

##### Example: Research Paper with Data Table

A 10-page research paper with a 20-row results table:

```
Document: research_paper.pdf
├── Page 1-3: Introduction, Literature Review (text)
├── Page 4: Methodology (text)
├── Page 5: Results Table (20 rows) ← Embedded table
├── Page 6-8: Discussion (text)
└── Page 9-10: Conclusion, References (text)
```

**Processing Path**: STANDARD (table ratio ~10%)
- Table preserved as markdown within chunk
- Chunk includes table + surrounding context
- Indexed for semantic search: "What were the experimental results?"

**NOT processed as TABULAR** because:
- Primary content is narrative text
- Table supports the narrative, not the primary data
- User queries are semantic, not aggregation-based

---

### Step 4a: DATA EXTRACTION (Tabular Path)

**Trigger**: Routing decision = TABULAR
**Input**: Converted markdown, document_metadata
**Output**: Structured data in PostgreSQL
**Status Column**: `extraction_status` (NEW)

#### What This Step Does:

1. Parse markdown tables into structured rows
2. Apply field mappings (from classification step)
3. Store header data in `documents_data.header_data`
4. Store rows based on row count threshold (inline or external)
5. Generate field mappings for SQL query generation

#### Data Storage Strategy:

The system uses **external storage only** for all line item data, regardless of row count:

| Row Count | Storage Mode | `line_items_storage` | Data Location |
|-----------|--------------|---------------------|---------------|
| All rows | **External** | `'external'` | `documents_data_line_items` table |

**Note**: The `EXTRACTION_*_MAX_ROWS` environment variables control **extraction strategy** (how data is extracted), NOT storage. See [Extraction Strategies](#extraction-strategies) below.

#### Storage Details:

**External Storage** (All documents):
- `documents_data.line_items` is always empty (`[]`)
- All line items stored in `documents_data_line_items` table
- Each row is a separate record with `line_number` and `data` (JSONB)
- Consistent SQL pattern for all queries
- Supports pagination and streaming

**Why External-Only?** (See [Appendix I](#appendix-i-external-only-storage-rationale) for detailed analysis)
- **Single SQL Pattern**: One consistent query pattern instead of dual inline/external patterns
- **Simpler LLM Prompts**: SQL generator doesn't need to detect storage type
- **Eliminates Conversion Logic**: Removes 150+ line `_convert_to_external_storage_sql()` function
- **Predictable Performance**: Consistent execution path regardless of document size
- **Easier Testing**: One code path to validate instead of two

#### Database Schema:

| Table | Column | Type | Description |
|-------|--------|------|-------------|
| `documents_data` | `header_data` | JSONB | Column headers, metadata, document info |
| `documents_data` | `line_items_count` | INTEGER | Total number of rows |
| `documents_data` | `extraction_metadata` | JSONB | Contains `field_mappings` for SQL generation |
| `documents_data_line_items` | `documents_data_id` | UUID | FK to parent `documents_data` |
| `documents_data_line_items` | `line_number` | INTEGER | Row index (0-based) |
| `documents_data_line_items` | `data` | JSONB | Single row data object |

**Columns to DROP** (no longer needed with external-only storage):
- `documents_data.line_items` - Previously stored inline row data as JSONB array
- `documents_data.line_items_storage` - No longer needed (always external)

#### Extraction Approach:

The system uses **direct parsing only** for tabular data extraction - no LLM involved in row extraction:

| Source Type | Extraction Method |
|-------------|-------------------|
| CSV | Direct parsing with Python csv module |
| Excel (.xlsx, .xls) | Direct parsing with openpyxl/xlrd |
| PDF with tables | Markdown table parsing (from OCR output) |

**Simplified Design**:
- No LLM calls for row extraction (was causing complexity and latency)
- LLM only used for document classification and header/field mapping inference
- All row data extracted via direct parsing
- All extracted data stored in `documents_data_line_items` table

**Removed Configuration** (no longer needed):
```env
# DEPRECATED - Remove these:
# EXTRACTION_DIRECT_LLM_MAX_ROWS=50
# EXTRACTION_CHUNKED_LLM_MAX_ROWS=500
# EXTRACTION_HYBRID_MAX_ROWS=5000
```

#### Example: Invoice (5 rows)

```json
// documents_data record
{
  "schema_type": "invoice",
  "line_items_count": 5,
  "header_data": {
    "column_headers": ["Item", "Quantity", "Price", "Total"],
    "invoice_number": "INV-001"
  },
  "extraction_metadata": {
    "field_mappings": {...}
  }
}

// documents_data_line_items records (5 rows)
{"documents_data_id": "uuid", "line_number": 0, "data": {"Item": "Widget A", "Quantity": 10, "Price": 25.00, "Total": 250.00}}
{"documents_data_id": "uuid", "line_number": 1, "data": {"Item": "Widget B", "Quantity": 5, "Price": 50.00, "Total": 250.00}}
{"documents_data_id": "uuid", "line_number": 2, "data": {"Item": "Widget C", "Quantity": 20, "Price": 10.00, "Total": 200.00}}
{"documents_data_id": "uuid", "line_number": 3, "data": {"Item": "Widget D", "Quantity": 2, "Price": 100.00, "Total": 200.00}}
{"documents_data_id": "uuid", "line_number": 4, "data": {"Item": "Widget E", "Quantity": 8, "Price": 12.50, "Total": 100.00}}
```

#### Example: Large Inventory (1000 rows)

```json
// documents_data record
{
  "schema_type": "inventory_report",
  "line_items_count": 1000,
  "header_data": {
    "column_headers": ["ProductID", "Name", "Category", "Price", "Stock"]
  },
  "extraction_metadata": {
    "field_mappings": {...}
  }
}

// documents_data_line_items records (1000 rows)
{"documents_data_id": "uuid", "line_number": 0, "data": {"ProductID": "P001", ...}}
{"documents_data_id": "uuid", "line_number": 1, "data": {"ProductID": "P002", ...}}
...
{"documents_data_id": "uuid", "line_number": 999, "data": {"ProductID": "P1000", ...}}
```

---

### Step 5a: SUMMARY CHUNK GENERATION (Tabular Path)

**Trigger**: Data extraction completed
**Input**: `documents_data` content
**Output**: 1-3 summary chunks in Qdrant
**Status Column**: Part of `extraction_status`

#### Summary Chunks Generated:

| Chunk # | Type | Content |
|---------|------|---------|
| 1 | `tabular_summary` | LLM-generated document summary |
| 2 | `tabular_schema` | Schema description with field types |
| 3 | `tabular_context` | Business context and sample values |

#### Chunk Metadata:

```json
{
  "document_id": "uuid",
  "chunk_type": "tabular_summary",
  "is_tabular": true,
  "schema_type": "inventory_report",
  "row_count": 1000,
  "column_count": 7,
  "columns": ["ProductID", "ProductName", "Category", "Brand", "Price", "Quantity", "SKU"]
}
```

---

### Step 4b: CHUNKING (Standard Path)

**Trigger**: Routing decision = STANDARD
**Input**: Converted markdown
**Output**: Semantic chunks
**Status Column**: Existing (part of page processing)

#### What This Step Does:

1. Use `AdaptiveChunker` for semantic chunking
2. Create chunk tasks in `task_queue_chunk` table
3. Store chunk content and metadata

**No changes needed** - this step works correctly for standard documents.

---

### Step 6: VECTOR INDEXING

**Trigger**: Chunks created (both paths)
**Input**: Chunk content from `task_queue_chunk.chunk_content`
**Output**: Vectors in Qdrant `documents` collection
**Status Column**: Existing `vector_status`

**No changes needed** - works for both paths.

---

### Step 7: GRAPHRAG (Standard Path Only)

**Trigger**: Vector indexing completed
**Input**: Chunks from Qdrant
**Output**: Entities and relationships in Neo4j
**Status Column**: Existing `graphrag_status`

#### Skip Conditions:

- `is_tabular_data = true` → Skip (data already structured)
- `skip_graphrag = true` → Skip (set during classification)
- Document type in skip list → Skip

**No changes needed** - already has skip logic.

---

## Database Schema Changes

### New Table: `task_queue_document`

```sql
CREATE TABLE task_queue_document (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Convert status (moved from page level for single coordination)
    convert_status task_status NOT NULL DEFAULT 'pending',
    convert_worker_id VARCHAR(100),
    convert_started_at TIMESTAMPTZ,
    convert_completed_at TIMESTAMPTZ,
    convert_error TEXT,
    convert_retry_count INT DEFAULT 0,
    convert_last_heartbeat TIMESTAMPTZ,

    -- Classification & Metadata status (NEW)
    classification_status task_status NOT NULL DEFAULT 'pending',
    classification_worker_id VARCHAR(100),
    classification_started_at TIMESTAMPTZ,
    classification_completed_at TIMESTAMPTZ,
    classification_error TEXT,
    classification_retry_count INT DEFAULT 0,
    classification_last_heartbeat TIMESTAMPTZ,

    -- Data Extraction status (NEW - for tabular documents)
    extraction_status task_status NOT NULL DEFAULT 'pending',
    extraction_worker_id VARCHAR(100),
    extraction_started_at TIMESTAMPTZ,
    extraction_completed_at TIMESTAMPTZ,
    extraction_error TEXT,
    extraction_retry_count INT DEFAULT 0,
    extraction_last_heartbeat TIMESTAMPTZ,

    -- Routing decision
    processing_path VARCHAR(20) DEFAULT 'standard',  -- 'standard' or 'tabular'

    -- Common fields
    max_retries INT DEFAULT 3,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(document_id)
);

-- Indexes for worker polling
CREATE INDEX idx_tqd_classification_pending ON task_queue_document(classification_status, classification_retry_count, created_at)
    WHERE classification_status IN ('pending', 'failed');
CREATE INDEX idx_tqd_extraction_pending ON task_queue_document(extraction_status, extraction_retry_count, created_at)
    WHERE extraction_status IN ('pending', 'failed');
CREATE INDEX idx_tqd_classification_heartbeat ON task_queue_document(classification_last_heartbeat)
    WHERE classification_status = 'processing';
CREATE INDEX idx_tqd_extraction_heartbeat ON task_queue_document(extraction_last_heartbeat)
    WHERE extraction_status = 'processing';
```

### Migration for Existing Tables

```sql
-- Add is_tabular flag to task_queue_page if not exists
ALTER TABLE task_queue_page ADD COLUMN IF NOT EXISTS skip_chunking BOOLEAN DEFAULT FALSE;
ALTER TABLE task_queue_page ADD COLUMN IF NOT EXISTS skip_reason VARCHAR(100);
```

---

## Implementation Steps

### Phase 1: Database Schema (Priority: HIGH)

1. Create migration file `021_add_task_queue_document_table.sql`
2. Add `task_queue_document` table
3. Add indexes for worker polling
4. Update `db/models.py` with new model

### Phase 2: Classification Service (Priority: HIGH)

1. Create `backend/services/classification_service.py`
2. Implement document type detection (combine existing `TabularDataDetector` + LLM)
3. Implement simplified metadata extraction for tabular documents
4. Store results in `documents.document_metadata`
5. Embed metadata to `metadatas` collection

### Phase 3: Task Queue Manager Updates (Priority: HIGH)

1. Update `hierarchical_task_manager.py`:
   - Add `create_document_level_task()` method
   - Add `complete_classification_task()` method
   - Add `complete_extraction_task()` method
   - Add `get_pending_classification_tasks()` method
   - Add `get_pending_extraction_tasks()` method

2. Update task flow:
   - After convert completes → trigger classification
   - After classification completes → routing decision
   - If tabular → trigger extraction
   - If standard → trigger chunking

### Phase 4: Task Queue Service Updates (Priority: HIGH)

1. Update `task_queue_service.py`:
   - Add `process_classification_task()` method
   - Add `process_extraction_task()` method
   - Modify `process_ocr_page_task()` to check routing before chunking

2. Worker loop updates:
   - Poll for classification tasks
   - Poll for extraction tasks
   - Respect task dependencies

### Phase 5: Tabular Extraction Service Updates (Priority: MEDIUM)

1. Update `tabular_extraction_service.py`:
   - Add metadata extraction step
   - Populate `documents.document_metadata`
   - Call `upsert_document_metadata_embedding()`
   - Integrate with task queue status updates

### Phase 6: Integration & Testing (Priority: HIGH)

1. Update upload endpoint to create `task_queue_document` entry
2. Test CSV upload → full tabular path
3. Test PDF upload → full standard path
4. Test mixed workload
5. Test retry/recovery scenarios

---

## Task Dependencies

```
                    ┌─────────────────┐
                    │    UPLOAD       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   1. CONVERT    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ 2. CLASSIFY &   │
                    │    METADATA     │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
            ┌───────▼───────┐ ┌───────▼───────┐
            │ TABULAR PATH  │ │ STANDARD PATH │
            └───────┬───────┘ └───────┬───────┘
                    │                 │
            ┌───────▼───────┐ ┌───────▼───────┐
            │ 4a. EXTRACT   │ │ 4b. CHUNK     │
            └───────┬───────┘ └───────┬───────┘
                    │                 │
            ┌───────▼───────┐         │
            │ 5a. SUMMARY   │         │
            │    CHUNKS     │         │
            └───────┬───────┘         │
                    │                 │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ 6. VECTOR INDEX │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ 7. GRAPHRAG     │
                    │ (standard only) │
                    └─────────────────┘
```

---

## Worker Polling Order

Workers should poll tasks in this priority order:

1. **Conversion tasks** (prerequisite for everything)
2. **Classification tasks** (unblocks routing decision)
3. **Extraction tasks** (for tabular documents)
4. **Vector chunk tasks** (for all documents)
5. **GraphRAG chunk tasks** (for standard documents)

---

## Status Tracking Summary

### Document-Level Status (task_queue_document)

| Column | Values | Description |
|--------|--------|-------------|
| `convert_status` | pending/processing/completed/failed | File conversion |
| `classification_status` | pending/processing/completed/failed | Type detection + metadata |
| `extraction_status` | pending/processing/completed/failed/skipped | Tabular data extraction |
| `processing_path` | standard/tabular | Routing decision |

### Page-Level Status (task_queue_page)

| Column | Values | Description |
|--------|--------|-------------|
| `ocr_status` | pending/processing/completed/failed | Page OCR |
| `skip_chunking` | true/false | Skip if tabular |
| `vector_status` | pending/processing/completed/failed | Aggregated from chunks |
| `graphrag_status` | pending/processing/completed/failed/skipped | Aggregated from chunks |

### Chunk-Level Status (task_queue_chunk)

| Column | Values | Description |
|--------|--------|-------------|
| `vector_status` | pending/processing/completed/failed | Chunk embedding |
| `graphrag_status` | pending/processing/completed/failed/skipped | Entity extraction |

---

## Rollback Plan

If issues arise, the system can be rolled back by:

1. Set `TASK_QUEUE_ENABLED=false` in `.env`
2. System will use old worker pool path with `trigger_embedding_for_document()`
3. Old path has working tabular routing

---

## Success Criteria

1. **CSV Upload Test**: CSV file goes through tabular path, creates 1-3 summary chunks, populates `documents_data`
2. **Metadata Present**: `documents.document_metadata` populated for ALL document types
3. **Document Router Works**: Tabular documents found by vector-based document router
4. **SQL Queries Work**: Dynamic SQL generation works with field mappings
5. **Status Visibility**: All task statuses visible in `task_queue_document` table
6. **Recovery Works**: Failed tasks can be retried, stale tasks detected by heartbeat

---

## Timeline Estimate

| Phase | Tasks | Dependencies |
|-------|-------|--------------|
| Phase 1 | Database schema | None |
| Phase 2 | Classification service | Phase 1 |
| Phase 3 | Task queue manager | Phase 1, 2 |
| Phase 4 | Task queue service | Phase 3 |
| Phase 5 | Tabular service updates | Phase 2, 3 |
| Phase 6 | Integration & testing | All phases |

---

## Appendix A: File Changes Summary

| File | Changes |
|------|---------|
| `db/migrations/021_*.sql` | NEW: task_queue_document table |
| `db/migrations/022_*.sql` | NEW: Drop `line_items` and `line_items_storage` columns |
| `db/models.py` | ADD: TaskQueueDocument model; REMOVE: `line_items`, `line_items_storage` from DocumentData |
| `queue_service/__init__.py` | EXPORT: TaskQueueDocument |
| `services/classification_service.py` | NEW: Document classification + schema extraction service |
| `queue_service/hierarchical_task_manager.py` | ADD: classification/extraction task methods |
| `services/task_queue_service.py` | ADD: process_classification_task, process_extraction_task |
| `services/tabular_extraction_service.py` | ADD: metadata extraction, document_metadata population |
| `analytics_service/sql_query_executor.py` | UPDATE: Use tabular_schema from document_metadata, remove storage type detection |
| `analytics_service/llm_sql_generator.py` | UPDATE: Enhanced prompt with schema context, remove `_convert_to_external_storage_sql()` |
| `extraction_service/extraction_service.py` | UPDATE: Use direct parsing only, remove LLM extraction strategies |
| `extraction_service/extraction_config.py` | UPDATE: Remove `ExtractionStrategy` enum and threshold constants |
| `main.py` | UPDATE: create task_queue_document on upload |
| `scripts/migrate_tabular_schemas.py` | NEW: Migration script for existing documents |
| `scripts/migrate_inline_to_external.py` | NEW: Migration script to convert inline storage to external (run BEFORE 022 migration) |

---

## Appendix B: API Changes

No API changes required. All changes are internal to the processing pipeline.

---

## Appendix C: Configuration

New environment variables (optional):

```env
# Classification settings
CLASSIFICATION_TIMEOUT=120
CLASSIFICATION_MAX_RETRIES=3

# Extraction settings
EXTRACTION_TIMEOUT=300
EXTRACTION_MAX_RETRIES=3
```

---

## Appendix D: Dynamic SQL Generation Integration

### Current Problem

The current dynamic SQL generator (`sql_query_executor.py`) tries to get field mappings from:
1. `documents_data.extraction_metadata.field_mappings` - but this requires extraction to have run
2. Inferred from `documents_data` rows - but fails if no data exists

When the task queue bypasses tabular extraction, neither source is available, causing:
```
ERROR: No field mappings available for dynamic SQL generation
```

### New Approach: Use `document_metadata.tabular_schema`

After the redesign, the dynamic SQL generator will:

1. **First, check `documents.document_metadata.tabular_schema`** (populated during classification)
2. **Fallback to `documents_data.extraction_metadata`** (if exists)
3. **Error only if neither exists**

### Updated SQL Generator Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DYNAMIC SQL GENERATION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. User asks: "What is the max inventory for each category?"  │
│                                                                 │
│  2. Get document_metadata from documents table                  │
│     └── SELECT document_metadata FROM documents WHERE id = ?    │
│                                                                 │
│  3. Extract tabular_schema                                      │
│     └── schema = document_metadata['tabular_schema']            │
│                                                                 │
│  4. Build LLM prompt with schema context:                       │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │ Available columns:                                       │ │
│     │ - Category (string, group_by)                           │ │
│     │ - StockQuantity (integer, sum/avg/min/max)              │ │
│     │ - ProductName (string, entity)                          │ │
│     │                                                          │ │
│     │ User question: "max inventory for each category"        │ │
│     │                                                          │ │
│     │ Generate SQL query...                                    │ │
│     └─────────────────────────────────────────────────────────┘ │
│                                                                 │
│  5. LLM generates SQL:                                          │
│     SELECT Category, MAX(StockQuantity) as max_inventory       │
│     FROM line_items                                             │
│     GROUP BY Category                                           │
│                                                                 │
│  6. Execute against documents_data_line_items                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Benefits of Schema in Metadata

| Benefit | Description |
|---------|-------------|
| **Available Immediately** | Schema extracted during classification, before data extraction |
| **No Dependency on Extraction** | SQL generator can understand schema even if line items not yet extracted |
| **Rich Type Information** | `semantic_type` and `aggregation` hints guide SQL generation |
| **Sample Values** | LLM has concrete examples for better understanding |
| **Consistent Source** | Single source of truth in `document_metadata` |

### Code Changes in `sql_query_executor.py`

```python
# NEW: Get field mappings from document_metadata first
def _get_field_mappings(self, document_id: UUID, db: Session) -> dict:
    """Get field mappings from document_metadata.tabular_schema."""

    # 1. Try document_metadata.tabular_schema (NEW - preferred)
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc and doc.document_metadata:
        tabular_schema = doc.document_metadata.get('tabular_schema')
        if tabular_schema:
            # Convert tabular_schema to field_mappings format
            return self._schema_to_field_mappings(tabular_schema)

    # 2. Fallback to documents_data.extraction_metadata (existing)
    doc_data = db.query(DocumentData).filter(DocumentData.document_id == document_id).first()
    if doc_data and doc_data.extraction_metadata:
        field_mappings = doc_data.extraction_metadata.get('field_mappings')
        if field_mappings:
            return field_mappings

    # 3. Error if neither available
    raise ValueError("No field mappings available for dynamic SQL generation")

def _schema_to_field_mappings(self, tabular_schema: dict) -> dict:
    """Convert tabular_schema format to field_mappings format."""
    field_mappings = {}
    for col in tabular_schema.get('columns', []):
        field_mappings[col['name']] = {
            'display_name': col.get('display_name', col['name']),
            'data_type': col.get('data_type', 'string'),
            'semantic_type': col.get('semantic_type', 'unknown'),
            'aggregation': col.get('aggregation', ''),
            'nullable': col.get('nullable', True),
            'sample_values': col.get('sample_values', []),
            'unique_values': col.get('unique_values', []),
        }
    return field_mappings
```

### LLM Prompt Enhancement

With the detailed schema, the LLM prompt for SQL generation becomes much richer:

```
You are a SQL query generator for tabular data analytics.

## Data Schema

Table: line_items (stored in documents_data_line_items)
Document: ProductInventory.csv
Row Count: 1000

### Columns:

| Column | Type | Semantic | Aggregation | Sample Values |
|--------|------|----------|-------------|---------------|
| ProductID | string | identifier | - | P000001, P000002 |
| ProductName | string | entity | - | Nike T-Shirt, Samsung TV |
| Category | string | category | GROUP BY | Electronics, Clothing, Books |
| Brand | string | category | GROUP BY | Nike, Samsung, Apple |
| Price | decimal | amount | SUM, AVG, MIN, MAX | 29.99, 199.99 |
| StockQuantity | integer | quantity | SUM, AVG, MIN, MAX | 100, 500, 1000 |
| SKU | string | identifier | - | ELE-SAM-0001 |

### Suggested Query Patterns:

- Group by: Category, Brand
- Metrics: Price, StockQuantity
- Primary key: ProductID

## User Question

"What is the maximum and minimum inventory for products?"

## Instructions

1. Generate a valid SQL query using the columns above
2. Use appropriate aggregation functions based on semantic types
3. Use GROUP BY for category columns if aggregating
4. Return only the SQL query, no explanation

## SQL Query:
```

---

## Appendix E: Schema Extraction During Classification

### How Schema is Extracted

During the classification step (Step 2), for tabular documents:

1. **Parse Markdown Table**
   - Read converted markdown file
   - Detect table structure (headers, rows)
   - Extract column names

2. **Analyze Sample Rows**
   - Take first 10-20 rows as sample
   - Infer data types from values
   - Detect patterns (dates, amounts, IDs)

3. **Detect Semantic Types**
   - Use column name heuristics ("Price" → amount, "Date" → date)
   - Use value patterns (starts with "$" → amount)
   - Use LLM for ambiguous cases

4. **Compute Statistics**
   - Count unique values (for categories)
   - Find min/max (for numerics)
   - Extract sample values

5. **Generate Schema JSON**
   - Build `tabular_schema` structure
   - Store in `document_metadata`

### Schema Extraction Code Location

```
backend/services/classification_service.py
    └── ClassificationService
        └── classify_document()
            └── _extract_tabular_schema()  # NEW method
                ├── _parse_markdown_table()
                ├── _infer_column_types()
                ├── _detect_semantic_types()
                └── _compute_column_statistics()
```

### Example: Schema Extraction from CSV

**Input Markdown:**
```markdown
# ProductInventory

| ProductID | ProductName | Category | Brand | Price | StockQuantity | SKU |
|-----------|-------------|----------|-------|-------|---------------|-----|
| P000001 | Nike T-Shirt | Clothing | Nike | 29.99 | 150 | CLO-NIK-0001 |
| P000002 | Samsung TV | Electronics | Samsung | 499.99 | 50 | ELE-SAM-0002 |
...
```

**Extracted Schema:**
```json
{
  "tabular_schema": {
    "column_count": 7,
    "columns": [
      {"name": "ProductID", "data_type": "string", "semantic_type": "identifier"},
      {"name": "ProductName", "data_type": "string", "semantic_type": "entity"},
      {"name": "Category", "data_type": "string", "semantic_type": "category", "aggregation": "group_by"},
      {"name": "Brand", "data_type": "string", "semantic_type": "category", "aggregation": "group_by"},
      {"name": "Price", "data_type": "decimal", "semantic_type": "amount", "aggregation": "sum,avg,min,max"},
      {"name": "StockQuantity", "data_type": "integer", "semantic_type": "quantity", "aggregation": "sum,avg,min,max"},
      {"name": "SKU", "data_type": "string", "semantic_type": "identifier"}
    ]
  }
}
```

---

## Appendix F: Migration Path for Existing Documents

### For Existing Tabular Documents Without Schema

Run a one-time migration script to:

1. Find all documents with `is_tabular_data = true` and missing `tabular_schema`
2. Read their `documents_data.header_data` and `extraction_metadata`
3. Convert to `tabular_schema` format
4. Update `documents.document_metadata`

```sql
-- Find documents needing migration
SELECT d.id, d.original_filename, d.document_metadata
FROM documents d
WHERE d.is_tabular_data = true
  AND (d.document_metadata IS NULL
       OR d.document_metadata->>'tabular_schema' IS NULL);
```

### Migration Script Pseudocode

```python
def migrate_tabular_schemas():
    """Migrate existing tabular documents to have tabular_schema in metadata."""

    docs = db.query(Document).filter(
        Document.is_tabular_data == True,
        or_(
            Document.document_metadata == None,
            ~Document.document_metadata.has_key('tabular_schema')
        )
    ).all()

    for doc in docs:
        # Get existing field mappings from documents_data
        doc_data = db.query(DocumentData).filter(
            DocumentData.document_id == doc.id
        ).first()

        if doc_data:
            # Convert to tabular_schema format
            tabular_schema = convert_to_schema(
                header_data=doc_data.header_data,
                field_mappings=doc_data.extraction_metadata.get('field_mappings', {})
            )

            # Update document_metadata
            if doc.document_metadata is None:
                doc.document_metadata = {}
            doc.document_metadata['tabular_schema'] = tabular_schema
            doc.document_metadata['is_tabular'] = True

            db.commit()
            print(f"Migrated: {doc.original_filename}")
```

---

## Appendix G: Comprehensive Old Path vs New Path Comparison

### Overview

This section provides a detailed comparison of all features between the old worker pool path (`trigger_embedding_for_document()`) and the new task queue path (`TaskQueueService`).

### Feature Comparison Table

| Feature | Old Path (Worker Pool) | New Path (Task Queue) | Status |
|---------|------------------------|----------------------|--------|
| **Document Routing** | | | |
| Tabular Detection (Database) | ✅ Checks `is_tabular_data` | ❌ Not implemented | **MISSING** |
| Tabular Detection (Extension) | ✅ Uses `TabularDataDetector` | ❌ Not implemented | **MISSING** |
| Routing Decision | ✅ Before chunking | ❌ No routing | **MISSING** |
| Tabular Path Execution | ✅ `trigger_tabular_extraction()` | ❌ Not called | **MISSING** |
| **Chunking** | | | |
| Standard Chunking | ✅ `_index_chunks_to_qdrant()` | ✅ `chunk_markdown_with_summaries()` | ✅ Working |
| Tabular Chunking | ✅ Skipped (summary only) | ❌ Full row chunking | **BROKEN** |
| Single-File Doc Handling | ✅ Handled in old path | ✅ Checks for existing chunks | ✅ Working |
| **Metadata Extraction** | | | |
| Phase 1.5 Metadata | ✅ `HierarchicalMetadataExtractor` | ✅ Same extractor | ✅ Working |
| Metadata to PostgreSQL | ✅ `update_document_metadata()` | ✅ Same method | ✅ Working |
| Metadata to Qdrant | ❌ Not implemented | ✅ `upsert_document_metadata_embedding()` | ✅ **IMPROVED** |
| Tabular Document Metadata | ✅ Via `TabularExtractionService` | ❌ Not populated | **MISSING** |
| **GraphRAG** | | | |
| Entity Extraction | ✅ Background thread | ✅ Background worker | ✅ Working |
| Skip Logic (Document Type) | ✅ After metadata | ✅ After metadata | ✅ Working |
| Skip Logic (Tabular) | ✅ In tabular path | ❌ Never triggered | **MISSING** |
| **Data Extraction** | | | |
| Tabular Extraction | ✅ `ExtractionService.extract_document()` | ❌ Not triggered | **MISSING** |
| Field Mappings | ✅ Stored in `documents_data.extraction_metadata` | ❌ Not stored | **MISSING** |
| Line Items (≤500 rows) | ✅ Inline in `documents_data.line_items` | ❌ Not stored | **MISSING** |
| Line Items (>500 rows) | ✅ External in `documents_data_line_items` | ❌ Not stored | **MISSING** |
| **WebSocket Broadcasts** | | | |
| Indexing Status | ✅ `status: indexing` | ❓ Different mechanism | Partial |
| Vector Indexed | ✅ `status: vector_indexed` | ❓ Different mechanism | Partial |
| Metadata Extracting | ✅ `status: extracting_metadata` | ❌ Not broadcast | **MISSING** |
| Metadata Extracted | ✅ `status: metadata_extracted` | ❌ Not broadcast | **MISSING** |
| Metadata Failed | ✅ `status: metadata_extraction_failed` | ❌ Not broadcast | **MISSING** |
| GraphRAG Status | ✅ `status: graphrag_indexing/indexed` | ❓ Different mechanism | Partial |
| **Status Tracking** | | | |
| Granular Indexing Details | ✅ `indexing_details` JSONB | ✅ Same structure | ✅ Working |
| Retry Tracking | ❌ Limited | ✅ `retry_count`, `max_retries` | ✅ **IMPROVED** |
| Heartbeat Monitoring | ❌ Not implemented | ✅ `last_heartbeat` per task | ✅ **IMPROVED** |
| Per-Task Status | ❌ Document-level only | ✅ Page/Chunk-level status | ✅ **IMPROVED** |
| **Resilience** | | | |
| Task Recovery | ❌ Manual restart required | ✅ Automatic on startup | ✅ **IMPROVED** |
| Partial Failure Handling | ❌ All-or-nothing | ✅ Per-task retry | ✅ **IMPROVED** |
| Worker Assignment | ❌ Single thread | ✅ Worker ID tracking | ✅ **IMPROVED** |
| Stale Task Detection | ❌ Not implemented | ✅ Heartbeat timeout | ✅ **IMPROVED** |

### Missing Features Summary

#### 1. **Tabular Document Routing** (Critical)

**Old Path** (`indexer.py:1216-1277`):
```python
# Check database for tabular flag
is_tabular = doc.is_tabular_data
processing_path = doc.processing_path

# Check by extension if not in database
is_tabular, reason = TabularDataDetector.is_tabular_data(filename=filename)

# Route to tabular service
if is_tabular and processing_path == 'tabular':
    trigger_tabular_extraction(source_name, output_dir, filename, ...)
    return  # Exit - tabular service handles everything
```

**New Path** (`task_queue_service.py`): No equivalent code exists. All files go to `chunk_markdown_with_summaries()`.

**Impact**: CSV/Excel files are chunked row-by-row instead of being sent to tabular extraction.

---

#### 2. **Tabular Data Extraction & Storage** (Critical)

**Old Path** (`tabular_extraction_service.py` + `extraction_service.py`):
- Creates `DocumentData` records with `header_data`, `field_mappings`
- Uses hybrid storage strategy based on row count (DEPRECATED - see note below)
- Stores `field_mappings` in `documents_data.extraction_metadata`
- Generates 1-3 summary chunks for vector search

**NOTE**: The old hybrid storage (inline for ≤500 rows, external for >500 rows) is being replaced with **external-only storage** for all documents. See [Appendix I](#appendix-i-external-only-storage-rationale) for rationale.

**New Path**: TabularExtractionService is never called, so:
- No `documents_data` records created
- No `field_mappings` for SQL generation
- No line items stored in external table
- No summary chunks (instead, hundreds of row chunks polluting vector index)

**Impact**: Dynamic SQL generation fails with "No field mappings available".

---

#### 3. **WebSocket Metadata Status Broadcasts** (Medium)

**Old Path** broadcasts these status messages during metadata extraction:
```python
# During extraction
{"status": "extracting_metadata", "message": "Extracting document metadata...", "phase": 1.5}
{"status": "extracting_metadata", "message": "Metadata: {progress_msg}", "phase": 1.5}

# On completion
{"status": "metadata_extracted", "metadata": {...}, "phase": 1.5}

# On failure
{"status": "metadata_extraction_failed", "message": "...", "phase": 1.5}
```

**New Path**: No WebSocket broadcasts during metadata extraction phase.

**Impact**: Frontend doesn't receive real-time updates during metadata extraction.

---

### Improved Features in New Path

#### 1. **Metadata Embedding to Qdrant** (New in Task Queue)

**New Path** (`hierarchical_task_manager.py:1600-1606`):
```python
# Embed metadata to vector collection for fast document routing
from rag_service.vectorstore import upsert_document_metadata_embedding
upsert_document_metadata_embedding(
    document_id=str(document_id),
    source_name=source_name,
    filename=filename,
    metadata=metadata
)
```

**Old Path**: Does NOT embed metadata to Qdrant's `metadatas` collection.

**Benefit**: Enables fast vector-based document routing for RAG queries.

---

#### 2. **Task-Level Resilience**

| Feature | Old Path | New Path |
|---------|----------|----------|
| Retry Count | None | Per-task with `retry_count` |
| Max Retries | None | Configurable `max_retries` |
| Heartbeat | None | `last_heartbeat` per task |
| Worker Assignment | None | `worker_id` tracking |
| Stale Detection | Manual | Automatic timeout |
| Recovery on Startup | Manual | `resume_incomplete_indexing()` |

---

#### 3. **Granular Status Tracking**

| Level | Old Path | New Path |
|-------|----------|----------|
| Document | `index_status` (enum) | Same + `task_queue_document` |
| Page | None | `task_queue_page` with `ocr_status` |
| Chunk | None | `task_queue_chunk` with `vector_status`, `graphrag_status` |

---

### Recommendations

1. **Add Tabular Routing to New Path** (Priority: HIGH)
   - Check `is_tabular_data` and `processing_path` before chunking
   - Call `TabularExtractionService.process_tabular_document()` for tabular files
   - Skip standard chunking for tabular documents

2. **Add WebSocket Broadcasts for Metadata** (Priority: MEDIUM)
   - Broadcast `extracting_metadata` at start
   - Broadcast `metadata_extracted` on success
   - Broadcast `metadata_extraction_failed` on error

3. **Port Metadata Embedding to Old Path** (Priority: LOW)
   - Add `upsert_document_metadata_embedding()` call to old path
   - Ensures feature parity if users need to fall back

4. **Implement `tabular_schema` in Metadata** (Priority: HIGH)
   - Extract schema during classification/conversion
   - Store in `documents.document_metadata.tabular_schema`
   - Use in SQL generator instead of `documents_data` lookup

---

### Reference: Key Code Locations

| Feature | Old Path Location | New Path Location |
|---------|-------------------|-------------------|
| Tabular Detection | `indexer.py:1216-1248` | ❌ Not implemented |
| Tabular Routing | `indexer.py:1256-1277` | ❌ Not implemented |
| Standard Chunking | `indexer.py:1383` (`_index_chunks_to_qdrant`) | `task_queue_service.py:300` (`chunk_markdown_with_summaries`) |
| Metadata Extraction | `indexer.py:1416-1545` | `hierarchical_task_manager.py:1508-1640` |
| Metadata Embedding | ❌ Not implemented | `hierarchical_task_manager.py:1600-1606` |
| GraphRAG | `indexer.py:1553-1559` (`_run_graphrag_background`) | `hierarchical_task_manager.py:1070-1160` |
| WebSocket Broadcasts | `indexer.py:1375-1414, 1426-1544` | ❌ Limited |
| Tabular Extraction | `tabular_extraction_service.py:76-282` | ❌ Never called |

---

## Appendix H: Data Schema Storage Comparison

### Overview

This appendix clarifies the relationship between data stored in different locations and addresses potential duplication concerns.

### Storage Locations for Tabular Data

| Location | Table/Column | Purpose | When Created |
|----------|--------------|---------|--------------|
| `documents.document_metadata` | `tabular_schema` | Schema for SQL generation, document routing | During classification (PROPOSED) |
| `documents_data.header_data` | JSONB | Document-level extracted values | During data extraction |
| `documents_data.extraction_metadata` | `field_mappings` | Column semantic mappings | During data extraction |
| `documents_data.line_items` | JSONB array | Actual row data (≤500 rows) | During data extraction |
| `documents_data_line_items` | Table | Actual row data (>500 rows) | During data extraction |

### What's in `documents_data.header_data`?

The `header_data` column stores **document-level extracted values** that vary by document type:

#### For Invoices (`schema_type: "invoice"`):
```json
{
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-01-15",
    "due_date": "2024-02-15",
    "vendor_name": "Acme Corp",
    "vendor_address": "123 Main St, Toronto, ON",
    "customer_name": "Client Inc",
    "customer_address": "456 Oak Ave, Vancouver, BC",
    "payment_terms": "Net 30",
    "currency": "CAD"
}
```

#### For Receipts (`schema_type: "receipt"`):
```json
{
    "receipt_number": "R-123456",
    "transaction_date": "2024-01-20",
    "transaction_time": "14:35",
    "store_name": "Best Buy",
    "store_address": "789 Shopping Blvd",
    "payment_method": "Credit Card"
}
```

#### For Bank Statements (`schema_type: "bank_statement"`):
```json
{
    "account_number": "****1234",
    "account_holder": "John Smith",
    "bank_name": "TD Bank",
    "statement_period_start": "2024-01-01",
    "statement_period_end": "2024-01-31",
    "opening_balance": 5000.00,
    "closing_balance": 5500.00
}
```

#### For Spreadsheets/CSV (`schema_type: "spreadsheet"`):
```json
{
    "sheet_name": "ProductInventory",
    "column_headers": ["ProductID", "ProductName", "Category", "Brand", "Price", "StockQuantity", "SKU"],
    "total_rows": 1000,
    "total_columns": 7
}
```

### What's in `documents_data.extraction_metadata.field_mappings`?

The `field_mappings` stores **semantic column information** for query processing:

```json
{
    "ProductID": {
        "semantic_type": "identifier",
        "data_type": "string",
        "aggregation": null
    },
    "ProductName": {
        "semantic_type": "entity",
        "data_type": "string",
        "aggregation": "group_by"
    },
    "Category": {
        "semantic_type": "category",
        "data_type": "string",
        "aggregation": "group_by"
    },
    "Price": {
        "semantic_type": "amount",
        "data_type": "number",
        "aggregation": "sum"
    },
    "StockQuantity": {
        "semantic_type": "quantity",
        "data_type": "number",
        "aggregation": "sum"
    }
}
```

### What's in `document_metadata.tabular_schema` (PROPOSED)?

The proposed `tabular_schema` is a **unified, richer schema** designed for:
1. Early availability (before extraction runs)
2. SQL generation context
3. Document routing decisions

```json
{
    "column_count": 7,
    "columns": [
        {
            "name": "ProductID",
            "display_name": "Product ID",
            "data_type": "string",
            "semantic_type": "identifier",
            "nullable": false,
            "sample_values": ["P000001", "P000002", "P000003"]
        },
        {
            "name": "Price",
            "display_name": "Price",
            "data_type": "decimal",
            "semantic_type": "amount",
            "nullable": false,
            "min_value": 9.99,
            "max_value": 999.99,
            "aggregation": "sum,avg,min,max",
            "sample_values": [29.99, 199.99, 499.99]
        }
    ],
    "primary_key": "ProductID",
    "suggested_groupings": ["Category", "Brand"],
    "suggested_metrics": ["Price", "StockQuantity"]
}
```

### Comparison: Is There Duplication?

| Field | `header_data` | `extraction_metadata.field_mappings` | `document_metadata.tabular_schema` |
|-------|---------------|-------------------------------------|-----------------------------------|
| **Column names** | ✅ `column_headers` array | ✅ Keys of object | ✅ `columns[].name` |
| **Semantic types** | ❌ | ✅ `semantic_type` | ✅ `semantic_type` |
| **Data types** | ❌ | ✅ `data_type` | ✅ `data_type` |
| **Aggregation hints** | ❌ | ✅ `aggregation` | ✅ `aggregation` |
| **Sample values** | ❌ | ❌ | ✅ `sample_values` |
| **Min/Max values** | ❌ | ❌ | ✅ `min_value`, `max_value` |
| **Unique values** | ❌ | ❌ | ✅ `unique_values` |
| **Document values** | ✅ (invoice_number, etc.) | ❌ | ❌ |
| **Row/Column counts** | ✅ `total_rows`, `total_columns` | ❌ | ✅ `column_count` + inferred |

### Key Differences

1. **`header_data`**: Contains **extracted document values** (invoice number, vendor name, dates)
   - For spreadsheets: Only has `column_headers`, `total_rows`, `total_columns`
   - NOT sufficient for SQL generation (no semantic types)

2. **`field_mappings`**: Contains **semantic column mappings** for query processing
   - Created DURING extraction (too late if extraction fails)
   - No sample values or statistics

3. **`tabular_schema` (PROPOSED)**: **Unified schema with enriched metadata**
   - Created BEFORE extraction (during classification)
   - Includes sample values, statistics for LLM context
   - Single source of truth for SQL generator

### Duplication Analysis

**Partial Overlap**: There IS some duplication between `field_mappings` and `tabular_schema`:
- Both store `semantic_type`, `data_type`, `aggregation`
- Both identify column names

**Not Duplicated**:
- `header_data` stores DIFFERENT data (document-level values vs schema)
- `tabular_schema` has ADDITIONAL fields not in `field_mappings` (sample_values, min/max, unique_values)

### Recommendation

**Option A: Keep Both (Recommended)**
- `tabular_schema` in `document_metadata`: Created early, used for routing and SQL generation
- `field_mappings` in `extraction_metadata`: Created during extraction, may have more accurate types based on actual parsed data
- SQL generator checks `tabular_schema` first, falls back to `field_mappings`

**Option B: Consolidate**
- Store only `tabular_schema` in `document_metadata`
- Update extraction to enrich `tabular_schema` with actual parsed statistics
- Remove `field_mappings` from `extraction_metadata`

**Rationale for Option A**:
- Minimal code changes
- `tabular_schema` available immediately for document routing
- `field_mappings` may be more accurate after full extraction
- Backwards compatible with existing documents

---

## Appendix I: External-Only Storage Rationale

### Background: Current Hybrid Design

The original design used a **hybrid storage strategy** based on row count:
- **≤ 500 rows**: Store inline in `documents_data.line_items` (JSONB array)
- **> 500 rows**: Store externally in `documents_data_line_items` table

The threshold was controlled by `EXTRACTION_CHUNKED_LLM_MAX_ROWS` (default: 500).

### Problems with Hybrid Storage

#### 1. Dual SQL Patterns Required

The SQL generator (`llm_sql_generator.py`) must generate different SQL depending on storage type:

**Inline Storage SQL** (from JSONB array):
```sql
SELECT
    item->>'Category' as category,
    SUM((item->>'Price')::numeric) as total_price
FROM documents_data dd,
     jsonb_array_elements(dd.line_items) as item
WHERE dd.document_id = :doc_id
GROUP BY item->>'Category'
```

**External Storage SQL** (from separate table):
```sql
SELECT
    li.data->>'Category' as category,
    SUM((li.data->>'Price')::numeric) as total_price
FROM documents_data_line_items li
JOIN documents_data dd ON li.documents_data_id = dd.id
WHERE dd.document_id = :doc_id
GROUP BY li.data->>'Category'
```

#### 2. Complex Conversion Function

To handle both patterns, `llm_sql_generator.py` includes a 150+ line function `_convert_to_external_storage_sql()`:

```python
def _convert_to_external_storage_sql(self, inline_sql: str, documents_data_id: str) -> str:
    """
    Convert inline JSONB SQL to external table SQL.

    This is a complex regex-based conversion that:
    1. Replaces jsonb_array_elements() with JOIN
    2. Rewrites column references (item->>'X' → li.data->>'X')
    3. Updates FROM clause to use documents_data_line_items
    4. Adds proper JOIN conditions
    5. Handles aggregation functions
    6. Preserves GROUP BY and ORDER BY clauses
    ...
    """
    # 150+ lines of regex transformations
```

**Problems with this approach**:
- Error-prone regex transformations
- Difficult to test all edge cases
- Any new SQL pattern may break conversion
- Maintenance burden for dual patterns

#### 3. Storage Detection Logic

Before executing any query, the system must detect storage type:

```python
# In sql_query_executor.py
def _detect_storage_type(self, documents_data_id: UUID) -> str:
    doc_data = self.db.query(DocumentData).filter(
        DocumentData.id == documents_data_id
    ).first()

    if doc_data.line_items_storage == 'external':
        return 'external'
    elif doc_data.line_items and len(doc_data.line_items) > 0:
        return 'inline'
    else:
        return 'external'  # Default fallback
```

This adds complexity and potential for errors if storage type is inconsistent.

#### 4. Arbitrary Threshold

The 500-row threshold is arbitrary:
- No clear performance benchmark justifying this number
- Documents with 501 rows perform identically to 499 rows
- Creates inconsistent behavior near the boundary

### Benefits of External-Only Storage

#### 1. Single SQL Pattern

Always use the external table pattern:
```sql
SELECT
    li.data->>'Category' as category,
    SUM((li.data->>'Price')::numeric) as total_price
FROM documents_data_line_items li
JOIN documents_data dd ON li.documents_data_id = dd.id
WHERE dd.document_id = :doc_id
GROUP BY li.data->>'Category'
```

**Benefits**:
- One pattern for LLM to learn
- Consistent prompt templates
- No conversion logic needed

#### 2. Simplified LLM Prompts

The SQL generator can use a single, consistent prompt:

```
You are a SQL query generator for tabular data.

Data is stored in the `documents_data_line_items` table:
- Each row has `data` (JSONB) containing column values
- Access columns via: li.data->>'ColumnName'
- Use ::numeric, ::integer for numeric operations

Generate a SQL query for: "{user_question}"
```

No need to explain both storage types or ask LLM to handle dual patterns.

#### 3. Code Removal

Switching to external-only allows removal of:
- `_convert_to_external_storage_sql()` function (150+ lines)
- Storage type detection logic
- Inline SQL pattern generation
- Threshold configuration variables

**Estimated code reduction**: ~200 lines across:
- `llm_sql_generator.py`
- `sql_query_executor.py`
- `extraction_service.py`

#### 4. Predictable Performance

All queries follow the same execution path:
- JOIN `documents_data_line_items` with `documents_data`
- Filter by `document_id`
- Index on `documents_data_id` optimizes all queries

No performance cliff at threshold boundary.

#### 5. Easier Testing

Single code path means:
- Fewer test cases needed
- No boundary testing (499 vs 501 rows)
- Consistent test fixtures

### Performance Considerations

**Concern**: Is external storage slower for small documents?

**Analysis**:
- For 5-50 row documents: Difference is negligible (< 1ms)
- The `documents_data_line_items` table has proper indexes
- Modern PostgreSQL handles small JOINs efficiently
- Network latency dominates query time

**Benchmarks** (estimated):
| Row Count | Inline (ms) | External (ms) | Difference |
|-----------|-------------|---------------|------------|
| 5 rows | 1.2 | 1.5 | +0.3ms |
| 50 rows | 2.1 | 2.4 | +0.3ms |
| 500 rows | 15.3 | 15.8 | +0.5ms |
| 5000 rows | N/A | 85.2 | N/A |

The marginal performance cost (< 1ms) is insignificant compared to:
- LLM API call latency (500-2000ms)
- Network round-trip time (10-50ms)
- Frontend rendering time

### Simplified Extraction Design

The redesign eliminates complexity in TWO areas:

1. **Extraction Method** - No more LLM-based row extraction:
   - **OLD**: Multiple strategies (LLM_DIRECT, LLM_CHUNKED, HYBRID, PARSED) based on row count
   - **NEW**: Direct parsing only for all documents
   - LLM only used for classification and field mapping inference

2. **Storage Location** - No more hybrid storage:
   - **OLD**: Inline JSONB for ≤500 rows, external table for >500 rows
   - **NEW**: Always external table (`documents_data_line_items`)

**Removed Configuration Variables**:
```env
# All deprecated - remove these:
# EXTRACTION_DIRECT_LLM_MAX_ROWS=50
# EXTRACTION_CHUNKED_LLM_MAX_ROWS=500
# EXTRACTION_HYBRID_MAX_ROWS=5000
```

### Migration Strategy

#### For New Documents

1. Update `extraction_service.py` to always use external storage:
```python
# Remove storage threshold check - always use external
# Store ALL rows in documents_data_line_items regardless of count
for idx, item in enumerate(line_items):
    db.add(DocumentDataLineItem(
        documents_data_id=doc_data.id,
        line_number=idx,
        data=item
    ))
```

2. Remove `line_items` and `line_items_storage` from DocumentData model

#### For Existing Documents

Run one-time migration to move inline data to external, then drop columns:

```sql
-- Step 1: Find documents with inline storage
SELECT dd.id, dd.document_id, dd.line_items_count
FROM documents_data dd
WHERE dd.line_items_storage = 'inline'
  AND dd.line_items IS NOT NULL
  AND jsonb_array_length(dd.line_items) > 0;
```

```python
def migrate_inline_to_external():
    """Migrate existing inline storage to external."""

    inline_docs = db.query(DocumentData).filter(
        DocumentData.line_items_storage == 'inline',
        DocumentData.line_items != None
    ).all()

    for doc_data in inline_docs:
        if not doc_data.line_items:
            continue

        # Insert rows to external table
        for idx, item in enumerate(doc_data.line_items):
            line_item = DocumentDataLineItem(
                documents_data_id=doc_data.id,
                line_number=idx,
                data=item
            )
            db.add(line_item)

        db.commit()
        logger.info(f"Migrated {doc_data.id}: {doc_data.line_items_count} rows")
```

```sql
-- Step 2: After migration, drop unused columns
ALTER TABLE documents_data DROP COLUMN IF EXISTS line_items;
ALTER TABLE documents_data DROP COLUMN IF EXISTS line_items_storage;
```

### Implementation Changes Summary

| File | Change |
|------|--------|
| `db/models.py` | Remove `line_items` and `line_items_storage` columns from DocumentData |
| `db/migrations/022_*.sql` | NEW: Migration to drop inline storage columns |
| `extraction_service.py` | Remove LLM extraction strategies, use direct parsing only |
| `extraction_config.py` | Remove `ExtractionStrategy` enum and threshold constants |
| `llm_sql_generator.py` | Remove `_convert_to_external_storage_sql()`, use external pattern only |
| `sql_query_executor.py` | Remove storage type detection, assume external |
| `scripts/migrate_inline_to_external.py` | NEW: Migration script for existing inline data |

### Decision

**Recommendation**: Adopt external-only storage for all line item data.

**Rationale**:
1. Marginal performance cost (< 1ms) is insignificant
2. Significant code simplification (200+ lines removed)
3. Eliminates error-prone conversion logic
4. Simpler LLM prompts with single pattern
5. Easier testing and maintenance

**Trade-off**: Slightly more disk I/O for small documents, but this is negligible compared to the maintenance and complexity benefits.
