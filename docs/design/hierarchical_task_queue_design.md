# Hierarchical Task Queue Design (v2)

## Overview

This document outlines a redesigned task queue system with:
- **Three-level hierarchy**: Document → Page → Chunk
- **Three processing phases**: OCR → Vector Index → GraphRAG Index
- **Sequential dependencies**: Each phase depends on the previous phase completing

## Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROCESSING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Phase 1: OCR              Phase 2: Vector Index      Phase 3: GraphRAG    │
│   ───────────────           ──────────────────────     ─────────────────    │
│   PDF → Markdown            Markdown → Embeddings      Chunks → Entities    │
│   (per page)                (per chunk)                (per chunk)          │
│                                                                              │
│   ┌──────────┐              ┌──────────┐              ┌──────────┐          │
│   │ Document │──depends──→  │ Document │──depends──→  │ Document │          │
│   │ OCR      │              │ Vector   │              │ GraphRAG │          │
│   └────┬─────┘              └────┬─────┘              └────┬─────┘          │
│        │                         │                         │                │
│   ┌────▼─────┐              ┌────▼─────┐              ┌────▼─────┐          │
│   │ Page OCR │              │ Page     │              │ Page     │          │
│   │ (N pages)│              │ Vector   │              │ GraphRAG │          │
│   └──────────┘              └────┬─────┘              └────┬─────┘          │
│                                  │                         │                │
│                             ┌────▼─────┐              ┌────▼─────┐          │
│                             │ Chunk    │              │ Chunk    │          │
│                             │ Vector   │              │ GraphRAG │          │
│                             │ (M each) │              │ (M each) │          │
│                             └──────────┘              └──────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Database Schema

### 1. Status Enum (Shared across all tables)

```sql
CREATE TYPE task_status AS ENUM ('pending', 'processing', 'completed', 'failed');
```

### 2. Documents Table (Existing - Add 3 status columns)

```sql
-- Add three separate status columns for each processing phase
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS ocr_status task_status DEFAULT 'pending',
ADD COLUMN IF NOT EXISTS vector_index_status task_status DEFAULT 'pending',
ADD COLUMN IF NOT EXISTS graphrag_index_status task_status DEFAULT 'pending';

-- Add timestamps for each phase
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS ocr_started_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS ocr_completed_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS ocr_error TEXT,
ADD COLUMN IF NOT EXISTS vector_started_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS vector_completed_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS vector_error TEXT,
ADD COLUMN IF NOT EXISTS graphrag_started_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS graphrag_completed_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS graphrag_error TEXT;

-- Worker tracking (which worker is processing this document)
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS current_worker_id VARCHAR(100),
ADD COLUMN IF NOT EXISTS last_heartbeat TIMESTAMP WITH TIME ZONE;
```

**Document Status Semantics:**

| Column | Meaning |
|--------|---------|
| `ocr_status` | Status of OCR processing (PDF → Markdown) |
| `vector_index_status` | Status of Vector indexing (Qdrant) |
| `graphrag_index_status` | Status of GraphRAG indexing (Neo4j) |

### 3. Task Queue Page Table (New)

Each page has THREE status columns - one for each processing phase:

```sql
CREATE TABLE task_queue_page (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Page identification
    page_number INTEGER NOT NULL,
    page_file_path VARCHAR(1000),  -- Path to _nohf.md file after OCR

    -- === THREE STATUS COLUMNS ===
    -- Phase 1: OCR (PDF page → Markdown)
    ocr_status task_status NOT NULL DEFAULT 'pending',
    ocr_worker_id VARCHAR(100),
    ocr_started_at TIMESTAMP WITH TIME ZONE,
    ocr_completed_at TIMESTAMP WITH TIME ZONE,
    ocr_error TEXT,
    ocr_retry_count INTEGER DEFAULT 0,
    ocr_last_heartbeat TIMESTAMP WITH TIME ZONE,

    -- Phase 2: Vector Index (Markdown → Qdrant)
    -- NOTE: Stays 'pending' until ocr_status = 'completed'
    vector_status task_status NOT NULL DEFAULT 'pending',
    vector_worker_id VARCHAR(100),
    vector_started_at TIMESTAMP WITH TIME ZONE,
    vector_completed_at TIMESTAMP WITH TIME ZONE,
    vector_error TEXT,
    vector_retry_count INTEGER DEFAULT 0,
    vector_last_heartbeat TIMESTAMP WITH TIME ZONE,

    -- Phase 3: GraphRAG Index (Chunks → Neo4j entities)
    -- NOTE: Stays 'pending' until vector_status = 'completed'
    graphrag_status task_status NOT NULL DEFAULT 'pending',
    graphrag_worker_id VARCHAR(100),
    graphrag_started_at TIMESTAMP WITH TIME ZONE,
    graphrag_completed_at TIMESTAMP WITH TIME ZONE,
    graphrag_error TEXT,
    graphrag_retry_count INTEGER DEFAULT 0,
    graphrag_last_heartbeat TIMESTAMP WITH TIME ZONE,

    -- Common metadata
    chunk_count INTEGER DEFAULT 0,  -- Number of chunks from this page
    max_retries INTEGER DEFAULT 3,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Ensure unique page per document
    UNIQUE(document_id, page_number)
);

-- Indexes for efficient task pickup
CREATE INDEX idx_tqp_ocr_status ON task_queue_page(ocr_status) WHERE ocr_status IN ('pending', 'failed');
CREATE INDEX idx_tqp_vector_status ON task_queue_page(vector_status) WHERE vector_status IN ('pending', 'failed');
CREATE INDEX idx_tqp_graphrag_status ON task_queue_page(graphrag_status) WHERE graphrag_status IN ('pending', 'failed');
CREATE INDEX idx_tqp_document ON task_queue_page(document_id);
CREATE INDEX idx_tqp_ocr_heartbeat ON task_queue_page(ocr_last_heartbeat) WHERE ocr_status = 'processing';
CREATE INDEX idx_tqp_vector_heartbeat ON task_queue_page(vector_last_heartbeat) WHERE vector_status = 'processing';
CREATE INDEX idx_tqp_graphrag_heartbeat ON task_queue_page(graphrag_last_heartbeat) WHERE graphrag_status = 'processing';
```

### 4. Task Queue Chunk Table (New)

Chunks only need TWO status columns (Vector and GraphRAG - no OCR at chunk level):

```sql
CREATE TABLE task_queue_chunk (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    page_id UUID NOT NULL REFERENCES task_queue_page(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Chunk identification
    chunk_id VARCHAR(64) NOT NULL,    -- External chunk ID (used in Qdrant)
    chunk_index INTEGER NOT NULL,      -- Position within page (0-indexed)
    chunk_content_hash VARCHAR(64),    -- For deduplication

    -- === TWO STATUS COLUMNS (no OCR at chunk level) ===
    -- Phase 2: Vector Index (Chunk → Qdrant embedding)
    vector_status task_status NOT NULL DEFAULT 'pending',
    vector_worker_id VARCHAR(100),
    vector_started_at TIMESTAMP WITH TIME ZONE,
    vector_completed_at TIMESTAMP WITH TIME ZONE,
    vector_error TEXT,
    vector_retry_count INTEGER DEFAULT 0,
    vector_last_heartbeat TIMESTAMP WITH TIME ZONE,

    -- Phase 3: GraphRAG Index (Chunk → Neo4j entities)
    -- NOTE: Stays 'pending' until vector_status = 'completed'
    graphrag_status task_status NOT NULL DEFAULT 'pending',
    graphrag_worker_id VARCHAR(100),
    graphrag_started_at TIMESTAMP WITH TIME ZONE,
    graphrag_completed_at TIMESTAMP WITH TIME ZONE,
    graphrag_error TEXT,
    graphrag_retry_count INTEGER DEFAULT 0,
    graphrag_last_heartbeat TIMESTAMP WITH TIME ZONE,

    -- Processing results
    entities_extracted INTEGER DEFAULT 0,
    relationships_extracted INTEGER DEFAULT 0,

    -- Common metadata
    max_retries INTEGER DEFAULT 3,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Ensure unique chunk per page
    UNIQUE(page_id, chunk_index)
);

-- Indexes for efficient task pickup
CREATE INDEX idx_tqc_vector_status ON task_queue_chunk(vector_status) WHERE vector_status IN ('pending', 'failed');
CREATE INDEX idx_tqc_graphrag_status ON task_queue_chunk(graphrag_status) WHERE graphrag_status IN ('pending', 'failed');
CREATE INDEX idx_tqc_page ON task_queue_chunk(page_id);
CREATE INDEX idx_tqc_document ON task_queue_chunk(document_id);
CREATE INDEX idx_tqc_vector_heartbeat ON task_queue_chunk(vector_last_heartbeat) WHERE vector_status = 'processing';
CREATE INDEX idx_tqc_graphrag_heartbeat ON task_queue_chunk(graphrag_last_heartbeat) WHERE graphrag_status = 'processing';
```

## Status Rules

### 1. Four Simple Statuses (Same for all phases)

| Status | Meaning |
|--------|---------|
| `pending` | Waiting to be picked up by a worker |
| `processing` | Worker is actively processing |
| `completed` | Successfully finished |
| `failed` | Failed after max retries |

### 2. Sequential Dependencies

```
OCR completed → enables → Vector Index
Vector Index completed → enables → GraphRAG Index
```

**Dependency Rules:**
- `vector_status` can only transition from `pending` → `processing` when `ocr_status = 'completed'`
- `graphrag_status` can only transition from `pending` → `processing` when `vector_status = 'completed'`

### 3. Status Bubble-Up Logic (Child → Parent)

For each phase separately, parent status is computed from children:

```
Parent.{phase}_status = f(Children.{phase}_status)

Rules (in priority order):
1. If ANY child is "processing" → parent = "processing"
2. If ANY child is "pending"    → parent = "pending"
3. If ALL children "completed"  → parent = "completed"
4. If ANY child "failed" (and none pending/processing) → parent = "failed"
```

**Example: OCR Status Bubble-Up**
```
Document (3 pages):
  - Page 1: ocr_status = completed
  - Page 2: ocr_status = processing
  - Page 3: ocr_status = pending

  → Document.ocr_status = "processing" (Rule 1)

After all pages complete:
  - Page 1: ocr_status = completed
  - Page 2: ocr_status = completed
  - Page 3: ocr_status = completed

  → Document.ocr_status = "completed" (Rule 3)
  → Document.vector_index_status can now be picked up!
```

## Task Pickup Priority Strategy

### Priority Order (within each phase)

```
For OCR Phase:
  Priority 1: Failed pages (OCR) - retry first
  Priority 2: Pending pages (OCR) - new work

For Vector Index Phase (only when OCR completed):
  Priority 1: Failed chunks (Vector) - retry first
  Priority 2: Pending chunks (Vector) - new work
  Priority 3: Failed pages (Vector) - if no chunks
  Priority 4: Pending pages (Vector) - if no chunks

For GraphRAG Phase (only when Vector completed):
  Priority 1: Failed chunks (GraphRAG) - retry first
  Priority 2: Pending chunks (GraphRAG) - new work
  Priority 3: Failed pages (GraphRAG) - if no chunks
  Priority 4: Pending pages (GraphRAG) - if no chunks
```

### SQL Queries for Task Pickup

#### Pick up OCR Task (Page Level)
```sql
-- Get next OCR page task
SELECT id, document_id, page_number
FROM task_queue_page
WHERE ocr_status IN ('pending', 'failed')
  AND ocr_retry_count < max_retries
ORDER BY
  CASE WHEN ocr_status = 'failed' THEN 0 ELSE 1 END,  -- Failed first
  ocr_retry_count ASC,
  created_at ASC
LIMIT 1
FOR UPDATE SKIP LOCKED;
```

#### Pick up Vector Index Task (Chunk Level)
```sql
-- Get next Vector chunk task (only where parent page OCR is completed)
SELECT c.id, c.page_id, c.document_id, c.chunk_id, c.chunk_index
FROM task_queue_chunk c
JOIN task_queue_page p ON c.page_id = p.id
WHERE c.vector_status IN ('pending', 'failed')
  AND c.vector_retry_count < c.max_retries
  AND p.ocr_status = 'completed'  -- Dependency check!
ORDER BY
  CASE WHEN c.vector_status = 'failed' THEN 0 ELSE 1 END,
  c.vector_retry_count ASC,
  c.created_at ASC
LIMIT 1
FOR UPDATE SKIP LOCKED;
```

#### Pick up GraphRAG Task (Chunk Level)
```sql
-- Get next GraphRAG chunk task (only where Vector is completed)
SELECT c.id, c.page_id, c.document_id, c.chunk_id, c.chunk_index
FROM task_queue_chunk c
WHERE c.graphrag_status IN ('pending', 'failed')
  AND c.graphrag_retry_count < c.max_retries
  AND c.vector_status = 'completed'  -- Dependency check!
ORDER BY
  CASE WHEN c.graphrag_status = 'failed' THEN 0 ELSE 1 END,
  c.graphrag_retry_count ASC,
  c.created_at ASC
LIMIT 1
FOR UPDATE SKIP LOCKED;
```

## Processing Flow

### Phase 1: OCR Processing

```
1. Document uploaded:
   - documents.ocr_status = 'pending'
   - documents.vector_index_status = 'pending'
   - documents.graphrag_index_status = 'pending'

2. Worker claims document for OCR:
   - documents.ocr_status = 'processing'
   - Convert PDF to images
   - Determine total_pages
   - Create task_queue_page records for each page:
     * ocr_status = 'pending'
     * vector_status = 'pending'
     * graphrag_status = 'pending'

3. Workers pick up page OCR tasks:
   - Claim page: ocr_status = 'processing', ocr_worker_id = X
   - Run OCR inference on page image
   - Save markdown to page_file_path
   - Mark page: ocr_status = 'completed' or 'failed'
   - Trigger bubble-up to document

4. When all pages OCR complete:
   - Document.ocr_status = 'completed'
   - Vector indexing can now begin!
```

### Phase 2: Vector Index Processing

```
1. Document OCR completed → Vector index can start

2. Worker claims document for Vector indexing:
   - documents.vector_index_status = 'processing'
   - Read all page markdown files
   - Chunk content using adaptive chunker
   - Create task_queue_chunk records for each chunk:
     * vector_status = 'pending'
     * graphrag_status = 'pending'
   - Update page.chunk_count

3. Workers pick up chunk Vector tasks:
   - Claim chunk: vector_status = 'processing'
   - Generate embedding via embedding service
   - Store in Qdrant
   - Mark chunk: vector_status = 'completed' or 'failed'
   - Trigger bubble-up: chunk → page → document

4. When all chunks Vector indexed:
   - Document.vector_index_status = 'completed'
   - GraphRAG indexing can now begin!
```

### Phase 3: GraphRAG Index Processing

```
1. Document Vector completed → GraphRAG can start

2. Workers pick up chunk GraphRAG tasks:
   - Claim chunk: graphrag_status = 'processing'
   - Extract entities and relationships via LLM
   - Store in Neo4j
   - Mark chunk: graphrag_status = 'completed' or 'failed'
   - Record: entities_extracted, relationships_extracted
   - Trigger bubble-up: chunk → page → document

3. When all chunks GraphRAG indexed:
   - Document.graphrag_index_status = 'completed'
   - Document is fully processed!
```

## Heartbeat & Stale Detection

### Heartbeat Update
Workers update heartbeat every 30 seconds while processing:

```sql
-- Update heartbeat for OCR task
UPDATE task_queue_page
SET ocr_last_heartbeat = NOW()
WHERE id = :page_id AND ocr_worker_id = :worker_id;

-- Update heartbeat for Vector chunk task
UPDATE task_queue_chunk
SET vector_last_heartbeat = NOW()
WHERE id = :chunk_id AND vector_worker_id = :worker_id;
```

### Stale Task Detection (Scheduler runs every 60s)

```sql
-- Release stale OCR tasks (no heartbeat for > 2 minutes)
UPDATE task_queue_page
SET ocr_status = 'pending',
    ocr_worker_id = NULL,
    ocr_started_at = NULL,
    ocr_last_heartbeat = NULL
WHERE ocr_status = 'processing'
  AND ocr_last_heartbeat < NOW() - INTERVAL '2 minutes';

-- Release stale Vector tasks
UPDATE task_queue_chunk
SET vector_status = 'pending',
    vector_worker_id = NULL,
    vector_started_at = NULL,
    vector_last_heartbeat = NULL
WHERE vector_status = 'processing'
  AND vector_last_heartbeat < NOW() - INTERVAL '2 minutes';

-- Release stale GraphRAG tasks
UPDATE task_queue_chunk
SET graphrag_status = 'pending',
    graphrag_worker_id = NULL,
    graphrag_started_at = NULL,
    graphrag_last_heartbeat = NULL
WHERE graphrag_status = 'processing'
  AND graphrag_last_heartbeat < NOW() - INTERVAL '2 minutes';
```

## Migration Path

### Phase 1: Add New Tables (Non-breaking)
1. Create new enum type `task_status`
2. Add new columns to `documents` table
3. Create `task_queue_page` table
4. Create `task_queue_chunk` table
5. Keep existing `task_queue` table for backward compatibility

### Phase 2: Migrate Data
1. For documents with `convert_status = 'converted'`:
   - Set `ocr_status = 'completed'`
2. For documents with `index_status = 'indexed'`:
   - Set `vector_index_status = 'completed'`
   - Set `graphrag_index_status = 'completed'`
3. Create page/chunk records from existing `ocr_details` and `indexing_details`

### Phase 3: Update Workers
1. Implement new `HierarchicalTaskQueueManager`
2. Update OCR processor to create page tasks
3. Update indexer to create chunk tasks
4. Implement status bubble-up logic

### Phase 4: Remove Legacy
1. Deprecate old `task_queue` table
2. Optionally keep `ocr_details` and `indexing_details` for detailed metadata

## Example Queries

### Get Document Progress (All Phases)
```sql
SELECT
    d.id,
    d.filename,
    d.ocr_status,
    d.vector_index_status,
    d.graphrag_index_status,
    -- OCR progress
    COUNT(p.id) as total_pages,
    COUNT(p.id) FILTER (WHERE p.ocr_status = 'completed') as ocr_completed_pages,
    COUNT(p.id) FILTER (WHERE p.ocr_status = 'failed') as ocr_failed_pages,
    -- Vector progress
    SUM(p.chunk_count) as total_chunks,
    (SELECT COUNT(*) FROM task_queue_chunk c WHERE c.document_id = d.id AND c.vector_status = 'completed') as vector_completed_chunks,
    -- GraphRAG progress
    (SELECT COUNT(*) FROM task_queue_chunk c WHERE c.document_id = d.id AND c.graphrag_status = 'completed') as graphrag_completed_chunks
FROM documents d
LEFT JOIN task_queue_page p ON d.id = p.document_id
WHERE d.id = :doc_id
GROUP BY d.id;
```

### Get Queue Statistics
```sql
SELECT
    'ocr_pages' as task_type,
    ocr_status as status,
    COUNT(*) as count
FROM task_queue_page
GROUP BY ocr_status

UNION ALL

SELECT
    'vector_chunks' as task_type,
    vector_status as status,
    COUNT(*) as count
FROM task_queue_chunk
GROUP BY vector_status

UNION ALL

SELECT
    'graphrag_chunks' as task_type,
    graphrag_status as status,
    COUNT(*) as count
FROM task_queue_chunk
GROUP BY graphrag_status;
```

### Find Documents Ready for Next Phase
```sql
-- Documents ready for Vector indexing (OCR completed)
SELECT id, filename
FROM documents
WHERE ocr_status = 'completed'
  AND vector_index_status = 'pending';

-- Documents ready for GraphRAG indexing (Vector completed)
SELECT id, filename
FROM documents
WHERE vector_index_status = 'completed'
  AND graphrag_index_status = 'pending';
```

## Benefits

1. **Clear Phase Separation**: Each phase (OCR, Vector, GraphRAG) has its own status
2. **Sequential Dependencies**: Enforced at database level
3. **Granular Retry**: Retry only failed chunks/pages, not entire documents
4. **Queryable Task State**: SQL queries instead of JSONB parsing
5. **Better Progress Tracking**: Exact counts per phase per level
6. **Heartbeat per Phase**: Each phase has its own heartbeat for stale detection
7. **Parallel Processing**: Multiple workers can process different phases simultaneously

## Implementation Files

### New Files Created
- `db/migrations/012_hierarchical_task_queue.sql` - Database migration
- `queue_service/models.py` - SQLAlchemy models for TaskQueuePage, TaskQueueChunk
- `queue_service/hierarchical_task_manager.py` - Main task queue manager
- `queue_service/hierarchical_worker_pool.py` - Worker pool implementation
- `queue_service/scheduler.py` - Updated scheduler for stale detection

### Configuration (.env)
```
# Hierarchical Task Queue Settings
TASK_QUEUE_ENABLED=true
TASK_QUEUE_CHECK_INTERVAL=60    # Scheduler interval (seconds)
WORKER_POLL_INTERVAL=5          # Worker poll interval (seconds)
HEARTBEAT_INTERVAL=30           # Heartbeat update interval (seconds)
STALE_TASK_TIMEOUT=120          # Stale detection timeout (seconds)
MAX_TASK_RETRIES=3              # Max retries before permanent failure
```

### Removed Legacy Files
The following legacy files have been removed:
- `queue_service/task_queue_manager.py` - Old single-level task manager (removed)
- `queue_service/queue_worker_pool.py` - Old worker pool (removed)
- `queue_service/README.md` - Old documentation (removed)

Note: The old `task_queue` table still exists in the database but is no longer used.

## Running the Migration

```bash
# 1. Apply the migration
psql -U postgres -d dots_ocr -f backend/db/migrations/012_hierarchical_task_queue.sql

# 2. Verify tables created
psql -U postgres -d dots_ocr -c "\dt task_queue_*"

# 3. Check document status columns
psql -U postgres -d dots_ocr -c "\d documents" | grep -E "(ocr_status|vector_status|graphrag_status)"
```
