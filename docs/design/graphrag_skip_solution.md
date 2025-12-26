# GraphRAG Smart Skip Solution

## Executive Summary

This document describes the implementation plan for intelligently skipping GraphRAG indexing for document types that don't benefit from entity/relationship extraction (tabular data, spreadsheets, invoices with line items, etc.) while ensuring these documents still appear as "completed" in the overall processing status.

---

## 1. Problem Statement

### Current Behavior
- When `GRAPH_RAG_INDEX_ENABLED=true`, ALL chunks are processed through GraphRAG
- This wastes resources on tabular/list content that produces noisy, low-value entities
- Excel files, CSVs, invoices, and receipts generate thousands of meaningless entities

### Desired Behavior
- Intelligently skip GraphRAG for content that doesn't benefit from entity extraction
- Documents with skipped GraphRAG should still show as "Ready/Completed"
- Maintain consistency with existing document type definitions

---

## 2. Existing Document Type System (Source of Truth)

### 2.1 Primary Document Types (`rag_service/chunking/document_types.py`)

The codebase defines **27 core document types** organized by category:

```python
DOCUMENT_TYPES = {
    # Business & Finance
    "receipt": "meals, restaurant bills, purchases, transactions, expenses",
    "invoice": "product purchases, orders, billing documents, vendor invoices",
    "financial_report": "financial statements, earnings reports, budget reports",
    "bank_statement": "bank account statements, transaction history",
    "expense_report": "expense claims, reimbursement requests",

    # Legal & Compliance
    "contract": "legal contracts, agreements, binding documents",
    "legal_brief": "legal arguments, court briefs, legal opinions",
    "policy": "terms of service, privacy policy, compliance documents",

    # Technical & Engineering
    "technical_doc": "API docs, specifications, technical references",
    "user_manual": "user guides, how-to documents, instructions",
    "datasheet": "product datasheets, component specifications",

    # Academic & Research
    "research_paper": "academic papers, research articles",
    "thesis": "dissertations, thesis documents",
    "case_study": "case studies, analysis of specific examples",

    # Medical & Healthcare
    "medical_record": "patient records, medical history",
    "clinical_report": "clinical reports, lab results",

    # Education & Training
    "course_material": "textbooks, course content",
    "tutorial": "step-by-step tutorials, learning guides",

    # Government & Public
    "government_form": "official forms, permits, licenses",
    "legislation": "laws, regulations, legislative documents",

    # Professional & HR
    "resume": "CV, job applications, career history",
    "job_description": "job postings, role descriptions",

    # Creative & Media
    "article": "news articles, blog posts, written content",
    "presentation": "slide decks, presentations",

    # General
    "report": "general reports, analysis documents",
    "memo": "internal memos, communications",
    "letter": "formal letters, correspondence",
    "email": "email messages",
    "meeting_notes": "meeting minutes, notes",
    "other": "uncategorized documents"
}
```

### 2.2 Extended Document Types (`rag_service/chunking/domain_patterns.py`)

The `DocumentType` enum provides **54 detailed types** for finer classification:

```python
class DocumentType(Enum):
    # Business & Finance (8 types)
    RECEIPT, INVOICE, FINANCIAL_REPORT, BANK_STATEMENT,
    TAX_DOCUMENT, EXPENSE_REPORT, PURCHASE_ORDER, QUOTATION

    # Legal & Compliance (9 types)
    CONTRACT, AGREEMENT, LEGAL_BRIEF, COURT_FILING, PATENT,
    REGULATORY_FILING, TERMS_OF_SERVICE, PRIVACY_POLICY, COMPLIANCE_DOC

    # Technical & Engineering (8 types)
    TECHNICAL_SPEC, API_DOCUMENTATION, USER_MANUAL, INSTALLATION_GUIDE,
    ARCHITECTURE_DOC, DATASHEET, SCHEMATIC, CODE_DOCUMENTATION

    # ... and more
```

### 2.3 Extraction-Eligible Types (`extraction_service/extraction_config.py`)

Already defines which document types support structured data extraction:

```python
EXTRACTABLE_DOCUMENT_TYPES = {
    # Financial - these have tabular data
    "invoice": "invoice_schema",
    "receipt": "receipt_schema",
    "bank_statement": "bank_statement_schema",
    "expense_report": "expense_schema",
    "purchase_order": "purchase_order_schema",

    # Logistics - shipping manifests, etc.
    "shipping_document": "shipping_schema",
    "bill_of_lading": "shipping_schema",

    # Spreadsheet types
    "spreadsheet": "generic_tabular_schema",
    "excel": "generic_tabular_schema",
    "csv": "generic_tabular_schema",
}
```

---

## 3. GraphRAG Skip Classification

### 3.1 Skip Categories

Based on the existing document type system, define which types should skip GraphRAG:

```python
# File: backend/rag_service/graphrag_skip_config.py

from rag_service.chunking.domain_patterns import DocumentType

class GraphRAGSkipReason:
    """Reasons for skipping GraphRAG indexing."""
    FILE_TYPE = "file_type"           # Based on file extension
    DOCUMENT_TYPE = "document_type"   # Based on extracted document type
    CONTENT_PATTERN = "content_pattern"  # Based on content analysis
    USER_DISABLED = "user_disabled"   # Manually disabled by user


# File extensions that ALWAYS skip GraphRAG (inherently tabular/structured)
GRAPHRAG_SKIP_FILE_EXTENSIONS = {
    # Spreadsheet formats
    '.xlsx', '.xls', '.xlsm', '.xlsb',
    # Delimited data
    '.csv', '.tsv',
    # Structured data
    '.json', '.xml', '.yaml', '.yml',
    # Technical/logs
    '.log', '.sql',
}


# Document types that should skip GraphRAG (aligned with existing definitions)
# These are types where entity extraction provides little value
GRAPHRAG_SKIP_DOCUMENT_TYPES = {
    # ===== Financial/Transactional (tabular line items) =====
    'invoice',           # Line items: product, qty, price
    'receipt',           # Transaction items, amounts
    'bank_statement',    # Transaction rows: date, desc, amount
    'expense_report',    # Expense line items
    'purchase_order',    # Order line items
    'quotation',         # Price quotations
    'tax_document',      # Tax forms, numeric data

    # ===== Spreadsheet/Data Types =====
    'spreadsheet',       # Generic spreadsheet
    'excel',             # Excel workbook
    'csv',               # CSV data
    'worksheet',         # Spreadsheet worksheet
    'data_export',       # Exported data tables
    'database_export',   # Database dumps

    # ===== Logistics (manifests, lists) =====
    'shipping_document', # Shipping manifests
    'shipping_manifest', # Item lists
    'bill_of_lading',    # Cargo lists
    'customs_declaration', # Item declarations
    'delivery_note',     # Delivery item lists
    'packing_list',      # Package contents

    # ===== Inventory/Stock =====
    'inventory_report',  # Stock lists
    'stock_report',      # Inventory counts
    'price_list',        # Product/price tables
    'catalog',           # Product catalogs

    # ===== Financial Statements (numeric tables) =====
    'financial_report',  # Financial tables
    'balance_sheet',     # Accounting tables
    'income_statement',  # P&L tables
    'cash_flow',         # Cash flow tables
    'ledger',            # Accounting ledger
    'journal_entry',     # Accounting entries

    # ===== Forms (structured fields) =====
    'government_form',   # Form fields, not narrative
    'application_form',  # Application fields
    'registration_form', # Registration data

    # ===== Technical Data =====
    'datasheet',         # Specification tables
    'schematic',         # Diagrams, not text
    'log_file',          # Log entries
}


# Document types that SHOULD be processed by GraphRAG (narrative content)
GRAPHRAG_PROCESS_DOCUMENT_TYPES = {
    # ===== Legal (rich in entities/relationships) =====
    'contract',          # Parties, terms, obligations
    'agreement',         # Parties, conditions
    'legal_brief',       # Cases, arguments, citations
    'court_filing',      # Parties, claims
    'patent',            # Inventors, claims, references
    'policy',            # Rules, conditions
    'terms_of_service',  # Terms, conditions
    'privacy_policy',    # Data handling policies
    'compliance_doc',    # Regulations, requirements

    # ===== Technical Narrative =====
    'technical_doc',     # Concepts, relationships
    'user_manual',       # Procedures, components
    'api_documentation', # Endpoints, parameters
    'architecture_doc',  # Components, interactions
    'installation_guide', # Steps, requirements

    # ===== Academic/Research =====
    'research_paper',    # Authors, citations, findings
    'thesis',            # Arguments, references
    'case_study',        # Subjects, outcomes
    'academic_article',  # Authors, findings

    # ===== Medical =====
    'medical_record',    # Patients, conditions, treatments
    'clinical_report',   # Findings, diagnoses

    # ===== Professional =====
    'resume',            # Person, skills, experience
    'job_description',   # Role, requirements

    # ===== Creative/Narrative =====
    'article',           # Topics, people, events
    'report',            # Findings, recommendations
    'memo',              # Topics, actions
    'letter',            # Parties, topics
    'meeting_notes',     # Attendees, decisions
    'presentation',      # Topics, key points
    'book_chapter',      # Narrative content
}


def should_skip_graphrag_for_document_type(document_type: str) -> tuple[bool, str]:
    """
    Determine if a document type should skip GraphRAG.

    Args:
        document_type: The document type string (lowercase)

    Returns:
        (should_skip: bool, reason: str or None)
    """
    if not document_type:
        return False, None

    doc_type_lower = document_type.lower().strip()

    # Check against skip list
    if doc_type_lower in GRAPHRAG_SKIP_DOCUMENT_TYPES:
        return True, f"document_type:{doc_type_lower}"

    # Check aliases and variations
    skip_aliases = {
        'invoice': ['bill', 'billing'],
        'receipt': ['meal_receipt', 'expense_receipt', 'purchase_receipt'],
        'bank_statement': ['account_statement', 'statement'],
        'spreadsheet': ['workbook', 'worksheet', 'table_data'],
        'inventory_report': ['inventory', 'stock_list'],
    }

    for skip_type, aliases in skip_aliases.items():
        if doc_type_lower in aliases:
            return True, f"document_type:{skip_type}"

    return False, None


def should_skip_graphrag_for_file(filename: str) -> tuple[bool, str]:
    """
    Determine if a file should skip GraphRAG based on extension.

    Args:
        filename: The filename with extension

    Returns:
        (should_skip: bool, reason: str or None)
    """
    from pathlib import Path
    ext = Path(filename).suffix.lower()

    if ext in GRAPHRAG_SKIP_FILE_EXTENSIONS:
        return True, f"file_type:{ext}"

    return False, None
```

### 3.2 Content Pattern Detection (Optional, for Mixed Documents)

For documents like PDFs that may contain both narrative and tabular content:

```python
# File: backend/rag_service/graphrag_skip_config.py (continued)

import re

def is_tabular_chunk_content(content: str) -> tuple[bool, str]:
    """
    Detect if chunk content is primarily tabular/list data.

    Used for mixed documents (e.g., PDF with embedded tables).

    Args:
        content: The chunk text content

    Returns:
        (is_tabular: bool, reason: str or None)
    """
    if not content or len(content) < 50:
        return False, None

    lines = content.strip().split('\n')
    if len(lines) < 3:
        return False, None

    # Check 1: Markdown table pattern
    pipe_lines = [l for l in lines if l.count('|') >= 2]
    separator_lines = [l for l in lines if re.match(r'^[\s|:-]+$', l)]
    if len(pipe_lines) >= 3 and len(separator_lines) >= 1:
        return True, "markdown_table"

    # Check 2: High numeric density (>40% numeric characters)
    content_no_space = re.sub(r'\s', '', content)
    if len(content_no_space) > 0:
        numeric_chars = len(re.findall(r'[\d.,]', content))
        if numeric_chars / len(content_no_space) > 0.4:
            return True, "high_numeric_density"

    # Check 3: Line item pattern (description + number + currency)
    line_item_pattern = r'.{5,50}\s+\d+\s*[xX×]?\s*[\$€£¥]?\d+[.,]\d{2}'
    matches = re.findall(line_item_pattern, content)
    if len(matches) >= 3:
        return True, "line_items_detected"

    # Check 4: Transaction pattern (date + description + amount)
    transaction_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}.{10,50}[\$€£¥]?\d+[.,]\d{2}'
    matches = re.findall(transaction_pattern, content)
    if len(matches) >= 3:
        return True, "transaction_list"

    # Check 5: Short repetitive lines (typical of tables)
    non_empty_lines = [l.strip() for l in lines if l.strip()]
    if len(non_empty_lines) >= 8:
        avg_length = sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
        if avg_length < 40:
            return True, "short_repetitive_lines"

    return False, None
```

---

## 4. Database Changes

### 4.1 Add SKIPPED Status to TaskStatus Enum

```sql
-- Migration: 019_add_skipped_task_status.sql

-- Add 'skipped' value to task_status enum
ALTER TYPE task_status ADD VALUE IF NOT EXISTS 'skipped' AFTER 'failed';

-- Add skip tracking columns to documents table
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS skip_graphrag BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS skip_graphrag_reason VARCHAR(100) DEFAULT NULL;

-- Add skip reason to chunk table
ALTER TABLE task_queue_chunk
ADD COLUMN IF NOT EXISTS graphrag_skip_reason VARCHAR(50) DEFAULT NULL;

-- Index for querying skipped documents
CREATE INDEX IF NOT EXISTS idx_documents_skip_graphrag
ON documents (skip_graphrag) WHERE skip_graphrag = TRUE;

COMMENT ON COLUMN documents.skip_graphrag IS 'Whether to skip GraphRAG indexing for this document';
COMMENT ON COLUMN documents.skip_graphrag_reason IS 'Reason for skipping: file_type:xxx, document_type:xxx';
COMMENT ON COLUMN task_queue_chunk.graphrag_skip_reason IS 'Reason chunk was skipped for GraphRAG';
```

### 4.2 Update Models

```python
# File: backend/queue_service/models.py

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # NEW: Intentionally not processed

# Define which statuses count as "done" for rollup
DONE_STATUSES = {TaskStatus.COMPLETED, TaskStatus.SKIPPED}
```

```python
# File: backend/db/models.py

class Document(Base):
    # ... existing columns ...

    # NEW: GraphRAG skip control
    skip_graphrag = Column(Boolean, default=False, nullable=False)
    skip_graphrag_reason = Column(String(100), nullable=True)
```

---

## 5. Backend Implementation

### 5.1 Update `hierarchical_task_manager.py`

#### 5.1.1 Import Skip Configuration

```python
# At top of file
from rag_service.graphrag_skip_config import (
    GRAPHRAG_SKIP_FILE_EXTENSIONS,
    GRAPHRAG_SKIP_DOCUMENT_TYPES,
    should_skip_graphrag_for_document_type,
    should_skip_graphrag_for_file,
)
from queue_service.models import DONE_STATUSES
```

#### 5.1.2 Update Status Rollup Logic

```python
def update_document_status(self, document_id: UUID, phase: PhaseType, db=None):
    """
    Update document status based on children (bubble-up logic).

    Rules (UPDATED - skipped counts as done):
    1. If ANY child is "processing" → parent = "processing"
    2. If ANY child is "pending"    → parent = "pending"
    3. If ALL children in DONE_STATUSES (completed/skipped) → parent = "completed"
    4. If ANY child "failed" (no pending/processing) → parent = "failed"
    """
    # ... existing query setup ...

    if phase == "graphrag":
        # Query with skipped count
        counts = db.query(
            func.count().filter(status_col == TaskStatus.PROCESSING).label("processing"),
            func.count().filter(status_col == TaskStatus.PENDING).label("pending"),
            func.count().filter(status_col == TaskStatus.COMPLETED).label("completed"),
            func.count().filter(status_col == TaskStatus.FAILED).label("failed"),
            func.count().filter(status_col == TaskStatus.SKIPPED).label("skipped"),  # NEW
            func.count().label("total")
        ).filter(TaskQueueChunk.document_id == document_id).first()

        # Calculate done count (completed + skipped)
        done_count = (counts.completed or 0) + (counts.skipped or 0)
    else:
        # ... existing logic for ocr/vector ...
        done_count = counts.completed or 0

    # Determine new status
    if counts.processing > 0:
        new_status = TaskStatus.PROCESSING
    elif counts.pending > 0:
        new_status = TaskStatus.PENDING
    elif done_count == counts.total:  # CHANGED: use done_count
        new_status = TaskStatus.COMPLETED
    elif counts.failed > 0:
        new_status = TaskStatus.FAILED
    else:
        new_status = TaskStatus.PENDING

    # ... rest of method ...
```

#### 5.1.3 Update `sync_document_indexing_details()`

```python
def sync_document_indexing_details(self, document_id: UUID, db=None):
    # ... existing setup ...

    # GraphRAG stats with skipped count
    graphrag_stats = db.query(
        func.count().label("total"),
        func.count().filter(TaskQueueChunk.graphrag_status == TaskStatus.COMPLETED).label("completed"),
        func.count().filter(TaskQueueChunk.graphrag_status == TaskStatus.PROCESSING).label("processing"),
        func.count().filter(TaskQueueChunk.graphrag_status == TaskStatus.FAILED).label("failed"),
        func.count().filter(TaskQueueChunk.graphrag_status == TaskStatus.PENDING).label("pending"),
        func.count().filter(TaskQueueChunk.graphrag_status == TaskStatus.SKIPPED).label("skipped"),  # NEW
        func.sum(TaskQueueChunk.entities_extracted).label("entities"),
        func.sum(TaskQueueChunk.relationships_extracted).label("relationships"),
    ).filter(TaskQueueChunk.document_id == document_id).first()

    if graphrag_stats and graphrag_stats.total > 0:
        # Done = completed + skipped
        done_count = (graphrag_stats.completed or 0) + (graphrag_stats.skipped or 0)

        # Determine status (skipped documents show as "completed")
        if graphrag_stats.processing > 0:
            graphrag_status = "processing"
        elif done_count == graphrag_stats.total:
            graphrag_status = "completed"  # All done (including skipped)
        elif graphrag_stats.failed > 0 and graphrag_stats.pending == 0 and graphrag_stats.processing == 0:
            graphrag_status = "failed"
        elif done_count > 0:
            graphrag_status = "partial"
        else:
            graphrag_status = "pending"

        doc.indexing_details["graphrag_indexing"] = {
            "status": graphrag_status,
            "total_chunks": graphrag_stats.total,
            "processed_chunks": graphrag_stats.completed or 0,
            "skipped_chunks": graphrag_stats.skipped or 0,  # NEW
            "failed_chunks": graphrag_stats.failed or 0,
            "pending_chunks": graphrag_stats.pending or 0,
            "processing_chunks": graphrag_stats.processing or 0,
            "entities_extracted": graphrag_stats.entities or 0,
            "relationships_extracted": graphrag_stats.relationships or 0,
            "updated_at": now.isoformat(),
        }

        # If all chunks skipped, add skip info
        if graphrag_stats.skipped == graphrag_stats.total and doc.skip_graphrag_reason:
            doc.indexing_details["graphrag_indexing"]["skip_reason"] = doc.skip_graphrag_reason
```

#### 5.1.4 Add Skip Detection Method

```python
def _should_skip_graphrag(self, chunk: TaskQueueChunk, db: Session) -> tuple[bool, str]:
    """
    Determine if a chunk should skip GraphRAG indexing.

    Checks in order:
    1. Document-level skip flag (set at upload or after metadata extraction)
    2. File extension
    3. Document type from metadata

    Returns:
        (should_skip: bool, reason: str or None)
    """
    doc = db.query(Document).filter(Document.id == chunk.document_id).first()
    if not doc:
        return False, None

    # 1. Check document-level skip flag (highest priority)
    if doc.skip_graphrag:
        return True, doc.skip_graphrag_reason or "document_disabled"

    # 2. Check file extension
    should_skip, reason = should_skip_graphrag_for_file(doc.filename)
    if should_skip:
        return True, reason

    # 3. Check document type from metadata
    if doc.document_metadata:
        doc_type = doc.document_metadata.get('document_type', '')
        should_skip, reason = should_skip_graphrag_for_document_type(doc_type)
        if should_skip:
            return True, reason

    return False, None
```

#### 5.1.5 Update `claim_graphrag_chunk_task()`

```python
def claim_graphrag_chunk_task(self, worker_id: str, db=None) -> Optional[ChunkTaskData]:
    """
    Claim the next available GraphRAG indexing chunk task.

    NEW: Automatically skips chunks from documents that shouldn't be GraphRAG indexed.
    """
    should_close_db = db is None
    if db is None:
        db = create_db_session()

    try:
        while True:  # Loop to handle skipped chunks
            # Find next GraphRAG chunk task where Vector is completed
            chunk = db.query(TaskQueueChunk).filter(
                TaskQueueChunk.graphrag_status.in_([TaskStatus.PENDING, TaskStatus.FAILED]),
                TaskQueueChunk.graphrag_retry_count < TaskQueueChunk.max_retries,
                TaskQueueChunk.vector_status == TaskStatus.COMPLETED
            ).order_by(
                TaskQueueChunk.graphrag_status.desc(),
                TaskQueueChunk.graphrag_retry_count.asc(),
                TaskQueueChunk.created_at.asc()
            ).with_for_update(skip_locked=True).first()

            if not chunk:
                return None

            # NEW: Check if this chunk should skip GraphRAG
            should_skip, skip_reason = self._should_skip_graphrag(chunk, db)

            if should_skip:
                # Mark as skipped (not failed)
                chunk.graphrag_status = TaskStatus.SKIPPED
                chunk.graphrag_skip_reason = skip_reason
                chunk.graphrag_completed_at = datetime.now(timezone.utc)
                db.commit()

                logger.info(f"⏭️ Skipped GraphRAG for chunk {chunk.chunk_id}: {skip_reason}")

                # Update parent statuses (skipped counts as done)
                self.update_page_status(chunk.page_id, "graphrag", db)
                self.update_document_status(chunk.document_id, "graphrag", db)
                self.sync_document_indexing_details(chunk.document_id, db)

                # Continue to find next non-skipped chunk
                continue

            # Claim the task (existing logic)
            now = datetime.now(timezone.utc)
            chunk.graphrag_status = TaskStatus.PROCESSING
            chunk.graphrag_worker_id = worker_id
            chunk.graphrag_started_at = now
            chunk.graphrag_last_heartbeat = now

            db.commit()

            logger.info(f"Worker {worker_id} claimed GraphRAG chunk task: chunk={chunk.chunk_id}")

            return ChunkTaskData(
                id=chunk.id,
                page_id=chunk.page_id,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                phase="graphrag",
                retry_count=chunk.graphrag_retry_count
            )

    except Exception as e:
        logger.error(f"Error claiming GraphRAG chunk task: {e}")
        db.rollback()
        return None
    finally:
        if should_close_db:
            db.close()
```

### 5.2 Set Skip Flag at Upload Time

```python
# File: backend/main.py - in upload endpoint

@app.post("/upload")
async def upload_file(...):
    # ... existing code to save file and create document ...

    # NEW: Set skip_graphrag flag for known file types
    from rag_service.graphrag_skip_config import should_skip_graphrag_for_file

    should_skip, skip_reason = should_skip_graphrag_for_file(file.filename)
    if should_skip:
        doc.skip_graphrag = True
        doc.skip_graphrag_reason = skip_reason
        logger.info(f"📊 Document {file.filename} will skip GraphRAG: {skip_reason}")

    db.commit()

    # ... rest of upload code ...
```

### 5.3 Update Skip Flag After Metadata Extraction

```python
# File: backend/queue_service/hierarchical_task_manager.py
# In _run_metadata_extraction() method

def _run_metadata_extraction(self, document_id: UUID, source_name: str, filename: str):
    # ... existing extraction code ...

    # After saving metadata, check if document type should skip GraphRAG
    from rag_service.graphrag_skip_config import should_skip_graphrag_for_document_type

    doc_type = metadata.get('document_type', '')
    should_skip, skip_reason = should_skip_graphrag_for_document_type(doc_type)

    if should_skip and not doc.skip_graphrag:
        doc.skip_graphrag = True
        doc.skip_graphrag_reason = skip_reason
        db.commit()
        logger.info(f"[Metadata] Set skip_graphrag=True for {source_name}: {skip_reason}")
```

---

## 6. Frontend/UI Changes

### 6.1 Status Display Mapping

```typescript
// File: frontend/src/utils/statusDisplay.ts

interface StatusDisplay {
  label: string;
  icon: string;
  color: string;
  tooltip: string;
}

const GRAPHRAG_STATUS_DISPLAY: Record<string, StatusDisplay> = {
  pending: {
    label: "Pending",
    icon: "clock",
    color: "gray",
    tooltip: "Waiting for vector indexing to complete"
  },
  processing: {
    label: "Processing",
    icon: "spinner",
    color: "blue",
    tooltip: "Extracting entities and relationships"
  },
  completed: {
    label: "Completed",
    icon: "check-circle",
    color: "green",
    tooltip: "Entity extraction complete"
  },
  failed: {
    label: "Failed",
    icon: "x-circle",
    color: "red",
    tooltip: "Error during entity extraction"
  },
  skipped: {
    label: "Skipped",
    icon: "fast-forward",
    color: "gray",
    tooltip: "Skipped: tabular/structured data"
  }
};

// For overall document status, treat skipped as completed
function isGraphRAGDone(status: string): boolean {
  return status === 'completed' || status === 'skipped';
}

function getDocumentOverallStatus(doc: Document): StatusDisplay {
  const ocrDone = doc.ocr_status === 'completed';
  const vectorDone = doc.vector_status === 'completed';
  const graphragDone = isGraphRAGDone(doc.graphrag_status);

  if (ocrDone && vectorDone && graphragDone) {
    return {
      label: "Ready",
      icon: "check-circle",
      color: "green",
      tooltip: "Document is ready for search"
    };
  }
  // ... other status checks
}
```

### 6.2 Progress Calculation

```typescript
// Include skipped chunks in progress calculation
function calculateGraphRAGProgress(indexingDetails: IndexingDetails): number {
  const graphrag = indexingDetails?.graphrag_indexing;
  if (!graphrag || !graphrag.total_chunks) return 0;

  const doneCount = (graphrag.processed_chunks || 0) + (graphrag.skipped_chunks || 0);
  return Math.round((doneCount / graphrag.total_chunks) * 100);
}
```

### 6.3 Document Details Display

```tsx
// File: frontend/src/components/DocumentDetails.tsx

function GraphRAGStatus({ doc }: { doc: Document }) {
  const graphrag = doc.indexing_details?.graphrag_indexing;
  const isSkipped = graphrag?.skipped_chunks === graphrag?.total_chunks;

  if (isSkipped) {
    return (
      <div className="status-row">
        <span className="status-icon">⏭️</span>
        <span className="status-label">GraphRAG: Skipped</span>
        <span className="status-detail text-muted">
          ({graphrag.skip_reason || "Tabular/structured data"})
        </span>
      </div>
    );
  }

  // ... normal status display
}
```

### 6.4 Document List Status Column

| Document | Type | Status | GraphRAG |
|----------|------|--------|----------|
| report.pdf | Report | ✅ Ready | ✅ 45 entities |
| data.xlsx | Excel | ✅ Ready | ⏭️ Skipped |
| invoice.pdf | Invoice | ✅ Ready | ⏭️ Skipped |
| contract.docx | Contract | ✅ Ready | ✅ 128 entities |

All documents show as "Ready" - the skipped ones don't block completion.

---

## 7. API Response Format

### 7.1 Document Status Response

```json
{
  "id": "670f8f3b-e9e2-4142-8fc1-720cbb8ef062",
  "filename": "product_purchase_orders.xlsx",
  "ocr_status": "completed",
  "vector_status": "completed",
  "graphrag_status": "completed",
  "skip_graphrag": true,
  "skip_graphrag_reason": "file_type:.xlsx",
  "is_ready": true,
  "indexing_details": {
    "ocr_processing": {
      "status": "completed",
      "total_pages": 1,
      "completed_pages": 1
    },
    "vector_indexing": {
      "status": "completed",
      "total_chunks": 15,
      "indexed_chunks": 15
    },
    "graphrag_indexing": {
      "status": "completed",
      "total_chunks": 15,
      "processed_chunks": 0,
      "skipped_chunks": 15,
      "skip_reason": "file_type:.xlsx",
      "entities_extracted": 0,
      "relationships_extracted": 0
    }
  }
}
```

Note: `graphrag_indexing.status` is "completed" because all chunks are done (skipped counts as done).

---

## 8. Summary: Files to Modify

### New Files
1. **`backend/rag_service/graphrag_skip_config.py`** - Skip configuration and detection logic
2. **`backend/db/migrations/019_add_skipped_task_status.sql`** - Database migration

### Modified Files
1. **`backend/queue_service/models.py`** - Add `SKIPPED` to TaskStatus, add `DONE_STATUSES`
2. **`backend/db/models.py`** - Add `skip_graphrag`, `skip_graphrag_reason` columns
3. **`backend/queue_service/hierarchical_task_manager.py`**:
   - Import skip config
   - Add `_should_skip_graphrag()` method
   - Update `claim_graphrag_chunk_task()` to skip eligible chunks
   - Update `update_document_status()` to count skipped as done
   - Update `update_page_status()` to count skipped as done
   - Update `sync_document_indexing_details()` to include skipped counts
   - Update `_run_metadata_extraction()` to set skip flag
4. **`backend/main.py`** - Set skip flag at upload time

### Frontend Files
1. **Status display utilities** - Handle "skipped" status
2. **Document list component** - Show skipped as complete
3. **Document details component** - Display skip reason
4. **Progress calculations** - Include skipped in done count

---

## 9. Testing Checklist

- [ ] Upload `.xlsx` file → should set `skip_graphrag=True` at upload
- [ ] Upload `.csv` file → should set `skip_graphrag=True` at upload
- [ ] Upload invoice PDF → should set `skip_graphrag=True` after metadata extraction
- [ ] Document with all skipped chunks shows as "Ready"
- [ ] Progress bar shows 100% when all chunks skipped
- [ ] Document list shows correct status for skipped documents
- [ ] API returns correct `indexing_details` with skipped counts
- [ ] Mixed document (PDF with tables) → tabular chunks skipped, narrative processed
