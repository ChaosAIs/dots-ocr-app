# Unified Document Intelligence Solution

## Overview

This document describes a comprehensive solution that combines:
1. **Automatic Structured Data Extraction** - Background extraction of structured JSON data from business documents
2. **Conversational Analytics** - Interactive chat-based querying with plan review and iteration
3. **Existing RAG Integration** - Seamless combination with existing document search capabilities

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED DOCUMENT INTELLIGENCE SOLUTION                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    DOCUMENT UPLOAD FLOW                              │    │
│  │                 (Background - No User Interaction)                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│     Upload → Task Queue → OCR → Vector → GraphRAG → Data Extraction         │
│                              │                             │                 │
│                              │                             ▼                 │
│                              │                    ┌──────────────────┐      │
│                              │                    │  documents_data  │      │
│                              │                    │     (JSON)       │      │
│                              │                    └──────────────────┘      │
│                              ▼                                               │
│                    Existing Pipeline (unchanged)                             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    CONVERSATIONAL ANALYTICS FLOW                     │    │
│  │                    (Interactive User Experience)                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│     User Query → Intent Classifier → Route:                                  │
│                                      │                                       │
│                     ┌────────────────┼────────────────┐                     │
│                     ▼                ▼                ▼                     │
│               document_search   data_analytics    hybrid                    │
│               (Existing RAG)    (New Analytics)   (Combined)                │
│                     │                │                │                     │
│                     ▼                ▼                ▼                     │
│               Qdrant/Neo4j    documents_data     Both paths                 │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    SESSION STATE MANAGEMENT                          │    │
│  │                (Redis Cache + PostgreSQL Persistence)                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│     Short-term: Redis (fast access, 24hr TTL)                               │
│     Long-term: PostgreSQL analytics_sessions table                          │
│     Recovery: Load from PostgreSQL on cache miss                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Background Data Extraction

### Integration with Existing Task Queue

Data extraction becomes the **4th phase** in the existing hierarchical task queue:

```
Existing Pipeline:
  Document → Page Tasks → [OCR] → Chunk Tasks → [Vector] → [GraphRAG]

Extended Pipeline:
  Document → Page Tasks → [OCR] → Chunk Tasks → [Vector] → [GraphRAG] → [Extraction]
                                                                              │
                                                            (auto-triggered when GraphRAG completes)
```

### Extractable Document Types

Not all documents need extraction. Only structured/tabular documents are eligible:

| Domain | Extractable Types |
|--------|-------------------|
| **Financial** | Invoice, Receipt, Bank Statement, Expense Report, Purchase Order, Tax Document |
| **Logistics** | Shipping Manifest, Bill of Lading, Customs Declaration, Delivery Note |
| **Inventory** | Inventory Report, Stock Count, Warehouse Receipt |
| **Retail** | Sales Report, Transaction Log, POS Summary |
| **HR** | Payroll Summary, Timesheet |

**Non-Extractable Types** (skip extraction phase):
- Contracts, Legal Filings
- Research Papers, Articles
- Emails, Memos
- Manuals, Guides

### Extraction Strategies

The system automatically selects the optimal extraction strategy based on document characteristics:

| Strategy | Row Count | Method | Use Case |
|----------|-----------|--------|----------|
| **Direct LLM** | < 50 rows | Single LLM call | Small invoices, receipts |
| **Chunked LLM** | 50-500 rows | Parallel chunks with merge | Monthly statements |
| **Hybrid** | 500-5000 rows | LLM for headers, rules for data | Quarterly reports |
| **Parsed** | > 5000 rows | Pattern matching, streaming | Annual transaction logs |

### Database Schema

```sql
-- Documents table extension
ALTER TABLE documents
ADD COLUMN extraction_eligible BOOLEAN DEFAULT NULL,
ADD COLUMN extraction_status VARCHAR(20) DEFAULT 'pending',
ADD COLUMN extraction_schema_type VARCHAR(64),
ADD COLUMN extraction_error TEXT;

-- Main extracted data storage
CREATE TABLE documents_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Schema Reference
    schema_type VARCHAR(64) NOT NULL,
    schema_version VARCHAR(16) DEFAULT '1.0',

    -- Extracted Data (structured JSON)
    header_data JSONB NOT NULL DEFAULT '{}',     -- Fixed fields (vendor, date, total)
    line_items JSONB DEFAULT '[]',               -- Repeating rows
    summary_data JSONB DEFAULT '{}',             -- Calculated/summary fields

    -- For large tables (> 500 rows)
    line_items_storage VARCHAR(16) DEFAULT 'inline',  -- 'inline' or 'external'
    line_items_count INTEGER DEFAULT 0,

    -- Validation & Quality
    validation_status VARCHAR(20) DEFAULT 'pending',
    overall_confidence DECIMAL(5,4),
    validation_results JSONB,

    -- Extraction Metadata
    extraction_method VARCHAR(32),       -- Strategy used
    extraction_model VARCHAR(64),        -- LLM/parser used
    extraction_duration_ms INTEGER,
    extraction_metadata JSONB,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(document_id)
);

-- Line items overflow for large tables
CREATE TABLE documents_data_line_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    documents_data_id UUID NOT NULL REFERENCES documents_data(id) ON DELETE CASCADE,
    line_number INTEGER NOT NULL,
    data JSONB NOT NULL,
    UNIQUE(documents_data_id, line_number)
);

-- Schema definitions
CREATE TABLE data_schemas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_type VARCHAR(64) NOT NULL UNIQUE,
    schema_version VARCHAR(16) DEFAULT '1.0',
    domain VARCHAR(32) NOT NULL,
    display_name VARCHAR(128),

    header_schema JSONB NOT NULL,
    line_items_schema JSONB,
    summary_schema JSONB,
    validation_rules JSONB,
    extraction_prompt TEXT,

    created_at TIMESTAMP DEFAULT NOW()
);

-- Semantic mappings (business concepts to fields)
CREATE TABLE semantic_mappings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    concept_name VARCHAR(128) NOT NULL,     -- e.g., "total_amount"
    concept_aliases TEXT[],                  -- ["total", "sum", "grand total"]
    applicable_schema_types VARCHAR(64)[],
    json_path VARCHAR(256) NOT NULL,
    data_type VARCHAR(32),
    default_aggregation VARCHAR(32),
    is_calculated BOOLEAN DEFAULT FALSE,
    calculation_formula TEXT
);

-- Entity registry (known companies, products, etc.)
CREATE TABLE entity_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID REFERENCES workspaces(id),
    entity_type VARCHAR(64) NOT NULL,
    canonical_name VARCHAR(256) NOT NULL,
    aliases TEXT[] DEFAULT '{}',
    document_count INTEGER DEFAULT 0,
    last_seen_at TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    UNIQUE(workspace_id, entity_type, canonical_name)
);
```

---

## Part 2: Conversational Analytics

### Session State Machine

```
┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌──────────────┐
│ INITIAL  │───▶│ CLASSIFYING │───▶│QUESTIONING│───▶│   PLANNING   │
└──────────┘    └─────────────┘    └──────────┘    └──────────────┘
     │                                   │                │
     │              ┌────────────────────┘                │
     │              │ (user provides info)               │
     │              ▼                                     ▼
     │         ┌──────────┐                        ┌──────────┐
     │         │ REFINING │◀──────────────────────▶│REVIEWING │
     │         └──────────┘   (user requests       └──────────┘
     │              │          changes)                  │
     │              │                                    │
     │              ▼                                    ▼
     │         ┌──────────┐                        ┌──────────┐
     └────────▶│EXECUTING │───────────────────────▶│ COMPLETE │
 (simple       └──────────┘                        └──────────┘
  queries)          │                                    │
                    │                                    │
                    ▼                                    ▼
               ┌──────────┐                        ┌──────────┐
               │  ERROR   │                        │ FOLLOW_UP│
               └──────────┘                        └──────────┘
```

### State Descriptions

| State | Description | User Interaction |
|-------|-------------|------------------|
| **INITIAL** | Query received, starting analysis | None |
| **CLASSIFYING** | Determining query intent | None |
| **QUESTIONING** | Gap analysis - gathering missing info | User answers questions |
| **PLANNING** | Generating execution plan | None |
| **REVIEWING** | Presenting plan for approval | User reviews, approves/modifies |
| **REFINING** | Updating plan based on feedback | User provides changes |
| **EXECUTING** | Running queries, aggregating data | Optional: view progress |
| **COMPLETE** | Results ready | View results |
| **FOLLOW_UP** | User asks follow-up question | New query in context |
| **ERROR** | Error occurred | Error message, retry option |

### Intent Classification

```
Query Intent Types:
├── DOCUMENT_SEARCH
│   └── "Find contracts mentioning liability"
│   └── "Summarize the Q3 report"
│
├── DATA_ANALYTICS
│   └── "Total sales by month for Company A"
│   └── "Average invoice amount in Q4 2024"
│
├── HYBRID
│   └── "Find all invoices over $10k and calculate total"
│   └── "List vendors with receipts and show spending per vendor"
│
└── GENERAL
    └── "What can you help me with?"
    └── "How do I upload files?"
```

### Session Storage

```sql
CREATE TABLE analytics_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Links
    chat_session_id UUID REFERENCES chat_sessions(id),
    user_id UUID NOT NULL REFERENCES users(id),
    workspace_id UUID NOT NULL REFERENCES workspaces(id),

    -- State
    state VARCHAR(32) NOT NULL DEFAULT 'INITIAL',
    state_entered_at TIMESTAMP DEFAULT NOW(),

    -- Query Context
    original_query TEXT NOT NULL,
    intent_classification JSONB,

    -- Gathered Info (from gap analysis)
    gathered_info JSONB DEFAULT '{}',
    /*
    Example:
    {
        "entities": [{"name": "Company A", "resolved_id": "uuid", "confidence": 0.95}],
        "time_range": {"start": "2024-01-01", "end": "2024-12-31"},
        "metrics": ["total_amount", "transaction_count"],
        "aggregation": "monthly"
    }
    */

    -- Plan
    current_plan JSONB,
    plan_version INTEGER DEFAULT 0,
    plan_history JSONB DEFAULT '[]',

    -- Execution
    execution_progress JSONB,
    cached_results JSONB,
    result_generated_at TIMESTAMP,

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP DEFAULT NOW() + INTERVAL '24 hours'
);
```

---

## Part 3: Redis Integration

### Configuration (.env)

```bash
# Redis Cache Configuration
REDIS_HOST=n8n-redis-1
REDIS_PORT=6379
REDIS_DB=1
REDIS_PASSWORD=
REDIS_SSL=false

# Session Cache Settings
REDIS_SESSION_TTL=86400           # 24 hours
REDIS_SESSION_PREFIX=analytics:session:

# Real-time Updates
REDIS_PUBSUB_CHANNEL=analytics:updates
```

### Session Caching Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Memory Architecture                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐            │
│  │  Short-Term     │   │  Medium-Term    │   │   Long-Term     │            │
│  │  (In-Request)   │   │  (Session)      │   │  (Persistent)   │            │
│  ├─────────────────┤   ├─────────────────┤   ├─────────────────┤            │
│  │ • Current msg   │   │ • Message hist  │   │ • Entity reg    │            │
│  │ • Tool results  │   │ • Gathered info │   │ • User prefs    │            │
│  │ • Working vars  │   │ • Plan state    │   │ • Query patterns│            │
│  │                 │   │ • Exec progress │   │ • Cached schemas│            │
│  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘            │
│           │                     │                     │                     │
│           │    LangChain        │    Redis Cache      │    PostgreSQL       │
│           │    Memory           │    (24hr TTL)       │    (permanent)      │
│           │                     │                     │                     │
│           └─────────────────────┴─────────────────────┘                     │
│                                 │                                           │
│                                 ▼                                           │
│                    ┌─────────────────────────┐                              │
│                    │   Session Recovery      │                              │
│                    │   (Redis → PostgreSQL)  │                              │
│                    └─────────────────────────┘                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Recovery Mechanism

1. **On new request**: Check Redis first
2. **Cache miss**: Load from PostgreSQL, populate Redis
3. **Disconnection**: Session persists in both stores
4. **Reconnection**: Resume from last known state
5. **Expiration**: Clean up from both Redis and PostgreSQL

---

## Part 4: Interactive Conversation Flow

### Example Session

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Example Interactive Session                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User: "Show me monthly sales comparison for Company A vs Company B         │
│         in 2024"                                                            │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  [State: INITIAL → CLASSIFYING]                                             │
│  System: Analyzing intent... (data_analytics, confidence: 0.94)             │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  [State: CLASSIFYING → QUESTIONING]                                         │
│  Assistant: I found 3 entities matching "Company A":                        │
│             1. ABC Corporation (45 invoices)                                │
│             2. A-Tech Solutions (12 invoices)                               │
│             3. Company Alpha LLC (8 invoices)                               │
│             Which one did you mean?                                         │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  User: "ABC Corporation"                                                    │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  [State: QUESTIONING - gathering more info]                                 │
│  Assistant: Got it. For "Company B", I found:                               │
│             1. Beta Industries (32 invoices)                                │
│             2. B&B Trading (5 invoices)                                     │
│             Which one?                                                      │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  User: "Beta Industries"                                                    │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  [State: QUESTIONING → PLANNING → REVIEWING]                                │
│  Assistant: Here's my plan:                                                 │
│                                                                              │
│  **Query Plan v1**                                                          │
│  1. Search invoices for ABC Corporation (Jan-Dec 2024)                      │
│  2. Search invoices for Beta Industries (Jan-Dec 2024)                      │
│  3. Aggregate monthly totals for each company                               │
│  4. Generate comparison chart and summary                                   │
│                                                                              │
│  Documents involved: ~77 invoices                                           │
│  Shall I proceed, or would you like to modify?                              │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  User: "Also include receipts, not just invoices"                           │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  [State: REVIEWING → REFINING → REVIEWING]                                  │
│  Assistant: Updated plan:                                                   │
│                                                                              │
│  **Query Plan v2** (Changed: +receipts)                                     │
│  1. Search invoices AND receipts for ABC Corporation (Jan-Dec 2024)         │
│  2. Search invoices AND receipts for Beta Industries (Jan-Dec 2024)         │
│  3. Aggregate monthly totals for each company                               │
│  4. Generate comparison chart and summary                                   │
│                                                                              │
│  Documents involved: ~112 documents                                         │
│  Ready to proceed?                                                          │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  User: "Yes, go ahead"                                                      │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  [State: REVIEWING → EXECUTING]                                             │
│  Assistant: Executing... (Step 1/4: Searching ABC Corporation docs)         │
│             ...                                                             │
│             (Step 4/4: Generating report)                                   │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  [State: EXECUTING → COMPLETE]                                              │
│  Assistant: Here's your comparison:                                         │
│                                                                              │
│  | Month    | ABC Corp  | Beta Ind  | Difference |                          │
│  |----------|-----------|-----------|------------|                          │
│  | Jan 2024 | $45,230   | $38,100   | +$7,130    |                          │
│  | Feb 2024 | $52,100   | $41,200   | +$10,900   |                          │
│  | ...      | ...       | ...       | ...        |                          │
│                                                                              │
│  ABC Corporation shows 18% higher average monthly sales.                    │
│  Would you like more details or a different view?                           │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  [State: COMPLETE → FOLLOW_UP]                                              │
│  User: "Show me just Q4 with a chart"                                       │
│                                                                              │
│  Assistant: (Continues with context from previous query...)                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 5: Access Control

### Document-Level Filtering

All analytics queries are automatically filtered by user's accessible documents:

```python
def get_user_accessible_documents(user_id: UUID, workspace_id: UUID) -> List[UUID]:
    """Get document IDs the user can access."""
    # Based on existing workspace membership and document ownership
    return db.query(Document.id).filter(
        or_(
            Document.owner_id == user_id,
            Document.workspace_id == workspace_id  # Workspace members can access
        )
    ).all()

def execute_analytics_query(user_id: UUID, workspace_id: UUID, sql: str):
    """Execute query with access control filter."""
    accessible_docs = get_user_accessible_documents(user_id, workspace_id)

    # Inject filter into query
    filtered_sql = f"""
        WITH accessible AS (
            SELECT id FROM documents WHERE id IN ({','.join(map(str, accessible_docs))})
        )
        SELECT * FROM ({sql}) AS query_result
        WHERE document_id IN (SELECT id FROM accessible)
    """
    return db.execute(filtered_sql)
```

---

## Part 6: Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Session Cache** | Redis (n8n-redis-1) | Fast session state, pub/sub |
| **Persistent Storage** | PostgreSQL | Sessions, extracted data, schemas |
| **Background Jobs** | Existing HierarchicalTaskQueue | Data extraction as 4th phase |
| **Vector Search** | Qdrant (existing) | Document search, metadata routing |
| **Graph Database** | Neo4j (existing) | Entity relationships |
| **LLM Backend** | vLLM/Ollama (existing) | Intent classification, extraction, generation |
| **Real-time Updates** | WebSocket + Redis Pub/Sub | Live progress updates |
| **Chat Infrastructure** | Existing ChatSession | Message persistence |

---

## Implementation Phases

### Phase 1: Infrastructure
- [x] Add Redis configuration to `.env`
- [ ] Create database migration for new tables
- [ ] Implement Redis session manager
- [ ] Add extraction phase to task queue

### Phase 2: Background Extraction
- [ ] Document eligibility classifier
- [ ] Extraction strategy selector
- [ ] Schema-based extraction (LLM + parsing)
- [ ] Validation pipeline
- [ ] Error handling and retry logic

### Phase 3: Conversational Analytics
- [ ] Intent classifier agent
- [ ] Gap analysis agent
- [ ] Plan builder agent
- [ ] SQL generator
- [ ] Report formatter

### Phase 4: Integration
- [ ] Connect to existing RAG for hybrid queries
- [ ] Access control enforcement
- [ ] Real-time progress via WebSocket
- [ ] Session recovery mechanisms

---

## Configuration Reference

### Environment Variables (.env)

```bash
# Redis Cache
REDIS_HOST=n8n-redis-1
REDIS_PORT=6379
REDIS_DB=1
REDIS_SESSION_TTL=86400

# Data Extraction
DATA_EXTRACTION_ENABLED=true
EXTRACTION_DIRECT_LLM_MAX_ROWS=50
EXTRACTION_CHUNKED_LLM_MAX_ROWS=500
EXTRACTION_HYBRID_MAX_ROWS=5000
EXTRACTION_MIN_CONFIDENCE=0.85

# Analytics
ANALYTICS_ENABLED=true
ANALYTICS_SESSION_EXPIRY_HOURS=24
ANALYTICS_MAX_PLAN_ITERATIONS=5
ANALYTICS_AUTO_CLASSIFY_INTENT=true
```

---

## Appendix: Schema Examples

### Invoice Schema

```json
{
  "schema_type": "invoice",
  "header_schema": {
    "type": "object",
    "properties": {
      "invoice_number": {"type": "string"},
      "invoice_date": {"type": "string", "format": "date"},
      "due_date": {"type": "string", "format": "date"},
      "vendor_name": {"type": "string"},
      "vendor_address": {"type": "string"},
      "customer_name": {"type": "string"},
      "customer_address": {"type": "string"},
      "subtotal": {"type": "number"},
      "tax_amount": {"type": "number"},
      "total_amount": {"type": "number"},
      "currency": {"type": "string", "default": "USD"}
    },
    "required": ["invoice_number", "total_amount"]
  },
  "line_items_schema": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "description": {"type": "string"},
        "quantity": {"type": "number"},
        "unit_price": {"type": "number"},
        "amount": {"type": "number"}
      }
    }
  }
}
```

### Bank Statement Schema

```json
{
  "schema_type": "bank_statement",
  "header_schema": {
    "type": "object",
    "properties": {
      "account_number": {"type": "string"},
      "account_holder": {"type": "string"},
      "bank_name": {"type": "string"},
      "statement_period_start": {"type": "string", "format": "date"},
      "statement_period_end": {"type": "string", "format": "date"},
      "opening_balance": {"type": "number"},
      "closing_balance": {"type": "number"},
      "total_deposits": {"type": "number"},
      "total_withdrawals": {"type": "number"}
    }
  },
  "line_items_schema": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "date": {"type": "string", "format": "date"},
        "description": {"type": "string"},
        "reference": {"type": "string"},
        "debit": {"type": "number"},
        "credit": {"type": "number"},
        "balance": {"type": "number"}
      }
    }
  }
}
```
