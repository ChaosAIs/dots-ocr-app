# Schema-Centric Data Management Design & Implementation Plan

## Executive Summary

This document outlines the comprehensive design for schema-centric structural data management in the OCR application. The design addresses gaps in the current implementation where schema definitions exist but are not actively utilized in extraction and query workflows.

**Key Design Decisions:**
- **Option B Storage**: All schema info cached in `extraction_metadata` column
- **Soft Linking**: `schema_type` string links `DocumentData` to `DataSchema`
- **Independent Processing**: Multi-schema queries processed separately without merging
- **Priority-based Lookup**: extraction_metadata → DataSchema table → Dynamic inference

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Schema Registry Design](#2-schema-registry-design)
3. [Data Extraction Workflow](#3-data-extraction-workflow)
4. [Query Workflow with Schema](#4-query-workflow-with-schema)
5. [Response Formatting](#5-response-formatting)
6. [Multi-Schema Query Handling](#6-multi-schema-query-handling)
7. [Implementation Plan](#7-implementation-plan)
8. [SQL Schema Seed Script](#8-sql-schema-seed-script)

---

## 1. Current State Analysis

### 1.1 Database Models

**DataSchema Table** (exists but NOT actively used):
```python
class DataSchema(Base):
    __tablename__ = "data_schemas"

    id: UUID
    schema_type: str              # e.g., "invoice", "receipt"
    schema_name: str              # Human-readable name
    description: str
    header_schema: dict           # JSON schema for header fields
    line_items_schema: dict       # JSON schema for line items
    summary_schema: dict          # JSON schema for summary fields
    field_mappings: dict          # Field name → data type mappings
    extraction_prompt: str        # LLM prompt for extraction
    validation_rules: dict        # Validation constraints
```

**DocumentData Table** (actively used):
```python
class DocumentData(Base):
    __tablename__ = "document_data"

    id: UUID
    document_id: UUID             # FK to documents
    schema_type: str              # Soft link to DataSchema
    header_data: dict             # Extracted header fields
    line_items: list              # Extracted line items
    summary_data: dict            # Extracted summary
    extraction_metadata: dict     # NEW: Cache field_mappings here
```

### 1.2 Document Type Hierarchy

```
Classification Types (29 types)
    ↓ maps to
Extractable Types (19 variations → 12 schema_types)
    ↓ stored in
DataSchema Table (12 formal schemas)
```

**Extractable Schema Types (12):**
| Schema Type | Document Type Variations |
|-------------|-------------------------|
| invoice | invoice |
| receipt | receipt |
| bank_statement | bank_statement, bank statement |
| financial_report | financial_report |
| expense_report | expense_report, expense report |
| purchase_order | purchase_order, purchase order |
| tax_document | tax_document, tax document |
| shipping_manifest | shipping_document, shipping_manifest, delivery_note |
| bill_of_lading | bill_of_lading, bill of lading |
| customs_declaration | customs_declaration |
| inventory_report | inventory_report, stock_report |
| spreadsheet | spreadsheet, excel, csv, worksheet |

### 1.3 Current Gaps Identified

1. **Schema Not Injected**: `header_schema` and `line_items_schema` never injected into SQL generation prompts
2. **Dynamic-Only**: Field mappings inferred dynamically, ignoring formal schemas
3. **No Caching**: Same schema re-inferred on every query
4. **Missing Prompts**: Only 4 extraction prompts defined (invoice, receipt, bank_statement, spreadsheet)

---

## 2. Schema Registry Design

### 2.1 Schema Structure

Each schema in `DataSchema` contains:

```json
{
  "header_schema": {
    "type": "object",
    "properties": {
      "invoice_number": {"type": "string", "description": "Unique invoice identifier"},
      "invoice_date": {"type": "string", "format": "date"},
      "total_amount": {"type": "number"}
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
  },
  "summary_schema": {
    "type": "object",
    "properties": {
      "subtotal": {"type": "number"},
      "tax_amount": {"type": "number"},
      "total_amount": {"type": "number"}
    }
  },
  "field_mappings": {
    "header_fields": {
      "invoice_number": {"type": "string", "source": "header"},
      "invoice_date": {"type": "date", "source": "header"}
    },
    "line_item_fields": {
      "description": {"type": "string", "source": "line_item"},
      "quantity": {"type": "number", "source": "line_item"}
    }
  }
}
```

### 2.2 Soft Linking Strategy

```
┌─────────────────┐         ┌─────────────────┐
│  DataSchema     │         │  DocumentData   │
├─────────────────┤         ├─────────────────┤
│ schema_type: PK │◄────────│ schema_type: FK │
│ header_schema   │  soft   │ header_data     │
│ line_items_sch  │  link   │ line_items      │
│ field_mappings  │         │ extraction_meta │
└─────────────────┘         └─────────────────┘
```

**Why Soft Link (not FK)?**
- Allows documents without formal schemas (e.g., unknown spreadsheets)
- Enables graceful degradation to dynamic inference
- Supports schema evolution without migration

---

## 3. Data Extraction Workflow

### 3.1 Complete Extraction Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                      DOCUMENT UPLOAD & EXTRACTION                     │
└──────────────────────────────────────────────────────────────────────┘

Step 1: File Upload
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DOCUMENT CLASSIFICATION (3-Level Hierarchical)                      │
│                                                                       │
│  Level 1: Chunk Summaries                                            │
│     └─► Each chunk → LLM summary                                     │
│                                                                       │
│  Level 2: Meta-Summary                                               │
│     └─► Combine chunk summaries → document summary                   │
│                                                                       │
│  Level 3: LLM Classification                                         │
│     └─► Summary → document_type (from 29 types)                      │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EXTRACTABILITY CHECK                                                │
│                                                                       │
│  if document_type in EXTRACTABLE_DOCUMENT_TYPES:                     │
│      schema_type = EXTRACTABLE_DOCUMENT_TYPES[document_type]         │
│      → proceed to extraction                                         │
│  elif document_type in NON_EXTRACTABLE_TYPES:                        │
│      → skip extraction (text-only document)                          │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SCHEMA LOOKUP (Priority Order)                                      │
│                                                                       │
│  1. Query DataSchema table by schema_type                            │
│     SELECT * FROM data_schemas WHERE schema_type = ?                 │
│                                                                       │
│  2. If found → Use formal schema                                     │
│     - header_schema for validation                                   │
│     - extraction_prompt for LLM                                      │
│     - field_mappings for SQL generation                              │
│                                                                       │
│  3. If not found (spreadsheet/unknown):                              │
│     → Use LLMSchemaAnalyzer for dynamic inference                    │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LLM EXTRACTION                                                      │
│                                                                       │
│  Strategy Selection (based on row count):                            │
│  ┌─────────────────┬──────────────────────────────────┐             │
│  │ Rows            │ Strategy                          │             │
│  ├─────────────────┼──────────────────────────────────┤             │
│  │ < 50            │ LLM_DIRECT (single call)          │             │
│  │ 50-500          │ LLM_CHUNKED (parallel chunks)     │             │
│  │ 500-5000        │ HYBRID (LLM headers + rules data) │             │
│  │ > 5000          │ PARSED (pattern matching only)    │             │
│  └─────────────────┴──────────────────────────────────┘             │
│                                                                       │
│  LLM Prompt = extraction_prompt from DataSchema                      │
│  Output = { header_data, line_items, summary_data }                  │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STORAGE (Option B: extraction_metadata caching)                     │
│                                                                       │
│  DocumentData record:                                                │
│  {                                                                    │
│    "schema_type": "invoice",                                         │
│    "header_data": { extracted header fields },                       │
│    "line_items": [ extracted items ],                                │
│    "summary_data": { extracted summary },                            │
│    "extraction_metadata": {                        ◄── NEW CACHE     │
│      "field_mappings": {                                             │
│        "header_fields": {                                            │
│          "invoice_number": {"type": "string", "source": "header"},   │
│          "invoice_date": {"type": "date", "source": "header"}        │
│        },                                                            │
│        "line_item_fields": {                                         │
│          "description": {"type": "string", "source": "line_item"},   │
│          "quantity": {"type": "number", "source": "line_item"}       │
│        }                                                             │
│      },                                                              │
│      "schema_version": "1.0",                                        │
│      "extracted_at": "2025-01-15T10:30:00Z"                         │
│    }                                                                  │
│  }                                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Code Changes for Extraction

**File: `/backend/extraction_service/data_extractor.py`**

```python
async def extract_structured_data(
    document_id: UUID,
    document_type: str,
    content: str,
    db: AsyncSession
) -> DocumentData:
    """Extract structured data using schema-centric approach."""

    # Step 1: Get schema_type from document_type
    schema_type = get_schema_for_document_type(document_type)
    if not schema_type:
        raise ValueError(f"No schema for document type: {document_type}")

    # Step 2: Lookup formal schema
    schema = await get_data_schema(db, schema_type)

    # Step 3: Get extraction prompt
    if schema and schema.extraction_prompt:
        prompt = schema.extraction_prompt
    else:
        prompt = get_extraction_prompt(schema_type)  # Fallback

    # Step 4: Extract via LLM
    extracted = await llm_extract(content, prompt)

    # Step 5: Build field_mappings for caching
    if schema and schema.field_mappings:
        field_mappings = schema.field_mappings
    else:
        field_mappings = infer_field_mappings(extracted)

    # Step 6: Store with extraction_metadata
    document_data = DocumentData(
        document_id=document_id,
        schema_type=schema_type,
        header_data=extracted.get("header_data", {}),
        line_items=extracted.get("line_items", []),
        summary_data=extracted.get("summary_data", {}),
        extraction_metadata={
            "field_mappings": field_mappings,
            "schema_version": schema.version if schema else "dynamic",
            "extracted_at": datetime.utcnow().isoformat()
        }
    )

    return document_data
```

---

## 4. Query Workflow with Schema

### 4.1 Field Mapping Lookup Priority

```
┌──────────────────────────────────────────────────────────────────────┐
│                    FIELD MAPPING LOOKUP PRIORITY                      │
└──────────────────────────────────────────────────────────────────────┘

Priority 1: extraction_metadata (per-document cache)
    │
    │  SELECT extraction_metadata->'field_mappings'
    │  FROM document_data WHERE document_id = ?
    │
    │  ✓ Fast (already in DocumentData)
    │  ✓ Document-specific (handles variations)
    │
    └─► If found → Use immediately
              │
              ▼ If not found

Priority 2: DataSchema table (formal schema)
    │
    │  SELECT field_mappings FROM data_schemas
    │  WHERE schema_type = ?
    │
    │  ✓ Canonical field definitions
    │  ✓ Consistent across documents
    │
    └─► If found → Use and cache to extraction_metadata
              │
              ▼ If not found

Priority 3: Dynamic inference (last resort)
    │
    │  Analyze header_data and line_items structure
    │  Infer types from actual values
    │
    │  ⚠ Slower (requires data analysis)
    │  ⚠ May be inconsistent
    │
    └─► Generate field_mappings and cache
```

### 4.2 SQL Generation with Schema Injection

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SQL GENERATION WORKFLOW                            │
└──────────────────────────────────────────────────────────────────────┘

Step 1: Query Analysis
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  USER QUERY ANALYSIS                                                 │
│                                                                       │
│  Input: "What is the total amount for all invoices in January?"      │
│                                                                       │
│  LLM Analysis Output:                                                │
│  {                                                                    │
│    "query_type": "aggregation",                                      │
│    "target_fields": ["total_amount"],                                │
│    "filters": [{"field": "date", "operator": "between", ...}],       │
│    "aggregations": ["sum"]                                           │
│  }                                                                    │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
Step 2: Schema & Field Mapping Retrieval
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FIELD MAPPING RETRIEVAL                                             │
│                                                                       │
│  For each document_id in query scope:                                │
│    1. Check extraction_metadata.field_mappings                       │
│    2. Fallback to DataSchema.field_mappings                          │
│    3. Fallback to dynamic inference                                  │
│                                                                       │
│  Result (injected into SQL prompt):                                  │
│  {                                                                    │
│    "header_fields": {                                                │
│      "invoice_number": {"type": "string", "source": "header"},       │
│      "invoice_date": {"type": "date", "source": "header"},           │
│      "total_amount": {"type": "number", "source": "header"}          │
│    },                                                                │
│    "line_item_fields": {                                             │
│      "description": {"type": "string", "source": "line_item"},       │
│      "quantity": {"type": "number", "source": "line_item"},          │
│      "unit_price": {"type": "number", "source": "line_item"},        │
│      "amount": {"type": "number", "source": "line_item"}             │
│    }                                                                  │
│  }                                                                    │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
Step 3: SQL Generation with Schema Context
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LLM SQL GENERATION PROMPT                                           │
│                                                                       │
│  You are a PostgreSQL expert. Generate SQL for the user's query.    │
│                                                                       │
│  ## Available Document Data Schema                          ◄── NEW  │
│                                                                       │
│  The document_data table has this structure:                         │
│  - header_data: JSONB containing header fields                       │
│  - line_items: JSONB array containing line item records              │
│  - summary_data: JSONB containing summary fields                     │
│                                                                       │
│  ## Field Mappings (with source information)               ◄── NEW   │
│                                                                       │
│  Header Fields (access via header_data->>'field_name'):              │
│  - invoice_number: string                                            │
│  - invoice_date: date                                                │
│  - total_amount: number                                              │
│                                                                       │
│  Line Item Fields (access via jsonb_array_elements):                 │
│  - description: string                                               │
│  - quantity: number                                                  │
│  - unit_price: number                                                │
│  - amount: number                                                    │
│                                                                       │
│  ## User Query                                                       │
│  {user_query}                                                        │
│                                                                       │
│  Generate valid PostgreSQL query.                                    │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
Step 4: SQL Execution & Error Correction
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EXECUTION WITH RETRY                                                │
│                                                                       │
│  1. Execute generated SQL                                            │
│  2. If error → Send error + schema back to LLM for correction       │
│  3. Retry up to 3 times                                              │
│  4. Return results or error                                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 Code Changes for Query

**File: `/backend/analytics_service/llm_sql_generator.py`**

```python
def _build_sql_generation_prompt(
    self,
    user_query: str,
    query_analysis: dict,
    field_mappings: dict,
    schema_type: str
) -> str:
    """Build SQL generation prompt WITH schema injection."""

    # Format field mappings with source info
    header_fields_str = self._format_fields(
        field_mappings.get("header_fields", {}),
        "header_data"
    )
    line_item_fields_str = self._format_fields(
        field_mappings.get("line_item_fields", {}),
        "line_items"
    )

    prompt = f"""You are a PostgreSQL expert. Generate SQL for the user's query.

## Document Data Schema

The document_data table structure:
- header_data: JSONB containing document header fields
- line_items: JSONB array containing line item records
- summary_data: JSONB containing summary/totals

## Available Fields

### Header Fields (access via header_data->>'field_name'):
{header_fields_str}

### Line Item Fields (access via jsonb_array_elements(line_items)):
{line_item_fields_str}

## Query Analysis
- Query Type: {query_analysis.get('query_type')}
- Target Fields: {query_analysis.get('target_fields')}
- Filters: {query_analysis.get('filters')}
- Aggregations: {query_analysis.get('aggregations')}

## User Query
{user_query}

Generate a valid PostgreSQL query. Use proper JSONB operators.
For line items, use: jsonb_array_elements(line_items) AS item
Access item fields via: (item->>'field_name')::type
"""
    return prompt
```

---

## 5. Response Formatting

### 5.1 Response Structure

```json
{
  "success": true,
  "query": "What are the total amounts by vendor?",
  "results": {
    "data": [
      {"vendor_name": "Acme Corp", "total_amount": 15000.00},
      {"vendor_name": "Beta Inc", "total_amount": 8500.00}
    ],
    "row_count": 2,
    "schema_info": {
      "schema_type": "invoice",
      "fields_used": ["vendor_name", "total_amount"],
      "field_sources": {
        "vendor_name": "header",
        "total_amount": "header"
      }
    }
  },
  "metadata": {
    "documents_queried": 15,
    "execution_time_ms": 245,
    "sql_generated": "SELECT ... FROM document_data ..."
  }
}
```

### 5.2 LLM Summary Generation

```python
async def generate_response_summary(
    query: str,
    results: list,
    schema_info: dict
) -> str:
    """Generate natural language summary of query results."""

    prompt = f"""Summarize the following query results in natural language.

Query: {query}

Schema Type: {schema_info['schema_type']}
Fields Used: {schema_info['fields_used']}

Results:
{json.dumps(results, indent=2)}

Provide a clear, concise summary that answers the user's question.
Include relevant totals, counts, or insights from the data.
"""

    summary = await llm_call(prompt)
    return summary
```

---

## 6. Multi-Schema Query Handling

### 6.1 Design Decision: Independent Processing

When a query spans documents with different schema types (e.g., invoices AND receipts), process each schema type **independently without merging**.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    MULTI-SCHEMA QUERY WORKFLOW                        │
└──────────────────────────────────────────────────────────────────────┘

User Query: "Show me all expenses from last month"
Document IDs: [doc1(invoice), doc2(receipt), doc3(invoice), doc4(receipt)]

Step 1: Group by Schema Type
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SCHEMA GROUPING                                                     │
│                                                                       │
│  {                                                                    │
│    "invoice": [doc1, doc3],                                          │
│    "receipt": [doc2, doc4]                                           │
│  }                                                                    │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
Step 2: Process Each Schema Independently (Parallel)
         │
         ├──────────────────────────────────────────┐
         ▼                                          ▼
┌─────────────────────────┐              ┌─────────────────────────┐
│  INVOICE PROCESSING      │              │  RECEIPT PROCESSING      │
│                          │              │                          │
│  Field Mappings:         │              │  Field Mappings:         │
│  - vendor_name           │              │  - store_name            │
│  - invoice_date          │              │  - transaction_date      │
│  - total_amount          │              │  - total_amount          │
│                          │              │                          │
│  SQL Query:              │              │  SQL Query:              │
│  SELECT vendor_name,     │              │  SELECT store_name,      │
│    invoice_date,         │              │    transaction_date,     │
│    total_amount          │              │    total_amount          │
│  FROM document_data      │              │  FROM document_data      │
│  WHERE document_id IN    │              │  WHERE document_id IN    │
│    (doc1, doc3)          │              │    (doc2, doc4)          │
│                          │              │                          │
│  Results:                │              │  Results:                │
│  [                       │              │  [                       │
│    {vendor: "Acme",      │              │    {store: "Walmart",    │
│     date: "2025-01-05",  │              │     date: "2025-01-10",  │
│     amount: 500}         │              │     amount: 45.99}       │
│  ]                       │              │  ]                       │
└─────────────────────────┘              └─────────────────────────┘
         │                                          │
         └──────────────────┬───────────────────────┘
                            ▼
Step 3: Return Separated Results (NO MERGE)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SEPARATED RESPONSE                                                  │
│                                                                       │
│  {                                                                    │
│    "results_by_type": {                                              │
│      "invoice": {                                                    │
│        "schema_info": { field mappings },                            │
│        "data": [ invoice results ],                                  │
│        "summary": "2 invoices totaling $1,500"                       │
│      },                                                              │
│      "receipt": {                                                    │
│        "schema_info": { field mappings },                            │
│        "data": [ receipt results ],                                  │
│        "summary": "2 receipts totaling $95.98"                       │
│      }                                                               │
│    },                                                                │
│    "combined_summary": "Found 4 expense documents..."                │
│  }                                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Code Implementation

**File: `/backend/analytics_service/sql_query_executor.py`**

```python
async def execute_multi_schema_query(
    self,
    user_query: str,
    document_ids: List[UUID],
    db: AsyncSession
) -> dict:
    """Execute query across documents with different schemas."""

    # Step 1: Group documents by schema_type
    schema_groups = await self._group_by_schema(document_ids, db)

    # Step 2: Process each schema group independently
    results_by_type = {}

    for schema_type, doc_ids in schema_groups.items():
        # Get field mappings for this schema
        field_mappings = await self._get_field_mappings(
            schema_type, doc_ids, db
        )

        # Generate and execute SQL for this schema
        sql = await self.sql_generator.generate_sql(
            user_query=user_query,
            document_ids=doc_ids,
            field_mappings=field_mappings,
            schema_type=schema_type
        )

        result = await self._execute_sql(sql, db)

        results_by_type[schema_type] = {
            "schema_info": {
                "schema_type": schema_type,
                "field_mappings": field_mappings,
                "document_count": len(doc_ids)
            },
            "data": result,
            "row_count": len(result)
        }

    # Step 3: Generate combined summary (NO data merge)
    combined_summary = await self._generate_multi_schema_summary(
        user_query, results_by_type
    )

    return {
        "success": True,
        "query": user_query,
        "results_by_type": results_by_type,
        "combined_summary": combined_summary
    }
```

---

## 7. Implementation Plan

### 7.1 Phase 1: Schema Infrastructure (Week 1)

| Task | Files | Description |
|------|-------|-------------|
| 1.1 | `data_schemas` table | Run seed SQL to populate all 12 schema definitions |
| 1.2 | `extraction_metadata` | Add migration if column doesn't exist |
| 1.3 | Schema lookup utilities | Create `get_data_schema()` and `get_field_mappings()` functions |

**Deliverables:**
- [ ] Execute schema seed SQL script
- [ ] Verify all 12 schemas in database
- [ ] Create schema lookup service

### 7.2 Phase 2: Extraction Integration (Week 2)

| Task | Files | Description |
|------|-------|-------------|
| 2.1 | `data_extractor.py` | Use DataSchema.extraction_prompt |
| 2.2 | `data_extractor.py` | Cache field_mappings to extraction_metadata |
| 2.3 | `extraction_config.py` | Remove hardcoded prompts (use DB) |

**Deliverables:**
- [ ] Extraction uses formal schema prompts
- [ ] Field mappings cached on every extraction
- [ ] Unit tests for schema-based extraction

### 7.3 Phase 3: Query Integration (Week 3)

| Task | Files | Description |
|------|-------|-------------|
| 3.1 | `sql_query_executor.py` | Implement priority-based field mapping lookup |
| 3.2 | `llm_sql_generator.py` | Inject schema into SQL generation prompt |
| 3.3 | `llm_sql_generator.py` | Add source tracking (header vs line_item) |

**Deliverables:**
- [ ] SQL generator receives field schema context
- [ ] Field sources properly indicated in prompts
- [ ] Unit tests for schema-injected SQL generation

### 7.4 Phase 4: Multi-Schema Queries (Week 4)

| Task | Files | Description |
|------|-------|-------------|
| 4.1 | `sql_query_executor.py` | Implement `_group_by_schema()` |
| 4.2 | `sql_query_executor.py` | Implement `execute_multi_schema_query()` |
| 4.3 | Response formatting | Return separated results by type |

**Deliverables:**
- [ ] Multi-schema queries work independently
- [ ] Results separated by schema type
- [ ] Combined summary generation

### 7.5 Phase 5: Testing & Validation (Week 5)

| Task | Description |
|------|-------------|
| 5.1 | Integration tests for all 12 schema types |
| 5.2 | Performance testing with large documents |
| 5.3 | Edge case handling (missing schemas, mixed types) |

---

## 8. SQL Schema Seed Script

```sql
-- =============================================================================
-- DATA SCHEMAS SEED SCRIPT
-- Populates the data_schemas table with all 12 extractable document types
-- =============================================================================

-- 1. INVOICE SCHEMA
INSERT INTO data_schemas (
    id,
    schema_type,
    schema_name,
    description,
    header_schema,
    line_items_schema,
    summary_schema,
    field_mappings,
    extraction_prompt,
    validation_rules,
    version,
    is_active,
    created_at,
    updated_at
) VALUES (
    gen_random_uuid(),
    'invoice',
    'Invoice',
    'Schema for invoice documents including vendor invoices, billing documents, and purchase invoices',
    '{
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string", "description": "Unique invoice identifier"},
            "invoice_date": {"type": "string", "format": "date", "description": "Date of invoice in YYYY-MM-DD format"},
            "due_date": {"type": "string", "format": "date", "description": "Payment due date"},
            "vendor_name": {"type": "string", "description": "Name of the vendor/supplier"},
            "vendor_address": {"type": "string", "description": "Full address of the vendor"},
            "customer_name": {"type": "string", "description": "Name of the customer/buyer"},
            "customer_address": {"type": "string", "description": "Full address of the customer"},
            "payment_terms": {"type": "string", "description": "Payment terms (e.g., Net 30)"},
            "currency": {"type": "string", "description": "Currency code (e.g., CAD, USD, EUR)"},
            "po_number": {"type": "string", "description": "Purchase order reference number"}
        },
        "required": ["invoice_number"]
    }'::jsonb,
    '{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Item description"},
                "quantity": {"type": "number", "description": "Quantity ordered"},
                "unit_price": {"type": "number", "description": "Price per unit"},
                "amount": {"type": "number", "description": "Line total (quantity × unit_price)"},
                "sku": {"type": "string", "description": "Product SKU or code"},
                "tax_rate": {"type": "number", "description": "Tax rate percentage"}
            },
            "required": ["description", "amount"]
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "subtotal": {"type": "number", "description": "Sum before taxes"},
            "tax_amount": {"type": "number", "description": "Total tax amount"},
            "discount_amount": {"type": "number", "description": "Total discounts applied"},
            "shipping_amount": {"type": "number", "description": "Shipping/freight charges"},
            "total_amount": {"type": "number", "description": "Final total amount due"}
        },
        "required": ["total_amount"]
    }'::jsonb,
    '{
        "header_fields": {
            "invoice_number": {"type": "string", "source": "header", "jsonb_path": "header_data->>''invoice_number''"},
            "invoice_date": {"type": "date", "source": "header", "jsonb_path": "header_data->>''invoice_date''"},
            "due_date": {"type": "date", "source": "header", "jsonb_path": "header_data->>''due_date''"},
            "vendor_name": {"type": "string", "source": "header", "jsonb_path": "header_data->>''vendor_name''"},
            "vendor_address": {"type": "string", "source": "header", "jsonb_path": "header_data->>''vendor_address''"},
            "customer_name": {"type": "string", "source": "header", "jsonb_path": "header_data->>''customer_name''"},
            "customer_address": {"type": "string", "source": "header", "jsonb_path": "header_data->>''customer_address''"},
            "payment_terms": {"type": "string", "source": "header", "jsonb_path": "header_data->>''payment_terms''"},
            "currency": {"type": "string", "source": "header", "jsonb_path": "header_data->>''currency''"},
            "po_number": {"type": "string", "source": "header", "jsonb_path": "header_data->>''po_number''"},
            "subtotal": {"type": "number", "source": "summary", "jsonb_path": "summary_data->>''subtotal''"},
            "tax_amount": {"type": "number", "source": "summary", "jsonb_path": "summary_data->>''tax_amount''"},
            "total_amount": {"type": "number", "source": "summary", "jsonb_path": "summary_data->>''total_amount''"}
        },
        "line_item_fields": {
            "description": {"type": "string", "source": "line_item", "jsonb_path": "item->>''description''"},
            "quantity": {"type": "number", "source": "line_item", "jsonb_path": "(item->>''quantity'')::numeric"},
            "unit_price": {"type": "number", "source": "line_item", "jsonb_path": "(item->>''unit_price'')::numeric"},
            "amount": {"type": "number", "source": "line_item", "jsonb_path": "(item->>''amount'')::numeric"},
            "sku": {"type": "string", "source": "line_item", "jsonb_path": "item->>''sku''"},
            "tax_rate": {"type": "number", "source": "line_item", "jsonb_path": "(item->>''tax_rate'')::numeric"}
        }
    }'::jsonb,
    'Extract structured data from this invoice document.

Return a JSON object with the following structure:
{
    "header_data": {
        "invoice_number": "string or null",
        "invoice_date": "YYYY-MM-DD or null",
        "due_date": "YYYY-MM-DD or null",
        "vendor_name": "string or null",
        "vendor_address": "string or null",
        "customer_name": "string or null",
        "customer_address": "string or null",
        "payment_terms": "string or null",
        "currency": "string or null (e.g., CAD, USD, EUR - ONLY if explicitly shown)",
        "po_number": "string or null"
    },
    "line_items": [
        {
            "description": "string",
            "quantity": number,
            "unit_price": number,
            "amount": number,
            "sku": "string or null",
            "tax_rate": number or null
        }
    ],
    "summary_data": {
        "subtotal": number or null,
        "tax_amount": number or null,
        "discount_amount": number or null,
        "shipping_amount": number or null,
        "total_amount": number
    }
}

IMPORTANT:
- Use null for any field where the value is NOT explicitly visible in the document.
- Do NOT assume or guess values. Only extract what is actually shown.
- For currency, only include if explicitly printed. Do not assume based on location.
- Ensure numbers are actual numbers, not strings.
- Dates should be in YYYY-MM-DD format.',
    '{
        "required_fields": ["invoice_number", "total_amount"],
        "date_format": "YYYY-MM-DD",
        "number_fields": ["quantity", "unit_price", "amount", "subtotal", "tax_amount", "total_amount"],
        "validations": [
            {"rule": "total_equals_subtotal_plus_tax", "fields": ["subtotal", "tax_amount", "total_amount"]},
            {"rule": "line_amount_equals_qty_times_price", "fields": ["quantity", "unit_price", "amount"]}
        ]
    }'::jsonb,
    '1.0',
    true,
    NOW(),
    NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    schema_name = EXCLUDED.schema_name,
    description = EXCLUDED.description,
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    validation_rules = EXCLUDED.validation_rules,
    version = EXCLUDED.version,
    updated_at = NOW();

-- 2. RECEIPT SCHEMA
INSERT INTO data_schemas (
    id, schema_type, schema_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'receipt',
    'Receipt',
    'Schema for receipt documents including retail purchases, restaurant bills, and transaction receipts',
    '{
        "type": "object",
        "properties": {
            "receipt_number": {"type": "string", "description": "Receipt or transaction number"},
            "transaction_date": {"type": "string", "format": "date", "description": "Date of transaction"},
            "transaction_time": {"type": "string", "description": "Time of transaction (HH:MM)"},
            "store_name": {"type": "string", "description": "Name of the store or merchant"},
            "store_address": {"type": "string", "description": "Store location address"},
            "store_phone": {"type": "string", "description": "Store phone number"},
            "cashier": {"type": "string", "description": "Cashier name or ID"},
            "payment_method": {"type": "string", "description": "Payment method (cash, card, etc.)"},
            "card_last_four": {"type": "string", "description": "Last 4 digits of card used"},
            "currency": {"type": "string", "description": "Currency code"}
        }
    }'::jsonb,
    '{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Item description"},
                "quantity": {"type": "number", "description": "Quantity purchased"},
                "unit_price": {"type": "number", "description": "Price per unit"},
                "amount": {"type": "number", "description": "Line total"}
            },
            "required": ["description", "amount"]
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "subtotal": {"type": "number"},
            "tax_amount": {"type": "number"},
            "tip_amount": {"type": "number"},
            "total_amount": {"type": "number"},
            "amount_tendered": {"type": "number"},
            "change_given": {"type": "number"}
        },
        "required": ["total_amount"]
    }'::jsonb,
    '{
        "header_fields": {
            "receipt_number": {"type": "string", "source": "header"},
            "transaction_date": {"type": "date", "source": "header"},
            "transaction_time": {"type": "string", "source": "header"},
            "store_name": {"type": "string", "source": "header"},
            "store_address": {"type": "string", "source": "header"},
            "payment_method": {"type": "string", "source": "header"},
            "currency": {"type": "string", "source": "header"},
            "subtotal": {"type": "number", "source": "summary"},
            "tax_amount": {"type": "number", "source": "summary"},
            "total_amount": {"type": "number", "source": "summary"}
        },
        "line_item_fields": {
            "description": {"type": "string", "source": "line_item"},
            "quantity": {"type": "number", "source": "line_item"},
            "unit_price": {"type": "number", "source": "line_item"},
            "amount": {"type": "number", "source": "line_item"}
        }
    }'::jsonb,
    'Extract structured data from this receipt.

Return a JSON object with:
{
    "header_data": {
        "receipt_number": "string or null",
        "transaction_date": "YYYY-MM-DD or null",
        "transaction_time": "HH:MM or null",
        "store_name": "string or null",
        "store_address": "string or null",
        "payment_method": "string or null",
        "currency": "string or null (ONLY if explicitly shown)"
    },
    "line_items": [
        {
            "description": "string",
            "quantity": number,
            "unit_price": number,
            "amount": number
        }
    ],
    "summary_data": {
        "subtotal": number or null,
        "tax_amount": number or null,
        "tip_amount": number or null,
        "total_amount": number
    }
}

IMPORTANT:
- Use null for any field not explicitly visible.
- For currency: ONLY set if currency code is EXPLICITLY printed.
- The "$" symbol alone does NOT indicate USD.',
    '{
        "required_fields": ["total_amount"],
        "date_format": "YYYY-MM-DD",
        "time_format": "HH:MM"
    }'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    validation_rules = EXCLUDED.validation_rules,
    updated_at = NOW();

-- 3. BANK STATEMENT SCHEMA
INSERT INTO data_schemas (
    id, schema_type, schema_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'bank_statement',
    'Bank Statement',
    'Schema for bank account statements and transaction history',
    '{
        "type": "object",
        "properties": {
            "account_number": {"type": "string", "description": "Account number (masked)"},
            "account_holder": {"type": "string", "description": "Account holder name"},
            "bank_name": {"type": "string", "description": "Name of the bank"},
            "branch_address": {"type": "string", "description": "Branch address"},
            "statement_period_start": {"type": "string", "format": "date"},
            "statement_period_end": {"type": "string", "format": "date"},
            "currency": {"type": "string", "description": "Account currency"}
        }
    }'::jsonb,
    '{
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
            },
            "required": ["date", "description"]
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "opening_balance": {"type": "number"},
            "total_deposits": {"type": "number"},
            "total_withdrawals": {"type": "number"},
            "closing_balance": {"type": "number"},
            "total_fees": {"type": "number"},
            "interest_earned": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_fields": {
            "account_number": {"type": "string", "source": "header"},
            "account_holder": {"type": "string", "source": "header"},
            "bank_name": {"type": "string", "source": "header"},
            "statement_period_start": {"type": "date", "source": "header"},
            "statement_period_end": {"type": "date", "source": "header"},
            "currency": {"type": "string", "source": "header"},
            "opening_balance": {"type": "number", "source": "summary"},
            "closing_balance": {"type": "number", "source": "summary"},
            "total_deposits": {"type": "number", "source": "summary"},
            "total_withdrawals": {"type": "number", "source": "summary"}
        },
        "line_item_fields": {
            "date": {"type": "date", "source": "line_item"},
            "description": {"type": "string", "source": "line_item"},
            "reference": {"type": "string", "source": "line_item"},
            "debit": {"type": "number", "source": "line_item"},
            "credit": {"type": "number", "source": "line_item"},
            "balance": {"type": "number", "source": "line_item"}
        }
    }'::jsonb,
    'Extract structured data from this bank statement.

Return a JSON object with:
{
    "header_data": {
        "account_number": "string (last 4 digits only for security)",
        "account_holder": "string or null",
        "bank_name": "string or null",
        "statement_period_start": "YYYY-MM-DD or null",
        "statement_period_end": "YYYY-MM-DD or null",
        "currency": "string or null"
    },
    "line_items": [
        {
            "date": "YYYY-MM-DD",
            "description": "string",
            "reference": "string or null",
            "debit": number or null,
            "credit": number or null,
            "balance": number or null
        }
    ],
    "summary_data": {
        "opening_balance": number or null,
        "total_deposits": number or null,
        "total_withdrawals": number or null,
        "closing_balance": number or null
    }
}

List all transactions in chronological order.',
    '{
        "required_fields": ["account_number"],
        "security_fields": ["account_number"],
        "mask_rules": {"account_number": "last_4_only"}
    }'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    updated_at = NOW();

-- 4. FINANCIAL REPORT SCHEMA
INSERT INTO data_schemas (
    id, schema_type, schema_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'financial_report',
    'Financial Report',
    'Schema for financial statements, earnings reports, and budget reports',
    '{
        "type": "object",
        "properties": {
            "report_title": {"type": "string"},
            "company_name": {"type": "string"},
            "report_period": {"type": "string"},
            "report_date": {"type": "string", "format": "date"},
            "report_type": {"type": "string", "description": "Income Statement, Balance Sheet, etc."},
            "currency": {"type": "string"},
            "fiscal_year": {"type": "string"}
        }
    }'::jsonb,
    '{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "line_item": {"type": "string"},
                "current_period": {"type": "number"},
                "prior_period": {"type": "number"},
                "variance": {"type": "number"},
                "variance_percent": {"type": "number"}
            }
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "total_revenue": {"type": "number"},
            "total_expenses": {"type": "number"},
            "net_income": {"type": "number"},
            "total_assets": {"type": "number"},
            "total_liabilities": {"type": "number"},
            "total_equity": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_fields": {
            "report_title": {"type": "string", "source": "header"},
            "company_name": {"type": "string", "source": "header"},
            "report_period": {"type": "string", "source": "header"},
            "report_date": {"type": "date", "source": "header"},
            "report_type": {"type": "string", "source": "header"},
            "currency": {"type": "string", "source": "header"},
            "total_revenue": {"type": "number", "source": "summary"},
            "total_expenses": {"type": "number", "source": "summary"},
            "net_income": {"type": "number", "source": "summary"}
        },
        "line_item_fields": {
            "category": {"type": "string", "source": "line_item"},
            "line_item": {"type": "string", "source": "line_item"},
            "current_period": {"type": "number", "source": "line_item"},
            "prior_period": {"type": "number", "source": "line_item"},
            "variance": {"type": "number", "source": "line_item"}
        }
    }'::jsonb,
    'Extract structured data from this financial report.

Return a JSON object with:
{
    "header_data": {
        "report_title": "string",
        "company_name": "string or null",
        "report_period": "string (e.g., Q1 2025, FY 2024)",
        "report_date": "YYYY-MM-DD or null",
        "report_type": "string (Income Statement, Balance Sheet, etc.)",
        "currency": "string or null"
    },
    "line_items": [
        {
            "category": "string (Revenue, Expenses, Assets, etc.)",
            "line_item": "string",
            "current_period": number,
            "prior_period": number or null,
            "variance": number or null
        }
    ],
    "summary_data": {
        "total_revenue": number or null,
        "total_expenses": number or null,
        "net_income": number or null,
        "total_assets": number or null,
        "total_liabilities": number or null,
        "total_equity": number or null
    }
}',
    '{"required_fields": ["report_title"]}'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    updated_at = NOW();

-- 5. EXPENSE REPORT SCHEMA
INSERT INTO data_schemas (
    id, schema_type, schema_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'expense_report',
    'Expense Report',
    'Schema for expense claims and reimbursement requests',
    '{
        "type": "object",
        "properties": {
            "report_id": {"type": "string"},
            "employee_name": {"type": "string"},
            "employee_id": {"type": "string"},
            "department": {"type": "string"},
            "report_period_start": {"type": "string", "format": "date"},
            "report_period_end": {"type": "string", "format": "date"},
            "submission_date": {"type": "string", "format": "date"},
            "approval_status": {"type": "string"},
            "currency": {"type": "string"}
        }
    }'::jsonb,
    '{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "format": "date"},
                "category": {"type": "string", "description": "Travel, Meals, Lodging, etc."},
                "description": {"type": "string"},
                "vendor": {"type": "string"},
                "amount": {"type": "number"},
                "receipt_attached": {"type": "boolean"},
                "billable": {"type": "boolean"},
                "project_code": {"type": "string"}
            }
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "total_claimed": {"type": "number"},
            "total_approved": {"type": "number"},
            "total_rejected": {"type": "number"},
            "by_category": {"type": "object"}
        }
    }'::jsonb,
    '{
        "header_fields": {
            "report_id": {"type": "string", "source": "header"},
            "employee_name": {"type": "string", "source": "header"},
            "department": {"type": "string", "source": "header"},
            "report_period_start": {"type": "date", "source": "header"},
            "report_period_end": {"type": "date", "source": "header"},
            "submission_date": {"type": "date", "source": "header"},
            "currency": {"type": "string", "source": "header"},
            "total_claimed": {"type": "number", "source": "summary"},
            "total_approved": {"type": "number", "source": "summary"}
        },
        "line_item_fields": {
            "date": {"type": "date", "source": "line_item"},
            "category": {"type": "string", "source": "line_item"},
            "description": {"type": "string", "source": "line_item"},
            "vendor": {"type": "string", "source": "line_item"},
            "amount": {"type": "number", "source": "line_item"},
            "project_code": {"type": "string", "source": "line_item"}
        }
    }'::jsonb,
    'Extract structured data from this expense report.

Return a JSON object with:
{
    "header_data": {
        "report_id": "string or null",
        "employee_name": "string or null",
        "employee_id": "string or null",
        "department": "string or null",
        "report_period_start": "YYYY-MM-DD or null",
        "report_period_end": "YYYY-MM-DD or null",
        "submission_date": "YYYY-MM-DD or null",
        "currency": "string or null"
    },
    "line_items": [
        {
            "date": "YYYY-MM-DD",
            "category": "string (Travel, Meals, Lodging, etc.)",
            "description": "string",
            "vendor": "string or null",
            "amount": number,
            "project_code": "string or null"
        }
    ],
    "summary_data": {
        "total_claimed": number,
        "total_approved": number or null
    }
}',
    '{"required_fields": ["employee_name", "total_claimed"]}'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    updated_at = NOW();

-- 6. PURCHASE ORDER SCHEMA
INSERT INTO data_schemas (
    id, schema_type, schema_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'purchase_order',
    'Purchase Order',
    'Schema for purchase orders and procurement documents',
    '{
        "type": "object",
        "properties": {
            "po_number": {"type": "string"},
            "po_date": {"type": "string", "format": "date"},
            "delivery_date": {"type": "string", "format": "date"},
            "vendor_name": {"type": "string"},
            "vendor_address": {"type": "string"},
            "vendor_contact": {"type": "string"},
            "ship_to_name": {"type": "string"},
            "ship_to_address": {"type": "string"},
            "bill_to_name": {"type": "string"},
            "bill_to_address": {"type": "string"},
            "payment_terms": {"type": "string"},
            "shipping_method": {"type": "string"},
            "currency": {"type": "string"}
        }
    }'::jsonb,
    '{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "line_number": {"type": "number"},
                "sku": {"type": "string"},
                "description": {"type": "string"},
                "quantity": {"type": "number"},
                "unit": {"type": "string"},
                "unit_price": {"type": "number"},
                "amount": {"type": "number"},
                "delivery_date": {"type": "string", "format": "date"}
            }
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "subtotal": {"type": "number"},
            "tax_amount": {"type": "number"},
            "shipping_amount": {"type": "number"},
            "discount_amount": {"type": "number"},
            "total_amount": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_fields": {
            "po_number": {"type": "string", "source": "header"},
            "po_date": {"type": "date", "source": "header"},
            "delivery_date": {"type": "date", "source": "header"},
            "vendor_name": {"type": "string", "source": "header"},
            "vendor_address": {"type": "string", "source": "header"},
            "ship_to_address": {"type": "string", "source": "header"},
            "payment_terms": {"type": "string", "source": "header"},
            "currency": {"type": "string", "source": "header"},
            "subtotal": {"type": "number", "source": "summary"},
            "total_amount": {"type": "number", "source": "summary"}
        },
        "line_item_fields": {
            "line_number": {"type": "number", "source": "line_item"},
            "sku": {"type": "string", "source": "line_item"},
            "description": {"type": "string", "source": "line_item"},
            "quantity": {"type": "number", "source": "line_item"},
            "unit": {"type": "string", "source": "line_item"},
            "unit_price": {"type": "number", "source": "line_item"},
            "amount": {"type": "number", "source": "line_item"}
        }
    }'::jsonb,
    'Extract structured data from this purchase order.

Return a JSON object with:
{
    "header_data": {
        "po_number": "string",
        "po_date": "YYYY-MM-DD or null",
        "delivery_date": "YYYY-MM-DD or null",
        "vendor_name": "string or null",
        "vendor_address": "string or null",
        "ship_to_address": "string or null",
        "payment_terms": "string or null",
        "currency": "string or null"
    },
    "line_items": [
        {
            "line_number": number,
            "sku": "string or null",
            "description": "string",
            "quantity": number,
            "unit": "string or null",
            "unit_price": number,
            "amount": number
        }
    ],
    "summary_data": {
        "subtotal": number or null,
        "tax_amount": number or null,
        "shipping_amount": number or null,
        "total_amount": number
    }
}',
    '{"required_fields": ["po_number", "total_amount"]}'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    updated_at = NOW();

-- 7. TAX DOCUMENT SCHEMA
INSERT INTO data_schemas (
    id, schema_type, schema_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'tax_document',
    'Tax Document',
    'Schema for tax forms, tax returns, and tax-related documents',
    '{
        "type": "object",
        "properties": {
            "form_type": {"type": "string", "description": "W-2, 1099, T4, etc."},
            "tax_year": {"type": "string"},
            "taxpayer_name": {"type": "string"},
            "taxpayer_id": {"type": "string", "description": "SSN/SIN (masked)"},
            "employer_name": {"type": "string"},
            "employer_id": {"type": "string"},
            "filing_status": {"type": "string"}
        }
    }'::jsonb,
    '{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "box_number": {"type": "string"},
                "description": {"type": "string"},
                "amount": {"type": "number"}
            }
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "gross_income": {"type": "number"},
            "federal_tax_withheld": {"type": "number"},
            "state_tax_withheld": {"type": "number"},
            "social_security_wages": {"type": "number"},
            "medicare_wages": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_fields": {
            "form_type": {"type": "string", "source": "header"},
            "tax_year": {"type": "string", "source": "header"},
            "taxpayer_name": {"type": "string", "source": "header"},
            "employer_name": {"type": "string", "source": "header"},
            "gross_income": {"type": "number", "source": "summary"},
            "federal_tax_withheld": {"type": "number", "source": "summary"}
        },
        "line_item_fields": {
            "box_number": {"type": "string", "source": "line_item"},
            "description": {"type": "string", "source": "line_item"},
            "amount": {"type": "number", "source": "line_item"}
        }
    }'::jsonb,
    'Extract structured data from this tax document.

Return a JSON object with:
{
    "header_data": {
        "form_type": "string (W-2, 1099, T4, etc.)",
        "tax_year": "string (e.g., 2024)",
        "taxpayer_name": "string or null",
        "taxpayer_id": "string (last 4 digits only)",
        "employer_name": "string or null",
        "employer_id": "string or null"
    },
    "line_items": [
        {
            "box_number": "string",
            "description": "string",
            "amount": number
        }
    ],
    "summary_data": {
        "gross_income": number or null,
        "federal_tax_withheld": number or null,
        "state_tax_withheld": number or null
    }
}

SECURITY: Only include last 4 digits of taxpayer IDs.',
    '{
        "required_fields": ["form_type", "tax_year"],
        "security_fields": ["taxpayer_id"],
        "mask_rules": {"taxpayer_id": "last_4_only"}
    }'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    updated_at = NOW();

-- 8. SHIPPING MANIFEST SCHEMA
INSERT INTO data_schemas (
    id, schema_type, schema_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'shipping_manifest',
    'Shipping Manifest',
    'Schema for shipping documents, delivery notes, and packing lists',
    '{
        "type": "object",
        "properties": {
            "manifest_number": {"type": "string"},
            "ship_date": {"type": "string", "format": "date"},
            "delivery_date": {"type": "string", "format": "date"},
            "shipper_name": {"type": "string"},
            "shipper_address": {"type": "string"},
            "consignee_name": {"type": "string"},
            "consignee_address": {"type": "string"},
            "carrier_name": {"type": "string"},
            "tracking_number": {"type": "string"},
            "shipping_method": {"type": "string"},
            "incoterms": {"type": "string"}
        }
    }'::jsonb,
    '{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "item_number": {"type": "string"},
                "description": {"type": "string"},
                "quantity": {"type": "number"},
                "unit": {"type": "string"},
                "weight": {"type": "number"},
                "weight_unit": {"type": "string"},
                "dimensions": {"type": "string"},
                "hs_code": {"type": "string"}
            }
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "total_packages": {"type": "number"},
            "total_weight": {"type": "number"},
            "total_volume": {"type": "number"},
            "freight_charges": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_fields": {
            "manifest_number": {"type": "string", "source": "header"},
            "ship_date": {"type": "date", "source": "header"},
            "delivery_date": {"type": "date", "source": "header"},
            "shipper_name": {"type": "string", "source": "header"},
            "consignee_name": {"type": "string", "source": "header"},
            "carrier_name": {"type": "string", "source": "header"},
            "tracking_number": {"type": "string", "source": "header"},
            "total_packages": {"type": "number", "source": "summary"},
            "total_weight": {"type": "number", "source": "summary"}
        },
        "line_item_fields": {
            "item_number": {"type": "string", "source": "line_item"},
            "description": {"type": "string", "source": "line_item"},
            "quantity": {"type": "number", "source": "line_item"},
            "unit": {"type": "string", "source": "line_item"},
            "weight": {"type": "number", "source": "line_item"},
            "hs_code": {"type": "string", "source": "line_item"}
        }
    }'::jsonb,
    'Extract structured data from this shipping manifest/delivery note.

Return a JSON object with:
{
    "header_data": {
        "manifest_number": "string or null",
        "ship_date": "YYYY-MM-DD or null",
        "delivery_date": "YYYY-MM-DD or null",
        "shipper_name": "string or null",
        "shipper_address": "string or null",
        "consignee_name": "string or null",
        "consignee_address": "string or null",
        "carrier_name": "string or null",
        "tracking_number": "string or null"
    },
    "line_items": [
        {
            "item_number": "string or null",
            "description": "string",
            "quantity": number,
            "unit": "string or null",
            "weight": number or null,
            "weight_unit": "string or null"
        }
    ],
    "summary_data": {
        "total_packages": number or null,
        "total_weight": number or null,
        "freight_charges": number or null
    }
}',
    '{"required_fields": ["manifest_number"]}'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    updated_at = NOW();

-- 9. BILL OF LADING SCHEMA
INSERT INTO data_schemas (
    id, schema_type, schema_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'bill_of_lading',
    'Bill of Lading',
    'Schema for bills of lading (ocean, air, truck freight documents)',
    '{
        "type": "object",
        "properties": {
            "bol_number": {"type": "string"},
            "booking_number": {"type": "string"},
            "vessel_name": {"type": "string"},
            "voyage_number": {"type": "string"},
            "port_of_loading": {"type": "string"},
            "port_of_discharge": {"type": "string"},
            "shipper_name": {"type": "string"},
            "shipper_address": {"type": "string"},
            "consignee_name": {"type": "string"},
            "consignee_address": {"type": "string"},
            "notify_party": {"type": "string"},
            "ship_date": {"type": "string", "format": "date"},
            "freight_terms": {"type": "string"}
        }
    }'::jsonb,
    '{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "container_number": {"type": "string"},
                "seal_number": {"type": "string"},
                "description": {"type": "string"},
                "package_count": {"type": "number"},
                "package_type": {"type": "string"},
                "gross_weight": {"type": "number"},
                "weight_unit": {"type": "string"},
                "measurement": {"type": "number"},
                "measurement_unit": {"type": "string"}
            }
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "total_containers": {"type": "number"},
            "total_packages": {"type": "number"},
            "total_gross_weight": {"type": "number"},
            "total_measurement": {"type": "number"},
            "freight_amount": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_fields": {
            "bol_number": {"type": "string", "source": "header"},
            "booking_number": {"type": "string", "source": "header"},
            "vessel_name": {"type": "string", "source": "header"},
            "port_of_loading": {"type": "string", "source": "header"},
            "port_of_discharge": {"type": "string", "source": "header"},
            "shipper_name": {"type": "string", "source": "header"},
            "consignee_name": {"type": "string", "source": "header"},
            "ship_date": {"type": "date", "source": "header"},
            "total_containers": {"type": "number", "source": "summary"},
            "total_gross_weight": {"type": "number", "source": "summary"}
        },
        "line_item_fields": {
            "container_number": {"type": "string", "source": "line_item"},
            "seal_number": {"type": "string", "source": "line_item"},
            "description": {"type": "string", "source": "line_item"},
            "package_count": {"type": "number", "source": "line_item"},
            "gross_weight": {"type": "number", "source": "line_item"},
            "measurement": {"type": "number", "source": "line_item"}
        }
    }'::jsonb,
    'Extract structured data from this bill of lading.

Return a JSON object with:
{
    "header_data": {
        "bol_number": "string",
        "booking_number": "string or null",
        "vessel_name": "string or null",
        "voyage_number": "string or null",
        "port_of_loading": "string or null",
        "port_of_discharge": "string or null",
        "shipper_name": "string or null",
        "consignee_name": "string or null",
        "ship_date": "YYYY-MM-DD or null",
        "freight_terms": "string or null"
    },
    "line_items": [
        {
            "container_number": "string or null",
            "seal_number": "string or null",
            "description": "string",
            "package_count": number,
            "package_type": "string or null",
            "gross_weight": number or null,
            "weight_unit": "string or null"
        }
    ],
    "summary_data": {
        "total_containers": number or null,
        "total_packages": number or null,
        "total_gross_weight": number or null,
        "freight_amount": number or null
    }
}',
    '{"required_fields": ["bol_number"]}'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    updated_at = NOW();

-- 10. CUSTOMS DECLARATION SCHEMA
INSERT INTO data_schemas (
    id, schema_type, schema_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'customs_declaration',
    'Customs Declaration',
    'Schema for customs declarations and import/export documents',
    '{
        "type": "object",
        "properties": {
            "declaration_number": {"type": "string"},
            "declaration_date": {"type": "string", "format": "date"},
            "declaration_type": {"type": "string", "description": "Import, Export, Transit"},
            "importer_name": {"type": "string"},
            "importer_address": {"type": "string"},
            "exporter_name": {"type": "string"},
            "exporter_address": {"type": "string"},
            "country_of_origin": {"type": "string"},
            "country_of_destination": {"type": "string"},
            "port_of_entry": {"type": "string"},
            "transport_mode": {"type": "string"},
            "currency": {"type": "string"}
        }
    }'::jsonb,
    '{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "line_number": {"type": "number"},
                "hs_code": {"type": "string"},
                "description": {"type": "string"},
                "quantity": {"type": "number"},
                "unit": {"type": "string"},
                "unit_value": {"type": "number"},
                "total_value": {"type": "number"},
                "origin_country": {"type": "string"},
                "duty_rate": {"type": "number"},
                "duty_amount": {"type": "number"}
            }
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "total_declared_value": {"type": "number"},
            "total_duty": {"type": "number"},
            "total_tax": {"type": "number"},
            "total_fees": {"type": "number"},
            "total_payable": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_fields": {
            "declaration_number": {"type": "string", "source": "header"},
            "declaration_date": {"type": "date", "source": "header"},
            "declaration_type": {"type": "string", "source": "header"},
            "importer_name": {"type": "string", "source": "header"},
            "exporter_name": {"type": "string", "source": "header"},
            "country_of_origin": {"type": "string", "source": "header"},
            "port_of_entry": {"type": "string", "source": "header"},
            "currency": {"type": "string", "source": "header"},
            "total_declared_value": {"type": "number", "source": "summary"},
            "total_duty": {"type": "number", "source": "summary"},
            "total_payable": {"type": "number", "source": "summary"}
        },
        "line_item_fields": {
            "hs_code": {"type": "string", "source": "line_item"},
            "description": {"type": "string", "source": "line_item"},
            "quantity": {"type": "number", "source": "line_item"},
            "unit_value": {"type": "number", "source": "line_item"},
            "total_value": {"type": "number", "source": "line_item"},
            "duty_rate": {"type": "number", "source": "line_item"},
            "duty_amount": {"type": "number", "source": "line_item"}
        }
    }'::jsonb,
    'Extract structured data from this customs declaration.

Return a JSON object with:
{
    "header_data": {
        "declaration_number": "string",
        "declaration_date": "YYYY-MM-DD or null",
        "declaration_type": "string (Import, Export, Transit)",
        "importer_name": "string or null",
        "exporter_name": "string or null",
        "country_of_origin": "string or null",
        "country_of_destination": "string or null",
        "port_of_entry": "string or null",
        "currency": "string or null"
    },
    "line_items": [
        {
            "hs_code": "string",
            "description": "string",
            "quantity": number,
            "unit": "string or null",
            "unit_value": number,
            "total_value": number,
            "duty_rate": number or null,
            "duty_amount": number or null
        }
    ],
    "summary_data": {
        "total_declared_value": number,
        "total_duty": number or null,
        "total_tax": number or null,
        "total_payable": number or null
    }
}',
    '{"required_fields": ["declaration_number", "total_declared_value"]}'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    updated_at = NOW();

-- 11. INVENTORY REPORT SCHEMA
INSERT INTO data_schemas (
    id, schema_type, schema_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'inventory_report',
    'Inventory Report',
    'Schema for inventory reports and stock level documents',
    '{
        "type": "object",
        "properties": {
            "report_id": {"type": "string"},
            "report_date": {"type": "string", "format": "date"},
            "warehouse_name": {"type": "string"},
            "warehouse_location": {"type": "string"},
            "report_period_start": {"type": "string", "format": "date"},
            "report_period_end": {"type": "string", "format": "date"},
            "report_type": {"type": "string", "description": "Physical Count, Cycle Count, etc."}
        }
    }'::jsonb,
    '{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "product_name": {"type": "string"},
                "category": {"type": "string"},
                "location": {"type": "string"},
                "quantity_on_hand": {"type": "number"},
                "quantity_reserved": {"type": "number"},
                "quantity_available": {"type": "number"},
                "unit_cost": {"type": "number"},
                "total_value": {"type": "number"},
                "reorder_point": {"type": "number"},
                "last_count_date": {"type": "string", "format": "date"}
            }
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "total_skus": {"type": "number"},
            "total_quantity": {"type": "number"},
            "total_value": {"type": "number"},
            "items_below_reorder": {"type": "number"},
            "items_out_of_stock": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_fields": {
            "report_id": {"type": "string", "source": "header"},
            "report_date": {"type": "date", "source": "header"},
            "warehouse_name": {"type": "string", "source": "header"},
            "warehouse_location": {"type": "string", "source": "header"},
            "report_type": {"type": "string", "source": "header"},
            "total_skus": {"type": "number", "source": "summary"},
            "total_quantity": {"type": "number", "source": "summary"},
            "total_value": {"type": "number", "source": "summary"}
        },
        "line_item_fields": {
            "sku": {"type": "string", "source": "line_item"},
            "product_name": {"type": "string", "source": "line_item"},
            "category": {"type": "string", "source": "line_item"},
            "location": {"type": "string", "source": "line_item"},
            "quantity_on_hand": {"type": "number", "source": "line_item"},
            "quantity_available": {"type": "number", "source": "line_item"},
            "unit_cost": {"type": "number", "source": "line_item"},
            "total_value": {"type": "number", "source": "line_item"}
        }
    }'::jsonb,
    'Extract structured data from this inventory report.

Return a JSON object with:
{
    "header_data": {
        "report_id": "string or null",
        "report_date": "YYYY-MM-DD or null",
        "warehouse_name": "string or null",
        "warehouse_location": "string or null",
        "report_type": "string or null"
    },
    "line_items": [
        {
            "sku": "string",
            "product_name": "string",
            "category": "string or null",
            "location": "string or null",
            "quantity_on_hand": number,
            "quantity_reserved": number or null,
            "quantity_available": number or null,
            "unit_cost": number or null,
            "total_value": number or null
        }
    ],
    "summary_data": {
        "total_skus": number or null,
        "total_quantity": number or null,
        "total_value": number or null,
        "items_below_reorder": number or null
    }
}',
    '{"required_fields": ["report_date"]}'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    updated_at = NOW();

-- 12. SPREADSHEET SCHEMA (Dynamic/Generic)
INSERT INTO data_schemas (
    id, schema_type, schema_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'spreadsheet',
    'Spreadsheet',
    'Dynamic schema for spreadsheets, CSVs, and tabular data with unknown structure',
    '{
        "type": "object",
        "properties": {
            "sheet_name": {"type": "string"},
            "column_headers": {"type": "array", "items": {"type": "string"}},
            "total_rows": {"type": "number"},
            "total_columns": {"type": "number"},
            "has_header_row": {"type": "boolean"},
            "data_start_row": {"type": "number"}
        }
    }'::jsonb,
    '{
        "type": "array",
        "description": "Dynamic structure - fields determined at extraction time",
        "items": {
            "type": "object",
            "additionalProperties": true
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "row_count": {"type": "number"},
            "column_count": {"type": "number"},
            "numeric_columns": {"type": "array", "items": {"type": "string"}},
            "date_columns": {"type": "array", "items": {"type": "string"}}
        }
    }'::jsonb,
    '{
        "header_fields": {
            "sheet_name": {"type": "string", "source": "header"},
            "total_rows": {"type": "number", "source": "header"},
            "total_columns": {"type": "number", "source": "header"}
        },
        "line_item_fields": {},
        "dynamic": true,
        "note": "Line item fields are inferred at extraction time based on column headers"
    }'::jsonb,
    'Extract structured data from this spreadsheet.

Identify the column headers and extract all data rows.

Return a JSON object with:
{
    "header_data": {
        "sheet_name": "string",
        "column_headers": ["col1", "col2", ...],
        "total_rows": number,
        "total_columns": number
    },
    "line_items": [
        {
            "row_number": number,
            ... // key-value pairs for each column
        }
    ],
    "summary_data": {
        "row_count": number,
        "column_count": number
    }
}

IMPORTANT:
- Preserve the original column names as keys in line_items.
- Convert numeric values to numbers, dates to YYYY-MM-DD format.
- Handle empty cells as null values.
- Detect and use the first row as headers if it contains column names.',
    '{
        "dynamic_schema": true,
        "infer_types": true,
        "header_detection": "auto"
    }'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    updated_at = NOW();

-- =============================================================================
-- VERIFICATION QUERY
-- =============================================================================
-- Run this to verify all schemas were inserted:
-- SELECT schema_type, schema_name, is_active, created_at FROM data_schemas ORDER BY schema_type;
```

---

## Appendix A: File Change Summary

| File | Changes |
|------|---------|
| `backend/db/models.py` | No changes needed (extraction_metadata exists) |
| `backend/extraction_service/data_extractor.py` | Add schema lookup, cache field_mappings |
| `backend/analytics_service/sql_query_executor.py` | Priority-based field mapping lookup, multi-schema grouping |
| `backend/analytics_service/llm_sql_generator.py` | Inject schema context into prompts |
| `backend/extraction_service/extraction_config.py` | Deprecate hardcoded prompts (use DB) |

---

## Appendix B: Testing Checklist

- [ ] Schema seed SQL executes without errors
- [ ] All 12 schemas visible in data_schemas table
- [ ] Invoice extraction uses DB prompt
- [ ] Field mappings cached to extraction_metadata
- [ ] SQL generator receives schema context
- [ ] Multi-schema query returns separated results
- [ ] Dynamic spreadsheet schema inference works
- [ ] Error correction with schema context works

---

*Document Version: 1.0*
*Last Updated: December 2024*
*Author: Schema Design Team*
