# SQL Generation and Execution System Design

## Overview

This document describes the architecture and workflow for converting natural language user queries into SQL queries, executing them against the PostgreSQL database, and returning formatted results with summaries.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY (Natural Language)                       │
│                     "What were total sales by customer in 2024?"                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         sql_query_executor.py                                    │
│                         (ORCHESTRATOR)                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  execute_dynamic_sql_query() / execute_dynamic_sql_query_async()        │    │
│  │  - Entry point for LLM-driven SQL workflow                              │    │
│  │  - Manages retry loop with error correction                             │    │
│  │  - Returns final formatted results                                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
           ┌──────────────┐    ┌──────────────────┐   ┌──────────────────┐
           │ Field Mapping │    │  LLM SQL Gen     │   │  SQL Execution   │
           │   Retrieval   │    │  (3 Rounds)      │   │  + Error Retry   │
           └──────────────┘    └──────────────────┘   └──────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         llm_sql_generator.py                                     │
│                         (SQL GENERATION TOOL)                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  generate_sql() - 3-round LLM analysis                                  │    │
│  │  correct_sql_error() - LLM-driven error correction                      │    │
│  │  generate_summary_report() - Result formatting with LLM                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Relationship

| Component | File | Role | Lines of Code |
|-----------|------|------|---------------|
| **SQL Query Executor** | `sql_query_executor.py` | Orchestrator - manages the entire workflow, executes SQL, handles retries | ~1,964 |
| **LLM SQL Generator** | `llm_sql_generator.py` | Tool - generates SQL using LLM, corrects errors, creates summaries | ~1,700 |

**Relationship**: The Executor **imports and calls** the Generator as a service:
```python
# In sql_query_executor.py (line 1062)
from analytics_service.llm_sql_generator import LLMSQLGenerator

generator = LLMSQLGenerator(llm_client)
sql_result = generator.generate_sql(query, field_mappings, doc_filter)
```

---

## Detailed Workflow

### Phase 1: Entry Point (Executor)

**File**: `sql_query_executor.py`
**Method**: `execute_dynamic_sql_query()` (line 1035) or `execute_dynamic_sql_query_async()` (line 1192)

```
User Query → Executor
     │
     ├─► 1. Get Field Mappings (HARDCODED LOGIC)
     │       └── _get_available_field_mappings() (line 1387)
     │           ├── Priority 1: header_data->'field_mappings' (cached)
     │           ├── Priority 2: DataSchema table lookup
     │           └── Priority 3: _infer_field_mappings_from_data() (line 1419)
     │
     ├─► 2. Build Document Filter (HARDCODED LOGIC)
     │       └── doc_filter = "dd.document_id IN ('uuid1', 'uuid2', ...)"
     │
     └─► 3. Call LLM SQL Generator → (continues to Phase 2)
```

**Hardcoded Logic in Executor**:
- Field mapping inference patterns (line 1458-1493)
- Semantic type classification (date, amount, entity, etc.)
- Data type inference from sample values (line 1555-1586)

---

### Phase 2: SQL Generation (Generator - 3 LLM Rounds)

**File**: `llm_sql_generator.py`
**Method**: `_generate_sql_multi_round()` (line 1073)

#### Round 1: Query Analysis (LLM INFERENCE)
**Purpose**: Understand user intent and extract query parameters

**Prompt**: `QUERY_ANALYSIS_PROMPT` (line 148-240)

```
User Query: "What were total sales by customer in 2024?"
                    │
                    ▼ (LLM Call)
                    │
┌───────────────────────────────────────────────────────┐
│ LLM Response (JSON):                                   │
│ {                                                      │
│   "grouping_order": ["customer"],                      │
│   "time_granularity": null,                            │
│   "time_filter_only": true,                            │
│   "time_filter_year": 2024,                            │
│   "aggregation_field": "amount",                       │
│   "aggregation_type": "sum",                           │
│   "entity_field": "Customer Name",                     │
│   "report_type": "summary",                            │
│   "explanation": "User wants total sales grouped..."   │
│ }                                                      │
└───────────────────────────────────────────────────────┘
```

**What LLM Determines**:
- `grouping_order`: Primary/secondary grouping hierarchy
- `time_granularity`: yearly, monthly, quarterly, or null
- `time_filter_only`: Filter vs Group by time
- `aggregation_type`: sum, min, max, min_max, avg, count
- `report_type`: summary, detail, comparison
- `entity_field`: Field to identify min/max records

---

#### Round 2: Field Mapping (LLM INFERENCE)
**Purpose**: Map user's requested fields to actual schema field names

**Prompt**: `FIELD_MAPPING_PROMPT` (line 243-285)

```
Round 1 Output + Field Schema
                    │
                    ▼ (LLM Call)
                    │
┌───────────────────────────────────────────────────────┐
│ LLM Response (JSON):                                   │
│ {                                                      │
│   "grouping_fields": [                                 │
│     {"requested": "customer",                          │
│      "actual_field": "Customer Name",                  │
│      "is_time_grouping": false}                        │
│   ],                                                   │
│   "aggregation_field": {                               │
│     "requested": "amount",                             │
│     "actual_field": "Total Sales"                      │
│   },                                                   │
│   "date_field": "Purchase Date",                       │
│   "success": true                                      │
│ }                                                      │
└───────────────────────────────────────────────────────┘
```

**What LLM Determines**:
- Maps "customer" → "Customer Name" (actual field)
- Maps "amount" → "Total Sales" (actual field)
- Identifies date field for time operations
- Validates all fields exist in schema

---

#### Round 3: SQL Generation (LLM INFERENCE)
**Purpose**: Generate verified PostgreSQL query

**Prompt**: `VERIFIED_SQL_PROMPT` (line 345-800)

```
Round 2 Output + Verified Parameters
                    │
                    ▼ (LLM Call)
                    │
┌───────────────────────────────────────────────────────────────────────┐
│ LLM Response (JSON):                                                   │
│ {                                                                      │
│   "sql": "WITH expanded_items AS (                                     │
│              SELECT dd.header_data,                                    │
│                     jsonb_array_elements(dd.line_items) as item        │
│              FROM documents_data dd                                    │
│              JOIN documents d ON dd.document_id = d.id                 │
│              WHERE dd.document_id IN (...)                             │
│           )                                                            │
│           SELECT                                                       │
│              item->>'Customer Name' as customer,                       │
│              ROUND(SUM((item->>'Total Sales')::numeric), 2) as total   │
│           FROM expanded_items                                          │
│           WHERE EXTRACT(YEAR FROM (...)) = 2024                        │
│           GROUP BY item->>'Customer Name'                              │
│           ORDER BY total DESC",                                        │
│   "explanation": "Groups sales by customer for 2024...",               │
│   "select_fields": ["customer", "total"],                              │
│   "group_by_fields": ["Customer Name"]                                 │
│ }                                                                      │
└───────────────────────────────────────────────────────────────────────┘
```

**SQL Templates Used (HARDCODED in prompts)**:
- CTE-based query structure with `expanded_items`
- JSONB access patterns: `item->>'field'`, `header_data->>'field'`
- Time grouping SQL: `EXTRACT(YEAR FROM ...)`, `TO_CHAR(..., 'YYYY-MM')`
- Aggregation patterns for sum, min, max, min_max, avg

---

### Phase 3: SQL Execution with Retry Loop

**File**: `sql_query_executor.py`
**Location**: Lines 1094-1190

```
Generated SQL
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  EXECUTION LOOP (max 3 correction attempts)                      │
│                                                                  │
│  attempt = 0                                                     │
│  while attempt <= max_correction_attempts:                       │
│      try:                                                        │
│          result = db.execute(text(current_sql))  ─────► SUCCESS  │
│          return formatted_results                                │
│      except Exception as e:                                      │
│          db.rollback()                                           │
│          if attempt < max:                                       │
│              corrected = generator.correct_sql_error(...)        │
│              current_sql = corrected.sql_query                   │
│              attempt += 1                                        │
│              continue  ◄────────────────────────────────────────┐│
│          else:                                                  ││
│              return error_response                              ││
│                                                                 ││
└─────────────────────────────────────────────────────────────────┘│
                              │                                    │
                              ▼ (On Error)                         │
┌─────────────────────────────────────────────────────────────────┐│
│  LLM ERROR CORRECTION (llm_sql_generator.py)                    ││
│                                                                  ││
│  Prompt: SQL_ERROR_CORRECTION_PROMPT (line 288-342)             ││
│                                                                  ││
│  Input:                                                          ││
│    - Original user query                                         ││
│    - Failed SQL                                                  ││
│    - PostgreSQL error message                                    ││
│    - Field schema                                                ││
│                                                                  ││
│  Output:                                                         ││
│    - Analysis of error cause                                     ││
│    - Corrected SQL query                                         ││
│    - Confidence score                                            ││
│                                                                  ││
└──────────────────────────────────────────────────────────────────┘
```

**Hardcoded Logic**:
- Transaction rollback on error
- Maximum retry count (default: 3)
- Correction history tracking

**LLM Inference**:
- Error analysis and root cause identification
- SQL correction based on error message

---

### Phase 4: Summary Report Generation

**File**: `llm_sql_generator.py`
**Method**: `generate_summary_report()` (line 99-145 prompt)

```
Query Results (JSON array)
          │
          ▼ (LLM Call)
          │
┌───────────────────────────────────────────────────────────────────────┐
│ LLM Response (JSON):                                                   │
│ {                                                                      │
│   "report_title": "Total Sales by Customer in 2024",                  │
│   "hierarchical_data": [                                               │
│     {                                                                  │
│       "group_name": "Acme Corp",                                       │
│       "group_total": 45000.00,                                         │
│       "sub_groups": [...],                                             │
│       "min_record": {"name": "Widget A", "amount": 150.00},            │
│       "max_record": {"name": "Widget Z", "amount": 5000.00}            │
│     }                                                                  │
│   ],                                                                   │
│   "grand_total": 125000.00,                                            │
│   "total_records": 42,                                                 │
│   "formatted_report": "In 2024, you had sales across 5 customers..."  │
│ }                                                                      │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                 COMPLETE WORKFLOW                                 │
└──────────────────────────────────────────────────────────────────────────────────┘

User Query: "Show me max and min receipts with details in 2025"
                    │
                    ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ [1] EXECUTOR: execute_dynamic_sql_query_async()                                   │
│     └─► Progress: "Analyzing data structure..."                                   │
│                                                                                   │
│ [2] EXECUTOR: _get_available_field_mappings()                       [HARDCODED]   │
│     ├─► Check header_data->'field_mappings'                                       │
│     ├─► Check DataSchema table                                                    │
│     └─► Infer from line_items structure                                           │
│                                                                                   │
│     Field Mappings Result:                                                        │
│     {                                                                             │
│       "Total Sales": {data_type: "number", source: "line_item", aggregation: "sum"}│
│       "Purchase Date": {data_type: "datetime", source: "line_item"}               │
│       "Customer Name": {data_type: "string", source: "line_item", aggregation: "group_by"}│
│     }                                                                             │
└───────────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ [3] EXECUTOR: Progress: "Building database query..."                              │
│                                                                                   │
│ [4] GENERATOR: generate_sql() → _generate_sql_multi_round()                       │
│                                                                                   │
│     ┌─────────────────────────────────────────────────────────────────────────┐   │
│     │ ROUND 1: Query Analysis                                   [LLM CALL]    │   │
│     │ Input: User query + Field schema                                        │   │
│     │ Output: {                                                               │   │
│     │   aggregation_type: "min_max",                                          │   │
│     │   time_filter_only: true,                                               │   │
│     │   time_filter_year: 2025,                                               │   │
│     │   report_type: "detail",                                                │   │
│     │   entity_field: "receipt_number"                                        │   │
│     │ }                                                                       │   │
│     └─────────────────────────────────────────────────────────────────────────┘   │
│                    │                                                              │
│                    ▼                                                              │
│     ┌─────────────────────────────────────────────────────────────────────────┐   │
│     │ ROUND 2: Field Mapping                                    [LLM CALL]    │   │
│     │ Input: Round 1 output + Field schema                                    │   │
│     │ Output: {                                                               │   │
│     │   aggregation_field: {actual_field: "Total Sales"},                     │   │
│     │   entity_field: {actual_field: "receipt_number"},                       │   │
│     │   date_field: "transaction_date"                                        │   │
│     │ }                                                                       │   │
│     └─────────────────────────────────────────────────────────────────────────┘   │
│                    │                                                              │
│                    ▼                                                              │
│     ┌─────────────────────────────────────────────────────────────────────────┐   │
│     │ ROUND 3: SQL Generation                                   [LLM CALL]    │   │
│     │ Input: Verified parameters + SQL templates                              │   │
│     │ Output: Complete PostgreSQL query with CTEs                             │   │
│     │                                                                         │   │
│     │ Uses HARDCODED SQL templates from VERIFIED_SQL_PROMPT:                  │   │
│     │ - min_max + detail template (line 546-608)                              │   │
│     │ - Proper JSONB access patterns                                          │   │
│     │ - WHERE clause placement rules                                          │   │
│     └─────────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ [5] GENERATOR: _add_table_filter()                               [HARDCODED]     │
│     └─► Inject document_id filter into CTE WHERE clause                          │
│                                                                                   │
│ [6] EXECUTOR: Progress: "Running database query..."                               │
│                                                                                   │
│ [7] EXECUTOR: db.execute(text(sql))                              [HARDCODED]     │
│     ├─► SUCCESS → Continue to [9]                                                │
│     └─► FAILURE → Go to [8]                                                      │
│                                                                                   │
│ [8] ERROR CORRECTION LOOP (up to 3 attempts):                                    │
│     ├─► db.rollback()                                            [HARDCODED]     │
│     ├─► GENERATOR: correct_sql_error()                           [LLM CALL]      │
│     │   Input: Failed SQL + Error message + Schema                               │
│     │   Output: Corrected SQL with fix explanation                               │
│     └─► Retry execution with corrected SQL                                       │
└───────────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ [9] EXECUTOR: Progress: "Found X records, preparing results..."                   │
│                                                                                   │
│ [10] GENERATOR: generate_summary_report()                        [LLM CALL]      │
│      Input: Query results + User query + Field mappings                          │
│      Output: {                                                                   │
│        hierarchical_data: [...],                                                 │
│        formatted_report: "In 2025, the maximum receipt was...",                  │
│        grand_total: 12500.00,                                                    │
│        min_record: {...},                                                        │
│        max_record: {...}                                                         │
│      }                                                                           │
│                                                                                   │
│ [11] EXECUTOR: Return final result                               [HARDCODED]     │
│      {                                                                           │
│        data: [...],                                                              │
│        summary: {...},                                                           │
│        metadata: {                                                               │
│          generated_sql: "...",                                                   │
│          correction_attempts: 0,                                                 │
│          ...                                                                     │
│        }                                                                         │
│      }                                                                           │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## LLM vs Hardcoded Logic Summary

### LLM Inference Points (5 total)

| Step | Location | Purpose |
|------|----------|---------|
| **Round 1** | `llm_sql_generator.py:1089-1118` | Query intent analysis |
| **Round 2** | `llm_sql_generator.py:1120-1140` | Field name mapping |
| **Round 3** | `llm_sql_generator.py:1142-1222` | SQL query generation |
| **Error Correction** | `llm_sql_generator.py:1224-1300` | Fix failed SQL |
| **Summary Report** | `llm_sql_generator.py` | Format results for display |

### Hardcoded Logic Points

| Component | Location | Purpose |
|-----------|----------|---------|
| **Field Mapping Retrieval** | `sql_query_executor.py:1387-1417` | Get schema from DB |
| **Field Type Inference** | `sql_query_executor.py:1419-1504` | Classify field semantics |
| **Data Type Detection** | `sql_query_executor.py:1555-1586` | Infer from sample values |
| **Document ID Filter** | `sql_query_executor.py:1076-1077` | Build access control filter |
| **Table Filter Injection** | `llm_sql_generator.py:1391-1466` | Add WHERE clause to SQL |
| **Retry Loop Logic** | `sql_query_executor.py:1094-1190` | Max attempts, rollback |
| **JSON Response Parsing** | `llm_sql_generator.py:1324-1348` | Extract JSON from LLM |
| **Heuristic SQL (fallback)** | `llm_sql_generator.py:1468-1599` | Non-LLM SQL generation |
| **Grouping Detection** | `sql_query_executor.py:893-924` | Keyword pattern matching |
| **Amount Filter Parsing** | `sql_query_executor.py:927-940` | Regex for "over/under $X" |

---

## SQL Templates in LLM Prompts

The `VERIFIED_SQL_PROMPT` contains hardcoded SQL templates that guide LLM generation:

### 1. Detail Report Template (line 400-422)
```sql
WITH expanded_items AS (
    SELECT dd.header_data, jsonb_array_elements(dd.line_items) as item
    FROM documents_data dd
    JOIN documents d ON dd.document_id = d.id
    {WHERE_CLAUSE}
)
SELECT
    header_data->>'receipt_number' as receipt_number,
    item->>'description' as item_description,
    ROUND((item->>'amount')::numeric, 2) as amount
FROM expanded_items
ORDER BY ...
```

### 2. Sum Aggregation Template (line 427-444)
```sql
SELECT {GROUPING_FIELDS},
       COUNT(*) as item_count,
       ROUND(SUM((item->>'amount')::numeric), 2) as total_amount
FROM expanded_items
GROUP BY {GROUP_BY_FIELDS}
```

### 3. Min/Max with Details Template (line 546-608)
```sql
WITH expanded_items AS (...),
receipt_totals AS (
    SELECT receipt_number, store_name, ..., SUM(amount) as total_amount
    GROUP BY ...
),
min_max_receipts AS (
    (SELECT receipt_number, 'MIN' FROM receipt_totals ORDER BY total_amount ASC LIMIT 1)
    UNION ALL
    (SELECT receipt_number, 'MAX' FROM receipt_totals ORDER BY total_amount DESC LIMIT 1)
)
SELECT mm.record_type, rt.*, ei.item_description, ei.item_amount
FROM min_max_receipts mm
JOIN receipt_totals rt ON ...
JOIN expanded_items ei ON ...
```

### 4. Average Per Receipt Template (line 449-481)
```sql
WITH expanded_items AS (...),
receipt_totals AS (
    SELECT receipt_number, SUM(amount) as receipt_total
    GROUP BY receipt_number
)
SELECT
    ROUND(AVG(receipt_total), 2) as average_per_receipt,
    COUNT(*) as total_receipts,
    ROUND(SUM(receipt_total), 2) as grand_total
FROM receipt_totals
```

---

## Database Schema

### documents_data Table
```sql
CREATE TABLE documents_data (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    schema_type VARCHAR(50),           -- 'invoice', 'receipt', 'spreadsheet'
    header_data JSONB,                 -- Document-level fields
    line_items JSONB,                  -- Array of row data
    line_items_count INTEGER,
    summary_data JSONB,
    extraction_method VARCHAR(50),
    validation_status VARCHAR(50),
    created_at TIMESTAMP
);
```

### JSONB Structure
```json
{
  "header_data": {
    "transaction_date": "2024-01-15",
    "store_name": "Walmart",
    "receipt_number": "REC-001",
    "field_mappings": {
      "Total Sales": {"data_type": "number", "source": "line_item", "aggregation": "sum"},
      "Customer Name": {"data_type": "string", "source": "line_item", "aggregation": "group_by"}
    }
  },
  "line_items": [
    {"Product": "Widget A", "Customer Name": "Acme Corp", "Total Sales": 1500.00, "Purchase Date": "2024-01-15"},
    {"Product": "Widget B", "Customer Name": "Acme Corp", "Total Sales": 2000.00, "Purchase Date": "2024-01-16"}
  ]
}
```

---

## Error Handling Flow

```
SQL Execution Error
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Common Errors Handled by LLM Correction:                       │
│                                                                │
│ 1. "column X does not exist"                                   │
│    → LLM checks schema for correct field name                  │
│                                                                │
│ 2. "column must appear in GROUP BY clause"                     │
│    → LLM adds missing column to GROUP BY                       │
│                                                                │
│ 3. "invalid input syntax for type numeric"                     │
│    → LLM adds NULLIF and type checking                         │
│                                                                │
│ 4. Wrong JSONB access (item vs header_data)                    │
│    → LLM checks 'source' in field mappings                     │
│                                                                │
│ 5. 'item' referenced in CTE WHERE clause                       │
│    → LLM moves filter to outer SELECT WHERE                    │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
Corrected SQL → Retry Execution
```

---

## Key Design Decisions

1. **Three-Round LLM Analysis**: Separates concerns for better accuracy
   - Round 1: Focus on understanding intent
   - Round 2: Focus on field mapping
   - Round 3: Focus on SQL syntax

2. **Self-Correcting SQL**: Up to 3 retry attempts with LLM-driven error analysis

3. **Fallback Heuristics**: Works without LLM using pattern matching (`_generate_sql_heuristic`)

4. **Dynamic Schema Inference**: No pre-defined schema required; infers from actual data

5. **Access Control**: All queries filtered by `accessible_doc_ids` for security

6. **CTE-Based SQL**: Uses Common Table Expressions for readability and JSONB expansion

---

## File References

| File | Key Lines | Description |
|------|-----------|-------------|
| [sql_query_executor.py](../backend/analytics_service/sql_query_executor.py) | 1035-1190 | Main execution loop |
| [sql_query_executor.py](../backend/analytics_service/sql_query_executor.py) | 1387-1504 | Field mapping retrieval |
| [llm_sql_generator.py](../backend/analytics_service/llm_sql_generator.py) | 56-97 | SQL generation prompt |
| [llm_sql_generator.py](../backend/analytics_service/llm_sql_generator.py) | 148-240 | Query analysis prompt |
| [llm_sql_generator.py](../backend/analytics_service/llm_sql_generator.py) | 243-285 | Field mapping prompt |
| [llm_sql_generator.py](../backend/analytics_service/llm_sql_generator.py) | 288-342 | Error correction prompt |
| [llm_sql_generator.py](../backend/analytics_service/llm_sql_generator.py) | 345-800 | Verified SQL prompt + templates |
| [llm_sql_generator.py](../backend/analytics_service/llm_sql_generator.py) | 1073-1222 | Multi-round generation |
| [llm_sql_generator.py](../backend/analytics_service/llm_sql_generator.py) | 1224-1300 | Error correction logic |
