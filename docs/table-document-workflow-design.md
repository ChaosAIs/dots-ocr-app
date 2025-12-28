# Table Document Workflow Design

## Overview

This document describes the end-to-end workflow for handling table-associated documents (Excel, CSV, spreadsheets, and other tabular data) in the DOTS OCR system. The design emphasizes LLM-driven analysis at each stage while maintaining efficiency through intelligent caching.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Document Type Detection](#2-document-type-detection)
3. [Data Extraction](#3-data-extraction)
4. [Field Mapping](#4-field-mapping)
5. [SQL Query Generation](#5-sql-query-generation)
6. [Report Layout Generation](#6-report-layout-generation)
7. [Data Flow Diagrams](#7-data-flow-diagrams)
8. [Database Schema](#8-database-schema)
9. [LLM Prompts Reference](#9-llm-prompts-reference)
10. [Configuration](#10-configuration)

---

## 1. Architecture Overview

### 1.1 Core Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DOTS OCR Table Workflow                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   Document   │───▶│    Data      │───▶│     SQL      │───▶│   Report   │ │
│  │  Detection   │    │  Extraction  │    │  Generation  │    │  Formatter │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│         │                   │                   │                   │        │
│         ▼                   ▼                   ▼                   ▼        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Field Mapping Layer                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │
│  │  │ Formal      │  │ Cached      │  │ LLM-Inferred│                   │   │
│  │  │ Schema      │  │ Mappings    │  │ Mappings    │                   │   │
│  │  │ (Priority 1)│  │ (Priority 2)│  │ (Priority 3)│                   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Storage Layer                                 │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │   │
│  │  │  data_schemas   │  │ documents_data  │  │ documents_data_     │   │   │
│  │  │  (formal)       │  │ (inline ≤500)   │  │ line_items (>500)   │   │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Design Principles

| Principle | Description |
|-----------|-------------|
| **LLM-Driven** | Each stage uses LLM for intelligent analysis, no hardcoded business logic |
| **Priority-Based Lookup** | Formal schema → Cached mappings → LLM inference |
| **Efficient Caching** | Field mappings cached in `extraction_metadata` after first inference |
| **Adaptive Storage** | Inline for ≤500 rows, overflow table for >500 rows |
| **Schema Independence** | Multi-schema queries processed independently without merging |

---

## 2. Document Type Detection

### 2.1 Detection Flow

```
┌─────────────────┐
│  Upload File    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  File Extension Analysis                 │
│  .xlsx, .xls → spreadsheet              │
│  .csv → spreadsheet                      │
│  .pdf → requires content analysis        │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  OCR Processing (Marker/MinerU)          │
│  Convert to Markdown with table tags     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Content Structure Analysis              │
│  Detect: |---|---| table patterns        │
│  Count columns, rows, headers            │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Schema Type Assignment                  │
│  • spreadsheet (generic tabular)         │
│  • invoice (if invoice patterns found)   │
│  • receipt, bank_statement, etc.         │
└─────────────────────────────────────────┘
```

### 2.2 Detection Criteria

| File Type | Extension | Schema Type | Detection Method |
|-----------|-----------|-------------|------------------|
| Excel | .xlsx, .xls | `spreadsheet` | File extension |
| CSV | .csv | `spreadsheet` | File extension |
| PDF with tables | .pdf | `spreadsheet` or specific | Content pattern analysis |
| Invoice PDF | .pdf | `invoice` | LLM content classification |

### 2.3 Content Pattern Detection

```python
# Markdown table detection regex
TABLE_PATTERN = r'\|[^\|]+\|'
SEPARATOR_PATTERN = r'\|[-:]+\|'

def detect_table_structure(content: str) -> Dict:
    """Detect markdown table structure in content."""
    lines = content.split('\n')

    # Find header row (first row with | separators)
    # Find separator row (|---|---|)
    # Count data rows

    return {
        'has_table': True/False,
        'column_count': int,
        'row_count': int,
        'column_headers': List[str]
    }
```

---

## 3. Data Extraction

### 3.1 Extraction Strategy Selection

```
┌─────────────────────────────────────────────────────────────────┐
│                    Strategy Selection Matrix                     │
├──────────────────┬──────────────┬───────────────────────────────┤
│  Document Size   │  Row Count   │  Strategy                     │
├──────────────────┼──────────────┼───────────────────────────────┤
│  Small (<10KB)   │  <50 rows    │  LLM_DIRECT                   │
│  Medium (10-100KB)│  50-500 rows │  LLM_CHUNKED or HYBRID        │
│  Large (>100KB)  │  >500 rows   │  HYBRID or PARSED             │
└──────────────────┴──────────────┴───────────────────────────────┘
```

### 3.2 Extraction Strategies

#### 3.2.1 LLM_DIRECT (Small Documents)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Content    │────▶│  LLM Call   │────▶│  Structured │
│  (full doc) │     │  (single)   │     │  JSON       │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Process:**
1. Send entire document content to LLM
2. LLM extracts header_data, line_items, summary_data
3. Parse JSON response

#### 3.2.2 HYBRID (Medium/Large Spreadsheets) - Recommended

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Content    │────▶│  Markdown   │────▶│  Structured │
│  (markdown) │     │  Parser     │     │  Data       │
└─────────────┘     └──────┬──────┘     └─────────────┘
                          │
                    (fallback)
                          │
                          ▼
                   ┌─────────────┐
                   │  LLM Call   │
                   └─────────────┘
```

**Process:**
1. **Priority 1**: Direct markdown table parsing (no LLM cost)
2. **Priority 2**: LLM extraction if parsing fails
3. Regex-based table detection: `|---|---|`
4. Parse column headers from first row
5. Parse data rows into key-value pairs

#### 3.2.3 LLM_CHUNKED (Large Documents)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Chunk 1    │────▶│  LLM (hdrs) │────▶│  Headers    │
│  (headers)  │     │             │     │  + Schema   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
┌─────────────┐     ┌─────────────┐            │
│  Chunk 2-N  │────▶│  LLM (rows) │────────────┘
│  (data)     │     │             │     ┌─────────────┐
└─────────────┘     └─────────────┘────▶│  Merged     │
                                        │  Result     │
                                        └─────────────┘
```

### 3.3 Extraction Output Structure

```json
{
  "header_data": {
    "sheet_name": "Sales Report Q4",
    "column_headers": ["Date", "Product", "Region", "Sales Amount", "Quantity"],
    "total_rows": 150,
    "total_columns": 5
  },
  "line_items": [
    {
      "row_number": 1,
      "Date": "2024-01-15",
      "Product": "Widget A",
      "Region": "North",
      "Sales Amount": 1500.00,
      "Quantity": 10
    },
    {
      "row_number": 2,
      "Date": "2024-01-16",
      "Product": "Widget B",
      "Region": "South",
      "Sales Amount": 2300.00,
      "Quantity": 15
    }
  ],
  "summary_data": {
    "row_count": 150,
    "column_count": 5
  }
}
```

### 3.4 Storage Decision

```python
CHUNKED_LLM_MAX_ROWS = 500  # Threshold for overflow storage

def determine_storage(line_items_count: int) -> str:
    if line_items_count > CHUNKED_LLM_MAX_ROWS:
        return "external"  # Store in documents_data_line_items table
    else:
        return "inline"    # Store in documents_data.line_items JSONB
```

---

## 4. Field Mapping

### 4.1 Field Mapping Priority

```
┌─────────────────────────────────────────────────────────────────┐
│                    Field Mapping Lookup                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Priority 1: extraction_metadata.field_mappings                  │
│  ├── Per-document cache                                          │
│  ├── Fastest lookup (already in documents_data)                  │
│  └── Contains document-specific mappings                         │
│                                                                  │
│  Priority 2: data_schemas.field_mappings                         │
│  ├── Formal schema from registry                                 │
│  ├── Shared across documents of same type                        │
│  └── Manually curated for known document types                   │
│                                                                  │
│  Priority 3: LLM-Driven Inference                                │
│  ├── Analyze actual data structure                               │
│  ├── Semantic pattern matching                                   │
│  └── Cache result in extraction_metadata for future queries      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Semantic Field Classification

The system uses pattern-based semantic classification for unknown fields:

```python
SEMANTIC_PATTERNS = [
    # (semantic_type, keywords, data_type, aggregation)
    ('date',     ['date', 'time', 'created', 'updated', 'timestamp'], 'datetime', None),
    ('amount',   ['total', 'amount', 'price', 'cost', 'revenue', 'sales', 'subtotal', 'fee', 'tax'], 'number', 'sum'),
    ('quantity', ['quantity', 'qty', 'count', 'units', 'items', 'number of'], 'number', 'sum'),
    ('entity',   ['customer', 'vendor', 'supplier', 'client', 'company', 'store', 'merchant'], 'string', 'group_by'),
    ('category', ['category', 'type', 'class', 'group', 'segment'], 'string', 'group_by'),
    ('product',  ['product', 'item', 'description', 'name', 'sku'], 'string', 'group_by'),
    ('region',   ['region', 'city', 'country', 'location', 'address', 'area'], 'string', 'group_by'),
    ('person',   ['sales rep', 'representative', 'agent', 'manager', 'assignee'], 'string', 'group_by'),
    ('method',   ['method', 'payment', 'shipping', 'channel'], 'string', 'group_by'),
    ('identifier', ['id', 'number', 'code', 'reference', 'invoice', 'receipt', 'order'], 'string', None),
]
```

### 4.3 Field Mapping Structure

```json
{
  "header_fields": {
    "sheet_name": {
      "semantic_type": "identifier",
      "data_type": "string",
      "source": "header",
      "aggregation": null,
      "original_name": "sheet_name"
    },
    "total_rows": {
      "semantic_type": "quantity",
      "data_type": "number",
      "source": "header",
      "aggregation": null
    }
  },
  "line_item_fields": {
    "Date": {
      "semantic_type": "date",
      "data_type": "datetime",
      "source": "line_item",
      "aggregation": null,
      "original_name": "Date"
    },
    "Sales Amount": {
      "semantic_type": "amount",
      "data_type": "number",
      "source": "line_item",
      "aggregation": "sum",
      "original_name": "Sales Amount"
    },
    "Region": {
      "semantic_type": "region",
      "data_type": "string",
      "source": "line_item",
      "aggregation": "group_by",
      "original_name": "Region"
    }
  }
}
```

### 4.4 Aggregation Rules

| Semantic Type | Default Aggregation | SQL Function |
|--------------|---------------------|--------------|
| `amount` | `sum` | `SUM(field)` |
| `quantity` | `sum` | `SUM(field)` |
| `entity` | `group_by` | `GROUP BY field` |
| `category` | `group_by` | `GROUP BY field` |
| `product` | `group_by` | `GROUP BY field` |
| `region` | `group_by` | `GROUP BY field` |
| `person` | `group_by` | `GROUP BY field` |
| `method` | `group_by` | `GROUP BY field` |
| `date` | `null` | Filter/Range |
| `identifier` | `null` | Filter |

---

## 5. SQL Query Generation

### 5.1 SQL Generation Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  User Query     │────▶│  Field Mapping  │────▶│  LLM SQL Gen    │
│  (NL)           │     │  Lookup         │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │  Generated SQL  │
                                               │  + Explanation  │
                                               └─────────────────┘
```

### 5.2 LLM Prompt for SQL Generation

The LLM receives:

1. **User Query**: Natural language question
2. **Field Mappings**: Available fields with semantic types
3. **Table Filter**: Document ID constraints
4. **Database Schema**: JSONB structure information

```
User Query: "Show me total sales by region for Q4 2024"

Available Fields:
- Date: semantic_type=date, data_type=datetime, source=line_item
- Region: semantic_type=region, data_type=string, aggregation=group_by
- Sales Amount: semantic_type=amount, data_type=number, aggregation=sum
- Product: semantic_type=product, data_type=string, aggregation=group_by

Generate PostgreSQL query against documents_data table where:
- line_items is a JSONB array
- Access fields via: jsonb_array_elements(line_items)->>'field_name'
```

### 5.3 Generated SQL Structure

```sql
-- Example: Total sales by region
SELECT
    li->>'Region' as region,
    SUM((li->>'Sales Amount')::numeric) as total_sales,
    COUNT(*) as transaction_count
FROM documents_data dd,
     jsonb_array_elements(dd.line_items) as li
WHERE dd.document_id IN ('uuid1', 'uuid2', 'uuid3')
  AND (li->>'Date')::date >= '2024-10-01'
  AND (li->>'Date')::date <= '2024-12-31'
GROUP BY li->>'Region'
ORDER BY total_sales DESC;
```

### 5.4 Multi-Schema Query Handling

When documents have different schema types, the system processes them independently:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Schema Query Flow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Group Documents by Schema Type                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  spreadsheet: [doc1, doc2, doc3]                           │ │
│  │  invoice: [doc4, doc5]                                      │ │
│  │  receipt: [doc6]                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Step 2: Process Each Group Independently                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ spreadsheet  │  │   invoice    │  │   receipt    │          │
│  │ SQL + Result │  │ SQL + Result │  │ SQL + Result │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  Step 3: Generate Combined Summary (LLM)                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Unified summary with results separated by document type    │ │
│  │  No data merging - respects schema differences              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.5 SQL Generation Result

```python
@dataclass
class SQLGenerationResult:
    success: bool
    sql_query: Optional[str]
    explanation: Optional[str]
    select_fields: List[str]
    grouping_fields: List[str]
    aggregation_fields: List[str]
    time_granularity: Optional[str]
    error: Optional[str]
```

---

## 6. Report Layout Generation

### 6.1 Report Generation Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Query Results  │────▶│  LLM Layout     │────▶│  Formatted      │
│  (raw data)     │     │  Designer       │     │  Report         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  Column Config  │
                        │  + Formatting   │
                        └─────────────────┘
```

### 6.2 Layout Types

| Layout Type | Use Case | Characteristics |
|-------------|----------|-----------------|
| `table` | Detailed data view | Columns with headers, row data |
| `hierarchical` | Grouped data | Parent-child structure |
| `summary` | Aggregated results | Key metrics, totals |
| `detailed` | Full breakdown | All fields with explanations |

### 6.3 Report Layout Configuration

```python
@dataclass
class ReportColumn:
    field_name: str
    display_name: str
    semantic_type: str
    format_type: str  # 'currency', 'date', 'number', 'percentage', 'text'
    width: Optional[int]
    alignment: str  # 'left', 'right', 'center'

@dataclass
class ReportLayout:
    layout_type: str
    title: str
    columns: List[ReportColumn]
    grouping: Optional[List[str]]
    totals: Optional[List[str]]
    chart_type: Optional[str]
```

### 6.4 LLM Layout Design Prompt

```
Given query results with the following structure:
- Columns: region (string), total_sales (number), transaction_count (number)
- Row count: 5
- Data sample: [{"region": "North", "total_sales": 45000, "transaction_count": 150}, ...]

Design an optimal report layout:
1. Choose layout type (table, hierarchical, summary, detailed)
2. Configure columns with appropriate formatting
3. Suggest any charts or visualizations
4. Determine if totals row is needed

Return JSON:
{
  "layout_type": "table",
  "columns": [...],
  "show_totals": true,
  "chart_recommendation": "bar_chart"
}
```

### 6.5 Column Formatting Rules

```python
FORMAT_RULES = {
    'currency': {
        'prefix': '$',
        'decimals': 2,
        'thousands_separator': True,
        'alignment': 'right'
    },
    'date': {
        'format': 'YYYY-MM-DD',  # or locale-specific
        'alignment': 'center'
    },
    'number': {
        'decimals': 0,
        'thousands_separator': True,
        'alignment': 'right'
    },
    'percentage': {
        'suffix': '%',
        'decimals': 1,
        'alignment': 'right'
    },
    'text': {
        'alignment': 'left',
        'max_length': 50
    }
}
```

### 6.6 Final Report Output

```markdown
## Sales by Region - Q4 2024

| Region | Total Sales | Transactions |
|--------|-------------|--------------|
| East   | $52,300.00  | 410          |
| North  | $45,000.00  | 320          |
| South  | $38,500.00  | 275          |
| West   | $31,200.00  | 245          |
| **Total** | **$167,000.00** | **1,250** |

### Summary
The East region leads with $52,300 in sales (31% of total), followed by North
at $45,000. Total revenue across all regions: $167,000 from 1,250 transactions.
```

---

## 7. Data Flow Diagrams

### 7.1 Complete End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Complete Table Document Workflow                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  UPLOAD PHASE                                                                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Upload  │───▶│   OCR    │───▶│  Type    │───▶│  Schema  │              │
│  │  File    │    │  Process │    │  Detect  │    │  Lookup  │              │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘              │
│                                                        │                     │
│  EXTRACTION PHASE                                      ▼                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │ │
│  │  │  Markdown    │───▶│  Extract     │───▶│  Infer Field │             │ │
│  │  │  Parse       │    │  Data        │    │  Mappings    │             │ │
│  │  └──────────────┘    └──────────────┘    └──────┬───────┘             │ │
│  │                                                  │                      │ │
│  │  ┌──────────────┐    ┌──────────────┐           │                      │ │
│  │  │  Save to DB  │◀───│  Cache Field │◀──────────┘                      │ │
│  │  │  (inline/ext)│    │  Mappings    │                                  │ │
│  │  └──────────────┘    └──────────────┘                                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  QUERY PHASE                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │ │
│  │  │  User Query  │───▶│  Load Field  │───▶│  LLM SQL     │             │ │
│  │  │  (NL)        │    │  Mappings    │    │  Generation  │             │ │
│  │  └──────────────┘    └──────────────┘    └──────┬───────┘             │ │
│  │                                                  │                      │ │
│  │  ┌──────────────┐    ┌──────────────┐           │                      │ │
│  │  │  Execute SQL │◀───│  Validate    │◀──────────┘                      │ │
│  │  │              │    │  SQL         │                                  │ │
│  │  └──────┬───────┘    └──────────────┘                                  │ │
│  │         │                                                               │ │
│  │         ▼                                                               │ │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │ │
│  │  │  Raw Results │───▶│  LLM Layout  │───▶│  Formatted   │             │ │
│  │  │              │    │  Design      │    │  Report      │             │ │
│  │  └──────────────┘    └──────────────┘    └──────────────┘             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Field Mapping Lookup Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Field Mapping Lookup                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐                                                     │
│  │  Request Field     │                                                     │
│  │  Mappings          │                                                     │
│  └─────────┬──────────┘                                                     │
│            │                                                                 │
│            ▼                                                                 │
│  ┌────────────────────┐     ┌─────────┐                                    │
│  │ Check extraction_  │────▶│  Found  │───▶ Return cached mappings         │
│  │ metadata cache     │     └─────────┘                                    │
│  └─────────┬──────────┘                                                     │
│            │ Not Found                                                       │
│            ▼                                                                 │
│  ┌────────────────────┐     ┌─────────┐                                    │
│  │ Check data_schemas │────▶│  Found  │───▶ Return formal schema           │
│  │ table              │     └─────────┘                                    │
│  └─────────┬──────────┘                                                     │
│            │ Not Found                                                       │
│            ▼                                                                 │
│  ┌────────────────────┐     ┌─────────────────────┐                        │
│  │ Infer from data    │────▶│  Classify fields    │                        │
│  │ structure          │     │  using patterns     │                        │
│  └────────────────────┘     └──────────┬──────────┘                        │
│                                        │                                     │
│                                        ▼                                     │
│                              ┌─────────────────────┐                        │
│                              │  Cache in           │                        │
│                              │  extraction_metadata│                        │
│                              └─────────────────────┘                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Database Schema

### 8.1 Core Tables

```sql
-- Formal schema definitions
CREATE TABLE data_schemas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_type VARCHAR(64) UNIQUE NOT NULL,
    schema_version VARCHAR(16) DEFAULT '1.0',
    domain VARCHAR(32) NOT NULL,
    display_name VARCHAR(128),
    description TEXT,
    header_schema JSONB NOT NULL DEFAULT '{}',
    line_items_schema JSONB,
    summary_schema JSONB,
    field_mappings JSONB DEFAULT '{}',
    extraction_prompt TEXT,
    validation_rules JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Extracted document data
CREATE TABLE documents_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    schema_type VARCHAR(64),
    header_data JSONB DEFAULT '{}',
    line_items JSONB DEFAULT '[]',      -- Inline storage (≤500 rows)
    summary_data JSONB DEFAULT '{}',
    line_items_storage VARCHAR(20) DEFAULT 'inline',  -- 'inline' or 'external'
    line_items_count INTEGER DEFAULT 0,
    extraction_method VARCHAR(50),
    extraction_duration_ms INTEGER,
    extraction_metadata JSONB DEFAULT '{}',  -- Contains cached field_mappings
    validation_status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Overflow storage for large documents (>500 rows)
CREATE TABLE documents_data_line_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    documents_data_id UUID REFERENCES documents_data(id) ON DELETE CASCADE,
    line_number INTEGER NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(documents_data_id, line_number)
);

-- Indexes
CREATE INDEX idx_documents_data_document_id ON documents_data(document_id);
CREATE INDEX idx_documents_data_schema_type ON documents_data(schema_type);
CREATE INDEX idx_line_items_data_id ON documents_data_line_items(documents_data_id);
CREATE INDEX idx_line_items_line_number ON documents_data_line_items(documents_data_id, line_number);
```

### 8.2 Extraction Metadata Structure

```json
{
  "extraction_metadata": {
    "schema_source": "formal",        // or "dynamic"
    "schema_version": "1.0",          // or "inferred"
    "cached_at": "2024-12-27T10:30:00Z",
    "field_mappings": {
      "header_fields": { ... },
      "line_item_fields": { ... }
    },
    "extraction_strategy": "hybrid",
    "parsing_method": "markdown_table"
  }
}
```

---

## 9. LLM Prompts Reference

### 9.1 Extraction Prompt (Generic Spreadsheet)

```
Extract structured data from this spreadsheet.

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
- Detect and use the first row as headers if it contains column names.
```

### 9.2 SQL Generation Prompt

```
Generate a PostgreSQL query to answer this question:
"{user_query}"

AVAILABLE FIELDS AND THEIR SEMANTICS:
{field_mappings_formatted}

DATABASE SCHEMA:
- Table: documents_data (alias: dd)
- Line items stored in: dd.line_items (JSONB array)
- Access line item fields: jsonb_array_elements(dd.line_items)->>'field_name'
- Cast numbers: (li->>'amount')::numeric
- Cast dates: (li->>'date')::date

DOCUMENT FILTER:
{table_filter}

QUERY GUIDELINES:
1. Use appropriate aggregations based on semantic types
2. Apply GROUP BY for category/entity/region fields
3. Use SUM for amount/quantity fields
4. Filter by date ranges when time-related queries
5. Always include the document filter in WHERE clause

Return JSON:
{
    "sql": "THE COMPLETE SQL QUERY",
    "explanation": "What this query does",
    "select_fields": ["field1", "field2"],
    "group_by_fields": ["field1", "field2"]
}
```

### 9.3 Report Layout Prompt

```
Design a report layout for these query results:

Query: "{user_query}"
Result Columns: {columns}
Row Count: {row_count}
Sample Data: {sample_data}
Field Mappings: {field_mappings}

Determine:
1. Best layout type (table, summary, hierarchical, detailed)
2. Column configuration (display names, formatting, alignment)
3. Whether to show totals row
4. Any chart recommendations

Return JSON:
{
    "layout_type": "table",
    "title": "Report Title",
    "columns": [
        {
            "field_name": "region",
            "display_name": "Region",
            "format_type": "text",
            "alignment": "left"
        },
        {
            "field_name": "total_sales",
            "display_name": "Total Sales",
            "format_type": "currency",
            "alignment": "right"
        }
    ],
    "show_totals": true,
    "totals_fields": ["total_sales"],
    "chart_type": "bar_chart"
}
```

### 9.4 Natural Language Summary Prompt

```
Generate a natural language summary of these query results:

Query: "{user_query}"
Data: {formatted_data}

Write a clear, concise summary that:
1. Directly answers the user's question
2. Highlights key findings and trends
3. Mentions notable outliers or patterns
4. Provides relevant totals or averages

Keep the summary to 2-3 sentences maximum.
```

---

## 10. Configuration

### 10.1 Environment Variables

```bash
# Extraction thresholds
EXTRACTION_MIN_CONFIDENCE=0.85
CHUNKED_LLM_MAX_ROWS=500

# LLM settings
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4000

# Storage paths
DOCUMENT_OUTPUT_DIR=/data/documents
```

### 10.2 Extraction Strategy Thresholds

```python
# extraction_config.py
LLM_DIRECT_MAX_ROWS = 50      # Max rows for single LLM call
CHUNKED_LLM_MAX_ROWS = 500    # Threshold for overflow storage
HYBRID_MIN_ROWS = 50          # Min rows to consider hybrid approach
PARSED_MIN_ROWS = 1000        # Min rows for pure parsing approach
```

### 10.3 Supported Schema Types

| Schema Type | Domain | Description |
|-------------|--------|-------------|
| `spreadsheet` | generic | Dynamic tabular data |
| `invoice` | financial | Vendor invoices |
| `receipt` | financial | Transaction receipts |
| `bank_statement` | financial | Bank transactions |
| `expense_report` | financial | Expense claims |
| `purchase_order` | financial | Purchase orders |
| `shipping_manifest` | logistics | Shipping documents |
| `inventory_report` | inventory | Stock reports |

---

## Appendix A: Code References

| Component | File | Key Functions |
|-----------|------|---------------|
| Extraction Service | `backend/extraction_service/extraction_service.py` | `extract_document()`, `_parse_markdown_table()` |
| SQL Generator | `backend/analytics_service/llm_sql_generator.py` | `generate_sql()`, `generate_sql_with_schema()` |
| SQL Executor | `backend/analytics_service/sql_query_executor.py` | `execute_multi_schema_query()` |
| Response Formatter | `backend/analytics_service/response_formatter.py` | `format_response()`, `design_layout()` |
| Schema Service | `backend/analytics_service/schema_service.py` | `get_field_mappings()`, `infer_schema()` |

---

## Appendix B: Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Empty line_items | Markdown table not detected | Check for proper `|---|---|` separator |
| Wrong field types | Pattern matching failed | Add custom patterns or use LLM inference |
| SQL errors | JSONB path incorrect | Verify field names match extraction |
| Slow queries | Large inline storage | Enable external storage for >500 rows |

### Logging

```python
# Enable debug logging for table workflow
import logging
logging.getLogger('extraction_service').setLevel(logging.DEBUG)
logging.getLogger('llm_sql_generator').setLevel(logging.DEBUG)
logging.getLogger('sql_query_executor').setLevel(logging.DEBUG)
```
