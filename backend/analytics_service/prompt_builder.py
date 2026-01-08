"""
Dynamic Prompt Builder for LLM-based SQL Generation

This module provides template-based prompt composition that:
1. Reduces token usage by including only relevant prompt sections
2. Adapts prompts based on query type and schema characteristics
3. Eliminates duplicate instructions across prompts
4. Integrates with DataSchema for schema-aware prompts

Design Principles:
- Modular prompt sections that can be composed dynamically
- Schema-driven field documentation
- Query-type-aware rule inclusion
- Minimal hardcoded logic - let LLM handle edge cases
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of analytics queries."""
    DETAIL = "detail"           # List individual records
    SUMMARY = "summary"         # Aggregated totals
    COUNT = "count"             # Count queries
    MIN_MAX = "min_max"         # Find min/max records
    COMPARISON = "comparison"   # Compare groups
    TREND = "trend"             # Time-based trends


class AggregationType(Enum):
    """Aggregation operations."""
    SUM = "sum"
    AVG = "avg"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    MIN_MAX = "min_max"


@dataclass
class QueryContext:
    """Context for building SQL generation prompts."""
    user_query: str
    schema_type: Optional[str] = None
    query_type: QueryType = QueryType.SUMMARY
    aggregation_type: Optional[AggregationType] = None
    has_time_filter: bool = False
    has_time_grouping: bool = False
    has_entity_grouping: bool = False
    count_level: str = "line_item"  # "document" or "line_item"
    field_mappings: Dict[str, Any] = field(default_factory=dict)


class PromptBuilder:
    """
    Builds dynamic prompts for LLM-based SQL generation.

    Instead of massive static prompts with all possible rules,
    this builder composes prompts based on:
    - Query type (detail, summary, count, min/max)
    - Schema characteristics
    - Detected query patterns

    This reduces token usage by 40-60% while improving accuracy.
    """

    # ==========================================================================
    # BASE PROMPT SECTIONS (Minimal, always included)
    # ==========================================================================

    BASE_SQL_CONTEXT = """You are a PostgreSQL SQL expert. Generate a query to answer the user's question.

## Database Schema (USE EXACT COLUMN NAMES)

### Table: documents_data (alias: dd)
| Column        | Type    | Description                        |
|---------------|---------|-------------------------------------|
| id            | UUID    | Primary key                         |
| document_id   | UUID    | FK to documents table               |
| schema_type   | VARCHAR | Document type (invoice, receipt)    |
| header_data   | JSONB   | Document-level fields               |
| summary_data  | JSONB   | Aggregated totals                   |

### Table: documents_data_line_items (alias: li)
| Column            | Type    | Description                           |
|-------------------|---------|---------------------------------------|
| id                | UUID    | Primary key                           |
| documents_data_id | UUID    | FK to documents_data.id               |
| line_number       | INTEGER | Position in original document         |
| data              | JSONB   | **THE LINE ITEM DATA** (one per row)  |

**CRITICAL**: The line item JSONB column is named `data`, NOT `line_items`!
Each row in documents_data_line_items represents ONE line item (not a JSONB array).
DO NOT use jsonb_array_elements() - the data is already normalized into rows.

## Field Access Patterns
- Header fields: `dd.header_data->>'field_name'` or `header_data->>'field_name'`
- Summary fields: `dd.summary_data->>'field_name'` or `summary_data->>'field_name'`
- Line item fields: `li.data->>'field_name'` (use alias `item` in CTE: `item->>'field_name'`)

## Required SQL Template
```sql
WITH items AS (
    SELECT
        dd.document_id,
        dd.header_data,
        dd.summary_data,
        li.data as item  -- Rename li.data to item for clarity
    FROM documents_data dd
    JOIN documents_data_line_items li ON li.documents_data_id = dd.id
    WHERE dd.document_id IN (...)  -- Filter added automatically
)
SELECT ... FROM items WHERE ...
```
"""

    # ==========================================================================
    # SCHEMA SECTION (Dynamic based on field_mappings)
    # ==========================================================================

    SCHEMA_SECTION_TEMPLATE = """
## Available Fields
{field_list}

**CRITICAL**: Use the exact `access_pattern` shown above in your SQL.
"""

    # ==========================================================================
    # QUERY TYPE SPECIFIC RULES (Only included when relevant)
    # ==========================================================================

    DETAIL_RULES = """
## Detail Report Rules
- SELECT all available fields from the schema
- Include document identifiers (receipt_number, invoice_number) for grouping
- Order fields logically: header fields first, then line item fields
- No aggregation - return individual records
"""

    SUMMARY_RULES = """
## Summary/Aggregation Rules
- GROUP BY the requested dimensions
- Use appropriate aggregate functions (SUM, AVG, COUNT)
- CRITICAL: Cast JSONB text to numeric for aggregations: `SUM((item->>'field')::numeric)`
- Include COUNT(*) as record_count
- Round monetary values to 2 decimal places
"""

    COUNT_DOCUMENT_RULES = """
## Document Count Rules
- You are counting DOCUMENTS (receipts/invoices), NOT line items
- Include `dd.document_id` in the CTE SELECT
- Use `COUNT(DISTINCT document_id)` in the outer SELECT
- Do NOT use `COUNT(*)` - that counts line items
"""

    COUNT_LINEITEM_RULES = """
## Line Item Count Rules
- You are counting individual line items/products
- Use `COUNT(*)` to count all matching line items
"""

    MIN_MAX_RULES = """
## MIN/MAX Query Rules
- Use subquery pattern to find record with min/max value:
  ```sql
  WHERE (field)::numeric = (SELECT MIN/MAX((field)::numeric) FROM items WHERE ...)
  ```
- Include ALL schema fields in SELECT to show complete record details
- Do NOT use `ORDER BY ... LIMIT 1` - it fails with NULL values
"""

    TIME_FILTER_RULES = """
## Time Filter Rules (Filtering, NOT grouping)
- Filter data to a specific time period without grouping by time
- Use safe year extraction: `SUBSTRING(date_field FROM '[0-9]{4}') = 'YYYY'`
- Never directly cast to ::date - date formats may be inconsistent
"""

    TIME_GROUPING_RULES = """
## Time Grouping Rules
- Group results by time period (yearly, monthly, quarterly)
- For yearly: `EXTRACT(YEAR FROM (date_field)::timestamp)::int as year`
- For monthly: `TO_CHAR((date_field)::timestamp, 'YYYY-MM') as month`
- For quarterly: `CONCAT(EXTRACT(YEAR FROM ...), '-Q', EXTRACT(QUARTER FROM ...)) as quarter`
"""

    # ==========================================================================
    # COMMON RULES (Included in most queries)
    # ==========================================================================

    TYPE_CASTING_RULES = """
## Type Casting (CRITICAL)
- The `->>` operator returns TEXT, so aggregations REQUIRE casting
- CORRECT: `SUM((item->>'amount')::numeric)`
- WRONG: `SUM(item->>'amount')` - ERROR: function sum(text) does not exist
- For SELECT (no aggregation): No cast needed, return as-is
- For ORDER BY numeric: Cast in ORDER BY clause only
"""

    TEXT_SEARCH_RULES = """
## Text Search (Case-Insensitive)
- Always use ILIKE for text matching (NOT LIKE)
- Example: `WHERE (item->>'description') ILIKE '%keyword%'`
"""

    FILTER_RULES = """
## Filter Operators
- Comparison: `>`, `<`, `>=`, `<=`, `=`, `!=`
- Range (inclusive): `BETWEEN x AND y`
- Compound (exclusive): `field > x AND field < y`
- List: `field IN ('a', 'b', 'c')`
- Text match: `field ILIKE '%pattern%'`
"""

    # ==========================================================================
    # OUTPUT FORMAT SECTION
    # ==========================================================================

    OUTPUT_FORMAT = """
## Response Format (JSON only)
```json
{
    "sql": "YOUR SQL QUERY",
    "explanation": "Brief explanation of what the query does",
    "columns": ["col1", "col2"],
    "needs_clarification": false
}
```
"""

    # ==========================================================================
    # SINGLE-SHOT UNIFIED PROMPT (Consolidated approach)
    # ==========================================================================

    UNIFIED_SQL_PROMPT = """You are a PostgreSQL SQL expert. Analyze the user's question and generate an appropriate SQL query.

## User Question
"{user_query}"

## Database Tables (CRITICAL - use exact column names)

**documents_data** (alias: dd):
- id (UUID): Primary key
- document_id (UUID): FK to documents table
- header_data (JSONB): Document-level fields
- summary_data (JSONB): Aggregated totals

**documents_data_line_items** (alias: li):
- id (UUID): Primary key
- documents_data_id (UUID): FK to dd.id
- line_number (INTEGER): Position in document
- data (JSONB): **THE LINE ITEM DATA** - NOT "line_items"!

**CRITICAL RULES**:
- Use `li.data` to access line item JSONB, NOT `li.line_items` (that column does not exist!)
- Each row in documents_data_line_items is ONE line item - do NOT use jsonb_array_elements()
- Alias `li.data as item` in the CTE, then use `item->>'field_name'`

## Available Fields
{field_schema}

## SQL Template (ALWAYS use this structure)
```sql
WITH items AS (
    SELECT dd.document_id, dd.header_data, dd.summary_data, li.data as item
    FROM documents_data dd
    JOIN documents_data_line_items li ON li.documents_data_id = dd.id
    -- Document filter will be added automatically
)
SELECT ... FROM items WHERE ...
```

## Key Rules
1. **Line Item Access**: Use `item->>'field'` after aliasing `li.data as item` (NEVER `li.line_items`!)
2. **Aggregations**: Cast to numeric - `SUM((item->>'field')::numeric)`
3. **Text Search**: Use ILIKE for case-insensitive matching
4. **Date Filtering**: Use `SUBSTRING(date FROM '[0-9]{{4}}') = 'YYYY'` for safe year extraction
5. **Detail Reports**: Include all fields, order header fields before line items
6. **Count Queries**: Use `COUNT(DISTINCT document_id)` for documents, `COUNT(*)` for items

{additional_rules}

## Response (JSON only)
{{
    "sql": "THE SQL QUERY",
    "explanation": "What the query does",
    "query_type": "detail|summary|count|min_max",
    "aggregation_type": "sum|avg|count|min|max|null",
    "grouping_fields": ["field1", "field2"],
    "time_granularity": "yearly|monthly|quarterly|null"
}}
"""

    # ==========================================================================
    # ERROR CORRECTION PROMPT
    # ==========================================================================

    ERROR_CORRECTION_PROMPT = """A PostgreSQL query failed. Analyze and fix it.

## Original Question
"{user_query}"

## Failed SQL
```sql
{failed_sql}
```

## Error Message
{error_message}

## CRITICAL: Correct Table Structure
The `documents_data_line_items` table has EXACTLY these columns:
- `id` (UUID) - Primary key
- `documents_data_id` (UUID) - FK to documents_data.id
- `line_number` (INTEGER) - Position in document
- `data` (JSONB) - **THIS IS THE LINE ITEM DATA**

**COMMON MISTAKE**: Using `li.line_items` - THIS COLUMN DOES NOT EXIST!
**CORRECT**: Use `li.data` (aliased as `item` in the CTE)

Each row in documents_data_line_items is ONE line item.
Do NOT use jsonb_array_elements() - the data is already normalized into rows.

## Available Fields
{field_schema}

## Common Fixes
1. **"column line_items does not exist"** → Use `li.data` not `li.line_items`
2. **"column item does not exist"** → Ensure CTE includes `li.data as item`
3. **"function sum(text)"** → Add ::numeric cast: `SUM((item->>'x')::numeric)`
4. **"invalid input syntax for numeric"** → Field is string, remove cast or handle NULL
5. **"date/time field value out of range"** → Use SUBSTRING for year extraction
6. **Case-sensitive search** → Change LIKE to ILIKE

## Response (JSON only)
{{
    "analysis": "What caused the error",
    "fix_applied": "Description of fix",
    "corrected_sql": "THE FIXED SQL QUERY",
    "confidence": 0.0-1.0
}}
"""

    # ==========================================================================
    # QUERY ANALYSIS PROMPT (Simplified single-round)
    # ==========================================================================

    QUERY_ANALYSIS_PROMPT = """Analyze this analytics question and extract query parameters.

## User Question
"{user_query}"

## Available Fields
{field_schema}

## Analysis Tasks
1. **Query Type**: What kind of result does the user want?
   - detail: List individual records with all fields
   - summary: Aggregated totals (SUM, AVG)
   - count: How many documents/items
   - min_max: Find records with minimum/maximum values

2. **Aggregation**: What operation on which field?
   - sum, avg, count, min, max, or null (for detail)

3. **Grouping**: What dimensions to group by?
   - Entity fields (customer, vendor)
   - Time periods (year, month, quarter)
   - Categories

4. **Time Handling**:
   - time_filter_only=true: Filter TO a period (e.g., "in 2025")
   - time_filter_only=false: Group BY period (e.g., "by month")

5. **Filters**: Any conditions to apply?

6. **Count Level** (for count queries):
   - document: Count receipts/invoices/documents
   - line_item: Count individual items/products

## Response (JSON only)
{{
    "query_type": "detail|summary|count|min_max",
    "aggregation_type": "sum|avg|count|min|max|null",
    "aggregation_field": "field_name or null",
    "grouping_fields": ["field1", "field2"],
    "time_granularity": "yearly|monthly|quarterly|null",
    "time_filter_only": true|false,
    "time_filter_year": 2025|null,
    "count_level": "document|line_item",
    "filters": {{}},
    "explanation": "What the user wants"
}}
"""

    def __init__(self, schema_service=None):
        """
        Initialize the prompt builder.

        Args:
            schema_service: Optional SchemaService for field mapping lookup
        """
        self.schema_service = schema_service

    def build_unified_sql_prompt(
        self,
        user_query: str,
        field_mappings: Dict[str, Any],
        context: Optional[QueryContext] = None
    ) -> str:
        """
        Build a unified SQL generation prompt.

        This is the primary method for generating SQL prompts.
        It includes only the rules relevant to the detected query type.

        Args:
            user_query: User's natural language query
            field_mappings: Field schema with access patterns
            context: Optional pre-analyzed query context

        Returns:
            Composed prompt string
        """
        # Format field schema for prompt
        field_schema = self._format_field_schema(field_mappings)

        # Build additional rules based on context
        additional_rules = self._build_additional_rules(context, user_query)

        prompt = self.UNIFIED_SQL_PROMPT.format(
            user_query=user_query,
            field_schema=field_schema,
            additional_rules=additional_rules
        )

        logger.debug(f"[PromptBuilder] Built unified prompt ({len(prompt)} chars)")
        return prompt

    def build_query_analysis_prompt(
        self,
        user_query: str,
        field_mappings: Dict[str, Any]
    ) -> str:
        """
        Build a query analysis prompt for understanding user intent.

        Args:
            user_query: User's natural language query
            field_mappings: Field schema

        Returns:
            Query analysis prompt
        """
        field_schema = self._format_field_schema(field_mappings)

        return self.QUERY_ANALYSIS_PROMPT.format(
            user_query=user_query,
            field_schema=field_schema
        )

    def build_error_correction_prompt(
        self,
        user_query: str,
        failed_sql: str,
        error_message: str,
        field_mappings: Dict[str, Any]
    ) -> str:
        """
        Build an error correction prompt for fixing failed SQL.

        Args:
            user_query: Original user query
            failed_sql: The SQL that failed
            error_message: Error from PostgreSQL
            field_mappings: Field schema

        Returns:
            Error correction prompt
        """
        field_schema = self._format_field_schema(field_mappings)

        return self.ERROR_CORRECTION_PROMPT.format(
            user_query=user_query,
            failed_sql=failed_sql,
            error_message=error_message,
            field_schema=field_schema
        )

    def build_sql_generation_prompt(
        self,
        user_query: str,
        field_mappings: Dict[str, Any],
        analysis_result: Dict[str, Any]
    ) -> str:
        """
        Build SQL generation prompt with analyzed query parameters.

        This is used in the two-round approach where query analysis
        has already been performed.

        Args:
            user_query: User's query
            field_mappings: Field schema
            analysis_result: Result from query analysis

        Returns:
            SQL generation prompt
        """
        field_schema = self._format_field_schema(field_mappings)

        # Build context from analysis
        context = QueryContext(
            user_query=user_query,
            query_type=self._parse_query_type(analysis_result.get('query_type', 'summary')),
            aggregation_type=self._parse_aggregation_type(analysis_result.get('aggregation_type')),
            has_time_filter=analysis_result.get('time_filter_only', False),
            has_time_grouping=analysis_result.get('time_granularity') is not None and not analysis_result.get('time_filter_only', False),
            count_level=analysis_result.get('count_level', 'line_item'),
            field_mappings=field_mappings
        )

        return self.build_unified_sql_prompt(user_query, field_mappings, context)

    def _format_field_schema(self, field_mappings: Dict[str, Any]) -> str:
        """
        Format field mappings as a concise schema for prompts.

        Supports both grouped format and flat format.
        """
        # Check for grouped format
        if any(k in field_mappings for k in ['header_mappings', 'line_item_mappings', 'summary_mappings']):
            return self._format_grouped_schema(field_mappings)
        else:
            return self._format_flat_schema(field_mappings)

    def _format_grouped_schema(self, grouped_mappings: Dict[str, List[Dict]]) -> str:
        """Format grouped field mappings (new format).

        Handles various input formats defensively:
        - Dict with 'canonical' key (expected)
        - List items [field_name, field_info]
        - String items (just field name)
        """
        lines = []

        def _get_canonical(m) -> tuple:
            """Extract canonical name and data_type from various formats."""
            if isinstance(m, dict):
                canonical = m.get('canonical', m.get('field_name', m.get('name', '')))
                data_type = m.get('data_type', 'string')
                return canonical, data_type
            elif isinstance(m, list) and len(m) >= 1:
                canonical = m[0] if isinstance(m[0], str) else ''
                data_type = m[1].get('data_type', 'string') if len(m) > 1 and isinstance(m[1], dict) else 'string'
                return canonical, data_type
            elif isinstance(m, str):
                return m, 'string'
            return '', 'string'

        # Header fields
        header_list = grouped_mappings.get('header_mappings', [])
        if header_list and isinstance(header_list, list):
            lines.append("### Header Fields (from header_data)")
            for m in header_list:
                canonical, data_type = _get_canonical(m)
                if canonical:
                    lines.append(f"- `{canonical}` ({data_type}) → `header_data->>'{canonical}'`")

        # Summary fields
        summary_list = grouped_mappings.get('summary_mappings', [])
        if summary_list and isinstance(summary_list, list):
            lines.append("\n### Summary Fields (from summary_data)")
            for m in summary_list:
                canonical, data_type = _get_canonical(m)
                if not data_type or data_type == 'string':
                    data_type = 'number'  # Default for summary fields
                if canonical:
                    lines.append(f"- `{canonical}` ({data_type}) → `summary_data->>'{canonical}'`")

        # Line item fields
        line_item_list = grouped_mappings.get('line_item_mappings', [])
        if line_item_list and isinstance(line_item_list, list):
            lines.append("\n### Line Item Fields (from item)")
            for m in line_item_list:
                canonical, data_type = _get_canonical(m)
                if canonical:
                    lines.append(f"- `{canonical}` ({data_type}) → `item->>'{canonical}'`")

        return "\n".join(lines)

    def _format_flat_schema(self, field_mappings: Dict[str, Dict[str, Any]]) -> str:
        """Format flat field mappings (legacy format)."""
        # Group by source
        header_fields = []
        summary_fields = []
        line_item_fields = []

        for field_name, mapping in field_mappings.items():
            if field_name in ['header_mappings', 'line_item_mappings', 'summary_mappings']:
                continue

            source = mapping.get('source', 'line_item')
            data_type = mapping.get('data_type', 'string')
            db_field = mapping.get('db_field', field_name)

            if source == 'header':
                header_fields.append(f"- `{db_field}` ({data_type}) → `header_data->>'{db_field}'`")
            elif source == 'summary':
                summary_fields.append(f"- `{db_field}` ({data_type}) → `summary_data->>'{db_field}'`")
            else:
                line_item_fields.append(f"- `{db_field}` ({data_type}) → `item->>'{db_field}'`")

        lines = []
        if header_fields:
            lines.append("### Header Fields")
            lines.extend(header_fields)
        if summary_fields:
            lines.append("\n### Summary Fields")
            lines.extend(summary_fields)
        if line_item_fields:
            lines.append("\n### Line Item Fields")
            lines.extend(line_item_fields)

        return "\n".join(lines) if lines else "No fields available."

    def _build_additional_rules(
        self,
        context: Optional[QueryContext],
        user_query: str
    ) -> str:
        """Build additional rules based on query context."""
        rules = []

        if not context:
            # Detect from query keywords
            query_lower = user_query.lower()

            if any(kw in query_lower for kw in ['list', 'show all', 'details', 'each']):
                rules.append(self.DETAIL_RULES)

            if any(kw in query_lower for kw in ['count', 'how many']):
                if any(kw in query_lower for kw in ['receipt', 'invoice', 'document', 'meal']):
                    rules.append(self.COUNT_DOCUMENT_RULES)
                else:
                    rules.append(self.COUNT_LINEITEM_RULES)

            if any(kw in query_lower for kw in ['min', 'max', 'lowest', 'highest', 'smallest', 'largest']):
                rules.append(self.MIN_MAX_RULES)

            if any(kw in query_lower for kw in ['by year', 'by month', 'monthly', 'yearly', 'quarterly']):
                rules.append(self.TIME_GROUPING_RULES)
            elif any(kw in query_lower for kw in ['in 2', 'for 2', 'during 2']):  # "in 2025", etc.
                rules.append(self.TIME_FILTER_RULES)
        else:
            # Use analyzed context
            if context.query_type == QueryType.DETAIL:
                rules.append(self.DETAIL_RULES)
            elif context.query_type == QueryType.COUNT:
                if context.count_level == "document":
                    rules.append(self.COUNT_DOCUMENT_RULES)
                else:
                    rules.append(self.COUNT_LINEITEM_RULES)
            elif context.query_type == QueryType.MIN_MAX:
                rules.append(self.MIN_MAX_RULES)

            if context.has_time_grouping:
                rules.append(self.TIME_GROUPING_RULES)
            elif context.has_time_filter:
                rules.append(self.TIME_FILTER_RULES)

        return "\n".join(rules) if rules else ""

    def _parse_query_type(self, query_type_str: str) -> QueryType:
        """Parse query type string to enum."""
        mapping = {
            'detail': QueryType.DETAIL,
            'summary': QueryType.SUMMARY,
            'count': QueryType.COUNT,
            'min_max': QueryType.MIN_MAX,
            'comparison': QueryType.COMPARISON,
            'trend': QueryType.TREND
        }
        return mapping.get(query_type_str, QueryType.SUMMARY)

    def _parse_aggregation_type(self, agg_type_str: Optional[str]) -> Optional[AggregationType]:
        """Parse aggregation type string to enum."""
        if not agg_type_str:
            return None
        mapping = {
            'sum': AggregationType.SUM,
            'avg': AggregationType.AVG,
            'count': AggregationType.COUNT,
            'min': AggregationType.MIN,
            'max': AggregationType.MAX,
            'min_max': AggregationType.MIN_MAX
        }
        return mapping.get(agg_type_str)


# =============================================================================
# RESULT FORMATTING PROMPTS
# =============================================================================

class ResultFormatterPrompts:
    """Prompts for LLM-driven result formatting."""

    SUMMARY_HEADER_PROMPT = """Generate a brief summary header for this query result.

## User Question
"{user_query}"

## Result Overview
- Records: {row_count}
- Columns: {columns}

## Statistics
{statistics}

## Sample Data
{sample_rows}

## Instructions
Write a brief (under 200 words) summary that:
1. States what the data shows
2. Highlights key totals/statistics
3. Notes the full data table follows below

Do NOT list individual records. Format as markdown.
"""

    COLUMN_FORMAT_PROMPT = """Determine the best display format for these columns.

## Columns
{columns}

## Sample Values
{sample_values}

## Response (JSON)
{{
    "column_name": {{
        "format": "currency|number|date|text|percentage",
        "alignment": "left|right|center",
        "display_name": "Human-Readable Name"
    }}
}}
"""
