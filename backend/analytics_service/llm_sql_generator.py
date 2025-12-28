"""
LLM-based SQL Generator for Dynamic Data Analysis

This module uses LLM to generate SQL queries based on:
1. User's natural language query
2. Dynamic data schema (field mappings from extracted data)
3. Available data in the database

Enhanced with multi-round LLM analysis:
- Round 1: Query intent and grouping analysis
- Round 2: Field mapping and identification
- Round 3: SQL generation with proper ordering

The generated SQL is executed against the JSONB line_items data.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysisResult:
    """Result of LLM query analysis."""
    grouping_fields: List[str] = field(default_factory=list)
    grouping_order: List[str] = field(default_factory=list)  # Ordered list of groupings
    aggregation_field: Optional[str] = None
    time_granularity: Optional[str] = None  # yearly, monthly, quarterly
    filters: Dict[str, Any] = field(default_factory=dict)
    report_type: str = "summary"  # summary, detail, comparison
    aggregation_type: str = "sum"  # sum, min, max, min_max, avg, count
    explanation: str = ""


@dataclass
class SQLGenerationResult:
    """Result of SQL generation."""
    sql_query: str
    explanation: str
    grouping_fields: List[str]
    aggregation_fields: List[str]
    time_granularity: Optional[str]  # yearly, quarterly, monthly, daily
    success: bool
    error: Optional[str] = None


class LLMSQLGenerator:
    """
    Generates SQL queries using LLM based on user query and data schema.
    """

    SQL_GENERATION_PROMPT = """You are a SQL expert. Generate a PostgreSQL query to answer the user's question.

## Data Structure
The data is stored in a PostgreSQL table with TWO separate JSONB columns:
- Table: documents_data (alias: dd)
- Column: header_data (JSONB object) - Contains document-level fields like transaction_date, store_name
- Column: line_items (JSONB array) - Contains item-level fields like description, amount, quantity

CRITICAL: Fields have different sources:
- source='header': Access via dd.header_data->>'field_name'
- source='line_item': Access via item->>'field_name' after jsonb_array_elements()

### Available Fields and Their Types:
{field_schema}

## User Query:
"{user_query}"

## Requirements:
1. Use jsonb_array_elements(dd.line_items) to expand line_items array
2. For header fields (source='header'): Use dd.header_data->>'field_name'
3. For line_item fields (source='line_item'): Use item->>'field_name'
4. Use proper type casting: ::numeric for numbers, ::timestamp for dates
5. For date grouping (usually from header_data):
   - Yearly: EXTRACT(YEAR FROM (dd.header_data->>'date_field')::timestamp)
   - Monthly: TO_CHAR((dd.header_data->>'date_field')::timestamp, 'YYYY-MM')
   - Quarterly: CONCAT(EXTRACT(YEAR FROM ...)::int, '-Q', EXTRACT(QUARTER FROM ...)::int)
6. Use ROUND() for monetary values to 2 decimal places
7. Always include COUNT(*) as item_count in aggregations
8. Order results logically (by date ascending, then by category)

## Response Format:
Return ONLY a JSON object with this exact structure:
{{
    "sql": "YOUR SQL QUERY HERE",
    "explanation": "Brief explanation of what the query does",
    "grouping_fields": ["field1", "field2"],
    "aggregation_fields": ["field_to_sum"],
    "time_granularity": "yearly" or "monthly" or "quarterly" or null
}}

Generate the SQL query now:"""

    SUMMARY_GENERATION_PROMPT = """Based on the query results, generate a summary report.

CRITICAL: You MUST use ONLY the actual data from the Query Results below. Do NOT invent, hallucinate, or make up any values or categories that are not present in the query results.

## User Query:
"{user_query}"

## Query Results (JSON) - USE ONLY THIS DATA:
{query_results}

## Available Field Mappings:
{field_schema}

## Requirements:
1. ONLY use column names and values that appear in the Query Results JSON above
2. Look at the actual column names in the query results (e.g., "Customer Name", "yearly", "total_amount")
3. If the query asks for MIN/MAX, identify the rows with minimum and maximum values
4. Do NOT invent categories like "Electronics" or "Furniture" unless they actually appear in the data
5. Format currency values with $ and proper formatting

## For MIN/MAX Queries:
If the user query asks for minimum/maximum values, identify:
- The row(s) with the minimum amount per grouping (e.g., per year)
- The row(s) with the maximum amount per grouping (e.g., per year)
- Include the entity name (customer, vendor, etc.) and the amount

## Response Format (JSON only):
{{
    "report_title": "Title based on actual query",
    "hierarchical_data": [
        {{
            "group_name": "actual_group_value_from_results",
            "group_total": actual_number,
            "sub_groups": [
                {{"name": "actual_name_from_results", "total": actual_number, "count": actual_count}}
            ],
            "min_record": {{"name": "entity_with_min", "amount": min_amount}},
            "max_record": {{"name": "entity_with_max", "amount": max_amount}}
        }}
    ],
    "summary_by_primary_grouping": {{"key1": value1, "key2": value2}},
    "grand_total": actual_sum,
    "total_records": actual_count,
    "formatted_report": "Human-readable report using ONLY actual data from results"
}}

Respond with JSON only:"""

    # Round 1: Query Analysis - Understand user intent and grouping requirements
    QUERY_ANALYSIS_PROMPT = """Analyze the following user query to understand their data analysis requirements.

## User Query:
"{user_query}"

## Available Data Fields (JSON Schema):
{field_schema}

## Task:
Analyze the query and identify:
1. What groupings/dimensions does the user want? (e.g., by customer, by year, by category)
2. What is the ORDER of groupings? (e.g., "first by customer, then by year" means customer is primary, year is secondary)
3. What metric/amount should be aggregated? (e.g., total sales, total amount, count)
4. What time granularity if any? (yearly, monthly, quarterly, or none)
5. Any filters or conditions?
6. What type of aggregation? Pay special attention to:
   - If user asks for "minimum", "lowest", "smallest", "least" -> aggregation_type = "min"
   - If user asks for "maximum", "highest", "largest", "most", "top" -> aggregation_type = "max"
   - If user asks for BOTH min AND max (e.g., "min and max", "lowest and highest") -> aggregation_type = "min_max"
   - If user asks for "total", "sum", or just wants amounts aggregated -> aggregation_type = "sum"
   - If user asks for "average", "mean" -> aggregation_type = "avg"
   - If user asks for "count", "how many" -> aggregation_type = "count"
7. CRITICAL - Report type detection:
   - "detail": User wants to see INDIVIDUAL ITEMS/RECORDS. Keywords: "details", "list", "show all", "each", "every", "items", "individual", "breakdown", "itemized", "what items", "what did we buy"
   - "summary": User wants AGGREGATED totals only. Keywords: "total", "sum", "how much", "aggregate"
   - "comparison": User wants to compare different groups

## Important:
- Pay close attention to ordering keywords like "first", "then", "under each", "within"
- "Group by X, then by Y" means X is the primary grouping, Y is secondary
- "Show Y under each X" means X is primary, Y is secondary
- For MIN/MAX queries, identify which entity (customer, vendor, product) should be shown with the min/max value
- IMPORTANT: If user says "list details", "show details", "list items", they want report_type="detail"

## Response Format (JSON only):
{{
    "grouping_order": ["primary_field", "secondary_field"],  // In order of hierarchy
    "time_granularity": "yearly" | "monthly" | "quarterly" | null,
    "aggregation_field": "field_name_to_sum",
    "aggregation_type": "sum" | "min" | "max" | "min_max" | "avg" | "count",
    "entity_field": "field_to_show_with_min_max",  // For min/max queries, which field identifies the entity (e.g., "Customer Name")
    "filters": {{}},  // Any filter conditions
    "report_type": "summary" | "detail" | "comparison",
    "explanation": "Brief explanation of what the user wants"
}}

Respond with JSON only:"""

    # Round 2: Field Mapping - Match user terms to actual field names
    FIELD_MAPPING_PROMPT = """Match the user's requested fields to the actual data schema fields.

## User's Requested Groupings (in order):
{requested_groupings}

## User's Requested Aggregation:
{requested_aggregation}

## Aggregation Type:
{aggregation_type}

## Entity Field (for min/max queries):
{entity_field}

## Time Granularity:
{time_granularity}

## Available Data Fields:
{field_schema}

## Task:
Map each requested grouping/aggregation to the EXACT field name from the schema.

For example:
- "customer" might map to "Customer Name" field
- "year" is a time grouping extracted from a date field
- "amount" might map to "Total Sales" or "Total Amount"

## Response Format (JSON only):
{{
    "grouping_fields": [
        {{"requested": "customer", "actual_field": "Customer Name", "is_time_grouping": false}},
        {{"requested": "year", "actual_field": "Purchase Date", "is_time_grouping": true, "granularity": "yearly"}}
    ],
    "aggregation_field": {{"requested": "amount", "actual_field": "Total Sales"}},
    "aggregation_type": "{aggregation_type}",
    "entity_field": {{"requested": "{entity_field}", "actual_field": "Customer Name"}},  // Map entity field for min/max
    "date_field": "Purchase Date",  // The date field to use for time groupings
    "success": true,
    "unmapped_fields": []  // Any fields that couldn't be mapped
}}

Respond with JSON only:"""

    # SQL Error Correction Prompt - Used when SQL execution fails
    SQL_ERROR_CORRECTION_PROMPT = """You are a SQL debugging expert. A PostgreSQL query failed with an error. Analyze the error and generate a corrected SQL query.

## Original User Query:
"{user_query}"

## Failed SQL Query:
```sql
{failed_sql}
```

## SQL Error Message:
{error_message}

## Data Schema (Available Fields):
{field_schema}

## Data Structure - CRITICAL:
The documents_data table has TWO separate JSONB columns:
- header_data (JSONB object): Document-level fields like transaction_date, store_name
- line_items (JSONB array): Item-level fields like description, amount, quantity

IMPORTANT: Check the 'source' and 'access_pattern' in the field schema above!
- source='header': Access via dd.header_data->>'field_name' or header_data->>'field_name' in CTE
- source='line_item': Access via item->>'field_name' after jsonb_array_elements()

COMMON MISTAKE: Trying to access header fields (like transaction_date) from line_items.
For example, if transaction_date is in header_data:
- WRONG: item->>'transaction_date' -- this field doesn't exist in line_items!
- CORRECT: header_data->>'transaction_date'

## Common Error Fixes:
1. "column X does not exist" in CTE: The 'item' column is only available AFTER jsonb_array_elements() runs. You CANNOT use item->>'...' in WHERE clause inside the CTE (WITH ... AS block). Move such filters to the outer SELECT.
2. "column must appear in GROUP BY": Add the column to GROUP BY clause.
3. "invalid input syntax for type numeric": Add NULLIF and type checking for numeric casts.
4. Date parsing errors: Use proper date format handling.
5. Aggregate function errors: Ensure aggregates are used with GROUP BY.
6. Empty results: Check if you're accessing a field from the wrong JSONB column (header_data vs line_items).

## Requirements:
1. Analyze the error message carefully
2. Identify the root cause
3. Generate a corrected SQL query that avoids the error
4. CRITICAL: Do NOT put any WHERE clause that references 'item' inside the CTE
5. Document filtering should go in the CTE WHERE clause using dd.document_id
6. Data filtering (on item fields) should go in the outer SELECT's WHERE clause

## Response Format (JSON only):
{{
    "analysis": "Brief explanation of what caused the error",
    "fix_applied": "Description of the fix applied",
    "corrected_sql": "THE COMPLETE CORRECTED SQL QUERY",
    "confidence": 0.0-1.0
}}

Respond with JSON only:"""

    # Round 3: SQL Generation with verified parameters
    VERIFIED_SQL_PROMPT = """Generate a PostgreSQL query using the verified field mappings.

## Original User Query:
"{user_query}"

## Verified Parameters:
- Grouping Fields (in order): {grouping_fields}
- Time Grouping: {time_grouping}
- Aggregation Field: {aggregation_field}
- Aggregation Type: {aggregation_type}
- Entity Field: {entity_field}
- Date Field: {date_field}
- Date Field Source: {date_field_source}
- Report Type: {report_type}
- Filters: {filters}

## Data Structure - CRITICAL:
The documents_data table has TWO separate JSONB columns:
- header_data (JSONB object): Document-level fields like transaction_date, store_name
- line_items (JSONB array): Item-level fields like description, amount, quantity

IMPORTANT: You must access fields from the correct source:
- Header fields (source='header'): Use dd.header_data->>'field_name'
- Line item fields (source='line_item'): Use item->>'field_name' after jsonb_array_elements()

For example, if the date is in header_data:
- Correct: EXTRACT(YEAR FROM (dd.header_data->>'transaction_date')::timestamp)
- WRONG: EXTRACT(YEAR FROM (item->>'transaction_date')::timestamp) -- date is NOT in line_items!

## SQL Templates by Report Type and Aggregation Type:

### For report_type = "detail" (show individual items):
When user wants to see INDIVIDUAL ITEMS with their details:
IMPORTANT: Always include document identifiers like receipt_number, invoice_number, or document_number from header_data!
```sql
WITH expanded_items AS (
    SELECT
        dd.header_data,
        jsonb_array_elements(dd.line_items) as item
    FROM documents_data dd
    JOIN documents d ON dd.document_id = d.id
    {{WHERE_CLAUSE}}
)
SELECT
    COALESCE(header_data->>'receipt_number', header_data->>'invoice_number', header_data->>'document_number', 'N/A') as receipt_number,
    TO_CHAR((header_data->>'transaction_date')::timestamp, 'YYYY-MM') as month,
    (header_data->>'transaction_date')::date as date,
    header_data->>'store_name' as store,
    header_data->>'store_address' as address,
    item->>'description' as item_description,
    ROUND((item->>'amount')::numeric, 2) as amount
FROM expanded_items
WHERE EXTRACT(YEAR FROM (header_data->>'transaction_date')::timestamp) = {{YEAR}}
ORDER BY (header_data->>'transaction_date')::timestamp, item->>'description'
```
NOTE: The receipt_number/invoice_number field is CRITICAL - it uniquely identifies each receipt/invoice document.

## SQL Templates by Aggregation Type (for summary reports):

### For aggregation_type = "sum" (default):
```sql
WITH expanded_items AS (
    SELECT
        dd.header_data,
        jsonb_array_elements(dd.line_items) as item
    FROM documents_data dd
    JOIN documents d ON dd.document_id = d.id
    {{WHERE_CLAUSE}}
)
SELECT
    {{SELECT_FIELDS}},
    COUNT(*) as item_count,
    ROUND(SUM((item->>'{{AMOUNT_FIELD}}')::numeric), 2) as total_amount
FROM expanded_items
GROUP BY {{GROUP_BY_FIELDS}}
ORDER BY {{ORDER_BY_FIELDS}}
```
NOTE: Access header fields as: header_data->>'field_name', line item fields as: item->>'field_name'

### For aggregation_type = "min_max" (finding entities with min AND max values per grouping):
Use a CTE approach with window functions to find BOTH min and max entities per group:
```sql
WITH expanded_items AS (
    SELECT
        dd.header_data,
        jsonb_array_elements(dd.line_items) as item
    FROM documents_data dd
    JOIN documents d ON dd.document_id = d.id
    {{WHERE_CLAUSE}}
),
aggregated AS (
    SELECT
        {{TIME_GROUP_FIELD}} as group_period,
        item->>'{{ENTITY_FIELD}}' as entity_name,
        ROUND(SUM((item->>'{{AMOUNT_FIELD}}')::numeric), 2) as total_amount
    FROM expanded_items
    GROUP BY group_period, entity_name
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY group_period ORDER BY total_amount ASC) as min_rank,
        ROW_NUMBER() OVER (PARTITION BY group_period ORDER BY total_amount DESC) as max_rank
    FROM aggregated
)
SELECT
    group_period,
    entity_name,
    total_amount,
    CASE
        WHEN min_rank = 1 THEN 'MIN'
        WHEN max_rank = 1 THEN 'MAX'
    END as record_type
FROM ranked
WHERE min_rank = 1 OR max_rank = 1
ORDER BY group_period, record_type DESC
```
NOTE: For time grouping from header_data, use: header_data->>'transaction_date' (NOT item!)

### For aggregation_type = "min" (finding entity with minimum value per grouping):
Use window functions to find the entity with the minimum amount per group:
```sql
WITH expanded_items AS (
    SELECT
        dd.header_data,
        jsonb_array_elements(dd.line_items) as item
    FROM documents_data dd
    JOIN documents d ON dd.document_id = d.id
    {{WHERE_CLAUSE}}
),
aggregated AS (
    SELECT
        {{TIME_GROUP_FIELD}} as group_period,
        item->>'{{ENTITY_FIELD}}' as entity_name,
        ROUND(SUM((item->>'{{AMOUNT_FIELD}}')::numeric), 2) as total_amount
    FROM expanded_items
    GROUP BY group_period, entity_name
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY group_period ORDER BY total_amount ASC) as rn
    FROM aggregated
)
SELECT group_period, entity_name, total_amount
FROM ranked
WHERE rn = 1
ORDER BY group_period
```

### For aggregation_type = "max" (finding entity with maximum value per grouping):
```sql
WITH expanded_items AS (
    SELECT
        dd.header_data,
        jsonb_array_elements(dd.line_items) as item
    FROM documents_data dd
    JOIN documents d ON dd.document_id = d.id
    {{WHERE_CLAUSE}}
),
aggregated AS (
    SELECT
        {{TIME_GROUP_FIELD}} as group_period,
        item->>'{{ENTITY_FIELD}}' as entity_name,
        ROUND(SUM((item->>'{{AMOUNT_FIELD}}')::numeric), 2) as total_amount
    FROM expanded_items
    GROUP BY group_period, entity_name
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY group_period ORDER BY total_amount DESC) as rn
    FROM aggregated
)
SELECT group_period, entity_name, total_amount
FROM ranked
WHERE rn = 1
ORDER BY group_period
```

## Requirements:
1. FIRST check report_type:
   - If report_type = "detail": Use the DETAIL template to show INDIVIDUAL ITEMS with date, store, description, amount. NO GROUP BY.
   - If report_type = "summary": Use the aggregation templates with GROUP BY.
2. Choose the appropriate SQL template based on aggregation_type (for summary reports)
3. CRITICAL - For time groupings, check the source of the date field:
   - If date is in header_data (source='header'): Use header_data->>'date_field'
   - If date is in line_items (source='line_item'): Use item->>'date_field'
   - yearly: EXTRACT(YEAR FROM (header_data->>'date_field')::timestamp)::int
   - monthly: TO_CHAR((header_data->>'date_field')::timestamp, 'YYYY-MM')
   - quarterly: CONCAT(EXTRACT(YEAR FROM ...)::int, '-Q', EXTRACT(QUARTER FROM ...)::int)
4. Non-time groupings: Check the source! Use header_data->>'field' or item->>'field' accordingly
5. CRITICAL: Do NOT add any WHERE clause inside the base CTE (WITH ... AS block). Document filtering will be added automatically.
6. For MIN/MAX queries: You MUST use the window function templates above to return ONLY the min/max records, not all records.
7. ALWAYS include header_data in the CTE SELECT so you can access header fields in outer queries.

## Response Format (JSON only):
{{
    "sql": "THE COMPLETE SQL QUERY",
    "explanation": "What this query does",
    "select_fields": ["field1", "field2"],
    "group_by_fields": ["field1", "field2"]
}}

Respond with JSON only:"""

    def __init__(self, llm_client=None, schema_service=None):
        """
        Initialize the SQL generator.

        Args:
            llm_client: LLM client for generating SQL. Must have a generate() method.
            schema_service: Optional SchemaService for formal schema lookup
        """
        self.llm_client = llm_client
        self.schema_service = schema_service

    def generate_sql_with_schema(
        self,
        user_query: str,
        document_ids: List[str],
        db=None
    ) -> SQLGenerationResult:
        """
        Generate SQL using schema-centric approach with dynamic field mapping lookup.

        This method:
        1. Looks up field mappings from extraction_metadata (cached)
        2. Falls back to DataSchema table if not cached
        3. Falls back to LLM-driven inference as last resort

        Args:
            user_query: User's natural language query
            document_ids: List of document IDs to query
            db: Database session for schema lookup

        Returns:
            SQLGenerationResult with generated SQL
        """
        from uuid import UUID

        # Build document filter
        doc_ids_str = ", ".join(f"'{doc_id}'" for doc_id in document_ids)
        table_filter = f"dd.document_id IN ({doc_ids_str})"

        # Get field mappings using schema service
        field_mappings = {}
        if self.schema_service:
            try:
                doc_uuids = [UUID(str(d)) for d in document_ids]
                field_mappings = self.schema_service.get_field_mappings(doc_uuids)
                logger.info(f"[SQL Generator] Retrieved {len(field_mappings)} field mappings via schema service")
            except Exception as e:
                logger.warning(f"[SQL Generator] Schema service lookup failed: {e}")

        # If no field mappings from schema service, try to get from database directly
        if not field_mappings and db:
            field_mappings = self._get_field_mappings_from_db(document_ids, db)

        # Generate SQL with field mappings
        return self.generate_sql(user_query, field_mappings, table_filter)

    def _get_field_mappings_from_db(
        self,
        document_ids: List[str],
        db
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get field mappings directly from database (extraction_metadata or data_schemas).

        Priority:
        1. extraction_metadata.field_mappings (per-document cache)
        2. data_schemas.field_mappings (formal schema)
        3. Dynamic inference from header_data/line_items

        Args:
            document_ids: Document IDs to get mappings for
            db: Database session

        Returns:
            Combined field mappings dictionary
        """
        from sqlalchemy import text

        try:
            doc_ids_str = ", ".join(f"'{doc_id}'" for doc_id in document_ids)

            # Priority 1: Check extraction_metadata
            result = db.execute(text(f"""
                SELECT extraction_metadata->'field_mappings' as field_mappings
                FROM documents_data
                WHERE document_id IN ({doc_ids_str})
                AND extraction_metadata->'field_mappings' IS NOT NULL
                AND jsonb_typeof(extraction_metadata->'field_mappings') = 'object'
                LIMIT 1
            """))

            row = result.fetchone()
            if row and row[0]:
                cached = dict(row[0])
                # Flatten header_fields and line_item_fields
                field_mappings = {}
                if 'header_fields' in cached:
                    field_mappings.update(cached['header_fields'])
                if 'line_item_fields' in cached:
                    field_mappings.update(cached['line_item_fields'])
                if field_mappings:
                    logger.info("[SQL Generator] Using field mappings from extraction_metadata")
                    return field_mappings

            # Priority 2: Check DataSchema table
            result = db.execute(text(f"""
                SELECT DISTINCT ds.field_mappings
                FROM documents_data dd
                JOIN data_schemas ds ON dd.schema_type = ds.schema_type
                WHERE dd.document_id IN ({doc_ids_str})
                AND ds.is_active = true
                AND ds.field_mappings IS NOT NULL
                LIMIT 1
            """))

            row = result.fetchone()
            if row and row[0]:
                formal = dict(row[0])
                field_mappings = {}
                if 'header_fields' in formal:
                    field_mappings.update(formal['header_fields'])
                if 'line_item_fields' in formal:
                    field_mappings.update(formal['line_item_fields'])
                if field_mappings:
                    logger.info("[SQL Generator] Using field mappings from DataSchema table")
                    return field_mappings

            # Priority 3: Infer from actual data
            logger.info("[SQL Generator] Inferring field mappings from document data")
            return self._infer_field_mappings_from_documents(document_ids, db)

        except Exception as e:
            logger.error(f"[SQL Generator] Database field mapping lookup failed: {e}")
            return {}

    def _infer_field_mappings_from_documents(
        self,
        document_ids: List[str],
        db
    ) -> Dict[str, Dict[str, Any]]:
        """
        Infer field mappings from actual document data structure.

        Args:
            document_ids: Document IDs to analyze
            db: Database session

        Returns:
            Inferred field mappings
        """
        from sqlalchemy import text

        try:
            doc_ids_str = ", ".join(f"'{doc_id}'" for doc_id in document_ids)

            # Get sample data
            result = db.execute(text(f"""
                SELECT
                    header_data,
                    line_items->0 as sample_item
                FROM documents_data
                WHERE document_id IN ({doc_ids_str})
                AND line_items IS NOT NULL
                LIMIT 1
            """))

            row = result.fetchone()
            if not row:
                return {}

            header_data = row[0] or {}
            sample_item = row[1] or {}

            field_mappings = {}

            # Process header fields
            for field in header_data.keys():
                if field not in ['field_mappings', 'column_headers']:
                    field_mappings[field] = self._classify_field_for_sql(field, 'header')

            # Process line item fields
            for field in sample_item.keys():
                if field != 'row_number':
                    field_mappings[field] = self._classify_field_for_sql(field, 'line_item')

            return field_mappings

        except Exception as e:
            logger.error(f"[SQL Generator] Field inference failed: {e}")
            return {}

    def _classify_field_for_sql(self, field_name: str, source: str) -> Dict[str, Any]:
        """
        Classify a field for SQL generation purposes.

        Args:
            field_name: Field name
            source: 'header' or 'line_item'

        Returns:
            Field classification
        """
        field_lower = field_name.lower().replace('_', ' ')

        # Semantic patterns
        patterns = [
            ('date', ['date', 'time', 'created', 'updated', 'timestamp'], 'datetime', None),
            ('amount', ['total', 'amount', 'price', 'cost', 'revenue', 'sales', 'subtotal', 'fee', 'tax'], 'number', 'sum'),
            ('quantity', ['quantity', 'qty', 'count', 'units', 'items'], 'number', 'sum'),
            ('entity', ['customer', 'vendor', 'supplier', 'client', 'company', 'store', 'merchant'], 'string', 'group_by'),
            ('category', ['category', 'type', 'class', 'group', 'segment'], 'string', 'group_by'),
            ('product', ['product', 'item', 'description', 'name', 'sku'], 'string', 'group_by'),
            ('region', ['region', 'city', 'country', 'location', 'address', 'area'], 'string', 'group_by'),
            ('person', ['sales rep', 'representative', 'agent', 'manager', 'assignee'], 'string', 'group_by'),
            ('method', ['method', 'payment', 'shipping', 'channel'], 'string', 'group_by'),
            ('identifier', ['id', 'number', 'code', 'reference'], 'string', None),
        ]

        for sem_type, keywords, data_type, aggregation in patterns:
            if any(kw in field_lower for kw in keywords):
                return {
                    'semantic_type': sem_type,
                    'data_type': data_type,
                    'source': source,
                    'aggregation': aggregation
                }

        return {
            'semantic_type': 'unknown',
            'data_type': 'string',
            'source': source,
            'aggregation': None
        }

    def generate_sql(
        self,
        user_query: str,
        field_mappings: Dict[str, Dict[str, Any]],
        table_filter: Optional[str] = None
    ) -> SQLGenerationResult:
        """
        Generate SQL query based on user query and field mappings.

        Uses multi-round LLM analysis when LLM client is available:
        - Round 1: Query analysis to understand intent and grouping order
        - Round 2: Field mapping to match user terms to actual schema fields
        - Round 3: SQL generation with verified parameters

        Args:
            user_query: User's natural language query
            field_mappings: Dynamic field mappings from extracted data
            table_filter: Optional WHERE clause filter for specific documents

        Returns:
            SQLGenerationResult with generated SQL and metadata
        """
        if not self.llm_client:
            logger.warning("No LLM client available, using heuristic SQL generation")
            return self._generate_sql_heuristic(user_query, field_mappings, table_filter)

        try:
            # Use multi-round LLM analysis for better accuracy
            return self._generate_sql_multi_round(user_query, field_mappings, table_filter)

        except Exception as e:
            logger.error(f"Multi-round LLM SQL generation failed: {e}")
            # Fallback to heuristic
            return self._generate_sql_heuristic(user_query, field_mappings, table_filter)

    def _generate_sql_multi_round(
        self,
        user_query: str,
        field_mappings: Dict[str, Dict[str, Any]],
        table_filter: Optional[str] = None
    ) -> SQLGenerationResult:
        """
        Generate SQL using multi-round LLM analysis.

        Round 1: Analyze query to understand grouping order and intent
        Round 2: Map user terms to actual schema field names
        Round 3: Generate SQL with verified parameters
        """
        field_schema = self._format_field_schema(field_mappings)
        field_schema_json = self._format_field_schema_json(field_mappings)

        # ========== Round 1: Query Analysis ==========
        logger.info(f"[LLM SQL] Round 1: Analyzing query intent...")
        analysis_prompt = self.QUERY_ANALYSIS_PROMPT.format(
            user_query=user_query,
            field_schema=field_schema_json
        )

        analysis_response = self.llm_client.generate(analysis_prompt)
        query_analysis = self._parse_json_response(analysis_response)

        if not query_analysis:
            logger.warning("[LLM SQL] Round 1 failed, falling back to heuristic")
            return self._generate_sql_heuristic(user_query, field_mappings, table_filter)

        # Extract aggregation type, entity field, and report type from Round 1
        aggregation_type = query_analysis.get('aggregation_type', 'sum')
        entity_field = query_analysis.get('entity_field', '')
        report_type = query_analysis.get('report_type', 'summary')

        logger.info(f"[LLM SQL] Round 1 result: grouping_order={query_analysis.get('grouping_order')}, "
                   f"time={query_analysis.get('time_granularity')}, "
                   f"aggregation={query_analysis.get('aggregation_field')}, "
                   f"aggregation_type={aggregation_type}, entity_field={entity_field}, "
                   f"report_type={report_type}")

        # ========== Round 2: Field Mapping ==========
        logger.info(f"[LLM SQL] Round 2: Mapping fields to schema...")
        mapping_prompt = self.FIELD_MAPPING_PROMPT.format(
            requested_groupings=json.dumps(query_analysis.get('grouping_order', [])),
            requested_aggregation=query_analysis.get('aggregation_field', 'amount'),
            aggregation_type=aggregation_type,
            entity_field=entity_field or 'none',
            time_granularity=query_analysis.get('time_granularity', 'none'),
            field_schema=field_schema_json
        )

        mapping_response = self.llm_client.generate(mapping_prompt)
        field_mapping_result = self._parse_json_response(mapping_response)

        if not field_mapping_result or not field_mapping_result.get('success', True):
            logger.warning("[LLM SQL] Round 2 failed, falling back to heuristic")
            return self._generate_sql_heuristic(user_query, field_mappings, table_filter)

        logger.info(f"[LLM SQL] Round 2 result: grouping_fields={field_mapping_result.get('grouping_fields')}, "
                   f"aggregation={field_mapping_result.get('aggregation_field')}, "
                   f"date_field={field_mapping_result.get('date_field')}")

        # ========== Round 3: SQL Generation ==========
        logger.info(f"[LLM SQL] Round 3: Generating SQL...")

        # Prepare grouping fields description
        grouping_fields_desc = []
        for gf in field_mapping_result.get('grouping_fields', []):
            if gf.get('is_time_grouping'):
                grouping_fields_desc.append(f"{gf.get('granularity', 'yearly')} (from {gf.get('actual_field', 'date')})")
            else:
                grouping_fields_desc.append(gf.get('actual_field', gf.get('requested', '')))

        # Determine time grouping info
        time_grouping_info = "None"
        for gf in field_mapping_result.get('grouping_fields', []):
            if gf.get('is_time_grouping'):
                time_grouping_info = f"{gf.get('granularity', 'yearly')} from field '{gf.get('actual_field')}'"
                break

        # Get entity field from Round 2 mapping or use the one from Round 1
        mapped_entity_field = field_mapping_result.get('entity_field', {})
        if isinstance(mapped_entity_field, dict):
            entity_field_actual = mapped_entity_field.get('actual_field', entity_field)
        else:
            entity_field_actual = entity_field

        # Use aggregation_type from Round 2 result or fall back to Round 1
        final_aggregation_type = field_mapping_result.get('aggregation_type', aggregation_type)

        # Determine the source of the date field (header vs line_item)
        date_field = field_mapping_result.get('date_field', 'transaction_date')
        date_field_source = 'header'  # Default to header since dates are typically document-level
        if date_field in field_mappings:
            date_field_source = field_mappings[date_field].get('source', 'header')

        sql_prompt = self.VERIFIED_SQL_PROMPT.format(
            user_query=user_query,
            grouping_fields=json.dumps(grouping_fields_desc),
            time_grouping=time_grouping_info,
            aggregation_field=field_mapping_result.get('aggregation_field', {}).get('actual_field', 'amount'),
            aggregation_type=final_aggregation_type,
            entity_field=entity_field_actual or 'description',
            date_field=date_field,
            date_field_source=date_field_source,
            report_type=report_type,
            filters=json.dumps(query_analysis.get('filters', {}))
        )

        sql_response = self.llm_client.generate(sql_prompt)
        sql_result = self._parse_json_response(sql_response)

        if not sql_result or not sql_result.get('sql'):
            logger.warning("[LLM SQL] Round 3 failed, falling back to heuristic")
            return self._generate_sql_heuristic(user_query, field_mappings, table_filter)

        # Build result
        sql_query = sql_result.get('sql', '')

        # Add table filter if provided
        if table_filter:
            sql_query = self._add_table_filter(sql_query, table_filter)

        # Extract grouping field names for result
        grouping_field_names = [
            gf.get('actual_field', gf.get('requested', ''))
            for gf in field_mapping_result.get('grouping_fields', [])
            if not gf.get('is_time_grouping')
        ]

        logger.info(f"[LLM SQL] Round 3 complete. Generated SQL:\n{sql_query[:200]}...")

        return SQLGenerationResult(
            sql_query=sql_query,
            explanation=sql_result.get('explanation', query_analysis.get('explanation', '')),
            grouping_fields=grouping_field_names,
            aggregation_fields=[field_mapping_result.get('aggregation_field', {}).get('actual_field', '')],
            time_granularity=query_analysis.get('time_granularity'),
            success=True
        )

    def correct_sql_error(
        self,
        user_query: str,
        failed_sql: str,
        error_message: str,
        field_mappings: Dict[str, Dict[str, Any]],
        table_filter: Optional[str] = None
    ) -> Optional[SQLGenerationResult]:
        """
        Use LLM to analyze SQL error and generate a corrected query.

        This method is called when SQL execution fails. It sends the failed SQL,
        error message, and schema to the LLM for analysis and correction.

        Args:
            user_query: Original user's natural language query
            failed_sql: The SQL query that failed
            error_message: The error message from PostgreSQL
            field_mappings: Dynamic field mappings from extracted data
            table_filter: Optional document filter to apply

        Returns:
            SQLGenerationResult with corrected SQL, or None if correction failed
        """
        if not self.llm_client:
            logger.warning("[SQL Correction] No LLM client available for SQL correction")
            return None

        try:
            field_schema_json = self._format_field_schema_json(field_mappings)

            # Format the error correction prompt
            prompt = self.SQL_ERROR_CORRECTION_PROMPT.format(
                user_query=user_query,
                failed_sql=failed_sql,
                error_message=error_message,
                field_schema=field_schema_json
            )

            logger.info(f"[SQL Correction] Sending error to LLM for correction: {error_message[:100]}...")

            # Call LLM for correction
            response = self.llm_client.generate(prompt)
            result = self._parse_json_response(response)

            if not result or not result.get('corrected_sql'):
                logger.warning("[SQL Correction] LLM failed to provide corrected SQL")
                return None

            corrected_sql = result.get('corrected_sql', '')
            confidence = result.get('confidence', 0.0)
            analysis = result.get('analysis', '')
            fix_applied = result.get('fix_applied', '')

            logger.info(f"[SQL Correction] LLM analysis: {analysis}")
            logger.info(f"[SQL Correction] Fix applied: {fix_applied}")
            logger.info(f"[SQL Correction] Confidence: {confidence}")

            # Apply table filter if provided and not already in the query
            if table_filter and 'document_id IN' not in corrected_sql:
                corrected_sql = self._add_table_filter(corrected_sql, table_filter)

            logger.info(f"[SQL Correction] Corrected SQL:\n{corrected_sql[:300]}...")

            return SQLGenerationResult(
                sql_query=corrected_sql,
                explanation=f"Corrected: {fix_applied}",
                grouping_fields=[],
                aggregation_fields=[],
                time_granularity=None,
                success=True,
                error=None
            )

        except Exception as e:
            logger.error(f"[SQL Correction] Failed to correct SQL: {e}")
            return None

    def _format_field_schema_json(self, field_mappings: Dict[str, Dict[str, Any]]) -> str:
        """Format field mappings as JSON for LLM prompts.

        Includes source information (header vs line_item) which is CRITICAL
        for the LLM to generate correct SQL that accesses fields from the
        right JSONB column.
        """
        schema_list = []
        for field_name, mapping in field_mappings.items():
            source = mapping.get('source', 'line_item')  # Default to line_item for backwards compatibility
            schema_list.append({
                "field_name": field_name,
                "data_type": mapping.get('data_type', 'string'),
                "semantic_type": mapping.get('semantic_type', 'unknown'),
                "source": source,  # CRITICAL: 'header' or 'line_item'
                "access_pattern": f"header_data->>'{field_name}'" if source == 'header' else f"item->>'{field_name}'",
                "aggregation": mapping.get('aggregation', 'none'),
                "description": mapping.get('description', ''),
                "aliases": mapping.get('aliases', [])
            })
        return json.dumps(schema_list, indent=2)

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response, handling various formats."""
        try:
            # Try to extract JSON from markdown code blocks
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            # Clean up the string
            json_str = json_str.strip()

            # Try to find JSON object boundaries if parsing fails
            if not json_str.startswith('{'):
                start = json_str.find('{')
                end = json_str.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = json_str[start:end]

            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"[LLM SQL] Failed to parse JSON response: {e}")
            logger.debug(f"[LLM SQL] Response was: {response[:500]}...")
            return None

    def _format_field_schema(self, field_mappings: Dict[str, Dict[str, Any]]) -> str:
        """Format field mappings as a readable schema description."""
        lines = []
        for field_name, mapping in field_mappings.items():
            semantic_type = mapping.get('semantic_type', 'unknown')
            data_type = mapping.get('data_type', 'string')
            aggregation = mapping.get('aggregation', 'none')
            description = mapping.get('description', '')

            lines.append(
                f"- {field_name}: {data_type} ({semantic_type})"
                f" [aggregation: {aggregation}] - {description}"
            )

        return "\n".join(lines)

    def _parse_llm_response(self, response: str) -> Optional[SQLGenerationResult]:
        """Parse LLM response to extract SQL and metadata."""
        try:
            # Extract JSON from response
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            result = json.loads(json_str.strip())

            return SQLGenerationResult(
                sql_query=result.get('sql', ''),
                explanation=result.get('explanation', ''),
                grouping_fields=result.get('grouping_fields', []),
                aggregation_fields=result.get('aggregation_fields', []),
                time_granularity=result.get('time_granularity'),
                success=True
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return None

    def _add_table_filter(self, sql: str, table_filter: str) -> str:
        """Add table filter to SQL query.

        Handles both CTE-based queries and simple queries.
        For CTEs, the filter goes inside the CTE; for simple queries, before GROUP BY.

        IMPORTANT: For CTE queries, we MUST NOT use `item` in the WHERE clause of the CTE
        because `item` is only defined after jsonb_array_elements() runs. The filter
        must only use dd.document_id or other table columns.

        Note: Invalid SQL with item references in CTE WHERE clauses are now handled
        by the LLM-driven error correction loop in execute_dynamic_sql_query().
        """

        # Check if this is a CTE-based query (WITH ... AS)
        if 'WITH' in sql.upper() and 'AS' in sql.upper():
            # For CTE queries, add filter inside the CTE's FROM clause
            # Look for the pattern: FROM documents_data dd ... JOIN ... )
            # We need to add WHERE before the closing parenthesis of the CTE

            # Check if there's already a valid WHERE clause in the CTE
            # Match the CTE section
            cte_match = re.search(r'WITH\s+\w+\s+AS\s*\((.*?)\)\s*SELECT', sql, re.IGNORECASE | re.DOTALL)
            if cte_match:
                cte_content = cte_match.group(1)
                if 'WHERE' in cte_content.upper():
                    # There's already a WHERE in CTE, add with AND
                    # Find the WHERE clause in the CTE and append our filter
                    sql = re.sub(
                        r'(JOIN documents d ON dd\.document_id = d\.id\s*WHERE\s+)([^\)]+)(\s*\))',
                        f'\\1{table_filter} AND \\2\\3',
                        sql,
                        flags=re.IGNORECASE
                    )
                    return sql

            # No WHERE clause in CTE, add one
            if 'JOIN documents d ON' in sql:
                # Add WHERE clause after the JOIN in the CTE
                sql = re.sub(
                    r'(JOIN documents d ON dd\.document_id = d\.id)(\s*\))',
                    f'\\1\n    WHERE {table_filter}\\2',
                    sql,
                    flags=re.IGNORECASE
                )
            else:
                # Fallback: add WHERE before the closing paren of CTE
                sql = re.sub(
                    r'(FROM documents_data dd[^)]*?)(\s*\))',
                    f'\\1\n    WHERE {table_filter}\\2',
                    sql,
                    flags=re.IGNORECASE
                )
        else:
            # Simple query (no CTE)
            if "WHERE" in sql.upper():
                # Add to existing WHERE
                sql = re.sub(
                    r'(WHERE\s+)',
                    f'WHERE {table_filter} AND ',
                    sql,
                    flags=re.IGNORECASE
                )
            else:
                # Add WHERE before GROUP BY or ORDER BY
                for keyword in ['GROUP BY', 'ORDER BY', 'LIMIT']:
                    if keyword in sql.upper():
                        sql = re.sub(
                            f'({keyword})',
                            f'WHERE {table_filter}\n\\1',
                            sql,
                            flags=re.IGNORECASE
                        )
                        break

        return sql

    def _generate_sql_heuristic(
        self,
        user_query: str,
        field_mappings: Dict[str, Dict[str, Any]],
        table_filter: Optional[str] = None
    ) -> SQLGenerationResult:
        """
        Generate SQL using heuristics when LLM is not available.

        This provides a fallback that handles common query patterns.
        """
        query_lower = user_query.lower()

        # Find key fields and their sources
        date_field = None
        date_field_source = 'header'  # Default to header for dates
        amount_field = None
        amount_field_source = 'line_item'  # Default to line_item for amounts
        category_fields = []

        for field_name, mapping in field_mappings.items():
            sem_type = mapping.get('semantic_type', '')
            source = mapping.get('source', 'line_item')
            if sem_type == 'date' and not date_field:
                date_field = field_name
                date_field_source = source
            elif sem_type == 'amount' and mapping.get('aggregation') == 'sum':
                amount_field = field_name
                amount_field_source = source
            elif mapping.get('aggregation') == 'group_by':
                category_fields.append((field_name, sem_type, source))
            # Also include entity-type fields (customer, vendor, etc.) as potential grouping fields
            elif sem_type == 'entity':
                category_fields.append((field_name, sem_type, source))

        # Determine correct accessor for date field
        date_accessor = f"header_data->>'{date_field}'" if date_field_source == 'header' else f"item->>'{date_field}'"
        amount_accessor = f"header_data->>'{amount_field}'" if amount_field_source == 'header' else f"item->>'{amount_field}'"

        # Detect time granularity
        time_granularity = None
        time_grouping_sql = ""

        if any(x in query_lower for x in ['by year', 'yearly', 'per year', 'each year', 'and year']):
            time_granularity = "yearly"
            if date_field:
                time_grouping_sql = f"EXTRACT(YEAR FROM ({date_accessor})::timestamp)::int as year"
        elif any(x in query_lower for x in ['by month', 'monthly', 'per month', 'each month', 'and month']):
            time_granularity = "monthly"
            if date_field:
                time_grouping_sql = f"TO_CHAR(({date_accessor})::timestamp, 'YYYY-MM') as month"
        elif any(x in query_lower for x in ['by quarter', 'quarterly', 'per quarter']):
            time_granularity = "quarterly"
            if date_field:
                time_grouping_sql = f"CONCAT(EXTRACT(YEAR FROM ({date_accessor})::timestamp)::int, '-Q', EXTRACT(QUARTER FROM ({date_accessor})::timestamp)::int) as quarter"

        # Detect category groupings
        grouping_fields = []
        grouping_sql_parts = []

        for field_name, sem_type, source in category_fields:
            # Check if this category is mentioned in query
            field_lower = field_name.lower()

            # Handle plural forms: category->categories, status->statuses, etc.
            sem_type_variations = [sem_type]
            if sem_type == 'category':
                sem_type_variations.extend(['categories', 'categorie'])
            elif sem_type == 'status':
                sem_type_variations.extend(['statuses'])
            elif sem_type == 'region':
                sem_type_variations.extend(['regions'])
            elif sem_type == 'product':
                sem_type_variations.extend(['products'])
            elif sem_type == 'entity':
                # Entity can be customer, vendor, client, supplier, etc.
                sem_type_variations.extend(['customer', 'customers', 'customer name', 'vendor', 'vendors',
                                           'client', 'clients', 'supplier', 'suppliers', 'entity', 'entities'])

            # Check various patterns
            matched = False
            for variation in sem_type_variations:
                if (variation in query_lower or
                    f"by {variation}" in query_lower or
                    f"and {variation}" in query_lower or
                    f"group by {variation}" in query_lower or
                    f"grouped by {variation}" in query_lower):
                    matched = True
                    break

            # Also check field name
            if not matched:
                if (field_lower in query_lower or
                    f"by {field_lower}" in query_lower or
                    f"and {field_lower}" in query_lower):
                    matched = True

            if matched:
                grouping_fields.append(field_name)
                # Use correct accessor based on source
                accessor = f"header_data->>'{field_name}'" if source == 'header' else f"item->>'{field_name}'"
                grouping_sql_parts.append(f"{accessor} as {field_name.lower().replace(' ', '_')}")

        # Build SQL
        select_parts = []
        group_by_parts = []
        order_by_parts = []

        # Determine grouping order based on query phrasing
        # Check if user wants category grouping first (e.g., "first by customer", "group by customer name, then by year")
        category_first = False
        if grouping_sql_parts and time_grouping_sql:
            # Check for explicit ordering keywords
            if 'first' in query_lower:
                # Find positions of 'first' relative to time and category keywords
                first_pos = query_lower.find('first')
                time_keywords = ['year', 'month', 'quarter']
                category_mentioned_after_first = False
                time_mentioned_after_first = False

                for keyword in time_keywords:
                    kw_pos = query_lower.find(keyword, first_pos)
                    if kw_pos != -1:
                        time_mentioned_after_first = True
                        break

                # Check if any category/entity keyword comes right after 'first'
                for field_name, _, _ in category_fields:
                    field_lower = field_name.lower()
                    if field_lower in query_lower[first_pos:first_pos+50]:
                        category_mentioned_after_first = True
                        break

                # Also check entity-related keywords after 'first'
                entity_keywords = ['customer', 'vendor', 'client', 'supplier']
                for kw in entity_keywords:
                    if kw in query_lower[first_pos:first_pos+50]:
                        category_mentioned_after_first = True
                        break

                if category_mentioned_after_first:
                    category_first = True

            # Also check natural ordering: which comes first in "by X, then by Y" pattern
            if not category_first:
                # Find first occurrence of category-related vs time-related keywords
                first_category_pos = len(query_lower)
                first_time_pos = len(query_lower)

                for field_name, _, _ in category_fields:
                    pos = query_lower.find(field_name.lower())
                    if pos != -1 and pos < first_category_pos:
                        first_category_pos = pos

                entity_keywords = ['customer', 'vendor', 'client']
                for kw in entity_keywords:
                    pos = query_lower.find(kw)
                    if pos != -1 and pos < first_category_pos:
                        first_category_pos = pos

                time_keywords = ['year', 'month', 'quarter']
                for kw in time_keywords:
                    pos = query_lower.find(kw)
                    if pos != -1 and pos < first_time_pos:
                        first_time_pos = pos

                if first_category_pos < first_time_pos:
                    category_first = True

        if category_first:
            # Add category groupings first
            for sql_part in grouping_sql_parts:
                select_parts.append(sql_part)
                alias = sql_part.split(' as ')[-1]
                group_by_parts.append(alias)
                order_by_parts.append(alias)

            # Then add time grouping
            if time_grouping_sql:
                select_parts.append(time_grouping_sql)
                if time_granularity == "yearly":
                    group_by_parts.append("year")
                    order_by_parts.append("year")
                elif time_granularity == "monthly":
                    group_by_parts.append("month")
                    order_by_parts.append("month")
                elif time_granularity == "quarterly":
                    group_by_parts.append("quarter")
                    order_by_parts.append("quarter")
        else:
            # Default: Add time grouping first
            if time_grouping_sql:
                select_parts.append(time_grouping_sql)
                if time_granularity == "yearly":
                    group_by_parts.append("year")
                    order_by_parts.append("year")
                elif time_granularity == "monthly":
                    group_by_parts.append("month")
                    order_by_parts.append("month")
                elif time_granularity == "quarterly":
                    group_by_parts.append("quarter")
                    order_by_parts.append("quarter")

            # Then add category groupings
            for sql_part in grouping_sql_parts:
                select_parts.append(sql_part)
                # Extract alias from "item->>'Field' as alias"
                alias = sql_part.split(' as ')[-1]
                group_by_parts.append(alias)
                order_by_parts.append(alias)

        # Add aggregations
        select_parts.append("COUNT(*) as item_count")
        if amount_field:
            select_parts.append(f"ROUND(SUM(({amount_accessor})::numeric), 2) as total_amount")

        # Build WHERE clause
        where_clause = ""
        if table_filter:
            where_clause = f"WHERE {table_filter}"

        # Build final SQL - include header_data so we can access both header and line item fields
        sql = f"""
WITH expanded_items AS (
    SELECT dd.header_data, jsonb_array_elements(dd.line_items) as item
    FROM documents_data dd
    JOIN documents d ON dd.document_id = d.id
    {where_clause}
)
SELECT
    {', '.join(select_parts)}
FROM expanded_items
"""

        if group_by_parts:
            sql += f"GROUP BY {', '.join(group_by_parts)}\n"

        if order_by_parts:
            sql += f"ORDER BY {', '.join(order_by_parts)}"

        return SQLGenerationResult(
            sql_query=sql.strip(),
            explanation=f"Aggregating {amount_field or 'data'} by {', '.join(grouping_fields) if grouping_fields else 'all'}"
                       + (f" and {time_granularity}" if time_granularity else ""),
            grouping_fields=grouping_fields,
            aggregation_fields=[amount_field] if amount_field else [],
            time_granularity=time_granularity,
            success=True
        )

    def generate_summary_report(
        self,
        user_query: str,
        query_results: List[Dict[str, Any]],
        field_mappings: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a formatted summary report from query results.

        Uses LLM for intelligent analysis when available, with heuristic fallback.

        Args:
            user_query: Original user query
            query_results: Results from SQL execution
            field_mappings: Field schema information

        Returns:
            Formatted report with hierarchical data and summaries
        """
        if not self.llm_client:
            return self._generate_summary_heuristic(query_results, field_mappings, user_query)

        # Use LLM-driven smart summary generation
        return self._generate_summary_with_llm(user_query, query_results, field_mappings)

    def _generate_summary_with_llm(
        self,
        user_query: str,
        query_results: List[Dict[str, Any]],
        field_mappings: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate summary report using LLM with a simple, natural approach.

        Converts SQL results to markdown table format for better LLM comprehension,
        then lets LLM generate a natural language summary.

        Args:
            user_query: Original user query
            query_results: Results from SQL execution
            field_mappings: Field schema information

        Returns:
            Formatted report with summary text
        """
        if not query_results:
            return {
                "report_title": "No Data Available",
                "hierarchical_data": [],
                "grand_total": 0,
                "total_records": 0,
                "formatted_report": "No data found for your query."
            }

        try:
            # Calculate basic stats from results
            sample = query_results[0]
            columns = list(sample.keys())

            # Find amount column for grand total
            amount_col = None
            for col in columns:
                col_lower = col.lower()
                if any(kw in col_lower for kw in ['amount', 'total', 'sales', 'sum', 'price', 'value']):
                    amount_col = col
                    break

            grand_total = 0
            if amount_col:
                grand_total = sum(float(r.get(amount_col, 0) or 0) for r in query_results)

            # Convert query results to markdown table format
            results_markdown = self._convert_results_to_markdown(query_results, columns)

            # Detect if user wants detailed listing vs summary
            detail_keywords = ['details', 'detail', 'list', 'show all', 'all items', 'individual',
                               'each', 'breakdown', 'itemized', 'every', 'specific']
            query_lower = user_query.lower()
            wants_details = any(keyword in query_lower for keyword in detail_keywords)

            # Determine instruction based on user intent
            if wants_details:
                detail_instruction = "4. IMPORTANT: The user asked for details/list - you MUST include ALL individual items from the data table in your response. Show each item with its details (date, description, amount, etc.). Do NOT just show a summary total."
                report_type = "detailed"
            else:
                detail_instruction = "4. Be concise but complete - summarize the data appropriately"
                report_type = "summary"

            # Simple prompt with markdown-formatted data
            prompt = f"""Based on the user's question and the SQL query results, write a clear {report_type} report.

## User's Question:
"{user_query}"

## SQL Query Results ({len(query_results)} rows):

{results_markdown}

## Instructions:
1. Directly answer the user's question based on the data above
2. If they asked for minimum/maximum values, clearly identify who has the min and max amounts for each year/grouping
3. Format amounts with $ and proper number formatting (e.g., $1,234.56)
{detail_instruction}
5. Use markdown formatting for readability

## CRITICAL - Hierarchical/Tree Layout for Grouped Data:
When showing detailed items, use a TREE/HIERARCHICAL structure instead of a flat table with duplicate values:
- If multiple items share the same receipt number, date, or restaurant, group them together
- Show the parent group (e.g., receipt number, restaurant, date) ONCE as a header
- List the child items (individual line items) indented under their parent
- This avoids repeating the same information multiple times

IMPORTANT: Use the ACTUAL receipt_number/invoice_number from the data as the primary identifier, NOT the month!
- If you see a column like "receipt_number" or "invoice_number", use THAT value in the header
- The month (YYYY-MM) is just for organizing, NOT the receipt identifier

Example of GOOD tree layout (using actual receipt number):
### Receipt #INV-2025-001 - YU SEAFOOD (2025-06-18)
**Address:** 123 Main St
| Item Name | Amount |
|-----------|--------|
| Crispy Milk Tart | $5.99 |
| Ginger Beef Puff | $5.99 |
**Subtotal:** $xx.xx

### Receipt #INV-2025-002 - Mr. Congee Garden LTD (2025-11-08)
**Address:** 456 Oak Ave
| Item Name | Amount |
|-----------|--------|
| Shrimp Fried Rice | $18.75 |
**Subtotal:** $xx.xx


6. Always include subtotals for each group and a grand total at the end

Write the {report_type} report in markdown format using hierarchical layout when appropriate:"""

            response = self.llm_client.generate(prompt)

            # Clean up the response - remove any code block markers
            formatted_report = response.strip()
            if formatted_report.startswith('```'):
                lines = formatted_report.split('\n')
                formatted_report = '\n'.join(
                    line for line in lines
                    if not line.startswith('```')
                )

            logger.info(f"[LLM Summary] Generated natural language summary ({len(formatted_report)} chars)")

            return {
                "report_title": "Query Results Summary",
                "hierarchical_data": [],  # Not needed for natural language response
                "grand_total": round(grand_total, 2),
                "total_records": len(query_results),
                "formatted_report": formatted_report
            }

        except Exception as e:
            logger.error(f"[LLM Summary] LLM summary generation failed: {e}")
            return self._generate_summary_heuristic(query_results, field_mappings, user_query)

    def _convert_results_to_markdown(
        self,
        query_results: List[Dict[str, Any]],
        columns: List[str]
    ) -> str:
        """
        Convert SQL query results to markdown table format.

        Args:
            query_results: List of result dictionaries
            columns: Column names

        Returns:
            Markdown formatted table string
        """
        if not query_results or not columns:
            return "No data available."

        lines = []

        # Header row
        header = "| " + " | ".join(columns) + " |"
        lines.append(header)

        # Separator row
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        lines.append(separator)

        # Data rows
        for row in query_results:
            values = []
            for col in columns:
                val = row.get(col, "")
                # Format numeric values
                if isinstance(val, (int, float)):
                    if 'amount' in col.lower() or 'total' in col.lower() or 'sales' in col.lower() or 'price' in col.lower():
                        values.append(f"${val:,.2f}")
                    else:
                        values.append(str(val))
                else:
                    values.append(str(val) if val is not None else "")
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)

    def _build_formatted_report(self, result: Dict[str, Any]) -> str:
        """Build a formatted markdown report from the result structure."""
        lines = []
        title = result.get('report_title', 'Data Summary Report')
        lines.append(f"### {title}\n")

        for group in result.get('hierarchical_data', []):
            group_name = group.get('group_name', 'Unknown')
            group_total = group.get('group_total', 0)
            lines.append(f"**{group_name}**")

            # Add min/max if present
            if 'min_record' in group:
                min_rec = group['min_record']
                lines.append(f"- Minimum: {min_rec.get('name', 'N/A')} - ${min_rec.get('amount', 0):,.2f}")
            if 'max_record' in group:
                max_rec = group['max_record']
                lines.append(f"- Maximum: {max_rec.get('name', 'N/A')} - ${max_rec.get('amount', 0):,.2f}")

            # Add sub-groups if present
            for sub in group.get('sub_groups', []):
                lines.append(f"  - {sub.get('name', 'N/A')}: ${sub.get('total', 0):,.2f}")

            lines.append(f"- Total: ${group_total:,.2f}")
            lines.append("")

        lines.append(f"**Grand Total: ${result.get('grand_total', 0):,.2f}**")
        lines.append(f"Total Records: {result.get('total_records', 0)}")

        return "\n".join(lines)

    def _generate_summary_heuristic(
        self,
        query_results: List[Dict[str, Any]],
        field_mappings: Dict[str, Dict[str, Any]],
        user_query: str = ""
    ) -> Dict[str, Any]:
        """Generate summary report using heuristics when LLM is not available.

        This is a fallback method that analyzes the actual query results structure
        and generates a summary without hardcoded assumptions about the data.

        Args:
            query_results: Results from SQL execution
            field_mappings: Field schema information
            user_query: Original user query (used to detect min/max requirements)
        """
        if not query_results:
            return {
                "report_title": "No Data Available",
                "hierarchical_data": [],
                "grand_total": 0,
                "total_records": 0,
                "formatted_report": "No data found for your query."
            }

        # Step 1: Analyze column structure from actual results
        sample = query_results[0]
        columns = list(sample.keys())

        # Detect column types dynamically
        amount_col = None
        entity_col = None
        time_col = None
        grouping_cols = []

        for col in columns:
            col_lower = col.lower()
            # Detect amount/value columns
            if any(kw in col_lower for kw in ['amount', 'total', 'sales', 'sum', 'price', 'value']):
                if amount_col is None:
                    amount_col = col
            # Detect count columns (skip these for grouping)
            elif any(kw in col_lower for kw in ['count', 'item_count']):
                continue
            # Detect time-based grouping columns
            elif any(kw in col_lower for kw in ['year', 'month', 'quarter', 'date', 'period']):
                time_col = col
                grouping_cols.append(col)
            # Detect entity columns (names, identifiers)
            elif any(kw in col_lower for kw in ['name', 'customer', 'vendor', 'client', 'entity', 'person']):
                entity_col = col
                grouping_cols.append(col)
            else:
                grouping_cols.append(col)

        # Step 2: Detect query intent from user query
        query_lower = user_query.lower() if user_query else ""
        is_min_max_query = any(kw in query_lower for kw in ['min', 'max', 'minimum', 'maximum', 'lowest', 'highest'])

        # Step 3: Calculate totals
        grand_total = sum(float(r.get(amount_col, 0) or 0) for r in query_results) if amount_col else 0
        total_records = len(query_results)

        # Step 4: Build report based on detected structure
        hierarchical_data = []

        # Determine primary and secondary grouping columns
        primary_col = time_col if time_col else (grouping_cols[0] if grouping_cols else None)
        secondary_col = entity_col if entity_col and entity_col != primary_col else None
        if not secondary_col and len(grouping_cols) > 1:
            secondary_col = [c for c in grouping_cols if c != primary_col][0] if primary_col else None

        # Build dynamic title based on actual columns
        title_parts = []
        if amount_col:
            title_parts.append(amount_col.replace('_', ' ').title())
        if is_min_max_query:
            title_parts.insert(0, "Min/Max")
        if primary_col:
            title_parts.append(f"by {primary_col.replace('_', ' ').title()}")
        report_title = " ".join(title_parts) if title_parts else "Data Summary"

        # Group data by primary column
        if primary_col:
            primary_groups = {}
            for row in query_results:
                key = row.get(primary_col)
                if key is None:
                    continue
                key = str(key)
                if key not in primary_groups:
                    primary_groups[key] = []
                primary_groups[key].append(row)

            for group_key in sorted(primary_groups.keys()):
                rows = primary_groups[group_key]
                group_total = sum(float(r.get(amount_col, 0) or 0) for r in rows) if amount_col else 0

                group_data = {
                    "group_name": group_key,
                    "group_total": round(group_total, 2),
                    "record_count": len(rows)
                }

                # For min/max queries, find min and max records
                if is_min_max_query and amount_col and secondary_col:
                    valid_rows = [r for r in rows if r.get(amount_col) is not None]
                    if valid_rows:
                        min_row = min(valid_rows, key=lambda r: float(r.get(amount_col, 0) or 0))
                        max_row = max(valid_rows, key=lambda r: float(r.get(amount_col, 0) or 0))
                        group_data["min_record"] = {
                            "name": str(min_row.get(secondary_col, 'Unknown')),
                            "amount": round(float(min_row.get(amount_col, 0) or 0), 2)
                        }
                        group_data["max_record"] = {
                            "name": str(max_row.get(secondary_col, 'Unknown')),
                            "amount": round(float(max_row.get(amount_col, 0) or 0), 2)
                        }

                # Add sub-groups if we have a secondary column
                elif secondary_col:
                    sub_groups = []
                    for row in rows:
                        sub_key = row.get(secondary_col)
                        if sub_key is None:
                            continue
                        sub_groups.append({
                            "name": str(sub_key),
                            "total": round(float(row.get(amount_col, 0) or 0), 2) if amount_col else 0,
                            "count": int(row.get('item_count', 1) or 1)
                        })
                    if sub_groups:
                        group_data["sub_groups"] = sub_groups

                hierarchical_data.append(group_data)
        else:
            # No grouping column found, create flat list
            for row in query_results:
                hierarchical_data.append({
                    "group_name": "All",
                    "group_total": round(float(row.get(amount_col, 0) or 0), 2) if amount_col else 0,
                    "record_count": 1
                })

        # Step 5: Build formatted report dynamically
        formatted_report = self._build_dynamic_formatted_report(
            report_title=report_title,
            hierarchical_data=hierarchical_data,
            grand_total=grand_total,
            total_records=total_records
        )

        # Step 6: Build summaries by each grouping dimension
        summaries = {}
        for col in grouping_cols:
            summary = {}
            for row in query_results:
                key = row.get(col)
                if key is None:
                    continue
                key = str(key)
                if key not in summary:
                    summary[key] = 0
                if amount_col:
                    summary[key] += float(row.get(amount_col, 0) or 0)
            col_name = col.replace(' ', '_').lower()
            summaries[f"summary_by_{col_name}"] = {k: round(v, 2) for k, v in summary.items()}

        return {
            "report_title": report_title,
            "hierarchical_data": hierarchical_data,
            "grand_total": round(grand_total, 2),
            "total_records": total_records,
            "formatted_report": formatted_report,
            **summaries
        }

    def _build_dynamic_formatted_report(
        self,
        report_title: str,
        hierarchical_data: List[Dict[str, Any]],
        grand_total: float,
        total_records: int
    ) -> str:
        """Build a formatted markdown report dynamically based on data structure.

        Args:
            report_title: Title for the report
            hierarchical_data: Grouped data structure
            grand_total: Sum of all amounts
            total_records: Total number of records
        """
        lines = [f"### {report_title}\n"]

        for group in hierarchical_data:
            group_name = group.get('group_name', 'Unknown')
            group_total = group.get('group_total', 0)
            record_count = group.get('record_count', 0)

            lines.append(f"**{group_name}**")

            # Add min/max records if present
            if 'min_record' in group:
                min_rec = group['min_record']
                lines.append(f"- Minimum: {min_rec.get('name', 'N/A')} - ${min_rec.get('amount', 0):,.2f}")
            if 'max_record' in group:
                max_rec = group['max_record']
                lines.append(f"- Maximum: {max_rec.get('name', 'N/A')} - ${max_rec.get('amount', 0):,.2f}")

            # Add sub-groups if present
            for sub in group.get('sub_groups', []):
                sub_name = sub.get('name', 'N/A')
                sub_total = sub.get('total', 0)
                lines.append(f"  - {sub_name}: ${sub_total:,.2f}")

            # Add group total
            lines.append(f"- Total: ${group_total:,.2f} ({record_count} records)")
            lines.append("")

        # Add grand total
        lines.append(f"**Grand Total: ${grand_total:,.2f}**")
        lines.append(f"Total Records: {total_records}")

        return "\n".join(lines)


def generate_analytics_sql(
    user_query: str,
    field_mappings: Dict[str, Dict[str, Any]],
    llm_client=None,
    document_filter: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to generate SQL for analytics queries.

    Args:
        user_query: Natural language query
        field_mappings: Dynamic field schema
        llm_client: Optional LLM client
        document_filter: Optional document filter

    Returns:
        Tuple of (SQL query string, metadata dict)
    """
    generator = LLMSQLGenerator(llm_client)
    result = generator.generate_sql(user_query, field_mappings, document_filter)

    metadata = {
        "explanation": result.explanation,
        "grouping_fields": result.grouping_fields,
        "aggregation_fields": result.aggregation_fields,
        "time_granularity": result.time_granularity,
        "success": result.success,
        "error": result.error
    }

    return result.sql_query, metadata
