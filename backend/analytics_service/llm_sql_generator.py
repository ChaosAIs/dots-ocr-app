"""
LLM-based SQL Generator for Dynamic Data Analysis

This module uses LLM to generate SQL queries based on:
1. User's natural language query
2. Dynamic data schema (field mappings from extracted data)
3. Available data in the database

The generated SQL is executed against the JSONB line_items data.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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
The data is stored in a PostgreSQL table with the following structure:
- Table: documents_data
- Column: line_items (JSONB array containing rows of data)
- Each line item is a JSON object with the following fields:

### Available Fields and Their Types:
{field_schema}

## User Query:
"{user_query}"

## Requirements:
1. Use jsonb_array_elements() to expand line_items array
2. Use proper type casting: (item->>'field_name')::numeric for numbers, (item->>'field_name')::timestamp for dates
3. For date grouping:
   - Yearly: EXTRACT(YEAR FROM (item->>'date_field')::timestamp)
   - Monthly: TO_CHAR((item->>'date_field')::timestamp, 'YYYY-MM')
   - Quarterly: CONCAT(EXTRACT(YEAR FROM (item->>'date_field')::timestamp), '-Q', EXTRACT(QUARTER FROM (item->>'date_field')::timestamp))
4. Use ROUND() for monetary values to 2 decimal places
5. Always include COUNT(*) as item_count in aggregations
6. Order results logically (by date ascending, then by category)

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

    SUMMARY_GENERATION_PROMPT = """Based on the query results, generate a comprehensive summary report.

## User Query:
"{user_query}"

## Query Results (JSON):
{query_results}

## Field Mappings:
{field_schema}

## Requirements:
Generate a well-formatted report that includes:
1. Main data grouped hierarchically as requested
2. Subtotals for each grouping level
3. Grand totals at the end
4. Summary by each dimension (e.g., total by year, total by category)

Format the report in a clear, readable structure with proper indentation and currency formatting.

Return the report as a JSON object with this structure:
{{
    "report_title": "Title of the report",
    "hierarchical_data": [
        {{
            "group_name": "2023",
            "group_total": 166217.58,
            "sub_groups": [
                {{"name": "Electronics", "total": 69916.60, "count": 27}},
                {{"name": "Furniture", "total": 32833.57, "count": 13}}
            ]
        }}
    ],
    "summary_by_year": {{"2023": 166217.58, "2024": 334346.63, "2025": 83626.91}},
    "summary_by_category": {{"Electronics": 238817.15, "Furniture": 102179.02}},
    "grand_total": 584191.11,
    "total_records": 200,
    "formatted_report": "Human-readable formatted report text"
}}"""

    def __init__(self, llm_client=None):
        """
        Initialize the SQL generator.

        Args:
            llm_client: LLM client for generating SQL. Must have a generate() method.
        """
        self.llm_client = llm_client

    def generate_sql(
        self,
        user_query: str,
        field_mappings: Dict[str, Dict[str, Any]],
        table_filter: Optional[str] = None
    ) -> SQLGenerationResult:
        """
        Generate SQL query based on user query and field mappings.

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
            # Format field schema for prompt
            field_schema = self._format_field_schema(field_mappings)

            prompt = self.SQL_GENERATION_PROMPT.format(
                field_schema=field_schema,
                user_query=user_query
            )

            logger.info(f"[LLM SQL Generator] Generating SQL for: {user_query[:100]}...")

            response = self.llm_client.generate(prompt)

            # Parse LLM response
            result = self._parse_llm_response(response)

            if result:
                # Add table filter if provided
                if table_filter and result.sql_query:
                    result.sql_query = self._add_table_filter(result.sql_query, table_filter)

                return result

            logger.warning("Failed to parse LLM response, falling back to heuristic")
            return self._generate_sql_heuristic(user_query, field_mappings, table_filter)

        except Exception as e:
            logger.error(f"LLM SQL generation failed: {e}")
            return self._generate_sql_heuristic(user_query, field_mappings, table_filter)

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
        """Add table filter to SQL query."""
        # Find WHERE clause or add one
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
                        f'WHERE {table_filter} \\1',
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

        # Find key fields
        date_field = None
        amount_field = None
        category_fields = []

        for field_name, mapping in field_mappings.items():
            sem_type = mapping.get('semantic_type', '')
            if sem_type == 'date' and not date_field:
                date_field = field_name
            elif sem_type == 'amount' and mapping.get('aggregation') == 'sum':
                amount_field = field_name
            elif mapping.get('aggregation') == 'group_by':
                category_fields.append((field_name, sem_type))

        # Detect time granularity
        time_granularity = None
        time_grouping_sql = ""

        if any(x in query_lower for x in ['by year', 'yearly', 'per year', 'each year', 'and year']):
            time_granularity = "yearly"
            if date_field:
                time_grouping_sql = f"EXTRACT(YEAR FROM (item->>'{date_field}')::timestamp)::int as year"
        elif any(x in query_lower for x in ['by month', 'monthly', 'per month', 'each month', 'and month']):
            time_granularity = "monthly"
            if date_field:
                time_grouping_sql = f"TO_CHAR((item->>'{date_field}')::timestamp, 'YYYY-MM') as month"
        elif any(x in query_lower for x in ['by quarter', 'quarterly', 'per quarter']):
            time_granularity = "quarterly"
            if date_field:
                time_grouping_sql = f"CONCAT(EXTRACT(YEAR FROM (item->>'{date_field}')::timestamp)::int, '-Q', EXTRACT(QUARTER FROM (item->>'{date_field}')::timestamp)::int) as quarter"

        # Detect category groupings
        grouping_fields = []
        grouping_sql_parts = []

        for field_name, sem_type in category_fields:
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
                grouping_sql_parts.append(f"item->>'{field_name}' as {field_name.lower().replace(' ', '_')}")

        # Build SQL
        select_parts = []
        group_by_parts = []
        order_by_parts = []

        # Add time grouping
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

        # Add category groupings
        for sql_part in grouping_sql_parts:
            select_parts.append(sql_part)
            # Extract alias from "item->>'Field' as alias"
            alias = sql_part.split(' as ')[-1]
            group_by_parts.append(alias)
            order_by_parts.append(alias)

        # Add aggregations
        select_parts.append("COUNT(*) as item_count")
        if amount_field:
            select_parts.append(f"ROUND(SUM((item->>'{amount_field}')::numeric), 2) as total_amount")

        # Build WHERE clause
        where_clause = ""
        if table_filter:
            where_clause = f"WHERE {table_filter}"

        # Build final SQL
        sql = f"""
WITH line_items AS (
    SELECT jsonb_array_elements(dd.line_items) as item
    FROM documents_data dd
    JOIN documents d ON dd.document_id = d.id
    {where_clause}
)
SELECT
    {', '.join(select_parts)}
FROM line_items
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
        Generate a formatted summary report from query results using LLM.

        Args:
            user_query: Original user query
            query_results: Results from SQL execution
            field_mappings: Field schema information

        Returns:
            Formatted report with hierarchical data and summaries
        """
        if not self.llm_client:
            return self._generate_summary_heuristic(query_results, field_mappings)

        try:
            field_schema = self._format_field_schema(field_mappings)

            prompt = self.SUMMARY_GENERATION_PROMPT.format(
                user_query=user_query,
                query_results=json.dumps(query_results[:50], indent=2, default=str),  # Limit for context
                field_schema=field_schema
            )

            response = self.llm_client.generate(prompt)

            # Parse response
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            return json.loads(json_str.strip())

        except Exception as e:
            logger.error(f"LLM summary generation failed: {e}")
            return self._generate_summary_heuristic(query_results, field_mappings)

    def _generate_summary_heuristic(
        self,
        query_results: List[Dict[str, Any]],
        field_mappings: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary report using heuristics."""
        if not query_results:
            return {
                "report_title": "No Data Available",
                "hierarchical_data": [],
                "grand_total": 0,
                "total_records": 0
            }

        # Detect grouping columns from results
        sample = query_results[0]
        grouping_cols = []
        amount_col = None

        for col in sample.keys():
            if col in ['total_amount', 'total_sales', 'amount', 'sum']:
                amount_col = col
            elif col not in ['item_count', 'count']:
                grouping_cols.append(col)

        # Calculate totals
        grand_total = sum(float(r.get(amount_col, 0) or 0) for r in query_results)
        total_records = sum(int(r.get('item_count', 0) or 0) for r in query_results)

        # Build hierarchical structure if multiple grouping levels
        hierarchical_data = []

        if len(grouping_cols) >= 2:
            # Group by first level
            primary_col = grouping_cols[0]
            secondary_col = grouping_cols[1]

            primary_groups = {}
            for row in query_results:
                primary_key = row.get(primary_col)
                # Skip rows with None primary key
                if primary_key is None:
                    continue
                primary_key = str(primary_key)
                if primary_key not in primary_groups:
                    primary_groups[primary_key] = []
                primary_groups[primary_key].append(row)

            for primary_key in sorted(primary_groups.keys()):
                rows = primary_groups[primary_key]
                group_total = sum(float(r.get(amount_col, 0) or 0) for r in rows)

                sub_groups = []
                for row in rows:
                    secondary_key = row.get(secondary_col)
                    # Skip rows with None secondary key
                    if secondary_key is None:
                        continue
                    sub_groups.append({
                        "name": str(secondary_key),
                        "total": float(row.get(amount_col, 0) or 0),
                        "count": int(row.get('item_count', 0) or 0)
                    })

                hierarchical_data.append({
                    "group_name": primary_key,
                    "group_total": group_total,
                    "sub_groups": sub_groups
                })
        else:
            # Single level grouping
            for row in query_results:
                group_key = row.get(grouping_cols[0]) if grouping_cols else 'All'
                # Skip rows with None key
                if group_key is None:
                    continue
                hierarchical_data.append({
                    "group_name": str(group_key),
                    "group_total": float(row.get(amount_col, 0) or 0),
                    "count": int(row.get('item_count', 0) or 0)
                })

        # Build summaries by each dimension
        summaries = {}
        for col in grouping_cols:
            summary = {}
            for row in query_results:
                key = row.get(col)
                # Skip None/null keys
                if key is None:
                    continue
                key = str(key)
                if key not in summary:
                    summary[key] = 0
                summary[key] += float(row.get(amount_col, 0) or 0)
            summaries[f"summary_by_{col}"] = summary

        return {
            "report_title": "Purchase Orders Summary Report",
            "hierarchical_data": hierarchical_data,
            "grand_total": round(grand_total, 2),
            "total_records": total_records,
            **summaries
        }


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
