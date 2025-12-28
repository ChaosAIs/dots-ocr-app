"""
LLM-Driven Response Formatter

This module provides intelligent response formatting for query results using LLM analysis.
It dynamically designs report layouts based on:
1. User's original query intent
2. The actual data structure and field schema
3. Result contents and patterns

Key Features:
- LLM-driven report layout design
- Dynamic column formatting based on semantic types
- Smart grouping and summary generation
- Flexible output formats (table, hierarchical, summary)
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ReportColumn:
    """Column definition for report layout."""
    field: str
    header: str
    format: str = "text"  # text, currency, number, date, percentage
    alignment: str = "left"  # left, right, center
    width: str = "auto"  # auto, narrow, wide


@dataclass
class ReportLayout:
    """Report layout configuration."""
    title: str
    description: str = ""
    layout_type: str = "table"  # table, hierarchical, summary, detailed
    columns: List[ReportColumn] = field(default_factory=list)
    grouping_enabled: bool = False
    group_by_fields: List[str] = field(default_factory=list)
    show_subtotals: bool = False
    show_grand_total: bool = True
    currency_symbol: str = "$"
    decimal_places: int = 2
    date_format: str = "YYYY-MM-DD"


# =============================================================================
# LLM Prompts for Dynamic Report Layout
# =============================================================================

REPORT_LAYOUT_PROMPT = """Design the optimal report layout for presenting this query result.

## User Query:
"{user_query}"

## Data Schema (field mappings):
{field_schema}

## Query Result Sample (first 5 rows):
{result_sample}

## Total Records: {total_records}

## Task:
Design a report layout that best presents this data to answer the user's query.

Consider:
1. What columns to show and their order (most important first)
2. How to group/organize the data
3. What summary statistics to include
4. How to format the output (table, list, hierarchical)
5. What title and description to use

## Response Format (JSON only):
{{
    "report_title": "Clear title for the report",
    "report_description": "Brief description of what the report shows",
    "layout_type": "table" | "hierarchical" | "summary" | "detailed",
    "columns": [
        {{
            "field": "field_name",
            "header": "Display Header",
            "format": "currency" | "number" | "date" | "text" | "percentage",
            "alignment": "left" | "right" | "center"
        }}
    ],
    "grouping": {{
        "enabled": true | false,
        "group_by": ["field1", "field2"],
        "show_subtotals": true | false
    }},
    "summary": {{
        "show_grand_total": true | false,
        "total_fields": ["field1"],
        "count_label": "Total Records"
    }},
    "formatting": {{
        "currency_symbol": "$",
        "decimal_places": 2,
        "date_format": "YYYY-MM-DD"
    }}
}}

Respond with JSON only:"""


NATURAL_LANGUAGE_SUMMARY_PROMPT = """Generate a clear, natural language summary of these query results.

## User's Question:
"{user_query}"

## Query Results:
{results_markdown}

## Report Layout:
- Title: {title}
- Type: {layout_type}
- Columns: {columns}

## Instructions:
1. Directly answer the user's question based on the data
2. Highlight key findings and patterns
3. Include specific numbers with proper formatting ($X,XXX.XX for currency)
4. If asking about min/max, clearly identify the records
5. Be concise but complete
6. Use markdown formatting for readability

## CRITICAL - Hierarchical/Tree Layout for Grouped Data:
When presenting detailed items, use a TREE/HIERARCHICAL structure instead of flat tables with duplicate values:
- Group items by their parent category (e.g., receipt, date, store/restaurant)
- Show the parent group ONCE as a section header
- List child items under their parent group
- Include subtotals per group and a grand total at the end

IMPORTANT: Use the ACTUAL receipt_number/invoice_number from the data as the primary identifier, NOT the month!
- If you see a column like "receipt_number" or "invoice_number", use THAT value in the header

Example format (using actual receipt number):
### Receipt #INV-2025-001 - YU SEAFOOD (2025-06-18)
**Address:** 123 Main St
| Item Name | Amount |
|-----------|--------|
| Crispy Milk Tart | $5.99 |
| Ginger Beef Puff | $5.99 |
**Subtotal:** $11.98

This avoids repeating the same receipt number, restaurant, or date for each row.

Write the summary:"""


class ResponseFormatter:
    """
    LLM-driven response formatter for query results.

    Features:
    - Dynamic report layout design using LLM
    - Smart column formatting based on semantic types
    - Flexible output formats
    - Natural language summaries
    """

    def __init__(self, llm_client=None):
        """
        Initialize the response formatter.

        Args:
            llm_client: LLM client for layout design and summary generation
        """
        self.llm_client = llm_client

    def format_query_results(
        self,
        user_query: str,
        results: List[Dict[str, Any]],
        field_mappings: Dict[str, Dict[str, Any]],
        schema_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format query results with LLM-driven layout design.

        Args:
            user_query: Original user query
            results: Query results (list of dicts)
            field_mappings: Field schema with semantic types
            schema_type: Optional schema type for context

        Returns:
            Formatted response with layout, data, and summary
        """
        if not results:
            return {
                "layout": {
                    "title": "No Results",
                    "layout_type": "summary"
                },
                "formatted_data": [],
                "summary": "No data found for your query.",
                "metadata": {
                    "row_count": 0,
                    "schema_type": schema_type
                }
            }

        # Step 1: Design report layout using LLM
        layout = self._design_report_layout(
            user_query=user_query,
            results=results,
            field_mappings=field_mappings
        )

        # Step 2: Format data according to layout
        formatted_data = self._format_data(results, layout, field_mappings)

        # Step 3: Generate summary
        summary = self._generate_summary(
            user_query=user_query,
            results=results,
            layout=layout,
            field_mappings=field_mappings
        )

        # Step 4: Build markdown table if applicable
        markdown_table = self._build_markdown_table(formatted_data, layout)

        return {
            "layout": {
                "title": layout.title,
                "description": layout.description,
                "layout_type": layout.layout_type,
                "columns": [
                    {
                        "field": c.field,
                        "header": c.header,
                        "format": c.format,
                        "alignment": c.alignment
                    }
                    for c in layout.columns
                ],
                "grouping": {
                    "enabled": layout.grouping_enabled,
                    "group_by": layout.group_by_fields
                }
            },
            "formatted_data": formatted_data,
            "markdown_table": markdown_table,
            "summary": summary,
            "metadata": {
                "row_count": len(results),
                "schema_type": schema_type,
                "formatted_at": datetime.utcnow().isoformat()
            }
        }

    def _design_report_layout(
        self,
        user_query: str,
        results: List[Dict[str, Any]],
        field_mappings: Dict[str, Dict[str, Any]]
    ) -> ReportLayout:
        """
        Use LLM to design optimal report layout.

        Args:
            user_query: User's original query
            results: Query results
            field_mappings: Field schema

        Returns:
            ReportLayout configuration
        """
        if not self.llm_client:
            return self._design_layout_heuristically(results, field_mappings)

        try:
            # Prepare sample data
            sample = results[:5]
            result_sample = json.dumps(sample, indent=2, default=str)

            # Format field schema
            field_schema = json.dumps(field_mappings, indent=2)

            prompt = REPORT_LAYOUT_PROMPT.format(
                user_query=user_query,
                field_schema=field_schema,
                result_sample=result_sample,
                total_records=len(results)
            )

            response = self.llm_client.generate(prompt)
            layout_config = self._parse_json_response(response)

            if layout_config:
                return self._build_layout_from_config(layout_config)

        except Exception as e:
            logger.warning(f"[Response Formatter] LLM layout design failed: {e}")

        return self._design_layout_heuristically(results, field_mappings)

    def _design_layout_heuristically(
        self,
        results: List[Dict[str, Any]],
        field_mappings: Dict[str, Dict[str, Any]]
    ) -> ReportLayout:
        """
        Design report layout using heuristics when LLM unavailable.

        Args:
            results: Query results
            field_mappings: Field schema

        Returns:
            ReportLayout configuration
        """
        if not results:
            return ReportLayout(
                title="Query Results",
                layout_type="summary",
                columns=[]
            )

        sample = results[0]
        columns = []

        for col in sample.keys():
            col_lower = col.lower()

            # Determine format based on semantic patterns
            if any(kw in col_lower for kw in ['amount', 'total', 'price', 'cost', 'sales', 'revenue']):
                format_type = 'currency'
                alignment = 'right'
            elif any(kw in col_lower for kw in ['date', 'time', 'created', 'updated']):
                format_type = 'date'
                alignment = 'center'
            elif any(kw in col_lower for kw in ['count', 'quantity', 'qty', 'number']):
                format_type = 'number'
                alignment = 'right'
            elif any(kw in col_lower for kw in ['rate', 'percent', 'ratio']):
                format_type = 'percentage'
                alignment = 'right'
            else:
                format_type = 'text'
                alignment = 'left'

            # Also check field_mappings for semantic type
            if col in field_mappings:
                mapping = field_mappings[col]
                sem_type = mapping.get('semantic_type', '')
                if sem_type == 'amount':
                    format_type = 'currency'
                    alignment = 'right'
                elif sem_type == 'date':
                    format_type = 'date'
                    alignment = 'center'
                elif sem_type == 'quantity':
                    format_type = 'number'
                    alignment = 'right'

            columns.append(ReportColumn(
                field=col,
                header=col.replace('_', ' ').title(),
                format=format_type,
                alignment=alignment
            ))

        return ReportLayout(
            title="Query Results",
            description=f"{len(results)} records found",
            layout_type="table",
            columns=columns,
            show_grand_total=True
        )

    def _build_layout_from_config(self, config: Dict[str, Any]) -> ReportLayout:
        """Build ReportLayout from LLM-generated config."""
        columns = []
        for col_config in config.get('columns', []):
            columns.append(ReportColumn(
                field=col_config.get('field', ''),
                header=col_config.get('header', ''),
                format=col_config.get('format', 'text'),
                alignment=col_config.get('alignment', 'left')
            ))

        grouping = config.get('grouping', {})
        formatting = config.get('formatting', {})

        return ReportLayout(
            title=config.get('report_title', 'Query Results'),
            description=config.get('report_description', ''),
            layout_type=config.get('layout_type', 'table'),
            columns=columns,
            grouping_enabled=grouping.get('enabled', False),
            group_by_fields=grouping.get('group_by', []),
            show_subtotals=grouping.get('show_subtotals', False),
            show_grand_total=config.get('summary', {}).get('show_grand_total', True),
            currency_symbol=formatting.get('currency_symbol', '$'),
            decimal_places=formatting.get('decimal_places', 2),
            date_format=formatting.get('date_format', 'YYYY-MM-DD')
        )

    def _format_data(
        self,
        results: List[Dict[str, Any]],
        layout: ReportLayout,
        field_mappings: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format data according to layout specifications.

        Args:
            results: Raw query results
            layout: Report layout configuration
            field_mappings: Field schema

        Returns:
            Formatted data rows
        """
        formatted_rows = []

        for row in results:
            formatted_row = {}

            for col in layout.columns:
                value = row.get(col.field)

                if value is None:
                    formatted_row[col.field] = ""
                elif col.format == 'currency':
                    try:
                        num_val = float(value)
                        formatted_row[col.field] = f"{layout.currency_symbol}{num_val:,.{layout.decimal_places}f}"
                    except (ValueError, TypeError):
                        formatted_row[col.field] = str(value)
                elif col.format == 'number':
                    try:
                        num_val = float(value)
                        if num_val == int(num_val):
                            formatted_row[col.field] = f"{int(num_val):,}"
                        else:
                            formatted_row[col.field] = f"{num_val:,.{layout.decimal_places}f}"
                    except (ValueError, TypeError):
                        formatted_row[col.field] = str(value)
                elif col.format == 'percentage':
                    try:
                        num_val = float(value)
                        formatted_row[col.field] = f"{num_val:.1f}%"
                    except (ValueError, TypeError):
                        formatted_row[col.field] = str(value)
                elif col.format == 'date':
                    formatted_row[col.field] = str(value) if value else ""
                else:
                    formatted_row[col.field] = str(value) if value else ""

            formatted_rows.append(formatted_row)

        return formatted_rows

    def _generate_summary(
        self,
        user_query: str,
        results: List[Dict[str, Any]],
        layout: ReportLayout,
        field_mappings: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate natural language summary of results.

        Args:
            user_query: Original user query
            results: Query results
            layout: Report layout
            field_mappings: Field schema

        Returns:
            Natural language summary
        """
        if not self.llm_client:
            return self._generate_summary_heuristically(results, layout)

        try:
            # Convert results to markdown table
            columns = [c.field for c in layout.columns] if layout.columns else list(results[0].keys())
            results_markdown = self._convert_to_markdown(results[:20], columns)

            prompt = NATURAL_LANGUAGE_SUMMARY_PROMPT.format(
                user_query=user_query,
                results_markdown=results_markdown,
                title=layout.title,
                layout_type=layout.layout_type,
                columns=", ".join(columns)
            )

            summary = self.llm_client.generate(prompt)
            return summary.strip()

        except Exception as e:
            logger.warning(f"[Response Formatter] LLM summary generation failed: {e}")

        return self._generate_summary_heuristically(results, layout)

    def _generate_summary_heuristically(
        self,
        results: List[Dict[str, Any]],
        layout: ReportLayout
    ) -> str:
        """Generate summary using heuristics when LLM unavailable."""
        if not results:
            return "No data found."

        lines = [f"### {layout.title}\n"]
        lines.append(f"**{len(results)}** records found.\n")

        # Find amount columns and calculate totals
        for col in layout.columns:
            if col.format == 'currency':
                try:
                    total = sum(float(r.get(col.field, 0) or 0) for r in results)
                    lines.append(f"- **{col.header} Total**: {layout.currency_symbol}{total:,.2f}")
                except:
                    pass

        return "\n".join(lines)

    def _build_markdown_table(
        self,
        formatted_data: List[Dict[str, Any]],
        layout: ReportLayout
    ) -> str:
        """
        Build a markdown table from formatted data.

        Args:
            formatted_data: Formatted data rows
            layout: Report layout

        Returns:
            Markdown table string
        """
        if not formatted_data or not layout.columns:
            return ""

        lines = []

        # Header row
        headers = [col.header for col in layout.columns]
        lines.append("| " + " | ".join(headers) + " |")

        # Separator with alignment
        separators = []
        for col in layout.columns:
            if col.alignment == 'right':
                separators.append("---:")
            elif col.alignment == 'center':
                separators.append(":---:")
            else:
                separators.append("---")
        lines.append("| " + " | ".join(separators) + " |")

        # Data rows
        for row in formatted_data:
            values = [str(row.get(col.field, "")) for col in layout.columns]
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)

    def _convert_to_markdown(
        self,
        results: List[Dict[str, Any]],
        columns: List[str]
    ) -> str:
        """Convert results to markdown table for LLM prompts."""
        if not results:
            return "No data"

        lines = []
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

        for row in results:
            values = []
            for col in columns:
                val = row.get(col, "")
                if isinstance(val, (int, float)):
                    if 'amount' in col.lower() or 'total' in col.lower():
                        values.append(f"${val:,.2f}")
                    else:
                        values.append(str(val))
                else:
                    values.append(str(val) if val else "")
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response."""
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            json_str = json_str.strip()

            if not json_str.startswith('{'):
                start = json_str.find('{')
                end = json_str.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = json_str[start:end]

            return json.loads(json_str)

        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"[Response Formatter] Failed to parse JSON: {e}")
            return None

    def format_multi_schema_results(
        self,
        user_query: str,
        results_by_type: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format multi-schema query results.

        Each schema type is formatted separately, then combined into a unified response.

        Args:
            user_query: Original user query
            results_by_type: Results grouped by schema type

        Returns:
            Combined formatted response
        """
        formatted_by_type = {}

        for schema_type, type_result in results_by_type.items():
            data = type_result.get('data', [])
            field_mappings = type_result.get('schema_info', {}).get('field_mappings', {})

            if data:
                formatted = self.format_query_results(
                    user_query=user_query,
                    results=data,
                    field_mappings=field_mappings,
                    schema_type=schema_type
                )
                formatted_by_type[schema_type] = formatted

        # Generate combined summary
        combined_summary = self._generate_combined_summary(
            user_query=user_query,
            formatted_by_type=formatted_by_type
        )

        return {
            "results_by_type": formatted_by_type,
            "combined_summary": combined_summary,
            "schema_types": list(formatted_by_type.keys()),
            "metadata": {
                "total_schemas": len(formatted_by_type),
                "formatted_at": datetime.utcnow().isoformat()
            }
        }

    def _generate_combined_summary(
        self,
        user_query: str,
        formatted_by_type: Dict[str, Any]
    ) -> str:
        """Generate combined summary for multi-schema results."""
        if not formatted_by_type:
            return "No data found for your query."

        if self.llm_client:
            try:
                sections = []
                for schema_type, formatted in formatted_by_type.items():
                    sections.append({
                        "schema_type": schema_type,
                        "row_count": formatted.get('metadata', {}).get('row_count', 0),
                        "summary": formatted.get('summary', '')
                    })

                prompt = f"""Combine these results from different document types into a unified summary.

User Query: "{user_query}"

Results by Type:
{json.dumps(sections, indent=2)}

Write a clear combined summary in markdown:"""

                return self.llm_client.generate(prompt).strip()

            except Exception as e:
                logger.warning(f"[Response Formatter] Combined summary failed: {e}")

        # Fallback
        lines = ["## Combined Results\n"]
        for schema_type, formatted in formatted_by_type.items():
            title = schema_type.replace('_', ' ').title()
            row_count = formatted.get('metadata', {}).get('row_count', 0)
            summary = formatted.get('summary', '')
            lines.append(f"### {title} ({row_count} records)")
            lines.append(summary)
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def format_query_results(
    user_query: str,
    results: List[Dict[str, Any]],
    field_mappings: Dict[str, Dict[str, Any]],
    llm_client=None,
    schema_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to format query results.

    Args:
        user_query: User's original query
        results: Query results
        field_mappings: Field schema
        llm_client: Optional LLM client
        schema_type: Optional schema type

    Returns:
        Formatted response
    """
    formatter = ResponseFormatter(llm_client)
    return formatter.format_query_results(
        user_query=user_query,
        results=results,
        field_mappings=field_mappings,
        schema_type=schema_type
    )
