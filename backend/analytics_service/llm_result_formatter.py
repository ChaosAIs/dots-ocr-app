"""
LLM-Driven Result Formatter for Analytics Queries

This module provides intelligent result formatting that:
1. Uses LLM to determine optimal column formatting based on data
2. Dynamically orders columns based on query context
3. Handles monetary vs non-monetary detection intelligently
4. Generates appropriate summaries for different result sizes

Design Principles:
- No hardcoded field patterns - LLM infers from data
- Schema-aware formatting using field_mappings
- Graceful degradation when LLM unavailable
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ColumnFormat:
    """Format specification for a column."""
    field_name: str
    display_name: str
    data_type: str  # currency, number, date, text, percentage
    alignment: str  # left, right, center
    is_monetary: bool
    decimal_places: int = 2


@dataclass
class FormattingConfig:
    """Complete formatting configuration for a result set."""
    columns: List[ColumnFormat]
    column_order: List[str]
    currency_symbol: str = "$"
    date_format: str = "YYYY-MM-DD"
    group_by_fields: List[str] = field(default_factory=list)
    show_subtotals: bool = False
    show_grand_total: bool = True


class LLMResultFormatter:
    """
    Formats query results using LLM-driven intelligence.

    Key features:
    - Dynamic column type detection
    - Context-aware ordering
    - Intelligent monetary/non-monetary handling
    - Adaptive summary generation
    """

    # Token limits for result processing
    MAX_SAFE_TOKENS = 80000
    MAX_ROWS_FOR_LLM = 40
    SAMPLE_ROWS = 10

    # Prompt for column format inference
    COLUMN_FORMAT_PROMPT = """Analyze these query result columns and determine the best display format.

## Columns and Sample Values
{column_samples}

## Field Schema (if available)
{field_schema}

## Task
For each column, determine:
1. display_name: Human-readable header (Title Case)
2. data_type: currency, number, date, text, or percentage
3. alignment: left (text), right (numbers), center (dates)
4. is_monetary: true if this represents money, false otherwise
5. decimal_places: number of decimal places (usually 2 for currency, 0 for counts)

## CRITICAL - Detecting Monetary vs Non-Monetary
- If column name contains "count", "qty", "quantity", "num", "total_receipts" → NOT monetary (is_monetary=false)
- If values are small integers like 7, 12, 45 AND column suggests count → NOT monetary
- If column name contains "amount", "total", "price", "cost", "avg" with dollar context → monetary (is_monetary=true)
- If values look like 142.12, 994.84 with 2 decimal places → likely monetary

## Response (JSON only)
{{
    "columns": [
        {{
            "field_name": "original_column_name",
            "display_name": "Human Readable Name",
            "data_type": "currency|number|date|text|percentage",
            "alignment": "left|right|center",
            "is_monetary": true|false,
            "decimal_places": 2
        }}
    ],
    "recommended_order": ["field1", "field2", "field3"]
}}
"""

    # Prompt for generating summary header for large result sets
    SUMMARY_HEADER_PROMPT = """Generate a brief summary header for this query result.

## User Question
"{user_query}"

## Result Overview
- Total Records: {row_count}
- Columns: {columns}

## Aggregated Statistics
{statistics}

## Sample Data (first {sample_count} rows)
{sample_markdown}

## Instructions
Write a brief (under 200 words) summary that:
1. States what the data shows
2. Highlights key totals/statistics
3. Notes the full data table follows

CRITICAL:
- Do NOT list individual records
- ONLY mention columns that actually exist in the data
- For counts (like "total_receipts: 7"), do NOT add $ sign - these are counts, not money
- For monetary values (like "grand_total: 994.84"), DO add $ sign
"""

    def __init__(self, llm_client=None, schema_service=None):
        """
        Initialize the formatter.

        Args:
            llm_client: LLM client for intelligent formatting
            schema_service: SchemaService for field metadata
        """
        self.llm_client = llm_client
        self.schema_service = schema_service
        self._format_cache: Dict[str, FormattingConfig] = {}

    def format_results(
        self,
        query_results: List[Dict[str, Any]],
        user_query: str,
        field_mappings: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format query results as markdown.

        Args:
            query_results: List of result dictionaries
            user_query: Original user query (for context)
            field_mappings: Optional field schema

        Returns:
            Formatted markdown string
        """
        if not query_results:
            return "No data available."

        # Get or create formatting config
        config = self._get_formatting_config(query_results, field_mappings)

        # Check result size
        row_count = len(query_results)
        if row_count > self.MAX_ROWS_FOR_LLM:
            # Large result: summary header + raw table
            return self._format_large_result(query_results, user_query, config)
        else:
            # Small result: full formatting
            return self._format_small_result(query_results, user_query, config)

    def _get_formatting_config(
        self,
        query_results: List[Dict[str, Any]],
        field_mappings: Optional[Dict[str, Any]] = None
    ) -> FormattingConfig:
        """Get or create formatting configuration."""
        if not query_results:
            return FormattingConfig(columns=[], column_order=[])

        # Get column names
        columns = list(query_results[0].keys())

        # Try LLM-based format inference
        if self.llm_client:
            try:
                config = self._infer_format_with_llm(query_results, columns, field_mappings)
                if config:
                    return config
            except Exception as e:
                logger.warning(f"[Formatter] LLM format inference failed: {e}")

        # Fallback to heuristic formatting
        return self._infer_format_heuristic(query_results, columns, field_mappings)

    def _infer_format_with_llm(
        self,
        query_results: List[Dict[str, Any]],
        columns: List[str],
        field_mappings: Optional[Dict[str, Any]]
    ) -> Optional[FormattingConfig]:
        """Use LLM to infer column formats."""
        # Build column samples
        samples = {}
        for col in columns:
            values = [str(r.get(col, ''))[:50] for r in query_results[:5]]
            samples[col] = values

        # Format field schema if available
        schema_str = "Not available"
        if field_mappings:
            schema_parts = []
            for name, mapping in field_mappings.items():
                if isinstance(mapping, dict):
                    schema_parts.append(f"- {name}: {mapping.get('data_type', 'unknown')} ({mapping.get('semantic_type', 'unknown')})")
            schema_str = "\n".join(schema_parts) if schema_parts else "Not available"

        prompt = self.COLUMN_FORMAT_PROMPT.format(
            column_samples=json.dumps(samples, indent=2),
            field_schema=schema_str
        )

        response = self.llm_client.generate(prompt)
        result = self._parse_json_response(response)

        if not result or 'columns' not in result:
            return None

        # Build config from LLM response
        col_formats = []
        for col_info in result.get('columns', []):
            col_formats.append(ColumnFormat(
                field_name=col_info.get('field_name', ''),
                display_name=col_info.get('display_name', col_info.get('field_name', '')),
                data_type=col_info.get('data_type', 'text'),
                alignment=col_info.get('alignment', 'left'),
                is_monetary=col_info.get('is_monetary', False),
                decimal_places=col_info.get('decimal_places', 2)
            ))

        return FormattingConfig(
            columns=col_formats,
            column_order=result.get('recommended_order', columns)
        )

    def _infer_format_heuristic(
        self,
        query_results: List[Dict[str, Any]],
        columns: List[str],
        field_mappings: Optional[Dict[str, Any]]
    ) -> FormattingConfig:
        """Infer column formats using heuristics."""
        col_formats = []

        for col in columns:
            col_lower = col.lower()

            # Get sample values
            sample_values = [r.get(col) for r in query_results[:10] if r.get(col) is not None]

            # Determine type from values and name
            is_numeric = all(isinstance(v, (int, float)) or self._is_numeric_string(v) for v in sample_values if v)

            # Detect column type
            # IMPORTANT: Check non-monetary keywords FIRST to avoid false positives
            # (e.g., "average_inventory" should NOT be treated as monetary just because it has "average")
            non_monetary_keywords = ['inventory', 'stock', 'units', 'products', 'items', 'count',
                                     'qty', 'quantity', 'num_', 'total_receipts', 'total_items',
                                     'record_count', 'product_count', 'item_count']

            if any(kw in col_lower for kw in ['date', 'time', 'created', 'updated']):
                data_type = 'date'
                alignment = 'center'
                is_monetary = False
            elif any(kw in col_lower for kw in non_monetary_keywords):
                # Non-monetary counts/quantities - check this BEFORE monetary keywords
                data_type = 'number'
                alignment = 'right'
                is_monetary = False
            elif any(kw in col_lower for kw in ['amount', 'price', 'cost', 'revenue', 'sales', 'value', 'fee', 'charge']):
                # Explicitly monetary keywords (removed 'avg', 'sum', 'total' as they're ambiguous)
                data_type = 'currency'
                alignment = 'right'
                is_monetary = True
            elif any(kw in col_lower for kw in ['rate', 'percent', 'pct']):
                data_type = 'percentage'
                alignment = 'right'
                is_monetary = False
            elif is_numeric:
                data_type = 'number'
                alignment = 'right'
                is_monetary = False
            else:
                data_type = 'text'
                alignment = 'left'
                is_monetary = False

            # Generate display name
            display_name = self._to_display_name(col)

            col_formats.append(ColumnFormat(
                field_name=col,
                display_name=display_name,
                data_type=data_type,
                alignment=alignment,
                is_monetary=is_monetary,
                decimal_places=2 if is_monetary else 0
            ))

        # Order columns
        column_order = self._order_columns(columns, col_formats)

        return FormattingConfig(
            columns=col_formats,
            column_order=column_order
        )

    def _format_large_result(
        self,
        query_results: List[Dict[str, Any]],
        user_query: str,
        config: FormattingConfig
    ) -> str:
        """Format large result set with summary header + raw table."""
        parts = []

        # Generate summary header
        if self.llm_client:
            try:
                header = self._generate_summary_header(query_results, user_query, config)
                parts.append(header)
            except Exception as e:
                logger.warning(f"[Formatter] Summary header generation failed: {e}")
                parts.append(f"## Query Results\n\nFound **{len(query_results):,}** records.\n")
        else:
            parts.append(f"## Query Results\n\nFound **{len(query_results):,}** records.\n")

        parts.append("\n\n---\n\n## Complete Data\n\n")

        # Build markdown table
        table = self._build_markdown_table(query_results, config)
        parts.append(table)

        parts.append(f"\n\n*Showing all {len(query_results):,} records*\n")

        return "".join(parts)

    def _format_small_result(
        self,
        query_results: List[Dict[str, Any]],
        user_query: str,
        config: FormattingConfig
    ) -> str:
        """Format small result set with full formatting."""
        # Build formatted table
        table = self._build_markdown_table(query_results, config)

        # Add summary statistics
        stats = self._calculate_statistics(query_results, config)
        if stats:
            table += "\n\n" + stats

        return table

    def _generate_summary_header(
        self,
        query_results: List[Dict[str, Any]],
        user_query: str,
        config: FormattingConfig
    ) -> str:
        """Generate summary header for large results using LLM."""
        # Calculate statistics
        stats = self._calculate_statistics_detailed(query_results, config)

        # Get sample rows
        sample_rows = query_results[:self.SAMPLE_ROWS]
        sample_table = self._build_markdown_table(sample_rows, config)

        # Get column names
        columns = [c.display_name for c in config.columns]

        prompt = self.SUMMARY_HEADER_PROMPT.format(
            user_query=user_query,
            row_count=len(query_results),
            columns=", ".join(columns),
            statistics=stats,
            sample_count=len(sample_rows),
            sample_markdown=sample_table
        )

        response = self.llm_client.generate(prompt)

        # Clean response
        cleaned = response.strip()
        if cleaned.startswith('```'):
            lines = cleaned.split('\n')
            cleaned = '\n'.join(l for l in lines if not l.startswith('```'))

        return cleaned

    def _build_markdown_table(
        self,
        query_results: List[Dict[str, Any]],
        config: FormattingConfig
    ) -> str:
        """Build markdown table with proper formatting."""
        if not query_results:
            return "No data available."

        # Get ordered columns
        columns = config.column_order if config.column_order else list(query_results[0].keys())

        # Build format lookup
        format_lookup = {c.field_name: c for c in config.columns}

        lines = []

        # Header row
        headers = []
        for col in columns:
            fmt = format_lookup.get(col)
            display = fmt.display_name if fmt else self._to_display_name(col)
            headers.append(display)
        lines.append("| " + " | ".join(headers) + " |")

        # Separator
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

        # Data rows
        for row in query_results:
            values = []
            for col in columns:
                val = row.get(col, "")
                fmt = format_lookup.get(col)
                formatted = self._format_value(val, fmt)
                values.append(formatted)
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)

    def _format_value(self, value: Any, fmt: Optional[ColumnFormat]) -> str:
        """Format a single value according to column format."""
        if value is None:
            return "-"

        if fmt is None:
            return str(value)

        # Convert string numbers
        numeric_value = value
        if isinstance(value, str):
            try:
                cleaned = value.replace(',', '').replace('$', '').strip()
                numeric_value = float(cleaned)
            except (ValueError, AttributeError):
                numeric_value = value

        if isinstance(numeric_value, (int, float)):
            if fmt.is_monetary:
                return f"${numeric_value:,.{fmt.decimal_places}f}"
            elif fmt.data_type == 'percentage':
                return f"{numeric_value:.{fmt.decimal_places}f}%"
            elif fmt.data_type == 'number':
                if fmt.decimal_places == 0 or numeric_value == int(numeric_value):
                    return f"{int(numeric_value):,}"
                return f"{numeric_value:,.{fmt.decimal_places}f}"

        # Text value
        str_val = str(value)
        # Escape pipe for markdown
        str_val = str_val.replace("|", "\\|")
        return str_val

    def _calculate_statistics(
        self,
        query_results: List[Dict[str, Any]],
        config: FormattingConfig
    ) -> str:
        """Calculate and format summary statistics."""
        if not query_results:
            return ""

        format_lookup = {c.field_name: c for c in config.columns}
        stats_parts = []

        for col in config.column_order:
            fmt = format_lookup.get(col)
            if not fmt or fmt.data_type not in ['currency', 'number']:
                continue

            values = []
            for row in query_results:
                val = row.get(col)
                if val is not None:
                    try:
                        if isinstance(val, str):
                            val = float(val.replace(',', '').replace('$', ''))
                        values.append(float(val))
                    except (ValueError, TypeError):
                        pass

            if values:
                total = sum(values)
                if fmt.is_monetary:
                    stats_parts.append(f"**Total {fmt.display_name}:** ${total:,.2f}")
                else:
                    stats_parts.append(f"**Total {fmt.display_name}:** {total:,.0f}")

        return " | ".join(stats_parts) if stats_parts else ""

    def _calculate_statistics_detailed(
        self,
        query_results: List[Dict[str, Any]],
        config: FormattingConfig
    ) -> str:
        """Calculate detailed statistics for summary header."""
        if not query_results:
            return "No data"

        format_lookup = {c.field_name: c for c in config.columns}
        lines = []

        for col in config.column_order:
            fmt = format_lookup.get(col)
            if not fmt:
                continue

            values = [row.get(col) for row in query_results if row.get(col) is not None]

            if not values:
                continue

            # Numeric columns
            if fmt.data_type in ['currency', 'number']:
                numeric_vals = []
                for v in values:
                    try:
                        if isinstance(v, str):
                            v = float(v.replace(',', '').replace('$', ''))
                        numeric_vals.append(float(v))
                    except:
                        pass

                if numeric_vals:
                    total = sum(numeric_vals)
                    avg = total / len(numeric_vals)
                    min_val = min(numeric_vals)
                    max_val = max(numeric_vals)

                    if fmt.is_monetary:
                        lines.append(f"**{fmt.display_name}**: Total ${total:,.2f}, Avg ${avg:,.2f}, Min ${min_val:,.2f}, Max ${max_val:,.2f}")
                    else:
                        lines.append(f"**{fmt.display_name}**: Total {total:,.0f}, Avg {avg:,.1f}, Min {min_val:,.0f}, Max {max_val:,.0f}")
            else:
                # Categorical
                unique = len(set(str(v) for v in values))
                lines.append(f"**{fmt.display_name}**: {unique} unique values")

        return "\n".join(lines) if lines else "No statistics available"

    def _order_columns(
        self,
        columns: List[str],
        formats: List[ColumnFormat]
    ) -> List[str]:
        """Order columns logically."""
        format_lookup = {f.field_name: f for f in formats}

        def priority(col: str) -> int:
            col_lower = col.lower()
            # Header/identifier fields first
            if any(kw in col_lower for kw in ['name', 'store', 'vendor', 'customer']):
                return 0
            if any(kw in col_lower for kw in ['number', 'id', 'receipt', 'invoice']):
                return 1
            if any(kw in col_lower for kw in ['date', 'time']):
                return 2
            # Item description
            if any(kw in col_lower for kw in ['description', 'product', 'item']):
                return 3
            # Quantity before amount
            if any(kw in col_lower for kw in ['quantity', 'qty', 'count']):
                return 4
            if any(kw in col_lower for kw in ['price', 'unit']):
                return 5
            # Amount/total last
            if any(kw in col_lower for kw in ['amount', 'total', 'sum']):
                return 6
            return 3  # Default middle

        return sorted(columns, key=priority)

    def _to_display_name(self, field_name: str) -> str:
        """Convert field name to display name."""
        # Replace underscores with spaces
        name = field_name.replace('_', ' ')
        # Title case
        return name.title()

    def _is_numeric_string(self, value: Any) -> bool:
        """Check if string represents a number."""
        if not isinstance(value, str):
            return False
        try:
            cleaned = value.replace(',', '').replace('$', '').strip()
            float(cleaned)
            return True
        except (ValueError, AttributeError):
            return False

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response."""
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                parts = response.split("```")
                if len(parts) >= 2:
                    json_str = parts[1]

            json_str = json_str.strip()
            if not json_str.startswith('{'):
                start = json_str.find('{')
                end = json_str.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = json_str[start:end]

            return json.loads(json_str)
        except Exception:
            return None
