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

from .llm_result_formatter import LLMResultFormatter

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


@dataclass
class ContentSizeEvaluation:
    """Result of content size evaluation for LLM token limit checking."""
    exceeds_limit: bool
    row_count: int
    column_count: int
    estimated_tokens: int
    markdown_table: str
    column_headers: List[str]


class ContentSizeEvaluator:
    """
    Evaluates SQL query result size to determine if it exceeds LLM token limits.

    When results are too large for LLM processing, the system will:
    1. Generate a summary header via LLM (using metadata + samples)
    2. Stream the raw markdown table directly (without LLM processing)
    """

    # Configuration constants
    LLM_MAX_SAFE_TOKENS = 80000      # Safe threshold (leaves headroom for prompt)
    LLM_MAX_SAFE_ROWS = 40           # Max rows for LLM formatting (output token limit) - increased to handle moderate datasets
    CHARS_PER_TOKEN = 4              # Approximate conversion ratio
    PROMPT_OVERHEAD_TOKENS = 3000    # Reserved for prompt/instructions
    SAMPLE_ROWS_FOR_LLM = 10         # Sample rows for LLM context
    TOP_VALUES_COUNT = 5             # Top N for category columns

    # Column type detection patterns
    CURRENCY_PATTERNS = ['amount', 'total', 'sales', 'price', 'cost', 'revenue', 'sum', 'value']
    DATE_PATTERNS = ['date', 'time', 'created', 'updated', 'timestamp']
    COUNT_PATTERNS = ['count', 'quantity', 'qty', 'num', 'number']

    def __init__(self, max_safe_tokens: int = None):
        """
        Initialize the evaluator.

        Args:
            max_safe_tokens: Override default token limit
        """
        self.max_safe_tokens = max_safe_tokens or self.LLM_MAX_SAFE_TOKENS

    def evaluate(self, query_results: List[Dict[str, Any]]) -> ContentSizeEvaluation:
        """
        Evaluate if query results exceed LLM token limits.

        Args:
            query_results: List of result dictionaries from SQL query

        Returns:
            ContentSizeEvaluation with size metrics and pre-built markdown table
        """
        if not query_results:
            return ContentSizeEvaluation(
                exceeds_limit=False,
                row_count=0,
                column_count=0,
                estimated_tokens=0,
                markdown_table="No data available.",
                column_headers=[]
            )

        # Get column headers from first row and reorder them
        raw_columns = list(query_results[0].keys())
        ordered_columns = self._order_columns(raw_columns)

        # Build the full markdown table with ordered columns
        markdown_table = self._build_markdown_table(query_results, ordered_columns, reorder_columns=False)

        # Estimate token count
        total_chars = len(markdown_table)
        estimated_tokens = (total_chars // self.CHARS_PER_TOKEN) + self.PROMPT_OVERHEAD_TOKENS
        row_count = len(query_results)

        # Check if exceeds limit (either token limit OR row count limit)
        # Row count limit prevents LLM output token exhaustion when formatting many rows
        exceeds_token_limit = estimated_tokens > self.max_safe_tokens
        exceeds_row_limit = row_count > self.LLM_MAX_SAFE_ROWS
        exceeds_limit = exceeds_token_limit or exceeds_row_limit

        if exceeds_limit:
            reason = []
            if exceeds_token_limit:
                reason.append(f"tokens ({estimated_tokens:,} > {self.max_safe_tokens:,})")
            if exceeds_row_limit:
                reason.append(f"rows ({row_count} > {self.LLM_MAX_SAFE_ROWS})")
            logger.info(
                f"[ContentSizeEvaluator] Result set exceeds LLM limit: {' and '.join(reason)}"
            )

        return ContentSizeEvaluation(
            exceeds_limit=exceeds_limit,
            row_count=len(query_results),
            column_count=len(ordered_columns),
            estimated_tokens=estimated_tokens,
            markdown_table=markdown_table,
            column_headers=ordered_columns  # Use ordered columns
        )

    def _order_columns(self, columns: List[str]) -> List[str]:
        """
        Order columns by logical priority for better readability.

        Priority:
        1. Header/Document fields (store, receipt, dates, payment)
        2. Item identifier fields (description, product_name)
        3. Quantity fields
        4. Price fields
        5. Amount/Total fields (last)

        NOTE: Data is normalized to canonical field names (snake_case).
        """
        # Define priority groups using canonical names (lower number = higher priority)
        header_fields = {'store_name', 'store_address', 'receipt_number', 'invoice_number',
                         'customer_name', 'vendor_name', 'payment_method', 'currency'}
        date_fields = {'transaction_date', 'invoice_date', 'order_date', 'date', 'due_date'}
        item_name_fields = {'description', 'product_name', 'sku', 'category'}
        quantity_fields = {'quantity'}
        price_fields = {'unit_price'}
        amount_fields = {'amount', 'total_amount', 'subtotal', 'total_value'}

        def get_priority(col: str) -> int:
            if col in header_fields:
                return 0
            if col in date_fields:
                return 1
            if col in item_name_fields:
                return 2
            if col in quantity_fields:
                return 3
            if col in price_fields:
                return 4
            if col in amount_fields:
                return 5
            return 3  # Default to middle priority

        return sorted(columns, key=get_priority)

    def _build_markdown_table(
        self,
        query_results: List[Dict[str, Any]],
        columns: List[str],
        reorder_columns: bool = True
    ) -> str:
        """
        Build a markdown table from query results with proper formatting.

        Args:
            query_results: List of result dictionaries
            columns: Column names (headers)
            reorder_columns: Whether to reorder columns by priority (default True)

        Returns:
            Formatted markdown table string
        """
        if not query_results or not columns:
            return "No data available."

        # Reorder columns for better readability
        if reorder_columns:
            columns = self._order_columns(columns)

        lines = []

        # Header row - use exact column names from SQL result
        header = "| " + " | ".join(columns) + " |"
        lines.append(header)

        # Separator row
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        lines.append(separator)

        # Data rows with proper formatting
        for row in query_results:
            # Check if this row represents a non-monetary item (e.g., "User Messages")
            is_non_monetary = self._is_non_monetary_item(row)

            values = []
            for col in columns:
                val = row.get(col, "")
                formatted_val = self._format_cell_value(val, col, is_non_monetary=is_non_monetary, item=row)
                values.append(formatted_val)
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)

    def build_grouped_markdown(
        self,
        query_results: List[Dict[str, Any]],
        group_by_columns: List[str] = None,
        item_columns: List[str] = None,
        field_mappings: Dict[str, Dict[str, Any]] = None
    ) -> str:
        """
        Build grouped markdown output - documents grouped with nested line items.

        Args:
            query_results: List of result dictionaries
            group_by_columns: Columns to group by (auto-detected from field_mappings if not provided)
            item_columns: Columns to show in item table (default: description, quantity, unit_price, amount)
            field_mappings: Field mappings from document metadata (used to detect document type)

        Returns:
            Grouped markdown string
        """
        if not query_results:
            return "No data available."

        # Determine grouping columns from field_mappings or data structure
        if not group_by_columns:
            group_by_columns = self._detect_grouping_columns(query_results, field_mappings)

        # Default item columns - filter to only those that exist in data
        # Data is now normalized to canonical field names (description, quantity, amount, etc.)
        all_columns = list(query_results[0].keys())

        if not item_columns:
            # Canonical item column names (after normalization)
            # Line item fields
            canonical_item_columns = ['description', 'quantity', 'unit_price', 'amount',
                                      'product_name', 'sku', 'category', 'date']
            # Summary fields (document-level totals from summary_data)
            summary_columns = ['subtotal', 'tax_amount', 'tip_amount', 'total_amount']

            item_columns = [c for c in canonical_item_columns if c in all_columns]

            # If no canonical columns found, use all non-group columns
            if not item_columns:
                item_columns = [c for c in all_columns if c not in group_by_columns and c not in summary_columns]

        # Identify summary columns present in data (for document-level display)
        summary_columns = ['subtotal', 'tax_amount', 'tip_amount', 'total_amount']
        present_summary_columns = [c for c in summary_columns if c in all_columns]

        # Group the data
        from collections import defaultdict
        groups = defaultdict(list)
        for row in query_results:
            # Build group key from available columns
            key_parts = []
            for col in group_by_columns:
                val = row.get(col, '')
                key_parts.append(str(val) if val else '')
            group_key = tuple(key_parts)
            groups[group_key].append(row)

        # Build grouped markdown
        lines = []
        valid_group_count = 0
        total_amount = 0.0
        total_valid_items = 0

        for group_key, items in groups.items():
            # Build item rows first to check if group has valid data
            display_cols = [c for c in item_columns if c in all_columns]
            if not display_cols:
                continue

            # Collect valid item rows for this group
            item_rows = []
            group_subtotal = 0.0
            for item in items:
                # Check if this row has any meaningful data in item columns
                has_data = False
                for col in display_cols:
                    val = item.get(col)
                    if val is not None and str(val).strip() not in ['', '-', 'None', 'null']:
                        has_data = True
                        break

                # Skip rows with all empty item values
                if not has_data:
                    continue

                # Check if this item's amount is non-monetary (e.g., "User Messages")
                is_non_monetary = self._is_non_monetary_item(item)

                values = []
                for col in display_cols:
                    val = item.get(col, "")
                    # Pass is_non_monetary flag for amount columns, and item for is_currency check
                    formatted_val = self._format_cell_value(val, col, is_non_monetary=is_non_monetary, item=item)
                    values.append(formatted_val)
                    # Track subtotal - only for monetary amounts
                    if col in ['amount', 'total', 'subtotal'] and isinstance(item.get(col), (int, float)):
                        # Don't include non-monetary amounts in the subtotal calculation
                        if not is_non_monetary:
                            group_subtotal += float(item.get(col, 0))
                item_rows.append("| " + " | ".join(values) + " |")

            # Skip groups with no valid items
            if not item_rows:
                continue

            valid_group_count += 1
            total_valid_items += len(item_rows)

            # Build header from group key
            header_parts = []
            for i, col in enumerate(group_by_columns):
                val = group_key[i] if i < len(group_key) else ''
                if val and col in ['store_name', 'customer_name', 'vendor_name']:
                    header_parts.insert(0, val)  # Entity name first
                elif val and col in ['receipt_number', 'invoice_number']:
                    header_parts.append(f"#{val}")
                elif val and 'date' in col.lower():
                    header_parts.append(f"({val})")

            header_text = " - ".join(header_parts) if header_parts else f"Group {valid_group_count}"
            lines.append(f"\n### 📄 {header_text}\n")

            # Add table header
            header = "| " + " | ".join(display_cols) + " |"
            lines.append(header)
            separator = "| " + " | ".join(["---"] * len(display_cols)) + " |"
            lines.append(separator)

            # Add item rows
            lines.extend(item_rows)

            # Display document-level summary fields (from summary_data)
            if present_summary_columns and items:
                # Get summary values from first item (they're the same for all items in group)
                first_item = items[0]
                summary_parts = []
                doc_total = None

                for col in present_summary_columns:
                    val = first_item.get(col)
                    if val is not None and val != '' and val != 'None':
                        try:
                            num_val = float(val)
                            if num_val > 0:
                                label = col.replace('_', ' ').title()
                                summary_parts.append(f"**{label}:** ${num_val:,.2f}")
                                if col == 'total_amount':
                                    doc_total = num_val
                        except (ValueError, TypeError):
                            pass

                if summary_parts:
                    lines.append("\n" + " | ".join(summary_parts))
                    if doc_total:
                        total_amount += doc_total
                elif group_subtotal > 0:
                    # Fallback to calculated subtotal if no summary fields
                    lines.append(f"\n**Subtotal: ${group_subtotal:,.2f}**")
                    total_amount += group_subtotal
            elif group_subtotal > 0:
                # Fallback to calculated subtotal if no summary fields
                lines.append(f"\n**Subtotal: ${group_subtotal:,.2f}**")
                total_amount += group_subtotal

            lines.append("\n---")

        # Add grand total
        if total_amount > 0:
            lines.append(f"\n**Grand Total: ${total_amount:,.2f}**")

        # Use valid counts (excluding empty rows/groups)
        lines.append(f"\n*{valid_group_count} documents, {total_valid_items} line items*")

        return "\n".join(lines)

    # Patterns that indicate the amount is NOT monetary (e.g., message credits, usage counts)
    # These are checked first, but only apply if amount == quantity (indicating a count)
    NON_MONETARY_DESCRIPTION_PATTERNS = [
        'user message', 'messages', 'credits', 'tokens', 'api call', 'request',
        'usage', 'units', 'hours', 'minutes', 'seconds', 'days',
        'points', 'miles'
    ]

    def _format_cell_value(self, value: Any, column_name: str, is_non_monetary: bool = False, item: Dict[str, Any] = None) -> str:
        """
        Format a cell value based on its type and column name.

        Args:
            value: The cell value
            column_name: Column name for type inference
            is_non_monetary: If True, skip currency formatting for amount columns
                           (used for items like "User Messages" where amount is count, not money)
            item: The full row data, used to check is_currency field

        Returns:
            Formatted string value
        """
        if value is None:
            return "-"

        col_lower = column_name.lower()

        # Try to convert string values to numbers for amount/price columns
        # SQL returns JSONB values as strings (e.g., "600.0" instead of 600.0)
        numeric_value = value
        if isinstance(value, str) and value.strip():
            try:
                # Try to parse as number
                cleaned = value.replace(',', '').replace('$', '').strip()
                numeric_value = float(cleaned)
            except (ValueError, AttributeError):
                numeric_value = value

        # Format numeric values
        if isinstance(numeric_value, (int, float)):
            # Check for unit_price column specifically - use is_currency field
            if 'unit_price' in col_lower or 'unit price' in col_lower:
                if item is not None:
                    # Check is_currency field (can be boolean or string from SQL)
                    is_currency_val = item.get('is_currency')
                    if is_currency_val is not None:
                        # Handle string values from SQL (e.g., 'true', 'false')
                        if isinstance(is_currency_val, str):
                            is_currency_val = is_currency_val.lower() == 'true'
                        if is_currency_val is False:
                            # Explicitly marked as not currency
                            if numeric_value == int(numeric_value):
                                return f"{int(numeric_value):,}"
                            return f"{numeric_value:,.2f}"
                # Default: unit_price is typically monetary
                return f"${numeric_value:,.2f}"

            # Currency columns - skip currency formatting if marked as non-monetary
            if any(pattern in col_lower for pattern in self.CURRENCY_PATTERNS):
                if is_non_monetary:
                    # Format as plain number for non-monetary amounts
                    if numeric_value == int(numeric_value):
                        return f"{int(numeric_value):,}"
                    return f"{numeric_value:,.2f}"
                return f"${numeric_value:,.2f}"
            # Count/quantity columns - show as integer
            elif any(pattern in col_lower for pattern in self.COUNT_PATTERNS):
                return f"{int(numeric_value):,}" if numeric_value == int(numeric_value) else f"{numeric_value:,.2f}"
            # Other numeric
            else:
                if numeric_value == int(numeric_value):
                    return str(int(numeric_value))
                return f"{numeric_value:,.2f}"

        # String values
        str_val = str(value)
        # Escape pipe characters in markdown tables
        str_val = str_val.replace("|", "\\|")
        return str_val

    def _is_non_monetary_item(self, item: Dict[str, Any]) -> bool:
        """
        Check if an item's amount should NOT be formatted as currency.

        Checks the `is_currency` boolean field from extraction.
        Falls back to heuristic detection if the field is missing.

        Examples:
        - "User Messages" with is_currency=false → non-monetary (465 messages)
        - "Seats" with is_currency=true → monetary ($50 per seat)

        Args:
            item: The row data containing description and other fields

        Returns:
            True if the item's amount should NOT be formatted as currency
        """
        # Check is_currency field (new schema)
        # Note: From SQL queries, this comes as string 'true'/'false'
        # From direct JSONB extraction, it's a boolean True/False
        is_currency = item.get('is_currency')
        if is_currency is not None:
            # Handle string values from SQL (e.g., 'true', 'false')
            if isinstance(is_currency, str):
                is_currency = is_currency.lower() == 'true'
            # is_currency=true means monetary, is_currency=false means non-monetary
            return not is_currency

        # Fallback to heuristic detection for data without is_currency
        unit_price = item.get('unit_price')
        quantity = item.get('quantity')
        amount = item.get('amount')

        # If unit_price exists and is non-null, it's likely a monetary item
        if unit_price is not None:
            return False

        # If amount equals quantity, it's likely a count (e.g., 465 messages)
        if quantity is not None and amount is not None:
            try:
                qty_val = float(quantity)
                amt_val = float(amount)
                if qty_val == amt_val and qty_val > 1:
                    return True
                if amt_val == 0 and qty_val > 0:
                    return True
            except (ValueError, TypeError):
                pass

        # Check description patterns for usage-based items
        description = str(item.get('description', '')).lower()
        for pattern in self.NON_MONETARY_DESCRIPTION_PATTERNS:
            if pattern in description:
                if unit_price is None:
                    return True

        return False

    def _detect_grouping_columns(
        self,
        query_results: List[Dict[str, Any]],
        field_mappings: Dict[str, Dict[str, Any]] = None
    ) -> List[str]:
        """
        Detect appropriate grouping columns based on field_mappings and data structure.

        Uses field_mappings metadata to determine document type (invoice vs receipt)
        and selects appropriate grouping columns.

        Args:
            query_results: Query result data
            field_mappings: Field mappings from document metadata

        Returns:
            List of column names to group by
        """
        if not query_results:
            return ['store_name', 'receipt_number', 'transaction_date']

        first_row_keys = set(query_results[0].keys())

        # Check field_mappings for document type indicators
        if field_mappings:
            # Invoice indicators in field_mappings
            invoice_fields = {'invoice_number', 'invoice_date', 'vendor_name', 'vendor_address', 'po_number', 'due_date'}
            # Receipt indicators in field_mappings
            receipt_fields = {'receipt_number', 'store_name', 'store_address', 'transaction_date', 'cashier'}

            mapping_keys = set(field_mappings.keys())
            invoice_matches = len(mapping_keys.intersection(invoice_fields))
            receipt_matches = len(mapping_keys.intersection(receipt_fields))

            if invoice_matches > receipt_matches:
                # This is invoice data
                return self._get_invoice_grouping_columns(first_row_keys)
            elif receipt_matches > invoice_matches:
                # This is receipt data
                return self._get_receipt_grouping_columns(first_row_keys)

        # Fallback: detect from actual data columns
        if 'invoice_number' in first_row_keys:
            return self._get_invoice_grouping_columns(first_row_keys)
        elif 'receipt_number' in first_row_keys:
            return self._get_receipt_grouping_columns(first_row_keys)

        # Default to receipt grouping
        return self._get_receipt_grouping_columns(first_row_keys)

    def _get_invoice_grouping_columns(self, available_columns: set) -> List[str]:
        """Get grouping columns for invoice documents."""
        # Priority order for invoice grouping
        invoice_group_priority = [
            ('vendor_name', 'invoice_number', 'invoice_date'),
            ('vendor_name', 'invoice_number'),
            ('invoice_number', 'invoice_date'),
            ('invoice_number',),
        ]

        for group_combo in invoice_group_priority:
            if all(col in available_columns for col in group_combo):
                return list(group_combo)

        # Fallback: use whatever invoice-related columns exist
        fallback = []
        for col in ['vendor_name', 'invoice_number', 'invoice_date']:
            if col in available_columns:
                fallback.append(col)
        return fallback if fallback else ['invoice_number']

    def _get_receipt_grouping_columns(self, available_columns: set) -> List[str]:
        """Get grouping columns for receipt documents."""
        # Priority order for receipt grouping
        receipt_group_priority = [
            ('store_name', 'receipt_number', 'transaction_date'),
            ('store_name', 'receipt_number'),
            ('receipt_number', 'transaction_date'),
            ('receipt_number',),
        ]

        for group_combo in receipt_group_priority:
            if all(col in available_columns for col in group_combo):
                return list(group_combo)

        # Fallback: use whatever receipt-related columns exist
        fallback = []
        for col in ['store_name', 'receipt_number', 'transaction_date']:
            if col in available_columns:
                fallback.append(col)
        return fallback if fallback else ['receipt_number']

    def get_sample_rows(
        self,
        query_results: List[Dict[str, Any]],
        max_rows: int = None
    ) -> List[Dict[str, Any]]:
        """
        Get a sample of rows for LLM context.

        Args:
            query_results: Full query results
            max_rows: Maximum rows to return

        Returns:
            Sample subset of results
        """
        max_rows = max_rows or self.SAMPLE_ROWS_FOR_LLM
        return query_results[:max_rows]

    def calculate_column_statistics(
        self,
        query_results: List[Dict[str, Any]],
        columns: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics for each column in the result set.

        For numeric columns: sum, avg, min, max, count
        For string columns: distinct count, top values

        Args:
            query_results: Full query results
            columns: Column names

        Returns:
            Dict mapping column name to statistics
        """
        stats = {}

        for col in columns:
            col_stats = {"column_name": col}
            values = [row.get(col) for row in query_results if row.get(col) is not None]

            if not values:
                stats[col] = {"column_name": col, "type": "empty", "non_null_count": 0}
                continue

            # Check if numeric
            numeric_values = []
            for v in values:
                if isinstance(v, (int, float)):
                    numeric_values.append(float(v))
                elif isinstance(v, str):
                    try:
                        # Try to parse as number (handle currency, commas)
                        cleaned = re.sub(r'[,$€£¥]', '', v.strip())
                        numeric_values.append(float(cleaned))
                    except (ValueError, AttributeError):
                        pass

            col_lower = col.lower()
            is_currency = any(pattern in col_lower for pattern in self.CURRENCY_PATTERNS)

            if len(numeric_values) == len(values):
                # All values are numeric
                col_stats["type"] = "numeric"
                col_stats["is_currency"] = is_currency
                col_stats["count"] = len(numeric_values)
                col_stats["sum"] = round(sum(numeric_values), 2)
                col_stats["avg"] = round(sum(numeric_values) / len(numeric_values), 2)
                col_stats["min"] = round(min(numeric_values), 2)
                col_stats["max"] = round(max(numeric_values), 2)
            else:
                # Treat as categorical/string
                col_stats["type"] = "categorical"
                col_stats["count"] = len(values)

                # Count distinct values
                unique_values = set(str(v) for v in values)
                col_stats["distinct_count"] = len(unique_values)

                # Get top N most frequent values
                value_counts = {}
                for v in values:
                    str_v = str(v)
                    value_counts[str_v] = value_counts.get(str_v, 0) + 1

                sorted_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
                col_stats["top_values"] = [
                    {"value": v, "count": c}
                    for v, c in sorted_counts[:self.TOP_VALUES_COUNT]
                ]

            stats[col] = col_stats

        return stats

    def format_statistics_for_llm(
        self,
        stats: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Format column statistics into a readable string for LLM prompt.

        Args:
            stats: Statistics dictionary from calculate_column_statistics

        Returns:
            Formatted string for LLM prompt
        """
        lines = []

        for col_name, col_stats in stats.items():
            col_type = col_stats.get("type", "unknown")

            if col_type == "numeric":
                is_currency = col_stats.get("is_currency", False)
                if is_currency:
                    lines.append(f"**{col_name}** (Currency):")
                    lines.append(f"  - Total: ${col_stats['sum']:,.2f}")
                    lines.append(f"  - Average: ${col_stats['avg']:,.2f}")
                    lines.append(f"  - Min: ${col_stats['min']:,.2f}")
                    lines.append(f"  - Max: ${col_stats['max']:,.2f}")
                else:
                    lines.append(f"**{col_name}** (Numeric):")
                    lines.append(f"  - Total: {col_stats['sum']:,.2f}")
                    lines.append(f"  - Average: {col_stats['avg']:,.2f}")
                    lines.append(f"  - Min: {col_stats['min']:,.2f}")
                    lines.append(f"  - Max: {col_stats['max']:,.2f}")

            elif col_type == "categorical":
                lines.append(f"**{col_name}** (Text/Category):")
                lines.append(f"  - Unique Values: {col_stats['distinct_count']}")
                top_vals = col_stats.get("top_values", [])
                if top_vals:
                    top_str = ", ".join([f"{v['value']} ({v['count']})" for v in top_vals[:3]])
                    lines.append(f"  - Top Values: {top_str}")

            elif col_type == "empty":
                lines.append(f"**{col_name}**: No data")

            lines.append("")  # Empty line between columns

        return "\n".join(lines)


class LLMSQLGenerator:
    """
    Generates SQL queries using LLM based on user query and data schema.
    """

    SQL_GENERATION_PROMPT = """You are a SQL expert. Generate a PostgreSQL query to answer the user's question.

## Data Structure
The data is stored in PostgreSQL tables with line items in external storage.

### Tables:
- documents_data (alias: dd) - Document metadata with header_data JSONB
- documents_data_line_items (alias: li) - Line items storage (data JSONB column)

CRITICAL: Fields have different sources:
- source='header': Access via dd.header_data->>'field_name' or header_data->>'field_name'
- source='line_item': Access via li.data->>'field_name' or item->>'field_name'

### Available Fields and Their Types:
{field_schema}

## User Query:
"{user_query}"

## Required SQL Structure (EXTERNAL STORAGE):
Always use this pattern for accessing line items:
```sql
WITH items AS (
    SELECT dd.header_data, li.data as item
    FROM documents_data dd
    JOIN documents_data_line_items li ON li.documents_data_id = dd.id
    {{WHERE_CLAUSE}}
)
SELECT ... FROM items ...
```

### Common requirements:
1. For header fields (source='header'): Use header_data->>'field_name'
2. For line_item fields (source='line_item'): Use item->>'field_name'
3. CRITICAL - Type casting for aggregations:
   - The ->> operator returns TEXT, so SUM/AVG REQUIRE casting!
   - CORRECT: SUM((item->>'quantity')::numeric)
   - CORRECT: AVG((item->>'amount')::numeric)
   - WRONG: SUM(item->>'quantity') -- ERROR: function sum(text) does not exist
4. Use ROUND() for monetary values to 2 decimal places
5. Always include COUNT(*) as item_count in aggregations
6. Order results logically
7. CRITICAL - Text search must be CASE-INSENSITIVE using ILIKE (not LIKE):
   - CORRECT: WHERE (item->>'description') ILIKE '%keyboard%'  -- Finds "Keyboard", "keyboard", "KEYBOARD"
   - WRONG: WHERE (item->>'description') LIKE '%keyboard%'  -- Case-sensitive, misses "Keyboard"
8. CRITICAL - When selecting line item amounts, ALWAYS include is_currency:
   - Add: item->>'is_currency' AS is_currency
   - This flag indicates if the amount is monetary (true) or a count/usage (false)
   - Example: SELECT item->>'description' AS description, (item->>'amount')::numeric AS amount, item->>'is_currency' AS is_currency FROM ...

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
6. NEVER add fields that don't exist in the Query Results - no "Description", "Tax", "Discount", "Supplier ID", etc.
7. If a field is not in the query results, it does NOT exist - do not include it in your response

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

    # Prompt for generating summary header when result set is too large for full LLM processing
    LARGE_RESULT_SUMMARY_PROMPT = """Generate a summary header for a large query result set.

The full data table will be shown separately after this summary. Your job is to provide
a concise overview that helps the user understand the data.

## User's Question:
"{user_query}"

## Dataset Overview:
- Total Records: {row_count:,}
- Columns: {column_names}

## Aggregated Statistics:
{statistics}

## Data Source:
- Document Type: {schema_type}
- Query Explanation: {sql_explanation}

## Sample Data (first {sample_count} rows):
{sample_markdown}

## Instructions:
Write a BRIEF summary header (under 300 words) that:
1. Provides a clear, descriptive title for this result
2. States the total record count
3. Highlights key statistics from the aggregated data above
4. Notes any patterns visible in the sample rows
5. Tells the user that the complete data table follows below

CRITICAL RULES:
- Do NOT list individual records - the full table will be shown after this summary
- Do NOT try to reproduce the data table
- Focus on providing context and key insights
- For currency values: PRESERVE the formatting from sample data
  - Values with $ (e.g., "$50.00") are monetary - keep the $ sign
  - Values without $ (e.g., "600") are non-monetary counts - do NOT add $ sign
- Keep the summary concise and scannable
- ONLY mention columns that appear in the "Columns" list above - NEVER assume or add fields that don't exist
- If a field is not listed in Columns, it does NOT exist in this data

Write the summary header in markdown format:"""

    # Round 1: Query Analysis - Understand user intent and grouping requirements
    QUERY_ANALYSIS_PROMPT = """Analyze the following user query to understand their data analysis requirements.

## User Query:
"{user_query}"

## Available Data Fields (JSON Schema):
{field_schema}

**IMPORTANT about field names:**
- Each field has "field_name" (user's terminology, e.g., "StockQuantity") and "db_field" (actual database name, e.g., "quantity")
- User may refer to fields by original names, but always use "db_field" in your response when referring to actual fields
- Match user terms to the appropriate db_field in your analysis

## Task:
Analyze the query and identify:
1. What groupings/dimensions does the user want? (e.g., by customer, by year, by category)
2. What is the ORDER of groupings? (e.g., "first by customer, then by year" means customer is primary, year is secondary)
3. What metric/amount should be aggregated? (e.g., total sales, total amount, count)
4. What time granularity if any? (yearly, monthly, quarterly, or none)
5. Any filters or conditions? IMPORTANT - Extract ALL filter conditions from the query:

   **Comparison operators:**
   - "stock < 50", "less than 50", "under 50", "below 50" -> {{"operator": "<", "value": 50}}
   - "stock > 100", "more than 100", "above 100", "over 100" -> {{"operator": ">", "value": 100}}
   - "stock = 0", "equals 0", "is 0", "exactly 0" -> {{"operator": "=", "value": 0}}
   - "stock <= 50", "at most 50", "50 or less", "no more than 50" -> {{"operator": "<=", "value": 50}}
   - "stock >= 100", "at least 100", "100 or more", "no less than 100" -> {{"operator": ">=", "value": 100}}

   **Range operator (BETWEEN) - INCLUSIVE bounds:**
   - "price between 50 and 100", "price from 50 to 100", "price 50-100" -> {{"operator": "between", "value": [50, 100]}}
   - "stock between 10 and 50" -> {{"operator": "between", "value": [10, 50]}}
   NOTE: BETWEEN is INCLUSIVE (>= AND <=). Only use "between" when user explicitly says "between".

   **Compound comparisons (NOT the same as BETWEEN) - EXCLUSIVE bounds:**
   CRITICAL: "lower than X but higher than Y" or "higher than Y but lower than X" means TWO SEPARATE conditions with EXCLUSIVE bounds (< and >), NOT BETWEEN!
   - "stock lower than 50 but higher than 30" -> TWO filters: {{"operator": "<", "value": 50}} AND {{"operator": ">", "value": 30}} (means > 30 AND < 50, EXCLUSIVE)
   - "price higher than 100 but lower than 500" -> TWO filters: {{"operator": ">", "value": 100}} AND {{"operator": "<", "value": 500}}
   - "inventory less than 100 but greater than 20" -> TWO filters: {{"operator": "<", "value": 100}} AND {{"operator": ">", "value": 20}}

   For compound conditions on the SAME field, use "compound" operator:
   - "stock lower than 50 but higher than 30" -> {{"operator": "compound", "conditions": [{{"op": ">", "value": 30}}, {{"op": "<", "value": 50}}]}}

   **List operator (IN):**
   - "category in Electronics, Books, Toys" -> {{"operator": "in", "value": ["Electronics", "Books", "Toys"]}}
   - "brand is Nike or Adidas or Puma" -> {{"operator": "in", "value": ["Nike", "Adidas", "Puma"]}}
   - "category is Electronics or Books" -> {{"operator": "in", "value": ["Electronics", "Books"]}}

   **NOT operators:**
   - "category not Electronics" -> {{"operator": "!=", "value": "Electronics"}}
   - "category not in Electronics, Books" -> {{"operator": "not_in", "value": ["Electronics", "Books"]}}

   **String matching (CASE-INSENSITIVE - always use ILIKE in PostgreSQL):**
   - "name contains Apple", "name like Apple" -> {{"operator": "ilike", "value": "%Apple%"}}
   - "name starts with A" -> {{"operator": "ilike", "value": "A%"}}
   - "item keyboard", "has keyboard" -> {{"operator": "ilike", "value": "%keyboard%"}}

   **Filter format:** filters = {{"FieldName": {{"operator": "op", "value": value}}}}
   - Match filter field to actual schema field name (e.g., "stock" matches "StockQuantity", "category" matches "Category")
   - For multiple filters: filters = {{"Field1": {{...}}, "Field2": {{...}}}}
6. What type of aggregation? Pay special attention to:
   - If user asks for "minimum", "lowest", "smallest", "least" -> aggregation_type = "min"
   - If user asks for "maximum", "highest", "largest", "most", "top" -> aggregation_type = "max"
   - If user asks for BOTH min AND max (e.g., "min and max", "lowest and highest") -> aggregation_type = "min_max"
   - If user asks for "total", "sum", or just wants amounts aggregated -> aggregation_type = "sum"
   - If user asks for "average", "mean" -> aggregation_type = "avg"
   - If user asks for "count", "how many" -> aggregation_type = "count"

   **CRITICAL for COUNT queries - Document vs Line Item counting:**
   Use the document types and entities from the Available Data Fields schema to determine what to count:

   **count_level="document"** - Count the DOCUMENTS themselves when user mentions:
   - Document type names from schema: receipt, invoice, order, meal, statement, manifest, report
   - Header-level entities (source='header'): vendor_name, store_name, receipt_number, invoice_number, transaction_date
   - Keywords: "how many receipts", "how many meals", "number of invoices", "count of orders"

   **count_level="line_item"** - Count individual LINE ITEMS when user mentions:
   - Line item fields (source='line_item'): item, product, description, quantity, unit_price
   - Item-level keywords: "how many items", "how many products", "number of entries", "total items purchased"

   **How to decide using schema context:**
   1. Look at the Available Data Fields - identify which are header-level (source='header') vs line-item-level (source='line_item')
   2. If user query mentions terms matching document types or header-level entities → count_level="document"
   3. If user query mentions terms matching line-item field names → count_level="line_item"
   4. If ambiguous, default to count_level="document" since users typically ask about document counts

   Examples:
   - Schema has document_type="receipt" → "how many meals/receipts in 2025" → count_level="document"
   - Schema has header field "vendor_name" → "how many vendors" → count documents grouped by vendor
   - Schema has line_item field "Item" → "how many items did I buy" → count_level="line_item"
   - Schema has line_item field "product" → "how many products" → count_level="line_item"
7. CRITICAL - Report type detection:
   - "detail": User wants to see INDIVIDUAL LINE ITEMS/RECORDS with their details. Keywords: "details", "included details", "include details", "with details", "list", "show all", "each", "every", "items", "individual", "breakdown", "itemized", "what items", "what did we buy", "line items", "products details", "product details"
   - "summary": User wants AGGREGATED totals only (no individual line items). Keywords: "total only", "sum only", "just the total", "aggregate only"
   - "comparison": User wants to compare different groups

   **CRITICAL for "detail" report_type**:
   - When report_type="detail", the SQL must include ALL columns from the Available Data Fields schema
   - Do NOT selectively include only 2-3 columns - include EVERY field in the schema
   - Example: If user asks "list products with details" and schema has 7 fields, SELECT must include all 7 fields

   IMPORTANT for MIN/MAX queries with details:
   - "max and min receipts with details" -> report_type="detail" (show the receipts AND their line items)
   - "max and min receipts included their details" -> report_type="detail"
   - "max and min receipts" -> report_type="summary" (just show receipt totals, no line items)

## CRITICAL - Time Filter vs Time Grouping:
This is VERY important for MIN/MAX queries:

**Time FILTER** (time_filter_only=true): User wants to FILTER data to a time period, NOT group by it.
- "in 2025" = filter to 2025, don't group by year
- "for 2025" = filter to 2025, don't group by year
- "maximum receipt in 2025" = find the ONE maximum across all of 2025
- "minimum and maximum in January" = find ONE min and ONE max for January

**Time GROUPING** (time_filter_only=false): User wants to GROUP results by time period.
- "by month" = group results by month
- "per year" = group results by year
- "monthly breakdown" = group by month
- "maximum receipt per month in 2025" = find max FOR EACH month
- "compare months" = group by month

Examples:
- "max and min receipts in 2025" -> time_filter_only=true, time_granularity=null (just filter to 2025, return ONE max and ONE min overall)
- "max and min receipts by month in 2025" -> time_filter_only=false, time_granularity="monthly" (return max/min FOR EACH month)
- "max and min receipts per month" -> time_filter_only=false, time_granularity="monthly"
- "total sales in Q1 2025" -> time_filter_only=true (filter to Q1, return ONE total)
- "total sales by quarter in 2025" -> time_filter_only=false, time_granularity="quarterly" (return total FOR EACH quarter)

## CRITICAL - Understanding "each meal" / "each receipt" / "per receipt":
When user asks about "each meal", "each receipt", "per meal", "per receipt", they are asking about DOCUMENT-LEVEL aggregation (each receipt/document), NOT line item level:
- "average amount for each meal in 2025" = average TOTAL per receipt (first sum line items per receipt, then average across receipts)
- "average per receipt" = average TOTAL per receipt
- "total for each meal" = total per receipt (NOT per line item)
- "how much did each meal cost" = total per receipt

The correct approach for "average per receipt":
1. First calculate the TOTAL for each receipt (sum of line items within each receipt)
2. Then calculate the AVERAGE of those receipt totals

This is different from "average per line item" which would just be AVG(item amount).

## Important:
- Pay close attention to ordering keywords like "first", "then", "under each", "within"
- "Group by X, then by Y" means X is the primary grouping, Y is secondary
- "Show Y under each X" means X is primary, Y is secondary
- For MIN/MAX queries, identify which entity (customer, vendor, product, receipt) should be shown with the min/max value
- IMPORTANT: If user says "list details", "show details", "list items", they want report_type="detail"
- For receipt-based queries, entity_field should typically be "receipt_number" to identify individual receipts
- When user asks "average for each meal/receipt", they want the average TOTAL per receipt, not average per line item

## Response Format (JSON only):
{{
    "grouping_order": ["primary_field", "secondary_field"],  // In order of hierarchy - EMPTY [] if no grouping wanted
    "time_granularity": "yearly" | "monthly" | "quarterly" | null,
    "time_filter_only": true | false,  // TRUE = just filter by time, FALSE = group by time
    "time_filter_year": 2025 | null,  // The year to filter by if time_filter_only is true
    "time_filter_period": "Q1" | "January" | null,  // Optional: specific quarter or month to filter
    "aggregation_field": "field_name_to_sum",
    "aggregation_type": "sum" | "min" | "max" | "min_max" | "avg" | "count",
    "count_level": "document" | "line_item",  // For count queries: "document" counts receipts/invoices, "line_item" counts individual items
    "entity_field": "field_to_show_with_min_max",  // For min/max queries, which field identifies the entity (e.g., "receipt_number")
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

**CRITICAL: Use db_field values in actual_field!**
- The schema shows "field_name" (user's original column name) and "db_field" (actual database field)
- In your response, ALWAYS use the "db_field" value for "actual_field"
- Example: If user says "stock" and schema has {{"field_name": "StockQuantity", "db_field": "quantity"}}, use "quantity" as actual_field

## Task:
Map each requested grouping/aggregation to the EXACT db_field from the schema.

For example:
- "customer" might map to "Customer Name" field -> use the db_field value
- "year" is a time grouping extracted from a date field
- "stock" maps to "quantity" (the db_field, not the original "StockQuantity")
- "amount" might map to "amount" db_field

## Response Format (JSON only):
{{
    "grouping_fields": [
        {{"requested": "customer", "actual_field": "vendor_name", "is_time_grouping": false}},
        {{"requested": "year", "actual_field": "transaction_date", "is_time_grouping": true, "granularity": "yearly"}}
    ],
    "aggregation_field": {{"requested": "stock", "actual_field": "quantity"}},
    "aggregation_type": "{aggregation_type}",
    "entity_field": {{"requested": "{entity_field}", "actual_field": "description"}},  // Map entity field for min/max
    "date_field": "transaction_date",  // The date field to use for time groupings (use db_field!)
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

## CRITICAL - USE EXACT FIELD NAMES FROM SCHEMA:
You MUST use the EXACT field names from the Data Schema above!
- Look at the 'field_name' values in the schema (e.g., 'Total Sales', 'Product', 'Purchase Date')
- DO NOT use generic names like 'amount', 'description', 'date', 'quantity'
- USE the actual field names exactly as shown in the schema

## Data Structure (CRITICAL - USE EXACT COLUMN NAMES):

### Table: documents_data (alias: dd)
- id (UUID): Primary key
- document_id (UUID): FK to documents table
- header_data (JSONB): Document-level fields (vendor_name, invoice_date, etc.)
- summary_data (JSONB): Aggregate totals (subtotal, tax_amount, total_amount, etc.)

### Table: documents_data_line_items (alias: li)
- id (UUID): Primary key
- documents_data_id (UUID): FK to dd.id
- line_number (INTEGER): Position in document
- data (JSONB): **THE LINE ITEM DATA** - NOT "line_items"!

**CRITICAL FIX FOR "column line_items does not exist"**:
- WRONG: `li.line_items` - THIS COLUMN DOES NOT EXIST!
- CORRECT: `li.data` - This is the actual JSONB column for line items
- Each row in documents_data_line_items is ONE line item (already normalized)
- Do NOT use jsonb_array_elements() - the data is already split into rows
- Alias as `li.data as item` in the CTE, then use `item->>'field_name'`

Check the 'source' and 'access_pattern' in the field schema!
- source='header': Access via `header_data->>'field_name'`
- source='summary': Access via `summary_data->>'field_name'`
- source='line_item': Access via `item->>'field_name'` after aliasing `li.data as item`

## Common Error Fixes:
1. **"column line_items does not exist"** → Use `li.data` not `li.line_items`! The column is named `data`.
2. **"column item does not exist" in CTE** → The 'item' alias is only available AFTER the CTE defines it. Move item->>'...' filters to the outer SELECT's WHERE clause.
3. "column must appear in GROUP BY": Add the column to GROUP BY clause.
3. **"invalid input syntax for type numeric"**:
   - CRITICAL: Check the data_type in the schema! The field being cast might be a STRING, not a number.
   - If error shows a value like "P000035", "SKU-001", etc., the field is a STRING - DO NOT cast to ::numeric
   - **BILINGUAL FIELD NAMES**: Field names containing "/" (e.g., "Item subtotal /Sous-total del'article") are LABELS, not data fields - DO NOT cast them to ::numeric!
   - For STRING fields: Use item->>'FieldName' without any cast
   - For NUMERIC fields only: Use (item->>'FieldName')::numeric with NULLIF for safety
   - Check the schema's data_type field: "string" = no cast, "number" = can cast to numeric
   - Common STRING fields that should NOT be cast: ProductID, SKU, Code, ID, Number (identifiers)
4. **Date/Time parsing errors** (CRITICAL - common issue):
   - **"invalid input syntax for type date"** or **"date/time field value out of range"**:
   - Date fields may contain INCONSISTENT formats: full datetime, date-only, time-only, or invalid values
   - **NEVER directly cast to ::date or ::timestamp** - always use safe extraction!
   - **SOLUTION - Use SUBSTRING to extract year for filtering:**
     ```sql
     -- Safe year extraction (works with any date format containing 4-digit year):
     WHERE SUBSTRING(header_data->>'transaction_date' FROM '[0-9]{{4}}') = '2025'
     -- Or for year comparison:
     WHERE COALESCE(SUBSTRING(header_data->>'transaction_date' FROM '^[0-9]{{4}}'), '0000') >= '2025'
     ```
   - **For ordering by date (handle mixed formats):**
     ```sql
     ORDER BY header_data->>'transaction_date' DESC  -- String ordering is usually sufficient
     ```
   - **DO NOT use**: (field)::date, TO_DATE(), TO_TIMESTAMP() directly on untrusted data
   - **Time-only values** like "13:23" will cause errors if cast to date - skip them with:
     ```sql
     WHERE header_data->>'transaction_date' ~ '^[0-9]{{4}}-'  -- Only process if starts with year
     ```
5. Wrong field name: Check the schema for exact field names (e.g., 'Total Sales' not 'amount', 'Product' not 'description').
6. **Case-sensitive text search returning no results**: If using LIKE for text matching, change to ILIKE for case-insensitive matching:
   - WRONG: WHERE (item->>'description') LIKE '%keyboard%'  -- Misses "Keyboard"
   - CORRECT: WHERE (item->>'description') ILIKE '%keyboard%'  -- Finds "Keyboard", "keyboard", etc.

## Requirements:
1. Analyze the error message carefully
2. Identify the root cause
3. Generate a corrected SQL query using EXACT field names from the schema
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
    VERIFIED_SQL_PROMPT = """Generate a PostgreSQL query based on the schema and parameters provided.

## Original User Query:
"{user_query}"

## Available Data Fields (use ONLY these fields):
{field_schema}

## Query Parameters:
- Report Type: {report_type}
- Aggregation Type: {aggregation_type}
- Aggregation Field: {aggregation_field}
- Entity Field: {entity_field}
- Grouping Fields: {grouping_fields}
- Filters: {filters}
- Date Field: {date_field} (source: {date_field_source})
- Time Grouping: {time_grouping}
- Time Filter Only: {time_filter_only}
- Time Filter Year: {time_filter_year}
- Time Filter Period: {time_filter_period}
- Count Level: {count_level} (for count queries: "document" = count receipts/invoices, "line_item" = count items)

## CRITICAL RULES:

### 1. Schema Adherence (MOST IMPORTANT):
- Use ONLY fields listed in "Available Data Fields" above
- **CRITICAL: Always use the "db_field" value (NOT "field_name") in SQL queries!**
  - The schema shows both: field_name (user's terminology) and db_field (actual database field)
  - User may say "StockQuantity" but the db_field is "quantity" - use "quantity" in SQL!
  - Example: If schema shows {{"field_name": "StockQuantity", "db_field": "quantity"}}, use: item->>'quantity'
- Data is normalized to canonical field names (description, quantity, amount, etc.)
- Use the exact "access_pattern" shown in the schema for each field
- For report_type="detail": Include ALL fields from the schema in SELECT
- For min/max queries (finding record with min or max value): Include ALL fields from the schema in SELECT
- Never invent or assume fields not in the schema

### 1.5 COUNT QUERIES - DOCUMENT vs LINE ITEM (READ THIS IF aggregation_type="count"):
**If count_level="document"**: You are counting DOCUMENTS (receipts/meals/invoices), NOT line items!
```sql
WITH expanded_items AS (
    SELECT dd.document_id, dd.header_data, dd.summary_data, li.data as item  -- MUST include document_id and summary_data
    FROM documents_data dd
    JOIN documents_data_line_items li ON li.documents_data_id = dd.id
    WHERE dd.document_id IN (...)
)
SELECT COUNT(DISTINCT document_id) AS document_count  -- Use DISTINCT document_id
FROM expanded_items
WHERE (header_data->>'transaction_date')::date BETWEEN ...
```
**If count_level="line_item"**: Count all line items with COUNT(*)

### 2. Data Type Handling (CRITICAL):
- **SELECT columns**: Just use item->>'FieldName' AS alias - NO type casting needed
- **ORDER BY numeric field**: Cast only in ORDER BY clause: ORDER BY (item->>'Field')::numeric
- **WHERE comparisons**: Cast only when comparing numbers or dates
- **Aggregations (SUM, AVG, MIN, MAX on numeric)**: ALWAYS cast to numeric! The ->> operator returns TEXT.
  - CORRECT: SUM((item->>'quantity')::numeric)
  - CORRECT: AVG((item->>'amount')::numeric)
  - WRONG: SUM(item->>'quantity') -- ERROR: function sum(text) does not exist
- **DO NOT cast in SELECT** for non-aggregated fields - return values as-is from JSONB
- **BILINGUAL FIELD NAMES**: Field names containing "/" (e.g., "Item subtotal /Sous-total") are LABELS, NOT numeric data - NEVER cast them to ::numeric!

### 3. Filter Application:
- Apply filters in OUTER WHERE clause (after CTE), not inside CTE
- Operators: <, >, =, <=, >=, !=, between, in, not_in, ilike, compound
- **CRITICAL FOR TEXT SEARCH**: Always use ILIKE for case-insensitive text matching (NOT LIKE)
  - Example: WHERE (item->>'description') ILIKE '%keyboard%'  -- Matches "Keyboard", "keyboard", "KEYBOARD"

**CRITICAL - Compound vs BETWEEN:**
- "lower than X but higher than Y" = TWO conditions: field > Y AND field < X (EXCLUSIVE bounds)
- "between X and Y" = field >= X AND field <= Y (INCLUSIVE bounds)
- Example: "inventory lower than 50 but higher than 30" means: (item->>'stockquantity')::numeric > 30 AND (item->>'stockquantity')::numeric < 50
- This is NOT the same as BETWEEN 30 AND 50!

### 4. SQL Structure:
```sql
WITH expanded_items AS (
    SELECT dd.document_id, dd.header_data, dd.summary_data, li.data as item  -- ALWAYS include document_id and summary_data
    FROM documents_data dd
    JOIN documents_data_line_items li ON li.documents_data_id = dd.id
    JOIN documents d ON dd.document_id = d.id
)
SELECT
    item->>'FieldName' AS FieldName,  -- NO casting in SELECT
    item->>'AnotherField' AS AnotherField
FROM expanded_items
WHERE -- filters here (cast only for comparisons)
ORDER BY (item->>'NumericField')::numeric DESC  -- cast only in ORDER BY if needed
```
NOTE: Always include `dd.document_id` and `dd.summary_data` in CTE - needed for count and summary fields.

### 5. Report Type Behavior:
- **detail**: SELECT all schema fields, return individual rows. ALWAYS include document identifiers (invoice_number, receipt_number) so rows can be grouped by document in the UI.
- **summary**: GROUP BY and aggregate (SUM, COUNT, AVG, MIN, MAX)
- **count**: Return COUNT with optional grouping

**CRITICAL for detail reports**: When listing line items from multiple documents, ALWAYS include the document identifier field (invoice_number or receipt_number) so users can tell which items belong to which document.

### 5.5 FIELD ORDERING IN SELECT (CRITICAL for readability):
Order fields in a LOGICAL HIERARCHY - document header fields first, then summary, then line item details:

**NOTE: Data is normalized to canonical field names. Use these standard names:**
- Line items: `description`, `quantity`, `unit_price`, `amount`, `sku`, `category`
- Headers: `store_name`, `transaction_date`, `receipt_number`, `invoice_number`, `vendor_name`, etc.
- Summary: `subtotal`, `tax_amount`, `tip_amount`, `total_amount`, `discount_amount`, `shipping_amount`

**Field Order Priority:**
1. **HEADER/DOCUMENT FIELDS** (from header_data) - GROUP THESE TOGETHER FIRST:
   - Business identity: store_name, vendor_name, customer_name
   - Document identifier: receipt_number, invoice_number
   - Temporal: transaction_date, invoice_date
   - Context: payment_method, currency

2. **SUMMARY FIELDS** (from summary_data) - DOCUMENT TOTALS:
   - subtotal (before tax/tip)
   - tax_amount (tax)
   - tip_amount (gratuity)
   - total_amount (grand total)

3. **LINE ITEM FIELDS** (from item) - AFTER header/summary fields:
   - description (WHAT was purchased - FIRST)
   - quantity (HOW MANY)
   - unit_price (PRICE PER UNIT)
   - amount (LINE TOTAL - LAST)

**Example - Correct field order for detail report:**
```sql
SELECT
    -- HEADER FIELDS (document-level)
    header_data->>'store_name' AS store_name,
    header_data->>'transaction_date' AS transaction_date,
    header_data->>'receipt_number' AS receipt_number,
    -- SUMMARY FIELDS (document totals)
    summary_data->>'subtotal' AS subtotal,
    summary_data->>'tax_amount' AS tax_amount,
    summary_data->>'total_amount' AS total_amount,
    -- LINE ITEM FIELDS (canonical names)
    item->>'description' AS description,      -- WHAT (first)
    item->>'quantity' AS quantity,            -- HOW MANY
    item->>'unit_price' AS unit_price,        -- PRICE EACH
    item->>'amount' AS amount                 -- LINE TOTAL (last)
FROM expanded_items
ORDER BY transaction_date, description
```

**CRITICAL RULE**: description field MUST come BEFORE amount/price fields!

### 6. Aggregation Patterns (ALWAYS cast to numeric for SUM/AVG):
- **sum**: GROUP BY grouping fields, SUM((item->>'field')::numeric)
- **avg**: Use two-step aggregation for per-document averages, AVG((item->>'field')::numeric)
- **count**: Two types based on count_level parameter - THIS IS CRITICAL:
  - **count_level="document"**: Count DISTINCT documents (receipts/invoices), NOT line items
    IMPORTANT: Include document_id and summary_data in the CTE!
    ```sql
    WITH expanded_items AS (
        SELECT dd.document_id, dd.header_data, dd.summary_data, li.data as item  -- MUST include document_id and summary_data
        FROM documents_data dd
        JOIN documents_data_line_items li ON li.documents_data_id = dd.id
        WHERE ...
    )
    SELECT COUNT(DISTINCT document_id) as document_count  -- Count DISTINCT documents
    FROM expanded_items
    WHERE ...
    ```
  - **count_level="line_item"**: Count individual line items
    ```sql
    SELECT COUNT(*) as item_count FROM expanded_items WHERE ...
    ```

  **CRITICAL**: When count_level="document", you MUST:
  1. Add `dd.document_id` to the CTE SELECT
  2. Use `COUNT(DISTINCT document_id)` in the outer SELECT
  3. Do NOT use `COUNT(*)` - that counts line items!
- **min/max**: Use subquery with MIN()/MAX() function to find the record. This properly handles NULL values.
  Example for MIN (finding record with minimum value):
  ```sql
  SELECT ... FROM expanded_items
  WHERE (item->>'AggField')::numeric = (
      SELECT MIN((item->>'AggField')::numeric) FROM expanded_items
  )
  ```
  Example for MAX (finding record with maximum value):
  ```sql
  SELECT ... FROM expanded_items
  WHERE (item->>'AggField')::numeric = (
      SELECT MAX((item->>'AggField')::numeric) FROM expanded_items
  )
  ```
  **IMPORTANT: Include ALL schema fields in SELECT, not just the aggregation field. The user wants to see the complete record.**
  **DO NOT use ORDER BY ... LIMIT 1 for min/max - it fails with NULL values.**

### 7. Important Notes:
- Do NOT add WHERE inside base CTE - document filtering is added automatically
- Always include header_data in CTE SELECT for access in outer queries
- For time groupings use: EXTRACT(YEAR/QUARTER FROM ...) or TO_CHAR(..., 'YYYY-MM')

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

        Supports both inline and external line item storage.

        Args:
            document_ids: Document IDs to analyze
            db: Database session

        Returns:
            Inferred field mappings
        """
        from sqlalchemy import text

        try:
            doc_ids_str = ", ".join(f"'{doc_id}'" for doc_id in document_ids)

            # First try inline storage: Get sample data from line_items array
            result = db.execute(text(f"""
                SELECT
                    header_data,
                    line_items->0 as sample_item
                FROM documents_data
                WHERE document_id IN ({doc_ids_str})
                AND line_items IS NOT NULL
                AND jsonb_array_length(line_items) > 0
                LIMIT 1
            """))

            row = result.fetchone()

            header_data = {}
            sample_item = {}

            if row:
                header_data = row[0] or {}
                sample_item = row[1] or {}

            # If no inline line_items found, check external storage
            if not sample_item:
                logger.info("[SQL Generator] No inline line_items, checking external storage")
                result = db.execute(text(f"""
                    SELECT
                        dd.header_data,
                        li.data as sample_item
                    FROM documents_data dd
                    JOIN documents_data_line_items li ON li.documents_data_id = dd.id
                    WHERE dd.document_id IN ({doc_ids_str})
                    LIMIT 1
                """))
                row = result.fetchone()
                if row:
                    header_data = row[0] or {}
                    sample_item = row[1] or {}
                    logger.info(f"[SQL Generator] Found external line_items with fields: {list(sample_item.keys())}")

            if not header_data and not sample_item:
                logger.warning("[SQL Generator] No data found for field inference")
                return {}

            field_mappings = {}

            # Process header fields
            for field in header_data.keys():
                if field not in ['field_mappings', 'column_headers']:
                    field_mappings[field] = self._classify_field_for_sql(field, 'header')

            # Process line item fields
            for field in sample_item.keys():
                if field != 'row_number':
                    field_mappings[field] = self._classify_field_for_sql(field, 'line_item')

            logger.info(f"[SQL Generator] Inferred {len(field_mappings)} field mappings: {list(field_mappings.keys())}")
            return field_mappings

        except Exception as e:
            logger.error(f"[SQL Generator] Field inference failed: {e}")
            return {}

    def _classify_field_for_sql(self, field_name: str, source: str) -> Dict[str, Any]:
        """
        Classify a field for SQL generation purposes.

        Recognizes canonical field names from the standardized schema and properly
        classifies them. Also handles bilingual/label detection to prevent
        misclassification of fields like "Item subtotal /Sous-total del'article".

        Args:
            field_name: Field name
            source: 'header', 'line_item', or 'summary'

        Returns:
            Field classification with semantic_type, data_type, source, aggregation
        """
        field_lower = field_name.lower().replace('_', ' ')

        # Check for bilingual/descriptive field names that should NOT be classified as numeric
        # These are labels or column headers, not actual data fields
        # Examples: "Item subtotal /Sous-total del'article", "Description / Description"
        is_bilingual_label = (
            '/' in field_name or  # Contains language separator (e.g., "English / French")
            len(field_name) > 50 or  # Very long field name (likely a description/label)
            field_name.count(' ') > 4  # Too many words to be a field name
        )

        # If it's a bilingual label, treat as string/description regardless of keywords
        if is_bilingual_label:
            logger.debug(f"[SQL Generator] Field '{field_name}' detected as bilingual/label - treating as string")
            return {
                'semantic_type': 'description',
                'data_type': 'string',
                'source': source,
                'aggregation': None
            }

        # CANONICAL FIELD RECOGNITION
        # These are standardized field names from the grouped field mappings
        canonical_fields = {
            # Header canonical fields
            'vendor_name': {'semantic_type': 'entity', 'data_type': 'string', 'aggregation': 'group_by'},
            'customer_name': {'semantic_type': 'entity', 'data_type': 'string', 'aggregation': 'group_by'},
            'store_name': {'semantic_type': 'entity', 'data_type': 'string', 'aggregation': 'group_by'},
            'invoice_number': {'semantic_type': 'identifier', 'data_type': 'string', 'aggregation': None},
            'receipt_number': {'semantic_type': 'identifier', 'data_type': 'string', 'aggregation': None},
            'invoice_date': {'semantic_type': 'date', 'data_type': 'datetime', 'aggregation': None},
            'transaction_date': {'semantic_type': 'date', 'data_type': 'datetime', 'aggregation': None},
            'due_date': {'semantic_type': 'date', 'data_type': 'datetime', 'aggregation': None},
            'payment_method': {'semantic_type': 'method', 'data_type': 'string', 'aggregation': 'group_by'},
            'currency': {'semantic_type': 'identifier', 'data_type': 'string', 'aggregation': None},

            # Line item canonical fields
            'description': {'semantic_type': 'product', 'data_type': 'string', 'aggregation': 'group_by'},
            'quantity': {'semantic_type': 'quantity', 'data_type': 'number', 'aggregation': 'sum'},
            'unit_price': {'semantic_type': 'amount', 'data_type': 'number', 'aggregation': None},
            'amount': {'semantic_type': 'amount', 'data_type': 'number', 'aggregation': 'sum'},
            'sku': {'semantic_type': 'identifier', 'data_type': 'string', 'aggregation': None},
            'category': {'semantic_type': 'category', 'data_type': 'string', 'aggregation': 'group_by'},

            # Summary canonical fields
            'subtotal': {'semantic_type': 'subtotal', 'data_type': 'number', 'aggregation': None},
            'tax_amount': {'semantic_type': 'tax', 'data_type': 'number', 'aggregation': None},
            'tip_amount': {'semantic_type': 'tip', 'data_type': 'number', 'aggregation': None},
            'discount_amount': {'semantic_type': 'discount', 'data_type': 'number', 'aggregation': None},
            'shipping_amount': {'semantic_type': 'amount', 'data_type': 'number', 'aggregation': None},
            'total_amount': {'semantic_type': 'total', 'data_type': 'number', 'aggregation': None},
        }

        # Check if this is a canonical field name (exact match)
        if field_name in canonical_fields:
            field_info = canonical_fields[field_name]
            return {
                'semantic_type': field_info['semantic_type'],
                'data_type': field_info['data_type'],
                'source': source,
                'aggregation': field_info['aggregation']
            }

        # Fallback to pattern-based classification for non-canonical fields
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
        table_filter: Optional[str] = None,
        storage_type: str = "external"
    ) -> SQLGenerationResult:
        """
        Generate SQL query based on user query and field mappings.

        Uses multi-round LLM analysis when LLM client is available:
        - Round 1: Query analysis to understand intent and grouping order
        - Round 2: Field mapping to match user terms to actual schema fields
        - Round 3: SQL generation with verified parameters

        NOTE: As of migration 022, storage_type defaults to "external".
        All line items are now stored in documents_data_line_items table.
        The "inline" storage (jsonb_array_elements) is deprecated.

        Args:
            user_query: User's natural language query
            field_mappings: Dynamic field mappings from extracted data
            table_filter: Optional WHERE clause filter for specific documents
            storage_type: "external" (default) - inline is deprecated

        Returns:
            SQLGenerationResult with generated SQL and metadata
        """
        logger.info("=" * 80)
        logger.info("[SQL Generator] ========== SQL GENERATION START ==========")
        logger.info("=" * 80)
        logger.info(f"[SQL Generator] User Query: {user_query}")
        logger.info(f"[SQL Generator] Field Mappings: {len(field_mappings)} fields")
        logger.info(f"[SQL Generator] Table Filter: {table_filter or 'None'}")
        logger.info(f"[SQL Generator] Storage Type: {storage_type}")
        logger.info("-" * 80)

        if not self.llm_client:
            logger.warning("[SQL Generator] No LLM client available")
            logger.info("[SQL Generator] Using HEURISTIC SQL generation (fallback)...")
            report_type = self._detect_report_type_from_query(user_query)
            return self._generate_sql_heuristic(user_query, field_mappings, table_filter, storage_type, report_type)

        try:
            logger.info("[SQL Generator] LLM client available - using MULTI-ROUND analysis")
            # Use multi-round LLM analysis for better accuracy
            return self._generate_sql_multi_round(user_query, field_mappings, table_filter, storage_type)

        except Exception as e:
            logger.error(f"[SQL Generator] Multi-round LLM SQL generation failed: {e}")
            logger.info("[SQL Generator] Falling back to HEURISTIC generation...")
            # Fallback to heuristic
            report_type = self._detect_report_type_from_query(user_query)
            return self._generate_sql_heuristic(user_query, field_mappings, table_filter, storage_type, report_type)

    def generate_min_max_separate_queries(
        self,
        user_query: str,
        field_mappings: Dict[str, Dict[str, Any]],
        table_filter: Optional[str] = None,
        storage_type: str = "external"
    ) -> Tuple[Optional[SQLGenerationResult], Optional[SQLGenerationResult]]:
        """
        Generate two separate SQL queries for MIN and MAX aggregations.

        This is used when the user asks for both min and max values (e.g., "which products
        have max inventory and min inventory"). Instead of generating one complex UNION ALL
        query, we generate two simple queries and let the executor run them separately.

        NOTE: As of migration 022, storage_type defaults to "external".

        Args:
            user_query: User's natural language query
            field_mappings: Dynamic field mappings from extracted data
            table_filter: Optional WHERE clause filter for specific documents
            storage_type: "external" (default) - inline is deprecated

        Returns:
            Tuple of (min_query_result, max_query_result)
        """
        logger.info("=" * 80)
        logger.info("[SQL Generator] ========== GENERATING SEPARATE MIN/MAX QUERIES ==========")
        logger.info("=" * 80)

        # Generate MIN query by modifying the user query
        min_query = self._modify_query_for_aggregation(user_query, "min")
        max_query = self._modify_query_for_aggregation(user_query, "max")

        logger.info(f"[SQL Generator] MIN query: {min_query}")
        logger.info(f"[SQL Generator] MAX query: {max_query}")

        # Generate both queries
        min_result = None
        max_result = None

        try:
            logger.info("[SQL Generator] Generating MIN query...")
            min_result = self.generate_sql(min_query, field_mappings, table_filter, storage_type)
        except Exception as e:
            logger.error(f"[SQL Generator] MIN query generation failed: {e}")

        try:
            logger.info("[SQL Generator] Generating MAX query...")
            max_result = self.generate_sql(max_query, field_mappings, table_filter, storage_type)
        except Exception as e:
            logger.error(f"[SQL Generator] MAX query generation failed: {e}")

        logger.info("[SQL Generator] Separate MIN/MAX queries generation complete")
        return (min_result, max_result)

    def _modify_query_for_aggregation(self, user_query: str, aggregation_type: str) -> str:
        """
        Modify a user query to ask for only MIN or MAX.

        Transforms queries like "which products have max and min inventory" into
        explicit queries that ask for the single product/item with the lowest/highest value.

        Args:
            user_query: Original user query with both min and max
            aggregation_type: "min" or "max"

        Returns:
            Modified query for single aggregation
        """
        query_lower = user_query.lower()

        if aggregation_type == "min":
            # Replace max/min combination with explicit min request
            modified = query_lower
            # Remove "max" references and make explicit
            modified = re.sub(r'\bmax(?:imum)?\s+(?:and|&)\s+min(?:imum)?\b', 'the single lowest', modified, flags=re.IGNORECASE)
            modified = re.sub(r'\bmin(?:imum)?\s+(?:and|&)\s+max(?:imum)?\b', 'the single lowest', modified, flags=re.IGNORECASE)
            modified = re.sub(r'\bhighest\s+(?:and|&)\s+lowest\b', 'the single lowest', modified, flags=re.IGNORECASE)
            modified = re.sub(r'\blowest\s+(?:and|&)\s+highest\b', 'the single lowest', modified, flags=re.IGNORECASE)
            # Handle "max xxx and min xxx" patterns - make more explicit
            modified = re.sub(r'\bmax(?:imum)?\s+(\w+)\s+(?:and|&)\s+min(?:imum)?\s+\1\b', r'the single lowest \1', modified, flags=re.IGNORECASE)
            modified = re.sub(r'\bmin(?:imum)?\s+(\w+)\s+(?:and|&)\s+max(?:imum)?\s+\1\b', r'the single lowest \1', modified, flags=re.IGNORECASE)
            # Add clarification suffix with explicit instruction to use MIN() subquery
            if 'single' not in modified:
                modified = modified + ' (return the record with the MINIMUM value. Use WHERE field = (SELECT MIN(field) FROM ...) subquery pattern. IMPORTANT: Include ALL schema fields in SELECT to show complete record details. DO NOT use ORDER BY LIMIT 1.)'
            return modified
        else:  # max
            modified = query_lower
            # Remove "min" references and make explicit
            modified = re.sub(r'\bmax(?:imum)?\s+(?:and|&)\s+min(?:imum)?\b', 'the single highest', modified, flags=re.IGNORECASE)
            modified = re.sub(r'\bmin(?:imum)?\s+(?:and|&)\s+max(?:imum)?\b', 'the single highest', modified, flags=re.IGNORECASE)
            modified = re.sub(r'\bhighest\s+(?:and|&)\s+lowest\b', 'the single highest', modified, flags=re.IGNORECASE)
            modified = re.sub(r'\blowest\s+(?:and|&)\s+highest\b', 'the single highest', modified, flags=re.IGNORECASE)
            # Handle "max xxx and min xxx" patterns - make more explicit
            modified = re.sub(r'\bmax(?:imum)?\s+(\w+)\s+(?:and|&)\s+min(?:imum)?\s+\1\b', r'the single highest \1', modified, flags=re.IGNORECASE)
            modified = re.sub(r'\bmin(?:imum)?\s+(\w+)\s+(?:and|&)\s+max(?:imum)?\s+\1\b', r'the single highest \1', modified, flags=re.IGNORECASE)
            # Add clarification suffix with explicit instruction to use MAX() subquery
            if 'single' not in modified:
                modified = modified + ' (return the record with the MAXIMUM value. Use WHERE field = (SELECT MAX(field) FROM ...) subquery pattern. IMPORTANT: Include ALL schema fields in SELECT to show complete record details. DO NOT use ORDER BY LIMIT 1.)'
            return modified

    def is_min_max_query(self, user_query: str) -> bool:
        """
        Detect if a user query is asking for both MIN and MAX values.

        Args:
            user_query: User's natural language query

        Returns:
            True if the query asks for both min and max
        """
        query_lower = user_query.lower()

        # Patterns that indicate both min and max are requested
        patterns = [
            r'\bmax(?:imum)?\s+(?:and|&)\s+min(?:imum)?\b',
            r'\bmin(?:imum)?\s+(?:and|&)\s+max(?:imum)?\b',
            r'\bhighest\s+(?:and|&)\s+lowest\b',
            r'\blowest\s+(?:and|&)\s+highest\b',
            r'\bmax(?:imum)?\s+\w+\s+(?:and|&)\s+min(?:imum)?\s+\w+\b',
            r'\bmin(?:imum)?\s+\w+\s+(?:and|&)\s+max(?:imum)?\s+\w+\b',
        ]

        for pattern in patterns:
            if re.search(pattern, query_lower):
                logger.info(f"[SQL Generator] Detected MIN/MAX query pattern: {pattern}")
                return True

        return False

    def _generate_sql_multi_round(
        self,
        user_query: str,
        field_mappings: Dict[str, Dict[str, Any]],
        table_filter: Optional[str] = None,
        storage_type: str = "external"
    ) -> SQLGenerationResult:
        """
        Generate SQL using multi-round LLM analysis.

        Round 1: Analyze query to understand grouping order and intent
        Round 2: Map user terms to actual schema field names
        Round 3: Generate SQL with verified parameters

        NOTE: As of migration 022, always uses external storage (documents_data_line_items).
        """
        field_schema = self._format_field_schema(field_mappings)
        field_schema_json = self._format_field_schema_json(field_mappings)

        # ========== Round 1: Query Analysis ==========
        logger.info("=" * 80)
        logger.info("[SQL Generator] ========== ROUND 1: QUERY ANALYSIS ==========")
        logger.info("=" * 80)
        logger.info("[SQL Generator] Analyzing user intent and grouping requirements...")

        analysis_prompt = self.QUERY_ANALYSIS_PROMPT.format(
            user_query=user_query,
            field_schema=field_schema_json
        )
        logger.debug(f"[SQL Generator] Analysis prompt length: {len(analysis_prompt)} chars")

        analysis_response = self.llm_client.generate(analysis_prompt)
        query_analysis = self._parse_json_response(analysis_response)

        if not query_analysis:
            logger.warning("[SQL Generator] Round 1 FAILED - could not parse LLM response")
            logger.info("[SQL Generator] Falling back to heuristic generation...")
            report_type = self._detect_report_type_from_query(user_query)
            return self._generate_sql_heuristic(user_query, field_mappings, table_filter, storage_type, report_type)

        # Extract aggregation type, entity field, and report type from Round 1
        aggregation_type = query_analysis.get('aggregation_type', 'sum')
        entity_field = query_analysis.get('entity_field', '')
        report_type = query_analysis.get('report_type', 'summary')

        # Extract time filter vs grouping parameters (NEW)
        time_filter_only = query_analysis.get('time_filter_only', False)
        time_filter_year = query_analysis.get('time_filter_year')
        time_filter_period = query_analysis.get('time_filter_period')

        # Extract count_level for count queries (MUST be logged here to debug)
        count_level_r1 = query_analysis.get('count_level', 'line_item')

        logger.info("-" * 80)
        logger.info("[SQL Generator] ROUND 1 RESULT:")
        logger.info(f"[SQL Generator]   - Grouping Order: {query_analysis.get('grouping_order')}")
        logger.info(f"[SQL Generator]   - Time Granularity: {query_analysis.get('time_granularity')}")
        logger.info(f"[SQL Generator]   - Aggregation Field: {query_analysis.get('aggregation_field')}")
        logger.info(f"[SQL Generator]   - Aggregation Type: {aggregation_type}")
        logger.info(f"[SQL Generator]   - Entity Field: {entity_field or 'None'}")
        logger.info(f"[SQL Generator]   - Report Type: {report_type}")
        logger.info(f"[SQL Generator]   - Time Filter Only: {time_filter_only}")
        logger.info(f"[SQL Generator]   - Time Filter Year: {time_filter_year}")
        if aggregation_type == 'count':
            logger.info(f"[SQL Generator]   - Count Level: {count_level_r1} (document=count receipts, line_item=count items)")
        logger.info(f"[SQL Generator]   - Explanation: {query_analysis.get('explanation', 'N/A')}")
        logger.info("-" * 80)

        # ========== Round 2: Field Mapping ==========
        logger.info("=" * 80)
        logger.info("[SQL Generator] ========== ROUND 2: FIELD MAPPING ==========")
        logger.info("=" * 80)
        logger.info("[SQL Generator] Mapping user terms to actual schema fields...")

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
            logger.warning("[SQL Generator] Round 2 FAILED - field mapping unsuccessful")
            logger.info("[SQL Generator] Falling back to heuristic generation...")
            return self._generate_sql_heuristic(user_query, field_mappings, table_filter, storage_type, report_type)

        logger.info("-" * 80)
        logger.info("[SQL Generator] ROUND 2 RESULT:")
        logger.info(f"[SQL Generator]   - Grouping Fields: {field_mapping_result.get('grouping_fields')}")
        logger.info(f"[SQL Generator]   - Aggregation Field: {field_mapping_result.get('aggregation_field')}")
        logger.info(f"[SQL Generator]   - Date Field: {field_mapping_result.get('date_field')}")
        logger.info(f"[SQL Generator]   - Entity Field: {field_mapping_result.get('entity_field')}")
        if field_mapping_result.get('unmapped_fields'):
            logger.warning(f"[SQL Generator]   - Unmapped Fields: {field_mapping_result.get('unmapped_fields')}")
        logger.info("-" * 80)

        # ========== Round 3: SQL Generation ==========
        logger.info("=" * 80)
        logger.info("[SQL Generator] ========== ROUND 3: SQL GENERATION ==========")
        logger.info("=" * 80)
        logger.info("[SQL Generator] Generating SQL with verified parameters...")

        # Prepare grouping fields description
        # Normalize grouping_fields to ensure each element is a dict
        raw_grouping_fields = field_mapping_result.get('grouping_fields', [])
        grouping_fields = []
        for gf in raw_grouping_fields:
            if isinstance(gf, dict):
                grouping_fields.append(gf)
            elif isinstance(gf, str):
                # Convert string to dict format
                grouping_fields.append({'actual_field': gf, 'requested': gf, 'is_time_grouping': False})
            elif isinstance(gf, list):
                # Handle nested list - flatten and convert each element
                for item in gf:
                    if isinstance(item, dict):
                        grouping_fields.append(item)
                    elif isinstance(item, str):
                        grouping_fields.append({'actual_field': item, 'requested': item, 'is_time_grouping': False})
            # Skip other types

        grouping_fields_desc = []
        for gf in grouping_fields:
            if gf.get('is_time_grouping'):
                grouping_fields_desc.append(f"{gf.get('granularity', 'yearly')} (from {gf.get('actual_field', 'date')})")
            else:
                grouping_fields_desc.append(gf.get('actual_field', gf.get('requested', '')))

        # Determine time grouping info
        time_grouping_info = "None"
        for gf in grouping_fields:
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

        # Check if grouped or legacy format and find date field source
        is_grouped_format = any(
            field_mappings.get(k) for k in ['header_mappings', 'line_item_mappings', 'summary_mappings']
        )
        if is_grouped_format:
            # Search for date_field in grouped format
            for mapping_key, source in [('header_mappings', 'header'), ('line_item_mappings', 'line_item'), ('summary_mappings', 'summary')]:
                mapping_list = field_mappings.get(mapping_key, [])
                if isinstance(mapping_list, list):
                    for item in mapping_list:
                        if isinstance(item, dict):
                            canonical = item.get('canonical', item.get('field_name', ''))
                            if canonical == date_field:
                                date_field_source = source
                                break
        elif date_field in field_mappings:
            # Legacy flat format
            mapping = field_mappings[date_field]
            if isinstance(mapping, dict):
                date_field_source = mapping.get('source', 'header')

        # Safely extract aggregation field (handle None values)
        agg_field_mapping = field_mapping_result.get('aggregation_field') or {}
        aggregation_field_actual = agg_field_mapping.get('actual_field', 'amount') if isinstance(agg_field_mapping, dict) else 'amount'

        logger.info(f"[SQL Generator] SQL Generation Parameters:")
        logger.info(f"[SQL Generator]   - Grouping Fields: {grouping_fields_desc}")
        logger.info(f"[SQL Generator]   - Time Grouping: {time_grouping_info}")
        logger.info(f"[SQL Generator]   - Time Filter Only: {time_filter_only}")
        logger.info(f"[SQL Generator]   - Aggregation Field: {aggregation_field_actual}")
        logger.info(f"[SQL Generator]   - Aggregation Type: {final_aggregation_type}")
        logger.info(f"[SQL Generator]   - Entity Field: {entity_field_actual or 'description'}")
        logger.info(f"[SQL Generator]   - Date Field: {date_field} (source: {date_field_source})")
        logger.info(f"[SQL Generator]   - Report Type: {report_type}")

        # Get count_level for count queries (document vs line_item counting)
        count_level = query_analysis.get('count_level', 'line_item')
        if final_aggregation_type == 'count':
            logger.info(f"[SQL Generator]   - Count Level: {count_level}")

        sql_prompt = self.VERIFIED_SQL_PROMPT.format(
            user_query=user_query,
            field_schema=field_schema_json,
            grouping_fields=json.dumps(grouping_fields_desc),
            time_grouping=time_grouping_info,
            time_filter_only=time_filter_only,
            time_filter_year=time_filter_year or 'null',
            time_filter_period=time_filter_period or 'null',
            aggregation_field=aggregation_field_actual,
            aggregation_type=final_aggregation_type,
            entity_field=entity_field_actual or 'description',
            date_field=date_field,
            date_field_source=date_field_source,
            report_type=report_type,
            filters=json.dumps(query_analysis.get('filters', {})),
            count_level=count_level
        )

        logger.info("[SQL Generator] Calling LLM to generate SQL...")
        sql_response = self.llm_client.generate(sql_prompt)
        sql_result = self._parse_json_response(sql_response)

        if not sql_result or not sql_result.get('sql'):
            logger.warning("[SQL Generator] Round 3 FAILED - could not generate valid SQL")
            logger.info("[SQL Generator] Falling back to heuristic generation...")
            return self._generate_sql_heuristic(user_query, field_mappings, table_filter, storage_type, report_type)

        # Build result
        sql_query = sql_result.get('sql', '')

        # Note: We no longer try to fix UNION syntax. For min_max queries,
        # the executor will run separate queries and combine results.

        # Convert to external storage SQL if needed
        if storage_type == 'external':
            logger.info("[SQL Generator] Converting SQL for external storage...")
            sql_query = self._convert_to_external_storage_sql(sql_query)

        # Fix document-level counts if LLM used COUNT(*) instead of COUNT(DISTINCT document_id)
        if count_level == 'document' and final_aggregation_type == 'count':
            sql_query = self._fix_document_level_count(sql_query)

        # Ensure summary_data fields (subtotal, total_amount) are included for detail reports
        if report_type == 'detail':
            sql_query = self._ensure_summary_data_fields(sql_query)

        # Add table filter if provided
        if table_filter:
            logger.info(f"[SQL Generator] Adding table filter: {table_filter[:50]}...")
            sql_query = self._add_table_filter(sql_query, table_filter)

        # Extract grouping field names for result
        grouping_field_names = [
            gf.get('actual_field', gf.get('requested', ''))
            for gf in field_mapping_result.get('grouping_fields', [])
            if not gf.get('is_time_grouping')
        ]

        logger.info("-" * 80)
        logger.info("[SQL Generator] ROUND 3 RESULT:")
        logger.info(f"[SQL Generator]   - SQL Generated: YES")
        logger.info(f"[SQL Generator]   - Explanation: {sql_result.get('explanation', 'N/A')[:100]}")
        logger.info("-" * 80)
        logger.info("[SQL Generator] Generated SQL Query:")
        logger.info("-" * 80)
        # Log SQL in chunks for readability
        sql_lines = sql_query.split('\n')
        for line in sql_lines[:20]:  # First 20 lines
            logger.info(f"[SQL]   {line}")
        if len(sql_lines) > 20:
            logger.info(f"[SQL]   ... ({len(sql_lines) - 20} more lines)")
        logger.info("-" * 80)

        logger.info("=" * 80)
        logger.info("[SQL Generator] ========== SQL GENERATION COMPLETE ==========")
        logger.info("=" * 80)

        return SQLGenerationResult(
            sql_query=sql_query,
            explanation=sql_result.get('explanation', query_analysis.get('explanation', '')),
            grouping_fields=grouping_field_names,
            aggregation_fields=[aggregation_field_actual] if aggregation_field_actual else [],
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

        Includes source information (header vs line_item vs summary) which is CRITICAL
        for the LLM to generate correct SQL that accesses fields from the
        right JSONB column.

        Supports two formats:
        1. NEW GROUPED FORMAT (from data_schemas.field_mappings):
           {
               "header_mappings": [{"canonical": "vendor_name", "data_type": "string", ...}],
               "line_item_mappings": [{"canonical": "amount", "data_type": "number", ...}],
               "summary_mappings": [{"canonical": "total_amount", "data_type": "number", ...}]
           }
        2. LEGACY FLAT FORMAT:
           {
               "vendor_name": {"data_type": "string", "source": "header", ...},
               "amount": {"data_type": "number", "source": "line_item", ...}
           }
        """
        # Check if this is the new grouped format
        is_grouped_format = any(
            field_mappings.get(k) for k in ['header_mappings', 'line_item_mappings', 'summary_mappings']
        )

        if is_grouped_format:
            return self._format_grouped_field_schema_json(field_mappings)
        else:
            return self._format_legacy_field_schema_json(field_mappings)

    def _format_grouped_field_schema_json(self, grouped_mappings: Dict[str, List[Dict]]) -> str:
        """Format grouped field mappings (new format) as JSON for LLM prompts.

        Converts grouped format to a flat schema list with proper source annotations.
        Handles various input formats defensively:
        - Dict with 'canonical' key (expected)
        - List items [field_name, field_info]
        - String items (just field name)
        """
        schema_list = []

        def _normalize_mapping(item: Any, source: str) -> Optional[Dict]:
            """Normalize a mapping item to the expected format."""
            if isinstance(item, dict):
                # Standard format with 'canonical' key
                canonical = item.get('canonical', item.get('field_name', item.get('name', '')))
                if canonical:
                    return {
                        'canonical': canonical,
                        'data_type': item.get('data_type', 'string'),
                        'patterns': item.get('patterns', [])
                    }
            elif isinstance(item, list) and len(item) >= 1:
                # List format: [field_name] or [field_name, field_info_dict]
                canonical = item[0] if isinstance(item[0], str) else ''
                field_info = item[1] if len(item) > 1 and isinstance(item[1], dict) else {}
                if canonical:
                    return {
                        'canonical': canonical,
                        'data_type': field_info.get('data_type', 'string') if isinstance(field_info, dict) else 'string',
                        'patterns': field_info.get('patterns', []) if isinstance(field_info, dict) else []
                    }
            elif isinstance(item, str) and item:
                # Simple string format
                return {
                    'canonical': item,
                    'data_type': 'string',
                    'patterns': []
                }
            logger.warning(f"[SQL Generator] Could not normalize mapping item: {type(item)} - {item}")
            return None

        # Process header mappings
        header_list = grouped_mappings.get('header_mappings', [])
        if not isinstance(header_list, list):
            logger.warning(f"[SQL Generator] header_mappings is not a list: {type(header_list)}")
            header_list = []

        for item in header_list:
            mapping = _normalize_mapping(item, 'header')
            if not mapping:
                continue
            canonical = mapping['canonical']
            schema_list.append({
                "field_name": canonical,
                "data_type": mapping.get('data_type', 'string'),
                "semantic_type": self._infer_semantic_type_from_canonical(canonical, 'header'),
                "source": "header",
                "access_pattern": f"header_data->>'{canonical}'",
                "aggregation": "group_by" if mapping.get('data_type') == 'string' else None,
                "patterns": mapping.get('patterns', []),
                "description": f"Header field: {canonical}"
            })

        # Process line item mappings
        line_item_list = grouped_mappings.get('line_item_mappings', [])
        if not isinstance(line_item_list, list):
            logger.warning(f"[SQL Generator] line_item_mappings is not a list: {type(line_item_list)}")
            line_item_list = []

        for item in line_item_list:
            mapping = _normalize_mapping(item, 'line_item')
            if not mapping:
                continue
            canonical = mapping['canonical']
            data_type = mapping.get('data_type', 'string')
            schema_list.append({
                "field_name": canonical,
                "data_type": data_type,
                "semantic_type": self._infer_semantic_type_from_canonical(canonical, 'line_item'),
                "source": "line_item",
                "access_pattern": f"item->>'{canonical}'",
                "aggregation": "sum" if data_type == 'number' and canonical in ['amount', 'quantity', 'unit_price'] else ("group_by" if data_type == 'string' else None),
                "patterns": mapping.get('patterns', []),
                "description": f"Line item field: {canonical}"
            })

        # Process summary mappings
        summary_list = grouped_mappings.get('summary_mappings', [])
        if not isinstance(summary_list, list):
            logger.warning(f"[SQL Generator] summary_mappings is not a list: {type(summary_list)}")
            summary_list = []

        for item in summary_list:
            mapping = _normalize_mapping(item, 'summary')
            if not mapping:
                continue
            canonical = mapping['canonical']
            schema_list.append({
                "field_name": canonical,
                "data_type": mapping.get('data_type', 'number'),
                "semantic_type": self._infer_semantic_type_from_canonical(canonical, 'summary'),
                "source": "summary",
                "access_pattern": f"summary_data->>'{canonical}'",
                "aggregation": None,  # Summary fields are already aggregated
                "patterns": mapping.get('patterns', []),
                "description": f"Summary field: {canonical}"
            })

        logger.info(f"[SQL Generator] Formatted {len(schema_list)} fields from grouped mappings "
                   f"(header={len(header_list)}, "
                   f"line_item={len(line_item_list)}, "
                   f"summary={len(summary_list)})")

        return json.dumps(schema_list, indent=2)

    def _format_legacy_field_schema_json(self, field_mappings: Dict[str, Dict[str, Any]]) -> str:
        """Format legacy flat field mappings as JSON for LLM prompts.

        IMPORTANT: For spreadsheet data, original column names are normalized during
        extraction (e.g., StockQuantity -> quantity). This method handles:
        - db_field: The ACTUAL field name in the database (always use this in SQL!)
        - field_name: The name in the mapping (could be original or db field)
        - original_column: The original column name from the spreadsheet
        - is_alias: If true, this is an alias entry pointing to an actual db_field
        """
        schema_list = []
        seen_db_fields = set()  # Track db_fields to avoid duplicates

        for field_name, mapping in field_mappings.items():
            # Skip grouped format keys if they somehow get here
            if field_name in ['header_mappings', 'line_item_mappings', 'summary_mappings']:
                continue

            # Get the actual database field name (may differ from field_name for aliases)
            db_field = mapping.get('db_field', field_name)

            # Skip duplicate entries for the same db_field (aliases point to same field)
            if mapping.get('is_alias') and db_field in seen_db_fields:
                continue

            seen_db_fields.add(db_field)

            source = mapping.get('source', 'line_item')  # Default to line_item for backwards compatibility

            # Build the schema entry with both names for LLM understanding
            schema_entry = {
                "field_name": field_name,
                "db_field": db_field,  # CRITICAL: LLM must use this in SQL queries!
                "data_type": mapping.get('data_type', 'string'),
                "semantic_type": mapping.get('semantic_type', 'unknown'),
                "source": source,
                "access_pattern": self._get_access_pattern(db_field, source),  # Use db_field for access
                "aggregation": mapping.get('aggregation', 'none'),
                "description": mapping.get('description', ''),
                "aliases": mapping.get('aliases', [])
            }

            # Include original column name if different (helps LLM understand user intent)
            original_column = mapping.get('original_column')
            if original_column and original_column != db_field:
                schema_entry["original_column"] = original_column
                schema_entry["note"] = f"User may refer to this as '{original_column}', but use '{db_field}' in SQL"

            schema_list.append(schema_entry)

        return json.dumps(schema_list, indent=2)

    def _get_access_pattern(self, field_name: str, source: str) -> str:
        """Get the SQL access pattern for a field based on its source."""
        if source == 'header':
            return f"header_data->>'{field_name}'"
        elif source == 'summary':
            return f"summary_data->>'{field_name}'"
        else:
            return f"item->>'{field_name}'"

    def _iterate_field_mappings(self, field_mappings: Dict[str, Any]):
        """
        Iterate over field mappings in a format-agnostic way.

        Supports two formats:
        1. NEW GROUPED FORMAT: {'header_mappings': [...], 'line_item_mappings': [...], ...}
        2. LEGACY FLAT FORMAT: {'field_name': {'data_type': '...', ...}}

        Yields:
            Tuple of (field_name, mapping_dict) where mapping_dict always has 'source' key
        """
        # Check if grouped format
        is_grouped_format = any(
            field_mappings.get(k) for k in ['header_mappings', 'line_item_mappings', 'summary_mappings']
        )

        if is_grouped_format:
            source_map = {
                'header_mappings': 'header',
                'line_item_mappings': 'line_item',
                'summary_mappings': 'summary'
            }
            for mapping_key, source in source_map.items():
                mapping_list = field_mappings.get(mapping_key, [])
                if not isinstance(mapping_list, list):
                    continue
                for item in mapping_list:
                    if isinstance(item, dict):
                        field_name = item.get('canonical', item.get('field_name', ''))
                        if field_name:
                            # Ensure 'source' is in the mapping
                            mapping_with_source = {**item, 'source': source}
                            yield (field_name, mapping_with_source)
                    elif isinstance(item, str):
                        yield (item, {'source': source, 'data_type': 'string'})
        else:
            # Legacy flat format
            for field_name, mapping in field_mappings.items():
                if field_name in ['header_mappings', 'line_item_mappings', 'summary_mappings']:
                    continue  # Skip if somehow grouped keys appear
                if isinstance(mapping, dict):
                    yield (field_name, mapping)

    def _infer_semantic_type_from_canonical(self, canonical: str, source: str) -> str:
        """Infer semantic type from canonical field name and source."""
        canonical_lower = canonical.lower()

        # Header field patterns
        if source == 'header':
            if any(kw in canonical_lower for kw in ['vendor', 'customer', 'store', 'merchant']):
                return 'entity'
            if any(kw in canonical_lower for kw in ['date', 'time']):
                return 'date'
            if any(kw in canonical_lower for kw in ['number', 'id', '#']):
                return 'identifier'
            if 'address' in canonical_lower:
                return 'address'
            if 'payment' in canonical_lower or 'method' in canonical_lower:
                return 'method'

        # Line item patterns
        if source == 'line_item':
            if canonical_lower == 'description' or canonical_lower == 'name':
                return 'product'
            if canonical_lower == 'quantity':
                return 'quantity'
            if canonical_lower in ['amount', 'unit_price', 'price']:
                return 'amount'
            if canonical_lower == 'sku':
                return 'identifier'
            if canonical_lower == 'category':
                return 'category'

        # Summary patterns
        if source == 'summary':
            if 'total' in canonical_lower:
                return 'total'
            if 'tax' in canonical_lower:
                return 'tax'
            if 'subtotal' in canonical_lower:
                return 'subtotal'
            if 'discount' in canonical_lower:
                return 'discount'
            if 'tip' in canonical_lower:
                return 'tip'

        return 'unknown'

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
        """Format field mappings as a readable schema description.

        Supports two formats:
        1. NEW GROUPED FORMAT: {'header_mappings': [...], 'line_item_mappings': [...], ...}
        2. LEGACY FLAT FORMAT: {'field_name': {'data_type': '...', ...}}
        """
        lines = []

        # Check if this is the new grouped format
        is_grouped_format = any(
            field_mappings.get(k) for k in ['header_mappings', 'line_item_mappings', 'summary_mappings']
        )

        if is_grouped_format:
            # Handle grouped format
            source_map = {
                'header_mappings': 'header',
                'line_item_mappings': 'line_item',
                'summary_mappings': 'summary'
            }
            for mapping_key, source in source_map.items():
                mapping_list = field_mappings.get(mapping_key, [])
                if not isinstance(mapping_list, list):
                    continue
                for item in mapping_list:
                    if isinstance(item, dict):
                        field_name = item.get('canonical', item.get('field_name', 'unknown'))
                        data_type = item.get('data_type', 'string')
                        semantic_type = item.get('semantic_type', 'unknown')
                        aggregation = item.get('aggregation', 'none')
                        description = item.get('description', '')
                        lines.append(
                            f"- {field_name}: {data_type} ({semantic_type}) [source: {source}]"
                            f" [aggregation: {aggregation}] - {description}"
                        )
                    elif isinstance(item, str):
                        lines.append(f"- {item}: string (unknown) [source: {source}]")
        else:
            # Handle legacy flat format
            for field_name, mapping in field_mappings.items():
                if not isinstance(mapping, dict):
                    # Skip non-dict mappings
                    continue
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

    def _convert_to_external_storage_sql(self, sql: str) -> str:
        """
        Convert SQL from inline line_items (jsonb_array_elements) to external storage
        (documents_data_line_items table).

        DEPRECATED: As of migration 022, all storage is external.
        This function is kept for backward compatibility with any cached/legacy
        SQL that might still use inline patterns.

        This is a post-processing step that rewrites SQL generated for inline storage
        to work with external storage.

        Inline pattern:
            jsonb_array_elements(dd.line_items) as item

        External pattern:
            JOIN documents_data_line_items li ON li.documents_data_id = dd.id
            ... li.data as item

        Args:
            sql: SQL query using inline storage pattern

        Returns:
            SQL query using external storage pattern
        """
        # Pattern: SELECT ... jsonb_array_elements(dd.line_items) as item FROM documents_data dd
        # Convert to: SELECT ... li.data as item FROM documents_data dd JOIN documents_data_line_items li ON li.documents_data_id = dd.id

        if 'jsonb_array_elements' not in sql.lower():
            logger.info("[SQL Converter] No jsonb_array_elements found, skipping conversion")
            # FIX: Check for undefined 'item' alias and replace with 'li.data'
            # This handles cases where LLM generates SQL using item->>'field' without defining item
            sql = self._fix_undefined_item_alias(sql)
            # FIX: Check for 'FROM expanded_items' without CTE definition
            # This handles cases where LLM references the CTE name from examples but doesn't define it
            sql = self._fix_missing_expanded_items_cte(sql)
            return sql

        # Step 1: Replace jsonb_array_elements(dd.line_items) or jsonb_array_elements(line_items) with li.data as item
        converted = re.sub(
            r'jsonb_array_elements\s*\(\s*(?:dd\.)?line_items\s*\)\s+as\s+item',
            'li.data as item',
            sql,
            flags=re.IGNORECASE
        )

        # Step 2: Add JOIN to documents_data_line_items after FROM documents_data
        # Handle multiple cases for different alias patterns
        if 'JOIN documents_data_line_items' in converted:
            # Already has the JOIN, skip
            pass
        elif re.search(r'FROM\s+documents_data\s+dd\s+JOIN\s+documents\s+d\s+ON', converted, re.IGNORECASE):
            # Pattern: FROM documents_data dd JOIN documents d ON ...
            converted = re.sub(
                r'FROM\s+documents_data\s+dd\s+JOIN\s+documents\s+d\s+ON',
                'FROM documents_data dd\n    JOIN documents_data_line_items li ON li.documents_data_id = dd.id\n    JOIN documents d ON',
                converted,
                flags=re.IGNORECASE
            )
        elif re.search(r'FROM\s+documents_data\s+JOIN\s+documents\s+ON', converted, re.IGNORECASE):
            # Pattern: FROM documents_data JOIN documents ON ... (no alias)
            # First add alias to documents_data
            converted = re.sub(
                r'FROM\s+documents_data\s+JOIN\s+documents\s+ON\s+documents_data\.document_id',
                'FROM documents_data dd\n    JOIN documents_data_line_items li ON li.documents_data_id = dd.id\n    JOIN documents d ON dd.document_id',
                converted,
                flags=re.IGNORECASE
            )
        elif re.search(r'FROM\s+documents_data\s+dd\b', converted, re.IGNORECASE):
            # Pattern: FROM documents_data dd (with alias, no JOIN to documents)
            converted = re.sub(
                r'FROM\s+documents_data\s+dd\b',
                'FROM documents_data dd\n    JOIN documents_data_line_items li ON li.documents_data_id = dd.id',
                converted,
                flags=re.IGNORECASE
            )
        elif re.search(r'FROM\s+documents_data\b', converted, re.IGNORECASE):
            # Pattern: FROM documents_data (no alias)
            converted = re.sub(
                r'FROM\s+documents_data\b',
                'FROM documents_data dd\n    JOIN documents_data_line_items li ON li.documents_data_id = dd.id',
                converted,
                flags=re.IGNORECASE
            )

        # Step 3: Fix any remaining alias issues
        # Replace "documents.id" with "d.id" when "documents d" alias is used
        if 'JOIN documents d ON' in converted or 'JOIN documents d ' in converted:
            converted = re.sub(r'\bdocuments\.id\b', 'd.id', converted)
            converted = re.sub(r'\bdocuments\.document_id\b', 'd.document_id', converted)

        # Replace "documents_data." with "dd." when alias is used
        if 'documents_data dd' in converted:
            converted = re.sub(r'\bdocuments_data\.([\w]+)', r'dd.\1', converted)

        # Step 4: Fix WHERE clause references to 'item' - replace with 'li.data'
        # In external storage, 'item' is just an alias defined in SELECT, can't be used in WHERE
        # We need to replace 'item' with 'li.data' ONLY in WHERE clauses (not in SELECT)
        if 'li.data as item' in converted:
            # Split SQL into parts: before WHERE, WHERE clause, after WHERE
            # Find the WHERE clause within the CTE and fix item references there

            # Pattern to match item->>'field' or (item->>'field')
            # Replace with li.data->>'field' or (li.data->>'field')
            # But ONLY in WHERE clauses, not in the outer SELECT

            # Find CTEs with WHERE clauses and fix them
            def fix_cte_where(match):
                cte_content = match.group(0)
                # Replace item references in WHERE part only
                # Find WHERE ... ) pattern within CTE
                where_pattern = r'(WHERE\s+.+?)(\)\s*$|\)\s*SELECT)'

                def replace_item_refs(where_match):
                    where_clause = where_match.group(1)
                    ending = where_match.group(2)
                    # Replace item->>' with li.data->>'
                    fixed = re.sub(r'\bitem\s*->>\'', "li.data->>'", where_clause)
                    fixed = re.sub(r'\bitem\s*->\'', "li.data->'", fixed)
                    fixed = re.sub(r'\(item->>', "(li.data->>'", fixed)
                    fixed = re.sub(r'\(item->', "(li.data->", fixed)
                    return fixed + ending

                return re.sub(where_pattern, replace_item_refs, cte_content, flags=re.IGNORECASE | re.DOTALL)

            # Match CTE definitions and fix item references in WHERE clauses inside CTEs only
            # Use a smarter approach: find CTEs and only replace item->li.data within them

            # Pattern to find CTE content including nested parentheses
            def find_and_fix_cte_where(sql_text):
                """Find CTEs and replace item with li.data in their WHERE clauses only."""
                # Split by 'FROM items' or similar to separate CTE from outer query
                # We only want to replace item->li.data inside the CTE WHERE, not in outer SELECT

                # Find the CTE block (everything between AS ( and the matching closing ))
                cte_pattern = r'(WITH\s+\w+\s+AS\s*\()(.*?)(\)\s*SELECT)'

                def fix_cte_content(match):
                    cte_start = match.group(1)
                    cte_body = match.group(2)
                    cte_end = match.group(3)

                    # Only replace item->li.data in the WHERE clause of the CTE body
                    # Look for WHERE ... that's inside the CTE
                    where_pattern = r'(WHERE\s+)(.*?)($)'

                    def fix_where(where_match):
                        where_keyword = where_match.group(1)
                        where_content = where_match.group(2)
                        where_end = where_match.group(3)

                        # Replace item references with li.data in WHERE content
                        fixed = re.sub(r'\bitem\s*->>\'', "li.data->>'", where_content)
                        fixed = re.sub(r'\bitem\s*->\'', "li.data->'", fixed)
                        fixed = re.sub(r'\(item->>', "(li.data->>'", fixed)
                        fixed = re.sub(r'\(item->', "(li.data->", fixed)
                        return where_keyword + fixed + where_end

                    fixed_body = re.sub(where_pattern, fix_where, cte_body, flags=re.IGNORECASE | re.DOTALL)
                    return cte_start + fixed_body + cte_end

                return re.sub(cte_pattern, fix_cte_content, sql_text, flags=re.IGNORECASE | re.DOTALL)

            converted = find_and_fix_cte_where(converted)

            # DO NOT replace item->li.data in outer SELECT/WHERE - 'item' is valid there as CTE alias

        logger.info(f"[SQL Converter] Converted SQL for external storage")
        logger.debug(f"[SQL Converter] Original:\n{sql[:200]}...")
        logger.debug(f"[SQL Converter] Converted:\n{converted[:200]}...")

        return converted

    def _fix_undefined_item_alias(self, sql: str) -> str:
        """
        Fix SQL that uses item->>'field' without defining the 'item' alias.

        This is a common LLM error where it generates:
            SELECT SUM((item->>'quantity')::numeric) FROM documents_data dd
            JOIN documents_data_line_items li ON li.documents_data_id = dd.id

        But 'item' is never defined. The correct SQL should use 'li.data' directly:
            SELECT SUM((li.data->>'quantity')::numeric) FROM documents_data dd
            JOIN documents_data_line_items li ON li.documents_data_id = dd.id

        Args:
            sql: SQL query that may have undefined 'item' references

        Returns:
            Fixed SQL query with 'item' replaced by 'li.data'
        """
        # Check if 'item' is used but not defined
        has_item_ref = re.search(r'\bitem\s*->>', sql, re.IGNORECASE)
        has_item_def = re.search(r'\bas\s+item\b', sql, re.IGNORECASE)

        if has_item_ref and not has_item_def:
            # 'item' is referenced but never defined - replace with li.data
            logger.info("[SQL Converter] Fixing undefined 'item' alias - replacing with 'li.data'")

            # Replace all item->>' with li.data->>'
            fixed = re.sub(r'\bitem\s*->>\'', "li.data->>'", sql)
            fixed = re.sub(r'\bitem\s*->\'', "li.data->'", fixed)
            fixed = re.sub(r'\(item->>', "(li.data->>'", fixed)
            fixed = re.sub(r'\(item->', "(li.data->", fixed)

            logger.debug(f"[SQL Converter] Fixed undefined item alias:\n  Before: {sql[:200]}...\n  After: {fixed[:200]}...")
            return fixed

        return sql

    def _fix_missing_expanded_items_cte(self, sql: str) -> str:
        """
        Fix SQL that references 'expanded_items' without defining the CTE.

        This is a common LLM error where it copies the CTE name from prompt examples
        but doesn't include the CTE definition:

        Invalid:
            SELECT SUM((li.data->>'quantity')::numeric) FROM expanded_items;

        Fixed:
            WITH expanded_items AS (
                SELECT dd.document_id, dd.header_data, dd.summary_data, li.data as item
                FROM documents_data dd
                JOIN documents_data_line_items li ON li.documents_data_id = dd.id
            )
            SELECT SUM((item->>'quantity')::numeric) FROM expanded_items;

        OR simpler direct query:
            SELECT SUM((li.data->>'quantity')::numeric)
            FROM documents_data dd
            JOIN documents_data_line_items li ON li.documents_data_id = dd.id;

        Args:
            sql: SQL query that may reference expanded_items without CTE

        Returns:
            Fixed SQL query with proper table references
        """
        # Check if 'expanded_items' is referenced but CTE is not defined
        has_expanded_items = re.search(r'\bFROM\s+expanded_items\b', sql, re.IGNORECASE)
        has_cte_def = re.search(r'\bWITH\s+expanded_items\s+AS\b', sql, re.IGNORECASE)

        if has_expanded_items and not has_cte_def:
            logger.info("[SQL Converter] Fixing missing expanded_items CTE - converting to direct table query")

            # Instead of adding the CTE, convert to direct table query
            # This is simpler and less error-prone

            # Replace 'FROM expanded_items' with the proper JOIN
            fixed = re.sub(
                r'\bFROM\s+expanded_items\b',
                'FROM documents_data dd\n    JOIN documents_data_line_items li ON li.documents_data_id = dd.id',
                sql,
                flags=re.IGNORECASE
            )

            # Also fix any 'item->>' references to 'li.data->>' since we're not using the CTE
            fixed = re.sub(r'\bitem\s*->>\'', "li.data->>'", fixed)
            fixed = re.sub(r'\bitem\s*->\'', "li.data->'", fixed)
            fixed = re.sub(r'\(item->>', "(li.data->>'", fixed)
            fixed = re.sub(r'\(item->', "(li.data->", fixed)

            logger.debug(f"[SQL Converter] Fixed missing CTE:\n  Before: {sql[:200]}...\n  After: {fixed[:200]}...")
            return fixed

        return sql

    def _fix_union_syntax(self, sql: str) -> str:
        """
        Fix common UNION ALL syntax errors where ORDER BY/LIMIT appears before UNION.

        In PostgreSQL, when using UNION ALL with ORDER BY/LIMIT on individual parts,
        each part must be wrapped in parentheses as a subquery.

        Invalid:
            SELECT ... ORDER BY x LIMIT 1 UNION ALL SELECT ... ORDER BY y LIMIT 1

        Valid:
            (SELECT ... ORDER BY x LIMIT 1) UNION ALL (SELECT ... ORDER BY y LIMIT 1)

        Args:
            sql: SQL query that may have invalid UNION syntax

        Returns:
            Fixed SQL query
        """
        if 'UNION' not in sql.upper():
            return sql

        # Check if there's an ORDER BY or LIMIT before UNION
        # Pattern: ... ORDER BY ... LIMIT ... UNION
        pattern = r'(ORDER\s+BY\s+[^\)]+?\s+LIMIT\s+\d+)\s*(UNION\s+ALL|UNION)'

        if not re.search(pattern, sql, re.IGNORECASE | re.DOTALL):
            return sql

        logger.info("[SQL Fixer] Detected ORDER BY/LIMIT before UNION, fixing syntax...")

        # Split by UNION ALL (case insensitive)
        parts = re.split(r'\s+(UNION\s+ALL|UNION)\s+', sql, flags=re.IGNORECASE)

        if len(parts) < 3:
            return sql

        # Reconstruct with parentheses
        fixed_parts = []
        for i, part in enumerate(parts):
            part = part.strip()
            if part.upper() in ('UNION ALL', 'UNION'):
                fixed_parts.append(part)
            elif part:
                # Remove any trailing ORDER BY that applies to the whole query
                # (appears after the last UNION part)
                if i == len(parts) - 1:
                    # Check if there's a final ORDER BY after the subquery
                    # Pattern: subquery content ORDER BY ... (at the end)
                    final_order_match = re.search(
                        r'^(.+?LIMIT\s+\d+)\s+(ORDER\s+BY\s+.+)$',
                        part,
                        re.IGNORECASE | re.DOTALL
                    )
                    if final_order_match:
                        subquery = final_order_match.group(1).strip()
                        final_order = final_order_match.group(2).strip()
                        fixed_parts.append(f"({subquery})")
                        fixed_parts.append(final_order)
                    else:
                        fixed_parts.append(f"({part})")
                else:
                    fixed_parts.append(f"({part})")

        # Join back together
        result_parts = []
        i = 0
        while i < len(fixed_parts):
            if fixed_parts[i].upper() in ('UNION ALL', 'UNION'):
                result_parts.append(f"\n{fixed_parts[i]}\n")
            elif fixed_parts[i].upper().startswith('ORDER BY'):
                result_parts.append(f"\n{fixed_parts[i]}")
            else:
                result_parts.append(fixed_parts[i])
            i += 1

        fixed_sql = ''.join(result_parts)
        logger.info("[SQL Fixer] Fixed UNION syntax")
        logger.debug(f"[SQL Fixer] Fixed SQL:\n{fixed_sql[:300]}...")

        return fixed_sql

    def _fix_document_level_count(self, sql: str) -> str:
        """
        Fix document-level count queries when LLM generates COUNT(*) instead of COUNT(DISTINCT document_id).

        This is a post-processing step for count_level="document" queries.

        The LLM should generate:
        - SELECT dd.document_id in the CTE
        - COUNT(DISTINCT document_id) in the outer SELECT

        But sometimes it generates COUNT(*) which counts line items instead of documents.
        This method fixes that.
        """
        import re

        original_sql = sql

        # Step 1: Check if CTE includes document_id
        cte_match = re.search(
            r'WITH\s+\w+\s+AS\s*\(\s*SELECT\s+(.*?)\s+FROM',
            sql,
            re.IGNORECASE | re.DOTALL
        )

        if cte_match:
            cte_select = cte_match.group(1)
            # Check if document_id is already in the CTE SELECT
            if 'document_id' not in cte_select.lower():
                # Add dd.document_id to the CTE SELECT
                logger.info("[SQL Fix] Adding document_id to CTE for document-level count")
                sql = re.sub(
                    r'(WITH\s+\w+\s+AS\s*\(\s*SELECT\s+)',
                    r'\1dd.document_id, ',
                    sql,
                    flags=re.IGNORECASE
                )

        # Step 2: Replace COUNT(*) with COUNT(DISTINCT document_id) in outer SELECT
        # Match patterns like: COUNT(*) AS xxx, COUNT(*) as xxx, COUNT(*)
        if re.search(r'\bCOUNT\s*\(\s*\*\s*\)', sql, re.IGNORECASE):
            logger.info("[SQL Fix] Replacing COUNT(*) with COUNT(DISTINCT document_id) for document-level count")
            sql = re.sub(
                r'\bCOUNT\s*\(\s*\*\s*\)',
                'COUNT(DISTINCT document_id)',
                sql,
                flags=re.IGNORECASE
            )

        if sql != original_sql:
            logger.info("[SQL Fix] Document-level count SQL fixed:")
            logger.debug(f"[SQL Fix] Before: {original_sql[:200]}...")
            logger.debug(f"[SQL Fix] After: {sql[:200]}...")

        return sql

    def _ensure_summary_data_fields(self, sql: str) -> str:
        """
        Ensure summary_data fields (subtotal, total_amount) are included in SELECT for detail reports.

        For detail/invoice reports, the UI formatter needs summary_data fields to display
        document-level subtotals and totals instead of calculating from line items.

        This method adds missing summary_data fields to both:
        1. The CTE SELECT (to make them available)
        2. The outer SELECT (to include them in results)

        Args:
            sql: SQL query that may be missing summary_data fields

        Returns:
            SQL query with summary_data fields included
        """
        import re

        original_sql = sql
        summary_fields = ['subtotal', 'tax_amount', 'total_amount']

        # Check if this is a CTE-based query
        cte_match = re.search(
            r'WITH\s+(\w+)\s+AS\s*\(\s*SELECT\s+(.*?)\s+FROM\s+documents_data',
            sql,
            re.IGNORECASE | re.DOTALL
        )

        if not cte_match:
            logger.debug("[SQL Fix] No CTE found, skipping summary_data field injection")
            return sql

        cte_name = cte_match.group(1)
        cte_select = cte_match.group(2)

        # Check which summary fields are already in CTE
        missing_in_cte = []
        for field in summary_fields:
            # Check for patterns like: dd.summary_data->>'subtotal', summary_data->>'subtotal', dd.summary_data
            if f"summary_data->>'{field}'" not in cte_select.lower() and 'summary_data' not in cte_select.lower():
                missing_in_cte.append(field)

        # If summary_data is already in CTE SELECT but individual fields aren't extracted, that's fine
        # because the outer SELECT can access them via summary_data->>'field'
        if 'summary_data' in cte_select.lower() or 'dd.summary_data' in cte_select.lower():
            # summary_data column is included, now check outer SELECT
            pass
        elif missing_in_cte:
            # Need to add dd.summary_data to CTE SELECT
            logger.info("[SQL Fix] Adding dd.summary_data to CTE for detail report")
            sql = re.sub(
                r'(WITH\s+\w+\s+AS\s*\(\s*SELECT\s+)',
                r'\1dd.summary_data, ',
                sql,
                flags=re.IGNORECASE
            )

        # Now check and fix the outer SELECT to include summary_data fields
        # Find the outer SELECT after the CTE
        outer_select_match = re.search(
            rf'\)\s*SELECT\s+(.*?)\s+FROM\s+{cte_name}',
            sql,
            re.IGNORECASE | re.DOTALL
        )

        if outer_select_match:
            outer_select = outer_select_match.group(1)

            # Check which summary fields are missing from outer SELECT
            fields_to_add = []
            for field in summary_fields:
                # Check for patterns like: summary_data->>'subtotal' AS subtotal, subtotal (as column alias)
                field_pattern = rf"(summary_data\s*->>\s*'{field}'|{field}\s+AS\s+|AS\s+{field}|\b{field}\b)"
                if not re.search(field_pattern, outer_select, re.IGNORECASE):
                    fields_to_add.append(f"summary_data->>'{field}' AS {field}")

            if fields_to_add:
                logger.info(f"[SQL Fix] Adding summary fields to outer SELECT: {fields_to_add}")
                # Add the fields at the end of the SELECT list
                fields_str = ", " + ", ".join(fields_to_add)
                # Insert before FROM in the outer SELECT
                sql = re.sub(
                    rf'(\)\s*SELECT\s+{re.escape(outer_select)})(\s+FROM\s+{cte_name})',
                    rf'\1{fields_str}\2',
                    sql,
                    flags=re.IGNORECASE
                )

        if sql != original_sql:
            logger.info("[SQL Fix] Summary data fields added to SQL for detail report")
            logger.debug(f"[SQL Fix] Before: {original_sql[:300]}...")
            logger.debug(f"[SQL Fix] After: {sql[:300]}...")

        return sql

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

    def _detect_report_type_from_query(self, user_query: str) -> str:
        """
        Detect report type from query keywords when LLM is not available.

        Returns 'detail' for list/detail queries, 'summary' otherwise.
        """
        query_lower = user_query.lower()

        # Keywords that indicate a detail/list report
        detail_keywords = [
            'list all', 'show all', 'get all',
            'list details', 'show details', 'get details',
            'all details', 'full details', 'all records',
            'all items', 'every item', 'each item',
            'all line items', 'every line item',
            'list the', 'show the items', 'display all'
        ]

        for keyword in detail_keywords:
            if keyword in query_lower:
                logger.info(f"[SQL Generator] Detected report_type='detail' from keyword: '{keyword}'")
                return 'detail'

        return 'summary'

    def _generate_sql_heuristic(
        self,
        user_query: str,
        field_mappings: Dict[str, Dict[str, Any]],
        table_filter: Optional[str] = None,
        storage_type: str = "external",
        report_type: str = "summary"
    ) -> SQLGenerationResult:
        """
        Generate SQL using heuristics when LLM is not available.

        This provides a fallback that handles common query patterns.

        NOTE: As of migration 022, always uses external storage (documents_data_line_items).

        Args:
            report_type: "summary" for aggregations, "detail" for all records with all columns
        """
        query_lower = user_query.lower()

        # Handle detail report type - return all records with all columns
        if report_type == "detail":
            return self._generate_detail_sql_heuristic(field_mappings, table_filter, storage_type)

        # Find key fields and their sources
        date_field = None
        date_field_source = 'header'  # Default to header for dates
        amount_field = None
        amount_field_source = 'line_item'  # Default to line_item for amounts
        category_fields = []

        for field_name, mapping in self._iterate_field_mappings(field_mappings):
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

        # Convert to external storage SQL if needed
        if storage_type == 'external':
            logger.info("[SQL Heuristic] Converting SQL for external storage...")
            sql = self._convert_to_external_storage_sql(sql)

        return SQLGenerationResult(
            sql_query=sql.strip(),
            explanation=f"Aggregating {amount_field or 'data'} by {', '.join(grouping_fields) if grouping_fields else 'all'}"
                       + (f" and {time_granularity}" if time_granularity else ""),
            grouping_fields=grouping_fields,
            aggregation_fields=[amount_field] if amount_field else [],
            time_granularity=time_granularity,
            success=True
        )

    def _generate_detail_sql_heuristic(
        self,
        field_mappings: Dict[str, Dict[str, Any]],
        table_filter: Optional[str] = None,
        storage_type: str = "external"
    ) -> SQLGenerationResult:
        """
        Generate SQL for detail report - return all records with all columns.

        This is used when report_type='detail' to list all line items with all fields.
        """
        logger.info("[SQL Heuristic] Generating DETAIL SQL (all records, all columns)...")

        # Collect all fields by source
        header_fields = []
        line_item_fields = []
        summary_fields = []

        for field_name, mapping in self._iterate_field_mappings(field_mappings):
            source = mapping.get('source', 'line_item')
            if source == 'header':
                header_fields.append(field_name)
            elif source == 'summary':
                summary_fields.append(field_name)
            else:
                line_item_fields.append(field_name)

        # Build SELECT parts
        select_parts = []

        # Add header fields
        for field in header_fields:
            select_parts.append(f"header_data->>'{field}' as {field.lower().replace(' ', '_')}")

        # Add summary fields (from summary_data column)
        for field in summary_fields:
            select_parts.append(f"summary_data->>'{field}' as {field.lower().replace(' ', '_')}")

        # Add line item fields
        for field in line_item_fields:
            select_parts.append(f"item->>'{field}' as {field.lower().replace(' ', '_')}")

        # Add row_number if not already included
        if 'row_number' not in line_item_fields:
            select_parts.append("item->>'row_number' as row_number")

        # If no fields defined, use generic approach
        if not select_parts:
            select_parts = [
                "header_data->>'{date_field}' as date".format(date_field=self._find_date_field(field_mappings) or 'date'),
                "item->>'description' as description",
                "item->>'quantity' as quantity",
                "item->>'amount' as amount",
                "item->>'row_number' as row_number"
            ]

        # Build WHERE clause
        where_clause = ""
        if table_filter:
            where_clause = f"WHERE {table_filter}"

        # Build SQL - include summary_data for summary fields
        sql = f"""
WITH expanded_items AS (
    SELECT dd.header_data, dd.summary_data, jsonb_array_elements(dd.line_items) as item
    FROM documents_data dd
    JOIN documents d ON dd.document_id = d.id
    {where_clause}
)
SELECT
    {', '.join(select_parts)}
FROM expanded_items
ORDER BY header_data->>'date' DESC, (item->>'row_number')::int ASC
"""

        # Convert to external storage SQL if needed
        if storage_type == 'external':
            logger.info("[SQL Heuristic] Converting detail SQL for external storage...")
            sql = self._convert_to_external_storage_sql(sql)

        all_fields = header_fields + summary_fields + line_item_fields
        logger.info(f"[SQL Heuristic] Detail SQL includes {len(all_fields)} fields: header={header_fields}, summary={summary_fields}, line_items={line_item_fields}")

        return SQLGenerationResult(
            sql_query=sql.strip(),
            explanation=f"Listing all line items with {len(all_fields)} fields",
            grouping_fields=[],
            aggregation_fields=[],
            time_granularity=None,
            success=True
        )

    def _find_date_field(self, field_mappings: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Find the date field in field mappings."""
        for field_name, mapping in self._iterate_field_mappings(field_mappings):
            if mapping.get('semantic_type') == 'date':
                return field_name
        return None

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
        field_mappings: Dict[str, Dict[str, Any]],
        schema_type: str = "unknown",
        sql_explanation: str = ""
    ) -> Dict[str, Any]:
        """
        Generate summary report using LLM with a simple, natural approach.

        Enhanced with content size evaluation:
        - For small result sets: LLM processes full data
        - For large result sets: LLM generates summary header only, then raw markdown table is appended

        Args:
            user_query: Original user query
            query_results: Results from SQL execution
            field_mappings: Field schema information
            schema_type: Type of document schema (for context)
            sql_explanation: Explanation of the generated SQL query

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
            # Step 1: Evaluate content size to determine processing strategy
            evaluator = ContentSizeEvaluator()
            evaluation = evaluator.evaluate(query_results)

            logger.info(
                f"[LLM Summary] Content size evaluation: {evaluation.row_count} rows, "
                f"~{evaluation.estimated_tokens:,} tokens, exceeds_limit={evaluation.exceeds_limit}"
            )

            # Calculate grand total for metadata
            columns = evaluation.column_headers
            amount_col = None
            for col in columns:
                col_lower = col.lower()
                if any(kw in col_lower for kw in ['amount', 'total', 'sales', 'sum', 'price', 'value']):
                    amount_col = col
                    break

            grand_total = 0
            if amount_col:
                grand_total = sum(float(r.get(amount_col, 0) or 0) for r in query_results)

            # Step 2: Choose processing strategy based on size
            if evaluation.exceeds_limit:
                # LARGE RESULT SET: Generate summary header + raw markdown table
                formatted_report = self._generate_large_result_report(
                    user_query=user_query,
                    query_results=query_results,
                    evaluation=evaluation,
                    evaluator=evaluator,
                    schema_type=schema_type,
                    sql_explanation=sql_explanation
                )
            else:
                # SMALL RESULT SET: LLM processes full data using LLMResultFormatter
                formatted_report = self._generate_small_result_report(
                    user_query=user_query,
                    query_results=query_results,
                    evaluation=evaluation,
                    field_mappings=field_mappings
                )

            return {
                "report_title": "Query Results Summary",
                "hierarchical_data": [],
                "grand_total": round(grand_total, 2),
                "total_records": len(query_results),
                "formatted_report": formatted_report
            }

        except Exception as e:
            logger.error(f"[LLM Summary] LLM summary generation failed: {e}")
            return self._generate_summary_heuristic(query_results, field_mappings, user_query)

    def _generate_large_result_report(
        self,
        user_query: str,
        query_results: List[Dict[str, Any]],
        evaluation: ContentSizeEvaluation,
        evaluator: ContentSizeEvaluator,
        schema_type: str,
        sql_explanation: str
    ) -> str:
        """
        Generate report for large result sets that exceed LLM token limits.

        Strategy:
        1. Generate summary header via LLM (using metadata + samples only)
        2. Append the raw markdown table directly (no LLM processing)

        Args:
            user_query: Original user query
            query_results: Full query results
            evaluation: Content size evaluation result
            evaluator: ContentSizeEvaluator instance
            schema_type: Document schema type
            sql_explanation: SQL query explanation

        Returns:
            Complete formatted report string
        """
        logger.info(
            f"[LLM Summary] Using large result strategy for {evaluation.row_count} rows "
            f"(~{evaluation.estimated_tokens:,} tokens)"
        )

        report_parts = []

        # Step 1: Calculate statistics for LLM context
        stats = evaluator.calculate_column_statistics(query_results, evaluation.column_headers)
        stats_formatted = evaluator.format_statistics_for_llm(stats)

        # Step 2: Get sample rows for LLM context
        sample_rows = evaluator.get_sample_rows(query_results)
        sample_markdown = evaluator._build_markdown_table(sample_rows, evaluation.column_headers)

        # Step 3: Build prompt for summary header only
        prompt = self.LARGE_RESULT_SUMMARY_PROMPT.format(
            user_query=user_query,
            row_count=evaluation.row_count,
            column_names=", ".join(evaluation.column_headers),
            statistics=stats_formatted,
            schema_type=schema_type or "unknown",
            sql_explanation=sql_explanation or "Query executed successfully",
            sample_count=len(sample_rows),
            sample_markdown=sample_markdown
        )

        # Step 4: Generate summary header via LLM
        try:
            summary_header = self.llm_client.generate(prompt)
            # Clean up the response
            summary_header = summary_header.strip()
            if summary_header.startswith('```'):
                lines = summary_header.split('\n')
                summary_header = '\n'.join(
                    line for line in lines
                    if not line.startswith('```')
                )
            report_parts.append(summary_header)
        except Exception as e:
            logger.error(f"[LLM Summary] Failed to generate summary header: {e}")
            # Fallback: Generate a simple header
            report_parts.append(f"## Query Results\n\n")
            report_parts.append(f"Found **{evaluation.row_count:,}** records matching your query.\n")

        # Step 5: Add separator between summary and data table
        report_parts.append("\n\n---\n\n")
        report_parts.append("## Complete Data Results\n\n")

        # Step 6: Append the raw markdown table directly (pre-built during evaluation)
        report_parts.append(evaluation.markdown_table)

        # Step 7: Add footer with row count
        report_parts.append(f"\n\n*Showing all {evaluation.row_count:,} records*\n")

        return "".join(report_parts)

    def _generate_small_result_report(
        self,
        user_query: str,
        query_results: List[Dict[str, Any]],
        evaluation: ContentSizeEvaluation,
        field_mappings: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> str:
        """
        Generate report for small result sets using LLMResultFormatter.

        Uses LLMResultFormatter for schema-aware, query-type-aware formatting
        instead of hardcoded receipt/invoice templates.

        Args:
            user_query: Original user query
            query_results: Query results
            evaluation: Content size evaluation (contains pre-built markdown table)
            field_mappings: Optional field schema for context

        Returns:
            Formatted report string
        """
        # Use LLMResultFormatter for intelligent, schema-aware formatting
        result_formatter = LLMResultFormatter(
            llm_client=self.llm_client,
            schema_service=self.schema_service
        )

        # Get the formatted markdown table from LLMResultFormatter
        formatted_table = result_formatter.format_results(
            query_results=query_results,
            user_query=user_query,
            field_mappings=field_mappings
        )

        # Now use LLM to generate a natural language summary based on the query and data
        # This prompt is generic and does NOT assume receipt/invoice structure
        actual_columns = list(query_results[0].keys()) if query_results else []
        columns_list = ", ".join(actual_columns)

        # Detect if user wants detailed listing vs summary
        detail_keywords = ['details', 'detail', 'list', 'show all', 'all items', 'individual',
                           'each', 'breakdown', 'itemized', 'every', 'specific']
        query_lower = user_query.lower()
        wants_details = any(keyword in query_lower for keyword in detail_keywords)
        report_type = "detailed" if wants_details else "summary"

        prompt = f"""Based on the user's question and the SQL query results, write a clear {report_type} report.

## User's Question:
"{user_query}"

## AVAILABLE COLUMNS IN THIS DATA (ONLY THESE EXIST):
{columns_list}

## SQL Query Results ({len(query_results)} rows):

{formatted_table}

## CRITICAL RULES:
1. **ONLY use columns that exist in the data above** - never invent fields
2. **Detect the data type from the query and columns**:
   - If columns contain aggregates like "total_", "sum_", "avg_", "count_", "min_", "max_" → This is AGGREGATE data, present as statistics
   - If columns contain identifiers like "id", "name", "description" with multiple rows → This is DETAIL data, present as a list/table
   - Match your output format to the data type - don't force invoice/receipt structure on inventory or other data types
3. **Format numbers appropriately - THIS IS CRITICAL**:
   - MONETARY values (price, cost, revenue, sales, fee, charge): Use $ prefix (e.g., $1,234.56)
   - NON-MONETARY values: Plain numbers WITHOUT $ sign:
     * Inventory counts/quantities (inventory, stock, units, products, items, qty): e.g., 267,159 units
     * Product counts: e.g., 1,000 products
     * Average inventory: e.g., 267.16 (NOT $267.16 - inventory is NOT money!)
     * Any count, quantity, or unit-based metric
   - Rule of thumb: If it's counting THINGS (products, items, inventory units), NO dollar sign
   - Rule of thumb: If it's counting MONEY (dollars, revenue, cost), USE dollar sign
4. **Present data naturally based on what it represents**:
   - Inventory data → Show as inventory summary/list (units, NOT dollars)
   - Sales data → Show as sales report
   - Aggregate statistics → Show as key metrics/statistics
   - Don't force hierarchical receipt/invoice layout on non-receipt data
5. Use markdown formatting for readability
6. Be concise but complete - directly answer the user's question

Write the {report_type} report in markdown format:"""

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

        return formatted_report

    async def stream_summary_report(
        self,
        user_query: str,
        query_results: List[Dict[str, Any]],
        field_mappings: Dict[str, Dict[str, Any]],
        schema_type: str = "unknown",
        sql_explanation: str = ""
    ):
        """
        Stream the summary report generation for real-time display.

        Enhanced with content size evaluation:
        - For small result sets: LLM processes full data (current behavior)
        - For large result sets: LLM generates summary header only, then raw markdown table is streamed

        Yields chunks of the report as they are generated by the LLM.

        Args:
            user_query: Original user query
            query_results: Results from SQL execution
            field_mappings: Field schema information
            schema_type: Type of document schema (for context)
            sql_explanation: Explanation of the generated SQL query

        Yields:
            Chunks of the formatted report text
        """
        if not self.llm_client or not hasattr(self.llm_client, 'astream'):
            # Fallback to non-streaming if streaming not available
            result = self.generate_summary_report(user_query, query_results, field_mappings)
            formatted_report = result.get('formatted_report', '')
            if formatted_report:
                yield formatted_report
            return

        if not query_results:
            yield "No data found for your query."
            return

        try:
            # Step 1: Evaluate content size to determine processing strategy
            evaluator = ContentSizeEvaluator()
            evaluation = evaluator.evaluate(query_results)

            logger.info(
                f"[LLM Summary] Content size evaluation: {evaluation.row_count} rows, "
                f"~{evaluation.estimated_tokens:,} tokens, exceeds_limit={evaluation.exceeds_limit}"
            )

            # Step 2: Choose processing strategy based on size
            if evaluation.exceeds_limit:
                # LARGE RESULT SET: Stream summary header + raw markdown table
                async for chunk in self._stream_large_result_response(
                    user_query=user_query,
                    query_results=query_results,
                    evaluation=evaluation,
                    evaluator=evaluator,
                    field_mappings=field_mappings,
                    schema_type=schema_type,
                    sql_explanation=sql_explanation
                ):
                    yield chunk
            else:
                # SMALL RESULT SET: Use current LLM-based formatting
                async for chunk in self._stream_small_result_response(
                    user_query=user_query,
                    query_results=query_results,
                    evaluation=evaluation,
                    field_mappings=field_mappings
                ):
                    yield chunk

            logger.info(f"[LLM Summary] Streamed summary report completed")

        except Exception as e:
            logger.error(f"[LLM Summary] Streaming summary generation failed: {e}")
            # Fallback to non-streaming
            result = self._generate_summary_heuristic(query_results, field_mappings, user_query)
            formatted_report = result.get('formatted_report', '')
            if formatted_report:
                yield formatted_report

    async def _stream_large_result_response(
        self,
        user_query: str,
        query_results: List[Dict[str, Any]],
        evaluation: ContentSizeEvaluation,
        evaluator: ContentSizeEvaluator,
        field_mappings: Dict[str, Dict[str, Any]],
        schema_type: str,
        sql_explanation: str
    ):
        """
        Stream response for large result sets that exceed LLM token limits.

        Strategy:
        1. Generate summary header via LLM (using metadata + samples only)
        2. Stream the raw markdown table directly (no LLM processing)

        Args:
            user_query: Original user query
            query_results: Full query results
            evaluation: Content size evaluation result
            evaluator: ContentSizeEvaluator instance
            field_mappings: Field schema information
            schema_type: Document schema type
            sql_explanation: SQL query explanation

        Yields:
            Chunks of the formatted report
        """
        logger.info(
            f"[LLM Summary] Using large result strategy for {evaluation.row_count} rows "
            f"(~{evaluation.estimated_tokens:,} tokens)"
        )

        # Step 1: Calculate statistics for LLM context
        stats = evaluator.calculate_column_statistics(query_results, evaluation.column_headers)
        stats_formatted = evaluator.format_statistics_for_llm(stats)

        # Step 2: Get sample rows for LLM context
        sample_rows = evaluator.get_sample_rows(query_results)
        sample_markdown = evaluator._build_markdown_table(sample_rows, evaluation.column_headers)

        # Step 3: Build prompt for summary header only
        prompt = self.LARGE_RESULT_SUMMARY_PROMPT.format(
            user_query=user_query,
            row_count=evaluation.row_count,
            column_names=", ".join(evaluation.column_headers),
            statistics=stats_formatted,
            schema_type=schema_type or "unknown",
            sql_explanation=sql_explanation or "Query executed successfully",
            sample_count=len(sample_rows),
            sample_markdown=sample_markdown
        )

        # Step 4: Stream the LLM-generated summary header
        try:
            async for chunk in self.llm_client.astream(prompt):
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error(f"[LLM Summary] Failed to generate summary header: {e}")
            # Fallback: Generate a simple header
            yield f"## Query Results\n\n"
            yield f"Found **{evaluation.row_count:,}** records matching your query.\n\n"

        # Step 5: Add separator between summary and data
        yield "\n\n---\n\n"
        yield "## Complete Data Results\n\n"

        # Step 6: Stream grouped markdown output (documents grouped with nested line items)
        # Use grouped format for better readability
        # Pass field_mappings to determine correct grouping columns based on document type
        grouped_markdown = evaluator.build_grouped_markdown(query_results, field_mappings=field_mappings)
        yield grouped_markdown

    async def _stream_small_result_response(
        self,
        user_query: str,
        query_results: List[Dict[str, Any]],
        evaluation: ContentSizeEvaluation,
        field_mappings: Dict[str, Dict[str, Any]] = None
    ):
        """
        Stream response for small result sets using schema-aware formatting.

        Uses LLMResultFormatter for intelligent formatting based on data type,
        not hardcoded receipt/invoice templates.

        Args:
            user_query: Original user query
            query_results: Query results
            evaluation: Content size evaluation (contains pre-built markdown table)
            field_mappings: Field mappings from document metadata

        Yields:
            Chunks of the formatted report
        """
        # Use LLMResultFormatter for intelligent, schema-aware formatting
        result_formatter = LLMResultFormatter(
            llm_client=self.llm_client,
            schema_service=self.schema_service
        )

        # Get the formatted markdown table from LLMResultFormatter
        formatted_table = result_formatter.format_results(
            query_results=query_results,
            user_query=user_query,
            field_mappings=field_mappings
        )

        # Get actual column names from the results
        actual_columns = list(query_results[0].keys()) if query_results else []
        columns_list = ", ".join(actual_columns)

        # Detect if user wants detailed listing vs summary
        detail_keywords = ['details', 'detail', 'list', 'show all', 'all items', 'individual',
                           'each', 'breakdown', 'itemized', 'every', 'specific']
        query_lower = user_query.lower()
        wants_details = any(keyword in query_lower for keyword in detail_keywords)
        report_type = "detailed" if wants_details else "summary"

        prompt = f"""Based on the user's question and the SQL query results, write a clear {report_type} report.

## User's Question:
"{user_query}"

## AVAILABLE COLUMNS IN THIS DATA (ONLY THESE EXIST):
{columns_list}

## SQL Query Results ({len(query_results)} rows):

{formatted_table}

## CRITICAL RULES:
1. **ONLY use columns that exist in the data above** - never invent fields
2. **Detect the data type from the query and columns**:
   - If columns contain aggregates like "total_", "sum_", "avg_", "count_", "min_", "max_" → This is AGGREGATE data, present as statistics
   - If columns contain identifiers like "id", "name", "description" with multiple rows → This is DETAIL data, present as a list/table
   - Match your output format to the data type - don't force invoice/receipt structure on inventory or other data types
3. **Format numbers appropriately - THIS IS CRITICAL**:
   - MONETARY values (price, cost, revenue, sales, fee, charge): Use $ prefix (e.g., $1,234.56)
   - NON-MONETARY values: Plain numbers WITHOUT $ sign:
     * Inventory counts/quantities (inventory, stock, units, products, items, qty): e.g., 267,159 units
     * Product counts: e.g., 1,000 products
     * Average inventory: e.g., 267.16 (NOT $267.16 - inventory is NOT money!)
     * Any count, quantity, or unit-based metric
   - Rule of thumb: If it's counting THINGS (products, items, inventory units), NO dollar sign
   - Rule of thumb: If it's counting MONEY (dollars, revenue, cost), USE dollar sign
4. **Present data naturally based on what it represents**:
   - Inventory data → Show as inventory summary/list (units, NOT dollars)
   - Sales data → Show as sales report
   - Aggregate statistics → Show as key metrics/statistics
   - Don't force hierarchical receipt/invoice layout on non-receipt data
5. Use markdown formatting for readability
6. Be concise but complete - directly answer the user's question

Write the {report_type} report in markdown format:"""

        # Stream the LLM-generated report
        try:
            async for chunk in self.llm_client.astream(prompt):
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error(f"[LLM Summary] Failed to generate formatted report: {e}")
            # Fallback: Generate simple title + raw table
            yield f"## Query Results\n\n"
            yield f"Found **{len(query_results):,}** records matching your query.\n\n"
            yield "### Complete Data Results\n\n"
            yield evaluation.markdown_table
            yield f"\n\n*Showing all {len(query_results)} records*\n"

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
            # Check is_currency flag for this row (can be boolean or string)
            is_currency = row.get('is_currency')
            if isinstance(is_currency, str):
                is_currency = is_currency.lower() == 'true'
            # Default to True (show as currency) if not specified
            if is_currency is None:
                is_currency = True

            for col in columns:
                val = row.get(col, "")
                col_lower = col.lower()

                # Try to convert string values to numbers (SQL returns JSONB as strings)
                numeric_val = val
                if isinstance(val, str) and val.strip():
                    try:
                        cleaned = val.replace(',', '').replace('$', '').strip()
                        numeric_val = float(cleaned)
                    except (ValueError, AttributeError):
                        numeric_val = val

                # Format numeric values
                if isinstance(numeric_val, (int, float)):
                    if 'amount' in col_lower or 'total' in col_lower or 'sales' in col_lower or 'price' in col_lower:
                        if is_currency:
                            values.append(f"${numeric_val:,.2f}")
                        else:
                            # Non-monetary amount - format without $ sign
                            if numeric_val == int(numeric_val):
                                values.append(f"{int(numeric_val):,}")
                            else:
                                values.append(f"{numeric_val:,.2f}")
                    elif 'quantity' in col_lower or 'count' in col_lower:
                        # Quantity/count columns - always show as integer
                        if numeric_val == int(numeric_val):
                            values.append(f"{int(numeric_val):,}")
                        else:
                            values.append(f"{numeric_val:,.2f}")
                    else:
                        values.append(str(int(numeric_val)) if numeric_val == int(numeric_val) else f"{numeric_val:,.2f}")
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
