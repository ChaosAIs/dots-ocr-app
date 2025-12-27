"""
SQL Query Executor for Analytics

Executes SQL queries against the documents_data table for analytics queries.
Supports filtering, aggregation, and report generation from extracted document data.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime
from decimal import Decimal

from sqlalchemy.orm import Session
from sqlalchemy import text, func, and_, or_, cast, Float
from sqlalchemy.dialects.postgresql import JSONB

from db.models import DocumentData, Document, DataSchema

logger = logging.getLogger(__name__)


class SQLQueryExecutor:
    """
    Executes SQL queries on documents_data table for analytics.

    Supports:
    - Filtering by schema type (invoice, receipt, etc.)
    - Filtering by time range
    - Filtering by entities (vendor, customer, etc.)
    - Aggregations (sum, count, avg, min, max)
    - Grouping by time periods (monthly, quarterly, yearly)
    - Access control filtering
    """

    def __init__(self, db: Session):
        self.db = db

    def execute_analytics_query(
        self,
        accessible_doc_ids: List[UUID],
        schema_types: Optional[List[str]] = None,
        time_range: Optional[Dict[str, str]] = None,
        entities: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        group_by: Optional[str] = None,  # 'monthly', 'quarterly', 'yearly', 'vendor', 'customer'
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """
        Execute an analytics query on documents_data.

        Args:
            accessible_doc_ids: List of document IDs user can access
            schema_types: Filter by document types (invoice, receipt, etc.)
            time_range: Dict with 'start' and 'end' dates (ISO format)
            entities: Filter by entity names (vendor, customer names)
            metrics: Metrics to calculate (total_amount, count, avg_amount, etc.)
            group_by: Grouping strategy
            filters: Additional JSONB field filters
            limit: Maximum rows to return

        Returns:
            Dict with 'data', 'summary', 'metadata'
        """
        logger.info(f"[SQL Query] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[SQL Query] ANALYTICS QUERY EXECUTION START")
        logger.info(f"[SQL Query] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[SQL Query] Parameters:")
        logger.info(f"[SQL Query]   • accessible_doc_ids: {len(accessible_doc_ids) if accessible_doc_ids else 0} documents")
        logger.info(f"[SQL Query]   • schema_types: {schema_types}")
        logger.info(f"[SQL Query]   • time_range: {time_range}")
        logger.info(f"[SQL Query]   • entities: {entities}")
        logger.info(f"[SQL Query]   • metrics: {metrics}")
        logger.info(f"[SQL Query]   • group_by: {group_by}")
        logger.info(f"[SQL Query]   • filters: {filters}")
        logger.info(f"[SQL Query]   • limit: {limit}")

        if not accessible_doc_ids:
            logger.warning(f"[SQL Query] No accessible documents - returning empty result")
            return {
                "data": [],
                "summary": {"total_records": 0},
                "metadata": {"error": "No accessible documents"}
            }

        try:
            # First, check what data exists in documents_data for these documents
            existing_data = self.db.query(
                DocumentData.document_id,
                DocumentData.schema_type,
                Document.original_filename
            ).join(
                Document, DocumentData.document_id == Document.id
            ).filter(
                DocumentData.document_id.in_(accessible_doc_ids)
            ).all()

            logger.info(f"[SQL Query] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"[SQL Query] AVAILABLE DATA IN documents_data TABLE:")
            logger.info(f"[SQL Query]   • Total records found: {len(existing_data)}")
            if existing_data:
                for doc in existing_data:
                    logger.info(f"[SQL Query]   • doc_id={str(doc.document_id)[:8]}..., schema_type='{doc.schema_type}', file='{doc.original_filename}'")
            else:
                logger.warning(f"[SQL Query]   • NO EXTRACTED DATA found for accessible documents!")
                logger.warning(f"[SQL Query]   • This means documents haven't been processed for data extraction")
            logger.info(f"[SQL Query] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            # Build the base query
            query = self.db.query(DocumentData).join(
                Document, DocumentData.document_id == Document.id
            ).filter(
                DocumentData.document_id.in_(accessible_doc_ids)
            )
            logger.info(f"[SQL Query] Base query built: filtering to {len(accessible_doc_ids)} accessible documents")

            # Apply schema type filter
            if schema_types:
                query = query.filter(DocumentData.schema_type.in_(schema_types))
                logger.info(f"[SQL Query] Applied schema filter: {schema_types}")

            # Apply time range filter (using header_data date fields)
            if time_range:
                query = self._apply_time_filter(query, time_range)
                logger.info(f"[SQL Query] Applied time range filter: {time_range}")

            # Apply entity filter
            if entities:
                query = self._apply_entity_filter(query, entities)
                logger.info(f"[SQL Query] Applied entity filter: {entities}")

            # Apply additional filters
            if filters:
                query = self._apply_jsonb_filters(query, filters)
                logger.info(f"[SQL Query] Applied JSONB filters: {filters}")

            # Execute query
            logger.info(f"[SQL Query] Executing query...")
            results = query.limit(limit).all()
            logger.info(f"[SQL Query] Query returned {len(results)} results")

            # Log details of results
            if results:
                logger.info(f"[SQL Query] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                logger.info(f"[SQL Query] QUERY RESULTS DETAILS:")
                for i, doc_data in enumerate(results[:5]):  # Log first 5
                    header = doc_data.header_data or {}
                    logger.info(f"[SQL Query]   [{i+1}] schema='{doc_data.schema_type}', header_data keys: {list(header.keys())}")
                    # Log key fields
                    for field in ['total_amount', 'total', 'amount', 'invoice_date', 'receipt_date', 'date', 'vendor_name', 'merchant_name']:
                        if field in header:
                            logger.info(f"[SQL Query]       • {field}: {header[field]}")
                if len(results) > 5:
                    logger.info(f"[SQL Query]   ... and {len(results) - 5} more results")
                logger.info(f"[SQL Query] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            # Process results based on metrics and grouping
            logger.info(f"[SQL Query] Processing results with metrics={metrics}, group_by={group_by}")
            processed_data = self._process_results(
                results,
                metrics=metrics or ['total_amount', 'count'],
                group_by=group_by
            )

            logger.info(f"[SQL Query] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"[SQL Query] FINAL SUMMARY:")
            logger.info(f"[SQL Query]   • total_records: {len(results)}")
            logger.info(f"[SQL Query]   • processed rows: {len(processed_data.get('rows', []))}")
            logger.info(f"[SQL Query]   • summary: {processed_data.get('summary', {})}")
            logger.info(f"[SQL Query] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            return {
                "data": processed_data["rows"],
                "summary": processed_data["summary"],
                "metadata": {
                    "total_records": len(results),
                    "schema_types": schema_types,
                    "time_range": time_range,
                    "group_by": group_by,
                    "query_time": datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Analytics query failed: {e}")
            return {
                "data": [],
                "summary": {"error": str(e)},
                "metadata": {"error": str(e)}
            }

    def _apply_time_filter(self, query, time_range: Dict[str, str]):
        """Apply time range filter based on document dates in header_data."""
        start_date = time_range.get('start')
        end_date = time_range.get('end')

        if start_date:
            # Check multiple possible date fields in header_data
            date_fields = ['invoice_date', 'receipt_date', 'transaction_date', 'date', 'statement_date']
            date_conditions = []
            for field in date_fields:
                date_conditions.append(
                    DocumentData.header_data[field].astext >= start_date
                )
            if date_conditions:
                query = query.filter(or_(*date_conditions))

        if end_date:
            date_fields = ['invoice_date', 'receipt_date', 'transaction_date', 'date', 'statement_date']
            date_conditions = []
            for field in date_fields:
                date_conditions.append(
                    DocumentData.header_data[field].astext <= end_date
                )
            if date_conditions:
                query = query.filter(or_(*date_conditions))

        return query

    def _apply_entity_filter(self, query, entities: List[str]):
        """Apply entity name filter (vendor, customer, etc.)."""
        entity_conditions = []
        entity_fields = ['vendor_name', 'customer_name', 'merchant_name', 'company_name', 'payee', 'payer']

        for entity in entities:
            entity_lower = entity.lower()
            for field in entity_fields:
                # Case-insensitive partial match
                entity_conditions.append(
                    func.lower(DocumentData.header_data[field].astext).contains(entity_lower)
                )

        if entity_conditions:
            query = query.filter(or_(*entity_conditions))

        return query

    def _apply_jsonb_filters(self, query, filters: Dict[str, Any]):
        """Apply additional JSONB field filters."""
        for field_path, value in filters.items():
            if isinstance(value, dict):
                # Range filter: {"total_amount": {"min": 10, "max": 100}}
                if 'min' in value:
                    query = query.filter(
                        cast(DocumentData.header_data[field_path].astext, Float) >= value['min']
                    )
                if 'max' in value:
                    query = query.filter(
                        cast(DocumentData.header_data[field_path].astext, Float) <= value['max']
                    )
            else:
                # Exact match
                query = query.filter(
                    DocumentData.header_data[field_path].astext == str(value)
                )

        return query

    def _process_results(
        self,
        results: List[DocumentData],
        metrics: List[str],
        group_by: Optional[str]
    ) -> Dict[str, Any]:
        """Process query results into structured data with aggregations."""
        if not results:
            return {"rows": [], "summary": {"total_records": 0}}

        # Extract data from each document
        # For documents with line_items (spreadsheets), expand each line item as a row
        rows = []
        for doc_data in results:
            # Check if this document has line items that should be expanded
            if doc_data.line_items and isinstance(doc_data.line_items, list) and len(doc_data.line_items) > 0:
                # Check if line_items are actual data rows (dicts) vs just column names (strings)
                first_item = doc_data.line_items[0]
                if isinstance(first_item, dict):
                    # Expand line items into individual rows for aggregation
                    for item in doc_data.line_items:
                        row = self._extract_line_item_data(doc_data, item)
                        rows.append(row)
                    logger.info(f"[SQL Query] Expanded {len(doc_data.line_items)} line items from document {doc_data.document_id}")
                else:
                    # Line items are column headers, use document-level data
                    row = self._extract_row_data(doc_data)
                    rows.append(row)
            else:
                # No line items, use document-level data
                row = self._extract_row_data(doc_data)
                rows.append(row)

        logger.info(f"[SQL Query] Total rows for aggregation: {len(rows)}")

        # Apply grouping if specified
        if group_by:
            grouped_data = self._group_data(rows, group_by, metrics)
            return grouped_data

        # Calculate summary metrics
        summary = self._calculate_summary(rows, metrics)

        return {
            "rows": rows,
            "summary": summary
        }

    def _extract_row_data(self, doc_data: DocumentData) -> Dict[str, Any]:
        """Extract relevant fields from DocumentData."""
        header = doc_data.header_data or {}
        summary = doc_data.summary_data or {}

        # Find the date field
        date_value = None
        for date_field in ['invoice_date', 'receipt_date', 'transaction_date', 'date', 'statement_date']:
            if date_field in header and header[date_field]:
                date_value = header[date_field]
                break

        # Find the amount field
        amount_value = None
        for amount_field in ['total_amount', 'total', 'amount', 'grand_total']:
            if amount_field in header and header[amount_field] is not None:
                amount_value = self._parse_number(header[amount_field])
                break
            if amount_field in summary and summary[amount_field] is not None:
                amount_value = self._parse_number(summary[amount_field])
                break

        # Find entity name
        entity_name = None
        for entity_field in ['vendor_name', 'merchant_name', 'customer_name', 'company_name']:
            if entity_field in header and header[entity_field]:
                entity_name = header[entity_field]
                break

        return {
            "document_id": str(doc_data.document_id),
            "schema_type": doc_data.schema_type,
            "date": date_value,
            "amount": amount_value,
            "entity_name": entity_name,
            "header_data": header,
            "summary_data": summary,
            "line_items_count": doc_data.line_items_count or 0
        }

    def _extract_line_item_data(self, doc_data: DocumentData, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant fields from a line item using dynamic field mappings.

        This method uses the field_mappings stored during extraction to identify
        which columns contain dates, amounts, categories, etc. - making it
        adaptable to any spreadsheet structure.

        Args:
            doc_data: Parent DocumentData
            item: Line item dictionary

        Returns:
            Normalized row data for aggregation
        """
        # Get field mappings from header_data (generated during extraction)
        header_data = doc_data.header_data or {}
        field_mappings = header_data.get("field_mappings", {})

        # If we have dynamic field mappings, use them
        if field_mappings:
            return self._extract_using_field_mappings(doc_data, item, field_mappings)

        # Fallback to hardcoded mappings for backwards compatibility
        return self._extract_using_hardcoded_mappings(doc_data, item)

    def _extract_using_field_mappings(
        self,
        doc_data: DocumentData,
        item: Dict[str, Any],
        field_mappings: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract data using dynamic field mappings from the stored schema.

        Args:
            doc_data: Parent DocumentData
            item: Line item dictionary
            field_mappings: Dict mapping column names to semantic types

        Returns:
            Normalized row data
        """
        result = {
            "document_id": str(doc_data.document_id),
            "schema_type": doc_data.schema_type,
            "raw_item": item
        }

        # Map semantic types to output fields
        semantic_to_output = {
            "date": "date",
            "amount": "amount",
            "quantity": "quantity",
            "category": "category",
            "entity": "entity_name",
            "product": "product",
            "status": "status",
            "region": "region",
            "method": "payment_method",
            "person": "sales_rep"
        }

        # Extract values based on semantic types
        for column_name, mapping in field_mappings.items():
            semantic_type = mapping.get("semantic_type", "unknown")
            output_field = semantic_to_output.get(semantic_type)

            if output_field and column_name in item:
                value = item[column_name]
                if value is not None:
                    # Convert based on data type
                    if mapping.get("data_type") == "number":
                        result[output_field] = self._parse_number(value)
                    elif mapping.get("data_type") == "datetime":
                        result[output_field] = str(value) if value else None
                    else:
                        result[output_field] = str(value) if value else None

        # Store all groupable fields for flexible grouping
        result["groupable_fields"] = {
            col: str(item.get(col)) if item.get(col) else None
            for col, mapping in field_mappings.items()
            if mapping.get("aggregation") == "group_by"
        }

        # Store all summable fields for flexible aggregation
        result["summable_fields"] = {
            col: self._parse_number(item.get(col))
            for col, mapping in field_mappings.items()
            if mapping.get("aggregation") == "sum" and item.get(col) is not None
        }

        return result

    def _extract_using_hardcoded_mappings(
        self,
        doc_data: DocumentData,
        item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fallback extraction using hardcoded field name patterns.
        Used for backwards compatibility with data extracted before field_mappings.

        Args:
            doc_data: Parent DocumentData
            item: Line item dictionary

        Returns:
            Normalized row data for aggregation
        """
        # Common field name mappings for different column naming conventions
        date_fields = ['Purchase Date', 'purchase_date', 'Date', 'date', 'Order Date', 'order_date',
                       'Transaction Date', 'transaction_date', 'Invoice Date', 'invoice_date']
        amount_fields = ['Total Sales', 'total_sales', 'Total', 'total', 'Amount', 'amount',
                        'Total Amount', 'total_amount', 'Grand Total', 'grand_total', 'Price', 'price']
        category_fields = ['Category', 'category', 'Type', 'type', 'Product Category', 'product_category']
        entity_fields = ['Customer Name', 'customer_name', 'Vendor', 'vendor', 'Customer', 'customer',
                        'Sales Rep', 'sales_rep', 'Merchant', 'merchant']
        product_fields = ['Product', 'product', 'Item', 'item', 'Description', 'description',
                         'Product Name', 'product_name']
        quantity_fields = ['Quantity', 'quantity', 'Qty', 'qty', 'Count', 'count']
        status_fields = ['Status', 'status', 'Order Status', 'order_status']
        region_fields = ['Region', 'region', 'Area', 'area', 'Location', 'location']

        # Extract date
        date_value = None
        for field in date_fields:
            if field in item and item[field]:
                date_value = str(item[field])
                break

        # Extract amount
        amount_value = None
        for field in amount_fields:
            if field in item and item[field] is not None:
                amount_value = self._parse_number(item[field])
                break

        # Extract category
        category_value = None
        for field in category_fields:
            if field in item and item[field]:
                category_value = str(item[field])
                break

        # Extract entity name
        entity_name = None
        for field in entity_fields:
            if field in item and item[field]:
                entity_name = str(item[field])
                break

        # Extract product name
        product_name = None
        for field in product_fields:
            if field in item and item[field]:
                product_name = str(item[field])
                break

        # Extract quantity
        quantity_value = None
        for field in quantity_fields:
            if field in item and item[field] is not None:
                quantity_value = self._parse_number(item[field])
                break

        # Extract status
        status_value = None
        for field in status_fields:
            if field in item and item[field]:
                status_value = str(item[field])
                break

        # Extract region
        region_value = None
        for field in region_fields:
            if field in item and item[field]:
                region_value = str(item[field])
                break

        return {
            "document_id": str(doc_data.document_id),
            "schema_type": doc_data.schema_type,
            "date": date_value,
            "amount": amount_value,
            "category": category_value,
            "entity_name": entity_name,
            "product": product_name,
            "quantity": quantity_value,
            "status": status_value,
            "region": region_value,
            "raw_item": item  # Keep original for advanced queries
        }

    def _parse_number(self, value) -> Optional[float]:
        """Parse a number from various formats."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, str):
            # Remove currency symbols and commas
            cleaned = re.sub(r'[,$€£¥]', '', value.strip())
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    def _group_data(
        self,
        rows: List[Dict],
        group_by: str,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Group data by specified dimension.

        Supports:
        - 'monthly': Group by YYYY-MM
        - 'quarterly': Group by YYYY-Q#
        - 'yearly': Group by YYYY
        - 'category': Group by category field
        - 'vendor', 'entity', 'customer': Group by entity name
        - 'product': Group by product name
        - 'region': Group by region
        - 'status': Group by status
        - 'schema_type': Group by document type
        - 'year_category': Multi-level grouping by year then category
        - 'category_year': Multi-level grouping by category then year
        """
        # Handle multi-level grouping
        if group_by == 'year_category':
            return self._group_data_multi_level(rows, ['yearly', 'category'], metrics)
        elif group_by == 'category_year':
            return self._group_data_multi_level(rows, ['category', 'yearly'], metrics)
        elif group_by == 'year_month_category':
            return self._group_data_three_level(rows, ['yearly', 'monthly', 'category'], metrics)
        elif group_by == 'year_category_month':
            return self._group_data_three_level(rows, ['yearly', 'category', 'monthly'], metrics)

        groups = {}

        for row in rows:
            # Determine group key
            key = self._get_group_key(row, group_by)

            if key not in groups:
                groups[key] = []
            groups[key].append(row)

        # Calculate metrics for each group
        grouped_rows = []
        for key, group_rows in sorted(groups.items()):
            group_summary = self._calculate_summary(group_rows, metrics)
            grouped_rows.append({
                "group": key,
                "count": len(group_rows),
                **group_summary
            })

        # Overall summary
        overall_summary = self._calculate_summary(rows, metrics)
        overall_summary["total_groups"] = len(groups)

        return {
            "rows": grouped_rows,
            "summary": overall_summary
        }

    def _get_group_key(self, row: Dict, group_by: str) -> str:
        """Get the group key for a row based on grouping type."""
        if group_by == 'monthly':
            return self._get_month_key(row.get('date'))
        elif group_by == 'quarterly':
            return self._get_quarter_key(row.get('date'))
        elif group_by == 'yearly':
            return self._get_year_key(row.get('date'))
        elif group_by == 'category':
            return row.get('category') or 'Unknown'
        elif group_by in ['vendor', 'entity', 'customer']:
            return row.get('entity_name') or 'Unknown'
        elif group_by == 'product':
            return row.get('product') or 'Unknown'
        elif group_by == 'region':
            return row.get('region') or 'Unknown'
        elif group_by == 'status':
            return row.get('status') or 'Unknown'
        elif group_by == 'schema_type':
            return row.get('schema_type') or 'Unknown'
        else:
            return 'All'

    def _group_data_multi_level(
        self,
        rows: List[Dict],
        levels: List[str],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Group data by multiple levels (e.g., year -> category).

        Args:
            rows: List of data rows
            levels: List of grouping levels (e.g., ['yearly', 'category'])
            metrics: Metrics to calculate

        Returns:
            Nested grouped data structure
        """
        if len(levels) < 2:
            return self._group_data(rows, levels[0] if levels else 'All', metrics)

        primary_level = levels[0]
        secondary_level = levels[1]

        # First level grouping
        primary_groups = {}
        for row in rows:
            key = self._get_group_key(row, primary_level)
            if key not in primary_groups:
                primary_groups[key] = []
            primary_groups[key].append(row)

        # Build hierarchical result
        grouped_rows = []
        category_totals = {}  # Track totals by category across all years

        for primary_key in sorted(primary_groups.keys()):
            primary_rows = primary_groups[primary_key]

            # Second level grouping within primary group
            secondary_groups = {}
            for row in primary_rows:
                sec_key = self._get_group_key(row, secondary_level)
                if sec_key not in secondary_groups:
                    secondary_groups[sec_key] = []
                secondary_groups[sec_key].append(row)

                # Track category totals
                if sec_key not in category_totals:
                    category_totals[sec_key] = []
                category_totals[sec_key].append(row)

            # Build secondary level data
            secondary_data = []
            for sec_key in sorted(secondary_groups.keys()):
                sec_rows = secondary_groups[sec_key]
                sec_summary = self._calculate_summary(sec_rows, metrics)
                secondary_data.append({
                    "group": sec_key,
                    "count": len(sec_rows),
                    **sec_summary
                })

            # Calculate primary level summary
            primary_summary = self._calculate_summary(primary_rows, metrics)

            grouped_rows.append({
                "group": primary_key,
                "count": len(primary_rows),
                "sub_groups": secondary_data,
                **primary_summary
            })

        # Calculate overall summary
        overall_summary = self._calculate_summary(rows, metrics)
        overall_summary["total_primary_groups"] = len(primary_groups)
        overall_summary["total_secondary_groups"] = len(category_totals)

        # Add category totals summary
        category_summary = []
        for cat_key in sorted(category_totals.keys()):
            cat_rows = category_totals[cat_key]
            cat_summary = self._calculate_summary(cat_rows, metrics)
            category_summary.append({
                "group": cat_key,
                "count": len(cat_rows),
                **cat_summary
            })

        return {
            "rows": grouped_rows,
            "summary": overall_summary,
            "category_totals": category_summary  # Summary by secondary grouping
        }

    def _get_month_key(self, date_str: Optional[str]) -> str:
        """Extract month key (YYYY-MM) from date string."""
        if not date_str:
            return 'Unknown'
        try:
            # Handle various date formats
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    dt = datetime.strptime(date_str[:10], fmt)
                    return dt.strftime('%Y-%m')
                except ValueError:
                    continue
            return date_str[:7] if len(date_str) >= 7 else 'Unknown'
        except Exception:
            return 'Unknown'

    def _get_quarter_key(self, date_str: Optional[str]) -> str:
        """Extract quarter key (YYYY-Q#) from date string."""
        if not date_str:
            return 'Unknown'
        try:
            month_key = self._get_month_key(date_str)
            if month_key == 'Unknown':
                return 'Unknown'
            year, month = month_key.split('-')
            quarter = (int(month) - 1) // 3 + 1
            return f"{year}-Q{quarter}"
        except Exception:
            return 'Unknown'

    def _get_year_key(self, date_str: Optional[str]) -> str:
        """Extract year key (YYYY) from date string."""
        if not date_str:
            return 'Unknown'
        try:
            month_key = self._get_month_key(date_str)
            if month_key == 'Unknown':
                return 'Unknown'
            return month_key.split('-')[0]
        except Exception:
            return 'Unknown'

    def _calculate_summary(self, rows: List[Dict], metrics: List[str]) -> Dict[str, Any]:
        """Calculate summary metrics for a set of rows."""
        summary = {
            "total_records": len(rows)
        }

        amounts = [r.get('amount') for r in rows if r.get('amount') is not None]

        if 'total_amount' in metrics or 'sum' in metrics:
            summary['total_amount'] = sum(amounts) if amounts else 0

        if 'count' in metrics:
            summary['count'] = len(rows)

        if 'avg_amount' in metrics or 'average' in metrics:
            summary['avg_amount'] = sum(amounts) / len(amounts) if amounts else 0

        if 'min_amount' in metrics or 'min' in metrics:
            summary['min_amount'] = min(amounts) if amounts else 0

        if 'max_amount' in metrics or 'max' in metrics:
            summary['max_amount'] = max(amounts) if amounts else 0

        return summary

    def get_available_data_summary(
        self,
        accessible_doc_ids: List[UUID]
    ) -> Dict[str, Any]:
        """Get summary of available extracted data for user."""
        if not accessible_doc_ids:
            return {"schemas": {}, "total_documents": 0}

        try:
            # Count documents by schema type
            results = self.db.query(
                DocumentData.schema_type,
                func.count(DocumentData.id).label('count')
            ).filter(
                DocumentData.document_id.in_(accessible_doc_ids)
            ).group_by(
                DocumentData.schema_type
            ).all()

            schemas = {r.schema_type: r.count for r in results}
            total = sum(schemas.values())

            return {
                "schemas": schemas,
                "total_documents": total,
                "available_types": list(schemas.keys())
            }
        except Exception as e:
            logger.error(f"Failed to get data summary: {e}")
            return {"schemas": {}, "total_documents": 0, "error": str(e)}

    def execute_natural_language_query(
        self,
        accessible_doc_ids: List[UUID],
        intent_classification: Dict[str, Any],
        query: str
    ) -> Dict[str, Any]:
        """
        Execute a query based on intent classification results.

        Args:
            accessible_doc_ids: List of document IDs user can access
            intent_classification: Result from IntentClassifier
            query: Original user query

        Returns:
            Query results with data and summary
        """
        # Extract parameters from intent classification
        schema_types = intent_classification.get('suggested_schemas', [])
        time_range = intent_classification.get('detected_time_range')
        entities = intent_classification.get('detected_entities', [])
        metrics = intent_classification.get('detected_metrics', ['total_amount', 'count'])

        # Determine grouping from query
        query_lower = query.lower()
        group_by = None

        # Check for multi-level grouping first (more specific patterns)
        has_year = any(x in query_lower for x in ['by year', 'yearly', 'per year', 'each year', 'group by year'])
        has_category = any(x in query_lower for x in ['by category', 'by categories', 'per category', 'each category', 'group by category'])

        if has_year and has_category:
            # Multi-level grouping: year -> category
            group_by = 'year_category'
            logger.info(f"[SQL Query] Detected multi-level grouping: year_category")
        elif 'monthly' in query_lower or 'by month' in query_lower:
            group_by = 'monthly'
        elif 'quarterly' in query_lower or 'by quarter' in query_lower:
            group_by = 'quarterly'
        elif has_year:
            group_by = 'yearly'
        elif has_category:
            group_by = 'category'
        elif 'by vendor' in query_lower or 'per vendor' in query_lower:
            group_by = 'vendor'
        elif 'by customer' in query_lower or 'per customer' in query_lower:
            group_by = 'customer'
        elif 'by product' in query_lower or 'per product' in query_lower:
            group_by = 'product'
        elif 'by region' in query_lower or 'per region' in query_lower:
            group_by = 'region'
        elif 'by status' in query_lower or 'per status' in query_lower:
            group_by = 'status'

        logger.info(f"[SQL Query] Detected grouping: {group_by}")

        # Detect amount filters from query
        filters = {}
        amount_match = re.search(r'over\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', query_lower)
        if amount_match:
            min_amount = float(amount_match.group(1).replace(',', ''))
            filters['total_amount'] = {'min': min_amount}

        amount_match = re.search(r'under\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', query_lower)
        if amount_match:
            max_amount = float(amount_match.group(1).replace(',', ''))
            if 'total_amount' in filters:
                filters['total_amount']['max'] = max_amount
            else:
                filters['total_amount'] = {'max': max_amount}

        logger.info(f"[SQL Executor] Executing analytics query: schemas={schema_types}, time_range={time_range}, entities={entities}, group_by={group_by}, filters={filters}")

        # First try with suggested schemas
        result = self.execute_analytics_query(
            accessible_doc_ids=accessible_doc_ids,
            schema_types=schema_types if schema_types else None,
            time_range=time_range,
            entities=entities if entities else None,
            metrics=metrics,
            group_by=group_by,
            filters=filters if filters else None
        )

        # If no data found and we had schema filters, retry without schema filter
        if result.get('summary', {}).get('total_records', 0) == 0 and schema_types:
            logger.info(f"[SQL Executor] No data found with schemas {schema_types}, retrying without schema filter")
            result = self.execute_analytics_query(
                accessible_doc_ids=accessible_doc_ids,
                schema_types=None,  # Search all schema types
                time_range=time_range,
                entities=entities if entities else None,
                metrics=metrics,
                group_by=group_by,
                filters=filters if filters else None
            )

        return result

    def execute_natural_language_query_with_llm(
        self,
        accessible_doc_ids: List[UUID],
        query: str,
        llm_client=None
    ) -> Dict[str, Any]:
        """
        Execute a natural language query using LLM for intelligent analysis.

        This method uses LLM to:
        1. Analyze the query to understand user intent
        2. Determine which fields to aggregate and group by
        3. Generate appropriate query parameters

        Args:
            accessible_doc_ids: List of document IDs user can access
            query: User's natural language query
            llm_client: LLM client for query analysis

        Returns:
            Query results with data and summary
        """
        # First, get field mappings from available documents
        field_mappings = self._get_available_field_mappings(accessible_doc_ids)

        if not field_mappings:
            logger.warning("[LLM Query] No field mappings available, falling back to standard query")
            return self.execute_natural_language_query(
                accessible_doc_ids=accessible_doc_ids,
                intent_classification={},
                query=query
            )

        # Use LLM to analyze the query
        if llm_client:
            try:
                from analytics_service.schema_analyzer import analyze_user_query
                query_analysis = analyze_user_query(query, field_mappings, llm_client)
                logger.info(f"[LLM Query] Analysis result: {query_analysis}")
            except Exception as e:
                logger.warning(f"[LLM Query] LLM analysis failed, using heuristics: {e}")
                query_analysis = self._analyze_query_heuristically(query, field_mappings)
        else:
            query_analysis = self._analyze_query_heuristically(query, field_mappings)

        # Map analysis results to query parameters
        group_by = self._map_query_analysis_to_grouping(query_analysis, field_mappings)
        metrics = ['total_amount', 'count']

        logger.info(f"[LLM Query] Executing with group_by={group_by}, explanation={query_analysis.get('explanation', '')}")

        # Execute the query
        result = self.execute_analytics_query(
            accessible_doc_ids=accessible_doc_ids,
            schema_types=None,
            metrics=metrics,
            group_by=group_by,
            filters=query_analysis.get('filter_conditions', {}) or None
        )

        # Add analysis metadata to result
        result['query_analysis'] = query_analysis

        return result

    def execute_dynamic_sql_query(
        self,
        accessible_doc_ids: List[UUID],
        query: str,
        llm_client=None,
        max_correction_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Execute a natural language query using LLM-generated dynamic SQL.

        This method:
        1. Gets the data schema from stored field mappings
        2. Uses LLM to generate appropriate SQL based on user query and schema
        3. Executes the generated SQL directly against the database
        4. If SQL fails, uses LLM to analyze the error and regenerate corrected SQL
        5. Retries up to max_correction_attempts times
        6. Returns formatted results with hierarchical summaries

        Args:
            accessible_doc_ids: List of document IDs user can access
            query: User's natural language query
            llm_client: LLM client for SQL generation and error correction
            max_correction_attempts: Maximum number of LLM correction attempts (default: 3)

        Returns:
            Query results with data, summaries, and generated SQL
        """
        from analytics_service.llm_sql_generator import LLMSQLGenerator

        # Get field mappings
        field_mappings = self._get_available_field_mappings(accessible_doc_ids)

        if not field_mappings:
            logger.warning("[Dynamic SQL] No field mappings available")
            return {
                "data": [],
                "summary": {"error": "No field mappings available for dynamic SQL generation"},
                "metadata": {"query": query}
            }

        # Build document filter for accessible documents
        doc_ids_str = ", ".join(f"'{str(doc_id)}'" for doc_id in accessible_doc_ids)
        doc_filter = f"dd.document_id IN ({doc_ids_str})"

        # Generate SQL using LLM
        generator = LLMSQLGenerator(llm_client)
        sql_result = generator.generate_sql(query, field_mappings, doc_filter)

        logger.info(f"[Dynamic SQL] Generated SQL:\n{sql_result.sql_query}")
        logger.info(f"[Dynamic SQL] Explanation: {sql_result.explanation}")

        if not sql_result.success or not sql_result.sql_query:
            return {
                "data": [],
                "summary": {"error": sql_result.error or "Failed to generate SQL"},
                "metadata": {"query": query}
            }

        # Execute SQL with retry loop for error correction
        current_sql = sql_result.sql_query
        attempt = 0
        last_error = None
        correction_history = []  # Track all attempts for debugging

        while attempt <= max_correction_attempts:
            try:
                # Execute the SQL
                logger.info(f"[Dynamic SQL] Executing SQL (attempt {attempt + 1}/{max_correction_attempts + 1})...")
                result = self.db.execute(text(current_sql))
                rows = result.fetchall()
                columns = result.keys()

                # Convert to list of dicts
                data = [dict(zip(columns, row)) for row in rows]

                logger.info(f"[Dynamic SQL] Query returned {len(data)} rows")

                # Generate summary report
                summary_report = generator.generate_summary_report(query, data, field_mappings)

                return {
                    "data": data,
                    "summary": summary_report,
                    "metadata": {
                        "query": query,
                        "generated_sql": current_sql,
                        "explanation": sql_result.explanation,
                        "grouping_fields": sql_result.grouping_fields,
                        "aggregation_fields": sql_result.aggregation_fields,
                        "time_granularity": sql_result.time_granularity,
                        "row_count": len(data),
                        "correction_attempts": attempt,
                        "correction_history": correction_history if correction_history else None
                    }
                }

            except Exception as e:
                error_message = str(e)
                last_error = error_message
                logger.error(f"[Dynamic SQL] Execution failed (attempt {attempt + 1}): {error_message}")

                # Rollback the transaction to recover from error state
                try:
                    self.db.rollback()
                    logger.info("[Dynamic SQL] Transaction rolled back after error")
                except Exception as rollback_error:
                    logger.warning(f"[Dynamic SQL] Rollback failed: {rollback_error}")

                # Track this attempt
                correction_history.append({
                    "attempt": attempt + 1,
                    "sql": current_sql[:500] + "..." if len(current_sql) > 500 else current_sql,
                    "error": error_message
                })

                # Check if we have more correction attempts
                if attempt < max_correction_attempts and llm_client:
                    logger.info(f"[Dynamic SQL] Attempting LLM-driven SQL correction (attempt {attempt + 1}/{max_correction_attempts})...")

                    # Use LLM to correct the SQL
                    corrected_result = generator.correct_sql_error(
                        user_query=query,
                        failed_sql=current_sql,
                        error_message=error_message,
                        field_mappings=field_mappings,
                        table_filter=doc_filter
                    )

                    if corrected_result and corrected_result.sql_query:
                        current_sql = corrected_result.sql_query
                        logger.info(f"[Dynamic SQL] LLM provided corrected SQL, retrying...")
                        attempt += 1
                        continue
                    else:
                        logger.warning("[Dynamic SQL] LLM could not provide corrected SQL")
                        break
                else:
                    if not llm_client:
                        logger.warning("[Dynamic SQL] No LLM client available for SQL correction")
                    else:
                        logger.warning(f"[Dynamic SQL] Max correction attempts ({max_correction_attempts}) reached")
                    break

        # All attempts failed
        return {
            "data": [],
            "summary": {"error": last_error or "SQL execution failed after all correction attempts"},
            "metadata": {
                "query": query,
                "generated_sql": sql_result.sql_query,
                "final_sql": current_sql,
                "error": last_error,
                "correction_attempts": attempt,
                "correction_history": correction_history
            }
        }

    def _get_available_field_mappings(self, accessible_doc_ids: List[UUID]) -> Dict[str, Dict[str, Any]]:
        """Get field mappings from documents with extracted data."""
        try:
            # Get documents with field_mappings in header_data
            # Use raw SQL to properly query JSONB for field_mappings key
            doc_ids_str = ", ".join(f"'{str(doc_id)}'" for doc_id in accessible_doc_ids)

            result = self.db.execute(text(f"""
                SELECT header_data->'field_mappings' as field_mappings
                FROM documents_data
                WHERE document_id IN ({doc_ids_str})
                AND header_data->'field_mappings' IS NOT NULL
                LIMIT 1
            """))

            row = result.fetchone()
            if row and row[0]:
                logger.info(f"[Field Mappings] Found field mappings with {len(row[0])} fields")
                return dict(row[0])

            # Fallback: try ORM query
            doc_data = self.db.query(DocumentData).filter(
                DocumentData.document_id.in_(accessible_doc_ids),
                DocumentData.header_data.isnot(None)
            ).first()

            if doc_data and doc_data.header_data:
                mappings = doc_data.header_data.get('field_mappings', {})
                if mappings:
                    logger.info(f"[Field Mappings] Found field mappings via ORM with {len(mappings)} fields")
                return mappings

            logger.warning("[Field Mappings] No field mappings found in accessible documents")
            return {}
        except Exception as e:
            logger.error(f"Failed to get field mappings: {e}")
            return {}

    def _analyze_query_heuristically(
        self,
        query: str,
        field_mappings: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fallback heuristic query analysis when LLM is not available."""
        try:
            from analytics_service.schema_analyzer import analyze_user_query
            return analyze_user_query(query, field_mappings, llm_client=None)
        except ImportError:
            # Basic fallback
            query_lower = query.lower()
            return {
                "primary_metric_field": None,
                "aggregation_type": "sum",
                "group_by_fields": [],
                "filter_conditions": {},
                "time_grouping": "yearly" if "year" in query_lower else None,
                "sort_order": "desc",
                "explanation": "Basic query analysis"
            }

    def _map_query_analysis_to_grouping(
        self,
        query_analysis: Dict[str, Any],
        field_mappings: Dict[str, Dict[str, Any]]
    ) -> Optional[str]:
        """Map LLM query analysis to grouping parameters."""
        group_by_fields = query_analysis.get('group_by_fields', [])
        time_grouping = query_analysis.get('time_grouping')

        # Determine grouping based on analysis
        has_time_grouping = time_grouping in ['yearly', 'quarterly', 'monthly']
        has_field_grouping = len(group_by_fields) > 0

        if has_time_grouping and has_field_grouping:
            # Multi-level grouping
            # Find the semantic type of the first group_by field
            first_group_field = group_by_fields[0]
            for field_name, mapping in field_mappings.items():
                if field_name == first_group_field or first_group_field.lower() in field_name.lower():
                    semantic_type = mapping.get('semantic_type', '')
                    if semantic_type == 'category':
                        return 'year_category'
                    elif semantic_type == 'region':
                        return 'year_category'  # Can be extended
                    elif semantic_type == 'status':
                        return 'year_category'
            return 'year_category'  # Default multi-level
        elif has_time_grouping:
            return time_grouping
        elif has_field_grouping:
            # Find semantic type of the group_by field
            first_group_field = group_by_fields[0]
            for field_name, mapping in field_mappings.items():
                if field_name == first_group_field or first_group_field.lower() in field_name.lower():
                    semantic_type = mapping.get('semantic_type', '')
                    if semantic_type in ['category', 'status', 'region', 'product', 'entity', 'person', 'method']:
                        return semantic_type
            return 'category'  # Default
        else:
            return None
