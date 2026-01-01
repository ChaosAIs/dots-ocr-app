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
    - Multi-schema queries (independent processing without merge)
    - LLM-driven field mapping lookup with priority-based caching
    """

    def __init__(self, db: Session, schema_service=None, llm_client=None):
        """
        Initialize the SQL query executor.

        Args:
            db: SQLAlchemy database session
            schema_service: Optional SchemaService for schema lookup
            llm_client: Optional LLM client for dynamic schema inference
        """
        self.db = db
        self.schema_service = schema_service
        self.llm_client = llm_client

    def _get_document_sources(self, doc_ids: List[UUID]) -> List[Dict[str, str]]:
        """
        Get document source names for the given document IDs.

        Args:
            doc_ids: List of document UUIDs

        Returns:
            List of dicts with 'document_id' and 'filename'
        """
        if not doc_ids:
            return []

        try:
            doc_ids_str = ", ".join(f"'{str(doc_id)}'" for doc_id in doc_ids)
            result = self.db.execute(text(f"""
                SELECT id, original_filename
                FROM documents
                WHERE id IN ({doc_ids_str})
            """))

            sources = []
            for row in result.fetchall():
                sources.append({
                    "document_id": str(row[0]),
                    "filename": row[1] or "Unknown"
                })
            return sources
        except Exception as e:
            logger.warning(f"[SQL Executor] Failed to get document sources: {e}")
            return []

    def _get_document_sources_from_results(self, data: List[Dict[str, Any]], sql_query: str) -> List[Dict[str, str]]:
        """
        Extract document sources from query results by modifying the SQL to include document_id.

        This runs a modified version of the query that extracts DISTINCT document_ids
        from the actual result set, then fetches the corresponding filenames.

        Args:
            data: The query result data (not used directly, but indicates query succeeded)
            sql_query: The generated SQL query

        Returns:
            List of dicts with 'document_id' and 'filename' for documents actually used
        """
        if not sql_query or not data:
            return []

        try:
            # The SQL typically has a CTE like:
            # WITH expanded_items AS (
            #     SELECT dd.header_data, li.data as item
            #     FROM documents_data dd
            #     JOIN documents_data_line_items li ON ...
            #     WHERE dd.document_id IN (...)
            # )
            # SELECT ... FROM expanded_items WHERE ...
            #
            # We need to rebuild it with dd.document_id included and extract unique values

            if 'WITH expanded_items AS' in sql_query or 'WITH items AS' in sql_query:
                # Extract document_id filter from the CTE's WHERE clause
                doc_filter_match = re.search(r"dd\.document_id\s+IN\s*\(([^)]+)\)", sql_query, re.IGNORECASE)

                if doc_filter_match:
                    doc_ids_in_filter = doc_filter_match.group(1)

                    # Find the outer WHERE clause (after the CTE ends with ")")
                    # Look for the pattern: ) SELECT ... FROM expanded_items WHERE ...
                    outer_where_match = re.search(
                        r'\)\s*SELECT[^W]+WHERE\s+(.+?)(?:ORDER BY|GROUP BY|LIMIT|;|$)',
                        sql_query,
                        re.IGNORECASE | re.DOTALL
                    )

                    if outer_where_match:
                        where_clause = outer_where_match.group(1).strip().rstrip(';')

                        # Build query to get document_ids from records matching the filter
                        doc_id_query = f"""
                            WITH expanded_items AS (
                                SELECT dd.document_id, dd.header_data, li.data as item
                                FROM documents_data dd
                                JOIN documents_data_line_items li ON li.documents_data_id = dd.id
                                WHERE dd.document_id IN ({doc_ids_in_filter})
                            )
                            SELECT DISTINCT document_id FROM expanded_items WHERE {where_clause}
                        """
                    else:
                        # No outer WHERE clause, get all document_ids from the CTE
                        doc_id_query = f"""
                            WITH expanded_items AS (
                                SELECT dd.document_id, dd.header_data, li.data as item
                                FROM documents_data dd
                                JOIN documents_data_line_items li ON li.documents_data_id = dd.id
                                WHERE dd.document_id IN ({doc_ids_in_filter})
                            )
                            SELECT DISTINCT document_id FROM expanded_items
                        """

                    # Execute to get document IDs
                    try:
                        logger.info(f"[SQL Executor] Extracting document sources with query:\n{doc_id_query[:300]}...")
                        result = self.db.execute(text(doc_id_query))
                        doc_ids = [str(row[0]) for row in result.fetchall()]
                        logger.info(f"[SQL Executor] Found {len(doc_ids)} unique document(s) in results")

                        if doc_ids:
                            # Get filenames for these document IDs
                            doc_ids_str = ", ".join(f"'{doc_id}'" for doc_id in doc_ids)
                            filename_result = self.db.execute(text(f"""
                                SELECT id, original_filename
                                FROM documents
                                WHERE id IN ({doc_ids_str})
                            """))

                            sources = []
                            for row in filename_result.fetchall():
                                sources.append({
                                    "document_id": str(row[0]),
                                    "filename": row[1] or "Unknown"
                                })
                            logger.info(f"[SQL Executor] Document sources: {[s['filename'] for s in sources]}")
                            return sources
                    except Exception as inner_e:
                        logger.warning(f"[SQL Executor] Failed to execute document_id extraction query: {inner_e}")
                        # Try rollback in case of error
                        try:
                            self.db.rollback()
                        except:
                            pass

            return []
        except Exception as e:
            logger.warning(f"[SQL Executor] Failed to extract document sources from results: {e}")
            return []

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
        # NOTE: As of migration 022, line items are stored externally in documents_data_line_items table
        rows = []
        for doc_data in results:
            # Check if this document has line items that should be expanded
            if doc_data.line_items_count and doc_data.line_items_count > 0:
                # Fetch line items from external storage
                line_items = self._fetch_line_items_from_external_storage(doc_data.id)
                if line_items:
                    # Check if line_items are actual data rows (dicts) vs just column names (strings)
                    first_item = line_items[0] if line_items else None
                    if first_item and isinstance(first_item, dict):
                        # Expand line items into individual rows for aggregation
                        for item in line_items:
                            row = self._extract_line_item_data(doc_data, item)
                            rows.append(row)
                        logger.info(f"[SQL Query] Expanded {len(line_items)} line items from document {doc_data.document_id}")
                    else:
                        # Line items are column headers, use document-level data
                        row = self._extract_row_data(doc_data)
                        rows.append(row)
                else:
                    # No line items found, use document-level data
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

    def _fetch_line_items_from_external_storage(self, documents_data_id) -> List[Dict[str, Any]]:
        """
        Fetch line items from external storage (documents_data_line_items table).

        As of migration 022, all line items are stored externally.

        Args:
            documents_data_id: UUID of the documents_data record

        Returns:
            List of line item dictionaries
        """
        try:
            from db.models import DocumentDataLineItem

            line_items = self.db.query(DocumentDataLineItem).filter(
                DocumentDataLineItem.documents_data_id == documents_data_id
            ).order_by(DocumentDataLineItem.line_number).all()

            return [item.data for item in line_items]

        except Exception as e:
            logger.error(f"[SQL Query] Failed to fetch line items from external storage: {e}")
            return []

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

        # If no date in line item, inherit from parent document's header_data
        if not result.get("date") and doc_data.header_data:
            header = doc_data.header_data
            date_fields = ['transaction_date', 'invoice_date', 'receipt_date', 'date', 'statement_date']
            for field in date_fields:
                if field in header and header[field]:
                    result["date"] = str(header[field])
                    break

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

        # Extract date - first try from line item, then fallback to header_data
        date_value = None
        for field in date_fields:
            if field in item and item[field]:
                date_value = str(item[field])
                break

        # If no date in line item, inherit from parent document's header_data
        if not date_value and doc_data.header_data:
            header = doc_data.header_data
            for field in date_fields:
                if field in header and header[field]:
                    date_value = str(header[field])
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

        # Check if user wants detailed listing - if so, don't group even if they mention "by month"
        detail_keywords = ['details', 'detail', 'list all', 'show all', 'all items', 'individual',
                           'each item', 'every item', 'itemized', 'line items', 'specific']
        wants_details = any(keyword in query_lower for keyword in detail_keywords)

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

                # Get document sources from actual query results (not just accessible docs)
                document_sources = self._get_document_sources_from_results(data, current_sql)
                # Fallback to accessible_doc_ids if extraction failed
                if not document_sources:
                    document_sources = self._get_document_sources(accessible_doc_ids)

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
                        "correction_history": correction_history if correction_history else None,
                        "document_sources": document_sources
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

        # All attempts failed (sync version)
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

    async def _execute_min_max_queries_async(
        self,
        generator,
        query: str,
        field_mappings: Dict[str, Dict[str, Any]],
        doc_filter: str,
        storage_type: str,
        send_progress,
        accessible_doc_ids: List[UUID] = None
    ) -> Dict[str, Any]:
        """
        Execute separate MIN and MAX queries and combine results.

        This method is called when the user asks for both min and max values (e.g.,
        "which products have max inventory and min inventory"). Instead of trying
        to execute one complex UNION ALL query, we run two simple queries and
        combine their results.

        Args:
            generator: LLMSQLGenerator instance
            query: Original user query
            field_mappings: Field mappings for SQL generation
            doc_filter: Document filter WHERE clause
            storage_type: "inline" or "external"
            send_progress: Async progress callback function
            accessible_doc_ids: List of document IDs used in the query (for source tracking)

        Returns:
            Combined query results with both MIN and MAX data
        """
        logger.info("[Dynamic SQL] Executing separate MIN/MAX queries")

        # Generate separate MIN and MAX queries
        min_result, max_result = generator.generate_min_max_separate_queries(
            query, field_mappings, doc_filter, storage_type
        )

        combined_data = []
        min_data = []
        max_data = []
        min_sql = None
        max_sql = None
        errors = []

        # Execute MIN query
        if min_result and min_result.success and min_result.sql_query:
            await send_progress("Running MIN query...")
            min_sql = min_result.sql_query
            logger.info(f"[Dynamic SQL] Executing MIN query:\n{min_sql[:200]}...")
            try:
                result = self.db.execute(text(min_sql))
                rows = result.fetchall()
                columns = result.keys()
                min_data = [dict(zip(columns, row)) for row in rows]
                # Add record_type marker to each row
                for row in min_data:
                    row['record_type'] = 'MIN'
                logger.info(f"[Dynamic SQL] MIN query returned {len(min_data)} rows")
            except Exception as e:
                logger.error(f"[Dynamic SQL] MIN query failed: {e}")
                errors.append(f"MIN query error: {str(e)}")
                # Rollback on error
                try:
                    self.db.rollback()
                except:
                    pass
        else:
            errors.append("Failed to generate MIN query")

        # Execute MAX query
        if max_result and max_result.success and max_result.sql_query:
            await send_progress("Running MAX query...")
            max_sql = max_result.sql_query
            logger.info(f"[Dynamic SQL] Executing MAX query:\n{max_sql[:200]}...")
            try:
                result = self.db.execute(text(max_sql))
                rows = result.fetchall()
                columns = result.keys()
                max_data = [dict(zip(columns, row)) for row in rows]
                # Add record_type marker to each row
                for row in max_data:
                    row['record_type'] = 'MAX'
                logger.info(f"[Dynamic SQL] MAX query returned {len(max_data)} rows")
            except Exception as e:
                logger.error(f"[Dynamic SQL] MAX query failed: {e}")
                errors.append(f"MAX query error: {str(e)}")
                # Rollback on error
                try:
                    self.db.rollback()
                except:
                    pass
        else:
            errors.append("Failed to generate MAX query")

        # Combine results (MIN records first, then MAX records)
        combined_data = min_data + max_data

        if not combined_data and errors:
            return {
                "data": [],
                "summary": {"error": "; ".join(errors)},
                "metadata": {"query": query}
            }

        # Calculate stats
        await send_progress(f"Found {len(combined_data)} records, preparing results...")

        # Find amount column for stats
        amount_col = None
        if combined_data:
            for col in combined_data[0].keys():
                col_lower = col.lower()
                if any(kw in col_lower for kw in ['amount', 'total', 'sales', 'sum', 'price', 'value', 'inventory', 'quantity']):
                    amount_col = col
                    break

        # Calculate min/max values
        min_value = None
        max_value = None
        if amount_col:
            if min_data:
                min_value = min(float(r.get(amount_col, 0) or 0) for r in min_data)
            if max_data:
                max_value = max(float(r.get(amount_col, 0) or 0) for r in max_data)

        # Get document sources from actual query results (not just accessible docs)
        document_sources = []
        # Try to extract from MIN data/SQL first, then MAX data/SQL
        if min_data and min_sql:
            document_sources = self._get_document_sources_from_results(min_data, min_sql)
        if not document_sources and max_data and max_sql:
            document_sources = self._get_document_sources_from_results(max_data, max_sql)
        # Fallback to accessible_doc_ids if extraction failed
        if not document_sources and accessible_doc_ids:
            document_sources = self._get_document_sources(accessible_doc_ids)

        return {
            "data": combined_data,
            "summary": {
                "report_title": "MIN/MAX Query Results",
                "min_value": min_value,
                "max_value": max_value,
                "min_records": len(min_data),
                "max_records": len(max_data),
                "total_records": len(combined_data),
            },
            "metadata": {
                "query": query,
                "generated_sql": f"MIN Query:\n{min_sql}\n\nMAX Query:\n{max_sql}",
                "explanation": "Executed as two separate queries for MIN and MAX",
                "grouping_fields": min_result.grouping_fields if min_result else [],
                "aggregation_fields": min_result.aggregation_fields if min_result else [],
                "time_granularity": min_result.time_granularity if min_result else None,
                "row_count": len(combined_data),
                "is_min_max_query": True,
                "errors": errors if errors else None,
                "document_sources": document_sources
            },
            # Include generator and field_mappings for streaming report
            "_stream_generator": generator,
            "_field_mappings": field_mappings
        }

    async def execute_dynamic_sql_query_async(
        self,
        accessible_doc_ids: List[UUID],
        query: str,
        llm_client=None,
        max_correction_attempts: int = 3,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Async version of execute_dynamic_sql_query with progress callbacks.

        Provides detailed progress updates during:
        1. Field mapping retrieval
        2. SQL generation by LLM
        3. SQL execution
        4. Report generation by LLM

        Args:
            accessible_doc_ids: List of document IDs user can access
            query: User's natural language query
            llm_client: LLM client for SQL generation and error correction
            max_correction_attempts: Maximum number of LLM correction attempts
            progress_callback: Optional async callback for progress updates

        Returns:
            Query results with data, summaries, and generated SQL
        """
        from analytics_service.llm_sql_generator import LLMSQLGenerator

        logger.info(f"[Dynamic SQL Async] Method called with progress_callback: {'set' if progress_callback else 'None'}")
        logger.info(f"[Dynamic SQL Async] Query: '{query[:80]}...'")
        logger.info(f"[Dynamic SQL Async] Accessible docs: {len(accessible_doc_ids)}")

        # Helper to send progress updates
        async def send_progress(message: str):
            logger.info(f"[Dynamic SQL] send_progress called with message: '{message}', callback is {'set' if progress_callback else 'None'}")
            if progress_callback:
                try:
                    logger.info(f"[Dynamic SQL] Calling progress_callback with: '{message}'")
                    await progress_callback(message)
                    logger.info(f"[Dynamic SQL] progress_callback completed for: '{message}'")
                except Exception as e:
                    logger.warning(f"[Dynamic SQL] Progress callback error: {e}")
            else:
                logger.warning(f"[Dynamic SQL] No progress_callback available, skipping: '{message}'")

        # Step 1: Get field mappings and storage type
        await send_progress("Analyzing data structure...")
        field_mappings, storage_type = self._get_field_mappings_and_storage_type(accessible_doc_ids)
        logger.info(f"[Dynamic SQL] Storage type detected: {storage_type}")

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

        # Step 2: Generate SQL using LLM (with storage type for proper table selection)
        await send_progress("Building database query...")
        generator = LLMSQLGenerator(llm_client)

        # Check if this is a min_max query - if so, generate and execute separate queries
        if generator.is_min_max_query(query):
            logger.info("[Dynamic SQL] Detected MIN/MAX query - using separate query approach")
            await send_progress("Generating separate MIN and MAX queries...")
            return await self._execute_min_max_queries_async(
                generator, query, field_mappings, doc_filter, storage_type, send_progress,
                accessible_doc_ids=accessible_doc_ids
            )

        sql_result = generator.generate_sql(query, field_mappings, doc_filter, storage_type)

        logger.info(f"[Dynamic SQL] Generated SQL:\n{sql_result.sql_query}")
        logger.info(f"[Dynamic SQL] Explanation: {sql_result.explanation}")

        if not sql_result.success or not sql_result.sql_query:
            return {
                "data": [],
                "summary": {"error": sql_result.error or "Failed to generate SQL"},
                "metadata": {"query": query}
            }

        # Step 3: Execute SQL with retry loop for error correction
        await send_progress("Running database query...")
        current_sql = sql_result.sql_query
        attempt = 0
        last_error = None
        correction_history = []

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

                # Step 4: Calculate basic stats (report will be streamed)
                await send_progress(f"Found {len(data)} records, preparing results...")

                # Calculate grand total for metadata
                amount_col = None
                for col in data[0].keys() if data else []:
                    col_lower = col.lower()
                    if any(kw in col_lower for kw in ['amount', 'total', 'sales', 'sum', 'price', 'value']):
                        amount_col = col
                        break
                grand_total = sum(float(r.get(amount_col, 0) or 0) for r in data) if amount_col else 0

                # Get document sources from actual query results (not just accessible docs)
                document_sources = self._get_document_sources_from_results(data, current_sql)
                # Fallback to accessible_doc_ids if extraction failed
                if not document_sources:
                    document_sources = self._get_document_sources(accessible_doc_ids)

                # Return data with streaming generator and field_mappings for report streaming
                return {
                    "data": data,
                    "summary": {
                        "report_title": "Query Results Summary",
                        "grand_total": round(grand_total, 2),
                        "total_records": len(data),
                    },
                    "metadata": {
                        "query": query,
                        "generated_sql": current_sql,
                        "explanation": sql_result.explanation,
                        "grouping_fields": sql_result.grouping_fields,
                        "aggregation_fields": sql_result.aggregation_fields,
                        "time_granularity": sql_result.time_granularity,
                        "row_count": len(data),
                        "correction_attempts": attempt,
                        "correction_history": correction_history if correction_history else None,
                        "document_sources": document_sources
                    },
                    # Include generator and field_mappings for streaming report
                    "_stream_generator": generator,
                    "_field_mappings": field_mappings
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
                    await send_progress(f"Refining query (attempt {attempt + 2})...")
                    logger.info(f"[Dynamic SQL] Attempting LLM-driven SQL correction...")

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

        # All attempts failed (async version)
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
        """
        Get field mappings from documents with extracted data.

        If explicit field_mappings don't exist, dynamically infer them from
        the actual data structure (header_data and line_items).
        """
        mappings, _ = self._get_field_mappings_and_storage_type(accessible_doc_ids)
        return mappings

    def _get_storage_type(self, accessible_doc_ids: List[UUID]) -> str:
        """
        Get the storage type for the given documents.

        NOTE: As of migration 022, all storage is external.
        This method is kept for backward compatibility and always returns "external".

        Returns:
            "external" (always)
        """
        # As of migration 022, all line items are stored externally
        # in the documents_data_line_items table
        logger.info("[Storage Type] Using external storage (migration 022)")
        return 'external'

    def _get_field_mappings_and_storage_type(self, accessible_doc_ids: List[UUID]) -> Tuple[Dict[str, Dict[str, Any]], str]:
        """
        Get field mappings and storage type from documents with extracted data.

        NOTE: As of migration 022, storage_type is always "external".

        Returns:
            Tuple of (field_mappings, storage_type)
        """
        storage_type = 'external'  # Always external as of migration 022

        try:
            doc_ids_str = ", ".join(f"'{str(doc_id)}'" for doc_id in accessible_doc_ids)

            # First try: Look for explicit field_mappings in header_data
            result = self.db.execute(text(f"""
                SELECT header_data->'field_mappings' as field_mappings
                FROM documents_data
                WHERE document_id IN ({doc_ids_str})
                AND header_data->'field_mappings' IS NOT NULL
                LIMIT 1
            """))

            row = result.fetchone()
            if row and row[0]:
                logger.info(f"[Field Mappings] Found explicit field mappings with {len(row[0])} fields, storage=external")
                return dict(row[0]), storage_type

            # Second: Dynamically infer field mappings from actual data structure
            logger.info("[Field Mappings] No explicit mappings found, inferring from data structure...")
            field_mappings = self._infer_field_mappings_from_data(accessible_doc_ids)

            return field_mappings, storage_type

        except Exception as e:
            logger.error(f"Failed to get field mappings: {e}")
            return {}, 'external'

    def _infer_field_mappings_from_data(self, accessible_doc_ids: List[UUID]) -> Dict[str, Dict[str, Any]]:
        """
        Dynamically infer field mappings from actual document data structure.

        Analyzes header_data and line_items to create semantic field mappings
        that the LLM can use for SQL generation.

        For spreadsheet data, uses flexible matching to map column headers
        like "Total Sales", "Purchase Date", etc. to semantic types.

        NOTE: As of migration 022, all line items are stored externally
        in documents_data_line_items table.
        """
        try:
            doc_ids_str = ", ".join(f"'{str(doc_id)}'" for doc_id in accessible_doc_ids)

            # Get document with line items (always external storage as of migration 022)
            result = self.db.execute(text(f"""
                SELECT
                    dd.id,
                    dd.schema_type,
                    dd.header_data,
                    dd.line_items_count
                FROM documents_data dd
                WHERE dd.document_id IN ({doc_ids_str})
                AND dd.line_items_count > 0
                LIMIT 1
            """))

            row = result.fetchone()
            if not row:
                logger.warning("[Field Mappings] No documents with line items found")
                return {}

            documents_data_id = row[0]
            schema_type = row[1]
            header_data = row[2] or {}
            line_items_count = row[3]

            # Fetch sample from external storage (documents_data_line_items table)
            logger.info(f"[Field Mappings] Fetching sample from external storage ({line_items_count} items)...")
            ext_result = self.db.execute(text(f"""
                SELECT data FROM documents_data_line_items
                WHERE documents_data_id = '{documents_data_id}'
                ORDER BY line_number
                LIMIT 1
            """))
            ext_row = ext_result.fetchone()
            if ext_row:
                sample_item = ext_row[0] or {}
                logger.info(f"[Field Mappings] Retrieved sample from external storage: {list(sample_item.keys())}")
            else:
                logger.warning("[Field Mappings] No sample found in external line items table")
                return {}

            # Build field mappings based on discovered fields
            field_mappings = {}

            # Map header fields (document-level)
            header_field_semantics = {
                'transaction_date': {'semantic_type': 'date', 'data_type': 'datetime', 'source': 'header'},
                'invoice_date': {'semantic_type': 'date', 'data_type': 'datetime', 'source': 'header'},
                'receipt_date': {'semantic_type': 'date', 'data_type': 'datetime', 'source': 'header'},
                'date': {'semantic_type': 'date', 'data_type': 'datetime', 'source': 'header'},
                'store_name': {'semantic_type': 'entity', 'data_type': 'string', 'source': 'header'},
                'vendor_name': {'semantic_type': 'entity', 'data_type': 'string', 'source': 'header'},
                'customer_name': {'semantic_type': 'entity', 'data_type': 'string', 'source': 'header'},
                'store_address': {'semantic_type': 'location', 'data_type': 'string', 'source': 'header'},
                'payment_method': {'semantic_type': 'method', 'data_type': 'string', 'source': 'header'},
                'currency': {'semantic_type': 'currency', 'data_type': 'string', 'source': 'header'},
                'receipt_number': {'semantic_type': 'identifier', 'data_type': 'string', 'source': 'header'},
            }

            for field in header_data.keys():
                if field in header_field_semantics:
                    field_mappings[field] = header_field_semantics[field]

            # For spreadsheet data, create field mappings directly from column headers
            # Let the LLM understand the semantics from the column names
            if schema_type == 'spreadsheet':
                field_mappings = self._create_spreadsheet_field_mappings(sample_item, header_data)
                logger.info(f"[Field Mappings] Created {len(field_mappings)} field mappings from spreadsheet columns: {list(field_mappings.keys())}")
                return field_mappings

            # Map line item fields (for non-spreadsheet documents)
            line_item_semantics = {
                'description': {'semantic_type': 'product', 'data_type': 'string', 'source': 'line_item', 'aggregation': 'group_by'},
                'item': {'semantic_type': 'product', 'data_type': 'string', 'source': 'line_item', 'aggregation': 'group_by'},
                'product': {'semantic_type': 'product', 'data_type': 'string', 'source': 'line_item', 'aggregation': 'group_by'},
                'amount': {'semantic_type': 'amount', 'data_type': 'number', 'source': 'line_item', 'aggregation': 'sum'},
                'total': {'semantic_type': 'amount', 'data_type': 'number', 'source': 'line_item', 'aggregation': 'sum'},
                'unit_price': {'semantic_type': 'price', 'data_type': 'number', 'source': 'line_item'},
                'quantity': {'semantic_type': 'quantity', 'data_type': 'number', 'source': 'line_item', 'aggregation': 'sum'},
                'category': {'semantic_type': 'category', 'data_type': 'string', 'source': 'line_item', 'aggregation': 'group_by'},
            }

            for field in sample_item.keys():
                if field in line_item_semantics:
                    field_mappings[field] = line_item_semantics[field]

            logger.info(f"[Field Mappings] Inferred {len(field_mappings)} field mappings from {schema_type} data: {list(field_mappings.keys())}")
            return field_mappings

        except Exception as e:
            logger.error(f"[Field Mappings] Failed to infer field mappings: {e}")
            return {}

    def _create_spreadsheet_field_mappings(
        self,
        sample_item: Dict[str, Any],
        header_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create field mappings for spreadsheet data directly from column headers.

        Instead of using hardcoded patterns, this method:
        1. Uses column_headers from header_data if available
        2. Infers data types from sample values
        3. Lets the LLM understand column semantics from the names directly

        Args:
            sample_item: First row of spreadsheet data (with actual values)
            header_data: Header metadata containing column_headers list

        Returns:
            Field mappings with actual column names as keys
        """
        field_mappings = {}

        # Get column names from header_data or sample_item keys
        column_headers = header_data.get('column_headers', [])
        if not column_headers:
            column_headers = [k for k in sample_item.keys() if k != 'row_number']

        for col_name in column_headers:
            # Skip internal fields
            if col_name in ['row_number', '_id']:
                continue

            # Get sample value to infer data type
            sample_value = sample_item.get(col_name)

            # Infer data type from sample value
            data_type = self._infer_data_type(sample_value)

            # Create field mapping - let LLM understand semantics from column name
            field_mappings[col_name] = {
                'data_type': data_type,
                'source': 'line_item',  # Spreadsheet data is in line_items
                'original_column': col_name,
                # For numeric fields, suggest they can be aggregated
                'aggregation': 'sum' if data_type == 'number' else 'group_by'
            }

        return field_mappings

    def _infer_data_type(self, value: Any) -> str:
        """
        Infer the data type from a sample value.

        Args:
            value: Sample value from the column

        Returns:
            Data type string: 'number', 'datetime', or 'string'
        """
        if value is None:
            return 'string'

        # Check if it's a number
        if isinstance(value, (int, float)):
            return 'number'

        # Check if it's a string that looks like a date/datetime
        if isinstance(value, str):
            import re
            # Common date/datetime patterns
            date_patterns = [
                r'^\d{4}-\d{2}-\d{2}',           # 2024-01-15 or 2024-01-15 00:00:00
                r'^\d{2}/\d{2}/\d{4}',           # 01/15/2024
                r'^\d{2}-\d{2}-\d{4}',           # 15-01-2024
                r'^\d{4}/\d{2}/\d{2}',           # 2024/01/15
            ]
            for pattern in date_patterns:
                if re.match(pattern, value.strip()):
                    return 'datetime'

        return 'string'

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

    # =========================================================================
    # Multi-Schema Query Support
    # =========================================================================

    def execute_multi_schema_query(
        self,
        accessible_doc_ids: List[UUID],
        query: str,
        llm_client=None
    ) -> Dict[str, Any]:
        """
        Execute query across documents with potentially different schemas.

        This method processes each schema type independently without merging,
        returning separated results for each schema type.

        Args:
            accessible_doc_ids: List of document IDs user can access
            query: User's natural language query
            llm_client: Optional LLM client for SQL generation

        Returns:
            Dict with results_by_type, combined_summary, and metadata
        """
        from analytics_service.llm_sql_generator import LLMSQLGenerator

        logger.info(f"[Multi-Schema Query] Processing {len(accessible_doc_ids)} documents")

        # Step 1: Group documents by schema type
        schema_groups = self._group_documents_by_schema(accessible_doc_ids)

        logger.info(f"[Multi-Schema Query] Found {len(schema_groups)} schema types: {list(schema_groups.keys())}")

        # Step 2: Process each schema group independently
        results_by_type = {}
        all_errors = []

        for schema_type, doc_ids in schema_groups.items():
            logger.info(f"[Multi-Schema Query] Processing {len(doc_ids)} documents of type '{schema_type}'")

            try:
                # Get field mappings for this schema type (priority-based lookup)
                field_mappings = self._get_field_mappings_for_schema(
                    schema_type=schema_type,
                    document_ids=doc_ids
                )

                if not field_mappings:
                    logger.warning(f"[Multi-Schema Query] No field mappings for {schema_type}, skipping")
                    continue

                # Build document filter
                doc_ids_str = ", ".join(f"'{str(doc_id)}'" for doc_id in doc_ids)
                doc_filter = f"dd.document_id IN ({doc_ids_str})"

                # Generate and execute SQL
                generator = LLMSQLGenerator(llm_client or self.llm_client)
                sql_result = generator.generate_sql(query, field_mappings, doc_filter)

                if not sql_result.success or not sql_result.sql_query:
                    logger.warning(f"[Multi-Schema Query] SQL generation failed for {schema_type}")
                    all_errors.append({
                        'schema_type': schema_type,
                        'error': sql_result.error or 'SQL generation failed'
                    })
                    continue

                # Execute SQL
                try:
                    result = self.db.execute(text(sql_result.sql_query))
                    rows = result.fetchall()
                    columns = result.keys()
                    data = [dict(zip(columns, row)) for row in rows]

                    # Generate summary for this schema type
                    summary = generator.generate_summary_report(query, data, field_mappings)

                    results_by_type[schema_type] = {
                        'schema_info': {
                            'schema_type': schema_type,
                            'field_mappings': field_mappings,
                            'document_count': len(doc_ids)
                        },
                        'data': data,
                        'row_count': len(data),
                        'summary': summary,
                        'generated_sql': sql_result.sql_query
                    }

                    logger.info(f"[Multi-Schema Query] {schema_type}: {len(data)} rows returned")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"[Multi-Schema Query] SQL execution failed for {schema_type}: {error_msg}")
                    all_errors.append({
                        'schema_type': schema_type,
                        'error': error_msg
                    })
                    self.db.rollback()

            except Exception as e:
                logger.error(f"[Multi-Schema Query] Processing failed for {schema_type}: {e}")
                all_errors.append({
                    'schema_type': schema_type,
                    'error': str(e)
                })

        # Step 3: Generate combined summary (without merging data)
        combined_summary = self._generate_multi_schema_summary(
            user_query=query,
            results_by_type=results_by_type,
            llm_client=llm_client or self.llm_client
        )

        return {
            'success': len(results_by_type) > 0,
            'query': query,
            'results_by_type': results_by_type,
            'combined_summary': combined_summary,
            'schema_types': list(results_by_type.keys()),
            'errors': all_errors if all_errors else None,
            'metadata': {
                'total_documents': len(accessible_doc_ids),
                'schemas_processed': len(results_by_type),
                'schemas_failed': len(all_errors),
                'query_time': datetime.utcnow().isoformat()
            }
        }

    def _group_documents_by_schema(
        self,
        document_ids: List[UUID]
    ) -> Dict[str, List[UUID]]:
        """
        Group documents by their schema type.

        Args:
            document_ids: List of document IDs

        Returns:
            Dict mapping schema_type to list of document IDs
        """
        try:
            doc_ids_str = ", ".join(f"'{str(doc_id)}'" for doc_id in document_ids)

            result = self.db.execute(text(f"""
                SELECT schema_type, document_id
                FROM documents_data
                WHERE document_id IN ({doc_ids_str})
            """))

            groups = {}
            for row in result.fetchall():
                schema_type = row[0] or 'unknown'
                doc_id = row[1]

                if schema_type not in groups:
                    groups[schema_type] = []
                groups[schema_type].append(doc_id)

            return groups

        except Exception as e:
            logger.error(f"[Multi-Schema Query] Failed to group documents: {e}")
            return {'unknown': document_ids}

    def _get_field_mappings_for_schema(
        self,
        schema_type: str,
        document_ids: List[UUID]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get field mappings for a schema type using priority-based lookup.

        Priority:
        1. extraction_metadata.field_mappings (per-document cache)
        2. DataSchema table (formal schema)
        3. LLM-driven inference

        Args:
            schema_type: The schema type
            document_ids: Document IDs of this schema type

        Returns:
            Field mappings dictionary
        """
        doc_ids_str = ", ".join(f"'{str(doc_id)}'" for doc_id in document_ids)

        # Priority 1: Check extraction_metadata
        try:
            result = self.db.execute(text(f"""
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
                    logger.info(f"[Field Mappings] Using cached mappings for {schema_type}")
                    return field_mappings
        except Exception as e:
            logger.warning(f"[Field Mappings] extraction_metadata lookup failed: {e}")

        # Priority 2: Check DataSchema table
        try:
            result = self.db.execute(text(f"""
                SELECT field_mappings
                FROM data_schemas
                WHERE schema_type = '{schema_type}'
                AND is_active = true
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
                    logger.info(f"[Field Mappings] Using formal schema for {schema_type}")
                    return field_mappings
        except Exception as e:
            logger.warning(f"[Field Mappings] DataSchema lookup failed: {e}")

        # Priority 3: Infer from data
        logger.info(f"[Field Mappings] Inferring mappings for {schema_type}")
        return self._infer_field_mappings_from_data(document_ids)

    def _generate_multi_schema_summary(
        self,
        user_query: str,
        results_by_type: Dict[str, Any],
        llm_client=None
    ) -> str:
        """
        Generate a combined summary for multi-schema query results.

        This summarizes results across all schema types without merging the data.

        Args:
            user_query: Original user query
            results_by_type: Results grouped by schema type
            llm_client: Optional LLM client for summary generation

        Returns:
            Combined summary text
        """
        if not results_by_type:
            return "No data found for your query."

        # Build summary sections for each schema type
        sections = []
        for schema_type, result in results_by_type.items():
            summary = result.get('summary', {})
            formatted = summary.get('formatted_report', '')
            row_count = result.get('row_count', 0)
            doc_count = result.get('schema_info', {}).get('document_count', 0)

            sections.append({
                'schema_type': schema_type,
                'document_count': doc_count,
                'row_count': row_count,
                'summary': formatted
            })

        # If LLM available, generate intelligent combined summary
        if llm_client:
            try:
                import json
                prompt = f"""Combine the following query results from different document types into a unified summary.

User Query: "{user_query}"

Results by Document Type:
{json.dumps(sections, indent=2)}

Write a clear, combined summary that:
1. Answers the user's original question
2. Shows results separated by document type
3. Highlights any notable differences or patterns across types
4. Provides grand totals where applicable

Format the response in markdown:"""

                combined = llm_client.generate(prompt)
                return combined.strip()

            except Exception as e:
                logger.warning(f"[Multi-Schema Summary] LLM summary failed: {e}")

        # Fallback: Simple concatenation
        lines = [f"## Query Results by Document Type\n"]
        for section in sections:
            lines.append(f"### {section['schema_type'].replace('_', ' ').title()}")
            lines.append(f"Documents: {section['document_count']}, Rows: {section['row_count']}")
            lines.append(section['summary'] or "(No summary available)")
            lines.append("")

        return "\n".join(lines)
