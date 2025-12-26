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
        rows = []
        for doc_data in results:
            row = self._extract_row_data(doc_data)
            rows.append(row)

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
        """Group data by specified dimension."""
        groups = {}

        for row in rows:
            # Determine group key
            if group_by == 'monthly':
                key = self._get_month_key(row.get('date'))
            elif group_by == 'quarterly':
                key = self._get_quarter_key(row.get('date'))
            elif group_by == 'yearly':
                key = self._get_year_key(row.get('date'))
            elif group_by in ['vendor', 'entity', 'customer']:
                key = row.get('entity_name') or 'Unknown'
            elif group_by == 'schema_type':
                key = row.get('schema_type') or 'Unknown'
            else:
                key = 'All'

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
        if 'monthly' in query_lower or 'by month' in query_lower:
            group_by = 'monthly'
        elif 'quarterly' in query_lower or 'by quarter' in query_lower:
            group_by = 'quarterly'
        elif 'yearly' in query_lower or 'by year' in query_lower:
            group_by = 'yearly'
        elif 'by vendor' in query_lower or 'per vendor' in query_lower:
            group_by = 'vendor'
        elif 'by customer' in query_lower or 'per customer' in query_lower:
            group_by = 'customer'

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
