"""
SQL tools for the SQL Agent.

These tools handle:
- Schema-aware SQL generation (via shared LLMSQLGenerator service)
- SQL execution with LLM-based error correction
- Result reporting

Quick Win Integration:
- Reuses LLMSQLGenerator from analytics_service for accurate SQL generation
- Reuses SchemaService for proper field mappings
- Reuses correct_sql_error() for LLM-based error correction
"""

import json
import logging
import os
import sys
import time
from typing import Annotated, Optional, Dict, Any, List

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agents.state.models import AgentOutput
from agents.config import AGENTIC_CONFIG

logger = logging.getLogger(__name__)

# Ensure backend path is in sys.path for imports
_BACKEND_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_PATH not in sys.path:
    sys.path.insert(0, _BACKEND_PATH)
    logger.info(f"[sql_tools] Added backend path to sys.path: {_BACKEND_PATH}")

# Import shared services from analytics_service (Quick Win)
_sql_generator = None
_schema_service = None


def _get_db_session():
    """Get database session - handles multiple import paths."""
    try:
        from db.database import get_db_session
        return get_db_session
    except ImportError:
        try:
            from database import get_db_session
            return get_db_session
        except ImportError:
            logger.error("[sql_tools] Failed to import get_db_session")
            return None


def _get_llm_client():
    """Get LLM client for SQL generation and error correction."""
    try:
        from rag_service.llm_service import get_llm_service
        llm_service = get_llm_service()
        return llm_service
    except Exception as e:
        logger.warning(f"[sql_tools] Failed to initialize LLM client: {e}")
        return None


def _get_sql_generator(db_session):
    """Get or create LLMSQLGenerator instance (cached)."""
    global _sql_generator
    if _sql_generator is None:
        try:
            from analytics_service.llm_sql_generator import LLMSQLGenerator
            llm_client = _get_llm_client()
            _sql_generator = LLMSQLGenerator(llm_client=llm_client)
            logger.info("[sql_tools] Initialized LLMSQLGenerator from analytics_service")
        except Exception as e:
            logger.warning(f"[sql_tools] Failed to initialize LLMSQLGenerator: {e}")
            return None
    return _sql_generator


def _get_schema_service(db_session):
    """Get or create SchemaService instance."""
    try:
        from analytics_service.schema_service import SchemaService
        llm_client = _get_llm_client()
        return SchemaService(db=db_session, llm_client=llm_client)
    except Exception as e:
        logger.warning(f"[sql_tools] Failed to initialize SchemaService: {e}")
        return None


def _validate_and_normalize_field_mappings(field_mappings: Any) -> Dict[str, Any]:
    """
    Validate and normalize field mappings to the expected grouped format.

    The LLMSQLGenerator expects:
    {
        'header_mappings': [{'canonical': 'field_name', 'data_type': '...', ...}],
        'line_item_mappings': [{'canonical': 'field_name', ...}],
        'summary_mappings': [{'canonical': 'field_name', ...}]
    }

    This function converts various formats to the expected format.
    """
    logger.info(f"[sql_tools] _validate_and_normalize_field_mappings input type: {type(field_mappings)}")

    if not field_mappings:
        logger.warning("[sql_tools] Empty field_mappings received")
        return {}

    if not isinstance(field_mappings, dict):
        logger.warning(f"[sql_tools] field_mappings is not a dict, got: {type(field_mappings)}")
        return {}

    # Log the keys and structure
    logger.info(f"[sql_tools] field_mappings keys: {list(field_mappings.keys())}")

    # Check if already in grouped format with proper structure
    grouped_keys = ['header_mappings', 'line_item_mappings', 'summary_mappings']
    is_grouped = any(k in field_mappings for k in grouped_keys)

    if is_grouped:
        logger.info("[sql_tools] Detected grouped format field_mappings")
        # Validate each mapping list contains dicts with 'canonical' key
        result = {}
        for key in grouped_keys:
            mappings = field_mappings.get(key, [])
            logger.info(f"[sql_tools] {key} type: {type(mappings)}, sample: {mappings[:2] if isinstance(mappings, list) and mappings else mappings}")

            if not isinstance(mappings, list):
                logger.warning(f"[sql_tools] {key} is not a list, got: {type(mappings)}")
                mappings = []

            # Ensure each item is a dict with 'canonical' key
            normalized = []
            for idx, item in enumerate(mappings):
                if isinstance(item, dict) and 'canonical' in item:
                    normalized.append(item)
                elif isinstance(item, dict):
                    # Try to extract canonical from other fields
                    if 'field_name' in item:
                        item['canonical'] = item['field_name']
                        normalized.append(item)
                    elif 'name' in item:
                        item['canonical'] = item['name']
                        normalized.append(item)
                    else:
                        logger.warning(f"[sql_tools] {key}[{idx}] is a dict but has no 'canonical', 'field_name', or 'name' key: {list(item.keys())}")
                elif isinstance(item, list):
                    # Handle case where item is a nested list (e.g., [field_name, field_info])
                    logger.warning(f"[sql_tools] {key}[{idx}] is a nested list: {item}")
                    if len(item) >= 2 and isinstance(item[0], str):
                        # Convert [field_name, field_info_dict] to proper format
                        field_name = item[0]
                        field_info = item[1] if isinstance(item[1], dict) else {}
                        normalized.append({
                            'canonical': field_name,
                            'data_type': field_info.get('data_type', 'string') if isinstance(field_info, dict) else 'string',
                            **({k: v for k, v in field_info.items()} if isinstance(field_info, dict) else {})
                        })
                    elif len(item) >= 1 and isinstance(item[0], str):
                        # Simple [field_name] list
                        normalized.append({'canonical': item[0], 'data_type': 'string'})
                elif isinstance(item, str):
                    # Handle case where item is just a string field name
                    logger.warning(f"[sql_tools] {key}[{idx}] is a string: {item}")
                    normalized.append({'canonical': item, 'data_type': 'string'})
                else:
                    logger.warning(f"[sql_tools] {key}[{idx}] has unexpected type: {type(item)}")

            result[key] = normalized
            logger.info(f"[sql_tools] Normalized {key}: {len(normalized)} items")
        return result

    # Legacy flat format: {'field_name': {'data_type': '...', 'source': '...'}}
    logger.info("[sql_tools] Detected legacy flat format field_mappings")
    header_mappings = []
    line_item_mappings = []
    summary_mappings = []

    for field_name, field_info in field_mappings.items():
        if not isinstance(field_info, dict):
            logger.warning(f"[sql_tools] Field '{field_name}' info is not a dict: {type(field_info)}")
            # Try to create a basic mapping if field_info is a simple value
            if isinstance(field_info, str):
                header_mappings.append({'canonical': field_name, 'data_type': field_info})
            continue
        source = field_info.get('source', 'header')
        mapping = {'canonical': field_name, **field_info}

        if source == 'line_item':
            line_item_mappings.append(mapping)
        elif source == 'summary':
            summary_mappings.append(mapping)
        else:
            header_mappings.append(mapping)

    logger.info(f"[sql_tools] Converted legacy format: header={len(header_mappings)}, line_item={len(line_item_mappings)}, summary={len(summary_mappings)}")

    return {
        'header_mappings': header_mappings,
        'line_item_mappings': line_item_mappings,
        'summary_mappings': summary_mappings
    }


def _get_field_mappings_for_documents(db_session, document_ids: List[str], schema_type: str = None) -> Dict[str, Any]:
    """
    Get field mappings for documents using SchemaService.

    This provides the critical field information needed for accurate SQL generation:
    - Field names and their database column locations (header_data, line_items, summary_data)
    - Data types for proper casting
    - Semantic types for understanding field meaning
    """
    try:
        schema_service = _get_schema_service(db_session)
        if not schema_service:
            return {}

        # Try to get field mappings from schema service
        if schema_type:
            schema = schema_service.get_schema(schema_type)
            if schema and schema.field_mappings:
                logger.info(f"[sql_tools] Got field mappings from DataSchema for type: {schema_type}")
                # Validate and normalize the schema field mappings
                return _validate_and_normalize_field_mappings(schema.field_mappings)

        # Fallback: Get field mappings from extraction_metadata in documents_data
        from sqlalchemy import text
        doc_ids_str = ", ".join([f"'{doc_id}'" for doc_id in document_ids])
        result = db_session.execute(text(f"""
            SELECT DISTINCT
                dd.schema_type,
                dd.extraction_metadata,
                dd.header_data,
                dd.summary_data
            FROM documents_data dd
            WHERE dd.document_id::text IN ({doc_ids_str})
            LIMIT 5
        """))

        rows = result.fetchall()
        if not rows:
            return {}

        # Build field mappings from actual data structure
        header_fields = {}
        line_item_fields = {}
        summary_fields = {}

        for row in rows:
            extraction_meta = row[1] if row[1] else {}
            header_data = row[2] if row[2] else {}
            summary_data = row[3] if row[3] else {}

            # Check extraction_metadata for field_mappings first
            if isinstance(extraction_meta, dict) and extraction_meta.get('field_mappings'):
                # Validate and normalize the extraction metadata field mappings
                normalized = _validate_and_normalize_field_mappings(extraction_meta['field_mappings'])
                if normalized:
                    return normalized

            # Build from actual header_data keys
            for key, value in header_data.items():
                if key not in ['column_headers', 'sheet_name', 'row_count', 'metadata']:
                    data_type = 'number' if isinstance(value, (int, float)) else 'string'
                    header_fields[key] = {
                        'data_type': data_type,
                        'source': 'header',
                        'access_pattern': f"header_data->>'{key}'"
                    }

            # Build from summary_data keys
            for key, value in summary_data.items():
                data_type = 'number' if isinstance(value, (int, float)) else 'string'
                summary_fields[key] = {
                    'data_type': data_type,
                    'source': 'summary',
                    'access_pattern': f"summary_data->>'{key}'"
                }

        return {
            'header_mappings': [{'canonical': k, **v} for k, v in header_fields.items()],
            'summary_mappings': [{'canonical': k, **v} for k, v in summary_fields.items()],
            'line_item_mappings': []
        }

    except Exception as e:
        logger.warning(f"[sql_tools] Failed to get field mappings: {e}")
        return {}


@tool
def generate_schema_aware_sql(
    task_description: str,
    document_ids: str,
    schema_group: str,
    aggregation_type: Optional[str],
    target_fields: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Generate SQL query aware of document schema grouping.

    QUICK WIN: Uses shared LLMSQLGenerator from analytics_service for accurate SQL generation.

    For combined documents (same schema):
    - Generate single query with document_id IN (...)

    For single document:
    - Generate focused query for that document

    Args:
        task_description: What to calculate/retrieve
        document_ids: JSON array of document IDs to query
        schema_group: JSON SchemaGroup with schema information
        aggregation_type: Type of aggregation (sum, count, avg, etc.)
        target_fields: JSON array of fields to target

    Returns:
        Generated SQL query string
    """
    try:
        doc_ids = json.loads(document_ids)
        group = json.loads(schema_group) if schema_group else {}
        fields = json.loads(target_fields) if target_fields else []

        common_fields = group.get("common_fields", fields)
        schema_type = group.get("schema_type", "unknown")

        # Build document filter for SQL
        doc_filter = ", ".join([f"'{d}'" for d in doc_ids])

        # Get database session
        get_db_session = _get_db_session()
        if not get_db_session:
            return json.dumps({"error": "Database session not available", "sql": ""})

        with get_db_session() as db:
            # QUICK WIN: Get field mappings using shared SchemaService
            field_mappings = _get_field_mappings_for_documents(db, doc_ids, schema_type)

            # QUICK WIN: Try to use LLMSQLGenerator for intelligent SQL generation
            sql_generator = _get_sql_generator(db)
            if sql_generator and field_mappings:
                try:
                    logger.info(f"[sql_tools] Using LLMSQLGenerator for task: {task_description[:50]}...")

                    # Generate SQL using the shared generator
                    result = sql_generator.generate_sql(
                        user_query=task_description,
                        field_mappings=field_mappings,
                        table_filter=f"dd.document_id IN ({doc_filter})"
                    )

                    if result and result.success and result.sql_query:
                        logger.info(f"[sql_tools] LLMSQLGenerator produced SQL successfully")
                        return json.dumps({
                            "sql": result.sql_query,
                            "explanation": result.explanation or f"Generated {aggregation_type or 'query'} for {len(doc_ids)} documents",
                            "confidence": 0.9,
                            "method": "llm_sql_generator"
                        })
                    else:
                        logger.warning(f"[sql_tools] LLMSQLGenerator failed: {result.error if result else 'No result'}")
                except Exception as gen_error:
                    logger.warning(f"[sql_tools] LLMSQLGenerator error: {gen_error}, falling back to template")

            # Fallback: Template-based SQL generation (improved - no invalid columns)
            logger.info("[sql_tools] Using template-based SQL generation (fallback)")

            # Check if this is a combined aggregation task
            # Combined tasks have descriptions like "Combined: sum...; count...; avg..."
            is_combined = "combined:" in task_description.lower()
            agg_parts = []

            # Helper to get field accessor for line items
            def get_line_item_accessor(field):
                return f"(li.data->>'{field}')::numeric"

            # Determine the aggregation SQL based on available fields
            # For combined aggregations, generate all in one query
            if is_combined or aggregation_type == "combined":
                # Parse the combined description to find which aggregations are needed
                desc_lower = task_description.lower()
                field = common_fields[0] if common_fields else "quantity"

                if "sum" in desc_lower:
                    agg_parts.append(f"SUM({get_line_item_accessor(field)}) as total_{field.replace(' ', '_')}")
                if "count" in desc_lower:
                    agg_parts.append("COUNT(*) as item_count")
                if "avg" in desc_lower or "average" in desc_lower:
                    agg_parts.append(f"AVG({get_line_item_accessor(field)}) as avg_{field.replace(' ', '_')}")
                if "min" in desc_lower:
                    agg_parts.append(f"MIN({get_line_item_accessor(field)}) as min_{field.replace(' ', '_')}")
                if "max" in desc_lower:
                    agg_parts.append(f"MAX({get_line_item_accessor(field)}) as max_{field.replace(' ', '_')}")

                if agg_parts:
                    agg_sql = ", ".join(agg_parts)
                    logger.info(f"[sql_tools] Generated combined aggregation SQL with {len(agg_parts)} aggregations")
                else:
                    agg_sql = "COUNT(*) as count"
            elif aggregation_type == "sum":
                if common_fields:
                    field = common_fields[0]
                    # Check field source for correct access pattern
                    agg_sql = f"SUM(CAST(COALESCE(header_data->>'{field}', summary_data->>'{field}', '0') AS NUMERIC)) as total_{field.replace(' ', '_')}"
                else:
                    agg_sql = "SUM(CAST(COALESCE(summary_data->>'total_amount', header_data->>'amount', '0') AS NUMERIC)) as total"
            elif aggregation_type == "count":
                agg_sql = "COUNT(*) as count"
            elif aggregation_type == "avg":
                if common_fields:
                    field = common_fields[0]
                    agg_sql = f"AVG(CAST(COALESCE(header_data->>'{field}', summary_data->>'{field}', '0') AS NUMERIC)) as avg_{field.replace(' ', '_')}"
                else:
                    agg_sql = "AVG(CAST(COALESCE(summary_data->>'total_amount', header_data->>'amount', '0') AS NUMERIC)) as average"
            elif aggregation_type == "min":
                field = common_fields[0] if common_fields else "amount"
                agg_sql = f"MIN(CAST(COALESCE(header_data->>'{field}', summary_data->>'{field}', '0') AS NUMERIC)) as min_{field.replace(' ', '_')}"
            elif aggregation_type == "max":
                field = common_fields[0] if common_fields else "amount"
                agg_sql = f"MAX(CAST(COALESCE(header_data->>'{field}', summary_data->>'{field}', '0') AS NUMERIC)) as max_{field.replace(' ', '_')}"
            else:
                # Default to selecting fields
                if common_fields:
                    field_selects = [f"COALESCE(header_data->>'{f}', summary_data->>'{f}') as \"{f}\"" for f in common_fields[:5]]
                    agg_sql = ", ".join(field_selects)
                else:
                    agg_sql = "header_data, summary_data"

            # FIXED: Use proper JOIN to get filename from documents table (not from documents_data)
            # For combined/line-item aggregations, include JOIN to documents_data_line_items
            if is_combined or "li.data" in agg_sql:
                sql = f"""
SELECT
    {agg_sql}
FROM documents_data dd
JOIN documents_data_line_items li ON li.documents_data_id = dd.id
JOIN documents d ON dd.document_id = d.id
WHERE dd.document_id IN ({doc_filter})
""".strip()
            else:
                sql = f"""
SELECT
    {agg_sql}
FROM documents_data dd
JOIN documents d ON dd.document_id = d.id
WHERE dd.document_id IN ({doc_filter})
""".strip()

        return json.dumps({
            "sql": sql,
            "explanation": f"Generated {aggregation_type or 'select'} query for {len(doc_ids)} documents",
            "confidence": 0.7,
            "method": "template_fallback"
        })

    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON input: {e}", "sql": ""})
    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        return json.dumps({"error": str(e), "sql": ""})


@tool
def execute_sql_with_retry(
    sql_query: str,
    max_retries: int,
    state: Annotated[dict, InjectedState]
) -> str:
    """Execute SQL query with LLM-based error correction retry.

    QUICK WIN: Uses correct_sql_error() from LLMSQLGenerator for intelligent error correction.

    On failure:
    1. Captures the PostgreSQL error message
    2. Sends error + failed SQL + schema to LLM for analysis
    3. LLM generates corrected SQL
    4. Retries with the corrected query

    Args:
        sql_query: SQL query to execute
        max_retries: Maximum number of retry attempts

    Returns:
        JSON with query results or error information
    """
    max_retries = min(max_retries, AGENTIC_CONFIG.get("max_sql_retries", 3))

    try:
        from sqlalchemy import text

        get_db_session = _get_db_session()
        if not get_db_session:
            return json.dumps({
                "success": False,
                "error": "Database module not found",
                "data": [],
                "row_count": 0
            })

        start_time = time.time()
        last_error = None
        current_sql = sql_query
        correction_history = []

        # Get user query from state for error correction context
        user_query = state.get("user_query", "")

        for attempt in range(max_retries + 1):
            try:
                with get_db_session() as db:
                    result = db.execute(text(current_sql))
                    rows = result.fetchall()
                    columns = list(result.keys()) if hasattr(result, 'keys') else []

                    # Convert to list of dicts with proper serialization
                    from decimal import Decimal
                    data = []
                    for row in rows:
                        row_dict = {}
                        for i, col in enumerate(columns):
                            value = row[i]
                            # Handle special types for JSON serialization
                            if value is None:
                                pass  # Keep None as is
                            elif isinstance(value, Decimal):
                                # Convert Decimal to float for JSON serialization
                                value = float(value)
                            elif hasattr(value, 'isoformat'):
                                value = value.isoformat()
                            elif isinstance(value, (bytes,)):
                                value = value.decode('utf-8', errors='replace')
                            elif isinstance(value, dict):
                                # Handle JSONB fields - ensure nested Decimals are converted
                                value = json.loads(json.dumps(value, default=str))
                            row_dict[col] = value
                        data.append(row_dict)

                    execution_time = int((time.time() - start_time) * 1000)

                    result_json = {
                        "success": True,
                        "data": data,
                        "row_count": len(data),
                        "execution_time_ms": execution_time,
                        "retries_used": attempt,
                        "sql_executed": current_sql
                    }

                    if correction_history:
                        result_json["corrections_applied"] = correction_history

                    return json.dumps(result_json)

            except Exception as e:
                last_error = str(e)
                logger.warning(f"[sql_tools] SQL execution attempt {attempt + 1} failed: {e}")

                if attempt < max_retries:
                    # QUICK WIN: Use LLM-based error correction
                    corrected_sql = _correct_sql_with_llm(
                        user_query=user_query,
                        failed_sql=current_sql,
                        error_message=last_error,
                        state=state
                    )

                    if corrected_sql and corrected_sql != current_sql:
                        logger.info(f"[sql_tools] LLM corrected SQL, retrying...")
                        correction_history.append({
                            "attempt": attempt + 1,
                            "error": last_error[:200],
                            "correction_applied": True
                        })
                        current_sql = corrected_sql
                    else:
                        logger.warning(f"[sql_tools] LLM could not correct SQL, retrying same query...")
                        correction_history.append({
                            "attempt": attempt + 1,
                            "error": last_error[:200],
                            "correction_applied": False
                        })
                        time.sleep(0.5)  # Brief delay before retry

                    continue

        return json.dumps({
            "success": False,
            "error": last_error,
            "data": [],
            "row_count": 0,
            "retries_used": max_retries,
            "sql_attempted": current_sql,
            "corrections_attempted": correction_history
        })

    except Exception as e:
        logger.error(f"[sql_tools] Error executing SQL: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "data": [],
            "row_count": 0
        })


def _correct_sql_with_llm(
    user_query: str,
    failed_sql: str,
    error_message: str,
    state: dict
) -> Optional[str]:
    """
    Use LLM to analyze SQL error and generate corrected query.

    QUICK WIN: Reuses correct_sql_error() from LLMSQLGenerator.

    Args:
        user_query: Original user's natural language query
        failed_sql: The SQL query that failed
        error_message: The error message from PostgreSQL
        state: Agent state (for getting document context)

    Returns:
        Corrected SQL query, or None if correction failed
    """
    try:
        get_db_session = _get_db_session()
        if not get_db_session:
            return None

        with get_db_session() as db:
            # Get SQL generator with LLM client
            sql_generator = _get_sql_generator(db)
            if not sql_generator:
                logger.warning("[sql_tools] No SQL generator available for error correction")
                return None

            # Get field mappings from state or extract from error context
            # Try to get document IDs from the failed SQL
            import re
            doc_ids_match = re.search(r"document_id\s+IN\s*\(([^)]+)\)", failed_sql, re.IGNORECASE)
            doc_ids = []
            if doc_ids_match:
                doc_ids_str = doc_ids_match.group(1)
                # Extract UUIDs from the string
                doc_ids = re.findall(r"'([^']+)'", doc_ids_str)

            # Get field mappings for context
            field_mappings = {}
            if doc_ids:
                field_mappings = _get_field_mappings_for_documents(db, doc_ids)

            # Call the shared error correction method
            result = sql_generator.correct_sql_error(
                user_query=user_query or "Execute the query",
                failed_sql=failed_sql,
                error_message=error_message,
                field_mappings=field_mappings
            )

            if result and result.success and result.sql_query:
                logger.info(f"[sql_tools] LLM corrected SQL successfully")
                return result.sql_query
            else:
                logger.warning(f"[sql_tools] LLM correction failed: {result.error if result else 'No result'}")
                return None

    except Exception as e:
        logger.error(f"[sql_tools] Error in SQL correction: {e}")
        return None


@tool
def report_sql_result(
    task_id: str,
    data: str,
    documents_used: str,
    schema_type: str,
    sql_executed: str,
    row_count: int,
    confidence: float,
    issues: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Report SQL agent results back to the retrieval supervisor.

    Args:
        task_id: ID of the task being reported
        data: JSON string of result data
        documents_used: JSON array of document IDs used
        schema_type: Schema type of processed documents
        sql_executed: The SQL query that was executed
        row_count: Number of rows returned
        confidence: Confidence score (0-1)
        issues: JSON array of any issues encountered

    Returns:
        JSON string with the result report
    """
    try:
        parsed_data = json.loads(data) if data else None
        parsed_docs = json.loads(documents_used) if documents_used else []
        parsed_issues = json.loads(issues) if issues else []

        # Determine status based on confidence and issues
        if confidence >= 0.7 and not parsed_issues:
            status = "success"
        elif confidence >= 0.5:
            status = "partial"
        else:
            status = "failed"

        logger.info(f"SQL Agent reporting result for task {task_id}: status={status}, confidence={confidence}")

        return json.dumps({
            "success": True,
            "task_id": task_id,
            "agent_name": "sql_agent",
            "status": status,
            "data": parsed_data,
            "documents_used": parsed_docs,
            "schema_type": schema_type,
            "sql_executed": sql_executed,
            "row_count": row_count,
            "confidence": confidence,
            "issues": parsed_issues,
            "message": f"SQL task {task_id} completed with {row_count} rows, confidence: {confidence:.2f}"
        })

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in report_sql_result: {e}")
        return json.dumps({
            "success": False,
            "task_id": task_id,
            "agent_name": "sql_agent",
            "status": "failed",
            "error": f"Failed to parse result data: {e}"
        })
    except Exception as e:
        logger.error(f"Error reporting SQL result: {e}")
        return json.dumps({
            "success": False,
            "task_id": task_id,
            "agent_name": "sql_agent",
            "status": "failed",
            "error": str(e)
        })
