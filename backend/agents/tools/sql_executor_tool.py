"""
SQL Executor Tool - Atomic SQL execution for agent flow.

This tool wraps the non-agent SQLQueryExecutor.execute_dynamic_sql_query()
method, providing a single atomic tool call that handles:
- Field mapping retrieval
- LLM-driven SQL generation
- SQL execution with retry/error correction
- Result formatting

This eliminates the ReAct overhead where LLM decides which tool to call next,
making SQL execution deterministic and efficient.
"""

import json
import logging
import os
import sys
import time
from typing import Annotated, Optional, List, Dict, Any
from uuid import UUID

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from agents.state.models import AgentOutput
from agents.config import AGENTIC_CONFIG

logger = logging.getLogger(__name__)

# Ensure backend path is in sys.path for imports
_BACKEND_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_PATH not in sys.path:
    sys.path.insert(0, _BACKEND_PATH)
    logger.info(f"[sql_executor_tool] Added backend path to sys.path: {_BACKEND_PATH}")


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
            logger.error("[sql_executor_tool] Failed to import get_db_session")
            return None


def _get_llm_client():
    """Get LLM client for SQL generation and error correction."""
    try:
        from rag_service.llm_service import get_llm_service
        llm_service = get_llm_service()
        return llm_service
    except Exception as e:
        logger.warning(f"[sql_executor_tool] Failed to initialize LLM client: {e}")
        return None


def _get_sql_executor(db_session):
    """Get SQLQueryExecutor instance."""
    try:
        from analytics_service.sql_query_executor import SQLQueryExecutor
        llm_client = _get_llm_client()
        return SQLQueryExecutor(db=db_session, llm_client=llm_client)
    except Exception as e:
        logger.warning(f"[sql_executor_tool] Failed to initialize SQLQueryExecutor: {e}")
        return None


def execute_sql_subtask_core(
    task_id: str,
    sub_task_description: str,
    user_query: str,
    document_ids: List[str],
    schema_type: Optional[str] = None,
    target_fields: Optional[List[str]] = None,
    aggregation_hint: Optional[str] = None,
    max_correction_attempts: int = 3
) -> Dict[str, Any]:
    """
    Core implementation of SQL subtask execution.

    This function can be called directly (for non-agent use) or via the tool wrapper.
    It reuses the SQLQueryExecutor.execute_dynamic_sql_query() from the non-agent flow.

    Args:
        task_id: Unique identifier for this task
        sub_task_description: What the planner wants to achieve
        user_query: Original user query for context
        document_ids: Pre-routed document IDs from planner
        schema_type: Optional schema type hint
        target_fields: Optional specific fields to focus on
        aggregation_hint: Optional aggregation hint (sum, count, avg, etc.)
        max_correction_attempts: Max SQL correction retries (default: 3)

    Returns:
        Dict with execution results
    """
    start_time = time.time()

    # Validate inputs
    if not document_ids:
        return {
            "success": False,
            "task_id": task_id,
            "error": "No document IDs provided",
            "data": [],
            "row_count": 0
        }

    # Convert string IDs to UUIDs
    try:
        doc_uuids = [UUID(doc_id) if isinstance(doc_id, str) else doc_id for doc_id in document_ids]
    except ValueError as e:
        return {
            "success": False,
            "task_id": task_id,
            "error": f"Invalid document ID format: {e}",
            "data": [],
            "row_count": 0
        }

    # Build the query to pass to the executor
    # Combine sub_task_description with context for better SQL generation
    enhanced_query = sub_task_description
    if aggregation_hint:
        enhanced_query = f"{sub_task_description} (aggregation: {aggregation_hint})"
    if target_fields:
        enhanced_query = f"{enhanced_query} focusing on fields: {', '.join(target_fields)}"

    logger.info(f"[sql_executor_tool] ========== EXECUTING SQL SUBTASK ==========")
    logger.info(f"[sql_executor_tool] Task ID: {task_id}")
    logger.info(f"[sql_executor_tool] Query: {enhanced_query[:200]}...")
    logger.info(f"[sql_executor_tool] Document count: {len(doc_uuids)}")
    logger.info(f"[sql_executor_tool] Document IDs: {[str(d) for d in doc_uuids[:5]]}{'...' if len(doc_uuids) > 5 else ''}")

    get_db_session = _get_db_session()
    if not get_db_session:
        return {
            "success": False,
            "task_id": task_id,
            "error": "Database session not available",
            "data": [],
            "row_count": 0
        }

    try:
        with get_db_session() as db:
            # Get or create SQLQueryExecutor
            sql_executor = _get_sql_executor(db)
            if not sql_executor:
                return {
                    "success": False,
                    "task_id": task_id,
                    "error": "SQLQueryExecutor not available",
                    "data": [],
                    "row_count": 0
                }

            # Get LLM client for SQL generation
            llm_client = _get_llm_client()

            # Use the proven non-agent flow method
            result = sql_executor.execute_dynamic_sql_query(
                accessible_doc_ids=doc_uuids,
                query=enhanced_query,
                llm_client=llm_client,
                max_correction_attempts=max_correction_attempts
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Extract data from result
            data = result.get("data", [])
            metadata = result.get("metadata", {})
            summary = result.get("summary", {})

            logger.info(f"[sql_executor_tool] ========== SQL EXECUTION RESULT ==========")
            logger.info(f"[sql_executor_tool] Data rows: {len(data)}")
            logger.info(f"[sql_executor_tool] Generated SQL: {metadata.get('generated_sql', 'N/A')[:300]}...")
            logger.info(f"[sql_executor_tool] Explanation: {metadata.get('explanation', 'N/A')}")
            if summary.get("error"):
                logger.warning(f"[sql_executor_tool] Summary error: {summary.get('error')}")

            # Check for errors
            error = metadata.get("error") or summary.get("error")
            if error and not data:
                return {
                    "success": False,
                    "task_id": task_id,
                    "error": error,
                    "data": [],
                    "row_count": 0,
                    "sql_generated": metadata.get("generated_sql"),
                    "execution_time_ms": execution_time_ms
                }

            # Serialize data for JSON (handle Decimal, datetime, etc.)
            serialized_data = _serialize_for_json(data)

            # Get document sources
            document_sources = metadata.get("document_sources", [])
            documents_used = [src.get("document_id") for src in document_sources if src.get("document_id")]
            if not documents_used:
                documents_used = [str(doc_id) for doc_id in doc_uuids]

            return {
                "success": True,
                "task_id": task_id,
                "data": serialized_data,
                "row_count": len(data),
                "sql_executed": metadata.get("generated_sql"),
                "explanation": metadata.get("explanation"),
                "documents_used": documents_used,
                "schema_type": schema_type or "unknown",
                "execution_metadata": {
                    "correction_attempts": metadata.get("correction_attempts", 0),
                    "grouping_fields": metadata.get("grouping_fields"),
                    "aggregation_fields": metadata.get("aggregation_fields"),
                    "time_granularity": metadata.get("time_granularity"),
                    "execution_time_ms": execution_time_ms
                },
                "summary": summary if isinstance(summary, dict) and not summary.get("error") else None
            }

    except Exception as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[sql_executor_tool] Error executing subtask: {e}", exc_info=True)
        return {
            "success": False,
            "task_id": task_id,
            "error": str(e),
            "data": [],
            "row_count": 0,
            "execution_time_ms": execution_time_ms
        }


def _serialize_for_json(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Serialize data for JSON output, handling special types."""
    from decimal import Decimal
    from datetime import datetime, date

    def serialize_value(value):
        if value is None:
            return None
        elif isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, (datetime, date)):
            return value.isoformat()
        elif isinstance(value, bytes):
            return value.decode('utf-8', errors='replace')
        elif isinstance(value, dict):
            return {k: serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [serialize_value(v) for v in value]
        elif isinstance(value, UUID):
            return str(value)
        else:
            return value

    return [
        {k: serialize_value(v) for k, v in row.items()}
        for row in data
    ]


@tool
def execute_sql_subtask(
    task_id: str,
    sub_task_description: str,
    document_ids: str,
    schema_type: Optional[str] = None,
    target_fields: Optional[str] = None,
    aggregation_hint: Optional[str] = None,
    state: Annotated[dict, InjectedState] = None
) -> str:
    """Execute a complete SQL analytics subtask using the non-agent flow.

    This tool handles the ENTIRE SQL workflow in a single atomic operation:
    1. Retrieves field mappings from schema
    2. Generates SQL using LLM
    3. Executes SQL against database
    4. Retries with LLM error correction if needed
    5. Returns formatted results

    NO additional tool calls needed - this replaces the generate_sql + execute_sql + report pattern.

    Args:
        task_id: Unique identifier for this task (from execution plan)
        sub_task_description: What to calculate/retrieve (from planner)
        document_ids: JSON array of document IDs to query
        schema_type: Optional schema type for the documents
        target_fields: Optional JSON array of specific fields to focus on
        aggregation_hint: Optional aggregation type (sum, count, avg, min, max, combined)

    Returns:
        JSON with:
        - success: bool
        - task_id: str
        - data: List[Dict] (result rows)
        - row_count: int
        - sql_executed: str
        - documents_used: List[str]
        - execution_metadata: Dict with timing and correction info
        - error: Optional[str] if failed
    """
    # Parse document_ids
    try:
        if document_ids and document_ids.strip():
            doc_ids_stripped = document_ids.strip()
            if doc_ids_stripped.startswith('['):
                doc_ids = json.loads(doc_ids_stripped)
            else:
                doc_ids = [d.strip().strip('"').strip("'") for d in doc_ids_stripped.split(',') if d.strip()]
        else:
            doc_ids = []
    except json.JSONDecodeError:
        doc_ids = [d.strip().strip('"').strip("'") for d in document_ids.split(',') if d.strip()]

    # Parse target_fields
    fields = None
    if target_fields and target_fields.strip():
        try:
            fields_stripped = target_fields.strip()
            if fields_stripped.startswith('['):
                fields = json.loads(fields_stripped)
            else:
                fields = [f.strip().strip('"').strip("'") for f in fields_stripped.split(',') if f.strip()]
        except json.JSONDecodeError:
            fields = [f.strip() for f in target_fields.split(',') if f.strip()]

    # Get user query from state for context
    user_query = ""
    if state:
        user_query = state.get("user_query", "")

    # Get max retries from config
    max_retries = AGENTIC_CONFIG.get("max_sql_retries", 3)

    # Execute the subtask
    result = execute_sql_subtask_core(
        task_id=task_id,
        sub_task_description=sub_task_description,
        user_query=user_query,
        document_ids=doc_ids,
        schema_type=schema_type,
        target_fields=fields,
        aggregation_hint=aggregation_hint,
        max_correction_attempts=max_retries
    )

    return json.dumps(result)


def create_agent_output_from_result(result: Dict[str, Any]) -> AgentOutput:
    """
    Convert execute_sql_subtask result to AgentOutput for the workflow state.

    This helper is used by the retrieval team to convert tool results
    into the standardized AgentOutput format.
    """
    success = result.get("success", False)

    # Determine status
    if success:
        row_count = result.get("row_count", 0)
        if row_count > 0:
            status = "success"
        else:
            status = "partial"  # Query succeeded but no data
    else:
        status = "failed"

    # Calculate confidence based on success and data quality
    confidence = 0.0
    if success:
        confidence = 0.9 if result.get("row_count", 0) > 0 else 0.5
        # Reduce confidence if corrections were needed
        corrections = result.get("execution_metadata", {}).get("correction_attempts", 0)
        if corrections > 0:
            confidence = max(0.5, confidence - (0.1 * corrections))

    # Collect issues
    issues = []
    if result.get("error"):
        issues.append(result["error"])

    # Debug: Log the data being set on AgentOutput
    data = result.get("data")
    data_info = "None"
    if data is not None:
        if isinstance(data, list):
            data_info = f"list[{len(data)}]"
        elif isinstance(data, dict):
            data_info = f"dict[{len(data)} keys]"
        else:
            data_info = f"{type(data).__name__}"
    logger.info(f"[create_agent_output_from_result] Creating AgentOutput: task_id={result.get('task_id')}, status={status}, row_count={result.get('row_count')}, data={data_info}")

    return AgentOutput(
        task_id=result.get("task_id", "unknown"),
        agent_name="sql_agent",
        status=status,
        data=result.get("data"),
        documents_used=result.get("documents_used", []),
        schema_type=result.get("schema_type", "unknown"),
        query_executed=result.get("sql_executed"),
        row_count=result.get("row_count"),
        confidence=confidence,
        issues=issues,
        execution_time_ms=result.get("execution_metadata", {}).get("execution_time_ms")
    )
