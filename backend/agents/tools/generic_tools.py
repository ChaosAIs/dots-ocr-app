"""
Generic document tools for the Generic Doc Agent.

These tools handle:
- Hybrid search combining vector and SQL approaches
- Data extraction from mixed documents
- Fallback RAG search
- Result reporting
"""

import json
import logging
import time
from typing import Annotated, Literal, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from agents.config import AGENTIC_CONFIG

logger = logging.getLogger(__name__)


@tool
def hybrid_document_search(
    query: str,
    document_ids: str,
    search_strategy: Literal["vector_first", "sql_first", "parallel"],
    state: Annotated[dict, InjectedState]
) -> str:
    """Perform hybrid search combining vector and SQL approaches.

    Used when document type is unclear or contains mixed structured/unstructured data.

    Strategies:
    - vector_first: Try semantic search, if low confidence try SQL extraction
    - sql_first: Try SQL on extracted data, if fails try vector search
    - parallel: Run both simultaneously, merge and rank results

    Args:
        query: Search query
        document_ids: JSON array of document IDs to search
        search_strategy: Which approach to prioritize

    Returns:
        Combined search results with confidence scores
    """
    try:
        doc_ids = json.loads(document_ids) if document_ids else []
        workspace_id = state.get("workspace_id", "")

        start_time = time.time()
        results = {
            "vector_results": None,
            "sql_results": None,
            "merged": None,
            "strategy_used": search_strategy
        }

        # Import tools for internal use
        from agents.tools.vector_tools import semantic_search
        from agents.tools.sql_tools import execute_sql_with_retry

        if search_strategy == "vector_first":
            # Step 1: Try vector search
            vector_result_str = semantic_search.invoke({
                "query": query,
                "document_ids": json.dumps(doc_ids),
                "top_k": 10,
                "filters": "{}",
                "state": state
            })
            vector_result = json.loads(vector_result_str)
            results["vector_results"] = vector_result

            # Check confidence
            if vector_result.get("success") and vector_result.get("total_found", 0) > 0:
                top_score = max(
                    (d.get("similarity_score", 0) for d in vector_result.get("documents", [])),
                    default=0
                )
                if top_score >= 0.7:
                    results["merged"] = vector_result
                    results["confidence"] = top_score
                    results["method"] = "vector_search"
                    execution_time = int((time.time() - start_time) * 1000)
                    results["execution_time_ms"] = execution_time
                    return json.dumps(results)

            # Step 2: Try SQL extraction as fallback
            sql_result = _try_sql_extraction(query, doc_ids, workspace_id, state)
            results["sql_results"] = sql_result

            # Merge results
            results["merged"] = _merge_results(vector_result, sql_result)
            results["confidence"] = max(
                vector_result.get("confidence", 0.5),
                sql_result.get("confidence", 0.5)
            )
            results["method"] = "hybrid_vector_first"

        elif search_strategy == "sql_first":
            # Step 1: Try SQL extraction
            sql_result = _try_sql_extraction(query, doc_ids, workspace_id, state)
            results["sql_results"] = sql_result

            if sql_result.get("success") and sql_result.get("confidence", 0) >= 0.7:
                results["merged"] = sql_result
                results["confidence"] = sql_result.get("confidence", 0.7)
                results["method"] = "sql_extraction"
                execution_time = int((time.time() - start_time) * 1000)
                results["execution_time_ms"] = execution_time
                return json.dumps(results)

            # Step 2: Fall back to vector search
            vector_result_str = semantic_search.invoke({
                "query": query,
                "document_ids": json.dumps(doc_ids),
                "top_k": 10,
                "filters": "{}",
                "state": state
            })
            vector_result = json.loads(vector_result_str)
            results["vector_results"] = vector_result

            # Merge results
            results["merged"] = _merge_results(sql_result, vector_result)
            results["confidence"] = max(
                sql_result.get("confidence", 0.5),
                0.6 if vector_result.get("success") else 0.3
            )
            results["method"] = "hybrid_sql_first"

        elif search_strategy == "parallel":
            # Run both in parallel (sequential in this implementation)
            vector_result_str = semantic_search.invoke({
                "query": query,
                "document_ids": json.dumps(doc_ids),
                "top_k": 10,
                "filters": "{}",
                "state": state
            })
            vector_result = json.loads(vector_result_str)
            results["vector_results"] = vector_result

            sql_result = _try_sql_extraction(query, doc_ids, workspace_id, state)
            results["sql_results"] = sql_result

            # Merge and rank
            results["merged"] = _merge_results(vector_result, sql_result)
            results["confidence"] = max(
                0.6 if vector_result.get("success") else 0.3,
                sql_result.get("confidence", 0.5)
            )
            results["method"] = "parallel_hybrid"

        execution_time = int((time.time() - start_time) * 1000)
        results["execution_time_ms"] = execution_time
        results["success"] = True

        return json.dumps(results)

    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "merged": None,
            "confidence": 0.0
        })


def _try_sql_extraction(query: str, doc_ids: list, workspace_id: str, state: dict) -> dict:
    """Try to extract and query structured data."""
    try:
        # Build a simple extraction query
        doc_filter = ", ".join([f"'{d}'" for d in doc_ids])

        sql = f"""
        SELECT
            document_id,
            filename,
            schema_type,
            header_data,
            summary_data
        FROM documents_data
        WHERE document_id IN ({doc_filter})
        """

        from agents.tools.sql_tools import execute_sql_with_retry

        result_str = execute_sql_with_retry.invoke({
            "sql_query": sql,
            "max_retries": 2,
            "state": state
        })
        result = json.loads(result_str)

        if result.get("success"):
            return {
                "success": True,
                "data": result.get("data", []),
                "confidence": 0.7,
                "method": "sql_extraction"
            }
        return {
            "success": False,
            "data": [],
            "confidence": 0.3,
            "error": result.get("error")
        }

    except Exception as e:
        return {
            "success": False,
            "data": [],
            "confidence": 0.0,
            "error": str(e)
        }


def _merge_results(result1: dict, result2: dict) -> dict:
    """Merge results from two search methods."""
    merged = {
        "sources": []
    }

    # Add vector results
    if result1 and result1.get("documents"):
        for doc in result1["documents"]:
            merged["sources"].append({
                "type": "vector",
                "content": doc.get("chunk_content", ""),
                "score": doc.get("similarity_score", 0),
                "document_id": doc.get("document_id", "")
            })

    # Add SQL results
    if result2 and result2.get("data"):
        for row in result2["data"]:
            merged["sources"].append({
                "type": "sql",
                "content": row,
                "score": 0.7,
                "document_id": row.get("document_id", "")
            })

    # Sort by score
    merged["sources"].sort(key=lambda x: x.get("score", 0), reverse=True)

    return merged


@tool
def extract_and_query(
    query: str,
    document_ids: str,
    extraction_fields: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Extract structured data from documents then query it.

    For documents that may have embedded structured data:
    - Tables in PDFs
    - Lists in documents
    - Key-value pairs in unstructured text

    Args:
        query: What to find/calculate
        document_ids: JSON array of document IDs to process
        extraction_fields: JSON array of fields to attempt to extract

    Returns:
        Extracted and queried results
    """
    try:
        doc_ids = json.loads(document_ids) if document_ids else []
        fields = json.loads(extraction_fields) if extraction_fields else []
        workspace_id = state.get("workspace_id", "")

        start_time = time.time()

        # Get document content
        extracted_data = []

        try:
            from sqlalchemy import text
            from backend.database import get_db_session

            with get_db_session() as db:
                for doc_id in doc_ids:
                    result = db.execute(
                        text("""
                            SELECT document_id, filename, header_data, summary_data, schema_type
                            FROM documents_data
                            WHERE document_id = :doc_id
                        """),
                        {"doc_id": doc_id}
                    )
                    row = result.fetchone()

                    if row:
                        doc_data = {
                            "document_id": row[0],
                            "filename": row[1],
                            "header_data": row[2] or {},
                            "summary_data": row[3] or {},
                            "schema_type": row[4]
                        }

                        # Extract requested fields
                        extracted = {"document_id": doc_id}
                        for field in fields:
                            # Check header_data
                            if field in (doc_data["header_data"] or {}):
                                extracted[field] = doc_data["header_data"][field]
                            # Check summary_data
                            elif field in (doc_data["summary_data"] or {}):
                                extracted[field] = doc_data["summary_data"][field]

                        if len(extracted) > 1:  # More than just document_id
                            extracted_data.append(extracted)

        except Exception as db_error:
            logger.warning(f"Database extraction failed: {db_error}")

        # Perform aggregation if query suggests it
        query_lower = query.lower()
        result_data = extracted_data

        if extracted_data:
            if "sum" in query_lower or "total" in query_lower:
                # Try to sum numeric fields
                for field in fields:
                    values = [
                        float(d.get(field, 0))
                        for d in extracted_data
                        if d.get(field) and str(d.get(field)).replace(".", "").isdigit()
                    ]
                    if values:
                        result_data = {
                            "aggregation": "sum",
                            "field": field,
                            "value": sum(values),
                            "count": len(values)
                        }
                        break

            elif "count" in query_lower:
                result_data = {
                    "aggregation": "count",
                    "value": len(extracted_data)
                }

            elif "average" in query_lower or "avg" in query_lower:
                for field in fields:
                    values = [
                        float(d.get(field, 0))
                        for d in extracted_data
                        if d.get(field) and str(d.get(field)).replace(".", "").isdigit()
                    ]
                    if values:
                        result_data = {
                            "aggregation": "average",
                            "field": field,
                            "value": sum(values) / len(values),
                            "count": len(values)
                        }
                        break

        execution_time = int((time.time() - start_time) * 1000)

        confidence = 0.7 if extracted_data else 0.3

        return json.dumps({
            "success": bool(extracted_data),
            "data": result_data,
            "extracted_count": len(extracted_data),
            "fields_found": list(set(
                field for d in extracted_data for field in d.keys() if field != "document_id"
            )),
            "confidence": confidence,
            "execution_time_ms": execution_time
        })

    except Exception as e:
        logger.error(f"Error in extract_and_query: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "data": [],
            "confidence": 0.0
        })


@tool
def fallback_rag_search(
    query: str,
    document_ids: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Fallback to standard RAG when other methods fail.

    This is the last resort - uses the existing RAG pipeline
    which combines document chunking, embedding, and LLM response.

    Args:
        query: Search query
        document_ids: JSON array of document IDs to search

    Returns:
        RAG search results
    """
    try:
        doc_ids = json.loads(document_ids) if document_ids else []
        workspace_id = state.get("workspace_id", "")

        start_time = time.time()

        # Try to use existing RAG service
        try:
            from backend.rag_service import RAGService

            rag_service = RAGService()
            results = rag_service.query(
                query=query,
                workspace_id=workspace_id,
                document_ids=doc_ids,
                top_k=5
            )

            execution_time = int((time.time() - start_time) * 1000)

            return json.dumps({
                "success": True,
                "method": "fallback_rag",
                "results": results,
                "confidence": results.get("confidence", 0.5),
                "execution_time_ms": execution_time,
                "note": "Used fallback RAG pipeline"
            })

        except ImportError:
            logger.warning("RAGService not available")

        # Ultimate fallback: Just return document content
        try:
            from sqlalchemy import text
            from backend.database import get_db_session

            doc_filter = ", ".join([f"'{d}'" for d in doc_ids])

            with get_db_session() as db:
                result = db.execute(text(f"""
                    SELECT document_id, filename, header_data, summary_data
                    FROM documents_data
                    WHERE document_id IN ({doc_filter})
                    LIMIT 10
                """))
                rows = result.fetchall()

                documents = []
                for row in rows:
                    documents.append({
                        "document_id": row[0],
                        "filename": row[1],
                        "content": {
                            "header": row[2],
                            "summary": row[3]
                        }
                    })

            execution_time = int((time.time() - start_time) * 1000)

            return json.dumps({
                "success": True,
                "method": "document_content_fallback",
                "results": documents,
                "confidence": 0.4,
                "execution_time_ms": execution_time,
                "note": "RAG unavailable - returning raw document content"
            })

        except Exception as db_error:
            logger.error(f"Fallback database query failed: {db_error}")

        execution_time = int((time.time() - start_time) * 1000)

        return json.dumps({
            "success": False,
            "method": "fallback_rag",
            "results": [],
            "confidence": 0.0,
            "execution_time_ms": execution_time,
            "error": "All fallback methods failed"
        })

    except Exception as e:
        logger.error(f"Error in fallback RAG search: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "results": [],
            "confidence": 0.0
        })


@tool
def report_generic_result(
    task_id: str,
    data: str,
    documents_used: str,
    confidence: float,
    method_used: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Report generic document agent results back to the retrieval supervisor.

    Args:
        task_id: ID of the task being reported
        data: JSON string of result data
        documents_used: JSON array of document IDs used
        confidence: Confidence score (0-1)
        method_used: Which method produced these results

    Returns:
        JSON string with the result report
    """
    try:
        # Handle various invalid values LLM might pass for data
        if not data or data.strip() in ("", "None", "null"):
            parsed_data = None
        else:
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError:
                parsed_data = None

        # Handle various invalid values LLM might pass for documents_used
        if not documents_used or documents_used.strip() in ("", "None", "null", "[]"):
            parsed_docs = []
        else:
            try:
                parsed_docs = json.loads(documents_used)
                if parsed_docs is None:
                    parsed_docs = []
            except json.JSONDecodeError:
                parsed_docs = []

        # Determine status
        if confidence >= 0.6:
            status = "success"
        elif confidence >= 0.4:
            status = "partial"
        else:
            status = "failed"

        issues = []
        if confidence < 0.6:
            issues.append(f"Low confidence using {method_used}")
        if not parsed_data:
            issues.append("No data retrieved")

        logger.info(
            f"Generic Doc Agent reporting result for task {task_id}: "
            f"status={status}, method={method_used}, confidence={confidence}"
        )

        return json.dumps({
            "success": True,
            "task_id": task_id,
            "agent_name": "generic_doc_agent",
            "status": status,
            "data": parsed_data,
            "documents_used": parsed_docs,
            "schema_type": "mixed",
            "confidence": confidence,
            "issues": issues,
            "method_used": method_used,
            "message": f"Generic task {task_id} completed using {method_used}, confidence: {confidence:.2f}"
        })

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in report_generic_result: {e}")
        return json.dumps({
            "success": False,
            "task_id": task_id,
            "agent_name": "generic_doc_agent",
            "status": "failed",
            "data": None,
            "confidence": 0.0,
            "issues": [f"Failed to parse data: {e}"],
            "error": str(e)
        })
    except Exception as e:
        logger.error(f"Error reporting generic result: {e}")
        return json.dumps({
            "success": False,
            "task_id": task_id,
            "agent_name": "generic_doc_agent",
            "status": "failed",
            "data": None,
            "confidence": 0.0,
            "issues": [str(e)],
            "error": str(e)
        })
