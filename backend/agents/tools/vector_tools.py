"""
Vector search tools for the Vector Agent.

These tools handle:
- Semantic similarity search via Qdrant
- Document content retrieval
- Result reporting
"""

import json
import logging
import time
from typing import Annotated, Optional

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agents.state.models import AgentOutput
from agents.config import AGENTIC_CONFIG

logger = logging.getLogger(__name__)


@tool
def semantic_search(
    query: str,
    document_ids: str,
    top_k: int,
    filters: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Perform semantic similarity search via Qdrant.

    Searches for documents or chunks that are semantically similar
    to the query using vector embeddings.

    Args:
        query: Search query text
        document_ids: JSON array of document IDs to search within
        top_k: Number of top results to return
        filters: JSON object with additional filters (schema_type, date_range, etc.)

    Returns:
        JSON with search results including:
        - documents: Array of matching documents/chunks
        - scores: Similarity scores
        - total_found: Total number of matches
    """
    try:
        doc_ids = json.loads(document_ids) if document_ids else []
        filter_dict = json.loads(filters) if filters else {}
        workspace_id = state.get("workspace_id", "")

        start_time = time.time()

        # Try to use existing Qdrant client
        try:
            from backend.rag_service.vector_store import VectorStoreService

            vector_service = VectorStoreService()

            # Build search filters
            search_filters = {}
            if doc_ids:
                search_filters["document_id"] = {"$in": doc_ids}
            if filter_dict.get("schema_type"):
                search_filters["schema_type"] = filter_dict["schema_type"]

            # Perform search
            results = vector_service.search(
                query=query,
                workspace_id=workspace_id,
                top_k=top_k,
                filters=search_filters
            )

            execution_time = int((time.time() - start_time) * 1000)

            return json.dumps({
                "success": True,
                "documents": results.get("documents", []),
                "scores": results.get("scores", []),
                "total_found": len(results.get("documents", [])),
                "execution_time_ms": execution_time
            })

        except ImportError:
            logger.warning("VectorStoreService not available, using mock search")

        # Fallback: Mock search results
        # In production, this should never happen
        mock_results = []
        if doc_ids:
            for i, doc_id in enumerate(doc_ids[:top_k]):
                mock_results.append({
                    "document_id": doc_id,
                    "chunk_content": f"Sample content for {doc_id} matching query: {query}",
                    "similarity_score": 0.9 - (i * 0.1),
                    "metadata": {
                        "chunk_index": 0,
                        "source": f"document_{doc_id}"
                    }
                })

        execution_time = int((time.time() - start_time) * 1000)

        return json.dumps({
            "success": True,
            "documents": mock_results,
            "scores": [r["similarity_score"] for r in mock_results],
            "total_found": len(mock_results),
            "execution_time_ms": execution_time,
            "note": "Using mock results - vector service unavailable"
        })

    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON input: {e}",
            "documents": [],
            "total_found": 0
        })
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "documents": [],
            "total_found": 0
        })


@tool
def report_vector_result(
    task_id: str,
    documents: str,
    confidence: float,
    state: Annotated[dict, InjectedState]
) -> str:
    """Report vector search results back to the retrieval supervisor.

    Args:
        task_id: ID of the task being reported
        documents: JSON array of retrieved documents with content and scores
        confidence: Overall confidence score (0-1)

    Returns:
        JSON string with the result report
    """
    try:
        # Handle various invalid values LLM might pass for documents
        if not documents or documents.strip() in ("", "None", "null", "[]"):
            parsed_docs = []
        else:
            try:
                parsed_docs = json.loads(documents)
                if parsed_docs is None:
                    parsed_docs = []
            except json.JSONDecodeError:
                parsed_docs = []

        # Extract document IDs used
        doc_ids_used = list(set(
            d.get("document_id", "") for d in parsed_docs if d.get("document_id")
        ))

        # Determine status
        if confidence >= 0.7 and parsed_docs:
            status = "success"
        elif confidence >= 0.5 or parsed_docs:
            status = "partial"
        else:
            status = "failed"

        # Build structured data from results
        result_data = {
            "retrieved_chunks": len(parsed_docs),
            "documents": parsed_docs,
            "top_score": max((d.get("similarity_score", 0) for d in parsed_docs), default=0)
        }

        logger.info(
            f"Vector Agent reporting result for task {task_id}: "
            f"status={status}, {len(parsed_docs)} docs, confidence={confidence}"
        )

        return json.dumps({
            "success": True,
            "task_id": task_id,
            "agent_name": "vector_agent",
            "status": status,
            "data": result_data,
            "documents_used": doc_ids_used,
            "confidence": confidence,
            "doc_count": len(parsed_docs),
            "message": f"Vector search task {task_id} completed: {len(parsed_docs)} documents, confidence: {confidence:.2f}"
        })

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in report_vector_result: {e}")
        return json.dumps({
            "success": False,
            "task_id": task_id,
            "agent_name": "vector_agent",
            "status": "failed",
            "error": f"Failed to parse documents: {e}"
        })
    except Exception as e:
        logger.error(f"Error reporting vector result: {e}")
        return json.dumps({
            "success": False,
            "task_id": task_id,
            "agent_name": "vector_agent",
            "status": "failed",
            "error": str(e)
        })
