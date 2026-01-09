"""
Summary Agent implementation.

The Summary Agent handles response synthesis:
- Aggregates results from all agents
- Formats the final response using shared LLMSQLGenerator (same as non-agent flow)
- Provides structured data output
"""

import json
import logging
import os
import sys
from typing import Optional, Dict, Any, List

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END

from agents.prompts.summary_prompts import SUMMARY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Ensure backend path is in sys.path for imports
_BACKEND_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_PATH not in sys.path:
    sys.path.insert(0, _BACKEND_PATH)


def _get_llm_sql_generator(llm_client=None):
    """Get LLMSQLGenerator instance for formatting (same as non-agent flow uses)."""
    try:
        from analytics_service.llm_sql_generator import LLMSQLGenerator
        return LLMSQLGenerator(llm_client=llm_client)
    except Exception as e:
        logger.warning(f"[SummaryAgent] Failed to initialize LLMSQLGenerator: {e}")
        return None


def _get_llm_client():
    """Get LLM client for summary generation."""
    try:
        from rag_service.llm_service import get_llm_service
        return get_llm_service()
    except Exception as e:
        logger.warning(f"[SummaryAgent] Failed to get LLM service: {e}")
        return None


def _build_document_id_to_filename_map(available_documents: List[Any]) -> Dict[str, str]:
    """
    Build a lookup map from document IDs to filenames.

    Args:
        available_documents: List of DocumentSource objects or dicts from state

    Returns:
        Dict mapping document_id -> filename
    """
    id_to_filename = {}

    for doc in available_documents:
        # Handle both Pydantic model and dict
        if hasattr(doc, 'model_dump'):
            doc = doc.model_dump()
        elif hasattr(doc, 'dict'):
            doc = doc.dict()
        elif not isinstance(doc, dict):
            continue

        doc_id = doc.get("document_id", "")
        filename = doc.get("filename", "")

        if doc_id and filename:
            id_to_filename[doc_id] = filename

    return id_to_filename


def _extract_sql_results_from_outputs(
    agent_outputs: List[Dict[str, Any]],
    id_to_filename_map: Dict[str, str] = None
) -> tuple:
    """
    Extract SQL query results from agent outputs for formatting.

    Args:
        agent_outputs: List of agent output dicts/models
        id_to_filename_map: Optional map to convert document IDs to filenames

    Returns:
        Tuple of (query_results: List[Dict], field_mappings: Dict, data_sources: List[str], schema_types: Set[str])
    """
    query_results = []
    data_sources = []
    field_mappings = {}
    schema_types = set()

    if id_to_filename_map is None:
        id_to_filename_map = {}

    for output in agent_outputs:
        # Handle both dict and Pydantic model
        if hasattr(output, 'model_dump'):
            output = output.model_dump()
        elif hasattr(output, 'dict'):
            output = output.dict()

        agent_name = output.get("agent_name", "")
        status = output.get("status", "")
        data = output.get("data")
        docs_used = output.get("documents_used", [])
        schema_type = output.get("schema_type", "")

        # Track schema types
        if schema_type:
            schema_types.add(schema_type.lower())

        # Track data sources - convert IDs to filenames
        for doc_id in docs_used:
            filename = id_to_filename_map.get(doc_id, doc_id)
            data_sources.append(filename)

        # Extract SQL results (primary data source for analytics)
        if agent_name == "sql_agent" and status == "success" and data:
            if isinstance(data, list):
                query_results.extend(data)
            elif isinstance(data, dict):
                query_results.append(data)

    # Remove duplicates from data sources
    data_sources = list(set(data_sources))

    logger.info(f"[SummaryAgent] Extracted {len(query_results)} rows from agent outputs, {len(data_sources)} sources, schema_types={schema_types}")

    return query_results, field_mappings, data_sources, schema_types


def _format_response_with_llm_generator(
    user_query: str,
    query_results: List[Dict[str, Any]],
    field_mappings: Dict[str, Any],
    data_sources: List[str],
    schema_types: set = None
) -> tuple:
    """
    Format response using shared LLMSQLGenerator (same as non-agent flow).

    For tabular/spreadsheet data, forces large result strategy to avoid
    sending full table records to LLM.

    Returns:
        Tuple of (final_response: str, structured_data: Dict)
    """
    llm_client = _get_llm_client()
    sql_generator = _get_llm_sql_generator(llm_client)

    if not sql_generator:
        logger.warning("[SummaryAgent] LLMSQLGenerator not available, using fallback formatting")
        return _fallback_format_response(query_results, data_sources), {}

    if not query_results:
        logger.info("[SummaryAgent] No query results to format")
        return "No data found for your query.", {}

    try:
        # Determine primary schema type
        schema_type = "unknown"
        if schema_types:
            # Prioritize tabular types
            tabular_types = {"spreadsheet", "csv", "excel", "tabular", "table"}
            for st in schema_types:
                if st in tabular_types:
                    schema_type = st
                    break
            if schema_type == "unknown":
                schema_type = next(iter(schema_types))

        logger.info(f"[SummaryAgent] Using LLMSQLGenerator for {len(query_results)} rows, schema_type={schema_type}")

        # Use the shared generate_summary_report method with schema_type
        report_result = sql_generator.generate_summary_report(
            user_query=user_query,
            query_results=query_results,
            field_mappings=field_mappings,
            schema_type=schema_type
        )

        formatted_report = report_result.get("formatted_report", "")

        # Add data sources section if not already present
        if formatted_report and data_sources and "Data Sources:" not in formatted_report:
            sources_text = "\n\n---\n**Data Sources:**\n"
            for source in data_sources:
                sources_text += f"- {source}\n"
            formatted_report += sources_text

        structured_data = {
            "report_title": report_result.get("report_title", "Query Results"),
            "grand_total": report_result.get("grand_total", 0),
            "total_records": report_result.get("total_records", len(query_results)),
            "query": user_query,
            "sources": data_sources
        }

        logger.info(f"[SummaryAgent] LLMSQLGenerator produced {len(formatted_report)} chars response")
        return formatted_report, structured_data

    except Exception as e:
        logger.error(f"[SummaryAgent] LLMSQLGenerator failed: {e}, using fallback")
        return _fallback_format_response(query_results, data_sources), {}


def _fallback_format_response(query_results: List[Dict[str, Any]], data_sources: List[str]) -> str:
    """
    Fallback formatting when LLMSQLGenerator is not available.

    Simple markdown formatting without LLM.
    """
    if not query_results:
        return "No data found for your query."

    response_parts = ["## Summary:\n"]

    # Format the first few results
    for i, result in enumerate(query_results[:10]):
        if isinstance(result, dict):
            for key, value in result.items():
                if value is not None and key not in ["task_id", "document_id"]:
                    label = key.replace("_", " ").title()
                    if isinstance(value, float):
                        formatted_value = f"{value:,.2f}"
                    elif isinstance(value, int):
                        formatted_value = f"{value:,}"
                    else:
                        formatted_value = str(value)
                    response_parts.append(f"- **{label}:** {formatted_value}\n")

    if len(query_results) > 10:
        response_parts.append(f"\n... and {len(query_results) - 10} more results\n")

    # Add data sources
    if data_sources:
        response_parts.append("\n**Data Sources:**\n")
        for source in list(set(data_sources)):
            response_parts.append(f"- {source}\n")

    return "".join(response_parts)


def create_summary_agent(
    llm: BaseChatModel,
    custom_prompt: Optional[str] = None
):
    """Create the Summary Agent.

    The Summary Agent aggregates all approved results and formats
    them into a coherent response for the user.

    IMPORTANT: This agent now uses the shared LLMSQLGenerator.generate_summary_report()
    method to ensure consistent response quality between agent and non-agent flows.

    Args:
        llm: Language model to use for the agent
        custom_prompt: Optional custom system prompt

    Returns:
        Compiled LangGraph agent
    """
    prompt = custom_prompt or SUMMARY_SYSTEM_PROMPT

    def summary_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main summary node that aggregates results and formats the response.

        Uses the shared LLMSQLGenerator for formatting (same as non-agent flow).
        """
        logger.info("[SummaryAgent] Starting summary process...")

        # Get user query for context
        user_query = state.get("user_query", "")

        # Build document ID to filename lookup map
        available_documents = state.get("available_documents", [])
        id_to_filename_map = _build_document_id_to_filename_map(available_documents)
        logger.info(f"[SummaryAgent] Built ID-to-filename map with {len(id_to_filename_map)} entries")

        # Step 1: Extract SQL results from agent outputs
        logger.info("[SummaryAgent] Step 1: Extracting results from agent outputs...")
        agent_outputs = state.get("agent_outputs", [])
        query_results, field_mappings, data_sources, schema_types = _extract_sql_results_from_outputs(
            agent_outputs, id_to_filename_map
        )

        # Also get data sources from state if available (convert IDs to filenames)
        state_sources = state.get("data_sources", [])
        if state_sources:
            converted_sources = [id_to_filename_map.get(s, s) for s in state_sources]
            data_sources = list(set(data_sources + converted_sources))

        logger.info(f"[SummaryAgent] Extracted {len(query_results)} rows, {len(data_sources)} sources, schema_types={schema_types}")

        # Step 2: Format using shared LLMSQLGenerator (same as non-agent flow)
        logger.info("[SummaryAgent] Step 2: Formatting with LLMSQLGenerator...")
        final_response, structured_data = _format_response_with_llm_generator(
            user_query=user_query,
            query_results=query_results,
            field_mappings=field_mappings,
            data_sources=data_sources,
            schema_types=schema_types
        )

        logger.info(f"[SummaryAgent] Final response length: {len(final_response)} chars")
        logger.info(f"[SummaryAgent] Data sources: {len(data_sources)}")

        return {
            "final_response": final_response,
            "data_sources": data_sources,
            "structured_data": structured_data
        }

    # Build a simple single-node graph
    builder = StateGraph(dict)
    builder.add_node("summary", summary_node)
    builder.set_entry_point("summary")
    builder.add_edge("summary", END)

    agent = builder.compile()

    logger.info("Created Summary Agent with shared LLMSQLGenerator (same as non-agent flow)")

    return agent
