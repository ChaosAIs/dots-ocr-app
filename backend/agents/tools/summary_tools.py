"""
Summary tools for the Summary Agent.

DEPRECATED: The format_response tool is no longer used by the Summary Agent.
The Summary Agent now uses the shared LLMSQLGenerator.generate_summary_report()
method directly to ensure consistent response quality between agent and non-agent flows.

These tools are kept for backwards compatibility but may be removed in future versions:
- aggregate_results: Aggregates agent outputs (kept for potential future use)
- format_response: DEPRECATED - use LLMSQLGenerator.generate_summary_report() instead
"""

import json
import logging
from typing import Annotated, List, Dict, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

logger = logging.getLogger(__name__)


@tool
def aggregate_results(
    state: Annotated[dict, InjectedState]
) -> str:
    """Aggregate all agent outputs into a unified data structure.

    Combines results from SQL, Vector, Graph, and Generic agents
    into a coherent structure for final formatting.

    This tool reads agent_outputs directly from state - no arguments needed.

    Returns:
        Aggregated data structure with all results organized by task
    """
    try:
        # Read agent_outputs directly from state
        outputs = state.get("agent_outputs", [])
        logger.info(f"[aggregate_results] Found {len(outputs)} outputs in state")

        # Ensure outputs is a list
        if not isinstance(outputs, list):
            outputs = [outputs] if outputs else []

        # Convert Pydantic models to dicts if needed
        normalized_outputs = []
        for o in outputs:
            if hasattr(o, 'model_dump'):
                normalized_outputs.append(o.model_dump())
            elif hasattr(o, 'dict'):
                normalized_outputs.append(o.dict())
            elif isinstance(o, dict):
                normalized_outputs.append(o)
            else:
                logger.warning(f"Unexpected output type: {type(o)}")
                continue
        outputs = normalized_outputs

        aggregated = {
            "by_task": {},
            "by_agent": {},
            "all_data": [],
            "documents_used": set(),
            "total_tasks": len(outputs),
            "successful_tasks": 0,
            "partial_tasks": 0,
            "failed_tasks": 0
        }

        for output in outputs:
            task_id = output.get("task_id", "unknown")
            agent_name = output.get("agent_name", "unknown")
            status = output.get("status", "unknown")
            data = output.get("data")
            docs = output.get("documents_used", [])

            # Organize by task
            aggregated["by_task"][task_id] = {
                "agent": agent_name,
                "status": status,
                "data": data,
                "confidence": output.get("confidence", 0),
                "documents": docs
            }

            # Organize by agent
            if agent_name not in aggregated["by_agent"]:
                aggregated["by_agent"][agent_name] = []
            aggregated["by_agent"][agent_name].append({
                "task_id": task_id,
                "data": data,
                "status": status
            })

            # Collect all data
            if data is not None:
                if isinstance(data, dict):
                    aggregated["all_data"].append({
                        "task_id": task_id,
                        **data
                    })
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            aggregated["all_data"].append({
                                "task_id": task_id,
                                **item
                            })
                        else:
                            aggregated["all_data"].append({
                                "task_id": task_id,
                                "value": item
                            })
                else:
                    aggregated["all_data"].append({
                        "task_id": task_id,
                        "value": data
                    })

            # Track documents
            aggregated["documents_used"].update(docs)

            # Count statuses
            if status == "success":
                aggregated["successful_tasks"] += 1
            elif status == "partial":
                aggregated["partial_tasks"] += 1
            else:
                aggregated["failed_tasks"] += 1

        # Convert set to list for JSON serialization
        aggregated["documents_used"] = list(aggregated["documents_used"])

        # Calculate success rate
        if aggregated["total_tasks"] > 0:
            aggregated["success_rate"] = (
                aggregated["successful_tasks"] / aggregated["total_tasks"]
            )
        else:
            aggregated["success_rate"] = 0

        return json.dumps(aggregated, indent=2, default=str)

    except Exception as e:
        logger.error(f"Error aggregating results: {e}")
        return json.dumps({
            "error": str(e),
            "by_task": {},
            "all_data": [],
            "success_rate": 0
        })


@tool
def format_response(
    aggregated_data: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Format the final response for user consumption.

    DEPRECATED: This function is no longer used by the Summary Agent.
    The Summary Agent now uses LLMSQLGenerator.generate_summary_report() directly
    to ensure consistent response quality between agent and non-agent flows.

    This function is kept for backwards compatibility but produces lower quality
    responses compared to the LLM-based formatting in LLMSQLGenerator.

    Creates a markdown-formatted response with:
    - Summary of findings
    - Data tables where appropriate
    - Source attribution
    - Confidence indicators

    Args:
        aggregated_data: JSON string of aggregated results (from aggregate_results tool)

    Returns:
        JSON string with formatted response
    """
    import warnings
    warnings.warn(
        "format_response is deprecated. Use LLMSQLGenerator.generate_summary_report() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    try:
        aggregated = json.loads(aggregated_data) if isinstance(aggregated_data, str) else aggregated_data

        # Get data sources and query from state
        sources = state.get("data_sources", [])
        original_query = state.get("user_query", "")

        # Also get sources from aggregated data if available
        if not sources and aggregated.get("documents_used"):
            sources = aggregated.get("documents_used", [])

        logger.info(f"[format_response] Query: {original_query[:50]}..., Sources: {len(sources)}")

        # Build the markdown response
        response_parts = []

        # Header
        response_parts.append("## Summary:\n")

        # Process results by task
        by_task = aggregated.get("by_task", {})

        for task_id, task_data in by_task.items():
            data = task_data.get("data")
            confidence = task_data.get("confidence", 0)

            if data is None:
                continue

            # Format based on data type
            if isinstance(data, dict):
                # Check for aggregation results
                if "aggregation" in data:
                    agg_type = data.get("aggregation", "")
                    field = data.get("field", "value")
                    value = data.get("value", 0)

                    # Format the value nicely
                    if isinstance(value, float):
                        formatted_value = f"{value:,.2f}"
                    else:
                        formatted_value = f"{value:,}" if isinstance(value, int) else str(value)

                    label = _get_label_for_aggregation(agg_type, field)
                    response_parts.append(f"- **{label}:** {formatted_value}\n")

                elif "total" in str(data).lower() or any(k.startswith("total") for k in data.keys()):
                    # Generic total/sum result
                    for key, value in data.items():
                        if value is not None:
                            label = _format_key_as_label(key)
                            if isinstance(value, float):
                                formatted_value = f"{value:,.2f}"
                            elif isinstance(value, int):
                                formatted_value = f"{value:,}"
                            else:
                                formatted_value = str(value)
                            response_parts.append(f"- **{label}:** {formatted_value}\n")

                else:
                    # Generic dict result
                    for key, value in data.items():
                        if key not in ["task_id", "document_id"] and value is not None:
                            label = _format_key_as_label(key)
                            response_parts.append(f"- **{label}:** {value}\n")

            elif isinstance(data, list):
                # List of results - might be documents or records
                if len(data) > 0:
                    if len(data) <= 5:
                        for item in data:
                            if isinstance(item, dict):
                                # Format as bullet points
                                item_str = ", ".join(
                                    f"{_format_key_as_label(k)}: {v}"
                                    for k, v in item.items()
                                    if k not in ["task_id", "document_id"]
                                )
                                response_parts.append(f"  - {item_str}\n")
                            else:
                                response_parts.append(f"  - {item}\n")
                    else:
                        response_parts.append(f"- **Results:** {len(data)} records found\n")

            else:
                # Simple value
                response_parts.append(f"- **Result:** {data}\n")

        # Add data sources section
        if sources:
            response_parts.append("\n**Data Sources:**\n")
            unique_sources = list(set(sources))
            for source in unique_sources:
                response_parts.append(f"- {source}\n")

        # Build final response
        final_response = "".join(response_parts)

        # If response is empty, provide a default message
        if not final_response.strip() or final_response.strip() == "## Summary:":
            final_response = """## Summary:

No specific results were retrieved for this query. This could mean:
- The requested data is not available in the documents
- The query could not be processed by the available data sources

Please try refining your query or check that the relevant documents have been uploaded.
"""

        # Calculate overall confidence
        confidences = [
            t.get("confidence", 0)
            for t in by_task.values()
            if t.get("confidence")
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Build structured data for programmatic access
        structured_data = {
            "query": original_query,
            "results": aggregated.get("all_data", []),
            "confidence": avg_confidence,
            "sources": sources,
            "task_count": aggregated.get("total_tasks", 0),
            "success_rate": aggregated.get("success_rate", 0)
        }

        logger.info(
            f"Summary Agent formatted response: {len(final_response)} chars, "
            f"{len(sources)} sources, confidence: {avg_confidence:.2f}"
        )

        return json.dumps({
            "success": True,
            "final_response": final_response,
            "data_sources": sources,
            "structured_data": structured_data,
            "confidence": avg_confidence,
            "message": "Response formatted and ready"
        })

    except Exception as e:
        logger.error(f"Error formatting response: {e}")

        # Return error response
        error_response = f"""## Summary:

An error occurred while formatting the results: {e}

Please try your query again or contact support if the issue persists.
"""

        return json.dumps({
            "success": False,
            "final_response": error_response,
            "data_sources": [],
            "structured_data": {"error": str(e)},
            "error": str(e),
            "message": f"Error formatting response: {e}"
        })


def _get_label_for_aggregation(agg_type: str, field: str) -> str:
    """Generate a human-readable label for an aggregation result."""
    field_label = _format_key_as_label(field)

    labels = {
        "sum": f"Total {field_label}",
        "count": f"Number of {field_label}s" if field != "value" else "Total Count",
        "avg": f"Average {field_label}",
        "average": f"Average {field_label}",
        "min": f"Minimum {field_label}",
        "max": f"Maximum {field_label}"
    }

    return labels.get(agg_type.lower(), f"{agg_type.title()} of {field_label}")


def _format_key_as_label(key: str) -> str:
    """Convert a snake_case or camelCase key to a human-readable label."""
    # Handle snake_case
    label = key.replace("_", " ")

    # Handle camelCase
    import re
    label = re.sub(r'([a-z])([A-Z])', r'\1 \2', label)

    # Capitalize words
    label = label.title()

    # Common replacements
    replacements = {
        "Id": "ID",
        "Url": "URL",
        "Api": "API",
        "Sql": "SQL",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)

    return label
