"""
SQL Agent implementation.

The SQL Agent handles structured/tabular data queries using a single atomic tool
that wraps the proven non-agent SQL execution flow.

Architecture Change (v2):
- Previously: ReAct agent with 3 tools (generate_sql, execute_sql, report_result)
- Now: Single tool call (execute_sql_subtask) that handles everything

This eliminates the LLM orchestration overhead within SQL execution,
making it deterministic and efficient while reusing the non-agent flow.
"""

import json
import logging
from typing import Optional, Dict, Any, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from agents.tools.sql_executor_tool import (
    execute_sql_subtask,
    execute_sql_subtask_core,
    create_agent_output_from_result
)
from agents.state.models import AgentOutput

logger = logging.getLogger(__name__)


# Simplified state for the SQL agent
class SQLAgentState(TypedDict):
    """State for the SQL Agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str
    # Task context from retrieval team
    task_id: str
    task_description: str
    document_ids: List[str]
    schema_type: str
    schema_group: Optional[Dict[str, Any]]
    aggregation_type: Optional[str]
    target_fields: Optional[List[str]]
    # Output
    agent_output: Optional[AgentOutput]


def create_sql_agent(
    llm: BaseChatModel,
    custom_prompt: Optional[str] = None
):
    """Create the SQL Agent.

    The SQL Agent uses a single atomic tool (execute_sql_subtask) that
    wraps the entire SQL generation + execution + retry flow.

    This is NOT a ReAct agent - it's a simple graph that:
    1. Extracts task context from state/messages
    2. Calls execute_sql_subtask_core directly
    3. Returns the result as AgentOutput

    Args:
        llm: Language model (not heavily used since execution is deterministic)
        custom_prompt: Optional custom prompt (for compatibility)

    Returns:
        Compiled LangGraph agent
    """

    def execute_sql_node(state: SQLAgentState) -> Dict[str, Any]:
        """Execute SQL subtask and return results.

        This node directly calls the SQL executor without LLM orchestration.
        The LLM is only used internally by the executor for SQL generation.
        """
        logger.info("[SQLAgent] Starting SQL execution node")

        try:
            # Extract task context from state
            task_id = state.get("task_id", "unknown") if state else "unknown"
            task_description = state.get("task_description", "") if state else ""
            document_ids = state.get("document_ids", []) if state else []
            schema_type = state.get("schema_type") if state else None
            schema_group = state.get("schema_group") if state else None
            aggregation_type = state.get("aggregation_type") if state else None
            target_fields = state.get("target_fields") if state else None
            user_query = state.get("user_query", "") if state else ""

            # If no task context in state, try to parse from messages
            if not task_description or not document_ids:
                messages = state.get("messages", []) if state else []
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        content = msg.content
                        # Try to parse structured task message
                        parsed = _parse_task_message(content)
                        if parsed:
                            task_id = parsed.get("task_id", task_id)
                            task_description = parsed.get("description", task_description)
                            document_ids = parsed.get("document_ids", document_ids)
                            schema_type = parsed.get("schema_type", schema_type)
                            schema_group = parsed.get("schema_group", schema_group)
                            aggregation_type = parsed.get("aggregation_type", aggregation_type)
                            target_fields = parsed.get("target_fields", target_fields)
                            break

            if not document_ids:
                logger.warning("[SQLAgent] No document IDs found in state or messages")
                error_output = AgentOutput(
                    task_id=task_id,
                    agent_name="sql_agent",
                    status="failed",
                    data=None,
                    documents_used=[],
                    schema_type=schema_type or "unknown",
                    confidence=0.0,
                    issues=["No document IDs provided for SQL execution"]
                )
                return {
                    "agent_output": error_output,
                    "messages": [AIMessage(content="SQL task failed: No document IDs provided")]
                }

            logger.info(f"[SQLAgent] Executing task '{task_id}' on {len(document_ids)} documents")
            if task_description:
                logger.info(f"[SQLAgent] Description: {task_description[:100]}...")

            # Get aggregation hint from schema_group if not provided directly
            agg_hint = aggregation_type
            if not agg_hint and schema_group:
                # Check if aggregation info is in schema_group
                if isinstance(schema_group, dict):
                    agg_hint = schema_group.get("aggregation_type")

            # Get target fields from schema_group if not provided
            fields = target_fields
            if not fields and schema_group:
                if isinstance(schema_group, dict):
                    fields = schema_group.get("common_fields", [])

            # Execute the SQL subtask using the core function (no LLM orchestration)
            result = execute_sql_subtask_core(
                task_id=task_id,
                sub_task_description=task_description,
                user_query=user_query,
                document_ids=document_ids,
                schema_type=schema_type,
                target_fields=fields,
                aggregation_hint=agg_hint
            )

            # Convert result to AgentOutput
            agent_output = create_agent_output_from_result(result)

            # Create response message
            if result.get("success"):
                row_count = result.get("row_count", 0)
                response = f"SQL task '{task_id}' completed successfully. Retrieved {row_count} rows."
            else:
                error = result.get("error", "Unknown error")
                response = f"SQL task '{task_id}' failed: {error}"

            logger.info(f"[SQLAgent] Task completed: {response}")

            return {
                "agent_output": agent_output,
                "messages": [AIMessage(content=response)]
            }

        except Exception as e:
            logger.error(f"[SQLAgent] Unexpected error in execute_sql_node: {e}", exc_info=True)
            error_output = AgentOutput(
                task_id=state.get("task_id", "unknown") if state else "unknown",
                agent_name="sql_agent",
                status="failed",
                data=None,
                documents_used=[],
                schema_type="unknown",
                confidence=0.0,
                issues=[f"Unexpected error: {str(e)}"]
            )
            return {
                "agent_output": error_output,
                "messages": [AIMessage(content=f"SQL task failed with unexpected error: {str(e)}")]
            }

    # Build simple graph
    builder = StateGraph(SQLAgentState)
    builder.add_node("execute_sql", execute_sql_node)
    builder.set_entry_point("execute_sql")
    builder.add_edge("execute_sql", END)

    logger.info("[SQLAgent] Created SQL Agent with atomic execution (no ReAct)")

    return builder.compile()


def _parse_task_message(content: str) -> Optional[Dict[str, Any]]:
    """Parse a structured task message from the retrieval team.

    Expected format (from retrieval_team._build_task_message):
    Task ID: task_1
    Task: Calculate total quantity across all documents
    Document IDs: ["doc1", "doc2"]
    Schema Type: tabular
    Aggregation: sum
    Target Fields: ["quantity"]
    Schema Group: {...}
    """
    if not content:
        return None

    result = {}
    lines = content.strip().split('\n')

    for line in lines:
        if ':' not in line:
            continue

        key, _, value = line.partition(':')
        key = key.strip().lower()
        value = value.strip()

        if key == "task id":
            result["task_id"] = value
        elif key == "task":
            result["description"] = value
        elif key == "document ids":
            try:
                result["document_ids"] = json.loads(value)
            except json.JSONDecodeError:
                # Try comma-separated
                result["document_ids"] = [v.strip().strip('"\'') for v in value.split(',') if v.strip()]
        elif key == "schema type":
            result["schema_type"] = value
        elif key == "aggregation":
            result["aggregation_type"] = value
        elif key == "target fields":
            try:
                result["target_fields"] = json.loads(value)
            except json.JSONDecodeError:
                result["target_fields"] = [v.strip().strip('"\'') for v in value.split(',') if v.strip()]
        elif key == "schema group":
            try:
                result["schema_group"] = json.loads(value)
            except json.JSONDecodeError:
                pass

    return result if result else None


# For backward compatibility - expose the tool for direct use if needed
sql_agent_tools = [execute_sql_subtask]
