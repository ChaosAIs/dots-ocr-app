"""
Main workflow assembly for the Agentic AI Flow system.

This module creates the complete multi-agent workflow using LangGraph,
connecting all agents into a coherent processing pipeline.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Literal, AsyncIterator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from agents.state.schema import AnalyticsAgentState, create_initial_state
from agents.state.models import ReviewDecisionType
from agents.agents import (
    create_planner_agent,
    create_retrieval_team,
    create_reviewer_agent,
    create_summary_agent
)
from agents.config import AGENTIC_CONFIG, AGENT_LLM_CONFIG

logger = logging.getLogger(__name__)


def _log_separator(char: str = "=", length: int = 80) -> str:
    """Create a log separator line."""
    return char * length


def _get_document_filenames(state: dict, max_files: int = 5) -> str:
    """Extract document filenames from state for logging.

    Args:
        state: The agent state containing available_documents
        max_files: Maximum number of filenames to include

    Returns:
        Comma-separated string of filenames, truncated if necessary
    """
    docs = state.get("available_documents", [])
    if not docs:
        return "(no documents)"

    filenames = []
    for doc in docs[:max_files]:
        if hasattr(doc, 'filename'):
            filenames.append(doc.filename)
        elif isinstance(doc, dict):
            filenames.append(doc.get('filename', doc.get('document_id', 'unknown')[:12]))
        else:
            filenames.append(str(doc)[:12])

    result = ", ".join(filenames)
    if len(docs) > max_files:
        result += f" (+{len(docs) - max_files} more)"

    return result


def _extract_final_response_from_messages_helper(messages) -> dict:
    """Extract final_response from tool call results in messages.

    Scans through messages looking for the format_response tool result
    and extracts the final_response, data_sources, and structured_data.

    This is a helper function defined early so it can be used by both
    async and sync agent wrappers.

    Args:
        messages: List of messages from the agent

    Returns:
        Dict with final_response, data_sources, structured_data or None
    """
    import json
    from langchain_core.messages import ToolMessage

    for msg in reversed(messages):  # Check most recent first
        # Check for ToolMessage with format_response result
        if isinstance(msg, ToolMessage) or (hasattr(msg, 'type') and msg.type == 'tool'):
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                try:
                    content = json.loads(msg.content)
                    if isinstance(content, dict):
                        # Check for final_response in the response
                        if content.get("final_response"):
                            logger.info("[ResponseExtractor] Found final_response in ToolMessage")
                            return {
                                "final_response": content.get("final_response", ""),
                                "data_sources": content.get("data_sources", []),
                                "structured_data": content.get("structured_data", {})
                            }
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug(f"[ResponseExtractor] Could not parse message content: {e}")
                    continue

        # Check regular message content that might contain JSON
        if hasattr(msg, 'content') and isinstance(msg.content, str) and 'final_response' in msg.content:
            try:
                content = msg.content
                start_idx = content.find('{')
                if start_idx != -1:
                    brace_count = 0
                    end_idx = start_idx
                    for i, char in enumerate(content[start_idx:], start_idx):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    if end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict) and parsed.get("final_response"):
                            logger.info("[ResponseExtractor] Found final_response in message content JSON")
                            return {
                                "final_response": parsed.get("final_response", ""),
                                "data_sources": parsed.get("data_sources", []),
                                "structured_data": parsed.get("structured_data", {})
                            }
            except (json.JSONDecodeError, TypeError):
                pass

    logger.warning("[ResponseExtractor] No final_response found in any messages")
    return None


def _wrap_agent_with_logging(agent, agent_name: str):
    """Wrap an agent with detailed logging.

    Args:
        agent: The LangGraph agent to wrap
        agent_name: Human-readable name for logging

    Returns:
        Wrapped agent function with logging
    """
    async def logged_agent(state: AnalyticsAgentState):
        start_time = time.time()

        # Extract key info from state for logging
        user_query = state.get("user_query", "")[:100]
        iteration = state.get("review_iteration", 0)

        logger.info(_log_separator("="))
        logger.info(f"[{agent_name}] ========== START ==========")
        logger.info(_log_separator("="))
        logger.info(f"[{agent_name}] Query: {user_query}...")

        # For SummaryAgent, show documents used by agents, not all available
        if agent_name == "SummaryAgent":
            outputs = state.get("agent_outputs", [])
            docs_used_ids = set()
            for output in outputs:
                used = output.documents_used if hasattr(output, 'documents_used') else output.get('documents_used', []) if isinstance(output, dict) else []
                docs_used_ids.update(used)
            logger.info(f"[{agent_name}] Documents Used: {len(docs_used_ids)} document(s)")
        else:
            doc_filenames = _get_document_filenames(state)
            logger.info(f"[{agent_name}] Documents: {doc_filenames}")

        logger.info(f"[{agent_name}] Review Iteration: {iteration}")

        # Log additional context based on agent type
        if agent_name == "PlannerAgent":
            doc_sources = state.get("available_documents", [])
            logger.info(f"[{agent_name}] Available Documents: {len(doc_sources)} documents")
        elif agent_name == "RetrievalTeam":
            plan = state.get("execution_plan")
            if plan:
                sub_tasks = plan.sub_tasks if hasattr(plan, 'sub_tasks') else plan.get('sub_tasks', [])
                logger.info(f"[{agent_name}] Execution Plan: {len(sub_tasks)} sub-tasks")
                for i, task in enumerate(sub_tasks[:5]):  # Log first 5 tasks
                    task_desc = task.description if hasattr(task, 'description') else task.get('description', '')
                    target = task.target_agent if hasattr(task, 'target_agent') else task.get('target_agent', '')
                    logger.info(f"[{agent_name}]   Task {i+1}: {task_desc[:50]}... -> {target}")
        elif agent_name == "ReviewerAgent":
            outputs = state.get("agent_outputs", [])
            plan = state.get("execution_plan")
            logger.info(f"[{agent_name}] Agent Outputs to Review: {len(outputs)}")
            # Log planned documents (from execution plan), not all available
            if plan:
                sub_tasks = plan.sub_tasks if hasattr(plan, 'sub_tasks') else plan.get('sub_tasks', [])
                planned_doc_ids = set()
                for task in sub_tasks:
                    doc_ids = task.document_ids if hasattr(task, 'document_ids') else task.get('document_ids', [])
                    planned_doc_ids.update(doc_ids)
                logger.info(f"[{agent_name}] Planned Document IDs: {len(planned_doc_ids)}")
            # Log documents actually used by agents
            docs_used = set()
            for output in outputs:
                used = output.documents_used if hasattr(output, 'documents_used') else output.get('documents_used', []) if isinstance(output, dict) else []
                docs_used.update(used)
            logger.info(f"[{agent_name}] Documents Actually Used: {len(docs_used)}")
        elif agent_name == "SummaryAgent":
            outputs = state.get("agent_outputs", [])
            data_sources = state.get("data_sources", [])
            # Log documents actually used by agents, not all available documents
            docs_used = set()
            for output in outputs:
                used = output.documents_used if hasattr(output, 'documents_used') else output.get('documents_used', []) if isinstance(output, dict) else []
                docs_used.update(used)
            logger.info(f"[{agent_name}] Agent Outputs: {len(outputs)}")
            logger.info(f"[{agent_name}] Data Sources: {len(data_sources)}")
            logger.info(f"[{agent_name}] Documents Actually Used: {len(docs_used)}")

        logger.info(_log_separator("-"))
        logger.info(f"[{agent_name}] Executing agent logic...")

        try:
            # Execute the actual agent
            result = await agent.ainvoke(state)

            elapsed_time = time.time() - start_time

            logger.info(_log_separator("-"))
            logger.info(f"[{agent_name}] Execution completed in {elapsed_time:.2f}s")

            # Log result details based on agent type
            if agent_name == "PlannerAgent":
                plan = result.get("execution_plan")
                if plan:
                    sub_tasks = plan.sub_tasks if hasattr(plan, 'sub_tasks') else plan.get('sub_tasks', [])
                    logger.info(f"[{agent_name}] Generated Plan: {len(sub_tasks)} sub-tasks")
                schema_groups = result.get("schema_groups", [])
                logger.info(f"[{agent_name}] Schema Groups: {len(schema_groups)}")
            elif agent_name == "RetrievalTeam":
                outputs = result.get("agent_outputs", [])
                logger.info(f"[{agent_name}] Generated Outputs: {len(outputs)}")
                for i, output in enumerate(outputs[:3]):  # Log first 3 outputs
                    out_agent = output.agent_name if hasattr(output, 'agent_name') else output.get('agent_name', '') if isinstance(output, dict) else ''
                    out_status = output.status if hasattr(output, 'status') else output.get('status', '') if isinstance(output, dict) else ''
                    logger.info(f"[{agent_name}]   Output {i+1}: {out_agent} - Status: {out_status}")
            elif agent_name == "ReviewerAgent":
                decision = result.get("review_decision")
                if decision:
                    dec_type = decision.decision if hasattr(decision, 'decision') else getattr(decision, 'decision', '')
                    reason = decision.reasoning if hasattr(decision, 'reasoning') else getattr(decision, 'reasoning', '')
                    logger.info(f"[{agent_name}] Decision: {dec_type}")
                    logger.info(f"[{agent_name}] Reason: {reason[:100] if reason else 'N/A'}...")
            elif agent_name == "SummaryAgent":
                # CRITICAL: Extract final_response from messages and add to result (async wrapper)
                response = result.get("final_response", "")
                if not response:
                    messages = result.get("messages", [])
                    extracted_response = _extract_final_response_from_messages_helper(messages)
                    if extracted_response:
                        result["final_response"] = extracted_response.get("final_response", "")
                        result["data_sources"] = extracted_response.get("data_sources", [])
                        result["structured_data"] = extracted_response.get("structured_data", {})
                        response = result["final_response"]
                        logger.info(f"[{agent_name}] Extracted final_response from tool messages (async)")

                logger.info(f"[{agent_name}] Final Response Length: {len(response)} chars")
                if response:
                    logger.info(f"[{agent_name}] Response Preview: {response[:200]}...")

            logger.info(_log_separator("="))
            logger.info(f"[{agent_name}] ========== END (Success) ==========")
            logger.info(_log_separator("="))

            return result

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(_log_separator("!"))
            logger.error(f"[{agent_name}] ========== ERROR ==========")
            logger.error(f"[{agent_name}] Error after {elapsed_time:.2f}s: {str(e)}")
            logger.error(_log_separator("!"))
            raise

    return logged_agent


def _extract_execution_plan_from_messages(messages) -> dict:
    """Extract execution_plan from tool call results in messages.

    Scans through messages looking for the create_execution_plan tool result
    and extracts the execution_plan data.

    Args:
        messages: List of messages from the agent

    Returns:
        Extracted execution_plan dict or None
    """
    import json
    from langchain_core.messages import ToolMessage

    for msg in reversed(messages):  # Check most recent first
        # Check for ToolMessage with execution plan result
        if isinstance(msg, ToolMessage) or (hasattr(msg, 'type') and msg.type == 'tool'):
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                try:
                    content = json.loads(msg.content)
                    if isinstance(content, dict):
                        # Check for execution_plan in the response
                        if content.get("execution_plan"):
                            logger.info("[PlanExtractor] Found execution_plan in ToolMessage")
                            return content["execution_plan"]
                        # Also check if success=True indicates valid plan
                        if content.get("success") and content.get("task_count", 0) > 0:
                            logger.info("[PlanExtractor] Found successful plan creation, extracting plan")
                            return content.get("execution_plan")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug(f"[PlanExtractor] Could not parse message content: {e}")
                    continue

        # Check for AIMessage with tool_calls containing plan arguments
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.get('name') == 'create_execution_plan':
                    logger.info("[PlanExtractor] Found create_execution_plan tool call, checking args")
                    args = tool_call.get('args', {})
                    if args:
                        # The args contain sub_tasks, schema_groups, etc. as JSON strings
                        # We need to parse them to reconstruct the plan
                        try:
                            sub_tasks_str = args.get('sub_tasks', '[]')
                            schema_groups_str = args.get('schema_groups', '[]')
                            execution_strategy = args.get('execution_strategy', 'parallel')
                            reasoning = args.get('reasoning', '')

                            # Parse JSON strings
                            sub_tasks = json.loads(sub_tasks_str) if isinstance(sub_tasks_str, str) else sub_tasks_str
                            schema_groups = json.loads(schema_groups_str) if isinstance(schema_groups_str, str) else schema_groups_str

                            if sub_tasks:
                                logger.info(f"[PlanExtractor] Reconstructed plan from tool call args: {len(sub_tasks)} tasks")
                                return {
                                    "sub_tasks": sub_tasks,
                                    "schema_groups": schema_groups,
                                    "execution_strategy": execution_strategy,
                                    "total_documents": sum(len(t.get('document_ids', [])) for t in sub_tasks),
                                    "documents_by_schema": {},
                                    "reasoning": reasoning
                                }
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"[PlanExtractor] Failed to parse tool call args: {e}")
                            continue

        # Also check regular message content that might contain JSON
        if hasattr(msg, 'content') and isinstance(msg.content, str) and 'execution_plan' in msg.content:
            try:
                # Try to find JSON in the message
                content = msg.content
                # Look for JSON object containing execution_plan
                start_idx = content.find('{')
                if start_idx != -1:
                    # Find matching closing brace
                    brace_count = 0
                    end_idx = start_idx
                    for i, char in enumerate(content[start_idx:], start_idx):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    if end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict) and parsed.get("execution_plan"):
                            logger.info("[PlanExtractor] Found execution_plan in message content JSON")
                            return parsed["execution_plan"]
            except (json.JSONDecodeError, TypeError):
                pass

    logger.warning("[PlanExtractor] No execution_plan found in any messages")
    return None


def _wrap_sync_agent_with_logging(agent, agent_name: str):
    """Wrap a synchronous agent with detailed logging.

    Args:
        agent: The LangGraph agent to wrap
        agent_name: Human-readable name for logging

    Returns:
        Wrapped agent function with logging
    """
    def logged_agent(state: AnalyticsAgentState):
        start_time = time.time()

        # Extract key info from state for logging
        user_query = state.get("user_query", "")[:100]
        iteration = state.get("review_iteration", 0)

        logger.info(_log_separator("="))
        logger.info(f"[{agent_name}] ========== START ==========")
        logger.info(_log_separator("="))
        logger.info(f"[{agent_name}] Query: {user_query}...")

        # For SummaryAgent, show documents used by agents, not all available
        if agent_name == "SummaryAgent":
            outputs = state.get("agent_outputs", [])
            docs_used_ids = set()
            for output in outputs:
                used = output.documents_used if hasattr(output, 'documents_used') else output.get('documents_used', []) if isinstance(output, dict) else []
                docs_used_ids.update(used)
            logger.info(f"[{agent_name}] Documents Used: {len(docs_used_ids)} document(s)")
        else:
            doc_filenames = _get_document_filenames(state)
            logger.info(f"[{agent_name}] Documents: {doc_filenames}")

        logger.info(f"[{agent_name}] Review Iteration: {iteration}")

        # Log additional context based on agent type
        if agent_name == "PlannerAgent":
            doc_sources = state.get("available_documents", [])
            logger.info(f"[{agent_name}] Available Documents: {len(doc_sources)} documents")
        elif agent_name == "RetrievalTeam":
            plan = state.get("execution_plan")
            if plan:
                sub_tasks = plan.sub_tasks if hasattr(plan, 'sub_tasks') else plan.get('sub_tasks', [])
                logger.info(f"[{agent_name}] Execution Plan: {len(sub_tasks)} sub-tasks")
                for i, task in enumerate(sub_tasks[:5]):  # Log first 5 tasks
                    task_desc = task.description if hasattr(task, 'description') else task.get('description', '')
                    target = task.target_agent if hasattr(task, 'target_agent') else task.get('target_agent', '')
                    logger.info(f"[{agent_name}]   Task {i+1}: {task_desc[:50]}... -> {target}")
            else:
                logger.warning(f"[{agent_name}] NO EXECUTION PLAN IN STATE!")
        elif agent_name == "ReviewerAgent":
            outputs = state.get("agent_outputs", [])
            plan = state.get("execution_plan")
            logger.info(f"[{agent_name}] Agent Outputs to Review: {len(outputs)}")
            # Log planned documents (from execution plan), not all available
            if plan:
                sub_tasks = plan.sub_tasks if hasattr(plan, 'sub_tasks') else plan.get('sub_tasks', [])
                planned_doc_ids = set()
                for task in sub_tasks:
                    doc_ids = task.document_ids if hasattr(task, 'document_ids') else task.get('document_ids', [])
                    planned_doc_ids.update(doc_ids)
                logger.info(f"[{agent_name}] Planned Document IDs: {len(planned_doc_ids)}")
            # Log documents actually used by agents
            docs_used = set()
            for output in outputs:
                used = output.documents_used if hasattr(output, 'documents_used') else output.get('documents_used', []) if isinstance(output, dict) else []
                docs_used.update(used)
            logger.info(f"[{agent_name}] Documents Actually Used: {len(docs_used)}")
        elif agent_name == "SummaryAgent":
            outputs = state.get("agent_outputs", [])
            data_sources = state.get("data_sources", [])
            # Log documents actually used by agents, not all available documents
            docs_used = set()
            for output in outputs:
                used = output.documents_used if hasattr(output, 'documents_used') else output.get('documents_used', []) if isinstance(output, dict) else []
                docs_used.update(used)
            logger.info(f"[{agent_name}] Agent Outputs: {len(outputs)}")
            logger.info(f"[{agent_name}] Data Sources: {len(data_sources)}")
            logger.info(f"[{agent_name}] Documents Actually Used: {len(docs_used)}")

        logger.info(_log_separator("-"))
        logger.info(f"[{agent_name}] Executing agent logic...")

        try:
            # Execute the actual agent
            result = agent.invoke(state)

            elapsed_time = time.time() - start_time

            logger.info(_log_separator("-"))
            logger.info(f"[{agent_name}] Execution completed in {elapsed_time:.2f}s")

            # Log result details based on agent type
            if agent_name == "PlannerAgent":
                # CRITICAL: Extract execution_plan from messages and add to result
                plan = result.get("execution_plan")
                if not plan:
                    # Try to extract from messages
                    messages = result.get("messages", [])
                    extracted_plan = _extract_execution_plan_from_messages(messages)
                    if extracted_plan:
                        from agents.state.models import ExecutionPlan, SubTask, SchemaGroup
                        # Convert dict to ExecutionPlan model
                        sub_tasks = [SubTask(**t) for t in extracted_plan.get("sub_tasks", [])]
                        schema_groups = [SchemaGroup(**g) for g in extracted_plan.get("schema_groups", [])]
                        plan = ExecutionPlan(
                            sub_tasks=sub_tasks,
                            schema_groups=schema_groups,
                            execution_strategy=extracted_plan.get("execution_strategy", "parallel"),
                            total_documents=extracted_plan.get("total_documents", 0),
                            documents_by_schema=extracted_plan.get("documents_by_schema", {}),
                            reasoning=extracted_plan.get("reasoning", "")
                        )
                        result["execution_plan"] = plan
                        result["schema_groups"] = schema_groups
                        logger.info(f"[{agent_name}] Extracted and stored execution_plan with {len(sub_tasks)} sub-tasks")

                if plan:
                    sub_tasks = plan.sub_tasks if hasattr(plan, 'sub_tasks') else plan.get('sub_tasks', [])
                    logger.info(f"[{agent_name}] Generated Plan: {len(sub_tasks)} sub-tasks")
                else:
                    logger.warning(f"[{agent_name}] No execution_plan found in result!")

                schema_groups = result.get("schema_groups", [])
                logger.info(f"[{agent_name}] Schema Groups: {len(schema_groups)}")
            elif agent_name == "RetrievalTeam":
                outputs = result.get("agent_outputs", [])
                logger.info(f"[{agent_name}] Generated Outputs: {len(outputs)}")
                for i, output in enumerate(outputs[:3]):  # Log first 3 outputs
                    out_agent = output.agent_name if hasattr(output, 'agent_name') else output.get('agent_name', '') if isinstance(output, dict) else ''
                    out_status = output.status if hasattr(output, 'status') else output.get('status', '') if isinstance(output, dict) else ''
                    logger.info(f"[{agent_name}]   Output {i+1}: {out_agent} - Status: {out_status}")
            elif agent_name == "ReviewerAgent":
                decision = result.get("review_decision")
                if decision:
                    dec_type = decision.decision if hasattr(decision, 'decision') else getattr(decision, 'decision', '')
                    reason = decision.reasoning if hasattr(decision, 'reasoning') else getattr(decision, 'reasoning', '')
                    logger.info(f"[{agent_name}] Decision: {dec_type}")
                    logger.info(f"[{agent_name}] Reason: {reason[:100] if reason else 'N/A'}...")
            elif agent_name == "SummaryAgent":
                # CRITICAL: Extract final_response from messages and add to result
                response = result.get("final_response", "")
                if not response:
                    # Try to extract from messages
                    messages = result.get("messages", [])
                    extracted_response = _extract_final_response_from_messages_helper(messages)
                    if extracted_response:
                        result["final_response"] = extracted_response.get("final_response", "")
                        result["data_sources"] = extracted_response.get("data_sources", [])
                        result["structured_data"] = extracted_response.get("structured_data", {})
                        response = result["final_response"]
                        logger.info(f"[{agent_name}] Extracted final_response from tool messages")

                logger.info(f"[{agent_name}] Final Response Length: {len(response)} chars")
                if response:
                    logger.info(f"[{agent_name}] Response Preview: {response[:200]}...")

            logger.info(_log_separator("="))
            logger.info(f"[{agent_name}] ========== END (Success) ==========")
            logger.info(_log_separator("="))

            return result

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(_log_separator("!"))
            logger.error(f"[{agent_name}] ========== ERROR ==========")
            logger.error(f"[{agent_name}] Error after {elapsed_time:.2f}s: {str(e)}")
            logger.error(_log_separator("!"))
            raise

    return logged_agent


def create_analytics_workflow(
    llm: Optional[BaseChatModel] = None,
    enable_checkpointing: bool = True,
    enable_graph_search: bool = True
):
    """Create the complete multi-agent analytics workflow.

    This function assembles all agents into a LangGraph workflow:
    1. Planner Agent - Query decomposition and planning
    2. Retrieval Team - Coordinated data retrieval
    3. Reviewer Agent - Quality control
    4. Summary Agent - Response synthesis

    Args:
        llm: Language model to use (creates default if not provided)
        enable_checkpointing: Whether to enable state checkpointing
        enable_graph_search: Whether to enable graph search agent (from UI toggle)

    Returns:
        Compiled LangGraph workflow
    """
    logger.info(_log_separator("="))
    logger.info("[WorkflowBuilder] ========== CREATING ANALYTICS WORKFLOW ==========")
    logger.info(_log_separator("="))

    # Create default LLM if not provided
    if llm is None:
        logger.info("[WorkflowBuilder] No LLM provided, creating default ChatOpenAI...")
        llm = ChatOpenAI(
            model=AGENT_LLM_CONFIG["planner"]["model"],
            temperature=AGENT_LLM_CONFIG["planner"]["temperature"],
            request_timeout=AGENTIC_CONFIG.get("planner_timeout", 60),  # Prevent indefinite hanging
            max_retries=2
        )
        logger.info(f"[WorkflowBuilder] Created LLM: model={AGENT_LLM_CONFIG['planner']['model']}, timeout={AGENTIC_CONFIG.get('planner_timeout', 60)}s")
    else:
        logger.info(f"[WorkflowBuilder] Using provided LLM: {type(llm).__name__}")

    # Create all agents with logging wrappers
    logger.info(_log_separator("-"))
    logger.info("[WorkflowBuilder] Creating agents...")
    logger.info("[WorkflowBuilder] Creating Planner Agent...")
    planner = create_planner_agent(llm)
    logger.info("[WorkflowBuilder] ✓ Planner Agent created")

    logger.info("[WorkflowBuilder] Creating Retrieval Team...")
    logger.info(f"[WorkflowBuilder] Graph Search Enabled: {enable_graph_search}")
    retrieval_team = create_retrieval_team(llm, enable_graph_search=enable_graph_search)
    logger.info("[WorkflowBuilder] ✓ Retrieval Team created")

    logger.info("[WorkflowBuilder] Creating Reviewer Agent...")
    reviewer = create_reviewer_agent(llm)
    logger.info("[WorkflowBuilder] ✓ Reviewer Agent created")

    logger.info("[WorkflowBuilder] Creating Summary Agent...")
    summarizer = create_summary_agent(llm)
    logger.info("[WorkflowBuilder] ✓ Summary Agent created")

    # Wrap agents with logging (for sync execution)
    logger.info(_log_separator("-"))
    logger.info("[WorkflowBuilder] Wrapping agents with logging...")
    planner_logged = _wrap_sync_agent_with_logging(planner, "PlannerAgent")
    retrieval_logged = _wrap_sync_agent_with_logging(retrieval_team, "RetrievalTeam")
    reviewer_logged = _wrap_sync_agent_with_logging(reviewer, "ReviewerAgent")
    summarizer_logged = _wrap_sync_agent_with_logging(summarizer, "SummaryAgent")
    logger.info("[WorkflowBuilder] ✓ All agents wrapped with logging")

    # Build the workflow graph
    logger.info(_log_separator("-"))
    logger.info("[WorkflowBuilder] Building workflow graph...")
    builder = StateGraph(AnalyticsAgentState)

    # Add nodes with logging wrappers
    builder.add_node("planner_agent", planner_logged)
    builder.add_node("retrieval_team", retrieval_logged)
    builder.add_node("reviewer_agent", reviewer_logged)
    builder.add_node("summary_agent", summarizer_logged)
    logger.info("[WorkflowBuilder] ✓ Added 4 nodes to graph")

    # Define edges
    logger.info("[WorkflowBuilder] Defining workflow edges...")
    builder.add_edge(START, "planner_agent")
    logger.info("[WorkflowBuilder]   START -> planner_agent")
    builder.add_edge("planner_agent", "retrieval_team")
    logger.info("[WorkflowBuilder]   planner_agent -> retrieval_team")
    builder.add_edge("retrieval_team", "reviewer_agent")
    logger.info("[WorkflowBuilder]   retrieval_team -> reviewer_agent")

    # Conditional routing from reviewer
    def route_after_review(state: AnalyticsAgentState) -> Literal["summary_agent", "retrieval_team"]:
        """Determine next step based on review decision."""
        logger.info(_log_separator("-"))
        logger.info("[Router] ========== ROUTING DECISION ==========")

        decision = state.get("review_decision")
        iteration = state.get("review_iteration", 0)
        max_iter = state.get("max_iterations", AGENTIC_CONFIG["max_review_iterations"])

        logger.info(f"[Router] Current iteration: {iteration}/{max_iter}")

        if decision:
            # Handle both model and dict forms
            if hasattr(decision, 'decision'):
                decision_type = decision.decision
                reason = getattr(decision, 'reasoning', 'No reason')
            else:
                decision_type = decision.get("decision") if isinstance(decision, dict) else getattr(decision, 'decision', '')
                reason = decision.get("reasoning", "No reason") if isinstance(decision, dict) else getattr(decision, 'reasoning', 'No reason')

            logger.info(f"[Router] Decision type: {decision_type}")
            logger.info(f"[Router] Reason: {reason[:100]}...")

            if decision_type == ReviewDecisionType.REFINE or decision_type == "refine":
                if iteration < max_iter:
                    logger.info(f"[Router] → Routing back to RETRIEVAL_TEAM for refinement (iteration {iteration + 1})")
                    logger.info(_log_separator("-"))
                    return "retrieval_team"
                else:
                    logger.info(f"[Router] Max iterations reached ({max_iter}), proceeding to summary")

        logger.info("[Router] → Routing to SUMMARY_AGENT")
        logger.info(_log_separator("-"))
        return "summary_agent"

    builder.add_conditional_edges(
        "reviewer_agent",
        route_after_review,
        {
            "summary_agent": "summary_agent",
            "retrieval_team": "retrieval_team"
        }
    )
    logger.info("[WorkflowBuilder]   reviewer_agent -> (conditional) summary_agent OR retrieval_team")

    builder.add_edge("summary_agent", END)
    logger.info("[WorkflowBuilder]   summary_agent -> END")
    logger.info("[WorkflowBuilder] ✓ All edges defined")

    # Compile with optional checkpointing
    logger.info(_log_separator("-"))
    logger.info("[WorkflowBuilder] Compiling workflow...")
    compile_kwargs = {}
    if enable_checkpointing:
        checkpointer = MemorySaver()
        compile_kwargs["checkpointer"] = checkpointer
        logger.info("[WorkflowBuilder] Checkpointing enabled with MemorySaver")

    workflow = builder.compile(**compile_kwargs)

    logger.info(_log_separator("="))
    logger.info("[WorkflowBuilder] ========== WORKFLOW CREATED SUCCESSFULLY ==========")
    logger.info(_log_separator("="))
    return workflow


def execute_analytics_query(
    user_query: str,
    workspace_id: str,
    llm: Optional[BaseChatModel] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    max_iterations: int = 3,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute an analytics query through the multi-agent workflow.

    This is the main entry point for synchronous query execution.

    Args:
        user_query: The user's natural language query
        workspace_id: Workspace ID for document access
        llm: Language model to use
        chat_history: Optional previous conversation history
        max_iterations: Maximum review iterations
        config: Optional LangGraph config

    Returns:
        Dictionary with response, data_sources, and metadata
    """
    import uuid

    start_time = time.time()

    logger.info(_log_separator("="))
    logger.info("[ExecuteQuery] ========== SYNC QUERY EXECUTION START ==========")
    logger.info(_log_separator("="))
    logger.info(f"[ExecuteQuery] Query: {user_query[:100]}...")
    logger.info(f"[ExecuteQuery] Workspace ID: {workspace_id}")
    logger.info(f"[ExecuteQuery] Max Iterations: {max_iterations}")
    logger.info(f"[ExecuteQuery] Chat History: {len(chat_history) if chat_history else 0} messages")

    # Ensure config has thread_id for checkpointing
    if config is None:
        config = {}
    if "configurable" not in config:
        config["configurable"] = {}
    if "thread_id" not in config["configurable"]:
        config["configurable"]["thread_id"] = f"analytics_{uuid.uuid4().hex[:12]}"
    logger.info(f"[ExecuteQuery] Thread ID: {config['configurable']['thread_id']}")

    # Create workflow
    logger.info(_log_separator("-"))
    logger.info("[ExecuteQuery] Creating workflow...")
    workflow = create_analytics_workflow(llm)

    # Create initial state
    logger.info("[ExecuteQuery] Creating initial state...")
    initial_state = create_initial_state(
        user_query=user_query,
        workspace_id=workspace_id,
        chat_history=chat_history,
        max_iterations=max_iterations
    )
    logger.info("[ExecuteQuery] ✓ Initial state created")

    # Execute workflow
    logger.info(_log_separator("-"))
    logger.info("[ExecuteQuery] Invoking workflow...")

    try:
        final_state = workflow.invoke(initial_state, config=config)

        elapsed_time = time.time() - start_time

        # Extract results
        result = {
            "success": True,
            "response": final_state.get("final_response"),
            "data_sources": final_state.get("data_sources", []),
            "structured_data": final_state.get("structured_data", {}),
            "execution_plan": _serialize_plan(final_state.get("execution_plan")),
            "agent_outputs": _serialize_outputs(final_state.get("agent_outputs", [])),
            "review_iterations": final_state.get("review_iteration", 0),
            "error": final_state.get("error")
        }

        logger.info(_log_separator("-"))
        logger.info("[ExecuteQuery] ========== QUERY EXECUTION RESULTS ==========")
        logger.info(f"[ExecuteQuery] Success: True")
        logger.info(f"[ExecuteQuery] Response Length: {len(result['response'] or '')} chars")
        logger.info(f"[ExecuteQuery] Data Sources: {len(result['data_sources'])}")
        logger.info(f"[ExecuteQuery] Agent Outputs: {len(result['agent_outputs'])}")
        logger.info(f"[ExecuteQuery] Review Iterations: {result['review_iterations']}")
        logger.info(f"[ExecuteQuery] Total Execution Time: {elapsed_time:.2f}s")
        logger.info(_log_separator("="))
        logger.info("[ExecuteQuery] ========== SYNC QUERY EXECUTION END (Success) ==========")
        logger.info(_log_separator("="))

        return result

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(_log_separator("!"))
        logger.error("[ExecuteQuery] ========== QUERY EXECUTION ERROR ==========")
        logger.error(f"[ExecuteQuery] Error after {elapsed_time:.2f}s: {str(e)}")
        logger.error(_log_separator("!"))
        return {
            "success": False,
            "response": f"An error occurred: {e}",
            "data_sources": [],
            "error": str(e)
        }


def _build_status_details(active_agent: str, event: Dict[str, Any], last_details: Dict[str, Any]) -> Dict[str, Any]:
    """Build detailed status information based on current agent and state.

    This extracts meaningful status information from the agent state to provide
    real-time feedback to the frontend about what the agent is doing.

    Args:
        active_agent: The currently active agent name
        event: Current state event from the workflow
        last_details: Previous status details to track changes

    Returns:
        Dictionary with status details for the current agent
    """
    details = {
        "agent": active_agent,
        "phase": "",
        "message": "",
        "sub_messages": [],
        "metrics": {}
    }

    # Extract common state info
    available_docs = event.get("available_documents", [])
    doc_count = len(available_docs)

    if active_agent == "planner_agent":
        details["phase"] = "planning"
        details["message"] = "Analyzing your question and creating an execution plan"

        # Check if we have execution plan
        plan = event.get("execution_plan")
        if plan:
            sub_tasks = plan.sub_tasks if hasattr(plan, 'sub_tasks') else plan.get('sub_tasks', [])
            schema_groups = event.get("schema_groups", [])
            details["sub_messages"] = [
                f"Created {len(sub_tasks)} sub-tasks for execution",
                f"Identified {len(schema_groups)} document groups by schema"
            ]
            details["metrics"]["sub_task_count"] = len(sub_tasks)
            details["metrics"]["schema_group_count"] = len(schema_groups)
        else:
            details["sub_messages"] = [
                f"Examining {doc_count} available documents",
                "Determining the best approach for your query"
            ]
            details["metrics"]["available_docs"] = doc_count

    elif active_agent == "retrieval_team":
        details["phase"] = "retrieval"
        iteration = event.get("review_iteration", 0)
        details["message"] = f"Retrieving and analyzing data (iteration {iteration + 1})"

        # Get execution plan details
        plan = event.get("execution_plan")
        agent_outputs = event.get("agent_outputs", [])

        if plan:
            sub_tasks = plan.sub_tasks if hasattr(plan, 'sub_tasks') else plan.get('sub_tasks', [])
            completed_tasks = len(agent_outputs)
            total_tasks = len(sub_tasks)

            # Describe what's being executed
            task_types = {}
            for task in sub_tasks:
                target = task.target_agent if hasattr(task, 'target_agent') else task.get('target_agent', '')
                task_types[target] = task_types.get(target, 0) + 1

            task_descriptions = []
            if task_types.get("sql_agent", 0) > 0:
                task_descriptions.append(f"Running {task_types['sql_agent']} SQL analytics queries")
            if task_types.get("rag_agent", 0) > 0:
                task_descriptions.append(f"Searching {task_types['rag_agent']} document contexts")
            if task_types.get("graph_search_agent", 0) > 0:
                task_descriptions.append(f"Exploring {task_types['graph_search_agent']} knowledge graph connections")

            details["sub_messages"] = task_descriptions if task_descriptions else [
                f"Executing {total_tasks} analysis tasks"
            ]
            details["metrics"]["total_tasks"] = total_tasks
            details["metrics"]["completed_tasks"] = completed_tasks

        # Add output status if we have results
        if agent_outputs:
            # Count successful outputs (status == "success")
            def is_successful(o):
                status = o.status if hasattr(o, 'status') else o.get('status', '') if isinstance(o, dict) else ''
                return status == "success"
            successful = sum(1 for o in agent_outputs if is_successful(o))
            details["sub_messages"].append(f"Completed {successful}/{len(agent_outputs)} data retrievals")
            details["metrics"]["successful_retrievals"] = successful

    elif active_agent == "reviewer_agent":
        details["phase"] = "review"
        iteration = event.get("review_iteration", 0)
        details["message"] = f"Reviewing data quality (iteration {iteration + 1})"

        agent_outputs = event.get("agent_outputs", [])
        review_decision = event.get("review_decision")

        details["sub_messages"] = [
            f"Analyzing {len(agent_outputs)} data retrieval results",
            "Checking for completeness and accuracy"
        ]
        details["metrics"]["outputs_to_review"] = len(agent_outputs)

        if review_decision:
            decision_type = review_decision.decision if hasattr(review_decision, 'decision') else getattr(review_decision, 'decision', '')
            reason = review_decision.reasoning if hasattr(review_decision, 'reasoning') else getattr(review_decision, 'reasoning', '')
            details["sub_messages"].append(f"Decision: {decision_type}")
            if reason:
                details["sub_messages"].append(f"Reason: {reason[:100]}...")
            details["metrics"]["decision"] = decision_type

    elif active_agent == "summary_agent":
        details["phase"] = "summarizing"
        details["message"] = "Generating your comprehensive report"

        agent_outputs = event.get("agent_outputs", [])
        data_sources = event.get("data_sources", [])

        details["sub_messages"] = [
            f"Synthesizing insights from {len(agent_outputs)} data sources",
            "Formatting response with relevant details"
        ]
        details["metrics"]["data_sources"] = len(data_sources)
        details["metrics"]["agent_outputs"] = len(agent_outputs)

        # Check if we have final response
        final_response = event.get("final_response")
        if final_response:
            details["sub_messages"].append(f"Generated response ({len(final_response)} characters)")
            details["metrics"]["response_length"] = len(final_response)

    else:
        # Generic status for any other agent
        details["phase"] = "processing"
        details["message"] = f"Processing with {active_agent}"
        details["sub_messages"] = [f"Working on your request"]

    return details


async def execute_analytics_query_async(
    user_query: str,
    workspace_id: str,
    llm: Optional[BaseChatModel] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    max_iterations: int = 3,
    config: Optional[Dict[str, Any]] = None,
    stream_progress: bool = True,
    enable_graph_search: bool = True,
    accessible_doc_ids: Optional[List[str]] = None
) -> AsyncIterator[Dict[str, Any]]:
    """Execute an analytics query asynchronously with streaming progress.

    This is the main entry point for async query execution with
    real-time progress updates.

    Args:
        user_query: The user's natural language query
        workspace_id: Workspace ID for document access
        llm: Language model to use
        chat_history: Optional previous conversation history
        max_iterations: Maximum review iterations
        config: Optional LangGraph config
        stream_progress: Whether to yield progress updates
        enable_graph_search: Whether to enable graph search agent (from UI toggle)
        accessible_doc_ids: List of document IDs the user has access to

    Yields:
        Progress updates and final result
    """
    import uuid

    start_time = time.time()
    event_count = 0

    logger.info(_log_separator("="))
    logger.info("[AsyncExecuteQuery] ========== ASYNC QUERY EXECUTION START ==========")
    logger.info(_log_separator("="))
    logger.info(f"[AsyncExecuteQuery] Query: {user_query[:100]}...")
    logger.info(f"[AsyncExecuteQuery] Workspace ID: {workspace_id}")
    logger.info(f"[AsyncExecuteQuery] Max Iterations: {max_iterations}")
    logger.info(f"[AsyncExecuteQuery] Chat History: {len(chat_history) if chat_history else 0} messages")
    logger.info(f"[AsyncExecuteQuery] Stream Progress: {stream_progress}")
    logger.info(f"[AsyncExecuteQuery] Graph Search Enabled: {enable_graph_search}")
    logger.info(f"[AsyncExecuteQuery] Accessible Documents: {len(accessible_doc_ids) if accessible_doc_ids else 0}")

    # Ensure config has thread_id for checkpointing
    if config is None:
        config = {}
    if "configurable" not in config:
        config["configurable"] = {}
    if "thread_id" not in config["configurable"]:
        config["configurable"]["thread_id"] = f"analytics_{uuid.uuid4().hex[:12]}"
    logger.info(f"[AsyncExecuteQuery] Thread ID: {config['configurable']['thread_id']}")

    # Create workflow
    logger.info(_log_separator("-"))
    logger.info("[AsyncExecuteQuery] Creating workflow...")
    workflow = create_analytics_workflow(llm, enable_graph_search=enable_graph_search)

    # Create initial state with accessible documents
    logger.info("[AsyncExecuteQuery] Creating initial state...")
    initial_state = create_initial_state(
        user_query=user_query,
        workspace_id=workspace_id,
        chat_history=chat_history,
        max_iterations=max_iterations,
        accessible_doc_ids=accessible_doc_ids
    )
    logger.info(f"[AsyncExecuteQuery] ✓ Initial state created with {len(initial_state.get('available_documents', []))} documents")

    logger.info(_log_separator("-"))
    logger.info("[AsyncExecuteQuery] Starting async stream execution...")

    try:
        final_state = None
        last_agent = None
        last_status_details = {}  # Track what status info we've already sent

        # Stream execution
        async for event in workflow.astream(initial_state, config=config, stream_mode="values"):
            event_count += 1
            final_state = event

            # Log state changes
            active_agent = event.get("active_agent")
            if active_agent and active_agent != last_agent:
                logger.info(f"[AsyncExecuteQuery] Event #{event_count}: Active Agent changed to '{active_agent}'")
                last_agent = active_agent

            if stream_progress:
                # Yield detailed progress updates with status information
                if active_agent:
                    logger.info(f"[AsyncExecuteQuery] Yielding progress update for '{active_agent}'")

                    # Build detailed status based on current agent and state
                    status_details = _build_status_details(active_agent, event, last_status_details)
                    last_status_details = status_details.copy()

                    yield {
                        "type": "progress",
                        "agent": active_agent,
                        "iteration": event.get("review_iteration", 0),
                        "status_details": status_details
                    }

        elapsed_time = time.time() - start_time

        # Yield final result
        if final_state:
            response_len = len(final_state.get("final_response") or "")
            data_sources_count = len(final_state.get("data_sources", []))
            agent_outputs_count = len(final_state.get("agent_outputs", []))

            logger.info(_log_separator("-"))
            logger.info("[AsyncExecuteQuery] ========== ASYNC QUERY EXECUTION RESULTS ==========")
            logger.info(f"[AsyncExecuteQuery] Success: True")
            logger.info(f"[AsyncExecuteQuery] Total Events Processed: {event_count}")
            logger.info(f"[AsyncExecuteQuery] Response Length: {response_len} chars")
            logger.info(f"[AsyncExecuteQuery] Data Sources: {data_sources_count}")
            logger.info(f"[AsyncExecuteQuery] Agent Outputs: {agent_outputs_count}")
            logger.info(f"[AsyncExecuteQuery] Review Iterations: {final_state.get('review_iteration', 0)}")
            logger.info(f"[AsyncExecuteQuery] Total Execution Time: {elapsed_time:.2f}s")

            yield {
                "type": "result",
                "success": True,
                "response": final_state.get("final_response"),
                "data_sources": final_state.get("data_sources", []),
                "structured_data": final_state.get("structured_data", {}),
                "execution_plan": _serialize_plan(final_state.get("execution_plan")),
                "agent_outputs": _serialize_outputs(final_state.get("agent_outputs", [])),
                "review_iterations": final_state.get("review_iteration", 0),
                "error": final_state.get("error")
            }

        logger.info(_log_separator("="))
        logger.info("[AsyncExecuteQuery] ========== ASYNC QUERY EXECUTION END (Success) ==========")
        logger.info(_log_separator("="))

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(_log_separator("!"))
        logger.error("[AsyncExecuteQuery] ========== ASYNC QUERY EXECUTION ERROR ==========")
        logger.error(f"[AsyncExecuteQuery] Error after {elapsed_time:.2f}s: {str(e)}")
        logger.error(f"[AsyncExecuteQuery] Events processed before error: {event_count}")
        logger.error(_log_separator("!"))
        yield {
            "type": "error",
            "success": False,
            "response": f"An error occurred: {e}",
            "error": str(e)
        }


def _serialize_plan(plan) -> Optional[Dict[str, Any]]:
    """Serialize execution plan for JSON response."""
    if plan is None:
        return None
    if hasattr(plan, 'dict'):
        return plan.dict()
    if isinstance(plan, dict):
        return plan
    return None


def _serialize_outputs(outputs: List) -> List[Dict[str, Any]]:
    """Serialize agent outputs for JSON response."""
    result = []
    for output in outputs:
        if hasattr(output, 'model_dump'):
            result.append(output.model_dump())
        elif hasattr(output, 'dict'):
            result.append(output.dict())
        elif isinstance(output, dict):
            result.append(output)
    return result


# Convenience function for creating workflow with specific LLM configs
def create_workflow_with_config(
    planner_model: str = "gpt-4o",
    retrieval_model: str = "gpt-4o-mini",
    reviewer_model: str = "gpt-4o",
    summary_model: str = "gpt-4o"
):
    """Create workflow with specific model configurations.

    Args:
        planner_model: Model for planner agent
        retrieval_model: Model for retrieval agents
        reviewer_model: Model for reviewer agent
        summary_model: Model for summary agent

    Returns:
        Compiled workflow
    """
    # For now, use a single LLM
    # In production, could create different LLMs for each agent
    llm = ChatOpenAI(model=planner_model, temperature=0.1)
    return create_analytics_workflow(llm)
