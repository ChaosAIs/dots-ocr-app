"""
Retrieval Team Supervisor implementation.

The Retrieval Team coordinates all retrieval agents:
- SQL Agent (atomic execution, no ReAct)
- Vector Agent
- Graph Agent
- Generic Doc Agent

Architecture Note:
The SQL Agent has been simplified to use atomic execution via execute_sql_subtask_core.
It no longer uses ReAct pattern with multiple tool calls, eliminating LLM orchestration
overhead within SQL execution.
"""

import logging
import time
from typing import Optional

from langchain_core.language_models import BaseChatModel

from agents.agents.sql_agent import create_sql_agent
from agents.agents.vector_agent import create_vector_agent
from agents.agents.graph_agent import create_graph_agent
from agents.agents.generic_doc_agent import create_generic_doc_agent
from agents.prompts.retrieval_prompts import RETRIEVAL_SUPERVISOR_PROMPT
from agents.config import AGENTIC_CONFIG

logger = logging.getLogger(__name__)


def _log_separator(char: str = "=", length: int = 80) -> str:
    """Create a log separator line."""
    return char * length


def create_retrieval_team(
    llm: BaseChatModel,
    custom_prompt: Optional[str] = None,
    enable_graph_search: bool = True
):
    """Create the Retrieval Team with supervisor.

    The Retrieval Team consists of four specialized agents coordinated
    by a supervisor that delegates tasks based on the execution plan.

    Args:
        llm: Language model to use for all agents
        custom_prompt: Optional custom prompt for supervisor
        enable_graph_search: Whether to enable graph search agent (from UI toggle)

    Returns:
        Compiled supervisor workflow
    """
    logger.info(_log_separator("-"))
    logger.info("[RetrievalTeam] ========== CREATING RETRIEVAL TEAM ==========")

    try:
        # Try to use langgraph-supervisor if available
        from langgraph_supervisor import create_supervisor

        logger.info("[RetrievalTeam] Using langgraph-supervisor for team coordination")

        # Create specialized agents
        logger.info("[RetrievalTeam] Creating SQL Agent...")
        sql_agent = create_sql_agent(llm)
        logger.info("[RetrievalTeam] ✓ SQL Agent created")

        logger.info("[RetrievalTeam] Creating Vector Agent...")
        vector_agent = create_vector_agent(llm)
        logger.info("[RetrievalTeam] ✓ Vector Agent created")

        logger.info("[RetrievalTeam] Creating Generic Doc Agent...")
        generic_doc_agent = create_generic_doc_agent(llm)
        logger.info("[RetrievalTeam] ✓ Generic Doc Agent created")

        agents = [sql_agent, vector_agent, generic_doc_agent]

        # Add graph agent if enabled (uses UI toggle, falls back to config)
        graph_enabled = enable_graph_search and AGENTIC_CONFIG.get("enable_graph_search", True)
        if graph_enabled:
            logger.info("[RetrievalTeam] Creating Graph Agent (graph search enabled from UI)...")
            graph_agent = create_graph_agent(llm)
            agents.append(graph_agent)
            logger.info("[RetrievalTeam] ✓ Graph Agent created")
        else:
            logger.info(f"[RetrievalTeam] Graph Agent skipped (UI toggle: {enable_graph_search}, config: {AGENTIC_CONFIG.get('enable_graph_search', True)})")

        prompt = custom_prompt or RETRIEVAL_SUPERVISOR_PROMPT

        # Create supervisor
        logger.info("[RetrievalTeam] Creating supervisor with %d agents...", len(agents))
        retrieval_supervisor = create_supervisor(
            agents=agents,
            model=llm,
            prompt=prompt,
            supervisor_name="retrieval_supervisor",
            output_mode="full_history"
        )

        logger.info("[RetrievalTeam] ✓ Retrieval Team created with %d agents using langgraph-supervisor", len(agents))
        logger.info(_log_separator("-"))

        return retrieval_supervisor

    except ImportError:
        logger.warning("[RetrievalTeam] langgraph-supervisor not available, using custom implementation")
        return _create_custom_retrieval_team(llm, custom_prompt, enable_graph_search)


def _create_custom_retrieval_team(
    llm: BaseChatModel,
    custom_prompt: Optional[str] = None,
    enable_graph_search: bool = True
):
    """Create a custom retrieval team without langgraph-supervisor.

    This is a fallback implementation that uses a simple StateGraph
    to coordinate the agents.

    IMPORTANT: This team receives the full AnalyticsAgentState from the
    main workflow, including the execution_plan set by the PlannerAgent.
    """
    logger.info(_log_separator("-"))
    logger.info("[RetrievalTeam] ========== CREATING CUSTOM RETRIEVAL TEAM ==========")

    from langgraph.graph import StateGraph, END
    from typing import TypedDict, Annotated, Sequence, Literal, Optional, List, Dict, Any
    from langchain_core.messages import BaseMessage, HumanMessage
    from langgraph.graph.message import add_messages
    import operator
    import json

    from agents.state.models import AgentOutput, ExecutionPlan, SchemaGroup, DocumentSource, ReviewDecision

    # Use the same state schema as the main workflow to ensure proper state passing
    class RetrievalState(TypedDict):
        """State schema that matches AnalyticsAgentState for proper state flow."""
        messages: Annotated[Sequence[BaseMessage], add_messages]
        user_query: str
        workspace_id: str
        chat_history: List[Dict[str, str]]
        available_documents: List[DocumentSource]
        schema_groups: List[SchemaGroup]
        execution_plan: Optional[ExecutionPlan]
        agent_outputs: Annotated[List[AgentOutput], operator.add]
        review_iteration: int
        max_iterations: int
        review_decision: Optional[ReviewDecision]
        final_response: Optional[str]
        data_sources: List[str]
        structured_data: Dict[str, Any]
        active_agent: Optional[str]
        error: Optional[str]
        # Internal retrieval team state
        current_task_index: int
        next_agent: Optional[str]

    # Create agents
    logger.info("[RetrievalTeam] Creating SQL Agent...")
    sql_agent = create_sql_agent(llm)
    logger.info("[RetrievalTeam] ✓ SQL Agent created")

    logger.info("[RetrievalTeam] Creating Vector Agent...")
    vector_agent = create_vector_agent(llm)
    logger.info("[RetrievalTeam] ✓ Vector Agent created")

    logger.info("[RetrievalTeam] Creating Generic Doc Agent...")
    generic_doc_agent = create_generic_doc_agent(llm)
    logger.info("[RetrievalTeam] ✓ Generic Doc Agent created")

    agents = {
        "sql_agent": sql_agent,
        "vector_agent": vector_agent,
        "generic_doc_agent": generic_doc_agent
    }

    # Add graph agent if enabled (uses UI toggle, falls back to config)
    graph_enabled = enable_graph_search and AGENTIC_CONFIG.get("enable_graph_search", True)
    if graph_enabled:
        logger.info("[RetrievalTeam] Creating Graph Agent (graph search enabled from UI)...")
        graph_agent = create_graph_agent(llm)
        agents["graph_agent"] = graph_agent
        logger.info("[RetrievalTeam] ✓ Graph Agent created")
    else:
        logger.info(f"[RetrievalTeam] Graph Agent skipped (UI toggle: {enable_graph_search}, config: {AGENTIC_CONFIG.get('enable_graph_search', True)})")

    def supervisor_node(state: RetrievalState):
        """Supervisor decides which agent to call next."""
        start_time = time.time()

        plan = state.get("execution_plan")
        outputs = state.get("agent_outputs", [])
        current_idx = state.get("current_task_index", 0)

        logger.info(_log_separator("-"))
        logger.info("[RetrievalSupervisor] ========== SUPERVISOR DECISION ==========")
        logger.info(f"[RetrievalSupervisor] Current task index: {current_idx}")
        logger.info(f"[RetrievalSupervisor] Completed outputs: {len(outputs)}")
        logger.info(f"[RetrievalSupervisor] Execution plan present: {plan is not None}")

        # Debug: Log state keys to understand what we're receiving
        state_keys = list(state.keys()) if isinstance(state, dict) else []
        logger.info(f"[RetrievalSupervisor] State keys: {state_keys}")

        if not plan:
            logger.warning("[RetrievalSupervisor] No execution plan found in state!")
            logger.info("[RetrievalSupervisor] → Returning with current outputs")
            logger.info(_log_separator("-"))
            return {"agent_outputs": outputs}

        # Get sub_tasks, handling both model and dict forms
        sub_tasks = plan.sub_tasks if hasattr(plan, 'sub_tasks') else plan.get('sub_tasks', [])

        if not sub_tasks:
            logger.warning("[RetrievalSupervisor] Execution plan has no sub-tasks!")
            logger.info("[RetrievalSupervisor] → Returning with current outputs")
            logger.info(_log_separator("-"))
            return {"agent_outputs": outputs}

        logger.info(f"[RetrievalSupervisor] Total sub-tasks: {len(sub_tasks)}")

        # Get completed task IDs
        completed_ids = {o.task_id for o in outputs if hasattr(o, 'task_id')}
        logger.info(f"[RetrievalSupervisor] Completed task IDs: {completed_ids}")

        # Find next pending task
        for i, task in enumerate(sub_tasks[current_idx:], start=current_idx):
            task_id = task.task_id if hasattr(task, 'task_id') else task.get('task_id', '')
            if task_id not in completed_ids:
                target_agent = task.target_agent if hasattr(task, 'target_agent') else task.get('target_agent', '')
                task_desc = task.description if hasattr(task, 'description') else task.get('description', '')

                elapsed = time.time() - start_time
                logger.info(f"[RetrievalSupervisor] Found pending task #{i}: {task_desc[:50]}...")
                logger.info(f"[RetrievalSupervisor] → Routing to: {target_agent}")
                logger.info(f"[RetrievalSupervisor] Decision time: {elapsed:.3f}s")
                logger.info(_log_separator("-"))

                return {
                    "current_task_index": i,
                    "next_agent": target_agent
                }

        # All tasks complete
        elapsed = time.time() - start_time
        logger.info("[RetrievalSupervisor] All tasks completed!")
        logger.info(f"[RetrievalSupervisor] Decision time: {elapsed:.3f}s")
        logger.info(_log_separator("-"))
        return {"next_agent": "complete"}

    def route_to_agent(state: RetrievalState) -> Literal["sql_agent", "vector_agent", "graph_agent", "generic_doc_agent", "complete"]:
        """Route to the appropriate agent."""
        next_agent = state.get("next_agent", "complete")
        logger.info(f"[RetrievalRouter] Routing to: {next_agent}")
        if next_agent in agents:
            return next_agent
        return "complete"

    def _get_current_task(state: RetrievalState) -> Optional[Dict[str, Any]]:
        """Get the current task from execution plan based on current_task_index."""
        plan = state.get("execution_plan")
        if not plan:
            return None

        sub_tasks = plan.sub_tasks if hasattr(plan, 'sub_tasks') else plan.get('sub_tasks', [])
        current_idx = state.get("current_task_index", 0)

        if current_idx < len(sub_tasks):
            task = sub_tasks[current_idx]
            # Convert to dict if it's a Pydantic model
            if hasattr(task, 'model_dump'):
                return task.model_dump()
            elif hasattr(task, 'dict'):
                return task.dict()
            return task
        return None

    def _get_schema_group_for_task(state: RetrievalState, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get the schema group associated with a task."""
        plan = state.get("execution_plan")
        if not plan:
            return None

        schema_groups = plan.schema_groups if hasattr(plan, 'schema_groups') else plan.get('schema_groups', [])
        schema_group_id = task.get("schema_group_id")

        for group in schema_groups:
            group_dict = group.model_dump() if hasattr(group, 'model_dump') else (group.dict() if hasattr(group, 'dict') else group)
            if group_dict.get("group_id") == schema_group_id:
                return group_dict

        # If no matching group, return first group with same schema_type
        task_schema = task.get("schema_type", "unknown")
        for group in schema_groups:
            group_dict = group.model_dump() if hasattr(group, 'model_dump') else (group.dict() if hasattr(group, 'dict') else group)
            if group_dict.get("schema_type") == task_schema:
                return group_dict

        return None

    def _build_task_message(state: RetrievalState, agent_name: str) -> str:
        """Build a focused task message for the agent with all needed context.

        This prevents the agent from seeing irrelevant tool calls from other agents
        (like Planner's identify_sub_questions) which can cause hallucinations.
        """
        task = _get_current_task(state)
        if not task:
            # Fallback to user query if no task found
            return state.get("user_query", "Process the documents")

        schema_group = _get_schema_group_for_task(state, task)

        # Build a structured task message with all context the agent needs
        parts = [
            f"Task ID: {task.get('task_id', 'unknown')}",
            f"Task: {task.get('description', 'Process documents')}",
            f"Document IDs: {json.dumps(task.get('document_ids', []))}",
            f"Schema Type: {task.get('schema_type', 'unknown')}",
        ]

        if task.get('aggregation_type'):
            parts.append(f"Aggregation: {task.get('aggregation_type')}")

        if task.get('target_fields'):
            parts.append(f"Target Fields: {json.dumps(task.get('target_fields', []))}")

        if schema_group:
            parts.append(f"Schema Group: {json.dumps(schema_group)}")

        return "\n".join(parts)

    def wrap_agent_with_output_extraction(agent, agent_name: str):
        """Wrap an agent to extract results and convert to AgentOutput.

        Handles two types of agents:
        1. SQL Agent (new): Returns agent_output directly in state (atomic execution)
        2. Other agents (ReAct): Return results via tool messages (report_*_result)

        IMPORTANT: We construct a fresh message context for each agent to prevent
        hallucinations from seeing other agents' tool calls in the message history.
        """
        import json
        from langchain_core.messages import ToolMessage, HumanMessage

        def wrapped_agent(state: RetrievalState):
            # Build a fresh state with only the task-specific message
            # This prevents the agent from seeing Planner's tool calls and hallucinating
            task_message = _build_task_message(state, agent_name)

            # Create a clean state copy with fresh messages
            agent_state = dict(state)
            agent_state["messages"] = [HumanMessage(content=task_message)]

            # For SQL agent, also pass task context directly in state
            if agent_name == "sql_agent":
                task = _get_current_task(state)
                if task:
                    agent_state["task_id"] = task.get("task_id", "unknown")
                    agent_state["task_description"] = task.get("description", "")
                    agent_state["document_ids"] = task.get("document_ids", [])
                    agent_state["schema_type"] = task.get("schema_type")
                    agent_state["aggregation_type"] = task.get("aggregation_type")
                    agent_state["target_fields"] = task.get("target_fields", [])
                    # Get schema group for this task
                    schema_group = _get_schema_group_for_task(state, task)
                    if schema_group:
                        agent_state["schema_group"] = schema_group
                agent_state["user_query"] = state.get("user_query", "")

            logger.info(f"[{agent_name}] Starting with fresh context: {task_message[:100]}...")

            # Run the agent
            # SQL agent is simpler and doesn't need high recursion limit
            # ReAct agents may need more steps
            config = {"recursion_limit": 5 if agent_name == "sql_agent" else 15}
            try:
                result = agent.invoke(agent_state, config=config)
            except Exception as e:
                error_msg = str(e)
                if "recursion" in error_msg.lower():
                    logger.error(f"[{agent_name}] Hit recursion limit - agent may be stuck in loop")
                    # Return an error state so we don't completely fail
                    result = {
                        "messages": state.get("messages", []),
                        "error": f"Agent {agent_name} exceeded recursion limit"
                    }
                else:
                    raise

            agent_outputs_to_add = []

            # Check for agent_output directly in result (SQL agent pattern)
            if "agent_output" in result and result["agent_output"] is not None:
                agent_output = result["agent_output"]
                if isinstance(agent_output, AgentOutput):
                    agent_outputs_to_add.append(agent_output)
                    # Debug: Log data details to trace data flow
                    data_info = "None"
                    if agent_output.data is not None:
                        if isinstance(agent_output.data, list):
                            data_info = f"list[{len(agent_output.data)}]"
                        elif isinstance(agent_output.data, dict):
                            data_info = f"dict[{len(agent_output.data)} keys]"
                        else:
                            data_info = f"{type(agent_output.data).__name__}"
                    logger.info(f"[{agent_name}] Got AgentOutput directly from state (task={agent_output.task_id}, status={agent_output.status}, row_count={agent_output.row_count}, data={data_info})")

            # Also check for tool messages (ReAct agents pattern)
            messages = result.get("messages", [])
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    # Check if this is a report_*_result tool call
                    tool_name = getattr(msg, 'name', '') or ''
                    if 'report' in tool_name.lower() and 'result' in tool_name.lower():
                        try:
                            content = msg.content
                            if isinstance(content, str):
                                data = json.loads(content)
                                # Extract output for BOTH success and failure cases
                                # This ensures task is marked as attempted even if it failed
                                if data.get("task_id"):
                                    # Use status from tool if available, otherwise map success boolean
                                    status_value = data.get("status")
                                    if status_value not in ("success", "partial", "failed"):
                                        status_value = "success" if data.get("success") else "failed"

                                    # Collect issues from both 'issues' array and 'error' field
                                    # Ensure all issues are strings (AgentOutput.issues expects List[str])
                                    raw_issues = data.get("issues", [])
                                    issues_list = []
                                    for issue in raw_issues:
                                        if isinstance(issue, str):
                                            issues_list.append(issue)
                                        elif isinstance(issue, dict):
                                            # Convert dict to string representation
                                            error_msg = issue.get("error", str(issue))
                                            issues_list.append(str(error_msg))
                                        else:
                                            issues_list.append(str(issue))

                                    # Add error field if present
                                    error_field = data.get("error")
                                    if error_field:
                                        error_str = str(error_field) if not isinstance(error_field, str) else error_field
                                        if error_str not in issues_list:
                                            issues_list.append(error_str)

                                    agent_output = AgentOutput(
                                        agent_name=data.get("agent_name", agent_name),
                                        task_id=data.get("task_id"),
                                        status=status_value,
                                        data=data.get("data"),
                                        documents_used=data.get("documents_used", []),
                                        confidence=data.get("confidence", 0.0),
                                        schema_type=data.get("schema_type", "unknown"),
                                        query_executed=data.get("sql_executed"),
                                        row_count=data.get("row_count"),
                                        issues=issues_list
                                    )
                                    agent_outputs_to_add.append(agent_output)
                                    logger.info(f"[{agent_name}] Extracted AgentOutput from tool message (task={data.get('task_id')}, status={status_value})")
                        except (json.JSONDecodeError, Exception) as e:
                            logger.warning(f"[{agent_name}] Failed to extract AgentOutput from tool message: {e}")

            # Add extracted outputs to result
            if agent_outputs_to_add:
                result["agent_outputs"] = agent_outputs_to_add

            return result

        return wrapped_agent

    # Build graph
    logger.info("[RetrievalTeam] Building custom StateGraph...")
    builder = StateGraph(RetrievalState)

    builder.add_node("supervisor", supervisor_node)
    for name, agent in agents.items():
        wrapped = wrap_agent_with_output_extraction(agent, name)
        builder.add_node(name, wrapped)
        logger.info(f"[RetrievalTeam]   Added node: {name} (with output extraction wrapper)")

    builder.set_entry_point("supervisor")
    logger.info("[RetrievalTeam]   Entry point: supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {name: name for name in agents} | {"complete": END}
    )
    logger.info("[RetrievalTeam]   Added conditional edges from supervisor")

    # All agents return to supervisor
    for name in agents:
        builder.add_edge(name, "supervisor")
        logger.info(f"[RetrievalTeam]   {name} -> supervisor")

    logger.info("[RetrievalTeam] ✓ Custom Retrieval Team created with %d agents", len(agents))
    logger.info(_log_separator("-"))

    return builder.compile()
