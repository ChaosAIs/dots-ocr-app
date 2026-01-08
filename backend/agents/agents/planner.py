"""
Planner Agent (Orchestrator) implementation.

The Planner Agent is responsible for:
- Query analysis and decomposition
- Document source routing
- Schema-aware task classification
- Execution plan generation
"""

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from agents.tools.routing_tools import (
    get_relevant_documents,
    analyze_document_schemas,
    group_documents_by_schema
)
from agents.tools.planning_tools import (
    identify_sub_questions,
    classify_task_agent,
    create_execution_plan
)
from agents.prompts.planner_prompts import PLANNER_SYSTEM_PROMPT
from agents.state.schema import PlannerAgentState

logger = logging.getLogger(__name__)


def create_planner_agent(
    llm: BaseChatModel,
    custom_prompt: Optional[str] = None
):
    """Create the Planner/Orchestrator agent.

    The Planner Agent analyzes user queries, routes to relevant documents,
    and creates execution plans for the retrieval team.

    Uses PlannerAgentState as state_schema to enable tools to access
    custom state fields like available_documents via InjectedState.

    Args:
        llm: Language model to use for the agent
        custom_prompt: Optional custom system prompt

    Returns:
        Compiled LangGraph agent
    """
    prompt = custom_prompt or PLANNER_SYSTEM_PROMPT

    tools = [
        get_relevant_documents,
        analyze_document_schemas,
        group_documents_by_schema,
        identify_sub_questions,
        classify_task_agent,
        create_execution_plan
    ]

    # Use PlannerAgentState as state_schema so InjectedState can access
    # custom fields like available_documents, workspace_id, user_query
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        name="planner_agent",
        state_schema=PlannerAgentState
    )

    logger.info("Created Planner Agent with %d tools and PlannerAgentState schema", len(tools))

    return agent
