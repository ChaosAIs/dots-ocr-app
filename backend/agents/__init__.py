"""
Agentic AI Flow for Multi-Query Analytics

This module implements a dynamic, self-orchestrating multi-agent system using
LangGraph 1.0+ and LangChain patterns for intelligent query processing.

Architecture:
- Planner Agent: Query decomposition and document routing
- Retrieval Team: SQL, Vector, Graph, and Generic Doc agents
- Reviewer Agent: Quality control and refinement
- Summary Agent: Response synthesis

Usage:
    from agents import create_analytics_workflow, execute_analytics_query

    workflow = create_analytics_workflow(llm)
    result = await execute_analytics_query(
        user_query="sum total inventory",
        workspace_id="ws_123",
        llm=llm
    )
"""

from agents.workflow import (
    create_analytics_workflow,
    execute_analytics_query,
    execute_analytics_query_async
)
from agents.state.schema import AnalyticsAgentState
from agents.state.models import (
    DocumentSource,
    SchemaGroup,
    SubTask,
    ExecutionPlan,
    AgentOutput,
    ReviewDecision
)
from agents.config import AGENTIC_CONFIG, AGENT_LLM_CONFIG

__all__ = [
    # Workflow
    "create_analytics_workflow",
    "execute_analytics_query",
    "execute_analytics_query_async",
    # State
    "AnalyticsAgentState",
    # Models
    "DocumentSource",
    "SchemaGroup",
    "SubTask",
    "ExecutionPlan",
    "AgentOutput",
    "ReviewDecision",
    # Config
    "AGENTIC_CONFIG",
    "AGENT_LLM_CONFIG",
]
