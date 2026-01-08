"""
Graph Agent implementation.

The Graph Agent handles entity relationship queries:
- Cypher query execution against Neo4j
- Entity and relationship traversal
- Result reporting
"""

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from agents.tools.graph_tools import (
    cypher_query,
    report_graph_result
)
from agents.prompts.graph_prompts import GRAPH_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def create_graph_agent(
    llm: BaseChatModel,
    custom_prompt: Optional[str] = None
):
    """Create the Graph Search Agent.

    The Graph Agent executes queries against the Neo4j knowledge graph
    to find entities and relationships.

    Args:
        llm: Language model to use for the agent
        custom_prompt: Optional custom system prompt

    Returns:
        Compiled LangGraph agent
    """
    prompt = custom_prompt or GRAPH_AGENT_SYSTEM_PROMPT

    tools = [
        cypher_query,
        report_graph_result
    ]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        name="graph_agent"
    )

    logger.info("Created Graph Agent with %d tools", len(tools))

    return agent
