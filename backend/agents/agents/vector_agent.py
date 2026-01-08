"""
Vector Agent implementation.

The Vector Agent handles semantic search operations:
- Similarity search via Qdrant
- Document content retrieval
- Result reporting
"""

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from agents.tools.vector_tools import (
    semantic_search,
    report_vector_result
)
from agents.prompts.vector_prompts import VECTOR_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def create_vector_agent(
    llm: BaseChatModel,
    custom_prompt: Optional[str] = None
):
    """Create the Vector Search Agent.

    The Vector Agent performs semantic similarity searches using
    vector embeddings stored in Qdrant.

    Args:
        llm: Language model to use for the agent
        custom_prompt: Optional custom system prompt

    Returns:
        Compiled LangGraph agent
    """
    prompt = custom_prompt or VECTOR_AGENT_SYSTEM_PROMPT

    tools = [
        semantic_search,
        report_vector_result
    ]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        name="vector_agent"
    )

    logger.info("Created Vector Agent with %d tools", len(tools))

    return agent
