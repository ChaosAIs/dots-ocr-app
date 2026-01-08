"""
Generic Document Agent implementation.

The Generic Doc Agent handles mixed/unknown document types:
- Hybrid search combining vector and SQL
- Data extraction from semi-structured documents
- Fallback RAG search
"""

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from agents.tools.generic_tools import (
    hybrid_document_search,
    extract_and_query,
    fallback_rag_search,
    report_generic_result
)
from agents.prompts.generic_prompts import GENERIC_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def create_generic_doc_agent(
    llm: BaseChatModel,
    custom_prompt: Optional[str] = None
):
    """Create the Generic Document Agent.

    The Generic Doc Agent is a versatile agent for handling documents
    that don't fit standard categories. It tries multiple strategies
    and serves as a fallback for other agents.

    Args:
        llm: Language model to use for the agent
        custom_prompt: Optional custom system prompt

    Returns:
        Compiled LangGraph agent
    """
    prompt = custom_prompt or GENERIC_AGENT_SYSTEM_PROMPT

    tools = [
        hybrid_document_search,
        extract_and_query,
        fallback_rag_search,
        report_generic_result
    ]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        name="generic_doc_agent"
    )

    logger.info("Created Generic Doc Agent with %d tools", len(tools))

    return agent
