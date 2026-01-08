"""
SQL Agent implementation.

The SQL Agent handles structured/tabular data queries with:
- Schema-aware SQL generation
- Execution with error correction
- Result reporting
"""

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from agents.tools.sql_tools import (
    generate_schema_aware_sql,
    execute_sql_with_retry,
    report_sql_result
)
from agents.prompts.sql_prompts import SQL_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def create_sql_agent(
    llm: BaseChatModel,
    custom_prompt: Optional[str] = None
):
    """Create the SQL Retrieval Agent.

    The SQL Agent generates and executes SQL queries for structured data,
    with automatic error correction and retry logic.

    Args:
        llm: Language model to use for the agent
        custom_prompt: Optional custom system prompt

    Returns:
        Compiled LangGraph agent
    """
    prompt = custom_prompt or SQL_AGENT_SYSTEM_PROMPT

    tools = [
        generate_schema_aware_sql,
        execute_sql_with_retry,
        report_sql_result
    ]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        name="sql_agent"
    )

    logger.info("Created SQL Agent with %d tools", len(tools))

    return agent
