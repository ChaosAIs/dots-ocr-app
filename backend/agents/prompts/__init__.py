"""
System prompts for all agents in the Agentic AI Flow system.
"""

from agents.prompts.planner_prompts import PLANNER_SYSTEM_PROMPT
from agents.prompts.sql_prompts import SQL_AGENT_SYSTEM_PROMPT
from agents.prompts.vector_prompts import VECTOR_AGENT_SYSTEM_PROMPT
from agents.prompts.graph_prompts import GRAPH_AGENT_SYSTEM_PROMPT
from agents.prompts.generic_prompts import GENERIC_AGENT_SYSTEM_PROMPT
from agents.prompts.reviewer_prompts import REVIEWER_SYSTEM_PROMPT
from agents.prompts.summary_prompts import SUMMARY_SYSTEM_PROMPT
from agents.prompts.retrieval_prompts import RETRIEVAL_SUPERVISOR_PROMPT

__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "SQL_AGENT_SYSTEM_PROMPT",
    "VECTOR_AGENT_SYSTEM_PROMPT",
    "GRAPH_AGENT_SYSTEM_PROMPT",
    "GENERIC_AGENT_SYSTEM_PROMPT",
    "REVIEWER_SYSTEM_PROMPT",
    "SUMMARY_SYSTEM_PROMPT",
    "RETRIEVAL_SUPERVISOR_PROMPT",
]
