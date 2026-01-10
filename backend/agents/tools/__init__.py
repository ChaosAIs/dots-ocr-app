"""
Tools for the Agentic AI Flow system.

This module contains all tool definitions used by the agents:
- Routing tools: Document source routing
- Planning tools: Query decomposition and plan creation
- SQL Executor tool: Atomic SQL execution (preferred)
- SQL tools: Legacy SQL tools (deprecated, use sql_executor_tool instead)
- Vector tools: Semantic search operations
- Graph tools: Knowledge graph queries
- Generic tools: Hybrid/fallback operations
- Review tools: Quality validation
- Summary tools: Result aggregation and formatting
"""

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
# New atomic SQL executor (preferred)
from agents.tools.sql_executor_tool import (
    execute_sql_subtask,
    execute_sql_subtask_core,
    create_agent_output_from_result
)
# Legacy SQL tools (deprecated)
from agents.tools.sql_tools import (
    generate_schema_aware_sql,
    execute_sql_with_retry,
    report_sql_result
)
from agents.tools.vector_tools import (
    semantic_search,
    report_vector_result
)
from agents.tools.graph_tools import (
    cypher_query,
    report_graph_result
)
from agents.tools.generic_tools import (
    hybrid_document_search,
    extract_and_query,
    fallback_rag_search,
    report_generic_result
)
from agents.tools.review_tools import (
    validate_completeness,
    check_data_quality,
    approve_and_continue,
    request_refinement
)
from agents.tools.summary_tools import (
    aggregate_results,
    format_response
)

__all__ = [
    # Routing
    "get_relevant_documents",
    "analyze_document_schemas",
    "group_documents_by_schema",
    # Planning
    "identify_sub_questions",
    "classify_task_agent",
    "create_execution_plan",
    # SQL Executor (preferred)
    "execute_sql_subtask",
    "execute_sql_subtask_core",
    "create_agent_output_from_result",
    # SQL (legacy, deprecated)
    "generate_schema_aware_sql",
    "execute_sql_with_retry",
    "report_sql_result",
    # Vector
    "semantic_search",
    "report_vector_result",
    # Graph
    "cypher_query",
    "report_graph_result",
    # Generic
    "hybrid_document_search",
    "extract_and_query",
    "fallback_rag_search",
    "report_generic_result",
    # Review
    "validate_completeness",
    "check_data_quality",
    "approve_and_continue",
    "request_refinement",
    # Summary
    "aggregate_results",
    "format_response",
]
