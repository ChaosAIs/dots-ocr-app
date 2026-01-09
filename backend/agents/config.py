"""
Configuration for the Agentic AI Flow system.

This module contains all configuration constants, feature flags,
and LLM settings for each agent in the system.
"""

from typing import Dict, Any, Literal
import os

# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

AGENTIC_CONFIG: Dict[str, Any] = {
    # Workflow settings
    "max_review_iterations": 3,
    "default_confidence_threshold": 0.7,
    "low_confidence_threshold": 0.5,
    "enable_streaming": True,

    # Qwen3 reasoning mode: "thinking" enables <think> blocks, "instruction" disables them
    # Reuses the same env var as non-agent flow (RAG_VLLM_REASONING_MODE)
    "reasoning_mode": os.getenv("RAG_VLLM_REASONING_MODE", "instruction"),

    # Agent timeouts (seconds) - per LLM call, not total agent time
    # Local models (Qwen3-4B/8B) may need 60-90s per call
    "planner_timeout": 120,      # Planner makes ~4 LLM calls
    "retrieval_timeout": 120,    # Retrieval agents need time for SQL/search
    "reviewer_timeout": 90,      # Reviewer makes 1-2 LLM calls
    "summary_timeout": 90,       # Summary makes 1-2 LLM calls
    "generic_agent_timeout": 90,

    # Feature flags
    "enable_graph_search": os.getenv("GRAPH_RAG_QUERY_ENABLED", "true").lower() == "true",
    "enable_vector_search": True,
    "enable_generic_agent": True,
    "fallback_to_legacy": True,  # Fall back to existing pipeline on failure

    # Checkpointing
    "checkpoint_backend": os.getenv("AGENT_CHECKPOINT_BACKEND", "memory"),  # "memory", "postgres", "redis"
    "checkpoint_ttl_hours": 24,

    # Parallel execution
    "max_parallel_tasks": 5,
    "enable_parallel_retrieval": True,

    # Retry settings
    "max_sql_retries": 3,
    "max_vector_retries": 2,
    "max_graph_retries": 2,

    # Result limits
    "max_documents_per_query": 50,
    "max_results_per_agent": 1000,
    "max_tokens_per_response": 4000,
}

# =============================================================================
# LLM CONFIGURATION PER AGENT
# =============================================================================

AGENT_LLM_CONFIG: Dict[str, Dict[str, Any]] = {
    "planner": {
        "model": os.getenv("PLANNER_MODEL", "gpt-4o"),
        "temperature": 0.1,
        "max_tokens": 2000,
        "description": "Query decomposition and document routing"
    },
    "sql_agent": {
        "model": os.getenv("SQL_AGENT_MODEL", "gpt-4o"),
        "temperature": 0.0,  # Deterministic for SQL generation
        "max_tokens": 1500,
        "description": "Structured data SQL queries"
    },
    "vector_agent": {
        "model": os.getenv("VECTOR_AGENT_MODEL", "gpt-4o-mini"),
        "temperature": 0.0,
        "max_tokens": 1000,
        "description": "Semantic search operations"
    },
    "graph_agent": {
        "model": os.getenv("GRAPH_AGENT_MODEL", "gpt-4o-mini"),
        "temperature": 0.0,
        "max_tokens": 1000,
        "description": "Entity relationship queries"
    },
    "generic_doc_agent": {
        "model": os.getenv("GENERIC_AGENT_MODEL", "gpt-4o"),
        "temperature": 0.1,
        "max_tokens": 1500,
        "description": "Hybrid/fallback document processing"
    },
    "retrieval_supervisor": {
        "model": os.getenv("SUPERVISOR_MODEL", "gpt-4o"),
        "temperature": 0.1,
        "max_tokens": 1000,
        "description": "Retrieval team coordination"
    },
    "reviewer": {
        "model": os.getenv("REVIEWER_MODEL", "gpt-4o"),
        "temperature": 0.1,
        "max_tokens": 1000,
        "description": "Quality control and validation"
    },
    "summary": {
        "model": os.getenv("SUMMARY_MODEL", "gpt-4o"),
        "temperature": 0.3,  # Slightly creative for natural responses
        "max_tokens": 2000,
        "description": "Response synthesis and formatting"
    },
}

# =============================================================================
# SCHEMA TYPE CLASSIFICATIONS
# =============================================================================

SCHEMA_CLASSIFICATIONS: Dict[str, Dict[str, Any]] = {
    "structured": {
        "types": ["tabular", "spreadsheet", "csv", "xlsx", "inventory_report", "financial_report"],
        "default_agent": "sql_agent",
        "description": "Highly structured data suitable for SQL queries"
    },
    "semi_structured": {
        "types": ["invoice", "receipt", "purchase_order", "bank_statement", "expense_report"],
        "default_agent": "sql_agent",  # If extracted fields available
        "fallback_agent": "vector_agent",
        "min_fields_for_sql": 3,  # Minimum extracted fields to use SQL
        "description": "Documents with extractable structured data"
    },
    "unstructured": {
        "types": ["document", "report", "memo", "email", "contract", "letter"],
        "default_agent": "vector_agent",
        "description": "Free-form text documents"
    },
    "mixed": {
        "types": ["mixed", "unknown", "custom"],
        "default_agent": "generic_doc_agent",
        "description": "Documents with mixed or unknown content types"
    }
}

# =============================================================================
# AGGREGATION MAPPINGS
# =============================================================================

AGGREGATION_KEYWORDS: Dict[str, str] = {
    # Sum variations
    "sum": "sum",
    "total": "sum",
    "add": "sum",
    "combine": "sum",
    "aggregate": "sum",

    # Count variations
    "count": "count",
    "number": "count",
    "how many": "count",
    "quantity": "count",

    # Average variations
    "average": "avg",
    "avg": "avg",
    "mean": "avg",

    # Min/Max variations
    "minimum": "min",
    "min": "min",
    "lowest": "min",
    "smallest": "min",
    "maximum": "max",
    "max": "max",
    "highest": "max",
    "largest": "max",

    # Group by
    "by": "group_by",
    "per": "group_by",
    "each": "group_by",
    "grouped": "group_by",
}

# =============================================================================
# AGENT ROUTING RULES
# =============================================================================

AGENT_ROUTING_RULES: Dict[str, Dict[str, Any]] = {
    "sql_agent": {
        "priority": 1,
        "supported_operations": ["sum", "count", "avg", "min", "max", "group_by", "filter", "join"],
        "requires_schema": True,
        "min_confidence": 0.7
    },
    "vector_agent": {
        "priority": 2,
        "supported_operations": ["search", "similarity", "extract", "summarize"],
        "requires_schema": False,
        "min_confidence": 0.6
    },
    "graph_agent": {
        "priority": 3,
        "supported_operations": ["relationships", "connections", "paths", "entities"],
        "requires_schema": False,
        "min_confidence": 0.6,
        "requires_graph_enabled": True
    },
    "generic_doc_agent": {
        "priority": 4,  # Lowest priority - used as fallback
        "supported_operations": ["hybrid", "extract", "fallback"],
        "requires_schema": False,
        "min_confidence": 0.5
    }
}

# =============================================================================
# ERROR MESSAGES
# =============================================================================

ERROR_MESSAGES: Dict[str, str] = {
    "no_documents": "No relevant documents found for the query.",
    "no_schema": "Could not determine document schema for processing.",
    "sql_generation_failed": "Failed to generate SQL query after {retries} attempts.",
    "vector_search_failed": "Semantic search failed to find relevant results.",
    "graph_search_failed": "Graph search failed - knowledge graph may not be available.",
    "all_agents_failed": "All retrieval agents failed to process the query.",
    "max_iterations_reached": "Maximum review iterations reached. Returning best available results.",
    "timeout": "Agent {agent_name} timed out after {timeout} seconds.",
    "low_confidence": "Results have low confidence ({confidence}). Consider refining your query.",
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG: Dict[str, Any] = {
    "log_agent_decisions": True,
    "log_tool_calls": True,
    "log_state_transitions": True,
    "log_level": os.getenv("AGENT_LOG_LEVEL", "INFO"),
    "include_timestamps": True,
    "include_token_usage": True,
}
