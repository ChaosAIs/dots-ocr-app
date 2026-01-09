"""
Shared utilities and services used by both agent and non-agent flows.

This module provides centralized implementations to ensure consistent behavior
across different processing paths.
"""

from shared.document_filter import (
    DocumentFilter,
    topic_based_prefilter,
    calculate_relevance_score,
    apply_adaptive_threshold,
    extract_query_hints,
    ADAPTIVE_THRESHOLD_RATIO,
    ENTITY_MATCH_BOOST,
    ENTITY_NO_MATCH_PENALTY,
    DOC_TYPE_BOOST,
    TABULAR_BOOST,
)

__all__ = [
    "DocumentFilter",
    "topic_based_prefilter",
    "calculate_relevance_score",
    "apply_adaptive_threshold",
    "extract_query_hints",
    "ADAPTIVE_THRESHOLD_RATIO",
    "ENTITY_MATCH_BOOST",
    "ENTITY_NO_MATCH_PENALTY",
    "DOC_TYPE_BOOST",
    "TABULAR_BOOST",
]
