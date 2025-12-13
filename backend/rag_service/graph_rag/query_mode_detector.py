"""
Query Mode Detector for GraphRAG.

This module implements LLM-based query mode detection to determine
the best retrieval strategy for a given query.

Query Modes:
- LOCAL: Entity-focused queries (e.g., "Who is John Smith?")
- GLOBAL: Relationship-focused queries (e.g., "How does X relate to Y?")
- HYBRID: Both entity and relationship queries
- NAIVE: Simple vector search (fallback)
"""

import json
import logging
import re
from typing import Tuple

from .base import QueryMode
from .prompts import QUERY_MODE_DETECTION_PROMPT
from .utils import clean_llm_response

logger = logging.getLogger(__name__)


class QueryModeDetector:
    """
    LLM-based query mode detection.

    Uses the LLM to analyze a query and determine:
    1. The best retrieval mode (LOCAL, GLOBAL, HYBRID, NAIVE)
    2. An enhanced version of the query for better retrieval
    """

    def __init__(self, llm_client=None):
        """
        Initialize the query mode detector.

        Args:
            llm_client: Optional LLM client (uses configured RAG LLM backend)
        """
        self.llm_client = llm_client
        self._cached_llm_client = None

    def _get_llm_client(self):
        """Get or create the LLM client using the configured RAG LLM backend."""
        if self.llm_client is not None:
            return self.llm_client

        if self._cached_llm_client is None:
            # Use the same LLM service as the RAG agent (vLLM or Ollama based on config)
            from ..llm_service import get_llm_service

            llm_service = get_llm_service()
            self._cached_llm_client = llm_service.get_query_model(
                temperature=0.1,
                num_ctx=2048,
                num_predict=256,
            )
            logger.info(f"QueryModeDetector using LLM service: {type(llm_service).__name__}")

        return self._cached_llm_client

    async def detect_mode(self, query: str) -> Tuple[QueryMode, str]:
        """
        Detect query mode and enhance query.

        Args:
            query: User query string

        Returns:
            Tuple of (QueryMode, enhanced_query)
        """
        try:
            llm = self._get_llm_client()
            prompt = QUERY_MODE_DETECTION_PROMPT.format(query=query)

            response = await llm.ainvoke([{"role": "user", "content": prompt}])
            result = response.content if hasattr(response, 'content') else str(response)
            result = clean_llm_response(result)

            # Parse JSON response
            mode, enhanced_query = self._parse_response(result, query)

            logger.info(f"Query mode detected: {mode.value} for query: {query[:50]}...")
            return mode, enhanced_query

        except Exception as e:
            logger.warning(f"Query mode detection failed: {e}, falling back to HYBRID")
            return QueryMode.HYBRID, query

    def _parse_response(self, response: str, original_query: str) -> Tuple[QueryMode, str]:
        """Parse the LLM response to extract mode and enhanced query."""
        try:
            # Try to parse as JSON
            # Handle potential markdown code blocks
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            mode_str = data.get("mode", "HYBRID").upper()
            enhanced_query = data.get("enhanced_query", original_query)

            # Map string to QueryMode enum
            mode_map = {
                "LOCAL": QueryMode.LOCAL,
                "GLOBAL": QueryMode.GLOBAL,
                "HYBRID": QueryMode.HYBRID,
                "NAIVE": QueryMode.NAIVE,
            }
            mode = mode_map.get(mode_str, QueryMode.HYBRID)

            return mode, enhanced_query

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response: {response[:100]}")
            # Try to extract mode from text
            response_upper = response.upper()
            if "LOCAL" in response_upper:
                return QueryMode.LOCAL, original_query
            elif "GLOBAL" in response_upper:
                return QueryMode.GLOBAL, original_query
            elif "NAIVE" in response_upper:
                return QueryMode.NAIVE, original_query
            else:
                return QueryMode.HYBRID, original_query

    def detect_mode_heuristic(self, query: str) -> Tuple[QueryMode, str]:
        """
        Heuristic-based query mode detection (fast fallback).

        Uses keyword patterns to determine query mode without LLM.

        Priority order (check HYBRID first since "how does X work" is common):
        1. HYBRID - "how does X work/function" (mechanism/process queries)
        2. GLOBAL - "how does X relate to Y" (cross-entity relationships)
        3. NAIVE - simple facts (when, how many, list)
        4. LOCAL - "what is X", "who is X" (entity definitions)
        5. Default - HYBRID (safest for complex queries)
        """
        query_lower = query.lower()

        # HYBRID patterns - check FIRST (how does X work/function)
        # These should NOT be confused with LOCAL "what is X"
        hybrid_patterns = [
            r'\bhow does\b.*\b(work|function|operate)\b',
            r'\bhow is\b.*\b(implemented|built|designed|structured)\b',
            r'\bexplain how\b',
            r'\bwhat are the (components|parts|steps|stages)\b',
            r'\bhow (to|can)\b',
            r'\bexplain the\b.*\b(process|system|mechanism|architecture)\b',
            r'\bwhat does\b.*\bdo\b',
        ]

        # GLOBAL patterns - relationships BETWEEN entities
        global_patterns = [
            r'\bhow does\b.*\brelate\b',
            r'\bconnection between\b',
            r'\brelationship between\b',
            r'\bcompare\b.*\b(and|with|to)\b',
            r'\bdifference between\b',
            r'\bsimilarity between\b',
            r'\bhow (do|does)\b.*\binteract\b',
            r'\bwhat links\b.*\bto\b',
        ]

        # NAIVE patterns - simple facts
        naive_patterns = [
            r'\blist\b',
            r'\bwhat date\b',
            r'\bwhen (was|is|did)\b',
            r'\bhow many\b',
            r'\bcount\b',
            r'\bwhat time\b',
        ]

        # LOCAL patterns - entity definitions ONLY
        local_patterns = [
            r'\bwho is\b',
            r'\bwhat is\b(?!.*\b(components|parts|steps|difference|connection)\b)',
            r'\btell me about\b',
            r'\bdefine\b',
            r'\bwhat does\b.*\bmean\b',
        ]

        # Check HYBRID first (most common complex queries)
        for pattern in hybrid_patterns:
            if re.search(pattern, query_lower):
                return QueryMode.HYBRID, query

        # Check GLOBAL (cross-entity relationships)
        for pattern in global_patterns:
            if re.search(pattern, query_lower):
                return QueryMode.GLOBAL, query

        # Check NAIVE (simple facts)
        for pattern in naive_patterns:
            if re.search(pattern, query_lower):
                return QueryMode.NAIVE, query

        # Check LOCAL (entity definitions)
        for pattern in local_patterns:
            if re.search(pattern, query_lower):
                return QueryMode.LOCAL, query

        # Default to HYBRID (safest for complex/ambiguous queries)
        return QueryMode.HYBRID, query

