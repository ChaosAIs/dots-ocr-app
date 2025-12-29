"""
Query Mode Detector for GraphRAG.

This module implements LLM-based query mode detection to determine
the best retrieval strategy for a given query.

Following Graph-R1 paper design:
- LOCAL: Entity-focused queries (e.g., "Who is John Smith?")
- GLOBAL: Relationship-focused queries (e.g., "How does X relate to Y?")
- HYBRID: Combined entity and relationship queries (default)
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
        logger.info("-" * 60)
        logger.info("[Query Mode Detector] MODE DETECTION")
        logger.info("-" * 60)
        logger.info(f"[Query Mode Detector] Query: {query[:80]}{'...' if len(query) > 80 else ''}")

        try:
            logger.info("[Query Mode Detector] Calling LLM for mode detection...")
            llm = self._get_llm_client()
            prompt = QUERY_MODE_DETECTION_PROMPT.format(query=query)

            response = await llm.ainvoke([{"role": "user", "content": prompt}])
            result = response.content if hasattr(response, 'content') else str(response)
            result = clean_llm_response(result)

            # Parse JSON response
            mode, enhanced_query = self._parse_response(result, query)

            logger.info("[Query Mode Detector] DETECTION RESULT:")
            logger.info(f"[Query Mode Detector]   - Mode: {mode.value}")
            if enhanced_query != query:
                logger.info(f"[Query Mode Detector]   - Enhanced query: {enhanced_query[:60]}...")
            return mode, enhanced_query

        except Exception as e:
            logger.warning(f"[Query Mode Detector] Detection failed: {e}, falling back to HYBRID")
            return QueryMode.HYBRID, query

    def _parse_response(self, response: str, original_query: str) -> Tuple[QueryMode, str]:
        """Parse the LLM response to extract mode and enhanced query (Graph-R1 modes only)."""
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

            # Map string to QueryMode enum (Graph-R1 modes only)
            mode_map = {
                "LOCAL": QueryMode.LOCAL,
                "GLOBAL": QueryMode.GLOBAL,
                "HYBRID": QueryMode.HYBRID,
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
            else:
                # Default to HYBRID for any unrecognized mode
                return QueryMode.HYBRID, original_query

    def detect_mode_heuristic(self, query: str) -> Tuple[QueryMode, str]:
        """
        Heuristic-based query mode detection (fast fallback).

        Uses keyword patterns to determine query mode without LLM.
        Following Graph-R1 paper design with 3 modes only.

        Priority order:
        1. GLOBAL - Cross-entity relationships ("how does X relate to Y")
        2. LOCAL - Single entity focus ("who is X", "what is X")
        3. HYBRID - Default for complex queries (mechanism, explanation, comparison)
        """
        logger.info("-" * 60)
        logger.info("[Query Mode Detector] HEURISTIC MODE DETECTION")
        logger.info("-" * 60)
        logger.info(f"[Query Mode Detector] Query: {query[:80]}{'...' if len(query) > 80 else ''}")

        query_lower = query.lower()

        # GLOBAL patterns - cross-entity relationships (most specific, check first)
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

        # LOCAL patterns - single entity definitions (specific entity queries)
        local_patterns = [
            r'\bwho is\b',
            r'\bwhat is\b(?!.*\b(components|parts|steps|difference|connection|relationship)\b)',
            r'\btell me about\b(?!.*\b(how|why|relationship)\b)',
            r'\bdefine\b',
            r'\bwhat does\b.*\bmean\b',
        ]

        # Check GLOBAL first (most specific)
        for pattern in global_patterns:
            if re.search(pattern, query_lower):
                logger.info("[Query Mode Detector] DETECTION RESULT:")
                logger.info(f"[Query Mode Detector]   - Mode: GLOBAL (matched pattern: {pattern})")
                return QueryMode.GLOBAL, query

        # Check LOCAL (single entity focus)
        for pattern in local_patterns:
            if re.search(pattern, query_lower):
                logger.info("[Query Mode Detector] DETECTION RESULT:")
                logger.info(f"[Query Mode Detector]   - Mode: LOCAL (matched pattern: {pattern})")
                return QueryMode.LOCAL, query

        # Default to HYBRID for all other queries
        logger.info("[Query Mode Detector] DETECTION RESULT:")
        logger.info("[Query Mode Detector]   - Mode: HYBRID (default)")
        return QueryMode.HYBRID, query

