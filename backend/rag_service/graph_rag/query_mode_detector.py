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

import os
import json
import logging
import re
from typing import Tuple, Optional

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
            llm_client: Optional LLM client (uses Ollama by default)
        """
        self.llm_client = llm_client
        self._ollama_client = None

    def _get_llm_client(self):
        """Get or create the LLM client."""
        if self.llm_client is not None:
            return self.llm_client

        if self._ollama_client is None:
            from langchain_ollama import ChatOllama

            host = os.getenv("OLLAMA_HOST", "localhost")
            port = os.getenv("OLLAMA_PORT", "11434")
            model = os.getenv("OLLAMA_MODEL", "qwen2.5:latest")

            self._ollama_client = ChatOllama(
                base_url=f"http://{host}:{port}",
                model=model,
                temperature=0.1,
            )

        return self._ollama_client

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
        """
        query_lower = query.lower()

        # Patterns for different modes
        local_patterns = [
            r'\bwho is\b', r'\bwhat is\b', r'\btell me about\b',
            r'\bdescribe\b', r'\bexplain\b.*\b(concept|term|entity)\b',
            r'\bdefine\b', r'\bwhat does\b.*\bmean\b',
        ]

        global_patterns = [
            r'\bhow does\b.*\brelate\b', r'\bconnection between\b',
            r'\brelationship between\b', r'\bcompare\b.*\band\b',
            r'\bdifference between\b', r'\bsimilarity between\b',
        ]

        naive_patterns = [
            r'\blist\b', r'\bwhat date\b', r'\bwhen\b',
            r'\bhow many\b', r'\bcount\b',
        ]

        # Check patterns
        for pattern in local_patterns:
            if re.search(pattern, query_lower):
                return QueryMode.LOCAL, query

        for pattern in global_patterns:
            if re.search(pattern, query_lower):
                return QueryMode.GLOBAL, query

        for pattern in naive_patterns:
            if re.search(pattern, query_lower):
                return QueryMode.NAIVE, query

        # Default to HYBRID
        return QueryMode.HYBRID, query

