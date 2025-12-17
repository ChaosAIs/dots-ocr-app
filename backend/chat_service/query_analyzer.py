"""
Query Analyzer - LLM-based query analysis for greeting detection and complexity assessment.

This module uses LLM to:
1. Detect if a query is a greeting or substantive question
2. Determine the optimal max_steps for Graph-RAG iterative reasoning based on query complexity
   (capped by GRAPH_RAG_MAX_STEPS environment variable)
"""
import logging
import json
import os
import re
from typing import Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    is_greeting: bool
    max_steps: int
    reasoning: str


def _get_max_steps_limit() -> int:
    """Get the maximum allowed steps from environment variable."""
    return int(os.getenv("GRAPH_RAG_MAX_STEPS", "3"))


# Prompt for combined greeting detection and complexity analysis
# Note: The max_steps value will be capped by GRAPH_RAG_MAX_STEPS env variable
QUERY_ANALYSIS_PROMPT = """Analyze the following user query and provide two assessments:

1. **Greeting Detection**: Is this query just a greeting/casual conversation, or does it require substantive information?
2. **Query Complexity**: How many iterative reasoning steps would be needed to answer this query?

User Query: "{query}"

Respond with ONLY a JSON object in this exact format:
{{
  "is_greeting": true/false,
  "max_steps": 1-{max_limit},
  "reasoning": "brief explanation"
}}

Guidelines:
- **is_greeting**:
  - true: Simple greetings like "hello", "hi", "how are you", casual conversation
  - false: Any question requiring information, explanation, or analysis

- **max_steps** (for non-greetings, range 1-{max_limit}):
  - 1: Simple, direct questions with clear answers (e.g., "What is X?", "Define Y")
  - 2: Questions requiring 2-3 pieces of information (e.g., "How does X work?")
  - 3: Questions comparing concepts or requiring multiple aspects (e.g., "Compare X and Y")
  - 4+: Complex questions requiring deep analysis or multi-concept relationships

Examples:
- "Hello!" → {{"is_greeting": true, "max_steps": 1, "reasoning": "Simple greeting"}}
- "What is Graph-R1?" → {{"is_greeting": false, "max_steps": 1, "reasoning": "Direct definition question"}}
- "How does Graph-R1 work?" → {{"is_greeting": false, "max_steps": 2, "reasoning": "Requires explanation of mechanism"}}
- "Compare Graph-R1 and traditional RAG" → {{"is_greeting": false, "max_steps": 3, "reasoning": "Requires understanding both concepts and comparison"}}

Your response (JSON only):"""


def analyze_query_with_llm(user_message: str, assistant_response: str = None) -> QueryAnalysis:
    """
    Analyze query using LLM to detect greetings and determine optimal max_steps.

    The max_steps value is capped by GRAPH_RAG_MAX_STEPS environment variable.
    This ensures the LLM determines complexity within the configured maximum limit.

    Args:
        user_message: The user's query
        assistant_response: Optional assistant response (for greeting detection context)

    Returns:
        QueryAnalysis with is_greeting and max_steps (capped by env variable)
    """
    try:
        from rag_service.llm_service import get_llm_service
        from langchain_core.messages import HumanMessage

        llm_service = get_llm_service()
        llm = llm_service.get_query_model(
            temperature=0.0,  # Deterministic for classification
            num_ctx=1024,
            num_predict=100
        )

        # Get max steps limit from environment
        max_limit = _get_max_steps_limit()

        # Create prompt with dynamic max_limit
        prompt = QUERY_ANALYSIS_PROMPT.format(query=user_message, max_limit=max_limit)

        # Get LLM response
        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip()

        # Parse JSON response (will cap at max_limit)
        analysis = _parse_llm_response(result, user_message, assistant_response, max_limit)

        logger.info(f"Query Analysis - Query: '{user_message[:50]}...', "
                   f"Is Greeting: {analysis.is_greeting}, Max Steps: {analysis.max_steps}/{max_limit}, "
                   f"Reasoning: {analysis.reasoning}")

        return analysis

    except Exception as e:
        logger.error(f"Error in LLM query analysis: {e}")
        # Fallback to heuristic
        return _fallback_analysis(user_message, assistant_response)


def _parse_llm_response(response: str, user_message: str, assistant_response: str = None, max_limit: int = None) -> QueryAnalysis:
    """
    Parse LLM JSON response and cap max_steps at the configured limit.

    Args:
        response: LLM response string
        user_message: Original user message
        assistant_response: Optional assistant response
        max_limit: Maximum allowed steps from environment variable

    Returns:
        QueryAnalysis with max_steps capped at max_limit
    """
    try:
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(response)

        is_greeting = data.get("is_greeting", False)
        max_steps = data.get("max_steps", 1)
        reasoning = data.get("reasoning", "")

        # Get max limit from parameter or environment
        if max_limit is None:
            max_limit = _get_max_steps_limit()

        # Validate max_steps range and cap at configured maximum
        max_steps = max(1, min(max_limit, int(max_steps)))

        return QueryAnalysis(
            is_greeting=bool(is_greeting),
            max_steps=max_steps,
            reasoning=reasoning
        )

    except Exception as e:
        logger.warning(f"Failed to parse LLM response: {e}, response: {response[:100]}")
        return _fallback_analysis(user_message, assistant_response)


def _fallback_analysis(user_message: str, assistant_response: str = None) -> QueryAnalysis:
    """
    Fallback heuristic-based analysis if LLM fails.

    Also caps max_steps at the configured limit.
    """
    # Simple heuristic for greeting detection
    greeting_patterns = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon']
    user_lower = user_message.lower().strip()

    is_greeting = False
    if len(user_lower.split()) <= 5:
        for pattern in greeting_patterns:
            if pattern in user_lower:
                is_greeting = True
                break

    # Get max limit from environment
    max_limit = _get_max_steps_limit()

    # Simple heuristic for max_steps (capped at max_limit)
    word_count = len(user_message.split())
    if word_count <= 5:
        max_steps = 1
    elif word_count <= 10:
        max_steps = min(2, max_limit)
    elif word_count <= 20:
        max_steps = min(3, max_limit)
    else:
        max_steps = min(4, max_limit)

    return QueryAnalysis(
        is_greeting=is_greeting,
        max_steps=max_steps,
        reasoning="Fallback heuristic analysis"
    )

