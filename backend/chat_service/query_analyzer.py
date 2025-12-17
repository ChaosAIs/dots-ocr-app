"""
Query Analyzer - LLM-based query analysis for greeting detection and complexity assessment.

This module uses LLM to:
1. Detect if a query is a greeting or substantive question
2. Determine the optimal max_steps for Graph-RAG iterative reasoning based on query complexity
"""
import logging
import json
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


# Prompt for combined greeting detection and complexity analysis
QUERY_ANALYSIS_PROMPT = """Analyze the following user query and provide two assessments:

1. **Greeting Detection**: Is this query just a greeting/casual conversation, or does it require substantive information?
2. **Query Complexity**: How many iterative reasoning steps would be needed to answer this query?

User Query: "{query}"

Respond with ONLY a JSON object in this exact format:
{{
  "is_greeting": true/false,
  "max_steps": 1-5,
  "reasoning": "brief explanation"
}}

Guidelines:
- **is_greeting**: 
  - true: Simple greetings like "hello", "hi", "how are you", casual conversation
  - false: Any question requiring information, explanation, or analysis

- **max_steps** (for non-greetings):
  - 1: Simple, direct questions with clear answers (e.g., "What is X?", "Define Y")
  - 2: Questions requiring 2-3 pieces of information (e.g., "How does X work?")
  - 3: Questions comparing concepts or requiring multiple aspects (e.g., "Compare X and Y")
  - 4: Complex questions requiring deep analysis (e.g., "Explain the relationship between X, Y, and Z")
  - 5: Very complex questions requiring comprehensive multi-step reasoning (e.g., "How do X, Y, and Z interact, and what are the implications?")

Examples:
- "Hello!" → {{"is_greeting": true, "max_steps": 1, "reasoning": "Simple greeting"}}
- "What is Graph-R1?" → {{"is_greeting": false, "max_steps": 1, "reasoning": "Direct definition question"}}
- "How does Graph-R1 work?" → {{"is_greeting": false, "max_steps": 2, "reasoning": "Requires explanation of mechanism"}}
- "Compare Graph-R1 and traditional RAG" → {{"is_greeting": false, "max_steps": 3, "reasoning": "Requires understanding both concepts and comparison"}}
- "Explain how Graph-R1 relates to knowledge graphs, vector databases, and LLMs" → {{"is_greeting": false, "max_steps": 4, "reasoning": "Multi-concept relationship analysis"}}

Your response (JSON only):"""


def analyze_query_with_llm(user_message: str, assistant_response: str = None) -> QueryAnalysis:
    """
    Analyze query using LLM to detect greetings and determine optimal max_steps.
    
    Args:
        user_message: The user's query
        assistant_response: Optional assistant response (for greeting detection context)
    
    Returns:
        QueryAnalysis with is_greeting and max_steps
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
        
        # Create prompt
        prompt = QUERY_ANALYSIS_PROMPT.format(query=user_message)
        
        # Get LLM response
        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip()
        
        # Parse JSON response
        analysis = _parse_llm_response(result, user_message, assistant_response)
        
        logger.info(f"Query Analysis - Query: '{user_message[:50]}...', "
                   f"Is Greeting: {analysis.is_greeting}, Max Steps: {analysis.max_steps}, "
                   f"Reasoning: {analysis.reasoning}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in LLM query analysis: {e}")
        # Fallback to heuristic
        return _fallback_analysis(user_message, assistant_response)


def _parse_llm_response(response: str, user_message: str, assistant_response: str = None) -> QueryAnalysis:
    """Parse LLM JSON response."""
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
        
        # Validate max_steps range
        max_steps = max(1, min(5, int(max_steps)))
        
        return QueryAnalysis(
            is_greeting=bool(is_greeting),
            max_steps=max_steps,
            reasoning=reasoning
        )
        
    except Exception as e:
        logger.warning(f"Failed to parse LLM response: {e}, response: {response[:100]}")
        return _fallback_analysis(user_message, assistant_response)


def _fallback_analysis(user_message: str, assistant_response: str = None) -> QueryAnalysis:
    """Fallback heuristic-based analysis if LLM fails."""
    # Simple heuristic for greeting detection
    greeting_patterns = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon']
    user_lower = user_message.lower().strip()
    
    is_greeting = False
    if len(user_lower.split()) <= 5:
        for pattern in greeting_patterns:
            if pattern in user_lower:
                is_greeting = True
                break
    
    # Simple heuristic for max_steps
    word_count = len(user_message.split())
    if word_count <= 5:
        max_steps = 1
    elif word_count <= 10:
        max_steps = 2
    elif word_count <= 20:
        max_steps = 3
    else:
        max_steps = 4
    
    return QueryAnalysis(
        is_greeting=is_greeting,
        max_steps=max_steps,
        reasoning="Fallback heuristic analysis"
    )

