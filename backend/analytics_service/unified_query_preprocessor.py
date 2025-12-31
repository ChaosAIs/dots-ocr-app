"""
Unified Query Preprocessor - Single LLM Call for All Pre-Processing

This module consolidates THREE separate LLM calls into ONE:
1. Context Analysis (pronouns, entities, topics)
2. Cache Analysis (dissatisfaction, self-contained check, cacheability)
3. Intent Classification (routing decision)

Benefits:
- 66% reduction in pre-processing latency (~1500ms -> ~500ms)
- 66% reduction in token cost
- Consistent entity/topic extraction across all components
- Single point of analysis for easier debugging
"""

import os
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime, timedelta
import calendar

logger = logging.getLogger(__name__)

# Environment variable to enable/disable unified preprocessing
UNIFIED_PREPROCESSING_ENABLED = os.getenv("UNIFIED_PREPROCESSING_ENABLED", "true").lower() == "true"


class QueryIntent(str, Enum):
    """Types of query intents."""
    DOCUMENT_SEARCH = "document_search"
    DATA_ANALYTICS = "data_analytics"
    HYBRID = "hybrid"
    GENERAL = "general"


class DissatisfactionType(str, Enum):
    """Types of user dissatisfaction signals."""
    NONE = "none"
    INCORRECT = "incorrect"
    UNCLEAR = "unclear"
    REFRESH_REQUEST = "refresh_request"
    VERIFICATION = "verification"


@dataclass
class ContextAnalysisResult:
    """Context analysis results."""
    topics: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    detected_pronouns: List[str] = field(default_factory=list)
    resolved_message: str = ""
    has_pronouns: bool = False


@dataclass
class CacheAnalysisResult:
    """Cache analysis results."""
    is_dissatisfied: bool = False
    dissatisfaction_type: DissatisfactionType = DissatisfactionType.NONE
    should_bypass_cache: bool = False
    should_invalidate_previous: bool = False
    is_self_contained: bool = True
    has_unresolved_references: bool = False
    is_cacheable: bool = True
    cache_reason: str = ""
    cache_key_question: str = ""


@dataclass
class IntentClassificationResult:
    """Intent classification results."""
    intent: QueryIntent = QueryIntent.DOCUMENT_SEARCH
    confidence: float = 0.8
    reasoning: str = ""
    requires_extracted_data: bool = False
    suggested_schemas: List[str] = field(default_factory=list)
    detected_entities: List[str] = field(default_factory=list)
    detected_metrics: List[str] = field(default_factory=list)
    detected_time_range: Optional[Dict[str, str]] = None


@dataclass
class UnifiedPreprocessResult:
    """
    Complete unified preprocessing result from single LLM call.

    Contains all analysis needed for query routing:
    - Context analysis (topics, entities, pronouns)
    - Cache analysis (dissatisfaction, self-contained, cacheable)
    - Intent classification (routing decision)
    """
    context: ContextAnalysisResult = field(default_factory=ContextAnalysisResult)
    cache: CacheAnalysisResult = field(default_factory=CacheAnalysisResult)
    intent: IntentClassificationResult = field(default_factory=IntentClassificationResult)

    # Metadata
    original_message: str = ""
    analysis_method: str = "unified_llm"  # "unified_llm", "individual_llm", "heuristic"
    processing_time_ms: float = 0.0


class UnifiedQueryPreprocessor:
    """
    Unified query preprocessor that performs all analysis in a single LLM call.

    Replaces three separate LLM calls:
    1. ContextAnalyzer._analyze_with_llm()
    2. QueryCacheAnalyzer._llm_analyze()
    3. IntentClassifier._llm_classify()

    This consolidation reduces latency by ~66% and token costs by ~66%.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the preprocessor.

        Args:
            llm_client: Optional LLM client with generate() method
        """
        self.llm_client = llm_client
        self._cached_llm_client = None

    def _get_llm_client(self):
        """Lazily initialize LLM client if not provided."""
        if self.llm_client:
            return self.llm_client

        if self._cached_llm_client:
            return self._cached_llm_client

        try:
            from rag_service.llm_service import get_llm_service
            from langchain_core.messages import HumanMessage

            llm_service = get_llm_service()
            if not llm_service.is_available():
                logger.warning("[UnifiedPreprocessor] LLM service not available")
                return None

            # Use a fast model for preprocessing
            chat_model = llm_service.get_query_model(
                temperature=0.1,
                num_ctx=4096,
                num_predict=1024
            )

            class LLMClientWrapper:
                def __init__(self, model):
                    self.model = model

                def generate(self, prompt: str) -> str:
                    response = self.model.invoke([HumanMessage(content=prompt)])
                    return response.content

            self._cached_llm_client = LLMClientWrapper(chat_model)
            logger.info("[UnifiedPreprocessor] LLM client initialized")
            return self._cached_llm_client

        except Exception as e:
            logger.warning(f"[UnifiedPreprocessor] Failed to create LLM client: {e}")
            return None

    def preprocess(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        previous_response: Optional[str] = None,
        available_schemas: Optional[List[str]] = None,
        use_llm: Optional[bool] = None
    ) -> UnifiedPreprocessResult:
        """
        Perform unified preprocessing with a single LLM call.

        This single call performs:
        1. Context analysis (topics, entities, pronouns)
        2. Cache analysis (dissatisfaction, self-contained, cacheable)
        3. Intent classification (routing decision)

        Args:
            message: Current user message
            chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            previous_response: The previous system response (for dissatisfaction check)
            available_schemas: List of available schema types for analytics
            use_llm: Override default LLM usage

        Returns:
            UnifiedPreprocessResult with all analysis results
        """
        import time
        start_time = time.time()

        logger.info("=" * 80)
        logger.info("[UnifiedPreprocessor] ========== UNIFIED PREPROCESSING START ==========")
        logger.info("=" * 80)
        logger.info(f"[UnifiedPreprocessor] Message: {message[:100]}...")
        logger.info(f"[UnifiedPreprocessor] Chat history length: {len(chat_history) if chat_history else 0}")
        logger.info(f"[UnifiedPreprocessor] Has previous response: {previous_response is not None}")
        logger.info(f"[UnifiedPreprocessor] Available schemas: {available_schemas}")

        should_use_llm = use_llm if use_llm is not None else UNIFIED_PREPROCESSING_ENABLED
        llm_client = self._get_llm_client() if should_use_llm else None

        if llm_client:
            try:
                result = self._unified_llm_analyze(
                    message,
                    chat_history,
                    previous_response,
                    available_schemas,
                    llm_client
                )
                result.processing_time_ms = (time.time() - start_time) * 1000
                self._log_result(result)
                return result
            except Exception as e:
                logger.warning(f"[UnifiedPreprocessor] LLM analysis failed: {e}")
                logger.info("[UnifiedPreprocessor] Falling back to heuristic analysis...")

        # Fallback to heuristics
        result = self._heuristic_analyze(message, chat_history, previous_response, available_schemas)
        result.processing_time_ms = (time.time() - start_time) * 1000
        self._log_result(result)
        return result

    def _unified_llm_analyze(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]]],
        previous_response: Optional[str],
        available_schemas: Optional[List[str]],
        llm_client
    ) -> UnifiedPreprocessResult:
        """Perform unified LLM-based analysis."""

        # Format inputs
        history_str = self._format_chat_history(chat_history)
        previous_response_str = previous_response[:500] if previous_response else "(No previous response)"
        schemas_str = ", ".join(available_schemas) if available_schemas else "invoice, receipt, bank_statement, expense_report, purchase_order, shipping_manifest, inventory_report, spreadsheet"

        prompt = self._build_unified_prompt(message, history_str, previous_response_str, schemas_str)

        response = llm_client.generate(prompt)

        # Parse JSON from response
        json_str = self._extract_json(response)
        result_dict = json.loads(json_str)

        return self._parse_llm_result(result_dict, message, available_schemas)

    def _build_unified_prompt(
        self,
        message: str,
        history_str: str,
        previous_response_str: str,
        schemas_str: str
    ) -> str:
        """Build the unified analysis prompt."""
        return f"""You are a query pre-processor for a document management system. Perform ALL of the following analyses in a SINGLE response.

## INPUT

Current User Message: "{message}"

Chat History (last 5 messages):
{history_str}

Previous System Response: "{previous_response_str}"

Available Document Types: {schemas_str}

## ANALYSIS TASKS

### 1. CONTEXT ANALYSIS
Extract from the conversation:
- Topics: Main subjects being discussed (e.g., "tire return", "invoice", "expenses")
- Entities: Named items mentioned (documents, people, products, dates)
- Pronouns: Any pronouns in current message that refer to previous context
- If pronouns exist, provide resolved_message with pronouns replaced by actual referents

### 2. CACHE ANALYSIS
Determine cache behavior:

A) DISSATISFACTION CHECK
- Is user expressing dissatisfaction with the previous response?
- Signals: "that's wrong", "check again", "not what I asked", "are you sure?"

B) SELF-CONTAINED CHECK
- Is the question understandable WITHOUT chat history?
- IMPORTANT: Possessive pronouns in general questions are SELF-CONTAINED:
  * "How to return my tire?" = SELF-CONTAINED (policy question)
  * "What is my order status?" = SELF-CONTAINED (general inquiry)
- NOT self-contained: "what about it?", "the previous one", "and the other?"

C) CACHE WORTHINESS
- Worth caching: policy questions, how-to questions, specific queries
- NOT worth caching: greetings, meta questions, highly temporal questions

### 3. INTENT CLASSIFICATION
Route the query:

- DATA_ANALYTICS: Numerical aggregations, sums, counts, averages, comparisons
  * "how many", "total", "average", "compare", "top N", "by month"

- DOCUMENT_SEARCH: Finding/reading documents, policies, procedures, how-to
  * "find documents", "what does X say", "how to", "return policy"
  * Questions about policies, procedures, terms, instructions

- HYBRID: Both search AND calculations needed
  * "Find invoices over $1000 and calculate total"

- GENERAL: ONLY for greetings or completely unrelated questions
  * "hello", "thanks", "how do I use this app"
  * If the question could be answered by documents, use DOCUMENT_SEARCH!

Respond with ONLY valid JSON (no markdown, no code blocks):
{{
  "context": {{
    "topics": ["topic1", "topic2"],
    "entities": {{
      "documents": [],
      "people": [],
      "products": [],
      "dates": []
    }},
    "detected_pronouns": [],
    "resolved_message": "message with pronouns resolved or original"
  }},
  "cache": {{
    "is_dissatisfied": false,
    "dissatisfaction_type": "none|incorrect|unclear|refresh_request|verification",
    "should_bypass_cache": false,
    "should_invalidate_previous": false,
    "is_self_contained": true,
    "has_unresolved_references": false,
    "is_cacheable": true,
    "cache_reason": "brief reason"
  }},
  "intent": {{
    "intent": "DOCUMENT_SEARCH|DATA_ANALYTICS|HYBRID|GENERAL",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "requires_extracted_data": false,
    "suggested_schemas": [],
    "detected_entities": [],
    "detected_metrics": []
  }}
}}"""

    def _format_chat_history(self, chat_history: Optional[List[Dict[str, str]]]) -> str:
        """Format chat history for the prompt."""
        if not chat_history:
            return "(No chat history)"

        lines = []
        for msg in chat_history[-5:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:200]
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*\n?', '', text)
            text = re.sub(r'\n?```\s*$', '', text)

        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return match.group(0)

        return text

    def _parse_llm_result(
        self,
        result: Dict[str, Any],
        original_message: str,
        available_schemas: Optional[List[str]]
    ) -> UnifiedPreprocessResult:
        """Parse LLM result dict into UnifiedPreprocessResult."""

        # Parse context
        context_data = result.get("context", {})
        entities = context_data.get("entities", {})
        context = ContextAnalysisResult(
            topics=context_data.get("topics", [])[:5],
            entities={
                "documents": entities.get("documents", [])[:5],
                "people": entities.get("people", [])[:5],
                "topics": context_data.get("topics", [])[:5],
                "objects": entities.get("products", [])[:5],
                "dates": entities.get("dates", [])[:5]
            },
            detected_pronouns=context_data.get("detected_pronouns", []),
            resolved_message=context_data.get("resolved_message", original_message),
            has_pronouns=len(context_data.get("detected_pronouns", [])) > 0
        )

        # Parse cache
        cache_data = result.get("cache", {})
        cache = CacheAnalysisResult(
            is_dissatisfied=cache_data.get("is_dissatisfied", False),
            dissatisfaction_type=DissatisfactionType(cache_data.get("dissatisfaction_type", "none").lower()),
            should_bypass_cache=cache_data.get("should_bypass_cache", False),
            should_invalidate_previous=cache_data.get("should_invalidate_previous", False),
            is_self_contained=cache_data.get("is_self_contained", True),
            has_unresolved_references=cache_data.get("has_unresolved_references", False),
            is_cacheable=cache_data.get("is_cacheable", True),
            cache_reason=cache_data.get("cache_reason", ""),
            cache_key_question=context.resolved_message or original_message
        )

        # Parse intent
        intent_data = result.get("intent", {})
        intent_str = intent_data.get("intent", "document_search").lower()

        # Extract time range from message
        time_range = self._extract_time_range(original_message.lower())

        intent = IntentClassificationResult(
            intent=QueryIntent(intent_str),
            confidence=float(intent_data.get("confidence", 0.8)),
            reasoning=intent_data.get("reasoning", ""),
            requires_extracted_data=intent_data.get("requires_extracted_data", False),
            suggested_schemas=intent_data.get("suggested_schemas", []),
            detected_entities=intent_data.get("detected_entities", []),
            detected_metrics=intent_data.get("detected_metrics", []),
            detected_time_range=time_range
        )

        return UnifiedPreprocessResult(
            context=context,
            cache=cache,
            intent=intent,
            original_message=original_message,
            analysis_method="unified_llm"
        )

    def _heuristic_analyze(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]]],
        previous_response: Optional[str],
        available_schemas: Optional[List[str]]
    ) -> UnifiedPreprocessResult:
        """Fallback heuristic-based analysis when LLM is unavailable."""

        message_lower = message.lower().strip()

        # Context analysis (simple)
        context = ContextAnalysisResult(
            topics=self._detect_topics_heuristic(message_lower),
            entities={"documents": [], "people": [], "topics": [], "objects": [], "dates": []},
            detected_pronouns=self._detect_pronouns_heuristic(message_lower),
            resolved_message=message,
            has_pronouns=False
        )
        context.has_pronouns = len(context.detected_pronouns) > 0

        # Cache analysis (simple)
        cache = CacheAnalysisResult(
            is_dissatisfied=self._detect_dissatisfaction_heuristic(message_lower),
            dissatisfaction_type=DissatisfactionType.NONE,
            should_bypass_cache=False,
            should_invalidate_previous=False,
            is_self_contained=not context.has_pronouns,
            has_unresolved_references=context.has_pronouns,
            is_cacheable=not self._is_greeting_heuristic(message_lower),
            cache_reason="Heuristic analysis",
            cache_key_question=message
        )

        if cache.is_dissatisfied:
            cache.should_bypass_cache = True

        # Intent classification (simple)
        intent = self._classify_intent_heuristic(message_lower, available_schemas)

        return UnifiedPreprocessResult(
            context=context,
            cache=cache,
            intent=intent,
            original_message=message,
            analysis_method="heuristic"
        )

    def _detect_topics_heuristic(self, message_lower: str) -> List[str]:
        """Detect topics using keyword matching."""
        topics = []
        topic_keywords = {
            "invoice": ["invoice", "bill", "billing"],
            "expense": ["expense", "cost", "spending"],
            "receipt": ["receipt", "purchase"],
            "return": ["return", "refund"],
            "shipping": ["shipping", "delivery"],
            "order": ["order", "purchase order"],
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in message_lower for kw in keywords):
                topics.append(topic)

        return topics[:5]

    def _detect_pronouns_heuristic(self, message_lower: str) -> List[str]:
        """Detect pronouns in message."""
        pronouns = ['it', 'they', 'them', 'this', 'that', 'these', 'those']
        detected = []
        words = re.findall(r'\b\w+\b', message_lower)
        for pronoun in pronouns:
            if pronoun in words:
                detected.append(pronoun)
        return detected

    def _detect_dissatisfaction_heuristic(self, message_lower: str) -> bool:
        """Detect dissatisfaction signals."""
        patterns = [
            r"\b(that's wrong|that is wrong|not correct|incorrect)\b",
            r"\b(check again|try again|refresh)\b",
            r"\b(are you sure|doesn't look right)\b",
        ]
        return any(re.search(p, message_lower) for p in patterns)

    def _is_greeting_heuristic(self, message_lower: str) -> bool:
        """Check if message is a greeting."""
        greetings = [r"^(hello|hi|hey|thanks|thank you|bye)(\s|$)"]
        return any(re.search(p, message_lower) for p in greetings)

    def _classify_intent_heuristic(
        self,
        message_lower: str,
        available_schemas: Optional[List[str]]
    ) -> IntentClassificationResult:
        """Classify intent using pattern matching."""

        analytics_patterns = [
            r'\b(total|sum|average|count|max|min)\b',
            r'\b(how many|how much)\b',
            r'\b(compare|comparison)\b',
        ]

        search_patterns = [
            r'\b(find|search|look for)\b',
            r'\b(how to|what is|tell me about)\b',
            r'\b(policy|procedure|return|refund)\b',
        ]

        analytics_score = sum(1 for p in analytics_patterns if re.search(p, message_lower))
        search_score = sum(1 for p in search_patterns if re.search(p, message_lower))

        if analytics_score > search_score:
            intent = QueryIntent.DATA_ANALYTICS
            confidence = 0.7
            reasoning = "Query contains analytics patterns"
        elif search_score > 0:
            intent = QueryIntent.DOCUMENT_SEARCH
            confidence = 0.7
            reasoning = "Query contains search patterns"
        else:
            intent = QueryIntent.DOCUMENT_SEARCH
            confidence = 0.5
            reasoning = "Default to document search"

        time_range = self._extract_time_range(message_lower)

        return IntentClassificationResult(
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            requires_extracted_data=intent == QueryIntent.DATA_ANALYTICS,
            suggested_schemas=[],
            detected_entities=[],
            detected_metrics=[],
            detected_time_range=time_range
        )

    def _extract_time_range(self, query_lower: str) -> Optional[Dict[str, str]]:
        """Extract time range from query."""
        now = datetime.now()
        current_year = now.year

        year_match = re.search(r'\b(20\d{2})\b', query_lower)
        mentioned_year = int(year_match.group(1)) if year_match else current_year

        # Quarter patterns
        if 'q1' in query_lower or 'first quarter' in query_lower:
            return {"start": f"{mentioned_year}-01-01", "end": f"{mentioned_year}-03-31"}
        elif 'q2' in query_lower or 'second quarter' in query_lower:
            return {"start": f"{mentioned_year}-04-01", "end": f"{mentioned_year}-06-30"}
        elif 'q3' in query_lower or 'third quarter' in query_lower:
            return {"start": f"{mentioned_year}-07-01", "end": f"{mentioned_year}-09-30"}
        elif 'q4' in query_lower or 'fourth quarter' in query_lower:
            return {"start": f"{mentioned_year}-10-01", "end": f"{mentioned_year}-12-31"}

        # Last X patterns
        last_match = re.search(r'last\s+(\d+)\s+(month|day|week|year)s?', query_lower)
        if last_match:
            count = int(last_match.group(1))
            unit = last_match.group(2)
            if unit == 'day':
                start = now - timedelta(days=count)
            elif unit == 'week':
                start = now - timedelta(weeks=count)
            elif unit == 'month':
                start = now - timedelta(days=count * 30)
            elif unit == 'year':
                start = now - timedelta(days=count * 365)
            else:
                start = now
            return {"start": start.strftime("%Y-%m-%d"), "end": now.strftime("%Y-%m-%d")}

        # This month/year
        if 'this month' in query_lower:
            return {"start": f"{now.year}-{now.month:02d}-01", "end": now.strftime("%Y-%m-%d")}
        elif 'this year' in query_lower:
            return {"start": f"{now.year}-01-01", "end": now.strftime("%Y-%m-%d")}

        # Only year mentioned
        if year_match and mentioned_year != current_year:
            return {"start": f"{mentioned_year}-01-01", "end": f"{mentioned_year}-12-31"}

        return None

    def _log_result(self, result: UnifiedPreprocessResult):
        """Log the preprocessing result."""
        logger.info("-" * 80)
        logger.info(f"[UnifiedPreprocessor] RESULT ({result.analysis_method}):")
        logger.info(f"[UnifiedPreprocessor] Processing time: {result.processing_time_ms:.2f}ms")
        logger.info(f"[UnifiedPreprocessor] --- CONTEXT ---")
        logger.info(f"[UnifiedPreprocessor]   Topics: {result.context.topics}")
        logger.info(f"[UnifiedPreprocessor]   Has pronouns: {result.context.has_pronouns}")
        logger.info(f"[UnifiedPreprocessor]   Detected pronouns: {result.context.detected_pronouns}")
        logger.info(f"[UnifiedPreprocessor] --- CACHE ---")
        logger.info(f"[UnifiedPreprocessor]   Dissatisfied: {result.cache.is_dissatisfied} ({result.cache.dissatisfaction_type.value})")
        logger.info(f"[UnifiedPreprocessor]   Self-contained: {result.cache.is_self_contained}")
        logger.info(f"[UnifiedPreprocessor]   Cacheable: {result.cache.is_cacheable}")
        logger.info(f"[UnifiedPreprocessor]   Bypass cache: {result.cache.should_bypass_cache}")
        logger.info(f"[UnifiedPreprocessor] --- INTENT ---")
        logger.info(f"[UnifiedPreprocessor]   Intent: {result.intent.intent.value}")
        logger.info(f"[UnifiedPreprocessor]   Confidence: {result.intent.confidence:.2f}")
        logger.info(f"[UnifiedPreprocessor]   Reasoning: {result.intent.reasoning}")
        logger.info(f"[UnifiedPreprocessor]   Requires extracted data: {result.intent.requires_extracted_data}")
        logger.info(f"[UnifiedPreprocessor]   Time range: {result.intent.detected_time_range}")
        logger.info("=" * 80)
        logger.info("[UnifiedPreprocessor] ========== UNIFIED PREPROCESSING END ==========")
        logger.info("=" * 80)


# Singleton instance
_preprocessor: Optional[UnifiedQueryPreprocessor] = None


def get_unified_preprocessor() -> UnifiedQueryPreprocessor:
    """Get or create the unified preprocessor singleton."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = UnifiedQueryPreprocessor()
    return _preprocessor


# Convenience function
def preprocess_query(
    message: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    previous_response: Optional[str] = None,
    available_schemas: Optional[List[str]] = None
) -> UnifiedPreprocessResult:
    """
    Convenience function to preprocess a query.

    Args:
        message: Current user message
        chat_history: List of previous messages
        previous_response: The previous system response
        available_schemas: List of available schema types

    Returns:
        UnifiedPreprocessResult with all analysis results
    """
    preprocessor = get_unified_preprocessor()
    return preprocessor.preprocess(message, chat_history, previous_response, available_schemas)
