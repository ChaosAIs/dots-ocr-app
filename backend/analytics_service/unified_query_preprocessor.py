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

            # Use a fast model for preprocessing with minimal token generation
            # Simplified prompt expects ~100 tokens response max
            chat_model = llm_service.get_query_model(
                temperature=0.1,
                num_ctx=2048,  # Reduced context - prompt is now much smaller
                num_predict=256  # Reduced from 1024 - simplified response needs ~100 tokens
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

        # Log the LLM response for debugging context resolution
        logger.info(f"[UnifiedPreprocessor] LLM raw response: {response[:500]}...")
        logger.info(f"[UnifiedPreprocessor] Parsed resolved_message: {result_dict.get('resolved_message', 'NOT FOUND')[:200]}")

        return self._parse_llm_result(result_dict, message, available_schemas)

    def _build_unified_prompt(
        self,
        message: str,
        history_str: str,
        previous_response_str: str,
        schemas_str: str
    ) -> str:
        """Build the unified analysis prompt - optimized for speed with minimal output."""
        return f"""Query preprocessor. Analyze and respond with JSON only.

Message: "{message}"
History: {history_str}
Previous response: "{previous_response_str}"

Tasks:
1. CONTEXT RESOLUTION (CRITICAL - MUST DO THIS FIRST):
   STEP 1: Check if the current message contains ANY of these references: "their", "them", "they", "it", "this", "that", "those", "these", "above", "same", "previous"
   STEP 2: If YES, look at the MOST RECENT USER message in History to find what entity/subject it refers to
   STEP 3: REPLACE the pronoun/reference with the actual subject + any conditions from history

   CRITICAL RULE: "their details", "list them", "show their info" = user wants details of items from PREVIOUS query!
   - If previous query asked about "meals/invoices/receipts in 2025" -> resolved_message = "list all meals/invoices/receipts details from 2025"
   - If previous query asked about "products with inventory < 50" -> resolved_message = "list all product details where inventory < 50"

   **CRITICAL - PRESERVE ENTITY NAMES (MUST DO):**
   - If user mentions a SPECIFIC vendor/company/store/person name, ALWAYS keep it in resolved_message!
   - Company/vendor names examples: "Augment Code", "Amazon", "Best Buy", "Walmart", "Apple", "Microsoft"
   - NEVER generalize or drop these names! They are filters, not optional context.

   EXAMPLE - ENTITY PRESERVATION:
   - Message: "list the augment code invoices details for me"
   - resolved_message MUST BE: "list augment code invoice details" (KEEP "augment code"!)
   - WRONG: "list all invoice details" (DROPPED the vendor name - THIS IS WRONG!)

   - Message: "show Amazon receipts from 2025"
   - resolved_message MUST BE: "show Amazon receipts from 2025" (KEEP "Amazon"!)
   - WRONG: "show all receipts from 2025" (DROPPED the store name - THIS IS WRONG!)

   - Message: "how many Best Buy purchases last month"
   - resolved_message MUST BE: "how many Best Buy purchases last month" (KEEP "Best Buy"!)

   EXAMPLE - PRONOUN RESOLUTION:
   - History: user: "how many meals invoices for 2025"
   - Current: "can you list all of their details?"
   - resolved_message MUST BE: "list all meals invoice details from 2025"
   - INTENT MUST BE: DATA_ANALYTICS

   EXAMPLE - CONDITION PRESERVATION:
   - History: user: "how many products with inventory lower than 50 but higher than 30?"
   - Current: "list all product details for above condition"
   - resolved_message MUST BE: "list all product details where inventory lower than 50 but higher than 30"
   - INTENT MUST BE: DATA_ANALYTICS

   CRITICAL - IGNORE RESULT COUNTS:
   - "document_count is 7", "the 7 invoices" = RESULT COUNTS, NOT database filters! IGNORE these numbers.
   - Use the ORIGINAL CONDITION (e.g., "from 2025") instead.

2. DISSATISFIED: Is user unhappy with previous response OR requesting fresh/latest data? ("wrong", "check again", "refresh", "are you sure", "latest data", "latest", "fresh data", "current data", "up to date", "most recent")

3. CACHEABLE: Worth caching? No for greetings/meta questions. Yes for factual queries. NO if user explicitly asks for "latest" or "fresh" data.

4. INTENT:
   - DATA_ANALYTICS: counts, totals, averages, comparisons, OR listing/showing details of invoices/receipts/meals/products/transactions (requires SQL on extracted data)
   - DOCUMENT_SEARCH: find/read policy docs, how-to guides, manuals, documentation
   - HYBRID: needs both analytics and document content
   - GENERAL: greetings only
   NOTE: If resolved_message is about listing invoice/receipt/meal/product details, intent MUST be DATA_ANALYTICS!

JSON only, no markdown:
{{"topics":["invoice"],"resolved_message":"list augment code invoice details","is_dissatisfied":false,"bypass_cache":false,"invalidate_previous":false,"is_cacheable":true,"intent":"DATA_ANALYTICS"}}"""

    def _format_chat_history(self, chat_history: Optional[List[Dict[str, str]]]) -> str:
        """Format chat history for the prompt.

        Includes USER messages fully and ASSISTANT messages summarized (to avoid token overflow).
        Assistant responses can be very large (analytics reports), so we only keep a brief summary.
        """
        if not chat_history:
            return "(No chat history)"

        formatted_messages = []
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content.strip():
                continue

            if role == "user":
                # User messages: keep more content (these contain the conditions we need)
                truncated = content[:400] if len(content) <= 400 else content[:400] + "..."
                formatted_messages.append(f"user: {truncated}")
            elif role == "assistant":
                # Assistant messages: heavily truncate - we only need to know the topic/subject
                # Extract just the first line or first 100 chars to indicate what was discussed
                first_line = content.split('\n')[0][:100]
                formatted_messages.append(f"assistant: {first_line}...")

        # Take last 6 messages (3 exchanges)
        recent_messages = formatted_messages[-6:]

        return "\n".join(recent_messages)

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
        """Parse LLM result dict into UnifiedPreprocessResult.

        Supports both simplified (flat) and legacy (nested) response formats.
        Simplified format is preferred for faster LLM responses.
        """

        # Check if this is simplified (flat) or legacy (nested) format
        is_simplified = "topics" in result and "context" not in result

        if is_simplified:
            # Parse simplified flat format
            topics = result.get("topics", [])[:5]
            resolved_message = result.get("resolved_message", original_message)

            # Detect if pronouns were resolved (message changed)
            has_pronouns = resolved_message != original_message and resolved_message != ""

            context = ContextAnalysisResult(
                topics=topics,
                entities={"documents": [], "people": [], "topics": topics, "objects": [], "dates": []},
                detected_pronouns=[],  # Not needed downstream
                resolved_message=resolved_message if resolved_message else original_message,
                has_pronouns=has_pronouns
            )

            # Parse cache fields from flat structure
            is_dissatisfied = result.get("is_dissatisfied", False)
            cache = CacheAnalysisResult(
                is_dissatisfied=is_dissatisfied,
                dissatisfaction_type=DissatisfactionType.REFRESH_REQUEST if is_dissatisfied else DissatisfactionType.NONE,
                should_bypass_cache=result.get("bypass_cache", is_dissatisfied),
                should_invalidate_previous=result.get("invalidate_previous", is_dissatisfied),
                is_self_contained=True,  # Default, not critical
                has_unresolved_references=False,  # Not used downstream
                is_cacheable=result.get("is_cacheable", True),
                cache_reason="",  # Not used downstream
                cache_key_question=context.resolved_message or original_message
            )

            # Parse intent from flat structure
            intent_str = result.get("intent", "document_search").lower()
            time_range = self._extract_time_range(original_message.lower())

            intent = IntentClassificationResult(
                intent=QueryIntent(intent_str),
                confidence=0.85,  # Default confidence for simplified format
                reasoning="",  # Not used downstream
                requires_extracted_data=intent_str == "data_analytics",
                suggested_schemas=[],  # Not used downstream
                detected_entities=[],  # Not used downstream
                detected_metrics=[],  # Not used downstream
                detected_time_range=time_range
            )
        else:
            # Parse legacy nested format for backward compatibility
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

            intent_data = result.get("intent", {})
            intent_str = intent_data.get("intent", "document_search").lower()
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

        # Detect pronouns and contextual references
        detected_pronouns = self._detect_pronouns_heuristic(message_lower)
        contextual_refs = self._detect_contextual_references(message_lower)
        has_context_refs = len(detected_pronouns) > 0 or len(contextual_refs) > 0

        # Try to resolve contextual references from chat history
        resolved_message = message
        if has_context_refs and chat_history:
            resolved_message = self._resolve_context_heuristic(message, chat_history, contextual_refs)

        # Context analysis (simple)
        context = ContextAnalysisResult(
            topics=self._detect_topics_heuristic(message_lower),
            entities={"documents": [], "people": [], "topics": [], "objects": [], "dates": []},
            detected_pronouns=detected_pronouns + contextual_refs,  # Include contextual refs as "pronouns" for downstream
            resolved_message=resolved_message,
            has_pronouns=has_context_refs
        )

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
        """Detect pronouns and contextual references in message."""
        pronouns = ['it', 'they', 'them', 'this', 'that', 'these', 'those', 'their']
        detected = []
        words = re.findall(r'\b\w+\b', message_lower)
        for pronoun in pronouns:
            if pronoun in words:
                detected.append(pronoun)
        return detected

    def _detect_contextual_references(self, message_lower: str) -> List[str]:
        """Detect contextual references that require resolution from chat history."""
        # Patterns that indicate the user is referencing previous conditions/criteria
        reference_patterns = [
            (r'\babove\s+condition', 'above condition'),
            (r'\bsame\s+criteria', 'same criteria'),
            (r'\bprevious\s+filter', 'previous filter'),
            (r'\bthose\s+results', 'those results'),
            (r'\bthe\s+same\b', 'the same'),
            (r'\bmentioned\s+before', 'mentioned before'),
            (r'\bfor\s+above\b', 'for above'),
            (r'\bsame\s+query', 'same query'),
            (r'\bbased\s+on\s+above', 'based on above'),
            (r'\bthat\s+criteria', 'that criteria'),
            (r'\babove\s+filter', 'above filter'),
            (r'\bprevious\s+condition', 'previous condition'),
            (r'\bsame\s+condition', 'same condition'),
            (r'\babove\s+query', 'above query'),
            (r'\bpreviously\s+mentioned', 'previously mentioned'),
        ]
        detected = []
        for pattern, label in reference_patterns:
            if re.search(pattern, message_lower):
                detected.append(label)
        return detected

    def _has_context_references(self, message_lower: str) -> bool:
        """Check if message has any context references that need resolution."""
        pronouns = self._detect_pronouns_heuristic(message_lower)
        contextual_refs = self._detect_contextual_references(message_lower)
        return len(pronouns) > 0 or len(contextual_refs) > 0

    def _resolve_context_heuristic(
        self,
        message: str,
        chat_history: List[Dict[str, str]],
        contextual_refs: List[str]
    ) -> str:
        """
        Heuristic-based resolution of contextual references from chat history.

        This is a fallback when LLM is not available. It extracts conditions from
        the most recent user message that contains filter/condition patterns.
        """
        if not chat_history or not contextual_refs:
            return message

        # Look for the most recent user message with conditions/filters
        condition_patterns = [
            # Time/date patterns (important for analytics follow-ups)
            r'(?:in|from|during|for)\s+(?:year\s+)?\d{4}',  # in 2025, from 2024, during year 2023
            r'(?:in|from|during)\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)',
            r'(?:last|this|past)\s+(?:week|month|quarter|year)',
            r'(?:q[1-4]|quarter\s*[1-4])\s*\d{4}',  # Q1 2025, quarter 1 2024
            # Numeric comparisons
            r'(?:where|with|having|that have|which have)?\s*\w+\s*(?:>|<|>=|<=|=|!=|between|lower than|higher than|greater than|less than|more than|equal to)\s*\d+',
            # Between patterns
            r'\w+\s+between\s+\d+\s+and\s+\d+',
            # Lower/higher than patterns
            r'(?:lower|higher|greater|less|more)\s+than\s+\d+',
            # Inventory/quantity specific patterns
            r'inventory\s+(?:lower|higher|greater|less|more)\s+than\s+\d+',
            r'inventory\s+(?:>|<|>=|<=)\s*\d+',
            # Price/amount patterns
            r'price\s+(?:>|<|>=|<=|lower than|higher than|greater than|less than|more than)\s*\d+',
        ]

        # Find the most recent user message with conditions
        previous_condition = None
        for msg in reversed(chat_history):
            if msg.get("role") == "user":
                msg_content = msg.get("content", "")
                msg_lower = msg_content.lower()

                # Try to extract conditions
                for pattern in condition_patterns:
                    match = re.search(pattern, msg_lower)
                    if match:
                        # Found a condition - extract the relevant part of the message
                        previous_condition = msg_content
                        break

                if previous_condition:
                    break

        if not previous_condition:
            return message

        # Extract the actual condition from the previous message
        # Pattern to find the WHERE-like clause
        condition_extract_patterns = [
            # Time/date patterns - check these first as they're common for analytics follow-ups
            (r'(?:in|from|during|for)\s+((?:year\s+)?\d{4})', 'year'),  # in 2025, from 2024
            (r'(?:in|from|during)\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?', 'month'),
            (r'(last|this|past)\s+(week|month|quarter|year)', 'relative'),
            # "inventory lower than X but higher than Y" pattern
            (r'(\w+)\s+(lower\s+than\s+\d+\s+(?:but\s+)?(?:higher|greater)\s+than\s+\d+)', 'comparison'),
            (r'(\w+)\s+(higher\s+than\s+\d+\s+(?:but\s+)?(?:lower|less)\s+than\s+\d+)', 'comparison'),
            (r'(\w+)\s+(between\s+\d+\s+and\s+\d+)', 'comparison'),
            (r'(\w+)\s*([<>=!]+\s*\d+)', 'comparison'),
            (r'(\w+)\s+(lower\s+than\s+\d+)', 'comparison'),
            (r'(\w+)\s+(higher\s+than\s+\d+)', 'comparison'),
            (r'(\w+)\s+(greater\s+than\s+\d+)', 'comparison'),
            (r'(\w+)\s+(less\s+than\s+\d+)', 'comparison'),
        ]

        extracted_condition = None
        for pattern_tuple in condition_extract_patterns:
            pattern, condition_type = pattern_tuple
            match = re.search(pattern, previous_condition.lower())
            if match:
                if condition_type == 'year':
                    # Time condition: "in 2025" -> "from 2025"
                    year_part = match.group(1)
                    extracted_condition = f"from {year_part}"
                elif condition_type == 'month':
                    # Month condition: "in January 2025"
                    month = match.group(1)
                    year = match.group(2) if match.lastindex >= 2 else ""
                    extracted_condition = f"from {month} {year}".strip()
                elif condition_type == 'relative':
                    # Relative time: "last month", "this year"
                    extracted_condition = f"{match.group(1)} {match.group(2)}"
                else:
                    # Comparison condition
                    field = match.group(1)
                    condition = match.group(2)
                    extracted_condition = f"{field} {condition}"
                break

        if not extracted_condition:
            # Fallback: just append the previous user question context
            extracted_condition = previous_condition

        # Replace the contextual reference with the actual condition
        resolved = message

        # Sort refs by length (longest first) to avoid partial replacements
        sorted_refs = sorted(contextual_refs, key=len, reverse=True)

        for ref in sorted_refs:
            # Different replacement strategies based on the reference type
            if 'for above' in ref or ref == 'above condition' or ref == 'above filter' or ref == 'above query':
                # Replace "for above condition" style with "where <condition>"
                ref_patterns = [
                    (rf'\bfor\s+{re.escape(ref)}', f'where {extracted_condition}'),
                    (rf'\b{re.escape(ref)}', f'where {extracted_condition}'),
                ]
            elif 'based on above' in ref or 'same criteria' in ref or 'same condition' in ref:
                # Replace "based on above" style - append the condition
                ref_patterns = [
                    (rf'\b{re.escape(ref)}', f'where {extracted_condition}'),
                ]
            elif ref == 'the same':
                # Replace "the same" with condition
                ref_patterns = [
                    (rf'\b{re.escape(ref)}\b', f'with {extracted_condition}'),
                ]
            elif ref == 'those results':
                # Replace "for those results" with condition
                ref_patterns = [
                    (rf'\bfor\s+{re.escape(ref)}\b', f'where {extracted_condition}'),
                    (rf'\b{re.escape(ref)}\b', f'where {extracted_condition}'),
                ]
            else:
                # Default: replace with the condition
                ref_patterns = [
                    (rf'\b{re.escape(ref)}\b', f'where {extracted_condition}'),
                ]

            for ref_pattern, replacement in ref_patterns:
                resolved = re.sub(ref_pattern, replacement, resolved, flags=re.IGNORECASE)

        # Strip result count references like "document_count is 7", "where count is 10"
        # These are result counts from previous queries, NOT valid database filters
        result_count_patterns = [
            r'\bwhere\s+(?:document_count|count|record_count|total_count)\s*(?:is|=|==)\s*\d+',
            r'\b(?:document_count|count|record_count|total_count)\s*(?:is|=|==)\s*\d+',
        ]
        for pattern in result_count_patterns:
            resolved = re.sub(pattern, '', resolved, flags=re.IGNORECASE)

        # Clean up any double spaces or awkward phrasing
        resolved = re.sub(r'\s+', ' ', resolved).strip()

        logger.info(f"[Heuristic] Resolved contextual reference: '{message}' -> '{resolved}'")
        return resolved

    def _detect_dissatisfaction_heuristic(self, message_lower: str) -> bool:
        """Detect dissatisfaction signals or requests for fresh/latest data."""
        patterns = [
            r"\b(that's wrong|that is wrong|not correct|incorrect)\b",
            r"\b(check again|try again|refresh)\b",
            r"\b(are you sure|doesn't look right)\b",
            # Patterns for requesting fresh/latest data (should bypass cache)
            r"\b(latest data|latest|fresh data|current data|up to date|most recent)\b",
            r"\bfrom (latest|current|fresh|newest)\b",
            r"\b(get|show|fetch|retrieve).*(latest|current|fresh|newest)\b",
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
