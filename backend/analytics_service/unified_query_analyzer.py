"""
Unified Query Analyzer - Single LLM Call for Complete Query Analysis

This module consolidates ALL query analysis into a SINGLE LLM call:
1. Context Resolution (pronouns, references)
2. Cache Analysis (dissatisfaction, bypass, cacheable)
3. Intent Classification (data_analytics, document_search, hybrid)
4. Query Enhancement (entities, topics, document types for routing)

Benefits:
- Single LLM call (~500ms) instead of 2-3 calls (~2000-3000ms)
- Single source of truth for all query metadata
- Consistent entity/topic extraction across all components
- Simplified debugging with one log section
- Guaranteed clean cache keys

This replaces:
- unified_query_preprocessor.py (merged here)
- query_cache_analyzer.py (deprecated - use this instead)
- _analyze_query_with_llm in rag_agent.py (skip when using this)
"""

import os
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Environment variable to enable/disable unified analysis
UNIFIED_QUERY_ANALYZER_ENABLED = os.getenv("UNIFIED_QUERY_ANALYZER_ENABLED", "true").lower() == "true"

# Cache control words/phrases that should be stripped from cache keys
CACHE_CONTROL_PATTERNS = [
    r'\bwithout\s+cache\b',
    r'\bno\s+cache\b',
    r'\bskip\s+cache\b',
    r'\bbypass\s+cache\b',
    r'\bcache\s*off\b',
    r'\bdisable\s+cache\b',
    r'\bdon\'?t\s+use\s+cache\b',
    r'\bdo\s+not\s+use\s+cache\b',
    r'\bignore\s+cache\b',
    r'\bfresh\s+data\b',
    r'\blatest\s+data\b',
    r'\bcurrent\s+data\b',
    r'\bmost\s+recent\s+data\b',
    r'\bup\s+to\s+date\s+data\b',
]


def strip_cache_control_words(query: str) -> str:
    """
    Strip cache control words/phrases from a query string.

    Args:
        query: The original query string

    Returns:
        Query string with cache control words removed
    """
    result = query
    for pattern in CACHE_CONTROL_PATTERNS:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)

    # Clean up extra whitespace and punctuation
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s*[,\.]+\s*$', '', result)
    result = re.sub(r'^\s*[,\.]+\s*', '', result)
    result = re.sub(r'\s+[,\.]+\s+', ' ', result)
    result = result.strip()

    return result


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
class QueryAnalysisResult:
    """
    Complete unified query analysis result from single LLM call.

    This single dataclass replaces multiple separate result types:
    - UnifiedPreprocessResult
    - UnifiedCacheAnalysis
    - IntentClassification
    - Query enhancement metadata

    All downstream components should use this single result.
    """
    # Original and resolved messages
    original_message: str = ""
    resolved_message: str = ""  # With pronouns resolved, cache words stripped

    # Cache analysis
    cache_key: str = ""  # Clean cache key (no cache control words)
    is_cacheable: bool = True
    bypass_cache: bool = False
    invalidate_previous: bool = False
    is_dissatisfied: bool = False
    dissatisfaction_type: DissatisfactionType = DissatisfactionType.NONE

    # Intent classification
    intent: QueryIntent = QueryIntent.DOCUMENT_SEARCH
    intent_confidence: float = 0.85
    requires_extracted_data: bool = False

    # Query enhancement for document routing
    enhanced_query: str = ""  # Optimized for vector search
    entities: List[str] = field(default_factory=list)  # Named entities (companies, people, products)
    topics: List[str] = field(default_factory=list)  # Subject categories
    document_type_hints: List[str] = field(default_factory=list)  # Likely document types

    # Time range detection
    time_range: Optional[Dict[str, str]] = None  # {"start": "2025-01-01", "end": "2025-12-31"}

    # Metadata
    analysis_method: str = "unified_llm"  # "unified_llm" or "heuristic"
    processing_time_ms: float = 0.0

    # Legacy compatibility - entities as dict for downstream code expecting old format
    @property
    def entities_dict(self) -> Dict[str, List[str]]:
        """Return entities in legacy dict format for compatibility."""
        return {
            "named_entities": self.entities,
            "topics": self.topics,
        }


class UnifiedQueryAnalyzer:
    """
    Unified query analyzer that performs ALL analysis in a single LLM call.

    This consolidates:
    1. UnifiedQueryPreprocessor - context, cache, intent
    2. QueryCacheAnalyzer - cache decisions
    3. _analyze_query_with_llm - entities, topics, document types

    Single LLM call reduces latency by ~66% and token costs by ~66%.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the analyzer.

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
                logger.warning("[UnifiedQueryAnalyzer] LLM service not available")
                return None

            # Use fast model with optimized parameters
            chat_model = llm_service.get_query_model(
                temperature=0.1,
                num_ctx=4096,
                num_predict=512
            )

            class LLMClientWrapper:
                def __init__(self, model):
                    self.model = model

                def generate(self, prompt: str) -> str:
                    response = self.model.invoke([HumanMessage(content=prompt)])
                    return response.content

            self._cached_llm_client = LLMClientWrapper(chat_model)
            logger.info("[UnifiedQueryAnalyzer] LLM client initialized")
            return self._cached_llm_client

        except Exception as e:
            logger.warning(f"[UnifiedQueryAnalyzer] Failed to create LLM client: {e}")
            return None

    def analyze(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        previous_response: Optional[str] = None,
        available_schemas: Optional[List[str]] = None,
        use_llm: Optional[bool] = None
    ) -> QueryAnalysisResult:
        """
        Perform complete unified query analysis with a single LLM call.

        This single call performs:
        1. Context resolution (pronouns, references)
        2. Cache analysis (dissatisfaction, bypass, cacheable)
        3. Intent classification (routing decision)
        4. Query enhancement (entities, topics, document types)

        Args:
            message: Current user message
            chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            previous_response: The previous system response (for dissatisfaction check)
            available_schemas: List of available schema types for analytics
            use_llm: Override default LLM usage

        Returns:
            QueryAnalysisResult with all analysis results
        """
        start_time = time.time()

        logger.info("=" * 80)
        logger.info("[UnifiedQueryAnalyzer] ========== UNIFIED ANALYSIS START ==========")
        logger.info("=" * 80)
        logger.info(f"[UnifiedQueryAnalyzer] Message: {message[:100]}...")
        logger.info(f"[UnifiedQueryAnalyzer] Chat history: {len(chat_history) if chat_history else 0} messages")

        should_use_llm = use_llm if use_llm is not None else UNIFIED_QUERY_ANALYZER_ENABLED
        llm_client = self._get_llm_client() if should_use_llm else None

        if llm_client:
            try:
                result = self._llm_analyze(
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
                logger.warning(f"[UnifiedQueryAnalyzer] LLM analysis failed: {e}")
                logger.info("[UnifiedQueryAnalyzer] Falling back to heuristic analysis...")

        # Fallback to heuristics
        result = self._heuristic_analyze(message, chat_history, previous_response, available_schemas)
        result.processing_time_ms = (time.time() - start_time) * 1000
        self._log_result(result)
        return result

    def _llm_analyze(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]]],
        previous_response: Optional[str],
        available_schemas: Optional[List[str]],
        llm_client
    ) -> QueryAnalysisResult:
        """Perform LLM-based unified analysis."""

        # Format inputs
        history_str = self._format_chat_history(chat_history)
        prev_response_str = previous_response[:500] if previous_response else "(No previous response)"
        schemas_str = ", ".join(available_schemas) if available_schemas else "invoice, receipt, bank_statement, expense_report, purchase_order, shipping_manifest, inventory_report, spreadsheet"

        prompt = self._build_unified_prompt(message, history_str, prev_response_str, schemas_str)

        response = llm_client.generate(prompt)

        # Parse JSON response
        json_str = self._extract_json(response)
        result_dict = json.loads(json_str)

        logger.info(f"[UnifiedQueryAnalyzer] LLM response: {response[:300]}...")

        return self._parse_llm_result(result_dict, message)

    def _build_unified_prompt(
        self,
        message: str,
        history_str: str,
        prev_response_str: str,
        schemas_str: str
    ) -> str:
        """Build the unified analysis prompt."""
        return f"""You are a query analyzer. Analyze the user message and provide a complete analysis in JSON format.

MESSAGE: "{message}"
CHAT HISTORY: {history_str}
PREVIOUS RESPONSE: "{prev_response_str}"
AVAILABLE DOCUMENT TYPES: {schemas_str}

ANALYSIS TASKS:

1. CONTEXT RESOLUTION:
   - If message contains pronouns (their, them, it, this, that, these, those) or references (above, same, previous), resolve them using chat history
   - STRIP cache control phrases from resolved_message: "without cache", "no cache", "skip cache", "bypass cache", "fresh data", "latest data", "current data"
   - PRESERVE entity names (company/vendor/person names): "Augment Code", "Amazon", "Best Buy", etc.

   Example: "list all meal receipts without cache" -> resolved_message: "list all meal receipts"
   Example: "show their details" (after asking about invoices) -> resolved_message: "show invoice details"

2. CACHE ANALYSIS:
   - is_dissatisfied: Is user unhappy? ("wrong", "check again", "are you sure")
   - bypass_cache: Should skip cache? (dissatisfied OR requesting fresh/latest data)
   - is_cacheable: Worth caching? (Yes for factual queries, No for greetings/meta)

3. INTENT CLASSIFICATION:
   - DATA_ANALYTICS: counts, totals, averages, comparisons, listing details of invoices/receipts/products (requires SQL)
   - DOCUMENT_SEARCH: find/read policy docs, how-to guides, manuals
   - HYBRID: needs both analytics and document content
   - GENERAL: greetings only

4. QUERY ENHANCEMENT FOR DOCUMENT ROUTING:
   - entities: ONLY proper nouns (company names, person names, product names). Example: ["Augment Code", "John Smith"]
   - topics: Subject categories for fuzzy matching. Example: ["meal", "food", "restaurant", "dining", "receipt"]
   - document_type_hints: Likely document types from: {schemas_str}
   - enhanced_query: Optimized query for vector search (detailed, includes synonyms)

CRITICAL RULES:
- entities = PROPER NOUNS ONLY (names you could look up). NOT categories like "meal", "travel", "invoice"
- topics = CATEGORIES/SUBJECT AREAS. Include related terms: for "meal" add ["food", "restaurant", "dining"]
- Keep multi-word names as single entity: "Augment Code" NOT ["Augment", "Code"]
- resolved_message must NOT contain cache control words

RESPOND WITH JSON ONLY (no markdown):
{{"resolved_message":"list all meal receipts","is_dissatisfied":false,"bypass_cache":true,"is_cacheable":true,"intent":"DATA_ANALYTICS","entities":[],"topics":["meal","food","restaurant","dining","receipt"],"document_type_hints":["receipt"],"enhanced_query":"list all meal receipts food dining restaurant expenses"}}"""

    def _format_chat_history(self, chat_history: Optional[List[Dict[str, str]]]) -> str:
        """Format chat history for the prompt."""
        if not chat_history:
            return "(No chat history)"

        formatted = []
        for msg in chat_history[-6:]:  # Last 3 exchanges
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content.strip():
                continue

            if role == "user":
                formatted.append(f"user: {content[:400]}")
            elif role == "assistant":
                # Truncate assistant messages heavily
                formatted.append(f"assistant: {content[:100]}...")

        return "\n".join(formatted)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*\n?', '', text)
            text = re.sub(r'\n?```\s*$', '', text)

        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return match.group(0)

        return text

    def _parse_llm_result(self, result: Dict[str, Any], original_message: str) -> QueryAnalysisResult:
        """Parse LLM result dict into QueryAnalysisResult."""

        resolved_message_raw = result.get("resolved_message", original_message)

        # Always apply strip_cache_control_words to resolved_message as safety fallback
        # This ensures cache hint phrases like "latest data", "fresh data", "no cache"
        # are removed from the query before it's passed to SQL generation or other agents
        resolved_message = strip_cache_control_words(resolved_message_raw)
        clean_cache_key = resolved_message  # Cache key is same as cleaned resolved message

        # Parse intent
        intent_str = result.get("intent", "document_search").lower()
        try:
            intent = QueryIntent(intent_str)
        except ValueError:
            intent = QueryIntent.DOCUMENT_SEARCH

        # Extract time range from message
        time_range = self._extract_time_range(original_message.lower())

        # Determine if requires extracted data
        requires_extracted_data = intent in [QueryIntent.DATA_ANALYTICS, QueryIntent.HYBRID]

        return QueryAnalysisResult(
            original_message=original_message,
            resolved_message=resolved_message,
            cache_key=clean_cache_key,
            is_cacheable=result.get("is_cacheable", True),
            bypass_cache=result.get("bypass_cache", False),
            invalidate_previous=result.get("invalidate_previous", False),
            is_dissatisfied=result.get("is_dissatisfied", False),
            dissatisfaction_type=DissatisfactionType.REFRESH_REQUEST if result.get("is_dissatisfied") else DissatisfactionType.NONE,
            intent=intent,
            intent_confidence=0.85,
            requires_extracted_data=requires_extracted_data,
            enhanced_query=result.get("enhanced_query", resolved_message),
            entities=result.get("entities", []),
            topics=result.get("topics", []),
            document_type_hints=result.get("document_type_hints", []),
            time_range=time_range,
            analysis_method="unified_llm"
        )

    def _heuristic_analyze(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]]],
        previous_response: Optional[str],
        available_schemas: Optional[List[str]]
    ) -> QueryAnalysisResult:
        """Fallback heuristic-based analysis."""

        message_lower = message.lower().strip()

        # Detect dissatisfaction
        is_dissatisfied = self._detect_dissatisfaction_heuristic(message_lower)
        bypass_cache = is_dissatisfied

        # Detect topics
        topics = self._detect_topics_heuristic(message_lower)

        # Detect document types
        doc_types = self._detect_document_types_heuristic(message_lower)

        # Classify intent
        intent = self._classify_intent_heuristic(message_lower)

        # Clean resolved message and cache key (remove cache control words)
        resolved_message = strip_cache_control_words(message)
        clean_cache_key = resolved_message  # Cache key is same as cleaned resolved message

        # Extract time range
        time_range = self._extract_time_range(message_lower)

        return QueryAnalysisResult(
            original_message=message,
            resolved_message=resolved_message,
            cache_key=clean_cache_key,
            is_cacheable=not self._is_greeting_heuristic(message_lower),
            bypass_cache=bypass_cache,
            invalidate_previous=False,
            is_dissatisfied=is_dissatisfied,
            dissatisfaction_type=DissatisfactionType.REFRESH_REQUEST if is_dissatisfied else DissatisfactionType.NONE,
            intent=intent,
            intent_confidence=0.7,
            requires_extracted_data=intent in [QueryIntent.DATA_ANALYTICS, QueryIntent.HYBRID],
            enhanced_query=resolved_message,
            entities=[],
            topics=topics,
            document_type_hints=doc_types,
            time_range=time_range,
            analysis_method="heuristic"
        )

    def _detect_dissatisfaction_heuristic(self, message_lower: str) -> bool:
        """Detect dissatisfaction signals."""
        patterns = [
            r"\b(that's wrong|not correct|incorrect)\b",
            r"\b(check again|try again|refresh)\b",
            r"\b(are you sure|doesn't look right)\b",
            r"\b(latest data|fresh data|current data|most recent)\b",
            r"\bwithout\s+cache\b",
            r"\bno\s+cache\b",
        ]
        return any(re.search(p, message_lower) for p in patterns)

    def _detect_topics_heuristic(self, message_lower: str) -> List[str]:
        """Detect topics using keyword matching."""
        topics = []
        topic_keywords = {
            "invoice": ["invoice", "bill", "billing"],
            "receipt": ["receipt", "meal", "food", "restaurant", "dining", "grocery"],
            "expense": ["expense", "cost", "spending", "travel"],
            "shipping": ["shipping", "delivery", "shipment"],
            "order": ["order", "purchase order", "po"],
            "inventory": ["inventory", "stock", "warehouse"],
            "bank": ["bank", "statement", "transaction", "account"],
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in message_lower for kw in keywords):
                topics.append(topic)
                # Add related topics
                if topic == "receipt":
                    topics.extend(["meal", "food", "restaurant", "dining"])
                elif topic == "expense":
                    topics.extend(["travel", "reimbursement"])

        return list(set(topics))[:10]

    def _detect_document_types_heuristic(self, message_lower: str) -> List[str]:
        """Detect likely document types."""
        doc_types = []
        type_keywords = {
            "invoice": ["invoice", "invoices", "bill"],
            "receipt": ["receipt", "receipts", "meal", "food", "restaurant"],
            "expense_report": ["expense", "travel expense"],
            "bank_statement": ["bank", "statement"],
            "purchase_order": ["purchase order", "po"],
            "shipping_manifest": ["shipping", "shipment"],
            "inventory_report": ["inventory", "stock"],
        }

        for doc_type, keywords in type_keywords.items():
            if any(kw in message_lower for kw in keywords):
                doc_types.append(doc_type)

        return doc_types[:3]

    def _classify_intent_heuristic(self, message_lower: str) -> QueryIntent:
        """Classify intent using pattern matching."""

        analytics_patterns = [
            r'\b(total|sum|average|count|max|min)\b',
            r'\b(how many|how much)\b',
            r'\b(compare|comparison)\b',
            r'\b(list all|show all|get all)\b',
        ]

        search_patterns = [
            r'\b(find|search|look for)\b',
            r'\b(how to|what is|tell me about)\b',
            r'\b(policy|procedure|guide)\b',
        ]

        analytics_score = sum(1 for p in analytics_patterns if re.search(p, message_lower))
        search_score = sum(1 for p in search_patterns if re.search(p, message_lower))

        if analytics_score > search_score:
            return QueryIntent.DATA_ANALYTICS
        elif search_score > 0:
            return QueryIntent.DOCUMENT_SEARCH
        else:
            return QueryIntent.DOCUMENT_SEARCH

    def _is_greeting_heuristic(self, message_lower: str) -> bool:
        """Check if message is a greeting."""
        greetings = [r"^(hello|hi|hey|thanks|thank you|bye)(\s|$)"]
        return any(re.search(p, message_lower) for p in greetings)

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

    def _log_result(self, result: QueryAnalysisResult):
        """Log the analysis result."""
        logger.info("-" * 80)
        logger.info(f"[UnifiedQueryAnalyzer] RESULT ({result.analysis_method}):")
        logger.info(f"[UnifiedQueryAnalyzer] Processing time: {result.processing_time_ms:.2f}ms")
        logger.info(f"[UnifiedQueryAnalyzer] --- MESSAGES ---")
        logger.info(f"[UnifiedQueryAnalyzer]   Original: {result.original_message[:60]}...")
        logger.info(f"[UnifiedQueryAnalyzer]   Resolved: {result.resolved_message[:60]}...")
        logger.info(f"[UnifiedQueryAnalyzer]   Cache key: {result.cache_key[:60]}...")
        logger.info(f"[UnifiedQueryAnalyzer] --- CACHE ---")
        logger.info(f"[UnifiedQueryAnalyzer]   Cacheable: {result.is_cacheable}")
        logger.info(f"[UnifiedQueryAnalyzer]   Bypass: {result.bypass_cache}")
        logger.info(f"[UnifiedQueryAnalyzer]   Dissatisfied: {result.is_dissatisfied}")
        logger.info(f"[UnifiedQueryAnalyzer] --- INTENT ---")
        logger.info(f"[UnifiedQueryAnalyzer]   Intent: {result.intent.value}")
        logger.info(f"[UnifiedQueryAnalyzer]   Confidence: {result.intent_confidence:.2f}")
        logger.info(f"[UnifiedQueryAnalyzer]   Requires extracted data: {result.requires_extracted_data}")
        logger.info(f"[UnifiedQueryAnalyzer] --- ROUTING ---")
        logger.info(f"[UnifiedQueryAnalyzer]   Entities: {result.entities}")
        logger.info(f"[UnifiedQueryAnalyzer]   Topics: {result.topics}")
        logger.info(f"[UnifiedQueryAnalyzer]   Doc types: {result.document_type_hints}")
        logger.info(f"[UnifiedQueryAnalyzer]   Time range: {result.time_range}")
        logger.info("=" * 80)
        logger.info("[UnifiedQueryAnalyzer] ========== UNIFIED ANALYSIS END ==========")
        logger.info("=" * 80)


# Singleton instance
_analyzer: Optional[UnifiedQueryAnalyzer] = None


def get_unified_query_analyzer() -> UnifiedQueryAnalyzer:
    """Get or create the unified query analyzer singleton."""
    global _analyzer
    if _analyzer is None:
        _analyzer = UnifiedQueryAnalyzer()
    return _analyzer


def analyze_query(
    message: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    previous_response: Optional[str] = None,
    available_schemas: Optional[List[str]] = None
) -> QueryAnalysisResult:
    """
    Convenience function to analyze a query.

    Args:
        message: Current user message
        chat_history: List of previous messages
        previous_response: The previous system response
        available_schemas: List of available schema types

    Returns:
        QueryAnalysisResult with all analysis results
    """
    analyzer = get_unified_query_analyzer()
    return analyzer.analyze(message, chat_history, previous_response, available_schemas)
