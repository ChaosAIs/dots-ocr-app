"""
Query Cache Analyzer - Unified Pre-Cache Analysis Service

This module implements a single LLM call that performs ALL pre-cache analysis:
- Dissatisfaction detection
- Question analysis (self-contained vs context-dependent)
- Question enhancement (if needed)
- Cache worthiness determination

This optimized approach reduces latency from 3 LLM calls to 1, saving 400-800ms
and 66% token cost.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class DissatisfactionType(str, Enum):
    """Types of user dissatisfaction signals."""
    NONE = "none"
    INCORRECT = "incorrect"
    UNCLEAR = "unclear"
    REFRESH_REQUEST = "refresh_request"
    VERIFICATION = "verification"


@dataclass
class DissatisfactionAnalysis:
    """Analysis of user dissatisfaction."""
    is_dissatisfied: bool = False
    type: DissatisfactionType = DissatisfactionType.NONE
    should_bypass_cache: bool = False
    should_invalidate_previous_cache: bool = False


@dataclass
class QuestionAnalysis:
    """Analysis of question characteristics."""
    is_self_contained: bool = True
    has_unresolved_references: bool = False
    reference_types: List[str] = field(default_factory=list)
    can_be_enhanced: bool = False


@dataclass
class QuestionEnhancement:
    """Enhanced question details."""
    enhanced_question: Optional[str] = None
    context_used: List[str] = field(default_factory=list)


@dataclass
class CacheDecision:
    """Cache worthiness decision."""
    is_cacheable: bool = True
    reason: str = ""
    cache_key_question: Optional[str] = None


@dataclass
class UnifiedCacheAnalysis:
    """
    Complete unified analysis result from single LLM call.

    Contains all decisions needed for cache routing:
    - Whether user is dissatisfied with previous response
    - Whether question is self-contained or needs enhancement
    - The enhanced question (if applicable)
    - Whether the question should be cached
    """
    dissatisfaction: DissatisfactionAnalysis = field(default_factory=DissatisfactionAnalysis)
    question_analysis: QuestionAnalysis = field(default_factory=QuestionAnalysis)
    enhancement: QuestionEnhancement = field(default_factory=QuestionEnhancement)
    cache_decision: CacheDecision = field(default_factory=CacheDecision)

    # Metadata
    original_question: str = ""
    analysis_method: str = "llm"  # "llm" or "heuristic"


class QueryCacheAnalyzer:
    """
    Pre-cache analyzer for determining cache behavior.

    Performs analysis for:
    1. Dissatisfaction detection
    2. Question self-containment check
    3. Question enhancement (if needed)
    4. Cache worthiness decision

    Can accept pre-computed results from UnifiedQueryPreprocessor
    to avoid duplicate LLM calls.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the analyzer.

        Args:
            llm_client: Optional LLM client with generate() method
        """
        self.llm_client = llm_client
        self._cached_llm_client = None

    def from_unified_result(self, unified_result) -> "UnifiedCacheAnalysis":
        """
        Convert UnifiedPreprocessResult to UnifiedCacheAnalysis.

        This allows reusing pre-computed results from the unified preprocessor
        instead of making a separate LLM call.

        Args:
            unified_result: UnifiedPreprocessResult from unified preprocessor

        Returns:
            UnifiedCacheAnalysis with cache decisions
        """
        cache = unified_result.cache
        context = unified_result.context

        return UnifiedCacheAnalysis(
            dissatisfaction=DissatisfactionAnalysis(
                is_dissatisfied=cache.is_dissatisfied,
                type=DissatisfactionType(cache.dissatisfaction_type.value) if hasattr(cache.dissatisfaction_type, 'value') else DissatisfactionType.NONE,
                should_bypass_cache=cache.should_bypass_cache,
                should_invalidate_previous_cache=cache.should_invalidate_previous
            ),
            question_analysis=QuestionAnalysis(
                is_self_contained=cache.is_self_contained,
                has_unresolved_references=cache.has_unresolved_references,
                reference_types=[],
                can_be_enhanced=context.has_pronouns
            ),
            enhancement=QuestionEnhancement(
                enhanced_question=context.resolved_message if context.has_pronouns else None,
                context_used=[]
            ),
            cache_decision=CacheDecision(
                is_cacheable=cache.is_cacheable,
                reason=cache.cache_reason,
                cache_key_question=cache.cache_key_question or unified_result.original_message
            ),
            original_question=unified_result.original_message,
            analysis_method="unified_preprocessor"
        )

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
                logger.warning("[CacheAnalyzer] LLM service not available")
                return None

            # Use a fast model for analysis
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
            logger.info("[CacheAnalyzer] LLM client initialized")
            return self._cached_llm_client

        except Exception as e:
            logger.warning(f"[CacheAnalyzer] Failed to create LLM client: {e}")
            return None

    def analyze(
        self,
        current_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        previous_response: Optional[str] = None,
        use_llm: Optional[bool] = None
    ) -> UnifiedCacheAnalysis:
        """
        Perform unified pre-cache analysis.

        This single call determines:
        - Is user dissatisfied with previous response?
        - Is the question self-contained or context-dependent?
        - Can it be enhanced to self-contained?
        - What is the enhanced question?
        - Is the question worth caching?

        Args:
            current_message: The current user message
            chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            previous_response: The previous system response (for dissatisfaction check)
            use_llm: Override default LLM usage

        Returns:
            UnifiedCacheAnalysis with all decisions
        """
        logger.info("=" * 80)
        logger.info("[CacheAnalyzer] ========== UNIFIED PRE-CACHE ANALYSIS ==========")
        logger.info("=" * 80)
        logger.info(f"[CacheAnalyzer] Current message: {current_message[:100]}...")
        logger.info(f"[CacheAnalyzer] Chat history length: {len(chat_history) if chat_history else 0}")
        logger.info(f"[CacheAnalyzer] Has previous response: {previous_response is not None}")

        should_use_llm = use_llm if use_llm is not None else True
        llm_client = self._get_llm_client() if should_use_llm else None

        if llm_client:
            try:
                result = self._llm_analyze(
                    current_message,
                    chat_history,
                    previous_response,
                    llm_client
                )
                self._log_analysis_result(result, "LLM")
                return result
            except Exception as e:
                logger.warning(f"[CacheAnalyzer] LLM analysis failed: {e}")
                logger.info("[CacheAnalyzer] Falling back to heuristic analysis...")

        # Fallback to heuristics
        result = self._heuristic_analyze(current_message, chat_history, previous_response)
        self._log_analysis_result(result, "Heuristic")
        return result

    def _llm_analyze(
        self,
        current_message: str,
        chat_history: Optional[List[Dict[str, str]]],
        previous_response: Optional[str],
        llm_client
    ) -> UnifiedCacheAnalysis:
        """Perform LLM-based unified analysis."""

        # Format chat history
        history_str = ""
        if chat_history:
            history_entries = []
            for msg in chat_history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")[:200]  # Truncate long messages
                history_entries.append(f"{role}: {content}")
            history_str = "\n".join(history_entries)
        else:
            history_str = "(No chat history)"

        previous_response_str = previous_response[:500] if previous_response else "(No previous response)"

        prompt = self._build_unified_prompt(current_message, history_str, previous_response_str)

        response = llm_client.generate(prompt)

        # Parse JSON from response
        json_str = self._extract_json(response)
        result_dict = json.loads(json_str)

        return self._parse_llm_result(result_dict, current_message)

    def _build_unified_prompt(
        self,
        current_message: str,
        history_str: str,
        previous_response_str: str
    ) -> str:
        """Build the unified analysis prompt."""
        return f"""You are a query pre-processor for a Q&A caching system. Analyze the user's message and provide ALL of the following assessments in a single response.

## Input Context

Current User Message: "{current_message}"

Chat History (last 5 messages):
{history_str}

Previous System Response: "{previous_response_str}"

## Analysis Tasks

Perform ALL of the following analyses:

### 1. DISSATISFACTION CHECK
Is the user expressing dissatisfaction with the previous response?
- Look for: complaints about correctness, requests to refresh/retry, confusion, negative sentiment
- Signals: "that's wrong", "check again", "not what I asked", "are you sure?", "refresh", "try again"

### 2. QUESTION ANALYSIS
Analyze the current message as a question:
- Is it self-contained (understandable without chat history)?
- Does it contain unresolved references that REQUIRE chat history to understand?

IMPORTANT - The following are NOT unresolved references (treat as SELF-CONTAINED):
- Possessive pronouns in general questions: "my tire", "my order", "my account", "my document"
- These are natural phrasing for policy/procedure questions and do NOT require chat history
- Examples of SELF-CONTAINED questions:
  * "How to return my tire?" → Self-contained (asking about return policy)
  * "What is my order status?" → Self-contained (general inquiry pattern)
  * "How to make sure my tire is able to return?" → Self-contained (policy question)

Unresolved references that make a question NOT self-contained:
- Pronouns referring to specific chat items: "what about it?", "the one I mentioned"
- Relative terms requiring context: "the previous one", "the second item", "the same thing"
- Follow-up fragments: "and the other?", "also that?"

### 3. QUESTION ENHANCEMENT (if applicable)
If the question is NOT self-contained but CAN be made self-contained using chat history:
- Create an enhanced version that replaces all references with actual subjects
- The enhanced question should be understandable in a brand new conversation

### 4. CACHE WORTHINESS
Should this question-answer pair be cached for future reuse?
- Worth caching: policy questions, how-to questions, procedure questions, specific queries, entity lookups
- Questions with "my/your" wording about general topics ARE cacheable (e.g., "how to return my tire?")
- NOT worth caching: meta questions about conversation, greetings, highly temporal ("what time is it")

Respond with ONLY valid JSON (no markdown, no code blocks, no explanation):
{{
  "dissatisfaction": {{
    "is_dissatisfied": true/false,
    "type": "incorrect|unclear|refresh_request|verification|none",
    "should_bypass_cache": true/false,
    "should_invalidate_previous_cache": true/false
  }},
  "question_analysis": {{
    "is_self_contained": true/false,
    "has_unresolved_references": true/false,
    "reference_types": ["pronoun", "relative_term", "implicit_subject"],
    "can_be_enhanced": true/false
  }},
  "enhancement": {{
    "enhanced_question": "the enhanced self-contained question or null",
    "context_used": ["key context item 1", "key context item 2"]
  }},
  "cache_decision": {{
    "is_cacheable": true/false,
    "reason": "brief explanation",
    "cache_key_question": "the question to use for cache operations (enhanced or original)"
  }}
}}"""

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*\n?', '', text)
            text = re.sub(r'\n?```\s*$', '', text)

        # Find JSON object
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return match.group(0)

        return text

    def _parse_llm_result(self, result: Dict[str, Any], original_question: str) -> UnifiedCacheAnalysis:
        """Parse LLM result dict into UnifiedCacheAnalysis."""

        # Parse dissatisfaction
        dissatisfaction_data = result.get("dissatisfaction", {})
        dissatisfaction = DissatisfactionAnalysis(
            is_dissatisfied=dissatisfaction_data.get("is_dissatisfied", False),
            type=DissatisfactionType(dissatisfaction_data.get("type", "none").lower()),
            should_bypass_cache=dissatisfaction_data.get("should_bypass_cache", False),
            should_invalidate_previous_cache=dissatisfaction_data.get("should_invalidate_previous_cache", False)
        )

        # Parse question analysis
        question_data = result.get("question_analysis", {})
        question_analysis = QuestionAnalysis(
            is_self_contained=question_data.get("is_self_contained", True),
            has_unresolved_references=question_data.get("has_unresolved_references", False),
            reference_types=question_data.get("reference_types", []),
            can_be_enhanced=question_data.get("can_be_enhanced", False)
        )

        # Parse enhancement
        enhancement_data = result.get("enhancement", {})
        enhanced_q = enhancement_data.get("enhanced_question")
        enhancement = QuestionEnhancement(
            enhanced_question=enhanced_q if enhanced_q and enhanced_q != "null" else None,
            context_used=enhancement_data.get("context_used", [])
        )

        # Parse cache decision
        cache_data = result.get("cache_decision", {})
        cache_key = cache_data.get("cache_key_question")
        cache_decision = CacheDecision(
            is_cacheable=cache_data.get("is_cacheable", True),
            reason=cache_data.get("reason", ""),
            cache_key_question=cache_key if cache_key and cache_key != "null" else original_question
        )

        return UnifiedCacheAnalysis(
            dissatisfaction=dissatisfaction,
            question_analysis=question_analysis,
            enhancement=enhancement,
            cache_decision=cache_decision,
            original_question=original_question,
            analysis_method="llm"
        )

    def _heuristic_analyze(
        self,
        current_message: str,
        chat_history: Optional[List[Dict[str, str]]],
        previous_response: Optional[str]
    ) -> UnifiedCacheAnalysis:
        """Fallback heuristic-based analysis when LLM is unavailable."""

        message_lower = current_message.lower().strip()

        # 1. Check for dissatisfaction signals
        dissatisfaction = self._detect_dissatisfaction_heuristic(message_lower)

        # 2. Analyze question characteristics
        question_analysis = self._analyze_question_heuristic(message_lower)

        # 3. Enhancement (limited without LLM)
        enhancement = QuestionEnhancement(
            enhanced_question=None,
            context_used=[]
        )

        # 4. Cache worthiness
        is_cacheable = (
            not dissatisfaction.is_dissatisfied and
            question_analysis.is_self_contained and
            not self._is_greeting_or_meta(message_lower)
        )

        cache_decision = CacheDecision(
            is_cacheable=is_cacheable,
            reason="Heuristic analysis" if is_cacheable else "Not cacheable (context-dependent or dissatisfaction)",
            cache_key_question=current_message if is_cacheable else None
        )

        return UnifiedCacheAnalysis(
            dissatisfaction=dissatisfaction,
            question_analysis=question_analysis,
            enhancement=enhancement,
            cache_decision=cache_decision,
            original_question=current_message,
            analysis_method="heuristic"
        )

    def _detect_dissatisfaction_heuristic(self, message_lower: str) -> DissatisfactionAnalysis:
        """Detect user dissatisfaction using patterns."""

        # Dissatisfaction patterns
        incorrect_patterns = [
            r"\b(that's wrong|that is wrong|not correct|incorrect|wrong answer|not right)\b",
            r"\b(no,?\s+that's not|no,?\s+that is not)\b",
        ]
        unclear_patterns = [
            r"\b(don't understand|do not understand|what do you mean|unclear|confusing)\b",
            r"\b(can you clarify|explain that|what does that mean)\b",
        ]
        refresh_patterns = [
            r"\b(refresh|try again|get latest|update|reload)\b",
            r"\b(check again|recheck|re-check)\b",
        ]
        verification_patterns = [
            r"\b(are you sure|double check|verify|is that right)\b",
            r"\b(that seems off|doesn't look right|looks wrong)\b",
        ]

        for pattern in incorrect_patterns:
            if re.search(pattern, message_lower):
                return DissatisfactionAnalysis(
                    is_dissatisfied=True,
                    type=DissatisfactionType.INCORRECT,
                    should_bypass_cache=True,
                    should_invalidate_previous_cache=True
                )

        for pattern in unclear_patterns:
            if re.search(pattern, message_lower):
                return DissatisfactionAnalysis(
                    is_dissatisfied=True,
                    type=DissatisfactionType.UNCLEAR,
                    should_bypass_cache=True,
                    should_invalidate_previous_cache=False
                )

        for pattern in refresh_patterns:
            if re.search(pattern, message_lower):
                return DissatisfactionAnalysis(
                    is_dissatisfied=True,
                    type=DissatisfactionType.REFRESH_REQUEST,
                    should_bypass_cache=True,
                    should_invalidate_previous_cache=False
                )

        for pattern in verification_patterns:
            if re.search(pattern, message_lower):
                return DissatisfactionAnalysis(
                    is_dissatisfied=True,
                    type=DissatisfactionType.VERIFICATION,
                    should_bypass_cache=True,
                    should_invalidate_previous_cache=False
                )

        return DissatisfactionAnalysis()

    def _analyze_question_heuristic(self, message_lower: str) -> QuestionAnalysis:
        """Analyze question characteristics using patterns."""

        reference_types = []

        # Check for pronouns
        pronoun_patterns = [
            r"\b(it|them|they|this|that|these|those)\b",
        ]
        for pattern in pronoun_patterns:
            if re.search(pattern, message_lower):
                # Check if it's a standalone pronoun reference (not part of a larger phrase)
                if re.search(r"^(what about|and|also|more about)\s+" + pattern[2:-2], message_lower):
                    reference_types.append("pronoun")
                    break

        # Check for relative terms
        relative_patterns = [
            r"\b(previous|above|earlier|before|last|same)\b",
            r"\b(the other one|the second one|the first one)\b",
        ]
        for pattern in relative_patterns:
            if re.search(pattern, message_lower):
                reference_types.append("relative_term")
                break

        # Check for implicit subject
        implicit_patterns = [
            r"^(and|also|what about|how about)\s+",
            r"^(more|less|another)\s+",
        ]
        for pattern in implicit_patterns:
            if re.search(pattern, message_lower):
                reference_types.append("implicit_subject")
                break

        has_unresolved = len(reference_types) > 0
        is_self_contained = not has_unresolved

        return QuestionAnalysis(
            is_self_contained=is_self_contained,
            has_unresolved_references=has_unresolved,
            reference_types=reference_types,
            can_be_enhanced=has_unresolved  # Could be enhanced if has references
        )

    def _is_greeting_or_meta(self, message_lower: str) -> bool:
        """Check if message is a greeting or meta question."""
        greeting_patterns = [
            r"^(hello|hi|hey|thanks|thank you|bye|goodbye)\b",
            r"^(how are you|what can you do|help me|who are you)\b",
        ]
        for pattern in greeting_patterns:
            if re.search(pattern, message_lower):
                return True
        return False

    def _log_analysis_result(self, result: UnifiedCacheAnalysis, method: str):
        """Log the analysis result."""
        logger.info("-" * 80)
        logger.info(f"[CacheAnalyzer] UNIFIED ANALYSIS RESULT ({method}):")
        logger.info(f"[CacheAnalyzer]   • Original question: {result.original_question[:80]}...")
        logger.info(f"[CacheAnalyzer]   • Dissatisfaction:")
        logger.info(f"[CacheAnalyzer]       - is_dissatisfied: {result.dissatisfaction.is_dissatisfied}")
        logger.info(f"[CacheAnalyzer]       - type: {result.dissatisfaction.type.value}")
        logger.info(f"[CacheAnalyzer]       - bypass_cache: {result.dissatisfaction.should_bypass_cache}")
        logger.info(f"[CacheAnalyzer]   • Question Analysis:")
        logger.info(f"[CacheAnalyzer]       - is_self_contained: {result.question_analysis.is_self_contained}")
        logger.info(f"[CacheAnalyzer]       - has_references: {result.question_analysis.has_unresolved_references}")
        logger.info(f"[CacheAnalyzer]       - can_be_enhanced: {result.question_analysis.can_be_enhanced}")
        logger.info(f"[CacheAnalyzer]   • Enhancement:")
        logger.info(f"[CacheAnalyzer]       - enhanced_question: {result.enhancement.enhanced_question or 'None'}")
        logger.info(f"[CacheAnalyzer]   • Cache Decision:")
        logger.info(f"[CacheAnalyzer]       - is_cacheable: {result.cache_decision.is_cacheable}")
        logger.info(f"[CacheAnalyzer]       - reason: {result.cache_decision.reason}")
        logger.info(f"[CacheAnalyzer]       - cache_key: {result.cache_decision.cache_key_question[:80] if result.cache_decision.cache_key_question else 'None'}...")
        logger.info("=" * 80)


# Singleton instance
_analyzer: QueryCacheAnalyzer = None


def get_query_cache_analyzer() -> QueryCacheAnalyzer:
    """Get or create the query cache analyzer singleton."""
    global _analyzer
    if _analyzer is None:
        _analyzer = QueryCacheAnalyzer()
    return _analyzer
