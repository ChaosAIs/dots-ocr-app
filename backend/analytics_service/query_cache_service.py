"""
Query Cache Service - Integration Layer for Chat Pipeline

This service integrates the query cache with the chat orchestration pipeline.
It provides a simple interface for:
1. Pre-cache analysis (single LLM call)
2. Cache lookup with permission validation
3. Background cache storage

Usage in chat pipeline:
    cache_service = get_query_cache_service()

    # Step 1: Analyze the question (single LLM call)
    analysis = cache_service.analyze_for_cache(
        question=user_message,
        chat_history=conversation_history,
        previous_response=last_response
    )

    # Step 2: Check if should bypass cache
    if analysis.dissatisfaction.should_bypass_cache:
        # Fresh query, optionally invalidate previous cache
        if analysis.dissatisfaction.should_invalidate_previous_cache:
            cache_service.invalidate_question(previous_question, workspace_id)
        # ... execute fresh query ...
    elif analysis.cache_decision.is_cacheable:
        # Step 3: Try cache lookup
        cache_key = analysis.cache_decision.cache_key_question
        cache_result = cache_service.lookup(
            question=cache_key,
            workspace_id=workspace_id,
            user_accessible_doc_ids=user_doc_ids,
            intent=intent
        )

        if cache_result.cache_hit:
            return cache_result.entry.answer  # Cache hit!

    # Step 4: Execute fresh query
    answer = ... execute full pipeline ...

    # Step 5: Store in cache (background, non-blocking)
    if analysis.cache_decision.is_cacheable:
        cache_service.store_async(
            question=cache_key,
            answer=answer,
            workspace_id=workspace_id,
            source_document_ids=doc_ids,
            intent=intent
        )
"""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any

from .query_cache_config import QueryCacheConfig, get_query_cache_config
from .query_cache_analyzer import (
    QueryCacheAnalyzer,
    UnifiedCacheAnalysis,
    get_query_cache_analyzer
)
from .query_cache_manager import (
    QueryCacheManager,
    CacheSearchResult,
    CacheEntry,
    get_query_cache_manager
)

logger = logging.getLogger(__name__)

# Thread pool for background cache operations
_executor: Optional[ThreadPoolExecutor] = None


class ResponseQualityEvaluator:
    """
    Evaluates response quality to determine if it's worth caching.

    Uses LLM to analyze whether a response actually answers the question
    or is just a "I don't know" / "no information found" type response.
    """

    def __init__(self):
        self._llm_client = None

    def _get_llm_client(self):
        """Lazily initialize LLM client."""
        if self._llm_client:
            return self._llm_client

        try:
            from rag_service.llm_service import get_llm_service
            from langchain_core.messages import HumanMessage

            llm_service = get_llm_service()
            if not llm_service.is_available():
                logger.warning("[ResponseEvaluator] LLM service not available")
                return None

            # Use a fast model for evaluation
            chat_model = llm_service.get_query_model(
                temperature=0.0,
                num_ctx=4096,
                num_predict=256
            )

            class LLMClientWrapper:
                def __init__(self, model):
                    self.model = model

                def generate(self, prompt: str) -> str:
                    response = self.model.invoke([HumanMessage(content=prompt)])
                    return response.content

            self._llm_client = LLMClientWrapper(chat_model)
            logger.info("[ResponseEvaluator] LLM client initialized")
            return self._llm_client

        except Exception as e:
            logger.warning(f"[ResponseEvaluator] Failed to create LLM client: {e}")
            return None

    def evaluate(self, question: str, answer: str, config: "QueryCacheConfig") -> tuple[bool, str]:
        """
        Evaluate if the response is worth caching.

        Args:
            question: The user's question
            answer: The generated answer
            config: Cache configuration

        Returns:
            Tuple of (is_worth_caching, reason)
        """
        # Quick checks first (no LLM needed)
        if len(answer.strip()) < config.min_response_length:
            return False, f"Response too short ({len(answer.strip())} chars < {config.min_response_length})"

        # Check for common "no answer" patterns
        no_answer_patterns = [
            "does not contain",
            "no information",
            "cannot find",
            "not found",
            "unable to find",
            "don't have",
            "do not have",
            "no relevant",
            "no data",
            "context does not",
            "provided context does not",
            "i couldn't find",
            "i could not find",
            "couldn't find any",
            "could not find any",
            "there is no information",
            "there are no results",
            "no results found",
            "not available",
            "not mentioned",
            "does not mention",
            "doesn't contain",
            "doesn't include",
            "does not include",
            "no documents",
            "no matching",
        ]

        answer_lower = answer.lower()
        for pattern in no_answer_patterns:
            if pattern in answer_lower:
                # Quick rejection based on pattern, but let LLM confirm if enabled
                if config.response_evaluation_enabled:
                    llm_result = self._llm_evaluate(question, answer)
                    if llm_result is not None:
                        return llm_result
                return False, f"Response indicates no information found (pattern: '{pattern}')"

        # If LLM evaluation is enabled, do a more thorough check
        if config.response_evaluation_enabled:
            llm_result = self._llm_evaluate(question, answer)
            if llm_result is not None:
                return llm_result

        # Default: assume response is worth caching
        return True, "Response passed quality checks"

    def _llm_evaluate(self, question: str, answer: str) -> Optional[tuple[bool, str]]:
        """
        Use LLM to evaluate response quality.

        Returns:
            Tuple of (is_worth_caching, reason) or None if LLM unavailable
        """
        llm_client = self._get_llm_client()
        if not llm_client:
            return None

        try:
            prompt = f"""You are evaluating whether an AI assistant's response should be cached for future similar questions.

Question: "{question[:500]}"

Response: "{answer[:1500]}"

Evaluate if this response ACTUALLY ANSWERS the question with useful information.

A response should NOT be cached if:
1. It says information is not found/available/provided
2. It only describes an error or technical issue
3. It asks the user to provide more context/documents
4. It's a generic "I don't know" type response
5. It doesn't provide substantive information related to the question

A response SHOULD be cached if:
1. It provides specific, useful information answering the question
2. It contains facts, procedures, or data relevant to the question
3. Even if partial, it gives meaningful content

Respond with ONLY a JSON object (no markdown, no explanation):
{{"should_cache": true/false, "reason": "brief explanation"}}"""

            response = llm_client.generate(prompt)

            # Parse the response
            import json
            import re

            # Extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```(?:json)?\s*\n?', '', response)
                response = re.sub(r'\n?```\s*$', '', response)

            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                result = json.loads(match.group(0))
                should_cache = result.get("should_cache", True)
                reason = result.get("reason", "LLM evaluation")
                logger.info(f"[ResponseEvaluator] LLM evaluation: should_cache={should_cache}, reason={reason}")
                return should_cache, reason

        except Exception as e:
            logger.warning(f"[ResponseEvaluator] LLM evaluation failed: {e}")

        return None


# Singleton for response evaluator
_response_evaluator: Optional[ResponseQualityEvaluator] = None


def _get_response_evaluator() -> ResponseQualityEvaluator:
    """Get or create the response evaluator singleton."""
    global _response_evaluator
    if _response_evaluator is None:
        _response_evaluator = ResponseQualityEvaluator()
    return _response_evaluator


def _get_executor() -> ThreadPoolExecutor:
    """Get or create the thread pool executor for background tasks."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cache_worker")
    return _executor


class QueryCacheService:
    """
    High-level query cache service for chat pipeline integration.

    Provides a unified interface for:
    - Pre-cache analysis (unified LLM call)
    - Cache lookup with permission validation
    - Non-blocking background cache storage
    - Cache invalidation
    """

    def __init__(
        self,
        analyzer: Optional[QueryCacheAnalyzer] = None,
        cache_manager: Optional[QueryCacheManager] = None,
        config: Optional[QueryCacheConfig] = None
    ):
        """
        Initialize the cache service.

        Args:
            analyzer: Query cache analyzer (optional, will use singleton if not provided)
            cache_manager: Query cache manager (optional, will use singleton if not provided)
            config: Cache configuration
        """
        self._analyzer = analyzer
        self._cache_manager = cache_manager
        self.config = config or get_query_cache_config()

    @property
    def analyzer(self) -> QueryCacheAnalyzer:
        """Get the analyzer instance."""
        if self._analyzer is None:
            self._analyzer = get_query_cache_analyzer()
        return self._analyzer

    @property
    def cache_manager(self) -> QueryCacheManager:
        """Get the cache manager instance."""
        if self._cache_manager is None:
            self._cache_manager = get_query_cache_manager()
        return self._cache_manager

    def is_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.config.cache_enabled

    def analyze_for_cache(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        previous_response: Optional[str] = None,
        use_llm: Optional[bool] = None
    ) -> UnifiedCacheAnalysis:
        """
        Perform unified pre-cache analysis.

        This single LLM call determines:
        - Is user dissatisfied with previous response?
        - Is the question self-contained or context-dependent?
        - Can it be enhanced to self-contained?
        - What is the enhanced question?
        - Is the question worth caching?

        Args:
            question: The current user message
            chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            previous_response: The previous system response
            use_llm: Override default LLM usage

        Returns:
            UnifiedCacheAnalysis with all decisions
        """
        if not self.config.cache_enabled or not self.config.pre_cache_analysis_enabled:
            # Return a simple analysis that allows caching
            from .query_cache_analyzer import (
                DissatisfactionAnalysis,
                QuestionAnalysis,
                QuestionEnhancement,
                CacheDecision
            )
            return UnifiedCacheAnalysis(
                dissatisfaction=DissatisfactionAnalysis(),
                question_analysis=QuestionAnalysis(),
                enhancement=QuestionEnhancement(),
                cache_decision=CacheDecision(
                    is_cacheable=True,
                    reason="Analysis disabled",
                    cache_key_question=question
                ),
                original_question=question,
                analysis_method="disabled"
            )

        return self.analyzer.analyze(
            current_message=question,
            chat_history=chat_history,
            previous_response=previous_response,
            use_llm=use_llm
        )

    def lookup(
        self,
        question: str,
        workspace_id: str,
        user_accessible_doc_ids: List[str],
        intent: Optional[str] = None
    ) -> CacheSearchResult:
        """
        Look up a cached answer with permission validation.

        The search process:
        1. Generate embedding for the question
        2. Find semantically similar cached questions
        3. For each candidate, check if user has access to ALL source documents
        4. Return first candidate that passes permission check

        Args:
            question: The question to search for (use enhanced question from analysis)
            workspace_id: Workspace ID
            user_accessible_doc_ids: List of document IDs the user can access
            intent: Optional intent filter

        Returns:
            CacheSearchResult with cache hit status and entry if found
        """
        if not self.config.cache_enabled:
            return CacheSearchResult(cache_hit=False)

        return self.cache_manager.search_cache(
            question=question,
            workspace_id=workspace_id,
            user_accessible_doc_ids=user_accessible_doc_ids,
            intent=intent
        )

    def store(
        self,
        question: str,
        answer: str,
        workspace_id: str,
        source_document_ids: List[str],
        intent: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Store a new cache entry (blocking).

        Args:
            question: The question (should be enhanced/self-contained)
            answer: The generated answer
            workspace_id: Workspace ID
            source_document_ids: List of document IDs used to generate the answer
            intent: Query intent type
            metadata: Additional metadata to store

        Returns:
            Cache entry ID if successful, None otherwise
        """
        if not self.config.cache_enabled:
            return None

        return self.cache_manager.store_cache_entry(
            question=question,
            answer=answer,
            workspace_id=workspace_id,
            source_document_ids=source_document_ids,
            intent=intent,
            metadata=metadata
        )

    def store_async(
        self,
        question: str,
        answer: str,
        workspace_id: str,
        source_document_ids: List[str],
        intent: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
        skip_quality_check: bool = False
    ) -> None:
        """
        Store a new cache entry in the background (non-blocking).

        This is the preferred method for cache storage as it doesn't
        block the user response.

        Before storing, the response is evaluated to determine if it's worth caching.
        Responses that don't actually answer the question (e.g., "information not found")
        are not cached.

        Args:
            question: The question (should be enhanced/self-contained)
            answer: The generated answer
            workspace_id: Workspace ID
            source_document_ids: List of document IDs used to generate the answer
            intent: Query intent type
            metadata: Additional metadata to store
            skip_quality_check: If True, skip response quality evaluation
        """
        if not self.config.cache_enabled:
            return

        def _store_background():
            try:
                # Evaluate response quality before caching
                if not skip_quality_check:
                    evaluator = _get_response_evaluator()
                    is_worth_caching, reason = evaluator.evaluate(question, answer, self.config)

                    if not is_worth_caching:
                        logger.info(f"[CacheService] Response NOT cached - {reason}")
                        logger.info(f"[CacheService]   Question: {question[:80]}...")
                        logger.info(f"[CacheService]   Answer preview: {answer[:100]}...")
                        return

                    logger.info(f"[CacheService] Response quality check passed: {reason}")

                entry_id = self.cache_manager.store_cache_entry(
                    question=question,
                    answer=answer,
                    workspace_id=workspace_id,
                    source_document_ids=source_document_ids,
                    intent=intent,
                    metadata=metadata
                )
                if entry_id:
                    logger.info(f"[CacheService] Background store completed: {entry_id[:8]}...")
            except Exception as e:
                logger.error(f"[CacheService] Background store failed: {e}")

        # Submit to thread pool
        executor = _get_executor()
        executor.submit(_store_background)
        logger.debug("[CacheService] Cache storage submitted to background worker")

    def invalidate_question(
        self,
        question: str,
        workspace_id: str
    ) -> int:
        """
        Invalidate cache entries matching a question.

        Args:
            question: The question to invalidate
            workspace_id: Workspace ID

        Returns:
            Number of entries invalidated
        """
        if not self.config.cache_enabled:
            return 0

        return self.cache_manager.invalidate_by_question(question, workspace_id)

    def invalidate_document(
        self,
        document_id: str,
        workspace_id: str
    ) -> int:
        """
        Invalidate all cache entries that used a specific document.

        Call this when a document is updated or deleted.

        Args:
            document_id: The document ID
            workspace_id: Workspace ID

        Returns:
            Number of entries invalidated
        """
        if not self.config.cache_enabled:
            return 0

        return self.cache_manager.invalidate_by_document(document_id, workspace_id)

    def record_negative_feedback(
        self,
        entry_id: str,
        workspace_id: str
    ) -> bool:
        """
        Record negative feedback for a cache entry.

        Call this when user expresses dissatisfaction with a cached answer.

        Args:
            entry_id: Cache entry ID
            workspace_id: Workspace ID

        Returns:
            True if successful
        """
        if not self.config.cache_enabled:
            return False

        return self.cache_manager.record_negative_feedback(entry_id, workspace_id)

    def cleanup_expired(self, workspace_id: str) -> int:
        """
        Clean up expired cache entries.

        Args:
            workspace_id: Workspace ID

        Returns:
            Number of entries cleaned up
        """
        if not self.config.cache_enabled:
            return 0

        return self.cache_manager.cleanup_expired(workspace_id)

    def get_stats(self, workspace_id: str) -> Dict[str, Any]:
        """
        Get cache statistics for a workspace.

        Args:
            workspace_id: Workspace ID

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "enabled": self.config.cache_enabled,
            "pre_analysis_enabled": self.config.pre_cache_analysis_enabled,
        }

        if self.config.cache_enabled:
            cache_stats = self.cache_manager.get_cache_stats(workspace_id)
            stats.update(cache_stats)

        return stats


# Singleton instance
_cache_service: Optional[QueryCacheService] = None


def get_query_cache_service() -> QueryCacheService:
    """Get or create the query cache service singleton."""
    global _cache_service
    if _cache_service is None:
        _cache_service = QueryCacheService()
    return _cache_service


# Convenience functions for common operations

def analyze_question_for_cache(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    previous_response: Optional[str] = None
) -> UnifiedCacheAnalysis:
    """
    Convenience function to analyze a question for caching.

    Args:
        question: The current user message
        chat_history: List of previous messages
        previous_response: The previous system response

    Returns:
        UnifiedCacheAnalysis with all decisions
    """
    service = get_query_cache_service()
    return service.analyze_for_cache(question, chat_history, previous_response)


def lookup_cached_answer(
    question: str,
    workspace_id: str,
    user_accessible_doc_ids: List[str],
    intent: Optional[str] = None
) -> CacheSearchResult:
    """
    Convenience function to look up a cached answer.

    Args:
        question: The question to search for
        workspace_id: Workspace ID
        user_accessible_doc_ids: List of document IDs the user can access
        intent: Optional intent filter

    Returns:
        CacheSearchResult with cache hit status and entry if found
    """
    service = get_query_cache_service()
    return service.lookup(question, workspace_id, user_accessible_doc_ids, intent)


def store_answer_in_cache(
    question: str,
    answer: str,
    workspace_id: str,
    source_document_ids: List[str],
    intent: str = "general",
    background: bool = True,
    skip_quality_check: bool = False
) -> Optional[str]:
    """
    Convenience function to store an answer in cache.

    Before storing, the response is evaluated to determine if it's worth caching.
    Responses that don't actually answer the question (e.g., "information not found")
    are not cached unless skip_quality_check=True.

    Args:
        question: The question
        answer: The answer
        workspace_id: Workspace ID
        source_document_ids: List of document IDs used
        intent: Query intent type
        background: If True, store in background (non-blocking)
        skip_quality_check: If True, skip response quality evaluation

    Returns:
        Cache entry ID if blocking, None if background
    """
    service = get_query_cache_service()
    if background:
        service.store_async(question, answer, workspace_id, source_document_ids, intent, skip_quality_check=skip_quality_check)
        return None
    else:
        return service.store(question, answer, workspace_id, source_document_ids, intent)
