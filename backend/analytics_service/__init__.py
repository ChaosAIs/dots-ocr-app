"""
Analytics Service Module

Provides conversational analytics capabilities including:
- Redis-backed session management
- Intent classification
- Query planning and execution
- Structured data extraction integration
- Semantic query caching with Qdrant
"""

from .redis_session_manager import RedisSessionManager
from .analytics_session_manager import AnalyticsSessionManager
from .intent_classifier import IntentClassifier, QueryIntent

# Query Cache exports
from .query_cache_config import QueryCacheConfig, get_query_cache_config
from .query_cache_analyzer import (
    QueryCacheAnalyzer,
    UnifiedCacheAnalysis,
    DissatisfactionType,
    get_query_cache_analyzer
)
from .query_cache_manager import (
    QueryCacheManager,
    CacheEntry,
    CacheSearchResult,
    get_query_cache_manager
)
from .query_cache_service import (
    QueryCacheService,
    get_query_cache_service,
    analyze_question_for_cache,
    lookup_cached_answer,
    store_answer_in_cache
)

__all__ = [
    # Session management
    'RedisSessionManager',
    'AnalyticsSessionManager',

    # Intent classification
    'IntentClassifier',
    'QueryIntent',

    # Query cache
    'QueryCacheConfig',
    'get_query_cache_config',
    'QueryCacheAnalyzer',
    'UnifiedCacheAnalysis',
    'DissatisfactionType',
    'get_query_cache_analyzer',
    'QueryCacheManager',
    'CacheEntry',
    'CacheSearchResult',
    'get_query_cache_manager',
    'QueryCacheService',
    'get_query_cache_service',
    'analyze_question_for_cache',
    'lookup_cached_answer',
    'store_answer_in_cache',
]
