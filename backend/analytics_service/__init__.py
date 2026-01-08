"""
Analytics Service Module

Provides conversational analytics capabilities including:
- Redis-backed session management
- Intent classification
- Query planning and execution
- Structured data extraction integration
- Semantic query caching with Qdrant
- LLM-driven SQL generation (V2 optimized)
- Dynamic prompt building
- LLM-driven result formatting
"""

from .redis_session_manager import RedisSessionManager
from .analytics_session_manager import AnalyticsSessionManager
from .intent_classifier import IntentClassifier, QueryIntent

# LLM SQL Generation V2 (Optimized)
from .prompt_builder import PromptBuilder, QueryContext, QueryType, AggregationType
from .llm_sql_generator_v2 import LLMSQLGeneratorV2, SQLGenerationResult, QueryAnalysis
from .llm_result_formatter import LLMResultFormatter, ColumnFormat, FormattingConfig

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

# Unified Query Preprocessor exports
from .unified_query_preprocessor import (
    UnifiedQueryPreprocessor,
    UnifiedPreprocessResult,
    get_unified_preprocessor,
    preprocess_query
)

__all__ = [
    # Session management
    'RedisSessionManager',
    'AnalyticsSessionManager',

    # Intent classification
    'IntentClassifier',
    'QueryIntent',

    # LLM SQL Generation V2 (Optimized)
    'PromptBuilder',
    'QueryContext',
    'QueryType',
    'AggregationType',
    'LLMSQLGeneratorV2',
    'SQLGenerationResult',
    'QueryAnalysis',
    'LLMResultFormatter',
    'ColumnFormat',
    'FormattingConfig',

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

    # Unified Query Preprocessor
    'UnifiedQueryPreprocessor',
    'UnifiedPreprocessResult',
    'get_unified_preprocessor',
    'preprocess_query',
]
