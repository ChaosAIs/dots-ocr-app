"""
Query Cache Configuration

Configuration settings for the semantic query cache system.
Provides environment-based configuration with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class QueryCacheConfig:
    """Configuration for query cache behavior."""

    # Feature flags
    cache_enabled: bool = field(
        default_factory=lambda: os.getenv("QUERY_CACHE_ENABLED", "true").lower() == "true"
    )
    pre_cache_analysis_enabled: bool = field(
        default_factory=lambda: os.getenv("QUERY_CACHE_PRE_ANALYSIS_ENABLED", "true").lower() == "true"
    )

    # Similarity thresholds by query type
    similarity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "document_search": float(os.getenv("QUERY_CACHE_THRESHOLD_DOC_SEARCH", "0.92")),
        "data_analytics": float(os.getenv("QUERY_CACHE_THRESHOLD_DATA_ANALYTICS", "0.90")),
        "hybrid": float(os.getenv("QUERY_CACHE_THRESHOLD_HYBRID", "0.88")),
        "general": float(os.getenv("QUERY_CACHE_THRESHOLD_GENERAL", "0.85")),
        "default": float(os.getenv("QUERY_CACHE_THRESHOLD_DEFAULT", "0.90")),
    })

    # TTL settings by query type (in seconds)
    ttl_settings: Dict[str, int] = field(default_factory=lambda: {
        "document_search": int(os.getenv("QUERY_CACHE_TTL_DOC_SEARCH", "604800")),      # 7 days
        "data_analytics": int(os.getenv("QUERY_CACHE_TTL_DATA_ANALYTICS", "3600")),     # 1 hour
        "hybrid": int(os.getenv("QUERY_CACHE_TTL_HYBRID", "86400")),                     # 24 hours
        "general": int(os.getenv("QUERY_CACHE_TTL_GENERAL", "2592000")),                 # 30 days
        "default": int(os.getenv("QUERY_CACHE_TTL_DEFAULT", "86400")),                   # 24 hours
    })

    # Cache size limits
    max_cache_entries_per_workspace: int = field(
        default_factory=lambda: int(os.getenv("QUERY_CACHE_MAX_ENTRIES", "10000"))
    )
    max_answer_length: int = field(
        default_factory=lambda: int(os.getenv("QUERY_CACHE_MAX_ANSWER_LENGTH", "50000"))
    )

    # Search settings
    search_limit: int = field(
        default_factory=lambda: int(os.getenv("QUERY_CACHE_SEARCH_LIMIT", "5"))
    )

    # Cleanup settings
    cleanup_batch_size: int = field(
        default_factory=lambda: int(os.getenv("QUERY_CACHE_CLEANUP_BATCH_SIZE", "1000"))
    )
    cleanup_interval_hours: int = field(
        default_factory=lambda: int(os.getenv("QUERY_CACHE_CLEANUP_INTERVAL_HOURS", "6"))
    )

    # Confidence scoring
    initial_confidence_score: float = field(
        default_factory=lambda: float(os.getenv("QUERY_CACHE_INITIAL_CONFIDENCE", "0.95"))
    )
    min_confidence_score: float = field(
        default_factory=lambda: float(os.getenv("QUERY_CACHE_MIN_CONFIDENCE", "0.5"))
    )
    confidence_decay_on_negative: float = field(
        default_factory=lambda: float(os.getenv("QUERY_CACHE_CONFIDENCE_DECAY", "0.25"))
    )
    max_negative_feedback_count: int = field(
        default_factory=lambda: int(os.getenv("QUERY_CACHE_MAX_NEGATIVE_FEEDBACK", "3"))
    )

    # Response quality evaluation (before caching)
    response_evaluation_enabled: bool = field(
        default_factory=lambda: os.getenv("QUERY_CACHE_RESPONSE_EVALUATION_ENABLED", "true").lower() == "true"
    )
    min_response_length: int = field(
        default_factory=lambda: int(os.getenv("QUERY_CACHE_MIN_RESPONSE_LENGTH", "50"))
    )

    # Qdrant settings
    collection_prefix: str = field(
        default_factory=lambda: os.getenv("QUERY_CACHE_COLLECTION_PREFIX", "query_cache_")
    )
    embedding_dimension: int = field(
        default_factory=lambda: int(os.getenv("QUERY_CACHE_EMBEDDING_DIM", "2560"))
    )

    def get_similarity_threshold(self, query_type: str) -> float:
        """Get similarity threshold for a specific query type."""
        return self.similarity_thresholds.get(
            query_type,
            self.similarity_thresholds.get("default", 0.90)
        )

    def get_ttl(self, query_type: str) -> int:
        """Get TTL for a specific query type."""
        return self.ttl_settings.get(
            query_type,
            self.ttl_settings.get("default", 86400)
        )


# Singleton instance
_config: QueryCacheConfig = None


def get_query_cache_config() -> QueryCacheConfig:
    """Get or create the query cache configuration singleton."""
    global _config
    if _config is None:
        _config = QueryCacheConfig()
    return _config
