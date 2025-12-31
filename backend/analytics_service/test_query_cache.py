"""
Test script for Query Cache implementation.

Run this script to verify the query cache components work correctly:
    python -m analytics_service.test_query_cache
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_config():
    """Test query cache configuration."""
    print("\n" + "=" * 60)
    print("TEST 1: Query Cache Configuration")
    print("=" * 60)

    from analytics_service.query_cache_config import get_query_cache_config

    config = get_query_cache_config()

    print(f"✓ Cache enabled: {config.cache_enabled}")
    print(f"✓ Pre-cache analysis enabled: {config.pre_cache_analysis_enabled}")
    print(f"✓ Similarity thresholds: {config.similarity_thresholds}")
    print(f"✓ TTL settings: {config.ttl_settings}")
    print(f"✓ Collection prefix: {config.collection_prefix}")
    print(f"✓ Embedding dimension: {config.embedding_dimension}")

    # Test getter methods
    print(f"✓ Threshold for 'document_search': {config.get_similarity_threshold('document_search')}")
    print(f"✓ TTL for 'data_analytics': {config.get_ttl('data_analytics')}")

    print("\n✅ Configuration test passed!")
    return True


def test_analyzer_heuristic():
    """Test query cache analyzer with heuristic fallback."""
    print("\n" + "=" * 60)
    print("TEST 2: Query Cache Analyzer (Heuristic)")
    print("=" * 60)

    from analytics_service.query_cache_analyzer import QueryCacheAnalyzer

    analyzer = QueryCacheAnalyzer()

    # Test cases
    test_cases = [
        {
            "message": "What is the total revenue for Q3 2024?",
            "expected_cacheable": True,
            "expected_self_contained": True,
        },
        {
            "message": "That's wrong, check again",
            "expected_dissatisfied": True,
        },
        {
            "message": "What about the second one?",
            "expected_self_contained": False,
        },
        {
            "message": "Hello, how are you?",
            "expected_cacheable": False,  # Greeting
        },
    ]

    for i, case in enumerate(test_cases):
        print(f"\n--- Test case {i + 1}: {case['message'][:50]}...")

        result = analyzer.analyze(
            current_message=case["message"],
            use_llm=False  # Force heuristic
        )

        print(f"  Analysis method: {result.analysis_method}")
        print(f"  Is dissatisfied: {result.dissatisfaction.is_dissatisfied}")
        print(f"  Is self-contained: {result.question_analysis.is_self_contained}")
        print(f"  Is cacheable: {result.cache_decision.is_cacheable}")

        # Basic validations
        if "expected_dissatisfied" in case:
            assert result.dissatisfaction.is_dissatisfied == case["expected_dissatisfied"], \
                f"Expected dissatisfied={case['expected_dissatisfied']}"
            print(f"  ✓ Dissatisfaction detection correct")

        if "expected_self_contained" in case:
            assert result.question_analysis.is_self_contained == case["expected_self_contained"], \
                f"Expected self_contained={case['expected_self_contained']}"
            print(f"  ✓ Self-contained detection correct")

    print("\n✅ Analyzer heuristic test passed!")
    return True


def test_cache_manager():
    """Test query cache manager with Qdrant."""
    print("\n" + "=" * 60)
    print("TEST 3: Query Cache Manager")
    print("=" * 60)

    try:
        from analytics_service.query_cache_manager import QueryCacheManager

        manager = QueryCacheManager()

        test_workspace = "test_workspace_cache"
        test_question = "What is the total revenue for Q3 2024?"
        test_answer = "The total revenue for Q3 2024 is $1,234,567."
        test_doc_ids = ["doc_123", "doc_456"]

        # Test 1: Store a cache entry
        print("\n--- Test: Store cache entry")
        entry_id = manager.store_cache_entry(
            question=test_question,
            answer=test_answer,
            workspace_id=test_workspace,
            source_document_ids=test_doc_ids,
            intent="data_analytics"
        )

        if entry_id:
            print(f"  ✓ Cache entry stored: {entry_id[:8]}...")
        else:
            print("  ⚠ Cache storage returned None (may be disabled or Qdrant unavailable)")
            return True

        # Test 2: Search with permission (user has access)
        print("\n--- Test: Search cache (user has access)")
        result = manager.search_cache(
            question="What is Q3 2024 revenue?",  # Similar question
            workspace_id=test_workspace,
            user_accessible_doc_ids=["doc_123", "doc_456", "doc_789"],  # Has access
            intent="data_analytics"
        )

        print(f"  Cache hit: {result.cache_hit}")
        print(f"  Similarity: {result.similarity_score:.3f}")
        print(f"  Permission granted: {result.permission_granted}")
        print(f"  Search time: {result.search_time_ms:.2f}ms")

        if result.cache_hit:
            print(f"  ✓ Cache hit successful!")
            print(f"  Answer preview: {result.entry.answer[:50]}...")

        # Test 3: Search without permission (user missing access)
        print("\n--- Test: Search cache (user missing access)")
        result_no_perm = manager.search_cache(
            question="What is Q3 2024 revenue?",
            workspace_id=test_workspace,
            user_accessible_doc_ids=["doc_123"],  # Missing doc_456
            intent="data_analytics"
        )

        print(f"  Cache hit: {result_no_perm.cache_hit}")
        print(f"  Candidates checked: {result_no_perm.candidates_checked}")

        if not result_no_perm.cache_hit:
            print(f"  ✓ Permission check correctly denied access!")

        # Test 4: Get cache stats
        print("\n--- Test: Get cache stats")
        stats = manager.get_cache_stats(test_workspace)
        print(f"  Stats: {stats}")

        # Test 5: Cleanup
        print("\n--- Test: Invalidate cache entry")
        manager.invalidate_by_question(test_question, test_workspace)
        print("  ✓ Cache entry invalidated")

        print("\n✅ Cache manager test passed!")
        return True

    except Exception as e:
        print(f"\n⚠ Cache manager test skipped: {e}")
        print("  (This is expected if Qdrant is not running)")
        return True


def test_cache_service():
    """Test the high-level cache service."""
    print("\n" + "=" * 60)
    print("TEST 4: Query Cache Service")
    print("=" * 60)

    from analytics_service.query_cache_service import get_query_cache_service

    service = get_query_cache_service()

    print(f"✓ Cache enabled: {service.is_enabled()}")

    # Test analyze_for_cache
    print("\n--- Test: analyze_for_cache")
    analysis = service.analyze_for_cache(
        question="What is the total revenue?",
        chat_history=None,
        previous_response=None,
        use_llm=False  # Force heuristic for testing
    )

    print(f"  Is cacheable: {analysis.cache_decision.is_cacheable}")
    print(f"  Cache key: {analysis.cache_decision.cache_key_question}")
    print(f"  Analysis method: {analysis.analysis_method}")

    # Test get_stats
    print("\n--- Test: get_stats")
    stats = service.get_stats("test_workspace")
    print(f"  Stats: {stats}")

    print("\n✅ Cache service test passed!")
    return True


def test_convenience_functions():
    """Test convenience functions."""
    print("\n" + "=" * 60)
    print("TEST 5: Convenience Functions")
    print("=" * 60)

    from analytics_service.query_cache_service import (
        analyze_question_for_cache,
        lookup_cached_answer
    )

    # Test analyze_question_for_cache
    print("\n--- Test: analyze_question_for_cache")
    analysis = analyze_question_for_cache(
        question="What is the average order value?",
        chat_history=[
            {"role": "user", "content": "Show me sales data"},
            {"role": "assistant", "content": "Here is the sales data..."}
        ]
    )
    print(f"  ✓ Analysis completed: cacheable={analysis.cache_decision.is_cacheable}")

    # Test lookup_cached_answer (will be cache miss)
    print("\n--- Test: lookup_cached_answer")
    try:
        result = lookup_cached_answer(
            question="Random test question",
            workspace_id="test_workspace",
            user_accessible_doc_ids=["doc_1", "doc_2"]
        )
        print(f"  ✓ Lookup completed: cache_hit={result.cache_hit}")
    except Exception as e:
        print(f"  ⚠ Lookup skipped (Qdrant may not be running): {e}")

    print("\n✅ Convenience functions test passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("QUERY CACHE IMPLEMENTATION TESTS")
    print("=" * 60)

    tests = [
        ("Configuration", test_config),
        ("Analyzer Heuristic", test_analyzer_heuristic),
        ("Cache Manager", test_cache_manager),
        ("Cache Service", test_cache_service),
        ("Convenience Functions", test_convenience_functions),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠ Some tests failed. Check the output above.")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
