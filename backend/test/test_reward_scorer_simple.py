"""
Simple unit tests for RewardScorer (no pytest required).
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_service.graph_rag.reward_scorer import RewardScorer


def test_perfect_format_with_query():
    """Test perfect format with query tag."""
    scorer = RewardScorer()
    response = """<think>
I need to search for information about cloud computing.
</think>
<query>cloud computing definition</query>"""
    
    score, details = scorer.compute_format_reward(response)
    
    assert score == 1.0, f"Expected 1.0, got {score}"
    assert details['has_think'] is True
    assert details['has_action'] is True
    assert details['valid_structure'] is True
    assert details['action_type'] == 'query'
    print("✅ test_perfect_format_with_query PASSED")


def test_perfect_format_with_answer():
    """Test perfect format with answer tag."""
    scorer = RewardScorer()
    response = """<think>
I have enough information to answer.
</think>
<answer>Cloud computing is a model for delivering computing services over the internet.</answer>"""
    
    score, details = scorer.compute_format_reward(response)
    
    assert score == 1.0, f"Expected 1.0, got {score}"
    assert details['has_think'] is True
    assert details['has_action'] is True
    assert details['valid_structure'] is True
    assert details['action_type'] == 'answer'
    print("✅ test_perfect_format_with_answer PASSED")


def test_partial_format_think_only():
    """Test partial format with only think tag."""
    scorer = RewardScorer()
    response = """<think>
I'm thinking about this problem.
</think>"""
    
    score, details = scorer.compute_format_reward(response)
    
    assert score == 0.6, f"Expected 0.6, got {score}"
    assert details['has_think'] is True
    assert details['has_action'] is False
    assert details['valid_structure'] is True
    print("✅ test_partial_format_think_only PASSED")


def test_no_tags():
    """Test response with no tags."""
    scorer = RewardScorer()
    response = "Just plain text without any tags."
    
    score, details = scorer.compute_format_reward(response)
    
    assert score == 0.3, f"Expected 0.3, got {score}"
    assert details['has_think'] is False
    assert details['has_action'] is False
    assert details['valid_structure'] is True
    print("✅ test_no_tags PASSED")


def test_malformed_tags():
    """Test malformed tags (unclosed)."""
    scorer = RewardScorer()
    response = """<think>
I'm thinking but forgot to close the tag
<query>search query</query>"""
    
    score, details = scorer.compute_format_reward(response)
    
    assert score == 0.4, f"Expected 0.4, got {score}"
    assert details['has_think'] is False
    assert details['has_action'] is True
    assert details['valid_structure'] is False
    print("✅ test_malformed_tags PASSED")


def test_good_answer():
    """Test a good quality answer."""
    scorer = RewardScorer()
    question = "What is cloud computing?"
    answer = """Cloud computing is a model for delivering computing services over the internet, 
including servers, storage, databases, networking, and software. Major providers include AWS, 
Azure, and Google Cloud Platform. It provides on-demand access to resources."""
    
    score, details = scorer.compute_answer_reward_heuristic(answer, question)
    
    assert score >= 0.6, f"Expected >= 0.6, got {score}"
    assert details['length_score'] > 0
    assert details['relevance_score'] > 0
    print(f"✅ test_good_answer PASSED (score: {score:.2f})")


def test_uncertain_answer():
    """Test answer with uncertainty."""
    scorer = RewardScorer()
    question = "What is quantum computing?"
    answer = "I don't know much about quantum computing. I'm not sure about the details."

    score, details = scorer.compute_answer_reward_heuristic(answer, question)

    # Uncertain answers should have 0 completeness score
    assert details['completeness_score'] == 0.0, f"Expected completeness=0.0, got {details['completeness_score']}"
    # Overall score should be lower than good answers
    assert score < 0.8, f"Expected < 0.8, got {score}"
    print(f"✅ test_uncertain_answer PASSED (score: {score:.2f})")


def test_empty_answer():
    """Test empty answer."""
    scorer = RewardScorer()
    question = "What is AI?"
    answer = ""
    
    score, details = scorer.compute_answer_reward_heuristic(answer, question)
    
    assert score == 0.0, f"Expected 0.0, got {score}"
    print("✅ test_empty_answer PASSED")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RUNNING REWARD SCORER TESTS")
    print("=" * 70 + "\n")
    
    tests = [
        test_perfect_format_with_query,
        test_perfect_format_with_answer,
        test_partial_format_think_only,
        test_no_tags,
        test_malformed_tags,
        test_good_answer,
        test_uncertain_answer,
        test_empty_answer,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

