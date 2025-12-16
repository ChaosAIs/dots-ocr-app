"""
Unit tests for RewardScorer.
"""

import pytest
from rag_service.graph_rag.reward_scorer import RewardScorer


class TestFormatReward:
    """Test format reward computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = RewardScorer()
    
    def test_perfect_format_with_query(self):
        """Test perfect format with query tag."""
        response = """<think>
I need to search for information about cloud computing.
</think>
<query>cloud computing definition</query>"""
        
        score, details = self.scorer.compute_format_reward(response)
        
        assert score == 1.0, f"Expected 1.0, got {score}"
        assert details['has_think'] is True
        assert details['has_action'] is True
        assert details['valid_structure'] is True
        assert details['action_type'] == 'query'
    
    def test_perfect_format_with_answer(self):
        """Test perfect format with answer tag."""
        response = """<think>
I have enough information to answer.
</think>
<answer>Cloud computing is a model for delivering computing services over the internet.</answer>"""
        
        score, details = self.scorer.compute_format_reward(response)
        
        assert score == 1.0, f"Expected 1.0, got {score}"
        assert details['has_think'] is True
        assert details['has_action'] is True
        assert details['valid_structure'] is True
        assert details['action_type'] == 'answer'
    
    def test_partial_format_think_only(self):
        """Test partial format with only think tag."""
        response = """<think>
I'm thinking about this problem.
</think>"""
        
        score, details = self.scorer.compute_format_reward(response)
        
        # Should get 0.3 for think + 0.3 for structure = 0.6
        assert score == 0.6, f"Expected 0.6, got {score}"
        assert details['has_think'] is True
        assert details['has_action'] is False
        assert details['valid_structure'] is True
    
    def test_partial_format_action_only(self):
        """Test partial format with only action tag."""
        response = """<query>search for information</query>"""
        
        score, details = self.scorer.compute_format_reward(response)
        
        # Should get 0.4 for action + 0.3 for structure = 0.7
        assert score == 0.7, f"Expected 0.7, got {score}"
        assert details['has_think'] is False
        assert details['has_action'] is True
        assert details['valid_structure'] is True
    
    def test_no_tags(self):
        """Test response with no tags."""
        response = "Just plain text without any tags."
        
        score, details = self.scorer.compute_format_reward(response)
        
        assert score == 0.3, f"Expected 0.3, got {score}"  # Only structure points
        assert details['has_think'] is False
        assert details['has_action'] is False
        assert details['valid_structure'] is True
    
    def test_malformed_tags_unclosed(self):
        """Test malformed tags (unclosed)."""
        response = """<think>
I'm thinking but forgot to close the tag
<query>search query</query>"""
        
        score, details = self.scorer.compute_format_reward(response)
        
        # Has action but invalid structure
        assert score == 0.4, f"Expected 0.4, got {score}"
        assert details['has_think'] is False  # Unclosed
        assert details['has_action'] is True
        assert details['valid_structure'] is False
    
    def test_malformed_tags_wrong_order(self):
        """Test tags in wrong order."""
        response = """<query>search first</query>
<think>then think</think>"""
        
        score, details = self.scorer.compute_format_reward(response)
        
        # Has both tags but invalid structure (wrong order)
        assert score == 0.7, f"Expected 0.7, got {score}"  # 0.3 + 0.4, no structure points
        assert details['has_think'] is True
        assert details['has_action'] is True
        assert details['valid_structure'] is False
    
    def test_duplicate_tags(self):
        """Test duplicate tags."""
        response = """<think>first thought</think>
<think>second thought</think>
<query>search query</query>"""
        
        score, details = self.scorer.compute_format_reward(response)
        
        # Has tags but invalid structure (duplicates)
        assert score == 0.7, f"Expected 0.7, got {score}"
        assert details['valid_structure'] is False


class TestAnswerReward:
    """Test answer reward computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = RewardScorer()
    
    def test_good_answer(self):
        """Test a good quality answer."""
        question = "What is cloud computing?"
        answer = """Cloud computing is a model for delivering computing services over the internet, 
including servers, storage, databases, networking, and software. Major providers include AWS, 
Azure, and Google Cloud Platform. It provides on-demand access to resources."""
        
        score, details = self.scorer.compute_answer_reward_heuristic(answer, question)
        
        # Should score reasonably high
        assert score >= 0.6, f"Expected >= 0.6, got {score}"
        assert details['length_score'] > 0
        assert details['relevance_score'] > 0
    
    def test_uncertain_answer(self):
        """Test answer with uncertainty."""
        question = "What is quantum computing?"
        answer = "I don't know much about quantum computing. I'm not sure about the details."
        
        score, details = self.scorer.compute_answer_reward_heuristic(answer, question)
        
        # Should score low due to uncertainty
        assert score < 0.5, f"Expected < 0.5, got {score}"
        assert details['completeness_score'] == 0.0
    
    def test_empty_answer(self):
        """Test empty answer."""
        question = "What is AI?"
        answer = ""
        
        score, details = self.scorer.compute_answer_reward_heuristic(answer, question)
        
        assert score == 0.0
    
    def test_very_short_answer(self):
        """Test very short answer."""
        question = "What is Python?"
        answer = "A language."
        
        score, details = self.scorer.compute_answer_reward_heuristic(answer, question)
        
        # Should score low due to length
        assert score < 0.4, f"Expected < 0.4, got {score}"
        assert details['length_score'] < 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

