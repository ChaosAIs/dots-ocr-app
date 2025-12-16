"""
Reward scorer for Graph-R1 style evaluation.

This module implements reward scoring mechanisms to evaluate the quality
of agent responses, following the Graph-R1 design principles.
"""

import re
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class RewardScorer:
    """
    Computes reward scores for agent responses.
    
    Implements two types of rewards:
    1. Format Reward: Evaluates response structure (tags, formatting)
    2. Answer Reward: Evaluates answer quality (heuristic-based)
    """
    
    def __init__(self):
        """Initialize the reward scorer."""
        logger.info("[RewardScorer] Initialized")
    
    def compute_format_reward(self, response: str) -> Tuple[float, Dict[str, Any]]:
        """
        Compute format reward based on response structure.
        
        Scoring breakdown:
        - Has <think> tags: +0.3 points
        - Has action tags (<query> or <answer>): +0.4 points
        - Proper structure (valid nesting): +0.3 points
        
        Args:
            response: The LLM response string
            
        Returns:
            Tuple of (score, details_dict)
            - score: Float between 0.0 and 1.0
            - details: Dict with breakdown information
        """
        score = 0.0
        details = {
            'has_think': False,
            'has_action': False,
            'valid_structure': False,
            'breakdown': {}
        }
        
        # Check 1: Has <think> tags (0.3 points)
        if '<think>' in response and '</think>' in response:
            score += 0.3
            details['has_think'] = True
            details['breakdown']['think_tags'] = 0.3
        else:
            details['breakdown']['think_tags'] = 0.0
        
        # Check 2: Has action tags (0.4 points)
        has_query = '<query>' in response and '</query>' in response
        has_answer = '<answer>' in response and '</answer>' in response
        
        if has_query or has_answer:
            score += 0.4
            details['has_action'] = True
            details['breakdown']['action_tags'] = 0.4
            details['action_type'] = 'query' if has_query else 'answer'
        else:
            details['breakdown']['action_tags'] = 0.0
        
        # Check 3: Proper structure (0.3 points)
        if self._validate_structure(response):
            score += 0.3
            details['valid_structure'] = True
            details['breakdown']['structure'] = 0.3
        else:
            details['breakdown']['structure'] = 0.0
        
        # Ensure score is in [0.0, 1.0]
        score = min(max(score, 0.0), 1.0)
        
        logger.debug(f"[RewardScorer] Format reward: {score:.2f}, details: {details}")
        
        return score, details
    
    def _validate_structure(self, response: str) -> bool:
        """
        Validate that tags are properly structured.
        
        Checks:
        - Tags are properly closed
        - No overlapping tags
        - Think comes before action
        
        Args:
            response: The response string
            
        Returns:
            True if structure is valid, False otherwise
        """
        # Check for proper tag closure
        think_open = response.count('<think>')
        think_close = response.count('</think>')
        query_open = response.count('<query>')
        query_close = response.count('</query>')
        answer_open = response.count('<answer>')
        answer_close = response.count('</answer>')
        
        # All opened tags must be closed
        if think_open != think_close:
            return False
        if query_open != query_close:
            return False
        if answer_open != answer_close:
            return False
        
        # Should have at most one of each tag type
        if think_open > 1 or query_open > 1 or answer_open > 1:
            return False
        
        # If has think tag, it should come before action tags
        if think_open > 0:
            think_pos = response.find('<think>')
            query_pos = response.find('<query>')
            answer_pos = response.find('<answer>')
            
            if query_pos != -1 and think_pos > query_pos:
                return False
            if answer_pos != -1 and think_pos > answer_pos:
                return False
        
        return True
    
    def compute_answer_reward_heuristic(
        self,
        answer: str,
        question: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute answer reward using heuristics (no ground truth).

        Heuristic scoring based on:
        - Answer length (not too short, not too long)
        - Completeness indicators
        - Confidence indicators
        - Relevance to question

        Args:
            answer: The answer text
            question: The original question

        Returns:
            Tuple of (score, details_dict)
            - score: Float between 0.0 and 1.0
            - details: Dict with scoring breakdown
        """
        score = 0.0
        details = {
            'length_score': 0.0,
            'completeness_score': 0.0,
            'confidence_score': 0.0,
            'relevance_score': 0.0,
        }

        if not answer or not answer.strip():
            return 0.0, details

        answer_lower = answer.lower()
        question_lower = question.lower()

        # 1. Length score (0.25 points)
        # Good answers are typically 50-500 characters
        answer_len = len(answer.strip())
        if answer_len < 20:
            details['length_score'] = 0.0
        elif answer_len < 50:
            details['length_score'] = 0.1
        elif answer_len <= 500:
            details['length_score'] = 0.25
        elif answer_len <= 1000:
            details['length_score'] = 0.2
        else:
            details['length_score'] = 0.15

        score += details['length_score']

        # 2. Completeness score (0.25 points)
        # Check for uncertainty markers (negative indicators)
        uncertainty_markers = [
            'i don\'t know',
            'not sure',
            'unclear',
            'cannot determine',
            'insufficient information',
            'no information',
            'unable to answer'
        ]

        has_uncertainty = any(marker in answer_lower for marker in uncertainty_markers)

        if has_uncertainty:
            details['completeness_score'] = 0.0
        else:
            # Check for completeness indicators
            completeness_indicators = [
                'based on',
                'according to',
                'specifically',
                'in particular',
                'for example',
                'such as'
            ]
            completeness_count = sum(1 for ind in completeness_indicators if ind in answer_lower)
            details['completeness_score'] = min(0.25, completeness_count * 0.1)

        score += details['completeness_score']

        # 3. Confidence score (0.25 points)
        # Check for hedging vs confident language
        hedging_words = ['maybe', 'perhaps', 'possibly', 'might', 'could be']
        confident_words = ['is', 'are', 'includes', 'provides', 'demonstrates']

        hedging_count = sum(1 for word in hedging_words if word in answer_lower)
        confident_count = sum(1 for word in confident_words if word in answer_lower)

        if hedging_count > confident_count:
            details['confidence_score'] = 0.1
        elif confident_count > 0:
            details['confidence_score'] = 0.25
        else:
            details['confidence_score'] = 0.15

        score += details['confidence_score']

        # 4. Relevance score (0.25 points)
        # Check if answer contains key terms from question
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        # Remove common stop words
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'who', 'which', 'are', 'do', 'does'}
        question_keywords = question_words - stop_words

        if question_keywords:
            answer_words = set(re.findall(r'\b\w+\b', answer_lower))
            overlap = len(question_keywords & answer_words)
            relevance_ratio = overlap / len(question_keywords)
            details['relevance_score'] = min(0.25, relevance_ratio * 0.5)
        else:
            details['relevance_score'] = 0.15

        score += details['relevance_score']

        # Ensure score is in [0.0, 1.0]
        score = min(max(score, 0.0), 1.0)

        logger.debug(f"[RewardScorer] Answer reward: {score:.2f}, details: {details}")

        return score, details

