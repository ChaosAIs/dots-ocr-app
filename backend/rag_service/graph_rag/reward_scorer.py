"""
Reward scorer for Graph-R1 style evaluation.

This module implements reward scoring mechanisms to evaluate the quality
of agent responses, following the Graph-R1 design principles.
"""

import re
import logging
import os
import json
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


class RewardScorer:
    """
    Computes reward scores for agent responses.

    Implements two types of rewards:
    1. Format Reward: Evaluates response structure (tags, formatting)
    2. Answer Reward: Evaluates answer quality (LLM-based or heuristic)
    """

    def __init__(self, llm_service=None):
        """
        Initialize the reward scorer.

        Args:
            llm_service: Optional LLM service for LLM-based answer evaluation
        """
        self.llm_service = llm_service
        self.use_llm_scoring = os.getenv("GRAPH_RAG_USE_LLM_ANSWER_SCORING", "true").lower() == "true"

        if self.use_llm_scoring and llm_service:
            logger.info("[RewardScorer] Initialized with LLM-based answer scoring")
        else:
            logger.info("[RewardScorer] Initialized with heuristic-based answer scoring")
    
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

    async def compute_answer_reward(
        self,
        answer: str,
        question: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute answer reward (automatically chooses LLM or heuristic method).

        Args:
            answer: The answer text
            question: The original question

        Returns:
            Tuple of (score, details_dict)
        """
        if self.use_llm_scoring and self.llm_service:
            return await self.compute_answer_reward_llm(answer, question)
        else:
            return self.compute_answer_reward_heuristic(answer, question)

    async def compute_answer_reward_llm(
        self,
        answer: str,
        question: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute answer reward using LLM-based evaluation.

        The LLM evaluates:
        - Completeness: Does the answer fully address the question?
        - Accuracy: Are there logical inconsistencies or contradictions?
        - Relevance: Is the answer on-topic?
        - Quality: Is the answer well-structured and clear?

        Args:
            answer: The answer text
            question: The original question

        Returns:
            Tuple of (score, details_dict)
            - score: Float between 0.0 and 1.0
            - details: Dict with scoring breakdown
        """
        if not self.llm_service:
            logger.warning("[RewardScorer] LLM service not available, falling back to heuristic")
            return self.compute_answer_reward_heuristic(answer, question)

        evaluation_prompt = f"""You are an answer quality evaluator. Evaluate the following answer to the question.

Question: {question}

Answer: {answer}

Evaluate the answer on these criteria (score each 0-10):

1. **Completeness**: Does the answer fully address all parts of the question?
   - Check if all requested information is provided
   - For counting questions, verify the count matches the listed items

2. **Accuracy**: Are there logical inconsistencies or contradictions?
   - Check for number mismatches (e.g., "10 items" but only lists 5)
   - Check for contradictory statements
   - Verify calculations are correct

3. **Relevance**: Is the answer on-topic and relevant to the question?
   - Check if non-relevant items are included (e.g., fees/charges in product lists)
   - Verify the answer addresses what was asked

4. **Clarity**: Is the answer well-structured and clear?
   - Check formatting and organization
   - Verify the answer is easy to understand

Respond in this exact JSON format:
{{
  "completeness": <score 0-10>,
  "accuracy": <score 0-10>,
  "relevance": <score 0-10>,
  "clarity": <score 0-10>,
  "issues": ["list", "of", "specific", "issues", "found"],
  "overall_score": <average score 0-10>
}}

Be strict in your evaluation. Deduct points for any issues found."""

        try:
            from langchain_core.messages import HumanMessage

            # Get LLM for evaluation (use query model for speed)
            llm = self.llm_service.get_query_model(temperature=0.1, num_predict=500)
            response_msg = await llm.ainvoke([HumanMessage(content=evaluation_prompt)])
            response = response_msg.content.strip()

            logger.debug(f"[RewardScorer] LLM evaluation response: {response[:200]}...")

            # Parse JSON response
            # Try to extract JSON from response (handle cases where LLM adds extra text)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group(0))

                # Normalize scores to 0.0-1.0 range
                completeness = float(eval_data.get("completeness", 5)) / 10.0
                accuracy = float(eval_data.get("accuracy", 5)) / 10.0
                relevance = float(eval_data.get("relevance", 5)) / 10.0
                clarity = float(eval_data.get("clarity", 5)) / 10.0

                # Overall score (weighted average)
                # Accuracy and completeness are most important
                score = (
                    accuracy * 0.35 +
                    completeness * 0.35 +
                    relevance * 0.20 +
                    clarity * 0.10
                )

                details = {
                    'method': 'llm',
                    'completeness': completeness,
                    'accuracy': accuracy,
                    'relevance': relevance,
                    'clarity': clarity,
                    'issues': eval_data.get('issues', []),
                    'llm_overall': float(eval_data.get('overall_score', 5)) / 10.0,
                }

                # Log issues if found
                if eval_data.get('issues'):
                    for issue in eval_data['issues']:
                        logger.warning(f"[RewardScorer] LLM detected issue: {issue}")

                logger.info(f"[RewardScorer] LLM answer score: {score:.2f} "
                           f"(accuracy={accuracy:.2f}, completeness={completeness:.2f})")

                return score, details
            else:
                logger.warning("[RewardScorer] Failed to parse LLM evaluation JSON, falling back to heuristic")
                return self.compute_answer_reward_heuristic(answer, question)

        except Exception as e:
            logger.error(f"[RewardScorer] Error in LLM evaluation: {e}, falling back to heuristic")
            return self.compute_answer_reward_heuristic(answer, question)

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
        - Consistency checks (detect contradictions)

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
            'consistency_score': 0.0,
        }

        if not answer or not answer.strip():
            return 0.0, details

        answer_lower = answer.lower()
        question_lower = question.lower()

        # 1. Length score (0.2 points - reduced from 0.25)
        # Good answers are typically 50-500 characters
        answer_len = len(answer.strip())
        if answer_len < 20:
            details['length_score'] = 0.0
        elif answer_len < 50:
            details['length_score'] = 0.08
        elif answer_len <= 500:
            details['length_score'] = 0.2
        elif answer_len <= 1000:
            details['length_score'] = 0.16
        else:
            details['length_score'] = 0.12

        score += details['length_score']

        # 2. Completeness score (0.2 points - reduced from 0.25)
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
            details['completeness_score'] = min(0.2, completeness_count * 0.08)

        score += details['completeness_score']

        # 3. Confidence score (0.2 points - reduced from 0.25)
        # Check for hedging vs confident language
        hedging_words = ['maybe', 'perhaps', 'possibly', 'might', 'could be']
        confident_words = ['is', 'are', 'includes', 'provides', 'demonstrates']

        hedging_count = sum(1 for word in hedging_words if word in answer_lower)
        confident_count = sum(1 for word in confident_words if word in answer_lower)

        if hedging_count > confident_count:
            details['confidence_score'] = 0.08
        elif confident_count > 0:
            details['confidence_score'] = 0.2
        else:
            details['confidence_score'] = 0.12

        score += details['confidence_score']

        # 4. Relevance score (0.2 points - reduced from 0.25)
        # Check if answer contains key terms from question
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        # Remove common stop words
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'who', 'which', 'are', 'do', 'does'}
        question_keywords = question_words - stop_words

        if question_keywords:
            answer_words = set(re.findall(r'\b\w+\b', answer_lower))
            overlap = len(question_keywords & answer_words)
            relevance_ratio = overlap / len(question_keywords)
            details['relevance_score'] = min(0.2, relevance_ratio * 0.4)
        else:
            details['relevance_score'] = 0.12

        score += details['relevance_score']

        # 5. Consistency score (0.2 points - NEW!)
        # Check for internal contradictions and counting accuracy
        details['consistency_score'] = self._check_consistency(answer, question)
        score += details['consistency_score']

        # Ensure score is in [0.0, 1.0]
        score = min(max(score, 0.0), 1.0)

        logger.debug(f"[RewardScorer] Answer reward: {score:.2f}, details: {details}")

        return score, details

    def _check_consistency(self, answer: str, question: str) -> float:
        """
        Check for internal consistency in the answer.

        Detects:
        - Number mismatches (e.g., "10 items" but only lists 5)
        - Contradictory statements
        - Missing expected items when counting
        - Non-product items in product lists (fees, taxes, etc.)

        Args:
            answer: The answer text
            question: The original question

        Returns:
            Consistency score (0.0 to 0.2)
        """
        score = 0.2  # Start with full points, deduct for issues
        answer_lower = answer.lower()
        question_lower = question.lower()

        # Check 1: Non-product items in product/electronic lists
        if any(word in question_lower for word in ['product', 'electronic', 'item', 'bought', 'purchased']):
            # Keywords that indicate fees/charges, not products
            non_product_keywords = [
                'shipping', 'delivery', 'freight', 'postage', 'handling',
                'tax', 'vat', 'gst', 'duty', 'customs',
                'fee', 'charge', 'surcharge', 'service charge',
                'discount', 'rebate', 'refund',
                'insurance', 'warranty fee',
                'frais', 'livraison', 'expédition',  # French: fees, delivery, shipping
            ]

            # Extract list items from answer
            list_items = re.findall(r'^\s*\d+[\.\)]\s+(.+)$', answer, re.MULTILINE)

            # Check each list item for non-product keywords
            non_product_count = 0
            for item in list_items:
                item_lower = item.lower()
                if any(keyword in item_lower for keyword in non_product_keywords):
                    non_product_count += 1
                    logger.warning(
                        f"[RewardScorer] Non-product item detected in list: '{item.strip()}'"
                    )

            if non_product_count > 0:
                # Penalty for including fees/charges as products
                penalty = min(0.1, non_product_count * 0.05)
                score -= penalty
                logger.warning(
                    f"[RewardScorer] Found {non_product_count} non-product items in product list"
                )

        # Check 2: Number-list mismatch for counting questions
        if any(word in question_lower for word in ['how many', 'count', 'number of', 'total']):
            # Count list items (numbered lists like "1.", "2.", etc.)
            list_items = re.findall(r'^\s*\d+[\.\)]\s+', answer, re.MULTILINE)

            # Count bullet points
            bullet_items = re.findall(r'^\s*[-*•]\s+', answer, re.MULTILINE)

            actual_list_count = max(len(list_items), len(bullet_items))

            if actual_list_count > 0:
                # Look for explicit count statements like "total of X", "X items", "X products"
                # This avoids picking up SKU numbers or other large numbers
                count_patterns = [
                    r'total\s+(?:of\s+)?(\d+)',
                    r'(\d+)\s+(?:electronic\s+)?(?:product|item)s?',
                    r'purchased\s+(\d+)',
                    r'bought\s+(\d+)',
                ]

                claimed_total = None
                for pattern in count_patterns:
                    matches = re.findall(pattern, answer_lower)
                    if matches:
                        # Get the first match (most likely the total)
                        claimed_total = int(matches[0])
                        break

                # If we found a claimed total, check if it matches the list
                if claimed_total is not None and claimed_total != actual_list_count:
                    # Mismatch detected!
                    logger.warning(
                        f"[RewardScorer] Consistency issue: Claims {claimed_total} items "
                        f"but lists {actual_list_count} items"
                    )
                    score -= 0.15  # Major penalty for number mismatch

        # Check 3: Contradictory phrases
        contradictions = [
            ('all', 'some'),
            ('always', 'sometimes'),
            ('never', 'occasionally'),
            ('complete', 'incomplete'),
            ('total', 'partial'),
        ]

        for word1, word2 in contradictions:
            if word1 in answer_lower and word2 in answer_lower:
                # Potential contradiction
                score -= 0.05
                break

        # Ensure score is non-negative
        score = max(score, 0.0)

        logger.debug(f"[RewardScorer] Consistency score: {score:.2f}")

        return score

