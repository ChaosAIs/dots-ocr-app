"""
Reward scorer following Graph-R1-Full-Solution source code exactly.

This module implements the EXACT reward scoring mechanisms from:
Graph-R1-Full-Solution/verl/utils/reward_score/qa_em_and_format.py

Key differences from reward_scorer.py:
1. Uses <|im_start|>assistant and <|im_end|> special tokens
2. Supports multi-turn conversations
3. Requires exact format: <think>...\n<query> or <think>...\n<answer>
4. Format reward can exceed 1.0 for multi-turn
5. Answer reward uses F1 score (requires ground truth)
"""

import re
import string
import logging
from typing import Tuple, Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def normalize_answer(s: str) -> str:
    """
    Normalize answer text for comparison.
    From Graph-R1-Full-Solution/verl/utils/reward_score/qa_em_and_format.py
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_solution(solution_str: str) -> Optional[str]:
    """
    Extract the answer from the solution string.
    From Graph-R1-Full-Solution/verl/utils/reward_score/qa_em_and_format.py
    """
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Compute F1 score between prediction and ground truth.
    Simplified version - for full implementation, use Graph-R1's cal_f1.
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    
    common = set(pred_tokens) & set(truth_tokens)
    num_same = len(common)
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


class GraphR1RewardScorer:
    """
    Reward scorer following Graph-R1-Full-Solution source code exactly.
    
    Based on: Graph-R1-Full-Solution/verl/utils/reward_score/qa_em_and_format.py
    """
    
    def __init__(self):
        """Initialize the reward scorer."""
        logger.info("[GraphR1RewardScorer] Initialized (following Graph-R1 source code)")
    
    def compute_format_reward(self, solution_str: str) -> Tuple[float, Dict[str, Any]]:
        """
        Compute format reward following Graph-R1 source code EXACTLY.
        
        From: Graph-R1-Full-Solution/verl/utils/reward_score/qa_em_and_format.py
        Lines 72-116
        
        Scoring:
        - Each intermediate turn with <think>...\n<query>: +0.5
        - Final turn with <think>...\n<answer>: +0.5
        - Can exceed 1.0 for multi-turn conversations
        
        Args:
            solution_str: The full solution string with special tokens
            
        Returns:
            Tuple of (score, details_dict)
        """
        if solution_str is None:
            return 0.0, {'error': 'None input'}
        
        try:
            # Extract assistant blocks between <|im_start|>assistant and <|im_end|>
            assistant_blocks = re.findall(
                r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', 
                solution_str, 
                re.DOTALL
            )
            
            format_reward = 0.0
            details = {
                'num_blocks': len(assistant_blocks),
                'intermediate_turns': 0,
                'has_final_answer': False,
                'blocks': []
            }
            
            # If no blocks found, return 0
            if not assistant_blocks:
                return 0.0, details
            
            # Check intermediate assistant blocks (all except last)
            for i, assistant_block in enumerate(assistant_blocks[:-1]):
                block_info = {'index': i, 'type': 'intermediate', 'valid': False}
                
                # Must have exactly one <think> and one <query>
                if (assistant_block.count('<think>') == 1 and 
                    assistant_block.count('</think>') == 1 and 
                    assistant_block.count('<query>') == 1 and 
                    assistant_block.count('</query>') == 1):
                    
                    # Must match exact format: <think>...\n<query>...
                    think_match = re.search(
                        r'^<think>(.*?)</think>\n<query>(.*?)</query>$', 
                        assistant_block, 
                        re.DOTALL
                    )
                    if think_match:
                        format_reward += 0.5
                        details['intermediate_turns'] += 1
                        block_info['valid'] = True
                
                details['blocks'].append(block_info)
            
            # Check the last assistant block
            if assistant_blocks:
                last_assistant_block = assistant_blocks[-1]
                block_info = {'index': len(assistant_blocks)-1, 'type': 'final', 'valid': False}
                
                # Must match exact format: <think>...\n<answer>...
                think_answer_match = re.search(
                    r'^<think>(.*?)</think>\n<answer>(.*?)</answer>$', 
                    last_assistant_block, 
                    re.DOTALL
                )
                if think_answer_match:
                    format_reward += 0.5
                    details['has_final_answer'] = True
                    block_info['valid'] = True
                
                details['blocks'].append(block_info)
            
        except Exception as e:
            logger.error(f"[GraphR1RewardScorer] Error in compute_format_reward: {e}")
            return 0.0, {'error': str(e)}
        
        return format_reward, details

    def compute_answer_reward(
        self,
        solution_str: str,
        ground_truth: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute answer reward using F1 score.

        From: Graph-R1-Full-Solution/verl/utils/reward_score/qa_em_and_format.py
        Lines 119-159

        Args:
            solution_str: The full solution string with special tokens
            ground_truth: The ground truth answer

        Returns:
            Tuple of (score, details_dict)
        """
        if solution_str is None or ground_truth is None:
            return 0.0, {'error': 'None input'}

        try:
            # Extract answer from last assistant block
            assistant_blocks = re.findall(
                r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',
                solution_str,
                re.DOTALL
            )

            if not assistant_blocks:
                return 0.0, {'error': 'No assistant blocks found'}

            solution_str = assistant_blocks[-1]
            answer = extract_solution(solution_str)

            answer_reward = 0.0
            details = {'has_answer': False, 'f1_score': 0.0}

            if answer is not None:
                details['has_answer'] = True
                # Compute F1 score against ground truth
                answer_reward = compute_f1_score(answer, ground_truth)
                details['f1_score'] = answer_reward

        except Exception as e:
            logger.error(f"[GraphR1RewardScorer] Error in compute_answer_reward: {e}")
            return 0.0, {'error': str(e)}

        return answer_reward, details

    def compute_combined_reward(
        self,
        solution_str: str,
        ground_truth: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute combined format + answer reward.

        From: Graph-R1-Full-Solution/verl/utils/reward_score/qa_em_and_format.py
        Lines 161-182

        Formula:
        - If format_reward >= 1.0: return -1.0 + format_reward + answer_reward
        - Else: return -1.0 + format_reward

        This means:
        - Negative reward unless format is perfect
        - Only add answer reward if format is perfect

        Args:
            solution_str: The full solution string with special tokens
            ground_truth: The ground truth answer

        Returns:
            Tuple of (combined_score, details_dict)
        """
        if solution_str is None or ground_truth is None:
            return 0.0, {'error': 'None input'}

        try:
            format_reward, format_details = self.compute_format_reward(solution_str)
            answer_reward, answer_details = self.compute_answer_reward(solution_str, ground_truth)

            # Cap format reward at 1.0 for combined calculation
            format_reward_capped = min(format_reward, 1.0)

            # Graph-R1 formula: -1.0 + format + (answer if format=1.0)
            if format_reward_capped == 1.0:
                combined_reward = -1.0 + format_reward_capped + answer_reward
            else:
                combined_reward = -1.0 + format_reward_capped

            details = {
                'format_reward': format_reward,
                'format_reward_capped': format_reward_capped,
                'answer_reward': answer_reward,
                'combined_reward': combined_reward,
                'format_details': format_details,
                'answer_details': answer_details,
            }

        except Exception as e:
            logger.error(f"[GraphR1RewardScorer] Error in compute_combined_reward: {e}")
            return 0.0, {'error': str(e)}

        return combined_reward, details

