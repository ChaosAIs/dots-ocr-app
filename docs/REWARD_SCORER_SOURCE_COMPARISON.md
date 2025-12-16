# Reward Scorer Implementation - Source Code Comparison

## ❌ Critical Finding: Implementation Does NOT Directly Reference Graph-R1 Source Code

After detailed analysis, **the current `reward_scorer.py` implementation does NOT accurately follow the Graph-R1-Full-Solution source code**. Here are the key discrepancies:

---

## 1. Format Reward Scoring - MAJOR DIFFERENCES

### Graph-R1 Source Code (`qa_em_and_format.py`)

<augment_code_snippet path="Graph-R1-Full-Solution/verl/utils/reward_score/qa_em_and_format.py" mode="EXCERPT">
```python
def compute_score_format(solution_str):
    """The scoring function for format reward."""
    # Lines 72-116
    
    format_reward = 0.0
    
    # Check for assistant blocks with special tokens
    assistant_blocks = re.findall(
        r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', 
        solution_str, 
        re.DOTALL
    )
    
    # For intermediate blocks: <think>...</think>\n<query>...</query>
    for i, assistant_block in enumerate(assistant_blocks[:-1]):
        if (assistant_block.count('<think>') == 1 and 
            assistant_block.count('</think>') == 1 and 
            assistant_block.count('<query>') == 1 and 
            assistant_block.count('</query>') == 1):
            
            think_match = re.search(
                r'^<think>(.*?)</think>\n<query>(.*?)</query>$', 
                assistant_block, 
                re.DOTALL
            )
            if think_match:
                format_reward += 0.5  # Each intermediate step: +0.5
    
    # For final block: <think>...</think>\n<answer>...</answer>
    if assistant_blocks:
        last_assistant_block = assistant_blocks[-1]
        think_answer_match = re.search(
            r'^<think>(.*?)</think>\n<answer>(.*?)</answer>$', 
            last_assistant_block, 
            re.DOTALL
        )
        if think_answer_match:
            format_reward += 0.5  # Final answer: +0.5
    
    return format_reward  # Can be > 1.0 for multi-turn!
```
</augment_code_snippet>

### Our Implementation (`reward_scorer.py`)

<augment_code_snippet path="backend/rag_service/graph_rag/reward_scorer.py" mode="EXCERPT">
```python
def compute_format_reward(self, response: str) -> Tuple[float, Dict[str, Any]]:
    """Compute format reward based on response structure."""
    # Lines 28-86
    
    score = 0.0
    
    # Check 1: Has <think> tags (0.3 points)
    if '<think>' in response and '</think>' in response:
        score += 0.3
    
    # Check 2: Has action tags (0.4 points)
    has_query = '<query>' in response and '</query>' in response
    has_answer = '<answer>' in response and '</answer>' in response
    if has_query or has_answer:
        score += 0.4
    
    # Check 3: Proper structure (0.3 points)
    if self._validate_structure(response):
        score += 0.3
    
    # Ensure score is in [0.0, 1.0]
    score = min(max(score, 0.0), 1.0)
    
    return score, details
```
</augment_code_snippet>

### Key Differences:

| Aspect | Graph-R1 Source | Our Implementation | Impact |
|--------|----------------|-------------------|--------|
| **Special Tokens** | Uses `<\|im_start\|>assistant` and `<\|im_end\|>` | ❌ Does NOT use these tokens | ⚠️ **CRITICAL** |
| **Scoring Method** | +0.5 per valid turn (can exceed 1.0) | +0.3/0.4/0.3 breakdown (capped at 1.0) | ⚠️ **MAJOR** |
| **Multi-turn Support** | ✅ Scores each turn separately | ❌ Single response only | ⚠️ **MAJOR** |
| **Exact Format Match** | Requires exact `<think>...\n<query>` format | Simple tag presence check | ⚠️ **MAJOR** |
| **Newline Requirement** | Requires `\n` between tags | No newline requirement | ⚠️ **MODERATE** |

---

## 2. Answer Reward Scoring - COMPLETELY DIFFERENT

### Graph-R1 Source Code

<augment_code_snippet path="Graph-R1-Full-Solution/verl/utils/reward_score/qa_em_and_format.py" mode="EXCERPT">
```python
def compute_score_answer(solution_str, ground_truth):
    """The scoring function for exact match (EM) with format reward."""
    # Lines 119-159
    
    # Extract answer from last assistant block
    assistant_blocks = re.findall(
        r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', 
        solution_str, 
        re.DOTALL
    )
    solution_str = assistant_blocks[-1]
    answer = extract_solution(solution_str)  # Extract from <answer> tags
    
    answer_reward = 0.0
    
    if answer is not None:
        # Use F1 score against ground truth
        answer_reward = cal_f1([ground_truth.tolist()], [answer])
    
    return answer_reward  # 0.0 to 1.0 based on F1 score
```
</augment_code_snippet>

### Our Implementation

<augment_code_snippet path="backend/rag_service/graph_rag/reward_scorer.py" mode="EXCERPT">
```python
def compute_answer_reward_heuristic(self, answer: str, question: str):
    """Compute answer reward using heuristics (no ground truth)."""
    # Lines 136-259
    
    score = 0.0
    
    # 1. Length score (0.25 points)
    # 2. Completeness score (0.25 points) - checks for uncertainty markers
    # 3. Confidence score (0.25 points) - hedging vs confident words
    # 4. Relevance score (0.25 points) - keyword overlap
    
    return score, details  # Heuristic-based, NO ground truth
```
</augment_code_snippet>

### Key Differences:

| Aspect | Graph-R1 Source | Our Implementation | Impact |
|--------|----------------|-------------------|--------|
| **Ground Truth** | ✅ Requires ground truth | ❌ No ground truth (heuristic) | ⚠️ **CRITICAL** |
| **Scoring Method** | F1 score (token overlap) | Length + completeness + confidence + relevance | ⚠️ **CRITICAL** |
| **Accuracy** | Objective (F1/EM) | Subjective (heuristics) | ⚠️ **CRITICAL** |

---

## 3. Combined Reward Function - MISSING

### Graph-R1 Source Code

```python
def compute_score_format_answer(solution_str, ground_truth):
    """Combined format + answer reward."""
    format_reward = compute_score_format(solution_str)
    answer_reward = compute_score_answer(solution_str, ground_truth)
    
    format_reward = min(format_reward, 1.0)
    if format_reward == 1.0:
        return -1.0 + format_reward + answer_reward  # Range: [0, 1]
    else:
        return -1.0 + format_reward  # Range: [-1, 0)
```

**Key insight**: The combined reward is **negative** unless format is perfect (1.0), then it adds answer reward.

### Our Implementation

❌ **MISSING** - We compute format and answer rewards separately but don't combine them with the `-1.0 + ...` formula.

---

## 4. Response Format - INCOMPATIBLE

### Graph-R1 Expected Format

```
<|im_start|>assistant
<think>
Reasoning here...
</think>
<query>Search query here</query>
<|im_end|>
<|im_start|>tool
<knowledge>Retrieved knowledge here</knowledge>
<|im_end|>
<|im_start|>assistant
<think>
Final reasoning...
</think>
<answer>Final answer here</answer>
<|im_end|>
```

### Our Current Format

```
<think>
Reasoning here...
</think>
<query>Search query here</query>
```

**No special tokens, no multi-turn structure.**

---

## Summary of Discrepancies

| Component | Alignment | Severity |
|-----------|-----------|----------|
| Format reward scoring method | ❌ 30% match | 🔴 CRITICAL |
| Special token handling | ❌ 0% match | 🔴 CRITICAL |
| Multi-turn support | ❌ 0% match | 🔴 CRITICAL |
| Answer reward method | ❌ 0% match (heuristic vs F1) | 🔴 CRITICAL |
| Combined reward formula | ❌ Missing | 🔴 CRITICAL |
| Response format | ❌ Incompatible | 🔴 CRITICAL |

**Overall Alignment: ~15%** ⚠️

---

## Conclusion

The current `reward_scorer.py` implementation:

1. ✅ **Captures the concept** of format and answer rewards
2. ✅ **Returns scores in 0.0-1.0 range**
3. ❌ **Does NOT follow the actual Graph-R1 source code implementation**
4. ❌ **Uses completely different scoring logic**
5. ❌ **Missing multi-turn support**
6. ❌ **Missing special token handling**
7. ❌ **Uses heuristics instead of F1/EM scoring**

The implementation is a **simplified, heuristic-based approximation** of the Graph-R1 design principles, NOT a faithful reproduction of the source code.

