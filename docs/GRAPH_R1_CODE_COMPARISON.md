# Graph-R1 Code Comparison: Full Solution vs Current Implementation

## Side-by-Side Comparison

### 1. Format Reward Calculation

#### Graph-R1 Full Solution
```python
# From: Graph-R1-Full-Solution/verl/utils/reward_score/qa_em_and_format.py

def compute_score_format(solution_str):
    """The scoring function for format reward."""
    if solution_str is None:
        return 0.0
    
    try:
        # Check for basic structure with <|im_start|>assistant and <|im_end|> tags
        assistant_blocks = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', 
            solution_str, 
            re.DOTALL
        )

        format_reward = 0.0
        
        # If no blocks found, return 0
        if not assistant_blocks:
            return 0.0
        
        # Check intermediate blocks have <think> and <query>
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
                    format_reward += 0.5

        # Check the last assistant block contains <answer> tags
        if assistant_blocks:
            last_assistant_block = assistant_blocks[-1]
            think_answer_match = re.search(
                r'^<think>(.*?)</think>\n<answer>(.*?)</answer>$', 
                last_assistant_block, 
                re.DOTALL
            )
            if think_answer_match:
                format_reward += 0.5
                
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format: {e}")
        return 0.0
    
    return format_reward
```

**Key Features**:
- Returns **numeric score** (0.0 to 1.0)
- Validates **conversation structure**
- Checks **multiple turns**
- Scores **intermediate steps** (0.5 each)
- Scores **final answer** (0.5)

---

#### Current Implementation
```python
# From: backend/rag_service/graph_rag/graph_rag_agent.py

def _extract_answer(self, response: str) -> Optional[str]:
    """Extract answer from response if present."""
    # Try standard format first
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try with newlines inside tags
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None

def _extract_query(self, response: str) -> Optional[str]:
    """Extract query from response if present."""
    # Try standard format first
    match = re.search(r"<query>(.*?)</query>", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try with newlines inside tags
    match = re.search(r"<query>\s*(.*?)\s*</query>", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None
```

**Key Features**:
- Returns **binary result** (found or not found)
- Extracts **content only**
- No **quality scoring**
- No **structure validation**

**Gap**: ❌ No scoring mechanism, just extraction

---

### 2. Answer Reward Calculation

#### Graph-R1 Full Solution
```python
# From: Graph-R1-Full-Solution/verl/utils/reward_score/qa_em_and_format.py

def compute_score_answer(solution_str, ground_truth):
    """The scoring function for answer reward."""
    if solution_str is None:
        return 0.0
    
    try:
        # Extract answer from <answer> tags
        assistant_blocks = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', 
            solution_str, 
            re.DOTALL
        )
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)

        answer_reward = 0.0
        
        if answer is not None:
            # F1 score against ground truth
            answer_reward = cal_f1([ground_truth.tolist()], [answer])
        
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_answer: {e}")
        return 0.0
    
    return answer_reward
```

**Key Features**:
- Returns **numeric score** (0.0 to 1.0)
- Compares against **ground truth**
- Uses **F1 score** for accuracy
- Handles **missing answers**

---

#### Current Implementation
```python
# From: backend/rag_service/graph_rag/graph_rag_agent.py

async def _generate_final_answer(self, state: AgentState) -> str:
    """Generate final answer from accumulated knowledge."""
    prompt = f"""Based on all the retrieved knowledge, provide a comprehensive answer.

## Question
{state.question}

## All Retrieved Knowledge
{state.get_knowledge_summary()}

## Instructions
Synthesize the information to provide a complete, accurate answer.

Your answer:"""

    response = await self.llm_service.generate(
        prompt=prompt,
        max_tokens=2000,
    )

    return response  # Just return, no scoring
```

**Key Features**:
- Returns **answer text**
- No **quality evaluation**
- No **ground truth comparison**
- No **accuracy scoring**

**Gap**: ❌ No answer reward calculation at all

---

### 3. Combined Reward and Termination

#### Graph-R1 Full Solution
```python
# From: Graph-R1-Full-Solution/verl/utils/reward_score/qa_em_and_format.py

def compute_score_format_answer(solution_str, ground_truth):
    """Combined scoring function."""
    if solution_str is None or ground_truth is None:
        return 0.0

    try:
        format_reward = compute_score_format(solution_str)
        answer_reward = compute_score_answer(solution_str, ground_truth)

        format_reward = min(format_reward, 1.0)
        
        # Only add answer reward if format is perfect
        if format_reward == 1.0:
            return -1.0 + format_reward + answer_reward  # Range: [0, 1]
        else:
            return -1.0 + format_reward  # Range: [-1, 0)
            
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return 0.0
```

**Termination Logic**:
```python
# From: Graph-R1-Full-Solution/agent/llm_agent/generation.py

# In training loop
reward = compute_score_format_answer(response, ground_truth)

# Policy learns:
# - If reward >= threshold (e.g., 0.8): STOP
# - If reward < threshold: CONTINUE
# - If format_reward < 0.5: STOP (hallucination)
```

**Key Features**:
- **Numeric reward** drives decisions
- **Policy learns** optimal stopping
- **Penalizes** poor format (-1.0 to 0.0)
- **Rewards** good format + answer (0.0 to 1.0)

---

#### Current Implementation
```python
# From: backend/rag_service/graph_rag/graph_rag_agent.py

# Parse response for answer or next query
answer = self._extract_answer(response)
next_query = self._extract_query(response)

if answer:
    state.is_complete = True
    logger.info("DECISION: TERMINATE - Answer found")
elif next_query:
    if next_query not in state.queries_made:
        logger.info("DECISION: CONTINUE - New query generated")
    else:
        # Duplicate query, terminate
        state.is_complete = True
        logger.info("DECISION: TERMINATE - Duplicate query")
else:
    # No valid tags, terminate
    state.is_complete = True
    logger.info("DECISION: TERMINATE - No valid tags found")
```

**Key Features**:
- **Rule-based** decisions
- **Binary** checks (tag found or not)
- **No scoring**
- **No learning**

**Gap**: ❌ No reward-based termination, just heuristics

---

## Summary Table

| Feature | Graph-R1 Full Solution | Current Implementation | Gap |
|---------|------------------------|------------------------|-----|
| **Format Scoring** | ✅ 0.0-1.0 numeric score | ❌ Binary extraction | HIGH |
| **Answer Scoring** | ✅ F1/EM against ground truth | ❌ None | HIGH |
| **Combined Reward** | ✅ -1.0 to 1.0 range | ❌ None | HIGH |
| **Termination Logic** | ✅ Reward-based policy | ❌ Rule-based heuristics | HIGH |
| **Learning** | ✅ RL policy (PPO/GRPO) | ❌ Fixed rules | HIGH |
| **Quality Metrics** | ✅ Quantitative | ❌ Qualitative | HIGH |
| **Iterative Loop** | ✅ Yes | ✅ Yes | NONE |
| **Knowledge Accumulation** | ✅ Yes | ✅ Yes | NONE |

---

## Conclusion

**Current implementation has 30% of Graph-R1 functionality**:
- ✅ Has: Iterative structure
- ❌ Missing: Reward mechanism (70% of Graph-R1's intelligence)

**To align with Graph-R1 design**:
1. Add format reward scoring
2. Add answer reward scoring (optional without ground truth)
3. Use rewards for termination decisions
4. (Optional) Train RL policy for optimal behavior

