# Graph-R1 Implementation Roadmap

## Current Status: ❌ Missing Core Reward Mechanism

**Date**: 2025-12-16  
**Analysis Document**: [GRAPH_R1_DESIGN_ANALYSIS.md](./GRAPH_R1_DESIGN_ANALYSIS.md)

---

## Quick Summary

### What We Have ✅
- Iterative reasoning loop (think-query-retrieve)
- Multi-step knowledge accumulation
- LLM-based decision making
- Tag-based response parsing

### What We're Missing ❌
- **Format Reward Scoring** (0.0-1.0 quality metric)
- **Answer Reward Scoring** (accuracy evaluation)
- **Reward-Based Termination** (intelligent stopping)
- **Reinforcement Learning Policy** (learned behavior)

### Alignment Score: **3/10**

Current implementation has the **structure** but lacks the **intelligence** of Graph-R1.

---

## Recommended Implementation Path

### Phase 1: Add Reward Scoring (2-3 weeks) ⭐ START HERE

**Goal**: Implement reward calculation without full RL training

#### Week 1: Format Reward
```bash
# Create new module
backend/rag_service/graph_rag/reward_scorer.py
```

**Tasks**:
1. Port `compute_format_reward()` from Graph-R1 solution
2. Adapt to current response format
3. Add unit tests
4. Integrate into agent's `_think_and_decide()` method

**Code Example**:
```python
# reward_scorer.py
def compute_format_reward(response: str) -> float:
    """
    Score response format quality (0.0 to 1.0).
    
    Checks:
    - Presence of <think> tags
    - Presence of <query> or <answer> tags
    - Proper structure and nesting
    """
    format_reward = 0.0
    
    # Check for <think> tags (0.3 points)
    if '<think>' in response and '</think>' in response:
        format_reward += 0.3
    
    # Check for action tags (0.4 points)
    has_query = '<query>' in response and '</query>' in response
    has_answer = '<answer>' in response and '</answer>' in response
    if has_query or has_answer:
        format_reward += 0.4
    
    # Check for proper structure (0.3 points)
    if _validate_structure(response):
        format_reward += 0.3
    
    return min(format_reward, 1.0)
```

#### Week 2: Answer Reward (Optional)
```python
def compute_answer_reward(answer: str, reference: str = None) -> float:
    """
    Score answer quality (0.0 to 1.0).
    
    Without ground truth: Use heuristics
    - Length check
    - Completeness indicators
    - Confidence markers
    
    With ground truth: Use F1/EM scoring
    """
    if reference is None:
        # Heuristic scoring
        return _heuristic_answer_score(answer)
    else:
        # F1 score against reference
        return _compute_f1_score(answer, reference)
```

#### Week 3: Integration
```python
# In graph_rag_agent.py
async def _think_and_decide(self, state: AgentState) -> str:
    response = await self.llm_service.generate(...)
    
    # NEW: Compute rewards
    format_score = compute_format_reward(response)
    logger.info(f"Format reward: {format_score:.2f}")
    
    # NEW: Use rewards for termination
    if format_score >= 0.8 and self._extract_answer(response):
        state.is_complete = True
        logger.info("High format score + answer found -> TERMINATE")
    elif format_score < 0.3:
        state.is_complete = True
        logger.info("Low format score (hallucination?) -> TERMINATE")
    
    return response
```

---

### Phase 2: Data Collection (1-2 weeks)

**Goal**: Gather data to optimize thresholds

#### Tasks:
1. Log all interactions with rewards
2. Collect format scores, answer scores, termination decisions
3. Analyze distributions
4. Identify optimal thresholds

**Logging Example**:
```python
# Add to agent
interaction_log = {
    "query": question,
    "step": state.step,
    "response": response,
    "format_reward": format_score,
    "answer_reward": answer_score,
    "terminated": state.is_complete,
    "reason": termination_reason,
}
logger.info(f"INTERACTION_LOG: {json.dumps(interaction_log)}")
```

---

### Phase 3: Threshold Optimization (1 week)

**Goal**: Learn optimal decision thresholds from data

#### Analysis:
```python
# Analyze collected logs
import pandas as pd

logs = pd.read_json("interaction_logs.jsonl", lines=True)

# Find optimal format threshold
successful_terminations = logs[logs['terminated'] & logs['answer_found']]
print(f"Avg format score at success: {successful_terminations['format_reward'].mean()}")

# Find hallucination threshold
hallucinations = logs[logs['format_reward'] < 0.5]
print(f"Hallucination rate: {len(hallucinations) / len(logs)}")
```

#### Update Thresholds:
```python
# In agent config
class AgentConfig:
    format_threshold_high: float = 0.85  # Learned from data
    format_threshold_low: float = 0.25   # Learned from data
    answer_threshold: float = 0.70       # Learned from data
```

---

### Phase 4: Full RL (Optional, 2-3 months)

**Goal**: Implement complete Graph-R1 with policy learning

**Only pursue if**:
- Phase 1-3 show insufficient improvement
- Have GPU resources for training
- Have labeled dataset with ground truth
- Need research-grade quality

**Components**:
1. VERL framework integration
2. PPO/GRPO training loop
3. Policy network
4. Reward model
5. Training infrastructure

**Effort**: 8-12 weeks, requires ML expertise

---

## Implementation Checklist

### Phase 1: Reward Scoring ⭐
- [ ] Create `backend/rag_service/graph_rag/reward_scorer.py`
- [ ] Implement `compute_format_reward()`
- [ ] Write unit tests for format reward
- [ ] Integrate into `graph_rag_agent.py`
- [ ] Update termination logic to use format scores
- [ ] Test with sample queries
- [ ] Document reward scoring in code

### Phase 2: Data Collection
- [ ] Add interaction logging
- [ ] Run on production queries (100+ samples)
- [ ] Export logs to analysis format
- [ ] Create analysis notebook

### Phase 3: Optimization
- [ ] Analyze reward distributions
- [ ] Identify optimal thresholds
- [ ] Update agent config
- [ ] A/B test old vs new thresholds
- [ ] Measure improvement metrics

### Phase 4: Full RL (Optional)
- [ ] Evaluate ROI of full RL
- [ ] Set up VERL environment
- [ ] Create training dataset
- [ ] Implement policy network
- [ ] Train and evaluate
- [ ] Deploy trained policy

---

## Success Metrics

### Phase 1 Success Criteria
- Format reward scores correlate with response quality
- Fewer hallucinations (low format score → early termination)
- Better termination decisions (high format + answer → stop)
- User satisfaction improves

### Phase 2 Success Criteria
- 100+ logged interactions
- Clear reward distribution patterns
- Identifiable optimal thresholds

### Phase 3 Success Criteria
- 10-20% improvement in termination accuracy
- Reduced average steps to answer
- Higher user satisfaction scores

---

## Timeline

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Phase 1: Reward Scoring | 3 weeks | Week 1 | Week 3 |
| Phase 2: Data Collection | 2 weeks | Week 4 | Week 5 |
| Phase 3: Optimization | 1 week | Week 6 | Week 6 |
| **Total (Recommended)** | **6 weeks** | - | - |
| Phase 4: Full RL (Optional) | 12 weeks | TBD | TBD |

---

## Resources Needed

### Phase 1-3 (Recommended Path)
- **Developer Time**: 1 developer, 6 weeks
- **Infrastructure**: None (uses existing)
- **Data**: Production query logs
- **Cost**: Low (development time only)

### Phase 4 (Full RL)
- **Developer Time**: 1-2 ML engineers, 12 weeks
- **Infrastructure**: GPU cluster for training
- **Data**: Labeled dataset (1000+ Q&A pairs with ground truth)
- **Cost**: High (GPU compute + ML expertise)

---

## Decision Point

**Recommendation**: Start with **Phase 1-3** (6 weeks)

**Rationale**:
- Low risk, high value
- Aligns with Graph-R1 principles
- No infrastructure changes
- Can evolve to Phase 4 if needed
- Provides data to justify Phase 4 investment

**Next Action**: Create `reward_scorer.py` module (Week 1, Day 1)

