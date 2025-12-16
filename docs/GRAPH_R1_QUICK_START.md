# Graph-R1 Implementation Quick Start Guide

## Overview

This guide provides a quick reference for implementing Graph-R1 reward-based termination in the GraphRAG agent.

## What You'll Build

A reward scoring system that evaluates:
1. **Format Reward** (0.0-1.0): Quality of response structure
2. **Answer Reward** (0.0-1.0): Quality of answer content
3. **Reward-Based Termination**: Intelligent stopping based on scores

## Quick Implementation (3 Weeks)

### Week 1: Core Reward Scorer

**Create:** `backend/rag_service/graph_rag/reward_scorer.py`

```python
class RewardScorer:
    def compute_format_reward(self, response: str) -> Tuple[float, Dict]:
        # Returns 0.0-1.0 based on tag structure
        # 0.3 for <think> tags
        # 0.4 for action tags (<query> or <answer>)
        # 0.3 for proper structure
```

**Test:** `backend/test/test_reward_scorer.py`

### Week 2: Integration

**Modify:** `backend/rag_service/graph_rag/graph_rag_agent.py`

1. Add `RewardScorer` to `__init__`
2. Add reward thresholds to `AgentConfig`
3. Compute rewards in `_think_and_decide()`
4. Update termination logic with reward-based decisions

**Key Decision Logic:**
- Format ≥ 0.8 + Answer found → TERMINATE ✅
- Format < 0.3 → TERMINATE ⚠️ (hallucination)
- Format ≥ 0.5 + Answer quality ≥ 0.6 → TERMINATE ✅
- Has query → CONTINUE 🔄

### Week 3: Testing

**Create:** `backend/test/test_agent_with_rewards.py`

Run integration tests with mock and real LLM.

## Data Collection (2 Weeks)

### Week 4: Logging

**Create:** `backend/rag_service/graph_rag/interaction_logger.py`

Enable in production to collect reward data.

### Week 5: Analysis

**Create:** `backend/analysis/analyze_rewards.py`

Analyze 100+ interactions to optimize thresholds.

## Key Files to Create

1. `reward_scorer.py` - Core scoring logic
2. `test_reward_scorer.py` - Unit tests
3. `interaction_logger.py` - Data collection
4. `analyze_rewards.py` - Threshold optimization
5. `reward_dashboard.py` - Real-time monitoring

## Key Files to Modify

1. `graph_rag_agent.py` - Add reward scoring and termination logic
2. `graph_rag.py` - Enable interaction logging

## Configuration

```python
@dataclass
class AgentConfig:
    enable_reward_based_termination: bool = True
    format_reward_high_threshold: float = 0.8
    format_reward_low_threshold: float = 0.3
    answer_reward_threshold: float = 0.6
```

## Testing Commands

```bash
# Unit tests
python -m pytest test/test_reward_scorer.py -v

# Integration test
python test/test_agent_with_rewards.py

# Real queries
python test/test_real_queries_with_rewards.py

# Analysis
python analysis/analyze_rewards.py

# Monitoring
python analysis/reward_dashboard.py
```

## Expected Results

- **Better termination decisions** (fewer false positives/negatives)
- **Quantifiable quality metrics** (objective scoring)
- **Data-driven optimization** (learned thresholds)
- **15-20% reduction** in average steps

## Alignment with Graph-R1

✅ Reward-based decision making
✅ Format reward scoring
✅ Answer quality evaluation
✅ Data-driven threshold optimization
❌ Full RL policy training (optional Phase 4)

## Next Steps

1. Follow detailed guide in `GRAPH_R1_DETAILED_IMPLEMENTATION.md`
2. Implement Phase 1 (Weeks 1-3)
3. Collect data in Phase 2 (Weeks 4-5)
4. Monitor and iterate in Phase 3 (Week 6+)

## Support

- Detailed implementation: `docs/GRAPH_R1_DETAILED_IMPLEMENTATION.md`
- Design analysis: `docs/GRAPH_R1_DESIGN_ANALYSIS.md`
- Code comparison: `docs/GRAPH_R1_CODE_COMPARISON.md`
- Roadmap: `docs/GRAPH_R1_IMPLEMENTATION_ROADMAP.md`

