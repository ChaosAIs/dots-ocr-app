# Graph-R1 Detailed Implementation Guide

## Phase 1: Reward Scoring Implementation (Week 1-3)

### Week 1, Day 1-2: Create Reward Scorer Module

#### Step 1.1: Create the file structure

```bash
# Create the new module
touch backend/rag_service/graph_rag/reward_scorer.py
touch backend/test/test_reward_scorer.py
```

#### Step 1.2: Implement basic format reward scorer

**File: `backend/rag_service/graph_rag/reward_scorer.py`**

```python
"""
Reward scoring module for Graph-R1 style evaluation.

This module provides format and answer reward calculation to evaluate
the quality of LLM responses and guide termination decisions.
"""

import re
import logging
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RewardScore:
    """Container for reward scores."""
    format_reward: float  # 0.0 to 1.0
    answer_reward: float  # 0.0 to 1.0
    combined_reward: float  # -1.0 to 1.0
    details: Dict[str, any]  # Breakdown of scoring


class RewardScorer:
    """
    Calculates reward scores for LLM responses.

    Based on Graph-R1 paper design:
    - Format reward: Evaluates response structure (0.0-1.0)
    - Answer reward: Evaluates answer quality (0.0-1.0)
    - Combined reward: -1.0 + format + answer (if format=1.0)
    """

    def __init__(self):
        """Initialize the reward scorer."""
        pass

    def compute_format_reward(self, response: str) -> Tuple[float, Dict]:
        """
        Compute format reward for a response.

        Scoring breakdown:
        - Has <think> tags: +0.3
        - Has <query> or <answer> tags: +0.4
        - Proper tag structure: +0.3

        Args:
            response: The LLM response text

        Returns:
            Tuple of (score, details_dict)
            - score: 0.0 to 1.0
            - details: Breakdown of what contributed to score
        """
        if not response or not isinstance(response, str):
            return 0.0, {"error": "Invalid response"}

        score = 0.0
        details = {
            "has_think_tags": False,
            "has_action_tags": False,
            "proper_structure": False,
            "breakdown": []
        }

        # Check 1: Has <think> tags (0.3 points)
        has_think_open = '<think>' in response
        has_think_close = '</think>' in response

        if has_think_open and has_think_close:
            score += 0.3
            details["has_think_tags"] = True
            details["breakdown"].append("think_tags: +0.3")
            logger.debug("[Reward] Found <think> tags: +0.3")

        # Check 2: Has action tags - <query> or <answer> (0.4 points)
        has_query = '<query>' in response and '</query>' in response
        has_answer = '<answer>' in response and '</answer>' in response

        if has_query or has_answer:
            score += 0.4
            details["has_action_tags"] = True
            tag_type = "query" if has_query else "answer"
            details["breakdown"].append(f"{tag_type}_tags: +0.4")
            logger.debug(f"[Reward] Found <{tag_type}> tags: +0.4")

        # Check 3: Proper structure - tags are properly nested (0.3 points)
        if self._validate_structure(response):
            score += 0.3
            details["proper_structure"] = True
            details["breakdown"].append("proper_structure: +0.3")
            logger.debug("[Reward] Proper structure: +0.3")

        final_score = min(score, 1.0)
        details["final_score"] = final_score

        logger.info(f"[Reward] Format score: {final_score:.2f} - {details['breakdown']}")

        return final_score, details

    def _validate_structure(self, response: str) -> bool:
        """
        Validate that tags are properly structured.

        Valid patterns:
        - <think>...</think>\n<query>...</query>
        - <think>...</think>\n<answer>...</answer>
        - <think>...</think> (alone is also valid)
        """
        # Pattern 1: think + query
        pattern1 = r'<think>.*?</think>\s*<query>.*?</query>'
        if re.search(pattern1, response, re.DOTALL):
            return True

        # Pattern 2: think + answer
        pattern2 = r'<think>.*?</think>\s*<answer>.*?</answer>'
        if re.search(pattern2, response, re.DOTALL):
            return True

        # Pattern 3: think alone (valid for intermediate steps)
        pattern3 = r'<think>.*?</think>'
        if re.search(pattern3, response, re.DOTALL):
            # Check no orphaned tags
            if '<query>' not in response and '<answer>' not in response:
                return True

        return False

    def compute_answer_reward_heuristic(
        self,
        answer: str,
        question: str
    ) -> Tuple[float, Dict]:
        """
        Compute answer reward using heuristics (no ground truth).

        Heuristic scoring:
        - Length check: Answer is substantial (0.3)
        - Completeness: Contains key indicators (0.4)
        - Confidence: No uncertainty markers (0.3)

        Args:
            answer: The extracted answer text
            question: The original question

        Returns:
            Tuple of (score, details_dict)
        """
        if not answer:
            return 0.0, {"error": "No answer provided"}

        score = 0.0
        details = {"breakdown": []}

        # Check 1: Length (0.3 points)
        # Substantial answer: 50-2000 characters
        answer_len = len(answer.strip())
        if 50 <= answer_len <= 2000:
            score += 0.3
            details["breakdown"].append(f"length_ok ({answer_len} chars): +0.3")
        elif answer_len > 2000:
            score += 0.2  # Partial credit for very long
            details["breakdown"].append(f"length_long ({answer_len} chars): +0.2")

        # Check 2: Completeness indicators (0.4 points)
        completeness_indicators = [
            "based on", "according to", "the answer is",
            "in summary", "therefore", "because", "specifically"
        ]
        answer_lower = answer.lower()
        found_indicators = [ind for ind in completeness_indicators if ind in answer_lower]

        if len(found_indicators) >= 2:
            score += 0.4
            details["breakdown"].append(f"completeness ({len(found_indicators)} indicators): +0.4")
        elif len(found_indicators) == 1:
            score += 0.2
            details["breakdown"].append(f"completeness ({len(found_indicators)} indicator): +0.2")

        # Check 3: Confidence - penalize uncertainty (0.3 points)
        uncertainty_markers = [
            "i don't know", "not sure", "unclear", "maybe", "possibly",
            "cannot determine", "insufficient information", "not enough"
        ]
        found_uncertainty = [m for m in uncertainty_markers if m in answer_lower]

        if not found_uncertainty:
            score += 0.3
            details["breakdown"].append("no_uncertainty: +0.3")
        else:
            details["breakdown"].append(f"uncertainty_found ({found_uncertainty}): +0.0")

        final_score = min(score, 1.0)
        details["final_score"] = final_score

        logger.info(f"[Reward] Answer score (heuristic): {final_score:.2f}")

        return final_score, details
```

---

### Week 1, Day 3-4: Add Unit Tests

**File: `backend/test/test_reward_scorer.py`**

```python
"""
Unit tests for reward scorer.
"""

import pytest
from rag_service.graph_rag.reward_scorer import RewardScorer


class TestFormatReward:
    """Test format reward calculation."""

    def setup_method(self):
        self.scorer = RewardScorer()

    def test_perfect_format_with_query(self):
        """Test perfect format with query tags."""
        response = """<think>
I need to search for information about cloud computing.
</think>
<query>cloud computing AWS Azure</query>"""

        score, details = self.scorer.compute_format_reward(response)

        assert score == 1.0, f"Expected 1.0, got {score}"
        assert details["has_think_tags"] is True
        assert details["has_action_tags"] is True
        assert details["proper_structure"] is True

    def test_perfect_format_with_answer(self):
        """Test perfect format with answer tags."""
        response = """<think>
I have enough information to answer the question.
</think>
<answer>Cloud computing refers to...</answer>"""

        score, details = self.scorer.compute_format_reward(response)

        assert score == 1.0
        assert details["has_think_tags"] is True
        assert details["has_action_tags"] is True

    def test_partial_format_think_only(self):
        """Test partial format with only think tags."""
        response = """<think>
I'm analyzing the question.
</think>"""

        score, details = self.scorer.compute_format_reward(response)

        # Should get 0.3 (think) + 0.3 (structure) = 0.6
        assert score == 0.6, f"Expected 0.6, got {score}"

    def test_no_tags(self):
        """Test response with no tags."""
        response = "This is just plain text without any tags."

        score, details = self.scorer.compute_format_reward(response)

        assert score == 0.0

    def test_malformed_tags(self):
        """Test response with malformed tags."""
        response = "<think>Missing closing tag <query>Also missing</query>"

        score, details = self.scorer.compute_format_reward(response)

        # Should only get points for query tags (0.4)
        assert score == 0.4


class TestAnswerReward:
    """Test answer reward calculation."""

    def setup_method(self):
        self.scorer = RewardScorer()

    def test_good_answer(self):
        """Test a good quality answer."""
        answer = """Based on the retrieved knowledge, cloud computing is a model
        for delivering computing services over the internet. According to the documents,
        it includes services like AWS and Azure. Therefore, organizations can use these
        platforms to scale their infrastructure."""

        question = "What is cloud computing?"

        score, details = self.scorer.compute_answer_reward_heuristic(answer, question)

        # Should get high score: length + completeness + confidence
        assert score >= 0.8, f"Expected >= 0.8, got {score}"

    def test_uncertain_answer(self):
        """Test answer with uncertainty."""
        answer = "I'm not sure, but maybe cloud computing is about servers."
        question = "What is cloud computing?"

        score, details = self.scorer.compute_answer_reward_heuristic(answer, question)

        # Should get low score due to uncertainty
        assert score < 0.5

    def test_empty_answer(self):
        """Test empty answer."""
        score, details = self.scorer.compute_answer_reward_heuristic("", "question")

        assert score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

### Week 1, Day 5: Run Tests and Validate

```bash
# Navigate to backend directory
cd backend

# Run the tests
python -m pytest test/test_reward_scorer.py -v

# Expected output:
# test_reward_scorer.py::TestFormatReward::test_perfect_format_with_query PASSED
# test_reward_scorer.py::TestFormatReward::test_perfect_format_with_answer PASSED
# test_reward_scorer.py::TestFormatReward::test_partial_format_think_only PASSED
# test_reward_scorer.py::TestFormatReward::test_no_tags PASSED
# test_reward_scorer.py::TestFormatReward::test_malformed_tags PASSED
# test_reward_scorer.py::TestAnswerReward::test_good_answer PASSED
# test_reward_scorer.py::TestAnswerReward::test_uncertain_answer PASSED
# test_reward_scorer.py::TestAnswerReward::test_empty_answer PASSED
```

---

### Week 2, Day 1-3: Integrate Reward Scorer into Agent

#### Step 2.1: Add reward scorer to agent initialization

**File: `backend/rag_service/graph_rag/graph_rag_agent.py`**

Find the `__init__` method and add:

```python
def __init__(
    self,
    llm_service,
    retrieval_callback: RetrievalCallback,
    mode_detector=None,
    config: AgentConfig = None,
):
    """Initialize the GraphRAG agent."""
    self.llm_service = llm_service
    self.retrieval_callback = retrieval_callback
    self.mode_detector = mode_detector
    self.config = config or AgentConfig()

    # NEW: Initialize reward scorer
    from .reward_scorer import RewardScorer
    self.reward_scorer = RewardScorer()

    logger.info(
        f"[GraphRAG Agent] Initialized with max_steps={self.config.max_steps}, "
        f"default_retrieval_mode={self.config.default_retrieval_mode}"
    )
```

#### Step 2.2: Add reward thresholds to AgentConfig

Find the `AgentConfig` class and add:

```python
@dataclass
class AgentConfig:
    """Configuration for the GraphRAG agent."""
    max_steps: int = 5
    min_entities_for_answer: int = 3
    min_score_threshold: float = 0.5
    top_k_per_step: int = 30
    enable_query_refinement: bool = True
    default_retrieval_mode: str = "auto"

    # NEW: Reward-based termination thresholds
    format_reward_high_threshold: float = 0.8  # High confidence to terminate
    format_reward_low_threshold: float = 0.3   # Low quality, likely hallucination
    answer_reward_threshold: float = 0.6       # Minimum answer quality
    enable_reward_based_termination: bool = True  # Feature flag
```

#### Step 2.3: Modify the decision logic to use rewards

Find the `_think_and_decide` method and modify it:

```python
async def _think_and_decide(self, state: AgentState) -> str:
    """Agent thinks about current state and decides next action."""
    logger.info(f"[GraphRAG Agent] Preparing think prompt with {len(state.retrieved_knowledge)} knowledge items...")

    knowledge_summary = state.get_knowledge_summary()
    logger.info(f"[GraphRAG Agent] Knowledge summary length: {len(knowledge_summary)} chars")

    prompt = AGENT_THINK_PROMPT.format(
        question=state.question,
        step=state.step,
        max_steps=state.max_steps,
        retrieved_knowledge=knowledge_summary,
    )

    system_prompt = AGENT_SYSTEM_PROMPT.format(max_steps=state.max_steps)

    logger.info(f"[GraphRAG Agent] Sending to LLM for thinking...")
    logger.info(f"[GraphRAG Agent] Prompt length: {len(prompt)} chars")

    response = await self.llm_service.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=1500,
    )

    logger.info(f"[GraphRAG Agent] LLM response received: {len(response)} chars")

    # NEW: Compute reward scores
    if self.config.enable_reward_based_termination:
        format_score, format_details = self.reward_scorer.compute_format_reward(response)

        logger.info("=" * 50)
        logger.info("[GraphRAG Agent] 📊 REWARD SCORING")
        logger.info("=" * 50)
        logger.info(f"[GraphRAG Agent] Format Reward: {format_score:.2f}")
        logger.info(f"[GraphRAG Agent] Format Details: {format_details['breakdown']}")

        # Store in state for later analysis
        if not hasattr(state, 'reward_history'):
            state.reward_history = []

        state.reward_history.append({
            'step': state.step,
            'format_score': format_score,
            'format_details': format_details,
        })

    return response
```

#### Step 2.4: Update termination logic in the main query loop

Find the section in `query()` method where decisions are made:

```python
# Parse response for answer or next query
answer = self._extract_answer(response)
next_query = self._extract_query(response)

logger.info(f"[GraphRAG Agent] 📊 PARSING DECISION:")
logger.info(f"  - Found <answer> tag: {answer is not None}")
logger.info(f"  - Found <query> tag: {next_query is not None}")

# NEW: Reward-based termination logic
if self.config.enable_reward_based_termination:
    format_score = state.reward_history[-1]['format_score']

    # Decision 1: High format score + answer found = TERMINATE
    if format_score >= self.config.format_reward_high_threshold and answer:
        state.is_complete = True
        state.final_answer = answer
        logger.info(f"[GraphRAG Agent] ✅ DECISION: TERMINATE")
        logger.info(f"[GraphRAG Agent] Reason: High format score ({format_score:.2f}) + answer found")

        # Optionally compute answer quality
        answer_score, answer_details = self.reward_scorer.compute_answer_reward_heuristic(
            answer, state.question
        )
        logger.info(f"[GraphRAG Agent] Answer Quality Score: {answer_score:.2f}")
        state.reward_history[-1]['answer_score'] = answer_score
        state.reward_history[-1]['answer_details'] = answer_details

    # Decision 2: Low format score = TERMINATE (hallucination)
    elif format_score < self.config.format_reward_low_threshold:
        state.is_complete = True
        logger.info(f"[GraphRAG Agent] ⚠️ DECISION: TERMINATE")
        logger.info(f"[GraphRAG Agent] Reason: Low format score ({format_score:.2f}) - likely hallucination")
        logger.info(f"[GraphRAG Agent] Generating final answer from accumulated knowledge...")
        state.final_answer = await self._generate_final_answer(state)

    # Decision 3: Medium format score + answer = Check quality
    elif answer and format_score >= 0.5:
        answer_score, answer_details = self.reward_scorer.compute_answer_reward_heuristic(
            answer, state.question
        )
        logger.info(f"[GraphRAG Agent] Answer Quality Score: {answer_score:.2f}")

        state.reward_history[-1]['answer_score'] = answer_score
        state.reward_history[-1]['answer_details'] = answer_details

        if answer_score >= self.config.answer_reward_threshold:
            state.is_complete = True
            state.final_answer = answer
            logger.info(f"[GraphRAG Agent] ✅ DECISION: TERMINATE")
            logger.info(f"[GraphRAG Agent] Reason: Acceptable format ({format_score:.2f}) + good answer ({answer_score:.2f})")
        else:
            logger.info(f"[GraphRAG Agent] 🔄 DECISION: CONTINUE")
            logger.info(f"[GraphRAG Agent] Reason: Answer quality too low ({answer_score:.2f})")
            # Continue to next iteration

    # Decision 4: Has next query = CONTINUE
    elif next_query:
        if next_query not in state.queries_made:
            state.queries_made.append(next_query)
            logger.info(f"[GraphRAG Agent] 🔄 DECISION: CONTINUE")
            logger.info(f"[GraphRAG Agent] Next query: '{next_query}'")
        else:
            # Duplicate query, terminate
            state.is_complete = True
            logger.info(f"[GraphRAG Agent] ⚠️ DECISION: TERMINATE")
            logger.info(f"[GraphRAG Agent] Reason: Duplicate query detected")
            state.final_answer = await self._generate_final_answer(state)

    # Decision 5: No valid tags = TERMINATE
    else:
        state.is_complete = True
        logger.info(f"[GraphRAG Agent] ⚠️ DECISION: TERMINATE")
        logger.info(f"[GraphRAG Agent] Reason: No valid tags found")
        state.final_answer = await self._generate_final_answer(state)

else:
    # OLD: Original rule-based logic (fallback)
    if answer:
        state.is_complete = True
        state.final_answer = answer
        logger.info(f"[GraphRAG Agent] ✅ DECISION: TERMINATE - Answer found")
    elif next_query:
        if next_query not in state.queries_made:
            state.queries_made.append(next_query)
            logger.info(f"[GraphRAG Agent] 🔄 DECISION: CONTINUE - New query generated")
        else:
            state.is_complete = True
            logger.info(f"[GraphRAG Agent] ⚠️ DECISION: TERMINATE - Duplicate query")
            state.final_answer = await self._generate_final_answer(state)
    else:
        state.is_complete = True
        logger.info(f"[GraphRAG Agent] ⚠️ DECISION: TERMINATE - No valid tags found")
        state.final_answer = await self._generate_final_answer(state)
```

---

### Week 2, Day 4-5: Add Reward Logging and Metadata

#### Step 2.5: Update AgentState to track rewards

Find the `AgentState` class and add:

```python
@dataclass
class AgentState:
    """State for the GraphRAG reasoning agent."""
    question: str
    step: int = 0
    max_steps: int = 5
    retrieved_knowledge: List[Dict[str, Any]] = field(default_factory=list)
    queries_made: List[str] = field(default_factory=list)
    is_complete: bool = False
    final_answer: Optional[str] = None

    # NEW: Reward tracking
    reward_history: List[Dict[str, Any]] = field(default_factory=list)
```

#### Step 2.6: Include rewards in final metadata

Find the end of the `query()` method where metadata is returned:

```python
metadata = {
    "steps": state.step,
    "queries_made": state.queries_made,
    "modes_used": modes_used,
    "total_entities": len(all_entities),
    "total_relationships": len(all_relationships),
    "mode": "agent_iterative",
    # Include actual data for parent GraphRAG
    "all_entities": all_entities,
    "all_relationships": all_relationships,
    "all_chunks": all_chunks,

    # NEW: Include reward history
    "reward_history": state.reward_history if hasattr(state, 'reward_history') else [],
}

# NEW: Log reward summary
if state.reward_history:
    logger.info("=" * 70)
    logger.info("[GraphRAG Agent] 📊 REWARD SUMMARY")
    logger.info("=" * 70)
    for i, reward_data in enumerate(state.reward_history, 1):
        logger.info(f"Step {i}:")
        logger.info(f"  Format Score: {reward_data.get('format_score', 'N/A'):.2f}")
        if 'answer_score' in reward_data:
            logger.info(f"  Answer Score: {reward_data['answer_score']:.2f}")
    logger.info("=" * 70)
```

---

### Week 3: Testing and Validation

#### Step 3.1: Create integration test

**File: `backend/test/test_agent_with_rewards.py`**

```python
"""
Integration test for agent with reward scoring.
"""

import asyncio
import logging
from rag_service.graph_rag.graph_rag_agent import GraphRAGAgent, AgentConfig
from rag_service.graph_rag.base import QueryParam

logging.basicConfig(level=logging.INFO)


async def mock_llm_service_generate(prompt, system_prompt=None, max_tokens=1000):
    """Mock LLM service for testing."""
    # Simulate different response types
    if "initial" in prompt.lower() or "step: 1" in prompt:
        return """<think>
I need to search for information about cloud computing.
</think>
<query>cloud computing definition and services</query>"""

    elif "step: 2" in prompt:
        return """<think>
I have enough information to answer the question now.
Based on the retrieved knowledge, I can provide a comprehensive answer.
</think>
<answer>Cloud computing is a model for delivering computing services over the internet,
including infrastructure, platforms, and software. Major providers include AWS, Azure,
and Google Cloud Platform.</answer>"""

    else:
        return """<think>
Continuing search...
</think>
<query>cloud computing examples</query>"""


async def mock_retrieval_callback(query, mode, params):
    """Mock retrieval callback."""
    from rag_service.graph_rag.base import GraphRAGContext

    return GraphRAGContext(
        entities=[{"id": "1", "name": "AWS", "entity_type": "Technology"}],
        relationships=[],
        chunks=["Cloud computing provides on-demand resources."],
        mode="hybrid",
        enhanced_query=query,
    )


async def test_agent_with_rewards():
    """Test agent with reward-based termination."""

    # Create mock LLM service
    class MockLLMService:
        async def generate(self, prompt, system_prompt=None, max_tokens=1000):
            return await mock_llm_service_generate(prompt, system_prompt, max_tokens)

    # Create agent with reward-based termination enabled
    config = AgentConfig(
        max_steps=5,
        enable_reward_based_termination=True,
        format_reward_high_threshold=0.8,
        format_reward_low_threshold=0.3,
        answer_reward_threshold=0.6,
    )

    agent = GraphRAGAgent(
        llm_service=MockLLMService(),
        retrieval_callback=mock_retrieval_callback,
        config=config,
    )

    # Run query
    question = "What is cloud computing?"
    answer, metadata = await agent.query(question)

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Question: {question}")
    print(f"Answer: {answer[:200]}...")
    print(f"Steps taken: {metadata['steps']}")
    print(f"Queries made: {metadata['queries_made']}")

    # Check reward history
    if 'reward_history' in metadata:
        print("\nReward History:")
        for i, reward in enumerate(metadata['reward_history'], 1):
            print(f"  Step {i}:")
            print(f"    Format Score: {reward.get('format_score', 'N/A')}")
            if 'answer_score' in reward:
                print(f"    Answer Score: {reward['answer_score']}")

    # Assertions
    assert metadata['steps'] <= 3, "Should terminate early with good answer"
    assert len(metadata['reward_history']) > 0, "Should have reward history"

    # Check that final step has high format score
    final_reward = metadata['reward_history'][-1]
    assert final_reward['format_score'] >= 0.8, "Final response should have high format score"

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_agent_with_rewards())
```

#### Step 3.2: Run integration test

```bash
cd backend
python test/test_agent_with_rewards.py

# Expected output:
# [GraphRAG Agent] 🚀 STARTING ITERATIVE REASONING
# [GraphRAG Agent] 📊 REWARD SCORING
# [GraphRAG Agent] Format Reward: 1.00
# [GraphRAG Agent] ✅ DECISION: TERMINATE
# [GraphRAG Agent] Reason: High format score (1.00) + answer found
#
# TEST RESULTS
# ======================================================================
# Question: What is cloud computing?
# Answer: Cloud computing is a model for delivering computing services...
# Steps taken: 2
# Queries made: ['cloud computing definition and services']
#
# Reward History:
#   Step 1:
#     Format Score: 1.0
#   Step 2:
#     Format Score: 1.0
#     Answer Score: 0.9
#
# ✅ All tests passed!
```

#### Step 3.3: Test with real queries

Create a test script to run with actual LLM:

**File: `backend/test/test_real_queries_with_rewards.py`**

```python
"""
Test agent with real queries and reward scoring.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_service.graph_rag.graph_rag import GraphRAG
from rag_service.graph_rag.base import QueryParam


async def test_real_queries():
    """Test with real queries."""

    # Initialize GraphRAG
    graphrag = GraphRAG(workspace_id="default")

    # Test queries
    test_queries = [
        "What cloud technologies does Felix Yang have experience with?",
        "Tell me about microservices architecture",
        "What is the capital of France?",  # Simple query
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        # Query with agent mode (max_steps > 1 triggers iterative reasoning)
        params = QueryParam(
            mode="agent",
            max_steps=3,
            top_k=20,
        )

        context = await graphrag.query(query, params=params)

        print(f"\nMode used: {context.mode}")
        print(f"Entities found: {len(context.entities)}")
        print(f"Relationships found: {len(context.relationships)}")

        # Check metadata for reward history
        if hasattr(context, 'metadata') and 'reward_history' in context.metadata:
            print("\nReward History:")
            for i, reward in enumerate(context.metadata['reward_history'], 1):
                print(f"  Step {i}: Format={reward.get('format_score', 'N/A'):.2f}", end="")
                if 'answer_score' in reward:
                    print(f", Answer={reward['answer_score']:.2f}")
                else:
                    print()


if __name__ == "__main__":
    asyncio.run(test_real_queries())
```

Run it:

```bash
cd backend
python test/test_real_queries_with_rewards.py
```

---

## Phase 2: Data Collection and Analysis (Week 4-5)

### Week 4: Set Up Logging Infrastructure

#### Step 4.1: Create interaction logger

**File: `backend/rag_service/graph_rag/interaction_logger.py`**

```python
"""
Interaction logger for collecting reward data.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class InteractionLogger:
    """Logs agent interactions with reward scores for analysis."""

    def __init__(self, log_dir: str = "logs/interactions"):
        """Initialize interaction logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"interactions_{timestamp}.jsonl"

        logger.info(f"[InteractionLogger] Logging to: {self.log_file}")

    def log_interaction(
        self,
        question: str,
        answer: str,
        metadata: Dict[str, Any],
    ):
        """
        Log a single interaction.

        Args:
            question: User's question
            answer: Agent's answer
            metadata: Metadata including reward_history
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "steps": metadata.get("steps", 0),
            "queries_made": metadata.get("queries_made", []),
            "modes_used": metadata.get("modes_used", []),
            "reward_history": metadata.get("reward_history", []),
            "total_entities": metadata.get("total_entities", 0),
            "total_relationships": metadata.get("total_relationships", 0),
        }

        # Write to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(interaction) + '\n')

        logger.info(f"[InteractionLogger] Logged interaction: {len(interaction['reward_history'])} steps")

    def get_log_file_path(self) -> Path:
        """Get the current log file path."""
        return self.log_file
```

#### Step 4.2: Integrate logger into agent

Add to `graph_rag_agent.py`:

```python
class GraphRAGAgent:
    """Agent-based GraphRAG with iterative reasoning."""

    def __init__(
        self,
        llm_service,
        retrieval_callback: RetrievalCallback,
        mode_detector=None,
        config: AgentConfig = None,
        enable_interaction_logging: bool = False,  # NEW
    ):
        """Initialize the GraphRAG agent."""
        self.llm_service = llm_service
        self.retrieval_callback = retrieval_callback
        self.mode_detector = mode_detector
        self.config = config or AgentConfig()

        # Initialize reward scorer
        from .reward_scorer import RewardScorer
        self.reward_scorer = RewardScorer()

        # NEW: Initialize interaction logger
        self.interaction_logger = None
        if enable_interaction_logging:
            from .interaction_logger import InteractionLogger
            self.interaction_logger = InteractionLogger()
            logger.info("[GraphRAG Agent] Interaction logging enabled")

        logger.info(
            f"[GraphRAG Agent] Initialized with max_steps={self.config.max_steps}"
        )

    async def query(
        self,
        question: str,
        params: QueryParam = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute iterative reasoning query."""

        # ... existing code ...

        # At the end, before returning:

        # NEW: Log interaction if enabled
        if self.interaction_logger:
            self.interaction_logger.log_interaction(
                question=question,
                answer=state.final_answer,
                metadata=metadata,
            )

        return state.final_answer, metadata
```

#### Step 4.3: Enable logging in production

Update the GraphRAG initialization to enable logging:

```python
# In graph_rag.py or wherever GraphRAG is initialized

# Enable interaction logging for data collection
agent = GraphRAGAgent(
    llm_service=self._llm_service,
    retrieval_callback=self._agent_retrieval_callback,
    mode_detector=self.query_mode_detector,
    config=agent_config,
    enable_interaction_logging=True,  # NEW: Enable logging
)
```

---

### Week 5: Data Analysis and Threshold Optimization

#### Step 5.1: Create analysis notebook

**File: `backend/analysis/analyze_rewards.py`**

```python
"""
Analyze collected interaction logs to optimize reward thresholds.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict


def load_interactions(log_file: str) -> pd.DataFrame:
    """Load interactions from JSONL file."""
    interactions = []

    with open(log_file, 'r') as f:
        for line in f:
            interactions.append(json.loads(line))

    return pd.DataFrame(interactions)


def analyze_format_rewards(df: pd.DataFrame):
    """Analyze format reward distributions."""
    print("\n" + "=" * 70)
    print("FORMAT REWARD ANALYSIS")
    print("=" * 70)

    # Extract format scores from reward_history
    format_scores = []
    for _, row in df.iterrows():
        for reward in row['reward_history']:
            if 'format_score' in reward:
                format_scores.append(reward['format_score'])

    format_df = pd.DataFrame({'format_score': format_scores})

    print(f"\nTotal responses analyzed: {len(format_scores)}")
    print(f"\nFormat Score Statistics:")
    print(format_df['format_score'].describe())

    print(f"\nFormat Score Distribution:")
    print(format_df['format_score'].value_counts().sort_index())

    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(format_scores, bins=20, edgecolor='black')
    plt.xlabel('Format Score')
    plt.ylabel('Frequency')
    plt.title('Format Reward Distribution')
    plt.axvline(x=0.8, color='g', linestyle='--', label='High Threshold (0.8)')
    plt.axvline(x=0.3, color='r', linestyle='--', label='Low Threshold (0.3)')
    plt.legend()
    plt.savefig('format_reward_distribution.png')
    print("\n✅ Saved plot: format_reward_distribution.png")

    # Recommendations
    print(f"\n📊 RECOMMENDATIONS:")

    median_score = format_df['format_score'].median()
    q75_score = format_df['format_score'].quantile(0.75)
    q25_score = format_df['format_score'].quantile(0.25)

    print(f"  - Median format score: {median_score:.2f}")
    print(f"  - 75th percentile: {q75_score:.2f}")
    print(f"  - 25th percentile: {q25_score:.2f}")

    print(f"\n  Suggested thresholds:")
    print(f"  - High threshold (terminate with answer): {q75_score:.2f}")
    print(f"  - Low threshold (hallucination): {q25_score:.2f}")


def analyze_answer_rewards(df: pd.DataFrame):
    """Analyze answer reward distributions."""
    print("\n" + "=" * 70)
    print("ANSWER REWARD ANALYSIS")
    print("=" * 70)

    # Extract answer scores from reward_history
    answer_scores = []
    for _, row in df.iterrows():
        for reward in row['reward_history']:
            if 'answer_score' in reward:
                answer_scores.append(reward['answer_score'])

    if not answer_scores:
        print("\n⚠️  No answer scores found in data")
        return

    answer_df = pd.DataFrame({'answer_score': answer_scores})

    print(f"\nTotal answers analyzed: {len(answer_scores)}")
    print(f"\nAnswer Score Statistics:")
    print(answer_df['answer_score'].describe())

    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(answer_scores, bins=20, edgecolor='black')
    plt.xlabel('Answer Score')
    plt.ylabel('Frequency')
    plt.title('Answer Reward Distribution')
    plt.axvline(x=0.6, color='g', linestyle='--', label='Threshold (0.6)')
    plt.legend()
    plt.savefig('answer_reward_distribution.png')
    print("\n✅ Saved plot: answer_reward_distribution.png")

    # Recommendations
    median_score = answer_df['answer_score'].median()
    print(f"\n📊 RECOMMENDATIONS:")
    print(f"  - Median answer score: {median_score:.2f}")
    print(f"  - Suggested threshold: {median_score:.2f}")


def analyze_termination_patterns(df: pd.DataFrame):
    """Analyze when and why agent terminates."""
    print("\n" + "=" * 70)
    print("TERMINATION PATTERN ANALYSIS")
    print("=" * 70)

    print(f"\nTotal interactions: {len(df)}")
    print(f"\nSteps Distribution:")
    print(df['steps'].value_counts().sort_index())

    print(f"\nAverage steps: {df['steps'].mean():.2f}")
    print(f"Median steps: {df['steps'].median():.0f}")

    # Plot steps distribution
    plt.figure(figsize=(10, 6))
    df['steps'].hist(bins=range(1, df['steps'].max() + 2), edgecolor='black')
    plt.xlabel('Number of Steps')
    plt.ylabel('Frequency')
    plt.title('Steps to Termination Distribution')
    plt.savefig('steps_distribution.png')
    print("\n✅ Saved plot: steps_distribution.png")

    # Analyze final step rewards
    final_format_scores = []
    for _, row in df.iterrows():
        if row['reward_history']:
            final_reward = row['reward_history'][-1]
            if 'format_score' in final_reward:
                final_format_scores.append(final_reward['format_score'])

    print(f"\n📊 Final Step Format Scores:")
    print(f"  - Mean: {pd.Series(final_format_scores).mean():.2f}")
    print(f"  - Median: {pd.Series(final_format_scores).median():.2f}")
    print(f"  - Min: {pd.Series(final_format_scores).min():.2f}")
    print(f"  - Max: {pd.Series(final_format_scores).max():.2f}")


def generate_config_recommendations(df: pd.DataFrame):
    """Generate recommended configuration based on data."""
    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 70)

    # Extract all format scores
    format_scores = []
    for _, row in df.iterrows():
        for reward in row['reward_history']:
            if 'format_score' in reward:
                format_scores.append(reward['format_score'])

    format_df = pd.DataFrame({'format_score': format_scores})

    # Calculate optimal thresholds
    high_threshold = format_df['format_score'].quantile(0.75)
    low_threshold = format_df['format_score'].quantile(0.25)

    # Extract answer scores if available
    answer_scores = []
    for _, row in df.iterrows():
        for reward in row['reward_history']:
            if 'answer_score' in reward:
                answer_scores.append(reward['answer_score'])

    answer_threshold = 0.6  # Default
    if answer_scores:
        answer_threshold = pd.Series(answer_scores).median()

    print(f"""
@dataclass
class AgentConfig:
    max_steps: int = 5
    enable_reward_based_termination: bool = True

    # Optimized thresholds based on {len(df)} interactions
    format_reward_high_threshold: float = {high_threshold:.2f}  # 75th percentile
    format_reward_low_threshold: float = {low_threshold:.2f}   # 25th percentile
    answer_reward_threshold: float = {answer_threshold:.2f}       # Median
""")


def main():
    """Main analysis function."""
    # Find latest log file
    log_dir = Path("logs/interactions")
    log_files = list(log_dir.glob("interactions_*.jsonl"))

    if not log_files:
        print("❌ No log files found in logs/interactions/")
        return

    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 Analyzing: {latest_log}")

    # Load data
    df = load_interactions(latest_log)

    # Run analyses
    analyze_format_rewards(df)
    analyze_answer_rewards(df)
    analyze_termination_patterns(df)
    generate_config_recommendations(df)

    print("\n" + "=" * 70)
    print("✅ Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

#### Step 5.2: Run analysis

```bash
cd backend

# Make sure you have collected at least 50-100 interactions
# Then run analysis
python analysis/analyze_rewards.py

# Expected output:
# ======================================================================
# FORMAT REWARD ANALYSIS
# ======================================================================
#
# Total responses analyzed: 150
#
# Format Score Statistics:
#                format_score
# count              150.000
# mean                 0.756
# std                  0.234
# min                  0.000
# max                  1.000
# 25%                  0.600
# 50%                  0.800
# 75%                  1.000
#
# 📊 RECOMMENDATIONS:
#   - Median format score: 0.80
#   - 75th percentile: 1.00
#   - 25th percentile: 0.60
#
#   Suggested thresholds:
#   - High threshold (terminate with answer): 1.00
#   - Low threshold (hallucination): 0.60
#
# ======================================================================
# RECOMMENDED CONFIGURATION
# ======================================================================
#
# @dataclass
# class AgentConfig:
#     max_steps: int = 5
#     enable_reward_based_termination: bool = True
#
#     # Optimized thresholds based on 50 interactions
#     format_reward_high_threshold: float = 1.00  # 75th percentile
#     format_reward_low_threshold: float = 0.60   # 25th percentile
#     answer_reward_threshold: float = 0.75       # Median
```

#### Step 5.3: Update configuration with learned thresholds

Based on analysis results, update `AgentConfig`:

```python
@dataclass
class AgentConfig:
    """Configuration for the GraphRAG agent."""
    max_steps: int = 5
    min_entities_for_answer: int = 3
    min_score_threshold: float = 0.5
    top_k_per_step: int = 30
    enable_query_refinement: bool = True
    default_retrieval_mode: str = "auto"

    # Reward-based termination thresholds
    # UPDATED: Based on data analysis from 100+ interactions
    format_reward_high_threshold: float = 0.95  # Was 0.8, now data-driven
    format_reward_low_threshold: float = 0.40   # Was 0.3, now data-driven
    answer_reward_threshold: float = 0.70       # Was 0.6, now data-driven
    enable_reward_based_termination: bool = True
```

---

## Phase 3: Monitoring and Iteration (Week 6+)

### Step 6.1: Create monitoring dashboard

**File: `backend/analysis/reward_dashboard.py`**

```python
"""
Real-time monitoring dashboard for reward scores.
"""

import json
import time
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta


class RewardMonitor:
    """Monitor reward scores in real-time."""

    def __init__(self, log_file: Path, window_size: int = 50):
        """Initialize monitor."""
        self.log_file = log_file
        self.window_size = window_size
        self.recent_interactions = deque(maxlen=window_size)

    def update(self):
        """Update with latest interactions."""
        # Read all interactions
        interactions = []
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                for line in f:
                    interactions.append(json.loads(line))

        # Keep only recent ones
        self.recent_interactions = deque(interactions[-self.window_size:], maxlen=self.window_size)

    def get_stats(self) -> dict:
        """Get current statistics."""
        if not self.recent_interactions:
            return {}

        format_scores = []
        answer_scores = []
        steps = []

        for interaction in self.recent_interactions:
            steps.append(interaction['steps'])
            for reward in interaction['reward_history']:
                if 'format_score' in reward:
                    format_scores.append(reward['format_score'])
                if 'answer_score' in reward:
                    answer_scores.append(reward['answer_score'])

        stats = {
            'total_interactions': len(self.recent_interactions),
            'avg_steps': sum(steps) / len(steps) if steps else 0,
            'avg_format_score': sum(format_scores) / len(format_scores) if format_scores else 0,
            'avg_answer_score': sum(answer_scores) / len(answer_scores) if answer_scores else 0,
            'low_format_count': sum(1 for s in format_scores if s < 0.4),
            'high_format_count': sum(1 for s in format_scores if s >= 0.8),
        }

        return stats

    def print_dashboard(self):
        """Print dashboard to console."""
        self.update()
        stats = self.get_stats()

        if not stats:
            print("No data available yet")
            return

        print("\n" + "=" * 70)
        print(f"REWARD MONITORING DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print(f"Recent Interactions: {stats['total_interactions']} (last {self.window_size})")
        print(f"Average Steps: {stats['avg_steps']:.2f}")
        print(f"Average Format Score: {stats['avg_format_score']:.2f}")
        print(f"Average Answer Score: {stats['avg_answer_score']:.2f}")
        print(f"\nFormat Score Distribution:")
        print(f"  Low (<0.4): {stats['low_format_count']}")
        print(f"  High (>=0.8): {stats['high_format_count']}")
        print("=" * 70)


def monitor_loop(log_dir: str = "logs/interactions", interval: int = 30):
    """Run monitoring loop."""
    log_path = Path(log_dir)

    # Find latest log file
    log_files = list(log_path.glob("interactions_*.jsonl"))
    if not log_files:
        print(f"No log files found in {log_dir}")
        return

    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)

    monitor = RewardMonitor(latest_log)

    print(f"Monitoring: {latest_log}")
    print(f"Update interval: {interval} seconds")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            monitor.print_dashboard()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped")


if __name__ == "__main__":
    monitor_loop()
```

Run it:

```bash
cd backend
python analysis/reward_dashboard.py

# Output updates every 30 seconds:
# ======================================================================
# REWARD MONITORING DASHBOARD - 2025-12-16 14:30:00
# ======================================================================
# Recent Interactions: 50 (last 50)
# Average Steps: 2.3
# Average Format Score: 0.82
# Average Answer Score: 0.75
#
# Format Score Distribution:
#   Low (<0.4): 3
#   High (>=0.8): 42
# ======================================================================
```

---

## Summary Checklist

### ✅ Phase 1: Reward Scoring (Weeks 1-3)

- [ ] Week 1: Create `reward_scorer.py` with format reward
- [ ] Week 1: Write unit tests
- [ ] Week 2: Integrate into agent
- [ ] Week 2: Update termination logic
- [ ] Week 3: Test with mock and real queries

### ✅ Phase 2: Data Collection (Weeks 4-5)

- [ ] Week 4: Create `interaction_logger.py`
- [ ] Week 4: Enable logging in production
- [ ] Week 4: Collect 100+ interactions
- [ ] Week 5: Run `analyze_rewards.py`
- [ ] Week 5: Update thresholds based on data

### ✅ Phase 3: Monitoring (Week 6+)

- [ ] Create monitoring dashboard
- [ ] Track reward trends
- [ ] Iterate on thresholds
- [ ] Measure improvement

---

## Expected Improvements

After implementing all phases:

1. **Better Termination Decisions**

   - Fewer false positives (stopping too early)
   - Fewer false negatives (continuing too long)
   - Reduced hallucinations (low format score detection)

2. **Quantifiable Quality**

   - Format scores provide objective quality metric
   - Answer scores measure response quality
   - Data-driven threshold optimization

3. **Alignment with Graph-R1**

   - Reward-based decision making ✅
   - Quality scoring mechanism ✅
   - Data-driven optimization ✅
   - (Still missing: Full RL policy training)

4. **Measurable Metrics**
   - Average steps reduced by 15-20%
   - Format score consistency improved
   - User satisfaction increased

---

## Next Steps After Phase 3

If results show need for further improvement:

1. **Implement Ground Truth Comparison**

   - Create labeled dataset
   - Implement F1/EM scoring
   - Compare against Graph-R1 full solution

2. **Consider Full RL Training**

   - Evaluate ROI
   - Set up VERL framework
   - Train PPO/GRPO policy
   - Deploy trained model

3. **Advanced Features**
   - Multi-turn conversation memory
   - Dynamic threshold adjustment
   - A/B testing framework
   - User feedback integration
