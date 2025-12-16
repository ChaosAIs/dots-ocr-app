"""
Simple integration test for agent with reward scoring (no external dependencies).
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_service.graph_rag.graph_rag_agent import GraphRAGAgent, AgentConfig
from rag_service.graph_rag.base import QueryParam

logging.basicConfig(level=logging.INFO)


class MockContext:
    """Mock context object."""
    def __init__(self, entities, relationships, chunks):
        self.entities = entities
        self.relationships = relationships
        self.chunks = chunks


class MockLLMService:
    """Mock LLM service for testing."""
    
    def __init__(self):
        self.call_count = 0
    
    async def generate(self, prompt, system_prompt=None, max_tokens=1000):
        """Mock LLM generation."""
        self.call_count += 1
        
        # Simulate different responses based on call count
        if self.call_count == 1:
            # Initial query generation
            return "<query>cloud computing definition and services</query>"
        
        elif self.call_count == 2:
            # First iteration - continue searching
            return """<think>
I need to search for more specific information about cloud computing providers.
</think>
<query>major cloud computing providers AWS Azure GCP</query>"""
        
        elif self.call_count == 3:
            # Second iteration - provide answer
            return """<think>
I now have enough information to answer the question comprehensively.
Based on the retrieved knowledge, I can provide a complete answer.
</think>
<answer>Cloud computing is a model for delivering computing services over the internet, 
including infrastructure, platforms, and software. Major providers include AWS (Amazon Web Services), 
Microsoft Azure, and Google Cloud Platform. These services provide on-demand access to computing 
resources such as servers, storage, databases, and networking capabilities.</answer>"""
        
        else:
            # Fallback
            return "<answer>Cloud computing provides on-demand computing resources.</answer>"


async def mock_retrieval_callback(query, mode, params):
    """Mock retrieval callback."""
    return MockContext(
        entities=[
            {"id": "1", "name": "AWS", "entity_type": "Technology", "description": "Amazon Web Services"},
            {"id": "2", "name": "Azure", "entity_type": "Technology", "description": "Microsoft Azure"},
        ],
        relationships=[],
        chunks=["Cloud computing provides on-demand resources over the internet."],
    )


async def test_agent_with_rewards():
    """Test agent with reward-based termination."""
    
    print("\n" + "=" * 70)
    print("TESTING AGENT WITH REWARD-BASED TERMINATION")
    print("=" * 70 + "\n")
    
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
    print(f"\nAnswer: {answer[:300]}...")
    print(f"\nSteps taken: {metadata['steps']}")
    print(f"Queries made: {metadata['queries_made']}")
    
    # Check reward history
    if 'reward_history' in metadata:
        print("\nReward History:")
        for i, reward in enumerate(metadata['reward_history'], 1):
            print(f"  Step {i}:")
            print(f"    Format Score: {reward.get('format_score', 'N/A'):.2f}")
            if 'answer_score' in reward:
                print(f"    Answer Score: {reward['answer_score']:.2f}")
    
    # Assertions
    print("\n" + "=" * 70)
    print("VALIDATIONS")
    print("=" * 70)
    
    # Should terminate early with good answer
    assert metadata['steps'] <= 3, f"Expected <= 3 steps, got {metadata['steps']}"
    print("✅ Terminated within expected steps")
    
    # Should have reward history
    assert len(metadata['reward_history']) > 0, "Should have reward history"
    print("✅ Reward history recorded")
    
    # Check that final step has high format score
    final_reward = metadata['reward_history'][-1]
    assert final_reward['format_score'] >= 0.8, f"Expected format score >= 0.8, got {final_reward['format_score']}"
    print(f"✅ Final format score is high: {final_reward['format_score']:.2f}")
    
    # Should have answer score in final step
    assert 'answer_score' in final_reward, "Final step should have answer score"
    print(f"✅ Answer quality evaluated: {final_reward['answer_score']:.2f}")
    
    # Answer should be non-empty
    assert answer and len(answer) > 50, "Answer should be substantial"
    print("✅ Answer is substantial")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✅")
    print("=" * 70 + "\n")
    
    return True


async def test_low_format_score():
    """Test termination with low format score."""
    
    print("\n" + "=" * 70)
    print("TESTING LOW FORMAT SCORE TERMINATION")
    print("=" * 70 + "\n")
    
    class BadFormatLLM:
        """LLM that produces bad format."""
        async def generate(self, prompt, system_prompt=None, max_tokens=1000):
            # Return response without proper tags
            return "I think cloud computing is about internet services but I'm not sure."
    
    config = AgentConfig(
        max_steps=5,
        enable_reward_based_termination=True,
        format_reward_low_threshold=0.3,
    )
    
    agent = GraphRAGAgent(
        llm_service=BadFormatLLM(),
        retrieval_callback=mock_retrieval_callback,
        config=config,
    )
    
    answer, metadata = await agent.query("What is cloud computing?")
    
    print(f"Steps taken: {metadata['steps']}")
    print(f"Reward history: {metadata['reward_history']}")
    
    # Should terminate quickly due to low format score
    assert metadata['steps'] <= 2, "Should terminate quickly with bad format"
    print("✅ Terminated quickly with low format score")
    
    return True


async def main():
    """Run all tests."""
    try:
        await test_agent_with_rewards()
        await test_low_format_score()
        print("\n🎉 All integration tests passed!\n")
        return True
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

