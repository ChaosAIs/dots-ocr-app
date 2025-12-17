"""
Test script for LLM-based query analyzer.
Tests greeting detection and complexity assessment.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat_service.query_analyzer import analyze_query_with_llm


def test_query_analyzer():
    """Test the LLM-based query analyzer with various queries."""
    
    test_cases = [
        # Greetings
        ("Hello!", "Simple greeting"),
        ("Hi there", "Casual greeting"),
        ("Good morning", "Polite greeting"),
        ("Hey, how are you?", "Greeting with question"),
        
        # Simple questions (max_steps = 1)
        ("What is Graph-R1?", "Simple definition question"),
        ("Define OCR", "Direct definition"),
        
        # Moderate questions (max_steps = 2-3)
        ("How does Graph-R1 work?", "Explanation question"),
        ("Explain the OCR process", "Process explanation"),
        ("What are the benefits of using Graph-R1?", "Benefits question"),
        
        # Complex questions (max_steps = 3-4)
        ("Compare Graph-R1 and traditional RAG", "Comparison question"),
        ("What is the relationship between Graph-R1 and knowledge graphs?", "Relationship question"),
        
        # Very complex questions (max_steps = 4-5)
        ("Explain how Graph-R1 relates to knowledge graphs, vector databases, and LLMs", "Multi-concept analysis"),
        ("How do entities, relationships, and embeddings interact in Graph-R1?", "Deep multi-part question"),
    ]
    
    print("=" * 80)
    print("Testing LLM-based Query Analyzer")
    print("=" * 80)
    print()
    
    results = []
    for query, description in test_cases:
        print(f"Query: '{query}'")
        print(f"Description: {description}")
        
        try:
            analysis = analyze_query_with_llm(query)
            
            print(f"  ✓ Is Greeting: {analysis.is_greeting}")
            print(f"  ✓ Max Steps: {analysis.max_steps}")
            print(f"  ✓ Reasoning: {analysis.reasoning}")
            
            results.append({
                "query": query,
                "is_greeting": analysis.is_greeting,
                "max_steps": analysis.max_steps,
                "reasoning": analysis.reasoning,
                "success": True
            })
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "query": query,
                "success": False,
                "error": str(e)
            })
        
        print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    successful = sum(1 for r in results if r.get("success", False))
    total = len(results)
    
    print(f"Total tests: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print()
    
    # Greeting detection summary
    greetings = [r for r in results if r.get("success") and r.get("is_greeting")]
    print(f"Detected as greetings: {len(greetings)}")
    for r in greetings:
        print(f"  - '{r['query']}' (max_steps={r['max_steps']})")
    print()
    
    # Complexity distribution
    complexity_dist = {}
    for r in results:
        if r.get("success") and not r.get("is_greeting"):
            steps = r.get("max_steps", 0)
            complexity_dist[steps] = complexity_dist.get(steps, 0) + 1
    
    print("Complexity distribution (non-greetings):")
    for steps in sorted(complexity_dist.keys()):
        count = complexity_dist[steps]
        print(f"  max_steps={steps}: {count} queries")
    
    print()
    print("=" * 80)
    
    return successful == total


if __name__ == "__main__":
    success = test_query_analyzer()
    sys.exit(0 if success else 1)

