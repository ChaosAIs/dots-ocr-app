#!/usr/bin/env python3
"""
Test script for ContextAnalyzer to verify pronoun detection and entity extraction.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat_service.context_analyzer import ContextAnalyzer

def test_context_analyzer():
    """Test the ContextAnalyzer with sample conversation."""
    
    analyzer = ContextAnalyzer()
    
    # Simulate conversation history
    history = [
        {"role": "user", "content": "List all the expense and purchases I did in 2025 and sort them by month"},
        {"role": "assistant", "content": """Here is a list of your purchases in 2025 based on the provided invoice data:

1. **Logitech M720 Triathlon Wireless Optical Mouse - Black**  
   - Price: $39.99  
   - Date: December 3, 2025  

2. **Monitor**  
   - Price: $159.00  
   - Date: February 10, 2025  

... (9 items total)

Total Purchases in 2025: 9 items
"""}
    ]
    
    # Test message with implicit reference
    test_message = "how much I spent totally"
    
    print("=" * 80)
    print("Testing ContextAnalyzer")
    print("=" * 80)
    print(f"\nConversation History ({len(history)} messages):")
    for i, msg in enumerate(history):
        print(f"  {i+1}. {msg['role']}: {msg['content'][:100]}...")
    
    print(f"\nTest Message: '{test_message}'")
    print("\nAnalyzing...")
    
    # Analyze the message
    analysis = analyzer.analyze_message(test_message, history)
    
    print("\n" + "=" * 80)
    print("Analysis Results:")
    print("=" * 80)
    print(f"Has Pronouns: {analysis.get('has_pronouns', False)}")
    print(f"Detected Pronouns: {analysis.get('detected_pronouns', [])}")
    print(f"Resolved Message: {analysis.get('resolved_message', test_message)}")
    print(f"\nExtracted Entities:")
    for entity_type, values in analysis.get('entities', {}).items():
        print(f"  {entity_type}: {values}")
    print(f"\nExtracted Topics: {analysis.get('topics', [])}")
    print("=" * 80)
    
    # Test another message
    test_message2 = "can you offer me a summary"
    print(f"\n\nTest Message 2: '{test_message2}'")
    print("Analyzing...")
    
    analysis2 = analyzer.analyze_message(test_message2, history)
    
    print("\n" + "=" * 80)
    print("Analysis Results 2:")
    print("=" * 80)
    print(f"Has Pronouns: {analysis2.get('has_pronouns', False)}")
    print(f"Detected Pronouns: {analysis2.get('detected_pronouns', [])}")
    print(f"Resolved Message: {analysis2.get('resolved_message', test_message2)}")
    print(f"\nExtracted Entities:")
    for entity_type, values in analysis2.get('entities', {}).items():
        print(f"  {entity_type}: {values}")
    print(f"\nExtracted Topics: {analysis2.get('topics', [])}")
    print("=" * 80)

if __name__ == "__main__":
    test_context_analyzer()

