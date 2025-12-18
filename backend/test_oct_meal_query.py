#!/usr/bin/env python3
"""
Test script to debug why October 2025 meal receipt query is failing.
This script will:
1. Run the same query through the RAG pipeline
2. Show exactly what context is retrieved
3. Test if LLM can answer correctly with explicit instructions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag_service.rag_agent import search_documents
from rag_service.llm_service import get_llm_service
from langchain_core.messages import HumanMessage, SystemMessage

def test_october_meal_query():
    """Test the October 2025 meal query."""

    query = "do we have meal in 2025 oct?"

    print("=" * 80)
    print(f"TESTING QUERY: {query}")
    print("=" * 80)

    # Call the search_documents tool
    result = search_documents.invoke({"query": query})

    print("\n" + "=" * 80)
    print("SEARCH RESULT (first 2000 chars):")
    print("=" * 80)
    print(result[:2000])
    print("\n... (truncated)")

    # Check if the result mentions October or 10/14/2025
    print("\n" + "=" * 80)
    print("DATE ANALYSIS:")
    print("=" * 80)

    import re
    dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', result)
    if dates:
        print(f"✅ Found {len(dates)} dates in result: {dates}")

        # Check for October dates (month = 10)
        oct_dates = [d for d in dates if d.startswith('10/') or d.startswith('10-')]
        if oct_dates:
            print(f"✅ Found {len(oct_dates)} October 2025 dates: {oct_dates}")
        else:
            print("❌ No October dates found")
    else:
        print("❌ No dates found in result")

    # Now test if LLM can answer correctly with the context
    print("\n" + "=" * 80)
    print("TESTING LLM RESPONSE:")
    print("=" * 80)

    llm_service = get_llm_service()
    llm = llm_service.get_chat_model(temperature=0.1)

    system_msg = f"""You are a helpful assistant. Answer the user's question based on the provided context.

IMPORTANT DATE INTERPRETATION:
- Dates may appear in various formats: "10/14/2025", "2025-10-14", "October 14, 2025", "Oct 14", etc.
- When a user asks about "October" or "Oct", match it with dates starting with "10/" or containing "October"
- Month 10 = October

Context:
{result}
"""

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=query)
    ]

    response = llm.invoke(messages)
    print(f"LLM Response: {response.content}")

if __name__ == "__main__":
    test_october_meal_query()

