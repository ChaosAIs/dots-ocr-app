#!/usr/bin/env python3
"""Debug scoring for meal receipts."""
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from db.database import get_db_session
from db.document_repository import DocumentRepository

def main():
    query_entities = ["meal receipts", "restaurant meals", "2025", "financial analysis", "expense tracking"]
    query_topics = ["financial reporting", "expense management", "data analysis", "receipt analysis", "monthly budgeting", "consumer spending"]
    
    with get_db_session() as db:
        repo = DocumentRepository(db)
        docs = repo.get_all_with_metadata()
        
        # Check Meal-04-14, Meal-06-14, Meal-06-18
        for filename in ["Meal-04-14.png", "Meal-06-14.png", "Meal-06-18.png"]:
            for doc in docs:
                if doc.filename == filename:
                    print("=" * 80)
                    print(f"File: {filename}")
                    print("=" * 80)
                    meta = doc.document_metadata
                    
                    print(f"\nSubject: {meta.get('subject_name')}")
                    print(f"Topics: {meta.get('topics')}")
                    print(f"\nSummary:")
                    print(f"  {meta.get('summary')}")
                    
                    if "hierarchical_summary" in meta:
                        print(f"\nMeta Summary:")
                        print(f"  {meta['hierarchical_summary'].get('meta_summary', 'N/A')[:200]}...")
                    
                    # Check for matches
                    summary_text = (meta.get("summary") or "").lower()
                    meta_summary = ""
                    if "hierarchical_summary" in meta and meta["hierarchical_summary"]:
                        meta_summary = (meta["hierarchical_summary"].get("meta_summary") or "").lower()
                    
                    combined = f"{summary_text} {meta_summary}"
                    
                    print(f"\nQuery entity matches in summary:")
                    for entity in query_entities:
                        if entity in combined:
                            print(f"  ✅ '{entity}' found")
                        else:
                            print(f"  ❌ '{entity}' NOT found")
                    
                    print(f"\nQuery topic matches in summary:")
                    for topic in query_topics[:3]:
                        if topic in combined:
                            print(f"  ✅ '{topic}' found")
                        else:
                            print(f"  ❌ '{topic}' NOT found")
                    
                    print()
                    break

if __name__ == "__main__":
    main()

