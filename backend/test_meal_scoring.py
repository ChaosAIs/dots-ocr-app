#!/usr/bin/env python3
"""
Test script to show detailed scoring for each meal receipt file.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from rag_service.document_router import DocumentRouter
from db.database import get_db_session
from db.document_repository import DocumentRepository

def main():
    # Query metadata from the user's query
    query_metadata = {
        "entities": ["meal receipts", "restaurant meals", "2025", "financial analysis", "expense tracking"],
        "topics": ["financial reporting", "expense management", "data analysis", "receipt analysis", "monthly budgeting", "consumer spending"],
        "document_type_hints": ["receipt", "invoice"],
        "intent": "analyze"
    }
    
    print("=" * 80)
    print("MEAL RECEIPT SCORING ANALYSIS")
    print("=" * 80)
    print("\nQuery Metadata:")
    print(f"  Entities: {query_metadata['entities']}")
    print(f"  Topics: {query_metadata['topics'][:3]}...")
    print(f"  Document Type Hints: {query_metadata['document_type_hints']}")
    print("\n" + "=" * 80)
    
    # Get all documents with metadata
    with get_db_session() as db:
        repo = DocumentRepository(db)
        docs = repo.get_all_with_metadata()
        
        # Filter to meal receipts only
        meal_docs = []
        for doc in docs:
            if doc.filename.startswith("Meal-") and doc.document_metadata:
                source_name = doc.filename.rsplit('.', 1)[0] if '.' in doc.filename else doc.filename
                meal_docs.append({
                    "source_name": source_name,
                    "filename": doc.filename,
                    "metadata": doc.document_metadata,
                })
        
        # Sort by filename
        meal_docs.sort(key=lambda x: x["filename"])
        
        print(f"\nFound {len(meal_docs)} meal receipt documents\n")
        
        # Create router and score each document
        router = DocumentRouter()
        
        for doc in meal_docs:
            print("-" * 80)
            print(f"📄 {doc['source_name']}")
            print("-" * 80)
            
            meta = doc["metadata"]
            print(f"  Document Type: {meta.get('document_type', 'N/A')}")
            print(f"  Subject Name: {meta.get('subject_name', 'N/A')}")
            print(f"  Confidence: {meta.get('confidence', 0.0):.2f}")
            
            # Show key entities
            entities = meta.get('key_entities', [])
            if entities:
                print(f"  Key Entities ({len(entities)}):")
                for ent in entities[:5]:  # Show first 5
                    print(f"    - {ent.get('name', 'N/A')} ({ent.get('type', 'N/A')})")
            else:
                print(f"  Key Entities: None")
            
            # Show topics
            topics = meta.get('topics', [])
            if topics:
                print(f"  Topics: {', '.join(topics[:5])}")
            else:
                print(f"  Topics: None")
            
            # Calculate score
            score = router._calculate_match_score(query_metadata, meta)
            print(f"\n  ⭐ FINAL SCORE: {score:.2f}")
            print()
        
        # Show routing decision
        print("=" * 80)
        print("ROUTING DECISION")
        print("=" * 80)
        
        scored_docs = router._score_documents(query_metadata, meal_docs)
        
        print("\nAll documents sorted by score:")
        for i, (source, score) in enumerate(scored_docs, 1):
            print(f"  {i}. {source}: {score:.2f}")
        
        # Apply filtering
        print("\nFiltering thresholds:")
        print(f"  Min score: {router.min_score}")
        print(f"  Score ratio: {router.score_ratio} (25% of top)")
        print(f"  Max score gap: {router.max_score_gap}")
        
        if scored_docs:
            top_score = scored_docs[0][1]
            ratio_threshold = top_score * router.score_ratio
            gap_threshold = top_score - router.max_score_gap
            
            print(f"\nCalculated thresholds:")
            print(f"  Top score: {top_score:.2f}")
            print(f"  Ratio threshold: {ratio_threshold:.2f} (25% of {top_score:.2f})")
            print(f"  Gap threshold: {gap_threshold:.2f} ({top_score:.2f} - {router.max_score_gap})")
            
            print("\nFiltering results:")
            for source, score in scored_docs:
                passes_min = score >= router.min_score
                passes_ratio = score >= ratio_threshold
                passes_gap = score >= gap_threshold
                passes_all = passes_min and passes_ratio and passes_gap
                
                status = "✅ PASS" if passes_all else "❌ FILTERED OUT"
                reasons = []
                if not passes_min:
                    reasons.append(f"min_score ({score:.2f} < {router.min_score})")
                if not passes_ratio:
                    reasons.append(f"ratio ({score:.2f} < {ratio_threshold:.2f})")
                if not passes_gap:
                    reasons.append(f"gap ({score:.2f} < {gap_threshold:.2f})")
                
                reason_str = f" - {', '.join(reasons)}" if reasons else ""
                print(f"  {status} {source}: {score:.2f}{reason_str}")

if __name__ == "__main__":
    main()

