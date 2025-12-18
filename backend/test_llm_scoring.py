#!/usr/bin/env python3
"""
Test LLM-based scoring for meal receipts.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from rag_service.document_router import DocumentRouter
from rag_service.llm_service import get_llm_service
from db.database import get_db_session
from db.document_repository import DocumentRepository

def main():
    # Original query
    original_query = "please analysis my meal receipts, and generate a summary report by month for 2025 and tell me, each month how much I spent in restaurant meal spent."
    
    # Query metadata
    query_metadata = {
        "entities": ["meal receipts", "restaurant meals", "2025", "financial analysis", "expense tracking"],
        "topics": ["financial reporting", "expense management", "data analysis", "receipt analysis", "monthly budgeting", "consumer spending"],
        "document_type_hints": ["receipt", "invoice"],
        "intent": "analyze"
    }
    
    print("=" * 80)
    print("LLM-BASED SCORING TEST")
    print("=" * 80)
    print(f"\nQuery: {original_query}")
    print(f"\nQuery Entities: {query_metadata['entities']}")
    print(f"Query Topics: {query_metadata['topics'][:3]}...")
    print("\n" + "=" * 80)
    
    # Get LLM service
    llm_service = get_llm_service()
    print(f"\nLLM Service: {llm_service}")
    
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
        
        # Create router with LLM service
        router = DocumentRouter(llm_service=llm_service)
        
        print(f"Router using LLM scoring: {router.use_llm_scoring}")
        print(f"LLM min score: {router.llm_min_score}")
        print(f"LLM score ratio: {router.llm_score_ratio}")
        print("\n" + "=" * 80)
        print("SCORING DOCUMENTS...")
        print("=" * 80 + "\n")
        
        # Score documents
        scored_docs = router._score_documents_llm(query_metadata, meal_docs, original_query)
        
        print("\n" + "=" * 80)
        print("SCORING RESULTS")
        print("=" * 80 + "\n")
        
        for i, (source, score) in enumerate(scored_docs, 1):
            print(f"{i}. {source}: {score:.2f}/10.0")
        
        # Apply filtering
        print("\n" + "=" * 80)
        print("FILTERING")
        print("=" * 80)
        
        filtered = router._apply_hybrid_filtering(scored_docs)
        
        print(f"\nFiltered to {len(filtered)} documents:")
        for source, score in filtered:
            print(f"  ✅ {source}: {score:.2f}/10.0")
        
        if len(filtered) < len(scored_docs):
            print(f"\nFiltered out {len(scored_docs) - len(filtered)} documents:")
            filtered_sources = {s for s, _ in filtered}
            for source, score in scored_docs:
                if source not in filtered_sources:
                    print(f"  ❌ {source}: {score:.2f}/10.0")

if __name__ == "__main__":
    main()

