#!/usr/bin/env python3
"""
Test script to verify that auto-indexing skips successfully indexed pages.

This script:
1. Checks the granular indexing status for documents
2. Verifies that _is_document_fully_indexed_in_db() checks granular status
3. Verifies that _get_successfully_indexed_page_files() returns correct pages
"""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.database import get_db_session
from db.document_repository import DocumentRepository
from rag_service.indexer import (
    _is_document_fully_indexed_in_db,
    _get_successfully_indexed_page_files,
    _get_failed_page_files,
)


def test_granular_status_check():
    """Test that granular status checking works correctly."""
    print("\n" + "="*80)
    print("Testing Granular Status Checking")
    print("="*80)
    
    with get_db_session() as db:
        repo = DocumentRepository(db)
        
        # Get all documents
        docs = repo.get_all()
        
        if not docs:
            print("❌ No documents found in database")
            return
        
        print(f"\n✅ Found {len(docs)} documents in database\n")
        
        for doc in docs:
            if doc.deleted_at:
                continue
            
            source_name = os.path.splitext(doc.filename)[0]
            
            print(f"\n📄 Document: {doc.filename}")
            print(f"   Source Name: {source_name}")
            print(f"   Index Status: {doc.index_status}")
            print(f"   Total Pages: {doc.total_pages}")
            print(f"   Indexed Chunks: {doc.indexed_chunks}")
            
            # Check indexing details
            indexing_details = doc.indexing_details or {}
            vector_indexing = indexing_details.get("vector_indexing", {})
            vector_status = vector_indexing.get("status", "N/A")
            pages = vector_indexing.get("pages", {})
            
            print(f"   Vector Status: {vector_status}")
            print(f"   Pages Tracked: {len(pages)}")
            
            # Count page statuses
            success_count = sum(1 for p in pages.values() if p.get("status") == "success")
            failed_count = sum(1 for p in pages.values() if p.get("status") == "failed")
            pending_count = sum(1 for p in pages.values() if p.get("status") == "pending")
            
            print(f"   Pages Success: {success_count}")
            print(f"   Pages Failed: {failed_count}")
            print(f"   Pages Pending: {pending_count}")
            
            # Test _is_document_fully_indexed_in_db
            is_fully_indexed = _is_document_fully_indexed_in_db(source_name)
            print(f"\n   🔍 _is_document_fully_indexed_in_db(): {is_fully_indexed}")
            
            # Test _get_successfully_indexed_page_files
            successful_pages = _get_successfully_indexed_page_files(source_name)
            print(f"   ✅ Successfully indexed pages: {len(successful_pages)}")
            if successful_pages and len(successful_pages) <= 5:
                for page_path in list(successful_pages)[:5]:
                    print(f"      - {page_path}")
            
            # Test _get_failed_page_files
            failed_pages = _get_failed_page_files(source_name)
            print(f"   ❌ Failed pages: {len(failed_pages)}")
            if failed_pages:
                for page_path in list(failed_pages)[:5]:
                    print(f"      - {page_path}")
            
            # Verify logic
            if failed_count > 0 and is_fully_indexed:
                print(f"\n   ⚠️  WARNING: Document has {failed_count} failed pages but is marked as fully indexed!")
            elif failed_count > 0 and not is_fully_indexed:
                print(f"\n   ✅ CORRECT: Document has {failed_count} failed pages and is NOT marked as fully indexed")
            elif failed_count == 0 and is_fully_indexed:
                print(f"\n   ✅ CORRECT: Document has no failed pages and is marked as fully indexed")
            elif failed_count == 0 and not is_fully_indexed:
                print(f"\n   ℹ️  INFO: Document has no failed pages but is not fully indexed (may be in progress)")


def main():
    """Main test function."""
    print("\n" + "="*80)
    print("Auto-Indexing Skip Test")
    print("="*80)
    
    try:
        test_granular_status_check()
        
        print("\n" + "="*80)
        print("Test Complete")
        print("="*80)
        print("\nKey Points:")
        print("1. Documents with failed pages should NOT be marked as fully indexed")
        print("2. _get_successfully_indexed_page_files() should return pages with status='success'")
        print("3. _get_failed_page_files() should return pages with status='failed'")
        print("4. Auto-indexing on startup will skip successfully indexed pages")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

