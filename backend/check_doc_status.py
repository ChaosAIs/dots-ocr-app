#!/usr/bin/env python3
"""
Quick script to check document status in the database.
"""
import sys
from db.database import get_db_session
from db.document_repository import DocumentRepository
import json

def main():
    with get_db_session() as db:
        repo = DocumentRepository(db)
        docs = repo.get_all()
        
        if not docs:
            print("No documents found in database")
            return
        
        # Sort by upload time, most recent first
        docs_sorted = sorted(docs, key=lambda d: d.upload_time, reverse=True)
        
        print(f"\n{'='*80}")
        print(f"Found {len(docs)} documents in database")
        print(f"{'='*80}\n")
        
        # Show the most recent 5 documents
        for i, doc in enumerate(docs_sorted[:5], 1):
            print(f"[{i}] {doc.filename}")
            print(f"    ID: {doc.id}")
            print(f"    Upload Time: {doc.upload_time}")
            print(f"    Convert Status: {doc.convert_status}")
            print(f"    Index Status: {doc.index_status}")
            print(f"    Indexed Chunks: {doc.indexed_chunks}")
            print(f"    Markdown Exists: {doc.markdown_exists}")
            
            if doc.indexing_details:
                print(f"\n    Indexing Details:")
                details = doc.indexing_details
                
                # Vector indexing
                if "vector_indexing" in details:
                    vi = details["vector_indexing"]
                    print(f"      Vector Indexing: {vi.get('status', 'N/A')}")
                    if "message" in vi:
                        print(f"        Message: {vi['message']}")
                
                # Metadata extraction
                if "metadata_extraction" in details:
                    me = details["metadata_extraction"]
                    print(f"      Metadata Extraction: {me.get('status', 'N/A')}")
                    if "message" in me:
                        print(f"        Message: {me['message']}")
                
                # GraphRAG indexing
                if "graphrag_indexing" in details:
                    gi = details["graphrag_indexing"]
                    print(f"      GraphRAG Indexing: {gi.get('status', 'N/A')}")
                    if "message" in gi:
                        print(f"        Message: {gi['message']}")
                    if "entities" in gi:
                        print(f"        Entities: {gi['entities']}")
                    if "relationships" in gi:
                        print(f"        Relationships: {gi['relationships']}")
            
            print()

if __name__ == "__main__":
    main()

