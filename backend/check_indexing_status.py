#!/usr/bin/env python3
"""
Check the indexing status for graph_r1.pdf document.
Shows page-level and chunk-level status updates.
"""

import os
import sys
import json
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.database import get_db_session
from db.document_repository import DocumentRepository


def format_timestamp(ts):
    """Format timestamp for display."""
    if not ts:
        return "N/A"
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def check_document_status(source_name="graph_r1"):
    """Check and display the indexing status for a document."""
    print("\n" + "="*100)
    print(f"Indexing Status Report for: {source_name}.pdf")
    print("="*100)
    print(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)
    
    with get_db_session() as db:
        repo = DocumentRepository(db)
        
        # Get document by filename
        from db.models import Document
        doc = db.query(Document).filter(
            Document.filename == f"{source_name}.pdf",
            Document.deleted_at.is_(None)
        ).first()
        
        if not doc:
            print(f"❌ Document not found: {source_name}.pdf")
            return
        
        # Document-level status
        print(f"\n📄 DOCUMENT LEVEL STATUS")
        print(f"   Filename: {doc.filename}")
        print(f"   Index Status: {doc.index_status}")
        print(f"   Total Pages: {doc.total_pages}")
        print(f"   Indexed Chunks: {doc.indexed_chunks}")
        print(f"   Created: {format_timestamp(doc.created_at)}")
        print(f"   Updated: {format_timestamp(doc.updated_at)}")
        
        # Indexing details
        indexing_details = doc.indexing_details or {}
        
        # Vector indexing status
        print(f"\n📊 PHASE 1: VECTOR INDEXING (Qdrant)")
        vector_indexing = indexing_details.get("vector_indexing", {})
        vector_status = vector_indexing.get("status", "N/A")
        vector_started = vector_indexing.get("started_at", "N/A")
        vector_completed = vector_indexing.get("completed_at", "N/A")
        
        print(f"   Status: {vector_status}")
        print(f"   Started: {vector_started}")
        print(f"   Completed: {vector_completed}")
        
        # Page-level status
        pages = vector_indexing.get("pages", {})
        print(f"\n   📑 Page-Level Status ({len(pages)} pages tracked):")
        
        success_pages = []
        failed_pages = []
        pending_pages = []
        
        for page_key in sorted(pages.keys()):
            page_info = pages[page_key]
            status = page_info.get("status", "unknown")
            chunks = page_info.get("chunks", 0)
            file_path = page_info.get("file_path", "N/A")
            
            if status == "success":
                success_pages.append((page_key, chunks, file_path))
            elif status == "failed":
                failed_pages.append((page_key, chunks, file_path))
            else:
                pending_pages.append((page_key, chunks, file_path))
        
        print(f"      ✅ Success: {len(success_pages)} pages")
        print(f"      ❌ Failed: {len(failed_pages)} pages")
        print(f"      ⏳ Pending: {len(pending_pages)} pages")
        
        # Show first 5 successful pages
        if success_pages:
            print(f"\n      Recent Successful Pages (showing first 5):")
            for page_key, chunks, file_path in success_pages[:5]:
                print(f"         {page_key}: {chunks} chunks - {os.path.basename(file_path)}")
        
        # Show all failed pages
        if failed_pages:
            print(f"\n      ❌ Failed Pages:")
            for page_key, chunks, file_path in failed_pages:
                print(f"         {page_key}: {chunks} chunks - {os.path.basename(file_path)}")
        
        # GraphRAG indexing status
        print(f"\n🕸️  PHASE 2: GRAPHRAG INDEXING (Neo4j)")
        graphrag_indexing = indexing_details.get("graphrag_indexing", {})
        graphrag_status = graphrag_indexing.get("status", "N/A")
        graphrag_started = graphrag_indexing.get("started_at", "N/A")
        graphrag_completed = graphrag_indexing.get("completed_at", "N/A")
        total_chunks = graphrag_indexing.get("total_chunks", 0)
        processed_chunks = graphrag_indexing.get("processed_chunks", 0)
        
        print(f"   Status: {graphrag_status}")
        print(f"   Started: {graphrag_started}")
        print(f"   Completed: {graphrag_completed}")
        print(f"   Progress: {processed_chunks}/{total_chunks} chunks")
        
        # Chunk-level status
        chunks_status = graphrag_indexing.get("chunks", {})
        print(f"\n   🧩 Chunk-Level Status ({len(chunks_status)} chunks tracked):")
        
        success_chunks = []
        failed_chunks = []
        pending_chunks = []
        
        for chunk_id, chunk_info in chunks_status.items():
            status = chunk_info.get("status", "unknown")
            entities = chunk_info.get("entities", 0)
            relationships = chunk_info.get("relationships", 0)
            
            if status == "success":
                success_chunks.append((chunk_id, entities, relationships))
            elif status == "failed":
                failed_chunks.append((chunk_id, entities, relationships))
            else:
                pending_chunks.append((chunk_id, entities, relationships))
        
        print(f"      ✅ Success: {len(success_chunks)} chunks")
        print(f"      ❌ Failed: {len(failed_chunks)} chunks")
        print(f"      ⏳ Pending: {len(pending_chunks)} chunks")
        
        # Show first 5 successful chunks
        if success_chunks:
            print(f"\n      Recent Successful Chunks (showing first 5):")
            for chunk_id, entities, rels in success_chunks[:5]:
                print(f"         {chunk_id[:8]}...: {entities} entities, {rels} relationships")
        
        # Show all failed chunks
        if failed_chunks:
            print(f"\n      ❌ Failed Chunks:")
            for chunk_id, entities, rels in failed_chunks:
                print(f"         {chunk_id[:8]}...: {entities} entities, {rels} relationships")
        
        # Summary
        print(f"\n" + "="*100)
        print(f"SUMMARY")
        print(f"="*100)
        print(f"Vector Indexing: {vector_status} - {len(success_pages)}/{len(pages)} pages successful")
        print(f"GraphRAG Indexing: {graphrag_status} - {len(success_chunks)}/{total_chunks} chunks successful")
        print(f"="*100 + "\n")


if __name__ == "__main__":
    import time
    
    # Check status every 5 seconds
    try:
        while True:
            check_document_status("graph_r1")
            print("\n⏳ Refreshing in 5 seconds... (Press Ctrl+C to stop)")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped by user")

