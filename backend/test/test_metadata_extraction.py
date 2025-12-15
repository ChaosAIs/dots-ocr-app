"""
Test script for metadata extraction on Felix Yang's resume.
"""
import sys
import os
import asyncio
import logging

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_service.indexer import reindex_document
from db.database import get_db_session
from db.document_repository import DocumentRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Test metadata extraction by re-indexing Felix Yang's resume."""
    
    source_name = "Felix Yang- Resume - 2025"
    filename = "Felix Yang- Resume - 2025.pdf"
    
    logger.info(f"Starting re-index of: {source_name}")
    logger.info("This will:")
    logger.info("  1. Delete existing Qdrant embeddings")
    logger.info("  2. Re-index to Qdrant (Phase 1)")
    logger.info("  3. Extract metadata (Phase 1.5) ← NEW!")
    logger.info("  4. Extract entities to Neo4j (Phase 2, background)")
    logger.info("")
    
    # Re-index the document
    try:
        chunks = reindex_document(source_name)
        logger.info(f"✅ Re-indexing complete: {chunks} chunks indexed")
        
        # Wait a bit for metadata extraction to complete
        logger.info("Waiting for metadata extraction to complete...")
        import time
        time.sleep(5)
        
        # Check if metadata was saved
        with get_db_session() as db:
            repo = DocumentRepository(db)
            doc = repo.get_by_filename(filename)
            
            if doc and doc.document_metadata:
                logger.info("✅ Metadata extraction successful!")
                logger.info(f"Document Type: {doc.document_metadata.get('document_type')}")
                logger.info(f"Subject Name: {doc.document_metadata.get('subject_name')}")
                logger.info(f"Subject Type: {doc.document_metadata.get('subject_type')}")
                logger.info(f"Title: {doc.document_metadata.get('title')}")
                logger.info(f"Author: {doc.document_metadata.get('author')}")
                logger.info(f"Summary: {doc.document_metadata.get('summary')}")
                logger.info(f"Topics: {doc.document_metadata.get('topics')}")
                logger.info(f"Confidence: {doc.document_metadata.get('confidence')}")
                logger.info(f"Key Entities: {len(doc.document_metadata.get('key_entities', []))} entities")
                logger.info(f"Processing Time: {doc.document_metadata.get('processing_stats', {}).get('processing_time_seconds')}s")
            else:
                logger.warning("⚠️  Metadata not found - may still be processing")
                
    except Exception as e:
        logger.error(f"❌ Error during re-indexing: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

