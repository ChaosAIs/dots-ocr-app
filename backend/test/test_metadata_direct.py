"""
Direct test of metadata extraction without re-indexing.
"""
import sys
import os
import asyncio
import logging

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_service.graph_rag.metadata_extractor import HierarchicalMetadataExtractor
from rag_service.vectorstore import get_vectorstore
from qdrant_client import models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def main():
    """Test metadata extraction directly on existing chunks."""
    
    source_name = "Felix Yang- Resume - 2025"
    
    logger.info(f"Fetching chunks for: {source_name}")
    
    # Get chunks from Qdrant
    vectorstore = get_vectorstore()
    
    # Search for all chunks from this source
    results = vectorstore.client.scroll(
        collection_name="dots_ocr_documents",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=source_name)
                )
            ]
        ),
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    
    points = results[0]
    logger.info(f"Found {len(points)} chunks")
    
    if not points:
        logger.error("No chunks found!")
        return 1
    
    # Convert to chunk format expected by metadata extractor
    chunks = []
    for point in points:
        chunks.append({
            "id": str(point.id),
            "content": point.payload.get("page_content", ""),
            "metadata": point.payload.get("metadata", {})
        })
    
    logger.info(f"Prepared {len(chunks)} chunks for metadata extraction")
    logger.info(f"Sample chunk content (first 200 chars): {chunks[0]['content'][:200]}")
    
    # Create extractor and run
    extractor = HierarchicalMetadataExtractor()
    
    def progress_callback(msg: str):
        logger.info(f"[Progress] {msg}")
    
    logger.info("Starting metadata extraction...")
    metadata = await extractor.extract_metadata(
        chunks=chunks,
        source_name=source_name,
        batch_size=10,
        progress_callback=progress_callback,
    )
    
    logger.info("✅ Metadata extraction complete!")
    logger.info(f"Document Type: {metadata.get('document_type')}")
    logger.info(f"Subject Name: {metadata.get('subject_name')}")
    logger.info(f"Subject Type: {metadata.get('subject_type')}")
    logger.info(f"Title: {metadata.get('title')}")
    logger.info(f"Author: {metadata.get('author')}")
    logger.info(f"Summary: {metadata.get('summary')}")
    logger.info(f"Topics: {metadata.get('topics')}")
    logger.info(f"Confidence: {metadata.get('confidence')}")
    logger.info(f"Key Entities: {len(metadata.get('key_entities', []))} entities")
    
    # Print full metadata as JSON
    import json
    logger.info(f"\nFull metadata:\n{json.dumps(metadata, indent=2)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

