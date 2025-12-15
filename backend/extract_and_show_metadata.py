"""
Extract metadata for Felix Yang resume and display it (without saving to DB).
"""
import sys
import os
import asyncio
import logging
import json

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_service.vectorstore import get_vectorstore
from rag_service.graph_rag.metadata_extractor import HierarchicalMetadataExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def main():
    """Extract and display metadata for Felix Yang resume."""
    
    source_name = "Felix Yang- Resume - 2025"
    
    logger.info(f"Fetching chunks for: {source_name}")
    
    # Get vectorstore
    vectorstore = get_vectorstore()
    
    # Fetch all chunks for this document
    all_chunks = []
    offset = None
    
    while True:
        results = vectorstore.client.scroll(
            collection_name=vectorstore.collection_name,
            scroll_filter={
                "must": [
                    {"key": "metadata.source", "match": {"value": source_name}}
                ]
            },
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        
        if not results[0]:
            break
            
        for point in results[0]:
            all_chunks.append({
                "id": point.id,
                "page_content": point.payload.get("page_content", ""),
                "metadata": point.payload,
            })
        
        offset = results[1]
        if offset is None:
            break
    
    logger.info(f"Found {len(all_chunks)} chunks")
    
    if not all_chunks:
        logger.error("No chunks found! Make sure the document is indexed.")
        return 1
    
    # Extract metadata
    logger.info("Starting metadata extraction...")
    extractor = HierarchicalMetadataExtractor()
    
    def progress_callback(msg: str):
        logger.info(f"[Progress] {msg}")
    
    metadata = await extractor.extract_metadata(
        chunks=all_chunks,
        source_name=source_name,
        batch_size=10,
        progress_callback=progress_callback,
    )
    
    logger.info("✅ Metadata extraction complete!")
    
    # Display results
    print("\n" + "="*80)
    print("EXTRACTED METADATA FOR: Felix Yang- Resume - 2025")
    print("="*80)
    print(f"\n📄 Document Type: {metadata.get('document_type')}")
    print(f"👤 Subject Name: {metadata.get('subject_name')}")
    print(f"🏷️  Subject Type: {metadata.get('subject_type')}")
    print(f"📝 Title: {metadata.get('title')}")
    print(f"✍️  Author: {metadata.get('author')}")
    print(f"📊 Confidence: {metadata.get('confidence')}")
    print(f"\n📋 Summary:")
    print(f"   {metadata.get('summary')}")
    print(f"\n🏷️  Topics ({len(metadata.get('topics', []))}):")
    for topic in metadata.get('topics', []):
        print(f"   - {topic}")
    print(f"\n🔑 Key Entities ({len(metadata.get('key_entities', []))}):")
    for entity in metadata.get('key_entities', []):
        print(f"   - {entity.get('name')} ({entity.get('type')}) - Score: {entity.get('score')}")
    
    print(f"\n⏱️  Processing Stats:")
    stats = metadata.get('processing_stats', {})
    print(f"   - Total Chunks: {stats.get('total_chunks')}")
    print(f"   - Processed Chunks: {stats.get('processed_chunks')}")
    print(f"   - LLM Calls: {stats.get('llm_calls')}")
    print(f"   - Processing Time: {stats.get('processing_time_seconds', 0):.1f}s")
    
    print("\n" + "="*80)
    print("FULL METADATA JSON")
    print("="*80)
    print(json.dumps(metadata, indent=2))
    print("="*80)
    
    # Save to file
    output_file = "/tmp/felix_yang_metadata.json"
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"\n✅ Metadata saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

