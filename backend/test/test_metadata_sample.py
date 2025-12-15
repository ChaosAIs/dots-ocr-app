"""
Test metadata extraction with sample resume data.
"""
import sys
import os
import asyncio
import logging
import json

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_service.graph_rag.metadata_extractor import HierarchicalMetadataExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Sample resume chunks (using page_content field like the indexer does)
SAMPLE_CHUNKS = [
    {
        "id": "chunk1",
        "page_content": "# Felix Yang\n\n**Senior Software Engineer**\n\nEmail: felix.yang@example.com | Phone: (555) 123-4567 | LinkedIn: linkedin.com/in/felixyang",
        "metadata": {"page": 1}
    },
    {
        "id": "chunk2",
        "page_content": "## Professional Summary\n\nHighly skilled software engineer with 10+ years of experience in full-stack development, cloud architecture, and AI/ML systems. Expert in Python, JavaScript, and distributed systems.",
        "metadata": {"page": 1}
    },
    {
        "id": "chunk3",
        "page_content": "## Work Experience\n\n### Senior Software Engineer | Tech Corp | 2020-Present\n\n- Led development of microservices architecture serving 10M+ users\n- Implemented GraphRAG system for document intelligence\n- Reduced API latency by 60% through optimization",
        "metadata": {"page": 1}
    },
    {
        "id": "chunk4",
        "page_content": "### Software Engineer | StartupCo | 2018-2020\n\n- Built real-time data processing pipeline using Apache Kafka\n- Developed React-based dashboard for analytics\n- Mentored junior developers",
        "metadata": {"page": 2}
    },
    {
        "id": "chunk5",
        "page_content": "## Skills\n\n**Programming Languages:** Python, JavaScript, TypeScript, Java, Go\n\n**Frameworks:** React, FastAPI, Django, Spring Boot\n\n**Cloud & DevOps:** AWS, Docker, Kubernetes, Terraform\n\n**Databases:** PostgreSQL, MongoDB, Redis, Neo4j",
        "metadata": {"page": 2}
    },
    {
        "id": "chunk6",
        "page_content": "## Education\n\n**Master of Science in Computer Science**\nStanford University | 2016-2018\n\n**Bachelor of Science in Software Engineering**\nUC Berkeley | 2012-2016",
        "metadata": {"page": 2}
    },
]


async def main():
    """Test metadata extraction with sample data."""
    
    logger.info("Testing metadata extraction with sample resume data")
    
    # Create extractor
    extractor = HierarchicalMetadataExtractor()
    
    def progress_callback(msg: str):
        logger.info(f"[Progress] {msg}")
    
    # Run extraction
    logger.info("Starting metadata extraction...")
    metadata = await extractor.extract_metadata(
        chunks=SAMPLE_CHUNKS,
        source_name="Felix Yang- Resume - 2025",
        batch_size=3,
        progress_callback=progress_callback,
    )
    
    logger.info("✅ Metadata extraction complete!")
    logger.info(f"\nExtracted Metadata:")
    logger.info(f"==================")
    logger.info(f"Document Type: {metadata.get('document_type')}")
    logger.info(f"Subject Name: {metadata.get('subject_name')}")
    logger.info(f"Subject Type: {metadata.get('subject_type')}")
    logger.info(f"Title: {metadata.get('title')}")
    logger.info(f"Author: {metadata.get('author')}")
    logger.info(f"Summary: {metadata.get('summary')}")
    logger.info(f"Topics: {metadata.get('topics')}")
    logger.info(f"Confidence: {metadata.get('confidence')}")
    logger.info(f"Key Entities ({len(metadata.get('key_entities', []))}):")
    for entity in metadata.get('key_entities', [])[:5]:
        logger.info(f"  - {entity.get('name')} ({entity.get('type')}) - Score: {entity.get('score')}")
    
    # Print full metadata as JSON
    logger.info(f"\nFull Metadata JSON:")
    logger.info(f"===================")
    print(json.dumps(metadata, indent=2))
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

