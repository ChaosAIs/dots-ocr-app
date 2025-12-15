"""
Test script for document routing functionality.

This script tests:
1. Query enhancement with metadata extraction
2. Document routing based on metadata matching
3. End-to-end search with routing
"""
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_query_enhancement():
    """Test query enhancement with metadata extraction."""
    from rag_service.rag_agent import _analyze_query_with_llm
    
    print("\n" + "="*80)
    print("TEST 1: Query Enhancement with Metadata Extraction")
    print("="*80)
    
    test_queries = [
        "What does Felix know about cloud?",
        "How do microservices communicate?",
        "Felix's work at BDO",
        "Tell me about authentication",
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 80)
        
        result = _analyze_query_with_llm(query)
        
        print(f"✅ Enhanced Query: {result['enhanced_query']}")
        print(f"👤 Entities: {result['metadata']['entities']}")
        print(f"📚 Topics: {result['metadata']['topics']}")
        print(f"📄 Document Types: {result['metadata']['document_type_hints']}")
        print(f"🎯 Intent: {result['metadata']['intent']}")


def test_document_router():
    """Test document routing with sample metadata."""
    from rag_service.document_router import DocumentRouter
    
    print("\n" + "="*80)
    print("TEST 2: Document Router")
    print("="*80)
    
    router = DocumentRouter()
    
    # Test query metadata
    test_cases = [
        {
            "name": "Felix Yang cloud query",
            "metadata": {
                "entities": ["Felix Yang", "AWS", "Azure"],
                "topics": ["cloud computing", "cloud platforms"],
                "document_type_hints": ["resume"],
                "intent": "Find Felix Yang's cloud experience"
            }
        },
        {
            "name": "Microservices architecture query",
            "metadata": {
                "entities": ["REST", "gRPC", "microservices"],
                "topics": ["microservices architecture", "distributed systems"],
                "document_type_hints": ["technical_doc", "manual"],
                "intent": "Understand microservices communication"
            }
        },
    ]
    
    for test_case in test_cases:
        print(f"\n📝 Test Case: {test_case['name']}")
        print("-" * 80)
        print(f"Query Metadata: {test_case['metadata']}")
        
        sources = router.route_query(test_case['metadata'])
        
        if sources:
            print(f"✅ Routed to {len(sources)} documents:")
            for source in sources:
                print(f"   - {source}")
        else:
            print("ℹ️  No specific routing, will search all documents")


def test_end_to_end():
    """Test end-to-end search with routing."""
    from rag_service.rag_agent import search_documents
    
    print("\n" + "="*80)
    print("TEST 3: End-to-End Search with Routing")
    print("="*80)
    
    test_queries = [
        "What cloud technologies does Felix Yang have experience with?",
        "Tell me about Felix's work experience",
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 80)
        
        try:
            result = search_documents(query)
            print(f"✅ Search completed")
            print(f"📊 Result length: {len(result)} characters")
            print(f"📄 Preview: {result[:300]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("\n🚀 Starting Document Router Tests")
    print("="*80)
    
    try:
        # Test 1: Query enhancement
        test_query_enhancement()
        
        # Test 2: Document routing
        test_document_router()
        
        # Test 3: End-to-end search
        test_end_to_end()
        
        print("\n" + "="*80)
        print("✅ All tests completed!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

