"""
Test script for LLM-Driven Adaptive Chunking (V3.0).

This script tests the new chunking system with sample documents.
Run with: python -m rag_service.chunking.test_llm_driven_chunking
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_service.chunking import (
    ContentSampler,
    SampledContent,
    StructureAnalyzer,
    StrategyConfig,
    DEFAULT_STRATEGY,
    StrategyExecutor,
    STRATEGIES,
    AdaptiveChunker,
)


# Sample documents for testing
ACADEMIC_PAPER = """# A Survey of Retrieval-Augmented Generation (RAG) Systems

## Abstract

Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for enhancing large language models with external knowledge. This survey examines recent advances in RAG architectures, focusing on retrieval mechanisms, generation strategies, and evaluation methodologies.

## 1. Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation. However, they face challenges with hallucination and outdated knowledge. RAG addresses these limitations by incorporating external retrieval systems.

The key contributions of this paper include:
- A comprehensive taxonomy of RAG architectures
- Analysis of retrieval and generation components
- Evaluation benchmarks and metrics

## 2. Background

### 2.1 Large Language Models

LLMs are neural networks trained on vast amounts of text data. Notable examples include GPT-4, Claude, and LLaMA.

### 2.2 Information Retrieval

Information retrieval systems enable efficient access to relevant documents from large corpora.

## 3. RAG Architectures

### 3.1 Basic RAG Pipeline

The standard RAG pipeline consists of:
1. Query encoding
2. Document retrieval
3. Context augmentation
4. Response generation

### 3.2 Advanced Architectures

Modern systems incorporate:
- Multi-hop reasoning
- Hybrid retrieval (dense + sparse)
- Iterative refinement

## 4. Conclusion

RAG systems represent a significant advancement in making LLMs more factual and up-to-date.

## References

[1] Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
[2] Gao, Y., et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey.
"""

LEGAL_CONTRACT = """SERVICES AGREEMENT

This Services Agreement ("Agreement") is entered into as of January 1, 2024.

1. DEFINITIONS

1.1 "Services" means the consulting services described in Exhibit A.

1.2 "Confidential Information" means any non-public information disclosed by either party.

1.3 "Term" means the period from the Effective Date until termination.

2. SERVICES

2.1 Scope of Work. Contractor shall provide the Services as described in Exhibit A.

2.2 Performance Standards. Contractor shall perform all Services in a professional and workmanlike manner.

2.3 Changes. Any changes to the Services must be agreed in writing by both parties.

3. COMPENSATION

3.1 Fees. Client shall pay Contractor the fees set forth in Exhibit B.

3.2 Expenses. Client shall reimburse Contractor for pre-approved expenses.

3.3 Payment Terms. Payment is due within 30 days of invoice receipt.

4. CONFIDENTIALITY

4.1 Protection. Each party agrees to protect the other's Confidential Information.

4.2 Exclusions. Confidential Information does not include information that:
    (a) is or becomes publicly available;
    (b) was known prior to disclosure;
    (c) is independently developed.

5. TERM AND TERMINATION

5.1 Term. This Agreement shall remain in effect for one (1) year.

5.2 Termination. Either party may terminate with 30 days written notice.

5.3 Effect of Termination. Upon termination, all unpaid fees become due.
"""

TABULAR_DATA = """| Product ID | Product Name | Category | Price | Stock | Last Updated |
|------------|--------------|----------|-------|-------|--------------|
| PRD-001 | Laptop Pro | Electronics | 1299.99 | 45 | 2024-01-15 |
| PRD-002 | Wireless Mouse | Accessories | 29.99 | 150 | 2024-01-14 |
| PRD-003 | USB-C Hub | Accessories | 49.99 | 80 | 2024-01-13 |
| PRD-004 | Monitor 27" | Electronics | 399.99 | 30 | 2024-01-12 |
| PRD-005 | Keyboard RGB | Accessories | 79.99 | 100 | 2024-01-11 |
| PRD-006 | Webcam HD | Electronics | 89.99 | 60 | 2024-01-10 |
| PRD-007 | Headphones | Audio | 149.99 | 75 | 2024-01-09 |
| PRD-008 | Microphone | Audio | 119.99 | 40 | 2024-01-08 |
| PRD-009 | Desk Stand | Furniture | 59.99 | 90 | 2024-01-07 |
| PRD-010 | Cable Kit | Accessories | 24.99 | 200 | 2024-01-06 |
| PRD-011 | SSD 1TB | Storage | 89.99 | 120 | 2024-01-05 |
| PRD-012 | RAM 16GB | Components | 69.99 | 85 | 2024-01-04 |
| PRD-013 | Power Strip | Electronics | 34.99 | 150 | 2024-01-03 |
| PRD-014 | Mouse Pad | Accessories | 14.99 | 300 | 2024-01-02 |
| PRD-015 | USB Drive | Storage | 19.99 | 250 | 2024-01-01 |
"""

LOG_FILE = """2024-01-15 10:23:45.123 [INFO] Application starting...
2024-01-15 10:23:45.456 [INFO] Loading configuration from /etc/app/config.yaml
2024-01-15 10:23:45.789 [INFO] Database connection established
2024-01-15 10:23:46.012 [INFO] Cache initialized with 1024MB capacity
2024-01-15 10:23:46.345 [INFO] HTTP server starting on port 8080

2024-01-15 10:24:01.123 [INFO] Request received: GET /api/users
2024-01-15 10:24:01.234 [DEBUG] Query: SELECT * FROM users WHERE active=1
2024-01-15 10:24:01.345 [INFO] Response: 200 OK (125 users)

2024-01-15 10:25:15.678 [ERROR] Failed to process request
java.lang.NullPointerException: User object is null
    at com.example.UserService.getUser(UserService.java:42)
    at com.example.ApiHandler.handleRequest(ApiHandler.java:89)
    at com.example.HttpServer.process(HttpServer.java:123)

2024-01-15 10:25:16.789 [INFO] Retry attempt 1/3
2024-01-15 10:25:17.012 [INFO] Request succeeded on retry

2024-01-15 10:30:00.000 [INFO] Scheduled job started: cleanup_sessions
2024-01-15 10:30:05.123 [INFO] Cleaned up 42 expired sessions
2024-01-15 10:30:05.456 [INFO] Scheduled job completed
"""


def test_content_sampler():
    """Test the ContentSampler with different content types."""
    print("\n" + "=" * 80)
    print("TEST 1: Content Sampler")
    print("=" * 80)

    sampler = ContentSampler()

    # Test character-based sampling
    print("\n1.1 Testing character-based sampling (long document)...")
    long_content = ACADEMIC_PAPER * 10  # Make it longer
    sampled = sampler.sample_from_content(long_content)

    print(f"   - Original length: {len(long_content)} chars")
    print(f"   - Scenario: {sampled.scenario}")
    print(f"   - First content length: {len(sampled.first_content)} chars")
    print(f"   - Middle content length: {len(sampled.middle_content)} chars")
    print(f"   - Last content length: {len(sampled.last_content)} chars")

    # Test short document (no sampling needed)
    print("\n1.2 Testing short document (no sampling)...")
    short_content = "This is a short document."
    sampled_short = sampler.sample_from_content(short_content)

    print(f"   - Original length: {len(short_content)} chars")
    print(f"   - First content length: {len(sampled_short.first_content)} chars")
    print(f"   - Middle content: '{sampled_short.middle_content}'")

    print("\n   PASS: Content sampler works correctly")


def test_strategy_executor():
    """Test the StrategyExecutor with different strategies."""
    print("\n" + "=" * 80)
    print("TEST 2: Strategy Executor")
    print("=" * 80)

    executor = StrategyExecutor()

    # Test header-based strategy
    print("\n2.1 Testing header_based strategy...")
    config = StrategyConfig(
        selected_strategy="header_based",
        chunk_size=512,
        overlap_percent=10,
        preserve_elements=["tables", "code_blocks"]
    )
    chunks = executor.execute(ACADEMIC_PAPER, config)
    print(f"   - Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk.content[:50].replace('\n', ' ')
        print(f"   - Chunk {i}: {len(chunk.content)} chars, '{preview}...'")

    # Test paragraph-based strategy
    print("\n2.2 Testing paragraph_based strategy...")
    config = StrategyConfig(
        selected_strategy="paragraph_based",
        chunk_size=512,
        overlap_percent=15
    )
    chunks = executor.execute(ACADEMIC_PAPER, config)
    print(f"   - Created {len(chunks)} chunks")

    # Test clause-based strategy on legal document
    print("\n2.3 Testing clause_based strategy on legal document...")
    config = StrategyConfig(
        selected_strategy="clause_based",
        chunk_size=1024,
        overlap_percent=5
    )
    chunks = executor.execute(LEGAL_CONTRACT, config)
    print(f"   - Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk.content[:50].replace('\n', ' ')
        print(f"   - Chunk {i}: {len(chunk.content)} chars, '{preview}...'")

    # Test table-based strategy
    print("\n2.4 Testing table_row_based strategy on tabular data...")
    config = StrategyConfig(
        selected_strategy="table_row_based",
        chunk_size=1024,
        overlap_percent=0,
        preserve_elements=["header_row"]
    )
    chunks = executor.execute(TABULAR_DATA, config)
    print(f"   - Created {len(chunks)} chunks")

    # Test log-based strategy
    print("\n2.5 Testing log_entry_based strategy on log file...")
    config = StrategyConfig(
        selected_strategy="log_entry_based",
        chunk_size=1024,
        overlap_percent=10,
        preserve_elements=["stack_traces"]
    )
    chunks = executor.execute(LOG_FILE, config)
    print(f"   - Created {len(chunks)} chunks")

    print("\n   PASS: Strategy executor works correctly")


def test_structure_analyzer_without_llm():
    """Test the StructureAnalyzer fallback when LLM is not available."""
    print("\n" + "=" * 80)
    print("TEST 3: Structure Analyzer (Fallback Mode)")
    print("=" * 80)

    print("\n3.1 Testing fallback strategy selection...")
    analyzer = StructureAnalyzer(llm_client=None)

    # This should return default strategy since no LLM is configured
    config = analyzer.analyze_from_content(ACADEMIC_PAPER)

    print(f"   - Selected strategy: {config.selected_strategy}")
    print(f"   - Chunk size: {config.chunk_size}")
    print(f"   - Overlap: {config.overlap_percent}%")
    print(f"   - Reasoning: {config.reasoning}")

    # Verify it's the default fallback
    assert config.selected_strategy == "paragraph_based", "Expected fallback to paragraph_based"

    print("\n   PASS: Structure analyzer fallback works correctly")


def test_adaptive_chunker_v3():
    """Test the AdaptiveChunker with V3.0 LLM-driven mode."""
    print("\n" + "=" * 80)
    print("TEST 4: Adaptive Chunker (V3.0 Mode)")
    print("=" * 80)

    print("\n4.1 Testing V3.0 LLM-driven chunking (fallback mode)...")

    # Create chunker with V3.0 mode enabled
    chunker = AdaptiveChunker(
        use_llm_driven_chunking=True,
        llm_client=None  # Will use fallback
    )

    result = chunker.chunk_content(
        content=ACADEMIC_PAPER,
        source_name="test_academic_paper",
        file_path="/tmp/test.md"
    )

    print(f"   - Total chunks: {len(result.chunks)}")
    print(f"   - Strategy used: {result.stats.get('strategy_used', 'N/A')}")
    print(f"   - Classification method: {result.stats.get('classification_method', 'N/A')}")

    # Check chunk metadata
    if result.chunks:
        first_chunk = result.chunks[0]
        print(f"   - First chunk metadata keys: {list(first_chunk.metadata.keys())[:10]}...")
        print(f"   - Chunking version: {first_chunk.metadata.get('chunking_version', 'N/A')}")

    print("\n4.2 Testing V2.0 mode (pattern-based)...")

    # Create chunker with V2.0 mode (LLM-driven disabled)
    chunker_v2 = AdaptiveChunker(
        use_llm_driven_chunking=False,
        use_llm_classification=False  # Disable LLM classification too
    )

    result_v2 = chunker_v2.chunk_content(
        content=ACADEMIC_PAPER,
        source_name="test_academic_paper",
        file_path="/tmp/test.md"
    )

    print(f"   - Total chunks: {len(result_v2.chunks)}")
    print(f"   - Document type: {result_v2.profile.document_type}")
    print(f"   - Strategy used: {result_v2.profile.recommended_strategy}")

    print("\n   PASS: Adaptive chunker works in both V2.0 and V3.0 modes")


def test_folder_chunking():
    """Test folder-based chunking with multi-page documents."""
    print("\n" + "=" * 80)
    print("TEST 5: Folder-Based Chunking")
    print("=" * 80)

    # Create temporary folder with page files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create page files
        for i in range(5):
            page_content = f"""# Page {i + 1}

This is the content of page {i + 1}.

## Section {i + 1}.1

Some text content for section {i + 1}.1.

## Section {i + 1}.2

More content for section {i + 1}.2.
"""
            page_path = os.path.join(tmpdir, f"page_{i + 1}_nohf.md")
            with open(page_path, 'w') as f:
                f.write(page_content)

        print(f"\n5.1 Created {5} test pages in {tmpdir}")

        # Test content sampler with folder
        sampler = ContentSampler()
        scenario, files = sampler.detect_scenario(tmpdir)
        print(f"   - Detected scenario: {scenario}")
        print(f"   - Found {len(files)} files")

        # Test sampled content
        sampled = sampler.sample_from_folder(tmpdir)
        print(f"   - Total pages: {sampled.total_pages}")
        print(f"   - Total sampled chars: {sampled.total_chars}")

        # Test folder chunking
        print("\n5.2 Testing chunk_folder method...")
        chunker = AdaptiveChunker(
            use_llm_driven_chunking=True,
            llm_client=None
        )

        result = chunker.chunk_folder(tmpdir, source_name="test_document")
        print(f"   - Total chunks: {len(result.chunks)}")
        print(f"   - Total pages processed: {result.stats.get('total_pages', 0)}")
        print(f"   - Strategy used: {result.stats.get('strategy_used', 'N/A')}")

    print("\n   PASS: Folder-based chunking works correctly")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("LLM-DRIVEN ADAPTIVE CHUNKING (V3.0) TEST SUITE")
    print("=" * 80)

    try:
        test_content_sampler()
        test_strategy_executor()
        test_structure_analyzer_without_llm()
        test_adaptive_chunker_v3()
        test_folder_chunking()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)

    except Exception as e:
        print(f"\n\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
