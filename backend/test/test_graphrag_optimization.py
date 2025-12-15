"""
Test script to validate GraphRAG performance optimizations.

This script tests:
1. Importance score filtering
2. Chunk filtering logic
3. Selective entity extraction
4. Configuration loading

Run with: python -m pytest backend/test/test_graphrag_optimization.py -v
"""

import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_service.graph_rag.entity_extractor import EntityExtractor
from rag_service.graph_rag.graph_indexer import _is_low_information_chunk
from rag_service.graph_rag.base import Entity


class TestImportanceFiltering:
    """Test importance score filtering."""

    def test_entity_extractor_default_min_score(self):
        """Test that EntityExtractor loads min_entity_score from env."""
        # Set env var
        os.environ["GRAPH_RAG_MIN_ENTITY_SCORE"] = "70"
        
        extractor = EntityExtractor()
        assert extractor.min_entity_score == 70
        
        # Cleanup
        del os.environ["GRAPH_RAG_MIN_ENTITY_SCORE"]

    def test_entity_extractor_custom_min_score(self):
        """Test that custom min_entity_score overrides env."""
        os.environ["GRAPH_RAG_MIN_ENTITY_SCORE"] = "60"
        
        extractor = EntityExtractor(min_entity_score=80)
        assert extractor.min_entity_score == 80
        
        # Cleanup
        del os.environ["GRAPH_RAG_MIN_ENTITY_SCORE"]

    def test_gleaning_disabled_by_default(self):
        """Test that gleaning is disabled by default (max_gleaning=0)."""
        os.environ["GRAPH_RAG_MAX_GLEANING"] = "0"
        
        extractor = EntityExtractor()
        assert extractor.max_gleaning == 0
        
        # Cleanup
        del os.environ["GRAPH_RAG_MAX_GLEANING"]


class TestChunkFiltering:
    """Test chunk filtering logic."""

    def test_empty_chunk_filtered(self):
        """Test that empty chunks are filtered."""
        assert _is_low_information_chunk("") is True
        assert _is_low_information_chunk("   ") is True
        assert _is_low_information_chunk("\n\n") is True

    def test_short_chunk_filtered(self):
        """Test that very short chunks are filtered."""
        assert _is_low_information_chunk("Short") is True
        assert _is_low_information_chunk("A bit longer but still short") is True

    def test_mostly_numbers_filtered(self):
        """Test that chunks with mostly numbers are filtered (tables/data)."""
        table_chunk = "123 456 789 012 345 678 901 234 567 890 123 456 789"
        assert _is_low_information_chunk(table_chunk) is True

    def test_low_alpha_ratio_filtered(self):
        """Test that chunks with few letters are filtered."""
        symbols_chunk = "!@#$%^&*()_+-=[]{}|;':\",./<>? 123 456 789"
        assert _is_low_information_chunk(symbols_chunk) is True

    def test_repetitive_chunk_filtered(self):
        """Test that highly repetitive chunks are filtered."""
        repetitive = "the the the the the the the the the the the the"
        assert _is_low_information_chunk(repetitive) is True

    def test_normal_chunk_not_filtered(self):
        """Test that normal text chunks are NOT filtered."""
        normal_text = """
        This is a normal paragraph with meaningful content about machine learning
        and artificial intelligence. It contains various concepts and ideas that
        should be extracted for the knowledge graph. The text has good vocabulary
        diversity and semantic value.
        """
        assert _is_low_information_chunk(normal_text) is False

    def test_technical_content_not_filtered(self):
        """Test that technical content with some numbers is NOT filtered."""
        technical = """
        The system processes 1000 requests per second using a distributed
        architecture with 5 microservices. Each service handles specific
        business logic and communicates via REST APIs.
        """
        assert _is_low_information_chunk(technical) is False


class TestSelectiveExtraction:
    """Test selective entity extraction prompt."""

    def test_prompt_emphasizes_quality(self):
        """Test that the extraction prompt emphasizes quality over quantity."""
        from rag_service.graph_rag.prompts import ENTITY_EXTRACTION_PROMPT
        
        # Check for key phrases that indicate selective extraction
        assert "ONLY the most important" in ENTITY_EXTRACTION_PROMPT
        assert "quality over quantity" in ENTITY_EXTRACTION_PROMPT
        assert "DO NOT extract" in ENTITY_EXTRACTION_PROMPT
        assert "5-15" in ENTITY_EXTRACTION_PROMPT  # Target entity count

    def test_prompt_includes_importance_guidelines(self):
        """Test that prompt includes importance score guidelines."""
        from rag_service.graph_rag.prompts import ENTITY_EXTRACTION_PROMPT
        
        assert "80-100" in ENTITY_EXTRACTION_PROMPT  # Critical entities
        assert "60-79" in ENTITY_EXTRACTION_PROMPT   # Important entities
        assert "importance_score" in ENTITY_EXTRACTION_PROMPT


class TestConfigurationLoading:
    """Test that configuration is loaded correctly from environment."""

    def test_graph_rag_enabled_flag(self):
        """Test GRAPH_RAG_ENABLED flag loading."""
        from rag_service.graph_rag import graph_indexer
        
        # Should be loaded from .env
        assert hasattr(graph_indexer, 'GRAPH_RAG_ENABLED')

    def test_min_chunk_length_config(self):
        """Test GRAPH_RAG_MIN_CHUNK_LENGTH loading."""
        from rag_service.graph_rag import graph_indexer
        
        assert hasattr(graph_indexer, 'GRAPH_RAG_MIN_CHUNK_LENGTH')
        assert graph_indexer.GRAPH_RAG_MIN_CHUNK_LENGTH >= 0

    def test_chunk_filtering_enabled_config(self):
        """Test GRAPH_RAG_ENABLE_CHUNK_FILTERING loading."""
        from rag_service.graph_rag import graph_indexer
        
        assert hasattr(graph_indexer, 'GRAPH_RAG_ENABLE_CHUNK_FILTERING')
        assert isinstance(graph_indexer.GRAPH_RAG_ENABLE_CHUNK_FILTERING, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

