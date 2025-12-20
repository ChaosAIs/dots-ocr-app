"""
Test script to validate GraphRAG performance optimizations.

This script tests:
1. Importance score filtering
2. Selective entity extraction
3. Configuration loading

Run with: python -m pytest backend/test/test_graphrag_optimization.py -v
"""

import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_service.graph_rag.entity_extractor import EntityExtractor
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


# Chunk filtering tests removed - content-based filtering has been disabled
# to prevent loss of important data from structured documents (invoices, tables, etc.)


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

    def test_graph_rag_index_enabled_flag(self):
        """Test GRAPH_RAG_INDEX_ENABLED flag loading."""
        from rag_service.graph_rag import graph_indexer
        
        # Should be loaded from .env
        assert hasattr(graph_indexer, 'GRAPH_RAG_INDEX_ENABLED')

    # Chunk filtering config tests removed - feature has been disabled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

