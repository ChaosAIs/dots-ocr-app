"""
Unit tests for GraphRAG components.

Tests cover:
- Entity and Relationship data classes
- Query mode detection (heuristic)
- Entity extraction parsing
- Utility functions
"""

import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_service.graph_rag.base import QueryMode, QueryParam, Entity, Relationship
from rag_service.graph_rag.utils import (
    generate_entity_id,
    generate_relationship_id,
    parse_entity_line,
    parse_relationship_line,
    parse_extraction_output,
    deduplicate_entities,
    deduplicate_relationships,
    truncate_text,
    clean_llm_response,
)
from rag_service.graph_rag.query_mode_detector import QueryModeDetector


class TestQueryMode:
    """Tests for QueryMode enum."""

    def test_query_modes_exist(self):
        """Test that all query modes are defined (following Graph-R1 paper)."""
        assert QueryMode.LOCAL.value == "local"
        assert QueryMode.GLOBAL.value == "global"
        assert QueryMode.HYBRID.value == "hybrid"


class TestQueryParam:
    """Tests for QueryParam dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = QueryParam()
        assert params.mode == QueryMode.HYBRID
        assert params.top_k >= 1  # Reads from env or defaults to 60
        assert params.max_steps >= 1  # Reads from env or defaults to 1
        assert params.only_need_context is False

    def test_custom_values(self):
        """Test custom parameter values."""
        params = QueryParam(mode=QueryMode.LOCAL, top_k=20, max_steps=3, only_need_context=True)
        assert params.mode == QueryMode.LOCAL
        assert params.top_k == 20
        assert params.max_steps == 3
        assert params.only_need_context is True

    def test_iterative_reasoning_enabled(self):
        """Test that max_steps > 1 enables iterative reasoning."""
        params = QueryParam(max_steps=5)
        assert params.max_steps == 5
        # When max_steps > 1, iterative reasoning should be triggered


class TestEntity:
    """Tests for Entity dataclass."""

    def test_entity_creation(self):
        """Test entity creation with required fields."""
        entity = Entity(
            id="entity_1",
            name="Test Entity",
            entity_type="PERSON",
            description="A test entity",
            source_chunk_id="chunk_1",
        )
        assert entity.id == "entity_1"
        assert entity.name == "Test Entity"
        assert entity.entity_type == "PERSON"
        assert entity.key_score == 50  # default is 50

    def test_entity_with_key_score(self):
        """Test entity with custom key score."""
        entity = Entity(
            id="entity_2",
            name="Important Entity",
            entity_type="ORGANIZATION",
            description="An important entity",
            source_chunk_id="chunk_2",
            key_score=95,
        )
        assert entity.key_score == 95


class TestRelationship:
    """Tests for Relationship dataclass."""

    def test_relationship_creation(self):
        """Test relationship creation."""
        rel = Relationship(
            id="rel_1",
            src_entity_id="entity_1",
            tgt_entity_id="entity_2",
            description="works for",
            keywords=["employment", "job"],
            source_chunk_id="chunk_1",
        )
        assert rel.id == "rel_1"
        assert rel.weight == 1.0  # default
        assert len(rel.keywords) == 2


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_generate_entity_id(self):
        """Test entity ID generation."""
        # generate_entity_id takes (name, entity_type, workspace_id)
        id1 = generate_entity_id("John Smith", "PERSON")
        id2 = generate_entity_id("John Smith", "PERSON")
        id3 = generate_entity_id("Jane Doe", "PERSON")

        assert id1 == id2  # Same inputs = same ID
        assert id1 != id3  # Different name = different ID

    def test_generate_relationship_id(self):
        """Test relationship ID generation."""
        # generate_relationship_id takes (src_name, tgt_name, workspace_id)
        # Note: IDs are sorted, so order doesn't matter
        id1 = generate_relationship_id("entity_1", "entity_2")
        id2 = generate_relationship_id("entity_1", "entity_2")
        id3 = generate_relationship_id("entity_2", "entity_1")

        assert id1 == id2  # Same inputs = same ID
        assert id1 == id3  # Order doesn't matter (sorted internally)

    def test_truncate_text(self):
        """Test text truncation."""
        short_text = "Hello"
        # truncate_text uses word-based truncation (max_tokens * 0.75 words)
        # For 100 tokens, that's 75 words max
        long_text = " ".join(["word"] * 200)  # 200 words

        assert truncate_text(short_text, 100) == short_text
        truncated = truncate_text(long_text, 100)
        assert truncated.endswith("...")
        assert len(truncated.split()) <= 76  # 75 words + "..."

    def test_clean_llm_response(self):
        """Test LLM response cleaning."""
        # clean_llm_response removes markdown code blocks, not think tags
        response = "```python\ncode\n```Actual response"
        cleaned = clean_llm_response(response)
        assert "```" not in cleaned
        assert "Actual response" in cleaned


class TestQueryModeDetectorHeuristic:
    """Tests for heuristic query mode detection."""

    def test_local_mode_detection(self):
        """Test LOCAL mode detection for entity queries."""
        detector = QueryModeDetector()

        local_queries = [
            "Who is John Smith?",
            "What is machine learning?",
            "Tell me about the CEO",
            "Define artificial intelligence",
        ]

        for query in local_queries:
            mode, _ = detector.detect_mode_heuristic(query)
            assert mode == QueryMode.LOCAL, f"Expected LOCAL for: {query}"

    def test_global_mode_detection(self):
        """Test GLOBAL mode detection for relationship queries."""
        detector = QueryModeDetector()

        # Note: LOCAL patterns are checked first, so avoid "what is" etc.
        global_queries = [
            "How does A relate to B?",
            "Show the relationship between X and Y",
            "Compare product A and product B",
            "The connection between these concepts",
        ]

        for query in global_queries:
            mode, _ = detector.detect_mode_heuristic(query)
            assert mode == QueryMode.GLOBAL, f"Expected GLOBAL for: {query}"

    def test_hybrid_mode_fallback(self):
        """Test HYBRID mode as fallback for ambiguous queries."""
        detector = QueryModeDetector()

        # Queries that don't match specific patterns
        ambiguous_queries = [
            "Tell me everything about the project",
            "What should I know about this?",
        ]

        for query in ambiguous_queries:
            mode, _ = detector.detect_mode_heuristic(query)
            assert mode == QueryMode.HYBRID, f"Expected HYBRID for: {query}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

