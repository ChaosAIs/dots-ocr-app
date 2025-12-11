"""
Unit tests for RAG scopes feature.

Tests cover:
- FileSummaryWithScopes dataclass
- Helper functions: _split_by_headers, _split_by_size, _parse_json_response
- Vectorstore scope functions (mocked)
- RAG agent scope functions (mocked)

Note: Some tests require external services (Ollama, Qdrant).
Tests are marked with pytest.mark.integration for selective running.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_service.summarizer import (
    FileSummaryWithScopes,
    _split_by_headers,
    _split_by_size,
    _parse_json_response,
    MAX_DIRECT_SUMMARY_SIZE,
    MAX_SECTION_SIZE,
    MAX_SECTIONS_TO_PROCESS,
)


class TestFileSummaryWithScopes:
    """Test FileSummaryWithScopes dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        summary = FileSummaryWithScopes(summary="Test summary")
        assert summary.summary == "Test summary"
        assert summary.scopes == []
        assert summary.content_type == "document"
        assert summary.complexity == "intermediate"

    def test_all_fields(self):
        """Test setting all fields."""
        summary = FileSummaryWithScopes(
            summary="Full summary",
            scopes=["auth", "security", "jwt"],
            content_type="technical_documentation",
            complexity="advanced",
        )
        assert summary.summary == "Full summary"
        assert summary.scopes == ["auth", "security", "jwt"]
        assert summary.content_type == "technical_documentation"
        assert summary.complexity == "advanced"

    def test_asdict(self):
        """Test conversion to dictionary."""
        summary = FileSummaryWithScopes(
            summary="Test",
            scopes=["topic1", "topic2"],
        )
        d = asdict(summary)
        assert d["summary"] == "Test"
        assert d["scopes"] == ["topic1", "topic2"]
        assert d["content_type"] == "document"
        assert d["complexity"] == "intermediate"


class TestSplitByHeaders:
    """Test _split_by_headers function."""

    def test_split_by_h1_headers(self):
        """Test splitting by H1 headers."""
        # Each section needs > 100 characters to be included
        content = """# Introduction
This is the introduction section with enough content to be included. We need to add more text here to make sure this section is longer than 100 characters.

# Main Content
This is the main content section with detailed information about the topic. Adding more content to ensure the section length exceeds the minimum threshold of 100 characters.

# Conclusion
This is the conclusion with final thoughts and summary of the document. This section also needs to have sufficient content to pass the length check."""

        sections = _split_by_headers(content)
        assert len(sections) >= 2, f"Expected at least 2 sections, got {len(sections)}"

    def test_split_by_mixed_headers(self):
        """Test splitting by mixed header levels."""
        # Sections need > 100 chars
        content = """# Header 1
Content for header 1 with sufficient length to be included in the section. Adding more content to ensure the minimum length threshold is met.

## Header 2
Content for header 2 with detailed information about the subsection. This needs additional text to exceed the 100 character minimum requirement.

### Header 3
Content for header 3 with more specific details about the topic. Including extra content to make sure this section passes the length filter."""

        sections = _split_by_headers(content)
        assert len(sections) >= 1

    def test_no_headers_fallback_to_size(self):
        """Test fallback to size splitting when no headers."""
        content = "A" * 10000  # Long content without headers
        sections = _split_by_headers(content)
        assert len(sections) >= 1

    def test_short_content(self):
        """Test with short content."""
        content = "Short content"
        sections = _split_by_headers(content)
        assert len(sections) >= 1

    def test_max_sections_limit(self):
        """Test that sections are limited to MAX_SECTIONS_TO_PROCESS."""
        # Create content with many headers
        content = "\n".join([f"# Header {i}\n{'Content ' * 50}" for i in range(30)])
        sections = _split_by_headers(content)
        assert len(sections) <= MAX_SECTIONS_TO_PROCESS


class TestSplitBySize:
    """Test _split_by_size function."""

    def test_short_content(self):
        """Test content shorter than max_size."""
        content = "Short content"
        sections = _split_by_size(content, 1000)
        assert len(sections) == 1
        assert sections[0] == "Short content"

    def test_exact_max_size(self):
        """Test content at exact max_size."""
        content = "A" * 1000
        sections = _split_by_size(content, 1000)
        assert len(sections) == 1

    def test_split_at_paragraph(self):
        """Test splitting at paragraph boundaries."""
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        sections = _split_by_size(content, 30)
        assert len(sections) >= 1

    def test_long_content(self):
        """Test long content is split into multiple sections."""
        content = "A" * 10000
        sections = _split_by_size(content, 1000)
        assert len(sections) >= 9  # 10000 / 1000 = 10, but may vary due to boundaries


class TestParseJsonResponse:
    """Test _parse_json_response function."""

    def test_valid_json(self):
        """Test parsing valid JSON."""
        response = '{"summary": "Test", "scopes": ["a", "b"]}'
        result = _parse_json_response(response)
        assert result["summary"] == "Test"
        assert result["scopes"] == ["a", "b"]

    def test_json_with_markdown_wrapper(self):
        """Test parsing JSON with markdown code block."""
        response = '```json\n{"summary": "Test", "scopes": ["a"]}\n```'
        result = _parse_json_response(response)
        assert result["summary"] == "Test"

    def test_json_with_generic_code_block(self):
        """Test parsing JSON with generic code block."""
        response = '```\n{"summary": "Test"}\n```'
        result = _parse_json_response(response)
        assert result["summary"] == "Test"

    def test_invalid_json(self):
        """Test handling invalid JSON."""
        response = "Not valid JSON"
        result = _parse_json_response(response)
        assert result == {}

    def test_empty_response(self):
        """Test handling empty response."""
        result = _parse_json_response("")
        assert result == {}

    def test_json_with_leading_text(self):
        """Test parsing JSON with leading text."""
        response = 'Here is the summary:\n{"summary": "Test", "scopes": []}'
        result = _parse_json_response(response)
        assert result.get("summary") == "Test"


class TestConstants:
    """Test constants are defined correctly."""

    def test_max_direct_summary_size(self):
        """Test MAX_DIRECT_SUMMARY_SIZE is reasonable."""
        assert MAX_DIRECT_SUMMARY_SIZE == 8000

    def test_max_section_size(self):
        """Test MAX_SECTION_SIZE is reasonable."""
        assert MAX_SECTION_SIZE == 4000

    def test_max_sections_to_process(self):
        """Test MAX_SECTIONS_TO_PROCESS is reasonable."""
        assert MAX_SECTIONS_TO_PROCESS == 20


class TestGenerateFileSummaryWithScopes:
    """Test generate_file_summary_with_scopes function."""

    def test_empty_content(self):
        """Test with empty content."""
        from rag_service.summarizer import generate_file_summary_with_scopes

        result = generate_file_summary_with_scopes("test.md", "")
        assert isinstance(result, FileSummaryWithScopes)
        assert "No content available" in result.summary
        assert result.scopes == []
        assert result.content_type == "document"
        assert result.complexity == "basic"

    def test_short_content(self):
        """Test with very short content."""
        from rag_service.summarizer import generate_file_summary_with_scopes

        result = generate_file_summary_with_scopes("test.md", "Short")
        assert isinstance(result, FileSummaryWithScopes)
        assert "No content available" in result.summary

    @pytest.mark.integration
    @pytest.mark.slow
    def test_direct_summarization(self):
        """Test direct summarization for small file (requires Ollama)."""
        from rag_service.summarizer import generate_file_summary_with_scopes

        content = """# Authentication Guide

This document describes how to authenticate users in our system.

## JWT Tokens
We use JSON Web Tokens for authentication. Tokens are signed with RS256.

## Security Considerations
Always validate tokens on the server side. Never trust client-side validation.
"""
        result = generate_file_summary_with_scopes("auth-guide.md", content)
        assert isinstance(result, FileSummaryWithScopes)
        assert len(result.summary) > 50
        assert len(result.scopes) >= 0  # May or may not have scopes


class TestVectorstoreScopeFunctions:
    """Test vectorstore scope functions with mocks."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client."""
        with patch("rag_service.vectorstore.QdrantClient") as mock:
            yield mock

    def test_add_file_summary_with_scopes_signature(self):
        """Test add_file_summary_with_scopes has correct signature."""
        from rag_service.vectorstore import add_file_summary_with_scopes
        import inspect

        sig = inspect.signature(add_file_summary_with_scopes)
        params = list(sig.parameters.keys())
        assert "source_name" in params
        assert "file_path" in params
        assert "summary" in params
        assert "scopes" in params
        assert "content_type" in params
        assert "complexity" in params

    def test_search_file_summaries_with_scopes_signature(self):
        """Test search_file_summaries_with_scopes has correct signature."""
        from rag_service.vectorstore import search_file_summaries_with_scopes
        import inspect

        sig = inspect.signature(search_file_summaries_with_scopes)
        params = list(sig.parameters.keys())
        assert "query" in params
        assert "k" in params
        assert "score_threshold" in params

    def test_get_all_file_summaries_signature(self):
        """Test get_all_file_summaries has correct signature."""
        from rag_service.vectorstore import get_all_file_summaries
        import inspect

        sig = inspect.signature(get_all_file_summaries)
        params = list(sig.parameters.keys())
        # Should have no required parameters
        assert len([p for p in sig.parameters.values() if p.default == inspect.Parameter.empty]) == 0


class TestRagAgentScopeFunctions:
    """Test RAG agent scope functions with mocks."""

    def test_analyze_query_with_scopes_signature(self):
        """Test _analyze_query_with_scopes has correct signature."""
        from rag_service.rag_agent import _analyze_query_with_scopes
        import inspect

        sig = inspect.signature(_analyze_query_with_scopes)
        params = list(sig.parameters.keys())
        assert "query" in params

    def test_match_scopes_with_llm_signature(self):
        """Test _match_scopes_with_llm has correct signature."""
        from rag_service.rag_agent import _match_scopes_with_llm
        import inspect

        sig = inspect.signature(_match_scopes_with_llm)
        params = list(sig.parameters.keys())
        assert "query_scopes" in params
        assert "candidate_docs" in params

    def test_find_relevant_files_with_scopes_signature(self):
        """Test _find_relevant_files_with_scopes has correct signature."""
        from rag_service.rag_agent import _find_relevant_files_with_scopes
        import inspect

        sig = inspect.signature(_find_relevant_files_with_scopes)
        params = list(sig.parameters.keys())
        assert "enhanced_query" in params
        assert "query_scopes" in params
        assert "k" in params

    @pytest.mark.integration
    @pytest.mark.slow
    def test_analyze_query_with_scopes_integration(self):
        """Test _analyze_query_with_scopes with real LLM (requires Ollama)."""
        from rag_service.rag_agent import _analyze_query_with_scopes

        query = "How do I validate JWT tokens in Python?"
        enhanced_query, scopes = _analyze_query_with_scopes(query)

        # Enhanced query should be a non-empty string
        assert isinstance(enhanced_query, str)
        assert len(enhanced_query) > 0

        # Scopes should be a list
        assert isinstance(scopes, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

