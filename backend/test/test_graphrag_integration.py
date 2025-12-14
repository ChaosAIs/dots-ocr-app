"""
Integration tests for GraphRAG pipeline.

Tests cover:
- PostgreSQL storage operations
- Entity extraction with LLM
- Full indexing pipeline
- Query mode detection and retrieval

Note: Some tests require external services (PostgreSQL, Ollama, Qdrant, Neo4j).
Tests are marked with pytest.mark.integration for selective running.
"""

import pytest
import sys
import os
import asyncio

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

from rag_service.graph_rag.base import QueryMode, QueryParam, Entity, Relationship
from rag_service.graph_rag.utils import (
    generate_entity_id,
    parse_entity_line,
    parse_relationship_line,
    parse_extraction_output,
)


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestEntityParsing:
    """Integration tests for entity parsing from LLM output."""

    def test_parse_valid_entity_line(self):
        """Test parsing a valid entity line."""
        line = "(entity|PERSON|John Smith|CEO of Acme Corp, leading innovation|85)"
        entity = parse_entity_line(line, source_chunk_id="chunk_1")

        assert entity is not None
        assert entity.name == "John Smith"
        assert entity.entity_type == "PERSON"
        assert entity.description == "CEO of Acme Corp, leading innovation"
        assert entity.key_score == 85
        assert entity.source_chunk_id == "chunk_1"

    def test_parse_entity_with_special_chars(self):
        """Test parsing entity with special characters."""
        line = "(entity|ORGANIZATION|Acme Corp.|A company founded in 1990, specializing in AI|90)"
        entity = parse_entity_line(line, source_chunk_id="chunk_2")

        assert entity is not None
        assert entity.name == "Acme Corp."
        assert entity.entity_type == "ORGANIZATION"

    def test_parse_invalid_entity_line(self):
        """Test parsing invalid entity lines returns None."""
        invalid_lines = [
            "Not an entity line",
            "(entity|PERSON)",  # Too few parts
            "(relationship|PERSON|John|Jane|knows)",  # Wrong prefix
        ]

        for line in invalid_lines:
            entity = parse_entity_line(line)
            assert entity is None, f"Expected None for: {line}"

    def test_parse_valid_relationship_line(self):
        """Test parsing a valid relationship line."""
        # Format: (relationship|<src>|<tgt>|<desc>|<keywords>|<weight>)
        line = "(relationship|John Smith|Acme Corp|WORKS_FOR|employment, job|1.0)"
        rel = parse_relationship_line(line, source_chunk_id="chunk_1")

        assert rel is not None
        assert rel.src_entity_id is not None
        assert rel.tgt_entity_id is not None
        assert rel.description == "WORKS_FOR"
        assert "employment" in rel.keywords

    def test_parse_extraction_output(self):
        """Test parsing full LLM extraction output."""
        llm_output = """
(entity|PERSON|Alice|Software engineer at TechCo|80)
(entity|ORGANIZATION|TechCo|Technology company|75)
(relationship|Alice|TechCo|WORKS_AT|employment, developer|1.0)
"""
        entities, relationships = parse_extraction_output(llm_output, "chunk_1")

        assert len(entities) == 2
        assert len(relationships) == 1
        assert entities[0].name == "Alice"
        assert entities[1].name == "TechCo"


# Check for optional dependencies
try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

try:
    import neo4j
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False


class TestPostgresKVStorage:
    """Integration tests for PostgreSQL KV storage."""

    @pytest.fixture
    def storage(self):
        """Create storage instance."""
        from rag_service.storage.postgres_kv_storage import PostgresKVStorage
        # PostgresKVStorage takes table_name, workspace_id, connection_string
        return PostgresKVStorage(
            table_name="graphrag_entities",
            workspace_id="test_workspace"
        )

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_ASYNCPG, reason="asyncpg not installed")
    @pytest.mark.skipif(
        os.getenv("POSTGRES_HOST") is None,
        reason="PostgreSQL not available"
    )
    async def test_upsert_and_get(self, storage):
        """Test upserting and retrieving entities."""
        test_entity = {
            "id": "test_entity_1",
            "entity_name": "Test Entity",
            "entity_type": "TEST",
            "description": "A test entity for integration testing",
        }

        # Upsert
        await storage.upsert({"test_entity_1": test_entity})

        # Get
        result = await storage.get_by_id("test_entity_1")
        assert result is not None
        assert result["entity_name"] == "Test Entity"

        # Cleanup
        await storage.delete(["test_entity_1"])

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_ASYNCPG, reason="asyncpg not installed")
    @pytest.mark.skipif(
        os.getenv("POSTGRES_HOST") is None,
        reason="PostgreSQL not available"
    )
    async def test_filter_by_workspace(self, storage):
        """Test filtering by workspace ID using filter_keys and get_by_ids."""
        # Insert test data
        await storage.upsert({
            "ws_entity_1": {"entity_name": "WS Entity 1"},
            "ws_entity_2": {"entity_name": "WS Entity 2"},
        })

        # Use filter_keys to check which keys exist
        existing_keys = await storage.filter_keys(["ws_entity_1", "ws_entity_2", "nonexistent"])
        assert "ws_entity_1" in existing_keys
        assert "ws_entity_2" in existing_keys
        assert "nonexistent" not in existing_keys

        # Use get_by_ids to retrieve the data
        results = await storage.get_by_ids(["ws_entity_1", "ws_entity_2"])
        assert len(results) == 2

        # Cleanup
        await storage.delete(["ws_entity_1", "ws_entity_2"])


class TestQueryModeDetection:
    """Integration tests for query mode detection."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        from rag_service.graph_rag.query_mode_detector import QueryModeDetector
        return QueryModeDetector()

    def test_heuristic_detection_accuracy(self, detector):
        """Test heuristic detection for various query types."""
        test_cases = [
            # (query, expected_mode)
            ("Who is the CEO of the company?", QueryMode.LOCAL),
            ("What is machine learning?", QueryMode.LOCAL),
            ("How does product A relate to product B?", QueryMode.GLOBAL),
            ("Compare A and B approaches", QueryMode.GLOBAL),  # pattern: compare.*and
            ("List all products", QueryMode.NAIVE),
            ("When was the company founded?", QueryMode.NAIVE),
        ]

        for query, expected in test_cases:
            mode, _ = detector.detect_mode_heuristic(query)
            assert mode == expected, f"Query: {query}, Expected: {expected}, Got: {mode}"


class TestNeo4jStorage:
    """Integration tests for Neo4j graph storage."""

    @pytest.fixture
    def storage(self):
        """Create Neo4j storage instance."""
        import rag_service.storage.neo4j_storage as neo4j_module
        from rag_service.storage.neo4j_storage import Neo4jStorage

        # Reset the module-level driver to get a fresh connection for this event loop
        neo4j_module._neo4j_driver = None

        storage = Neo4jStorage(workspace_id="test_workspace")
        yield storage

        # Reset driver after test to prevent reuse across event loops
        neo4j_module._neo4j_driver = None

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_NEO4J, reason="neo4j not installed")
    @pytest.mark.skipif(
        os.getenv("NEO4J_URI") is None or os.getenv("NEO4J_PASSWORD", "").startswith("your_"),
        reason="Neo4j not configured (set NEO4J_URI and NEO4J_PASSWORD in .env)"
    )
    async def test_upsert_and_get_node(self, storage):
        """Test upserting and retrieving a node."""
        node_id = "test_node_1"
        node_data = {
            "name": "Test Node",
            "entity_type": "TEST",
            "description": "A test node",
        }

        # Upsert node
        await storage.upsert_node(node_id, node_data)

        # Get node
        result = await storage.get_node(node_id)
        assert result is not None
        assert result["name"] == "Test Node"

        # Cleanup
        await storage.delete_node(node_id)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_NEO4J, reason="neo4j not installed")
    @pytest.mark.skipif(
        os.getenv("NEO4J_URI") is None or os.getenv("NEO4J_PASSWORD", "").startswith("your_"),
        reason="Neo4j not configured (set NEO4J_URI and NEO4J_PASSWORD in .env)"
    )
    async def test_upsert_edge(self, storage):
        """Test upserting an edge between nodes."""
        # Create source and target nodes
        await storage.upsert_node("edge_src", {"name": "Source"})
        await storage.upsert_node("edge_tgt", {"name": "Target"})

        # Upsert edge
        await storage.upsert_edge(
            src_id="edge_src",
            tgt_id="edge_tgt",
            edge_data={"relationship": "CONNECTS_TO", "weight": 1.0}
        )

        # Get edges for source node
        edges = await storage.get_node_edges("edge_src")
        assert len(edges) > 0

        # Cleanup
        await storage.delete_node("edge_src")
        await storage.delete_node("edge_tgt")


class TestGraphRAGOrchestrator:
    """Integration tests for the main GraphRAG orchestrator."""

    @pytest.fixture
    def graphrag(self):
        """Create GraphRAG instance."""
        from rag_service.graph_rag.graph_rag import GraphRAG
        return GraphRAG(workspace_id="test_workspace")

    def test_format_context_with_entities(self, graphrag):
        """Test formatting context with entities."""
        from rag_service.graph_rag.graph_rag import GraphRAGContext

        # GraphRAGContext uses Dict types, not Entity/Relationship types
        context = GraphRAGContext(
            entities=[
                {
                    "id": "e1",
                    "name": "John Smith",
                    "entity_type": "PERSON",
                    "description": "CEO of Acme Corp",
                }
            ],
            relationships=[
                {
                    "id": "r1",
                    "src_entity_id": "e1",
                    "tgt_entity_id": "e2",
                    "description": "LEADS",
                    "keywords": "leadership, management",
                }
            ],
            chunks=[{"page_content": "Acme Corp is a leading tech company."}],
            mode=QueryMode.HYBRID,
            enhanced_query="Tell me about Acme Corp",
        )

        formatted = graphrag.format_context(context)

        assert "John Smith" in formatted
        assert "CEO of Acme Corp" in formatted
        assert "LEADS" in formatted
        assert "Acme Corp is a leading tech company" in formatted

    def test_query_param_defaults(self, graphrag):
        """Test QueryParam default values."""
        params = QueryParam()

        assert params.mode == QueryMode.HYBRID
        assert params.top_k == 60
        assert params.only_need_context is False


class TestEntityExtractorIntegration:
    """Integration tests for entity extraction (requires Ollama)."""

    @pytest.fixture
    def extractor(self):
        """Create EntityExtractor instance."""
        from rag_service.graph_rag.entity_extractor import EntityExtractor
        # EntityExtractor uses Ollama client by default, configured via environment
        return EntityExtractor(max_gleaning=2)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_simple_extraction(self, extractor):
        """Test simple entity extraction from text."""
        text = """
        John Smith is the CEO of Acme Corporation. He founded the company in 2010
        in San Francisco, California. The company specializes in artificial
        intelligence and machine learning solutions.
        """

        entities, relationships = await extractor.extract_simple(
            text=text,
            chunk_id="test_chunk",
        )

        # Should extract at least some entities
        entity_names = [e.name.lower() for e in entities]

        # Check for expected entities (case-insensitive)
        assert any("john" in name for name in entity_names), "Should find John Smith"
        assert any("acme" in name for name in entity_names), "Should find Acme Corporation"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
