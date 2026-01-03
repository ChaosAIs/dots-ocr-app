"""
GraphRAG Module - Knowledge Graph enhanced RAG.

This module provides Graph-R1 inspired GraphRAG capabilities:
- Entity and relationship extraction from documents
- Knowledge graph storage and querying
- Multi-mode query detection (LOCAL, GLOBAL, HYBRID, NAIVE)
- Graph-based retrieval integrated with vector search
"""

from .base import (
    QueryParam,
    QueryMode,
    Entity,
    Relationship,
    BaseKVStorage,
    BaseVectorStorage,
    BaseGraphStorage,
)
from .entity_extractor import EntityExtractor
from .graph_indexer import (
    GraphRAGIndexer,
    index_chunks_sync,
    delete_graphrag_by_source_sync,
    delete_graphrag_by_document_id_sync,
    GRAPH_RAG_INDEX_ENABLED,
)
from .query_mode_detector import QueryModeDetector
from .graph_rag import GraphRAG, GraphRAGContext, GRAPH_RAG_QUERY_ENABLED
from .utils import (
    generate_entity_id,
    generate_relationship_id,
    parse_extraction_output,
    deduplicate_entities,
    deduplicate_relationships,
)

__all__ = [
    # Base classes
    "QueryParam",
    "QueryMode",
    "Entity",
    "Relationship",
    "BaseKVStorage",
    "BaseVectorStorage",
    "BaseGraphStorage",
    # Entity extraction
    "EntityExtractor",
    # Graph indexing
    "GraphRAGIndexer",
    "index_chunks_sync",
    "delete_graphrag_by_source_sync",
    "delete_graphrag_by_document_id_sync",
    "GRAPH_RAG_INDEX_ENABLED",    
    # Query processing
    "QueryModeDetector",
    "GraphRAG",
    "GraphRAGContext",
    "GRAPH_RAG_QUERY_ENABLED",
    # Utilities
    "generate_entity_id",
    "generate_relationship_id",
    "parse_extraction_output",
    "deduplicate_entities",
    "deduplicate_relationships",
]

