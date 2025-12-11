# GraphRAG Usage Guide

This document describes how to set up and use the GraphRAG (Graph-Enhanced Retrieval-Augmented Generation) feature in the Dots OCR application.

## Overview

GraphRAG enhances traditional RAG by incorporating knowledge graph capabilities:

- **Entity Extraction**: Extracts entities (people, organizations, concepts) from documents
- **Relationship Discovery**: Identifies relationships between entities
- **Graph-Based Retrieval**: Uses graph traversal for more contextual search results
- **Query Mode Detection**: Automatically selects the best retrieval strategy

## Prerequisites

### Required Services

1. **PostgreSQL** (already configured for the app)
2. **Qdrant** (already configured for vector storage)
3. **Ollama** with qwen2.5 model (already configured)
4. **Neo4j** (optional, for advanced graph features)

### Dependencies

Install optional dependencies for full functionality:

```bash
cd backend
pip install asyncpg neo4j
```

## Configuration

### Environment Variables

Add these to your `backend/.env` file:

```bash
# GraphRAG Core Settings
GRAPH_RAG_ENABLED=true              # Enable/disable GraphRAG
GRAPH_RAG_DEFAULT_MODE=auto         # auto, local, global, hybrid, naive

# Entity Extraction Settings
GRAPH_RAG_MAX_GLEANING=3            # Max iterations for entity extraction
GRAPH_RAG_ENTITY_BATCH_SIZE=5       # Chunks per batch during extraction
GRAPH_RAG_TOP_K=60                  # Max entities/relationships to retrieve

# Neo4j Configuration (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
```

## Database Setup

Run the migration to create GraphRAG tables:

```bash
PGPASSWORD='your_password' psql -h localhost -p 6400 -U postgres -d dots_ocr \
  -f backend/db/migrations/002_add_graphrag_tables.sql
```

This creates the following tables:

- `graphrag_doc_full` - Full document content
- `graphrag_chunks` - Document chunks
- `graphrag_entities` - Extracted entities
- `graphrag_hyperedges` - Relationships between entities
- `graphrag_llm_cache` - LLM response cache

## Query Modes

GraphRAG supports four query modes:

| Mode       | Description                    | Best For                  |
| ---------- | ------------------------------ | ------------------------- |
| **LOCAL**  | Entity-focused retrieval       | "Who is X?", "What is Y?" |
| **GLOBAL** | Relationship-focused retrieval | "How does A relate to B?" |
| **HYBRID** | Both entity and relationship   | Complex questions         |
| **NAIVE**  | Traditional vector search      | Simple lookups, lists     |

### Automatic Mode Detection

When `GRAPH_RAG_DEFAULT_MODE=auto`, the system automatically detects the best mode using:

1. LLM-based analysis (more accurate)
2. Heuristic fallback (faster, pattern-based)

## API Usage

### Indexing Documents

Documents are automatically indexed with GraphRAG when uploaded (if enabled):

```python
from rag_service.indexer import index_document_now

# Index a document with GraphRAG
index_document_now(
    file_path="/path/to/document.md",
    workspace_id="my_workspace"
)
```

### Querying with GraphRAG

```python
from rag_service.rag_agent import search_documents

# Search with automatic mode detection
results = search_documents(
    query="Who is the CEO of Acme Corp?",
    workspace_id="my_workspace"
)
# GraphRAG context is included in results when enabled
```

### Direct GraphRAG Usage

```python
import asyncio
from rag_service.graph_rag import GraphRAG, QueryParam, QueryMode

# Create GraphRAG instance
graphrag = GraphRAG(workspace_id="my_workspace")

# Query with specific mode
async def query_graph():
    context = await graphrag.query(
        query="How does product A relate to product B?",
        param=QueryParam(mode=QueryMode.GLOBAL, top_k=20)
    )

    # Format for LLM consumption
    formatted = graphrag.format_context(context)
    return formatted

result = asyncio.run(query_graph())
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Query Processing                        │
├─────────────────────────────────────────────────────────────┤
│  Query Mode Detector  →  GraphRAG Orchestrator              │
│        ↓                       ↓                            │
│  LOCAL/GLOBAL/HYBRID     Entity + Relationship Retrieval    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Storage Layer                           │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL KV     Qdrant Vectors     Neo4j Graph           │
│  (entities,        (embeddings)       (relationships)       │
│   chunks, cache)                                             │
└─────────────────────────────────────────────────────────────┘
```

## Testing

Run the GraphRAG tests:

```bash
cd backend

# Unit tests (no external services required)
python -m pytest test/test_graphrag.py -v

# Integration tests (requires PostgreSQL, Neo4j)
python -m pytest test/test_graphrag_integration.py -v

# Skip slow tests (LLM-based extraction)
python -m pytest test/test_graphrag_integration.py -v -m "not slow"
```

## Troubleshooting

### Common Issues

1. **"GraphRAG context is empty"**

   - Check if `GRAPH_RAG_ENABLED=true` in `.env`
   - Verify documents have been indexed with GraphRAG
   - Check PostgreSQL tables for entities

2. **"Module not found: asyncpg"**

   - Install: `pip install asyncpg`

3. **"Module not found: neo4j"**

   - Install: `pip install neo4j`
   - Neo4j is optional; PostgreSQL-based storage works without it

4. **"Entity extraction produces no results"**
   - Verify Ollama is running with qwen2.5 model
   - Check document content is meaningful text
   - Try increasing `GRAPH_RAG_MAX_GLEANING`

### Debugging

Enable debug logging:

```python
import logging
logging.getLogger("rag_service.graph_rag").setLevel(logging.DEBUG)
```

Check entity counts in database:

```sql
SELECT COUNT(*) FROM graphrag_entities WHERE workspace_id = 'your_workspace';
SELECT COUNT(*) FROM graphrag_hyperedges WHERE workspace_id = 'your_workspace';
```

## Performance Tuning

### For Large Documents

- Decrease `GRAPH_RAG_ENTITY_BATCH_SIZE` for memory efficiency
- Increase `GRAPH_RAG_MAX_GLEANING` for better extraction coverage
- Use `QueryMode.NAIVE` for simple factual queries

### For High Query Volume

- Enable LLM response caching (automatic with `graphrag_llm_cache` table)
- Use heuristic mode detection instead of LLM-based
- Reduce `GRAPH_RAG_TOP_K` for faster retrieval

## Feature Flag

To disable GraphRAG without removing configuration:

```bash
GRAPH_RAG_ENABLED=false
```

The application will fall back to standard vector-based RAG when disabled.
