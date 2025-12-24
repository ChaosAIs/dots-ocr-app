# GraphRAG Integration Upgrade Plan

## Integrating Graph-R1 Knowledge Graph RAG into dots-ocr-app

**Version:** 1.0
**Date:** 2024-12-10
**Author:** AI Assistant

---

> **⚠️ NOTE: This is an outdated planning document.**
>
> The actual implementation differs from this plan in the following ways:
> - **Entity/relationship embeddings** are stored natively in Neo4j (not in separate Qdrant collections)
> - **Qdrant** only contains: `documents` (document chunks), `metadatas` (document metadata for routing)
> - **Neo4j** stores entities and relationships with native vector indexes for semantic search
>
> For current architecture, see [GRAPHRAG_USAGE.md](./GRAPHRAG_USAGE.md).

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture](#2-current-architecture)
3. [Target Architecture](#3-target-architecture)
4. [Existing Resources to Reuse](#4-existing-resources-to-reuse)
5. [New Components](#5-new-components)
6. [Storage Design](#6-storage-design)
7. [Query Flow](#7-query-flow)
8. [Multi-Round Conversation Design](#8-multi-round-conversation-design)
9. [Reinforcement Learning Clarification](#9-reinforcement-learning-clarification)
10. [Implementation Phases](#10-implementation-phases)
11. [API Changes](#11-api-changes)
12. [Configuration](#12-configuration)
13. [Testing Strategy](#13-testing-strategy)
14. [Risk Mitigation](#14-risk-mitigation)
15. [Timeline](#15-timeline)

---

## 1. Executive Summary

This document outlines the complete upgrade plan to integrate **Graph-R1 Knowledge Graph RAG** capabilities into the existing **dots-ocr-app** solution. The integration enhances RAG quality by adding:

- **Knowledge Graph**: Entity and relationship extraction for structured understanding
- **Multi-mode Query**: LOCAL (entity-centric), GLOBAL (relationship-centric), HYBRID, NAIVE modes
- **LLM Query Mode Detection**: Automatic query classification using Option 1 (LLM-based)
- **Graph-based Retrieval**: Enhanced context retrieval using graph traversal

### Key Benefits

| Benefit                          | Description                                              |
| -------------------------------- | -------------------------------------------------------- |
| **Better Context Understanding** | Entities and relationships provide structured knowledge  |
| **Query-Optimized Retrieval**    | Different modes for different query types                |
| **Reduced Hallucination**        | Graph provides verified entity relationships             |
| **Scalable Architecture**        | Reuses existing Neo4j, Qdrant, PostgreSQL infrastructure |

---

## 2. Current Architecture

### 2.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Current dots-ocr-app                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐    ┌─────────────────────┐                         │
│  │   Document Upload   │───▶│   OCR Conversion    │                         │
│  │   (FastAPI)         │    │   (Qwen3/DeepSeek)  │                         │
│  └─────────────────────┘    └─────────────────────┘                         │
│                                       │                                      │
│                                       ▼                                      │
│                             ┌─────────────────────┐                         │
│                             │   Markdown Output   │                         │
│                             └─────────────────────┘                         │
│                                       │                                      │
│                                       ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                           RAG Service                                    ││
│  │  ┌───────────────┐   ┌───────────────┐   ┌───────────────────────────┐  ││
│  │  │   Indexer     │──▶│   Chunker     │──▶│   Qdrant Vector Store     │  ││
│  │  │               │   │               │   │   - documents             │  ││
│  │  │               │   │               │   │   - file_summaries        │  ││
│  │  │               │   │               │   │   - chunk_summaries       │  ││
│  │  └───────────────┘   └───────────────┘   └───────────────────────────┘  ││
│  │                                                      │                   ││
│  │                                                      ▼                   ││
│  │                                          ┌───────────────────────────┐  ││
│  │                                          │      RAG Agent            │  ││
│  │                                          │   (LangGraph + Ollama)    │  ││
│  │                                          │   - Query Analysis        │  ││
│  │                                          │   - 3-Phase Retrieval     │  ││
│  │                                          │   - Response Generation   │  ││
│  │                                          └───────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        PostgreSQL Database                               ││
│  │   - documents table (metadata, status)                                   ││
│  │   - document_status_log (audit trail)                                    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Current RAG Flow

```
User Query → Query Analysis (LLM) → Enhanced Query
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                          ▼
          Phase 1: File Summaries                    Phase 2: Chunk Summaries
          (Find relevant files)                      (Find relevant chunks)
                    │                                          │
                    └────────────────────┬────────────────────┘
                                         ▼
                              Phase 3: Full Chunks
                              (Retrieve chunk content)
                                         │
                                         ▼
                              LLM Response Generation
```

### 2.3 Current Limitations

| Limitation                | Impact                                      |
| ------------------------- | ------------------------------------------- |
| No entity understanding   | Can't answer "What entities are mentioned?" |
| No relationship awareness | Can't answer "How does X relate to Y?"      |
| Single retrieval strategy | Same approach for all query types           |
| No graph traversal        | Can't follow entity connections             |

---

## 3. Target Architecture

### 3.1 Enhanced Architecture with GraphRAG

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Enhanced dots-ocr-app with GraphRAG                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐    ┌─────────────────────┐                         │
│  │   Document Upload   │───▶│   OCR Conversion    │                         │
│  │   (FastAPI)         │    │   (Qwen3/DeepSeek)  │                         │
│  └─────────────────────┘    └─────────────────────┘                         │
│                                       │                                      │
│                                       ▼                                      │
│                             ┌─────────────────────┐                         │
│                             │   Markdown Output   │                         │
│                             └─────────────────────┘                         │
│                                       │                                      │
│              ┌────────────────────────┴─────────────────────────┐           │
│              ▼                                                   ▼           │
│  ┌───────────────────────────────────┐   ┌───────────────────────────────┐  │
│  │      Traditional Indexer          │   │     NEW: Entity Extractor     │  │
│  │  - Chunk markdown                 │   │  - LLM-based extraction       │  │
│  │  - Generate embeddings            │   │  - Entity identification      │  │
│  │  - Store in Qdrant                │   │  - Relationship detection     │  │
│  └───────────────────────────────────┘   └───────────────────────────────┘  │
│              │                                         │                     │
│              ▼                                         ▼                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Storage Layer                                   │  │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐  │  │
│  │  │  Qdrant VDB     │ │  PostgreSQL KV  │ │      Neo4j Graph        │  │  │
│  │  │  - documents    │ │  - chunks       │ │  - Entity nodes         │  │  │
│  │  │  - file_summaries│ │  - entities    │ │  - Relationship edges   │  │  │
│  │  │  - chunk_summaries│ │  - hyperedges │ │  - Graph traversal      │  │  │
│  │  │  - entity_embed │ │  - llm_cache    │ │                         │  │  │
│  │  │  - edge_embed   │ │                 │ │                         │  │  │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                       │                                      │
│                                       ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     NEW: GraphRAG Query Engine                         │  │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐  │  │
│  │  │ Query Mode      │ │ Graph Query     │ │   Enhanced RAG Agent    │  │  │
│  │  │ Detector (LLM)  │▶│ Orchestrator    │▶│   - Graph context       │  │  │
│  │  │ - LOCAL         │ │ - Entity search │ │   - Vector context      │  │  │
│  │  │ - GLOBAL        │ │ - Edge search   │ │   - Combined retrieval  │  │  │
│  │  │ - HYBRID        │ │ - Graph traverse│ │                         │  │  │
│  │  │ - NAIVE         │ │                 │ │                         │  │  │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Query Mode Decision Flow

```
                              User Query
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Query Mode Detector   │
                    │   (LLM-based Option 1)  │
                    └─────────────────────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
           ▼                      ▼                      ▼
    ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
    │    LOCAL    │       │   GLOBAL    │       │   HYBRID    │
    │  "Who is X?"│       │ "How does   │       │ "Explain    │
    │  "What is Y"│       │  X relate   │       │  everything │
    │             │       │  to Y?"     │       │  about X"   │
    └─────────────┘       └─────────────┘       └─────────────┘
           │                      │                      │
           ▼                      ▼                      ▼
    ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
    │ Entity VDB  │       │ Hyperedge   │       │ Both paths  │
    │ Search      │       │ VDB Search  │       │ combined    │
    │     │       │       │     │       │       │             │
    │     ▼       │       │     ▼       │       │             │
    │ Get Entity  │       │ Get Edge    │       │             │
    │ Description │       │ Entities    │       │             │
    │     │       │       │     │       │       │             │
    │     ▼       │       │     ▼       │       │             │
    │ 1-hop Edges │       │ Get Entity  │       │             │
    │             │       │ Descriptions│       │             │
    └─────────────┘       └─────────────┘       └─────────────┘
           │                      │                      │
           └──────────────────────┼──────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Combine with Chunk    │
                    │   Context (if needed)   │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   LLM Response Gen      │
                    └─────────────────────────┘
```

---

## 4. Existing Resources to Reuse

### 4.1 Infrastructure Services

| Service        | Current Usage                     | GraphRAG Usage                                        |
| -------------- | --------------------------------- | ----------------------------------------------------- |
| **PostgreSQL** | Document metadata, status logs    | KV Storage (chunks, entities, edges, LLM cache)       |
| **Qdrant**     | Vector embeddings (3 collections) | Extended with entity/edge embeddings (5 collections)  |
| **Neo4j**      | Available but unused              | Graph storage (entity nodes, relationship edges)      |
| **Ollama**     | LLM for OCR, RAG responses        | Entity extraction, query mode detection, response gen |

### 4.2 Existing Code to Extend

| Component           | File                                 | Extension                            |
| ------------------- | ------------------------------------ | ------------------------------------ |
| **Indexer**         | `backend/rag_service/indexer.py`     | Add entity extraction after chunking |
| **RAG Agent**       | `backend/rag_service/rag_agent.py`   | Add graph context to retrieval       |
| **Vector Store**    | `backend/rag_service/vectorstore.py` | Add entity/edge collections          |
| **Database Models** | `backend/db/models.py`               | Add GraphRAG tables                  |
| **Embeddings**      | `backend/rag_service/vectorstore.py` | Reuse `LocalQwen3Embedding`          |

### 4.3 Current Configuration (from .env)

```bash
# PostgreSQL - REUSE
POSTGRES_HOST=localhost
POSTGRES_PORT=6400
POSTGRES_DB=dots_ocr
POSTGRES_USER=dots_ocr
POSTGRES_PASSWORD=dots_ocr

# Qdrant - REUSE
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Ollama - REUSE
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_MODEL=qwen2.5:latest

# Neo4j - ADD NEW
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

---

## 5. New Components

### 5.1 Directory Structure

```
backend/
├── rag_service/
│   ├── indexer.py                    # MODIFY: Add entity extraction
│   ├── rag_agent.py                  # MODIFY: Add graph context
│   ├── vectorstore.py                # MODIFY: Add new collections
│   │
│   ├── graph_rag/                    # NEW DIRECTORY
│   │   ├── __init__.py
│   │   ├── base.py                   # Abstract base classes
│   │   ├── prompts.py                # All LLM prompts
│   │   ├── query_mode_detector.py    # LLM-based mode detection
│   │   ├── entity_extractor.py       # Entity/relationship extraction
│   │   ├── graph_rag.py              # Main orchestrator
│   │   └── utils.py                  # Utility functions
│   │
│   └── storage/                      # NEW DIRECTORY
│       ├── __init__.py
│       ├── postgres_kv_storage.py    # PostgreSQL KV implementation
│       ├── neo4j_storage.py          # Neo4j graph implementation
│       └── qdrant_adapter.py         # Qdrant vector adapter
│
├── db/
│   ├── models.py                     # MODIFY: Add GraphRAG models
│   └── migrations/
│       └── 002_add_graphrag_tables.sql  # NEW: Migration script
```

### 5.2 New Python Classes

#### 5.2.1 Base Classes (`graph_rag/base.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional, Any

@dataclass
class QueryParam:
    """Query parameters for GraphRAG"""
    mode: Literal["local", "global", "hybrid", "naive"] = "hybrid"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    top_k: int = 60
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 4000
    max_token_for_local_context: int = 4000

class BaseKVStorage(ABC):
    """Abstract base class for Key-Value storage"""

    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[dict]: ...

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict]: ...

    @abstractmethod
    async def filter_keys(self, keys: list[str]) -> set[str]: ...

    @abstractmethod
    async def upsert(self, data: dict[str, dict]) -> None: ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None: ...

class BaseVectorStorage(ABC):
    """Abstract base class for Vector storage"""

    @abstractmethod
    async def query(self, query: str, top_k: int = 10) -> list[dict]: ...

    @abstractmethod
    async def upsert(self, data: dict[str, dict]) -> None: ...

class BaseGraphStorage(ABC):
    """Abstract base class for Graph storage"""

    @abstractmethod
    async def has_node(self, node_id: str) -> bool: ...

    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[dict]: ...

    @abstractmethod
    async def upsert_node(self, node_id: str, node_data: dict) -> None: ...

    @abstractmethod
    async def upsert_edge(self, src_id: str, tgt_id: str, edge_data: dict) -> None: ...

    @abstractmethod
    async def get_node_edges(self, node_id: str) -> list[tuple[str, str, dict]]: ...
```

#### 5.2.2 Query Mode Detector (`graph_rag/query_mode_detector.py`)

```python
from langchain_ollama import ChatOllama
from .prompts import QUERY_MODE_DETECTION_PROMPT

class QueryModeDetector:
    """LLM-based query mode detection (Option 1)"""

    def __init__(self, llm: ChatOllama):
        self.llm = llm

    async def detect_mode(self, query: str) -> tuple[str, str]:
        """
        Detect query mode and enhance query.

        Returns:
            tuple[str, str]: (mode, enhanced_query)
            mode is one of: "local", "global", "hybrid", "naive"
        """
        prompt = QUERY_MODE_DETECTION_PROMPT.format(query=query)
        response = await self.llm.ainvoke(prompt)

        # Parse response to extract mode and enhanced query
        mode, enhanced_query = self._parse_response(response.content)
        return mode, enhanced_query

    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response to extract mode and enhanced query"""
        lines = response.strip().split('\n')
        mode = "hybrid"  # default
        enhanced_query = ""

        for line in lines:
            if line.startswith("MODE:"):
                mode = line.replace("MODE:", "").strip().lower()
            elif line.startswith("QUERY:"):
                enhanced_query = line.replace("QUERY:", "").strip()

        # Validate mode
        if mode not in ["local", "global", "hybrid", "naive"]:
            mode = "hybrid"

        return mode, enhanced_query or query
```

---

## 6. Storage Design

### 6.1 PostgreSQL KV Storage Tables

```sql
-- Migration: 002_add_graphrag_tables.sql

-- Full document content (for reference)
CREATE TABLE IF NOT EXISTS graphrag_doc_full (
    id VARCHAR(64) PRIMARY KEY,
    workspace_id VARCHAR(64) NOT NULL DEFAULT 'default',
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document chunks with embeddings reference
CREATE TABLE IF NOT EXISTS graphrag_chunks (
    id VARCHAR(64) PRIMARY KEY,
    workspace_id VARCHAR(64) NOT NULL DEFAULT 'default',
    full_doc_id VARCHAR(64) REFERENCES graphrag_doc_full(id),
    content TEXT NOT NULL,
    tokens INTEGER,
    chunk_order_index INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extracted entities
CREATE TABLE IF NOT EXISTS graphrag_entities (
    id VARCHAR(64) PRIMARY KEY,
    workspace_id VARCHAR(64) NOT NULL DEFAULT 'default',
    entity_name VARCHAR(512) NOT NULL,
    entity_type VARCHAR(128),
    description TEXT,
    source_chunk_id VARCHAR(64) REFERENCES graphrag_chunks(id),
    key_score INTEGER DEFAULT 50,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Hyperedges (relationships)
CREATE TABLE IF NOT EXISTS graphrag_hyperedges (
    id VARCHAR(64) PRIMARY KEY,
    workspace_id VARCHAR(64) NOT NULL DEFAULT 'default',
    src_entity_id VARCHAR(64) REFERENCES graphrag_entities(id),
    tgt_entity_id VARCHAR(64) REFERENCES graphrag_entities(id),
    description TEXT,
    keywords TEXT,
    weight FLOAT DEFAULT 1.0,
    source_chunk_id VARCHAR(64) REFERENCES graphrag_chunks(id),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- LLM response cache
CREATE TABLE IF NOT EXISTS graphrag_llm_cache (
    id VARCHAR(64) PRIMARY KEY,
    workspace_id VARCHAR(64) NOT NULL DEFAULT 'default',
    prompt_hash VARCHAR(64) NOT NULL,
    response TEXT NOT NULL,
    model_name VARCHAR(128),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(workspace_id, prompt_hash)
);

-- Indexes for performance
CREATE INDEX idx_graphrag_chunks_workspace ON graphrag_chunks(workspace_id);
CREATE INDEX idx_graphrag_chunks_doc ON graphrag_chunks(full_doc_id);
CREATE INDEX idx_graphrag_entities_workspace ON graphrag_entities(workspace_id);
CREATE INDEX idx_graphrag_entities_name ON graphrag_entities(entity_name);
CREATE INDEX idx_graphrag_hyperedges_workspace ON graphrag_hyperedges(workspace_id);
CREATE INDEX idx_graphrag_hyperedges_src ON graphrag_hyperedges(src_entity_id);
CREATE INDEX idx_graphrag_hyperedges_tgt ON graphrag_hyperedges(tgt_entity_id);
CREATE INDEX idx_graphrag_llm_cache_hash ON graphrag_llm_cache(workspace_id, prompt_hash);
```

### 6.2 Qdrant Vector Collections

| Collection                 | Purpose                         | Embedding Content                  |
| -------------------------- | ------------------------------- | ---------------------------------- |
| `dots_ocr_documents`       | Existing chunk embeddings       | Chunk text                         |
| `dots_ocr_file_summaries`  | Existing file summaries         | File summary text                  |
| `dots_ocr_chunk_summaries` | Existing chunk summaries        | Chunk summary text                 |
| `dots_ocr_entities`        | **NEW** Entity embeddings       | Entity name + description          |
| `dots_ocr_hyperedges`      | **NEW** Relationship embeddings | Keywords + src + tgt + description |

### 6.3 Neo4j Graph Schema

```cypher
// Entity nodes
CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Entity node properties:
// - id: VARCHAR(64) - unique identifier
// - name: VARCHAR(512) - entity name
// - type: VARCHAR(128) - entity type (person, organization, etc.)
// - description: TEXT - entity description
// - source_chunk_id: VARCHAR(64) - source chunk reference
// - workspace_id: VARCHAR(64) - workspace isolation

// Relationship edges
// (Entity)-[:RELATES_TO {
//   id: VARCHAR(64),
//   description: TEXT,
//   keywords: TEXT,
//   weight: FLOAT,
//   source_chunk_id: VARCHAR(64)
// }]->(Entity)
```

---

## 7. Query Flow

### 7.1 Complete Query Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GraphRAG Query Flow                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 1: Query Mode Detection (LLM-based)                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  User Query: "How does the authentication system work?"                 ││
│  │                              │                                          ││
│  │                              ▼                                          ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │  LLM Query Mode Detector                                         │   ││
│  │  │  Prompt: "Analyze this query and determine:                      │   ││
│  │  │  1. Query type: LOCAL (entity-focused), GLOBAL (relationship),   │   ││
│  │  │     HYBRID (both), or NAIVE (simple lookup)                      │   ││
│  │  │  2. Enhanced query for better retrieval"                         │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  │                              │                                          ││
│  │                              ▼                                          ││
│  │  Output: MODE=hybrid, QUERY="authentication system architecture flow"  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  Step 2: Entity/Relationship Search (based on mode)                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                         ││
│  │  ┌─────────────────┐         ┌─────────────────┐                       ││
│  │  │ Entity VDB      │         │ Hyperedge VDB   │                       ││
│  │  │ Search          │         │ Search          │                       ││
│  │  │ (LOCAL path)    │         │ (GLOBAL path)   │                       ││
│  │  └────────┬────────┘         └────────┬────────┘                       ││
│  │           │                           │                                 ││
│  │           ▼                           ▼                                 ││
│  │  Found: "AUTH_SYSTEM",       Found: "AUTH_SYSTEM->USER_DB",            ││
│  │         "USER_DB",                   "AUTH_SYSTEM->TOKEN_SERVICE"      ││
│  │         "TOKEN_SERVICE"                                                 ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  Step 3: Graph Traversal (Neo4j)                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                         ││
│  │  For each entity found:                                                 ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │  MATCH (e:Entity {name: "AUTH_SYSTEM"})-[r]->(related)          │   ││
│  │  │  RETURN e, r, related                                            │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  │                                                                         ││
│  │  Result: Entity descriptions + 1-hop relationships                     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  Step 4: Chunk Context Retrieval (Qdrant)                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                         ││
│  │  Get source chunks for entities/relationships                          ││
│  │  Additional vector search for related chunks                           ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  Step 5: Context Assembly & Response Generation                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                         ││
│  │  Combined Context:                                                      ││
│  │  - Entity descriptions (from graph)                                     ││
│  │  - Relationship descriptions (from graph)                               ││
│  │  - Source chunks (from vector store)                                    ││
│  │                              │                                          ││
│  │                              ▼                                          ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │  LLM Response Generation                                         │   ││
│  │  │  "Based on the context, the authentication system..."           │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Mode-Specific Retrieval Strategies

| Mode       | Entity Search | Edge Search | Graph Traversal        | Chunk Retrieval         |
| ---------- | ------------- | ----------- | ---------------------- | ----------------------- |
| **LOCAL**  | ✅ Primary    | ❌ Skip     | ✅ 1-hop from entities | ✅ Entity source chunks |
| **GLOBAL** | ❌ Skip       | ✅ Primary  | ✅ Get edge entities   | ✅ Edge source chunks   |
| **HYBRID** | ✅ Yes        | ✅ Yes      | ✅ Both paths          | ✅ All source chunks    |
| **NAIVE**  | ❌ Skip       | ❌ Skip     | ❌ Skip                | ✅ Direct vector search |

---

## 8. Multi-Round Conversation Design

Graph-R1 uses **two distinct multi-round conversation patterns** to improve accuracy:

### 8.1 Entity Extraction Gleaning Loop (Indexing Phase)

During document indexing, Graph-R1 uses an iterative "gleaning" loop to extract entities and relationships more thoroughly. This pattern ensures no important knowledge fragments are missed.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Entity Extraction Gleaning Loop                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 1: Initial Extraction                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  LLM extracts entities from document chunk                            │   │
│  │  → "(entity|person|John Smith|CEO of Acme Corp|90)"                  │   │
│  │  → "(hyper-relation|John Smith founded Acme Corp in 2010|8)"         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                         │
│  Step 2: Build Conversation History                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  history = [                                                          │   │
│  │    {"role": "user", "content": "<initial extraction prompt>"},       │   │
│  │    {"role": "assistant", "content": "<extracted entities>"}          │   │
│  │  ]                                                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                         │
│  Step 3: Continue Prompt (Gleaning)                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Prompt: "MANY knowledge fragments with entities were missed in the   │   │
│  │           last extraction. Add them below using the same format:"    │   │
│  │                                                                       │   │
│  │  LLM finds additional entities it missed initially                    │   │
│  │  → "(entity|organization|Acme Corp|Technology company|85)"           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                         │
│  Step 4: Check if More Iterations Needed                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Prompt: "Please check whether knowledge fragments cover all the      │   │
│  │           given text. Answer YES | NO if there are knowledge         │   │
│  │           fragments that need to be added."                          │   │
│  │                                                                       │   │
│  │  If YES → Go back to Step 3 (up to max_gleaning iterations)          │   │
│  │  If NO  → Exit loop with all accumulated entities                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation Pattern:**

```python
# From Graph-R1: operate.py (lines 324-339)
async def extract_entities_with_gleaning(chunk: str, max_gleaning: int = 3):
    # Step 1: Initial extraction
    final_result = await use_llm_func(hint_prompt)
    history = pack_user_ass_to_openai_messages(hint_prompt, final_result)

    # Steps 2-4: Gleaning loop
    for now_glean_index in range(max_gleaning):
        # Continue extraction
        glean_result = await use_llm_func(continue_prompt, history_messages=history)
        history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
        final_result += glean_result

        # Check if we should continue
        if now_glean_index == max_gleaning - 1:
            break
        if_loop_result = await use_llm_func(if_loop_prompt, history_messages=history)
        if if_loop_result.strip().lower() != "yes":
            break

    return parse_entities(final_result)
```

**Key Prompts:**

| Prompt                       | Purpose                                                                     |
| ---------------------------- | --------------------------------------------------------------------------- |
| `entity_extract_prompt`      | Initial entity extraction from chunk                                        |
| `entiti_continue_extraction` | "MANY knowledge fragments were missed. Add them below..."                   |
| `entiti_if_loop_extraction`  | "Please check whether knowledge fragments cover all text. Answer YES \| NO" |

### 8.2 Think-Query-Retrieve-Rethink Reasoning Cycle (Query Phase)

This is the **core innovation** of Graph-R1 - an iterative reasoning loop during query answering that allows the LLM to gather information step-by-step.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 "Think–Generate Query–Retrieve–Rethink" Cycle               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User Question: "Which magazine came out first, Tit-Bits or Illustreret?"   │
│                                    ↓                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ TURN 1: THINK                                                         │   │
│  │ <think>                                                               │   │
│  │   I need to find when Tit-Bits magazine was first published...       │   │
│  │ </think>                                                              │   │
│  │ <query>{"query": "Tit-Bits magazine founding date"}</query>          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ RETRIEVE: Knowledge from graph                                        │   │
│  │ <knowledge>                                                           │   │
│  │   {"<knowledge>": "Tit-Bits was founded in 1881", "<coherence>": 0.9}│   │
│  │ </knowledge>                                                          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ TURN 2: RETHINK (with accumulated context)                            │   │
│  │ <think>                                                               │   │
│  │   Now I know Tit-Bits was founded in 1881. I need to find when       │   │
│  │   Illustreret Nyhedsblad was founded to compare...                   │   │
│  │ </think>                                                              │   │
│  │ <query>{"query": "Illustreret Nyhedsblad founding date"}</query>     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ RETRIEVE: More knowledge from graph                                   │   │
│  │ <knowledge>                                                           │   │
│  │   {"<knowledge>": "Illustreret Nyhedsblad was founded in 1851"}      │   │
│  │ </knowledge>                                                          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ TURN 3: FINAL ANSWER (LLM decides it has enough information)          │   │
│  │ <think>                                                               │   │
│  │   Tit-Bits: 1881, Illustreret Nyhedsblad: 1851                       │   │
│  │   1851 < 1881, so Illustreret Nyhedsblad came first.                 │   │
│  │ </think>                                                              │   │
│  │ <answer>Illustreret Nyhedsblad</answer>                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Design Elements:**

| Element                      | Description                                    |
| ---------------------------- | ---------------------------------------------- |
| `<think>...</think>`         | LLM's visible chain-of-thought reasoning       |
| `<query>...</query>`         | Search query to retrieve from knowledge graph  |
| `<knowledge>...</knowledge>` | Retrieved knowledge injected back into context |
| `<answer>...</answer>`       | Final answer (terminates the loop)             |
| `MAX_TURNS`                  | Maximum iterations allowed (default: 20)       |

**Implementation Pattern:**

```python
# From Graph-R1: agent/vllm_infer/run.py (lines 167-198)
async def query_with_reasoning(question: str, max_turns: int = 10):
    messages = [{"role": "user", "content": build_initial_prompt(question)}]

    for turn in range(max_turns):
        # Get LLM response
        response = await llm.generate(messages)

        # Check if LLM wants to search or answer
        if "<query>" in response:
            # Extract query and retrieve from knowledge graph
            query = extract_query(response)
            knowledge = await graphrag.retrieve(query)

            # Append response and knowledge to conversation
            messages[0]["content"] += response + format_knowledge(knowledge)

        elif "<answer>" in response:
            # LLM has decided it has enough information
            return extract_answer(response)

    return "Unable to find answer within turn limit"
```

### 8.3 History Message Formatting

Graph-R1 uses a simple utility to convert prompts and responses into OpenAI message format:

```python
# From Graph-R1: graphr1/utils.py (lines 174-178)
def pack_user_ass_to_openai_messages(*args: str):
    """Convert alternating prompt/response pairs to OpenAI message format"""
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content}
        for i, content in enumerate(args)
    ]

# Example usage:
# pack_user_ass_to_openai_messages(prompt1, response1, prompt2, response2)
# Returns:
# [
#   {"role": "user", "content": prompt1},
#   {"role": "assistant", "content": response1},
#   {"role": "user", "content": prompt2},
#   {"role": "assistant", "content": response2}
# ]
```

### 8.4 Integration into dots-ocr-app

For our implementation, we will adopt both patterns:

| Pattern                          | Usage in dots-ocr-app             | Component             |
| -------------------------------- | --------------------------------- | --------------------- |
| **Gleaning Loop**                | Entity extraction during indexing | `entity_extractor.py` |
| **Think-Query-Retrieve-Rethink** | Complex query answering           | `rag_agent.py`        |

**Simplified Implementation for dots-ocr-app:**

```python
# In rag_service/graph_rag/entity_extractor.py
class EntityExtractor:
    async def extract_with_gleaning(self, chunk: str, max_gleaning: int = 2):
        """Extract entities with iterative gleaning for thoroughness"""
        history = []
        all_entities = await self._initial_extraction(chunk)
        history = self._build_history(chunk, all_entities)

        for i in range(max_gleaning):
            additional = await self._continue_extraction(history)
            if not additional:
                break
            all_entities.extend(additional)
            history = self._append_history(history, additional)

            should_continue = await self._check_continue(history)
            if not should_continue:
                break

        return self._deduplicate(all_entities)

# In rag_service/rag_agent.py
class EnhancedRAGAgent:
    async def query_with_reasoning(self, question: str, max_turns: int = 5):
        """Answer questions using iterative reasoning with graph retrieval"""
        context = self._build_initial_context(question)

        for turn in range(max_turns):
            response = await self._reason(context)

            if self._has_final_answer(response):
                return self._extract_answer(response)

            if self._needs_more_info(response):
                query = self._extract_search_query(response)
                knowledge = await self.graphrag.retrieve(query)
                context = self._append_knowledge(context, response, knowledge)

        # Fallback: generate answer with available context
        return await self._generate_final_answer(context)
```

---

## 9. Reinforcement Learning Clarification

### 9.1 Important: RL Training is OPTIONAL

Graph-R1 has **two separate components**:

| Component              | Purpose                                     | Required for dots-ocr-app?                  |
| ---------------------- | ------------------------------------------- | ------------------------------------------- |
| **GraphRAG Framework** | Entity extraction, graph storage, retrieval | ✅ **Required** - This is what we integrate |
| **RL Training**        | Fine-tune LLM to reason better with graphs  | ❌ **Optional** - Academic enhancement      |

### 9.2 What the RL Training Does

The RL training in Graph-R1 is a **research methodology** to improve an existing LLM's ability to:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     RL Training Process (OPTIONAL)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: Base LLM (e.g., Qwen2.5-3B-Instruct)                                │
│                         ↓                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  GRPO/REINFORCE++/PPO Training Loop                                  │   │
│  │  ────────────────────────────────────────────────────────────────    │   │
│  │  1. LLM generates: <think>...</think><query>...</query>              │   │
│  │  2. System retrieves knowledge from graph                            │   │
│  │  3. LLM continues reasoning with new knowledge                       │   │
│  │  4. LLM produces: <answer>...</answer>                               │   │
│  │  5. Compare answer with ground truth                                 │   │
│  │  6. Calculate reward:                                                │   │
│  │     - Format reward: Did LLM use correct <think>/<query>/<answer>?   │   │
│  │     - Answer reward: Is the answer correct (F1 score)?               │   │
│  │  7. Update LLM weights via policy gradient                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                         ↓                                                    │
│  Output: Fine-tuned LLM that reasons better with graphs                     │
│                                                                              │
│  Hardware Required: 4x 48GB GPUs (for training only)                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.3 The Reward Function

Graph-R1 uses a combined reward function:

```python
# From Graph-R1: verl/utils/reward_score/qa_em_and_format.py
def compute_score_format_answer(solution_str, ground_truth):
    # Format reward: Did the LLM follow the correct structure?
    format_reward = compute_score_format(solution_str)  # 0-1 score

    # Answer reward: Is the answer correct?
    answer_reward = compute_score_answer(solution_str, ground_truth)  # F1 score

    # Combined reward
    if format_reward == 1.0:
        return -1.0 + format_reward + answer_reward  # Range: 0 to 1
    else:
        return -1.0 + format_reward  # Penalty if format is wrong
```

### 9.4 Our Approach: Prompting-Based (No RL Required)

For dots-ocr-app, we use a **prompting-based approach** that works with your existing Qwen2.5 LLM:

| Aspect                | Without RL (Our Approach)  | With RL (Research)                      |
| --------------------- | -------------------------- | --------------------------------------- |
| **Setup Complexity**  | Simple - just prompts      | Complex - needs training infrastructure |
| **Hardware Required** | Any GPU running Ollama     | 4x 48GB GPUs for training               |
| **Answer Quality**    | Good (depends on base LLM) | Better (model learns optimal strategy)  |
| **Format Compliance** | Prompt engineering         | Model learns format reliably            |
| **When to Stop**      | Heuristic (max turns)      | Model learns optimal stopping           |
| **Recommended For**   | ✅ Production use          | Research or high-value domains          |

### 9.5 Why Prompting-Based Works for Us

1. ✅ **Your existing Qwen2.5:14b is capable** - Modern LLMs follow instructions well
2. ✅ **No additional training infrastructure** - Works with current Ollama setup
3. ✅ **The multi-turn reasoning works with prompts** - We just need good system prompts
4. ✅ **Entity extraction gleaning works with prompts** - Iterative extraction improves coverage
5. ✅ **Feature flag allows easy testing** - Can compare with/without GraphRAG

### 9.6 System Prompts for Multi-Turn Reasoning

```python
# In graph_rag/prompts.py

MULTI_TURN_REASONING_SYSTEM_PROMPT = """You are a helpful assistant that answers
questions using a knowledge base.

When answering, follow this process:
1. Think about what information you need in <think>...</think>
2. If you need to search the knowledge base, use: <query>{"query": "your search"}</query>
3. When you have enough information, provide your answer in <answer>...</answer>

Rules:
- Always show your reasoning in <think> tags
- You can search multiple times if needed
- Only provide <answer> when you are confident
- Keep searches focused and specific

Example:
<think>
The user asks about X. I need to find information about Y first.
</think>
<query>{"query": "information about Y"}</query>

[After receiving knowledge]

<think>
Now I know Y. Based on this, I can answer about X.
</think>
<answer>The answer is...</answer>
"""
```

### 9.7 Summary

| Decision                                   | Rationale                                      |
| ------------------------------------------ | ---------------------------------------------- |
| **No RL training required**                | Prompting works for production use             |
| **Use existing Qwen2.5:14b**               | Already available via Ollama                   |
| **Implement gleaning loop**                | Improves entity extraction coverage            |
| **Implement think-query-retrieve-rethink** | Enables complex query answering                |
| **Use heuristic stopping**                 | `max_turns` parameter (default: 5)             |
| **Can upgrade later**                      | RL training is an enhancement, not requirement |

---

## 10. Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Goal:** Set up storage infrastructure and base classes

| Task                         | Files                                    | Priority | Est. Days |
| ---------------------------- | ---------------------------------------- | -------- | --------- |
| Create directory structure   | `graph_rag/`, `storage/`                 | P0       | 0.5       |
| Implement base classes       | `graph_rag/base.py`                      | P0       | 1         |
| Create PostgreSQL KV storage | `storage/postgres_kv_storage.py`         | P0       | 2         |
| Create Neo4j graph storage   | `storage/neo4j_storage.py`               | P0       | 2         |
| Create Qdrant adapter        | `storage/qdrant_adapter.py`              | P0       | 1         |
| Add database migrations      | `migrations/002_add_graphrag_tables.sql` | P0       | 0.5       |
| Add Neo4j config to .env     | `.env`                                   | P0       | 0.5       |

**Deliverables:**

- All storage implementations working
- Database tables created
- Neo4j connection established

### Phase 2: Entity Extraction (Week 3-4)

**Goal:** Extract entities and relationships from documents

| Task                               | Files                           | Priority | Est. Days |
| ---------------------------------- | ------------------------------- | -------- | --------- |
| Create prompts file                | `graph_rag/prompts.py`          | P0       | 1         |
| Implement entity extractor         | `graph_rag/entity_extractor.py` | P0       | 3         |
| Implement utility functions        | `graph_rag/utils.py`            | P0       | 1         |
| Integrate with indexer             | `rag_service/indexer.py`        | P1       | 2         |
| Add entity/edge Qdrant collections | `rag_service/vectorstore.py`    | P1       | 1         |

**Deliverables:**

- Entity extraction working
- Entities stored in PostgreSQL, Qdrant, Neo4j
- Indexer creates knowledge graph during document processing

### Phase 3: Query Engine (Week 5-6)

**Goal:** Implement query mode detection and graph-based retrieval

| Task                            | Files                              | Priority | Est. Days |
| ------------------------------- | ---------------------------------- | -------- | --------- |
| Implement query mode detector   | `graph_rag/query_mode_detector.py` | P0       | 2         |
| Implement GraphRAG orchestrator | `graph_rag/graph_rag.py`           | P0       | 3         |
| Integrate with RAG agent        | `rag_service/rag_agent.py`         | P1       | 2         |
| Add feature flag                | `.env`, config                     | P1       | 0.5       |

**Deliverables:**

- Query mode detection working
- Graph-based retrieval integrated
- Feature flag for enable/disable

### Phase 4: Testing & Optimization (Week 7-8)

**Goal:** Comprehensive testing and performance optimization

| Task                             | Files                                | Priority | Est. Days |
| -------------------------------- | ------------------------------------ | -------- | --------- |
| Unit tests for storage           | `tests/test_storage.py`              | P1       | 2         |
| Unit tests for entity extraction | `tests/test_entity_extractor.py`     | P1       | 2         |
| Integration tests                | `tests/test_graphrag_integration.py` | P1       | 2         |
| Performance benchmarks           | `tests/benchmark_graphrag.py`        | P2       | 1         |
| Documentation                    | `docs/GRAPHRAG_USAGE.md`             | P2       | 1         |

**Deliverables:**

- All tests passing
- Performance benchmarks documented
- User documentation complete

---

## 11. API Changes

### 11.1 New Endpoints

```python
# No new endpoints required - GraphRAG is integrated into existing RAG flow
# The /api/rag/query endpoint will automatically use GraphRAG when enabled
```

### 11.2 Modified Endpoints

```python
# POST /api/rag/query
# Request body (unchanged):
{
    "query": "How does authentication work?",
    "workspace_id": "default"  # optional
}

# Response (enhanced):
{
    "response": "Based on the knowledge graph...",
    "sources": [...],
    "metadata": {
        "query_mode": "hybrid",           # NEW: detected mode
        "enhanced_query": "...",          # NEW: LLM-enhanced query
        "entities_found": ["AUTH_SYSTEM", "USER_DB"],  # NEW
        "relationships_found": [...]       # NEW
    }
}
```

### 11.3 Configuration Endpoint

```python
# GET /api/config
# Response (enhanced):
{
    "graph_rag_enabled": true,    # NEW
    "graph_rag_mode": "auto",     # NEW: auto, local, global, hybrid, naive
    ...
}
```

---

## 12. Configuration

### 12.1 Environment Variables

```bash
# Add to backend/.env

# ============================================
# GraphRAG Configuration
# ============================================

# Feature flag
GRAPH_RAG_ENABLED=true

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password
NEO4J_DATABASE=neo4j

# GraphRAG Settings
GRAPH_RAG_DEFAULT_MODE=auto          # auto, local, global, hybrid, naive
GRAPH_RAG_ENTITY_TOP_K=20            # Top entities to retrieve
GRAPH_RAG_EDGE_TOP_K=20              # Top edges to retrieve
GRAPH_RAG_CHUNK_TOP_K=10             # Top chunks per entity/edge

# Entity Extraction Settings
ENTITY_EXTRACT_MAX_GLEANING=2        # Max extraction iterations
ENTITY_SUMMARY_MAX_TOKENS=500        # Max tokens for entity summary

# Chunking Settings (for GraphRAG)
GRAPH_RAG_CHUNK_SIZE=1200            # Tokens per chunk
GRAPH_RAG_CHUNK_OVERLAP=100          # Overlap tokens
```

### 12.2 Feature Flag Logic

```python
# In rag_agent.py

import os

GRAPH_RAG_ENABLED = os.getenv("GRAPH_RAG_ENABLED", "false").lower() == "true"

async def query(self, user_query: str, workspace_id: str = "default"):
    if GRAPH_RAG_ENABLED:
        # Use GraphRAG pipeline
        return await self._graphrag_query(user_query, workspace_id)
    else:
        # Use existing vector-only pipeline (NAIVE mode)
        return await self._vector_query(user_query, workspace_id)
```

---

## 13. Testing Strategy

### 13.1 Unit Tests

| Component             | Test File                     | Coverage                             |
| --------------------- | ----------------------------- | ------------------------------------ |
| PostgreSQL KV Storage | `test_postgres_kv_storage.py` | CRUD operations, workspace isolation |
| Neo4j Graph Storage   | `test_neo4j_storage.py`       | Node/edge operations, traversal      |
| Qdrant Adapter        | `test_qdrant_adapter.py`      | Vector search, upsert                |
| Query Mode Detector   | `test_query_mode_detector.py` | Mode classification accuracy         |
| Entity Extractor      | `test_entity_extractor.py`    | Entity/relationship extraction       |

### 13.2 Integration Tests

| Test Scenario       | Description                                   |
| ------------------- | --------------------------------------------- |
| End-to-end indexing | Document → Chunks → Entities → Graph          |
| Query flow          | Query → Mode detection → Retrieval → Response |
| Mode switching      | Verify correct retrieval for each mode        |
| Fallback behavior   | GraphRAG disabled → NAIVE mode                |

### 13.3 Performance Benchmarks

| Metric                       | Target          | Measurement                      |
| ---------------------------- | --------------- | -------------------------------- |
| Query mode detection latency | < 500ms         | Time from query to mode decision |
| Entity extraction throughput | > 10 chunks/sec | Chunks processed per second      |
| Graph traversal latency      | < 100ms         | Time for 1-hop traversal         |
| End-to-end query latency     | < 3s            | Total query response time        |

---

## 14. Risk Mitigation

### 14.1 Identified Risks

| Risk                      | Impact         | Mitigation                      |
| ------------------------- | -------------- | ------------------------------- |
| Neo4j connection failures | Query failures | Fallback to NAIVE mode          |
| LLM extraction errors     | Bad entities   | Validation + retry logic        |
| Performance degradation   | Slow queries   | Caching, async operations       |
| Data inconsistency        | Wrong results  | Transaction support, validation |

### 14.2 Fallback Strategy

```python
async def query_with_fallback(query: str, mode: str = "auto"):
    try:
        if mode == "auto":
            mode = await detect_query_mode(query)

        if mode in ["local", "global", "hybrid"]:
            return await graphrag_query(query, mode)
        else:
            return await naive_query(query)

    except Neo4jConnectionError:
        logger.warning("Neo4j unavailable, falling back to NAIVE mode")
        return await naive_query(query)

    except Exception as e:
        logger.error(f"GraphRAG error: {e}, falling back to NAIVE mode")
        return await naive_query(query)
```

---

## 15. Timeline

### 15.1 Gantt Chart

```
Week 1  │████████████████████████████████│ Phase 1: Foundation (Storage)
Week 2  │████████████████████████████████│ Phase 1: Foundation (Storage)
Week 3  │████████████████████████████████│ Phase 2: Entity Extraction
Week 4  │████████████████████████████████│ Phase 2: Entity Extraction
Week 5  │████████████████████████████████│ Phase 3: Query Engine
Week 6  │████████████████████████████████│ Phase 3: Query Engine
Week 7  │████████████████████████████████│ Phase 4: Testing
Week 8  │████████████████████████████████│ Phase 4: Testing & Documentation
```

### 15.2 Milestones

| Milestone            | Target Date   | Deliverable                               |
| -------------------- | ------------- | ----------------------------------------- |
| M1: Storage Ready    | End of Week 2 | All storage implementations working       |
| M2: Extraction Ready | End of Week 4 | Entity extraction integrated with indexer |
| M3: Query Ready      | End of Week 6 | Full GraphRAG query pipeline working      |
| M4: Production Ready | End of Week 8 | All tests passing, documentation complete |

---

## Appendix A: Prompts

### A.1 Query Mode Detection Prompt

```python
QUERY_MODE_DETECTION_PROMPT = """You are a query analyzer for a knowledge graph RAG system.

Analyze the following user query and determine:
1. The best retrieval mode
2. An enhanced version of the query for better retrieval

MODES:
- LOCAL: Use when the query asks about a specific entity (person, organization, concept)
  Examples: "Who is John Smith?", "What is machine learning?", "Tell me about Company X"

- GLOBAL: Use when the query asks about relationships between entities
  Examples: "How does X relate to Y?", "What is the connection between A and B?"

- HYBRID: Use when the query requires both entity information and relationships
  Examples: "Explain the authentication system and how it connects to the database"

- NAIVE: Use for simple factual lookups or when graph context isn't needed
  Examples: "What is the date of the meeting?", "List all files"

USER QUERY: {query}

Respond in this exact format:
MODE: <mode>
QUERY: <enhanced query for retrieval, 50-100 words max>
"""
```

### A.2 Entity Extraction Prompt

See `Graph-R1-Full-Solution/graphr1/prompt.py` for the complete entity extraction prompt.

---

## Appendix B: Code Examples

### B.1 PostgreSQL KV Storage Implementation

```python
# storage/postgres_kv_storage.py

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from typing import Optional
from ..graph_rag.base import BaseKVStorage

class PostgreSQLKVStorage(BaseKVStorage):
    def __init__(self, session: AsyncSession, table_name: str, workspace_id: str = "default"):
        self.session = session
        self.table_name = table_name
        self.workspace_id = workspace_id

    async def get_by_id(self, id: str) -> Optional[dict]:
        # Implementation using SQLAlchemy async
        ...

    async def upsert(self, data: dict[str, dict]) -> None:
        # Batch upsert implementation
        ...
```

### B.2 Neo4j Graph Storage Implementation

```python
# storage/neo4j_storage.py

from neo4j import AsyncGraphDatabase
from typing import Optional
from ..graph_rag.base import BaseGraphStorage

class Neo4jStorage(BaseGraphStorage):
    def __init__(self, uri: str, user: str, password: str, workspace_id: str = "default"):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self.workspace_id = workspace_id

    async def upsert_node(self, node_id: str, node_data: dict) -> None:
        async with self.driver.session() as session:
            await session.run(
                """
                MERGE (e:Entity {id: $id, workspace_id: $workspace_id})
                SET e += $data
                """,
                id=node_id,
                workspace_id=self.workspace_id,
                data=node_data
            )

    async def get_node_edges(self, node_id: str) -> list[tuple[str, str, dict]]:
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {id: $id, workspace_id: $workspace_id})-[r]->(related)
                RETURN e.id, related.id, properties(r)
                """,
                id=node_id,
                workspace_id=self.workspace_id
            )
            return [(r["e.id"], r["related.id"], r["properties(r)"]) async for r in result]
```

---

## Appendix C: Comparison with Graph-R1

| Aspect              | Graph-R1                  | dots-ocr-app GraphRAG                |
| ------------------- | ------------------------- | ------------------------------------ |
| KV Storage          | JsonKVStorage, PostgreSQL | PostgreSQL (reuse existing)          |
| Vector Storage      | NanoVectorDB, Milvus      | Qdrant (reuse existing)              |
| Graph Storage       | NetworkX, Neo4j           | Neo4j (reuse existing)               |
| LLM                 | OpenAI, HuggingFace       | Ollama/qwen2.5 (reuse existing)      |
| Embeddings          | OpenAI                    | LocalQwen3Embedding (reuse existing) |
| Query Mode          | Manual parameter          | LLM-based auto-detection             |
| Async Support       | Full async                | Full async                           |
| Workspace Isolation | Yes                       | Yes                                  |

---

## Appendix D: Simplified RAG Architecture with Scopes (Implemented)

### D.1 Overview

As of December 2024, we implemented a **simplified RAG architecture** that removes chunk summaries and enhances file summaries with topic scopes. This provides better document filtering with less complexity.

```
BEFORE (Original 3-Phase):
┌─────────────────────────────────────────────────────────────────┐
│  File Summaries → Chunk Summaries → Raw Chunks → GraphRAG      │
│       ↓               ↓                ↓            ↓          │
│   (vectorized)    (vectorized)    (vectorized)  (vectorized)   │
└─────────────────────────────────────────────────────────────────┘

AFTER (Simplified with Scopes):
┌─────────────────────────────────────────────────────────────────┐
│  File Summaries (with scopes) → Raw Chunks → GraphRAG          │
│       ↓                            ↓            ↓              │
│   (vectorized)                (vectorized)  (vectorized)       │
│                                                                 │
│  New metadata: { scopes: ["topic1", "topic2", ...] }           │
└─────────────────────────────────────────────────────────────────┘
```

### D.2 Enhanced File Summary Structure

Each file summary now includes:

| Field          | Description                    | Example                                        |
| -------------- | ------------------------------ | ---------------------------------------------- |
| `summary`      | 500-word comprehensive summary | "This document covers..."                      |
| `scopes`       | 8-15 topic keywords            | ["authentication", "jwt", "security"]          |
| `content_type` | Document type                  | "technical_documentation", "tutorial", "guide" |
| `complexity`   | Complexity level               | "basic", "intermediate", "advanced"            |

### D.3 Query Flow with LLM Scope Matching

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENHANCED QUERY FLOW (LLM Scope Matching)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: QUERY ANALYSIS (LLM)                                               │
│  └── Input: "How do I validate JWT tokens?"                                │
│  └── Output:                                                                │
│      {                                                                      │
│        "enhanced_query": "JWT token validation authentication security",   │
│        "query_scopes": ["jwt", "authentication", "token validation"]       │
│      }                                                                      │
│                                                                             │
│  Step 2: FILE SUMMARY SEARCH + LLM SCOPE MATCHING                           │
│  └── Vector search on file summaries → Get candidates                      │
│  └── LLM matches query_scopes with document scopes                         │
│  └── Semantic matching: "login" ≈ "authentication" ✓                       │
│  └── Output: ["auth-guide.md", "security-docs.md"]                         │
│                                                                             │
│  Step 3: DIRECT CHUNK SEARCH (filtered by relevant files)                   │
│  └── Vector search on raw chunks                                           │
│  └── Filtered to only search within relevant files                         │
│                                                                             │
│  Step 4: GRAPHRAG CONTEXT                                                   │
│  └── Entity and relationship enrichment                                    │
│                                                                             │
│  Step 5: LLM RESPONSE GENERATION                                            │
│  └── Combine chunk content + graph context → Generate answer               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### D.4 LLM-Based Scope Matching

Instead of keyword matching, we use LLM to semantically match scopes:

| Keyword Matching           | LLM-Based Matching                      |
| -------------------------- | --------------------------------------- |
| "JWT" ≠ "JSON Web Token"   | "JWT" = "JSON Web Token" ✓              |
| "login" ≠ "authentication" | "login" ≈ "authentication" ✓            |
| "DB" ≠ "database"          | "DB" = "database" ✓                     |
| No semantic understanding  | Understands context and relationships ✓ |

### D.5 Map-Reduce for Large Files

For files larger than 8000 characters, we use a map-reduce approach:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MAP-REDUCE SUMMARIZATION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Large Document (> 8000 chars)                                              │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SPLIT: By markdown headers or size (max 4000 chars per section)    │   │
│  │  → Section 1, Section 2, Section 3, ... (max 20 sections)           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  MAP: Summarize each section independently                          │   │
│  │  → Summary 1, Summary 2, Summary 3, ...                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  REDUCE: Combine section summaries into final summary with scopes   │   │
│  │  → { summary: "...", scopes: [...], content_type: "...", ... }      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### D.6 Files Modified

| File                                 | Changes                                                                                                                              |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| `backend/rag_service/summarizer.py`  | Added `FileSummaryWithScopes` dataclass, `generate_file_summary_with_scopes()` with map-reduce support                               |
| `backend/rag_service/vectorstore.py` | Added `add_file_summary_with_scopes()`, `search_file_summaries_with_scopes()`, `get_all_file_summaries()`                            |
| `backend/rag_service/rag_agent.py`   | Added `_analyze_query_with_scopes()`, `_match_scopes_with_llm()`, `_find_relevant_files_with_scopes()`, updated `search_documents()` |
| `backend/rag_service/indexer.py`     | Updated to use `generate_file_summary_with_scopes()` and `add_file_summary_with_scopes()`                                            |

### D.7 Benefits of Simplified Architecture

| Aspect                 | Before                   | After                            |
| ---------------------- | ------------------------ | -------------------------------- |
| LLM calls per document | Many (chunk summaries)   | Few (file summary only)          |
| Indexing time          | Longer                   | Shorter                          |
| Storage overhead       | Higher (chunk summaries) | Lower                            |
| Query accuracy         | Good                     | Better (semantic scope matching) |
| Complexity             | Higher                   | Lower                            |

### D.8 Backward Compatibility

- The simplified architecture is fully backward compatible
- Existing chunk summary functions remain in the codebase
- New file summaries with scopes work alongside existing file summaries
- GraphRAG integration remains unchanged

---

**End of Document**
