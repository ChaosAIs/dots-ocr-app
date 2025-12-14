-- Migration: 002_add_graphrag_tables.sql
-- Add GraphRAG tables for entity and relationship storage
-- Run with: psql -U postgres -d dots_ocr -f 002_add_graphrag_tables.sql

-- Full document content (for reference)
CREATE TABLE IF NOT EXISTS graphrag_doc_full (
    id VARCHAR(64) PRIMARY KEY,
    workspace_id VARCHAR(64) NOT NULL DEFAULT 'default',
    page_content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document chunks with embeddings reference
CREATE TABLE IF NOT EXISTS graphrag_chunks (
    id VARCHAR(64) PRIMARY KEY,
    workspace_id VARCHAR(64) NOT NULL DEFAULT 'default',
    full_doc_id VARCHAR(64) REFERENCES graphrag_doc_full(id) ON DELETE CASCADE,
    page_content TEXT NOT NULL,
    tokens INTEGER,
    chunk_order_index INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extracted entities
CREATE TABLE IF NOT EXISTS graphrag_entities (
    id VARCHAR(64) PRIMARY KEY,
    workspace_id VARCHAR(64) NOT NULL DEFAULT 'default',
    entity_name VARCHAR(512) NOT NULL,
    entity_type VARCHAR(128),
    description TEXT,
    source_chunk_id VARCHAR(64) REFERENCES graphrag_chunks(id) ON DELETE SET NULL,
    key_score INTEGER DEFAULT 50,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Hyperedges (relationships between entities)
CREATE TABLE IF NOT EXISTS graphrag_hyperedges (
    id VARCHAR(64) PRIMARY KEY,
    workspace_id VARCHAR(64) NOT NULL DEFAULT 'default',
    src_entity_id VARCHAR(64) REFERENCES graphrag_entities(id) ON DELETE CASCADE,
    tgt_entity_id VARCHAR(64) REFERENCES graphrag_entities(id) ON DELETE CASCADE,
    description TEXT,
    keywords TEXT,
    weight FLOAT DEFAULT 1.0,
    source_chunk_id VARCHAR(64) REFERENCES graphrag_chunks(id) ON DELETE SET NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- LLM response cache for entity extraction and query processing
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

-- graphrag_doc_full indexes
CREATE INDEX IF NOT EXISTS idx_graphrag_doc_full_workspace ON graphrag_doc_full(workspace_id);

-- graphrag_chunks indexes
CREATE INDEX IF NOT EXISTS idx_graphrag_chunks_workspace ON graphrag_chunks(workspace_id);
CREATE INDEX IF NOT EXISTS idx_graphrag_chunks_doc ON graphrag_chunks(full_doc_id);
CREATE INDEX IF NOT EXISTS idx_graphrag_chunks_order ON graphrag_chunks(workspace_id, full_doc_id, chunk_order_index);

-- graphrag_entities indexes
CREATE INDEX IF NOT EXISTS idx_graphrag_entities_workspace ON graphrag_entities(workspace_id);
CREATE INDEX IF NOT EXISTS idx_graphrag_entities_name ON graphrag_entities(entity_name);
CREATE INDEX IF NOT EXISTS idx_graphrag_entities_type ON graphrag_entities(workspace_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_graphrag_entities_chunk ON graphrag_entities(source_chunk_id);

-- graphrag_hyperedges indexes
CREATE INDEX IF NOT EXISTS idx_graphrag_hyperedges_workspace ON graphrag_hyperedges(workspace_id);
CREATE INDEX IF NOT EXISTS idx_graphrag_hyperedges_src ON graphrag_hyperedges(src_entity_id);
CREATE INDEX IF NOT EXISTS idx_graphrag_hyperedges_tgt ON graphrag_hyperedges(tgt_entity_id);
CREATE INDEX IF NOT EXISTS idx_graphrag_hyperedges_chunk ON graphrag_hyperedges(source_chunk_id);

-- graphrag_llm_cache indexes
CREATE INDEX IF NOT EXISTS idx_graphrag_llm_cache_hash ON graphrag_llm_cache(workspace_id, prompt_hash);

-- Full-text search indexes for entity descriptions and names
CREATE INDEX IF NOT EXISTS idx_graphrag_entities_name_gin ON graphrag_entities USING gin(to_tsvector('english', entity_name));
CREATE INDEX IF NOT EXISTS idx_graphrag_entities_desc_gin ON graphrag_entities USING gin(to_tsvector('english', COALESCE(description, '')));
