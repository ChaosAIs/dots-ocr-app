-- Migration: Drop unused GraphRAG tables
-- Date: 2024-12-14
-- Description: Remove graphrag_doc_full, graphrag_chunks, graphrag_entities, graphrag_hyperedges
--              These tables are redundant because:
--              - Chunks are stored in Qdrant (dots_ocr_documents collection)
--              - Entities and relationships are stored in Neo4j
--              Keep graphrag_llm_cache for potential future LLM response caching

-- Drop indexes first
DROP INDEX IF EXISTS idx_graphrag_doc_full_workspace;
DROP INDEX IF EXISTS idx_graphrag_chunks_workspace;
DROP INDEX IF EXISTS idx_graphrag_chunks_full_doc;
DROP INDEX IF EXISTS idx_graphrag_entities_workspace;
DROP INDEX IF EXISTS idx_graphrag_entities_type;
DROP INDEX IF EXISTS idx_graphrag_hyperedges_workspace;
DROP INDEX IF EXISTS idx_graphrag_hyperedges_source;
DROP INDEX IF EXISTS idx_graphrag_hyperedges_target;

-- Drop tables
DROP TABLE IF EXISTS graphrag_hyperedges;
DROP TABLE IF EXISTS graphrag_entities;
DROP TABLE IF EXISTS graphrag_chunks;
DROP TABLE IF EXISTS graphrag_doc_full;

-- Note: graphrag_llm_cache is intentionally kept for future use

