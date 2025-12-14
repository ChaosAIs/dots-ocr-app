-- Migration: 003_rename_content_to_page_content.sql
-- Rename 'content' column to 'page_content' in graphrag tables for LangChain consistency
-- Run with: psql -U postgres -d dots_ocr -f 003_rename_content_to_page_content.sql

-- Rename content column in graphrag_doc_full table
ALTER TABLE graphrag_doc_full 
RENAME COLUMN content TO page_content;

-- Rename content column in graphrag_chunks table
ALTER TABLE graphrag_chunks 
RENAME COLUMN content TO page_content;

-- Note: graphrag_entities and graphrag_hyperedges use 'description' not 'content'
-- so they don't need to be changed

-- Add comment for documentation
COMMENT ON COLUMN graphrag_doc_full.page_content IS 'Full document content (LangChain standard field name)';
COMMENT ON COLUMN graphrag_chunks.page_content IS 'Chunk text content (LangChain standard field name)';

