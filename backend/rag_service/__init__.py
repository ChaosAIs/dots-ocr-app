# RAG Service Module
# Provides Agentic RAG Chatbot functionality for searching documents

from .local_qwen_embedding import LocalQwen3Embedding
from .vectorstore import get_vectorstore, get_retriever, delete_documents_by_source
from .markdown_chunker import chunk_markdown_file
from .indexer import (
    index_existing_documents,
    start_watching_output,
    trigger_embedding_for_document,
    reindex_document,
)
from .rag_agent import create_agent_executor
from .chat_api import router as chat_router

__all__ = [
    "LocalQwen3Embedding",
    "get_vectorstore",
    "get_retriever",
    "delete_documents_by_source",
    "chunk_markdown_file",
    "index_existing_documents",
    "start_watching_output",
    "trigger_embedding_for_document",
    "reindex_document",
    "create_agent_executor",
    "chat_router",
]

