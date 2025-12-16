"""
Selective re-indexing module for failed chunks.

This module provides functions to re-index only failed chunks/pages,
skipping successful ones for performance optimization.

Key Features:
- Retrieves failed chunks from Qdrant (for GraphRAG failures) or markdown files (for vector failures)
- Re-indexes only failed items, not the entire document
- Updates granular status after successful re-indexing
- Supports selective re-indexing by phase (vector, metadata, GraphRAG)
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain.schema import Document

logger = logging.getLogger(__name__)


def get_chunks_from_markdown_files(
    source_name: str,
    output_dir: str,
    file_paths: List[str]
) -> List[Document]:
    """
    Re-read and chunk specific markdown files.

    Used for vector indexing failures where chunks are NOT in Qdrant.

    Args:
        source_name: Document source name
        output_dir: Output directory path
        file_paths: List of markdown file paths to re-chunk

    Returns:
        List of Document objects with chunks from specified files
    """
    from .markdown_chunker import chunk_markdown_with_summaries

    all_chunks = []

    for file_path in file_paths:
        try:
            logger.info(f"[Selective Re-index] Re-chunking file: {file_path}")
            result = chunk_markdown_with_summaries(file_path, source_name, generate_summaries=True)

            if result.chunks:
                all_chunks.extend(result.chunks)
                logger.info(f"[Selective Re-index] Got {len(result.chunks)} chunks from {file_path}")

        except Exception as e:
            logger.error(f"[Selective Re-index] Failed to re-chunk {file_path}: {e}")

    return all_chunks


def get_failed_chunks_for_reindex(
    doc,
    source_name: str,
    output_dir: str,
    phase: str
) -> List[Document]:
    """
    Retrieve failed chunks based on failure type (hybrid approach).

    Args:
        doc: Document model instance
        source_name: Document source name
        output_dir: Output directory path
        phase: "vector", "graphrag", or "metadata"

    Returns:
        List of Document objects with chunk content
    """
    from .vectorstore import get_chunks_by_ids, get_all_chunks_for_source
    from db.document_repository import DocumentRepository

    if not doc.indexing_details:
        logger.warning(f"[Selective Re-index] No indexing_details for {doc.filename}")
        return []

    if phase == "vector":
        # Vector failures: chunks NOT in Qdrant
        # Must re-read from markdown files
        vector = doc.indexing_details.get("vector_indexing", {})
        failed_pages = vector.get("pages", {})

        file_paths = [
            page_info["file_path"]
            for page_info in failed_pages.values()
            if page_info.get("status") == "failed"
        ]

        if not file_paths:
            logger.info(f"[Selective Re-index] No failed pages for vector indexing")
            return []

        logger.info(f"[Selective Re-index] Re-indexing {len(file_paths)} failed pages for vector")
        return get_chunks_from_markdown_files(source_name, output_dir, file_paths)

    elif phase == "graphrag":
        # GraphRAG failures: chunks ARE in Qdrant
        # Fast retrieval from Qdrant
        graphrag = doc.indexing_details.get("graphrag_indexing", {})
        failed_chunk_ids = [
            chunk_id
            for chunk_id, chunk_info in graphrag.get("chunks", {}).items()
            if chunk_info.get("status") == "failed"
        ]

        if not failed_chunk_ids:
            logger.info(f"[Selective Re-index] No failed chunks for GraphRAG")
            return []

        logger.info(f"[Selective Re-index] Re-indexing {len(failed_chunk_ids)} failed chunks for GraphRAG")
        return get_chunks_by_ids(failed_chunk_ids, [source_name])

    elif phase == "metadata":
        # Metadata is document-level, need all chunks
        vector = doc.indexing_details.get("vector_indexing", {})

        if vector.get("status") == "completed":
            # Get all chunks from Qdrant
            logger.info(f"[Selective Re-index] Getting all chunks from Qdrant for metadata extraction")
            return get_all_chunks_for_source(source_name)
        else:
            # Fall back to markdown files
            logger.info(f"[Selective Re-index] Getting all chunks from markdown for metadata extraction")
            doc_dir = Path(output_dir) / source_name
            file_paths = [str(f) for f in doc_dir.glob("*_nohf.md")]
            return get_chunks_from_markdown_files(source_name, output_dir, file_paths)

    return []


def reindex_failed_vector_pages(
    doc,
    source_name: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Re-index failed pages for vector indexing.

    Args:
        doc: Document model instance
        source_name: Document source name
        output_dir: Output directory path

    Returns:
        Dictionary with re-indexing results
    """
    from .vectorstore import get_vectorstore
    from db.database import get_db_session
    from db.document_repository import DocumentRepository

    logger.info(f"[Selective Re-index] Starting vector re-indexing for {doc.filename}")

    # Get failed chunks
    failed_chunks = get_failed_chunks_for_reindex(doc, source_name, output_dir, "vector")

    if not failed_chunks:
        logger.info(f"[Selective Re-index] No failed chunks to re-index for vector")
        return {"pages_reindexed": 0, "chunks_reindexed": 0}

    # Group chunks by page (file_path)
    chunks_by_page = {}
    for chunk in failed_chunks:
        file_path = chunk.metadata.get("file_path", "")
        if file_path not in chunks_by_page:
            chunks_by_page[file_path] = []
        chunks_by_page[file_path].append(chunk)

    pages_reindexed = 0
    chunks_reindexed = 0

    # Re-index each page
    for file_path, page_chunks in chunks_by_page.items():
        try:
            # Extract page number
            page_number = page_chunks[0].metadata.get("page") if page_chunks else None
            chunk_ids = [c.metadata.get("chunk_id") for c in page_chunks]

            # Re-index to Qdrant
            vectorstore = get_vectorstore()
            vectorstore.add_documents(page_chunks)

            # Update status: success
            with get_db_session() as db:
                repo = DocumentRepository(db)
                doc = repo.get_by_filename(doc.filename)
                if doc:
                    repo.update_vector_indexing_status(
                        doc,
                        page_file_path=file_path,
                        chunk_ids=chunk_ids,
                        status="success",
                        page_number=page_number
                    )

            pages_reindexed += 1
            chunks_reindexed += len(page_chunks)
            logger.info(f"[Selective Re-index] Re-indexed page {file_path}: {len(page_chunks)} chunks")

        except Exception as e:
            logger.error(f"[Selective Re-index] Failed to re-index page {file_path}: {e}")

            # Update status: still failed
            try:
                with get_db_session() as db:
                    repo = DocumentRepository(db)
                    doc = repo.get_by_filename(doc.filename)
                    if doc:
                        chunk_ids = [c.metadata.get("chunk_id") for c in page_chunks]
                        page_number = page_chunks[0].metadata.get("page") if page_chunks else None
                        repo.update_vector_indexing_status(
                            doc,
                            page_file_path=file_path,
                            chunk_ids=chunk_ids,
                            status="failed",
                            error=str(e),
                            page_number=page_number
                        )
            except Exception as db_error:
                logger.warning(f"Could not update failed status: {db_error}")

    logger.info(
        f"[Selective Re-index] Vector re-indexing complete: "
        f"{pages_reindexed} pages, {chunks_reindexed} chunks"
    )

    return {
        "pages_reindexed": pages_reindexed,
        "chunks_reindexed": chunks_reindexed
    }



def reindex_failed_graphrag_chunks(
    doc,
    source_name: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Re-index failed chunks for GraphRAG.

    Args:
        doc: Document model instance
        source_name: Document source name
        output_dir: Output directory path

    Returns:
        Dictionary with re-indexing results
    """
    from .graph_rag import index_chunks_sync, GRAPH_RAG_ENABLED

    if not GRAPH_RAG_ENABLED:
        logger.warning(f"[Selective Re-index] GraphRAG is disabled, skipping")
        return {"chunks_reindexed": 0, "entities_extracted": 0, "relationships_extracted": 0}

    logger.info(f"[Selective Re-index] Starting GraphRAG re-indexing for {doc.filename}")

    # Get failed chunks from Qdrant
    failed_chunks = get_failed_chunks_for_reindex(doc, source_name, output_dir, "graphrag")

    if not failed_chunks:
        logger.info(f"[Selective Re-index] No failed chunks to re-index for GraphRAG")
        return {"chunks_reindexed": 0, "entities_extracted": 0, "relationships_extracted": 0}

    # Convert to GraphRAG format
    graphrag_chunks = [
        {
            "id": chunk.metadata.get("chunk_id"),
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        }
        for chunk in failed_chunks
    ]

    # Re-index with GraphRAG
    try:
        num_entities, num_rels = index_chunks_sync(
            graphrag_chunks,
            source_name,
            filename_with_ext=doc.filename
        )

        logger.info(
            f"[Selective Re-index] GraphRAG re-indexing complete: "
            f"{len(failed_chunks)} chunks, {num_entities} entities, {num_rels} relationships"
        )

        return {
            "chunks_reindexed": len(failed_chunks),
            "entities_extracted": num_entities,
            "relationships_extracted": num_rels
        }

    except Exception as e:
        logger.error(f"[Selective Re-index] GraphRAG re-indexing failed: {e}", exc_info=True)
        return {"chunks_reindexed": 0, "entities_extracted": 0, "relationships_extracted": 0, "error": str(e)}


def reindex_metadata(
    doc,
    source_name: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Re-extract metadata for a document.

    Args:
        doc: Document model instance
        source_name: Document source name
        output_dir: Output directory path

    Returns:
        Dictionary with re-indexing results
    """
    from .graph_rag.metadata_extractor import HierarchicalMetadataExtractor
    from db.database import get_db_session
    from db.document_repository import DocumentRepository
    import asyncio

    logger.info(f"[Selective Re-index] Starting metadata re-extraction for {doc.filename}")

    # Get all chunks
    all_chunks = get_failed_chunks_for_reindex(doc, source_name, output_dir, "metadata")

    if not all_chunks:
        logger.warning(f"[Selective Re-index] No chunks available for metadata extraction")
        return {"status": "failed", "error": "No chunks available"}

    # Convert to metadata extractor format
    chunks_data = [
        {
            "id": chunk.metadata.get("chunk_id"),
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        }
        for chunk in all_chunks
    ]

    # Extract metadata
    try:
        extractor = HierarchicalMetadataExtractor()

        # Run async extraction
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        metadata = loop.run_until_complete(
            extractor.extract_metadata(
                chunks=chunks_data,
                source_name=source_name,
                batch_size=10,
                progress_callback=None
            )
        )

        # Save to database
        with get_db_session() as db:
            repo = DocumentRepository(db)
            doc = repo.get_by_filename(doc.filename)
            if doc:
                repo.update_document_metadata(doc, metadata)
                repo.update_metadata_extraction_status(doc, "completed")

        logger.info(
            f"[Selective Re-index] Metadata re-extraction complete: "
            f"{metadata.get('document_type')} | {metadata.get('subject_name')}"
        )

        return {
            "status": "completed",
            "document_type": metadata.get("document_type"),
            "subject_name": metadata.get("subject_name")
        }

    except Exception as e:
        logger.error(f"[Selective Re-index] Metadata re-extraction failed: {e}", exc_info=True)

        # Update status
        try:
            with get_db_session() as db:
                repo = DocumentRepository(db)
                doc = repo.get_by_filename(doc.filename)
                if doc:
                    repo.update_metadata_extraction_status(doc, "failed", error=str(e))
        except Exception as db_error:
            logger.warning(f"Could not update metadata failed status: {db_error}")

        return {"status": "failed", "error": str(e)}


