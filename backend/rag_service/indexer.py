"""
Document indexer with three-phase indexing and watchdog-based file watching.
Monitors the output folder for new markdown files and indexes them.

Three-Phase Indexing Architecture:
==================================
Phase 1 (Synchronous/Background): Chunk embedding to Qdrant
    - Chunks markdown files and embeds them to Qdrant vector database
    - Makes documents immediately queryable after completion (~5 seconds)
    - Runs in background thread to not block main application

Phase 1.5 (Synchronous): Document metadata extraction
    - Extracts document metadata using hierarchical summarization
    - Identifies document type, subject, topics, and key entities
    - Saves metadata to PostgreSQL for intelligent document routing
    - Completes in ~30-60 seconds, making metadata available quickly
    - Runs synchronously to ensure metadata is available before GraphRAG

Phase 2 (Background/Async): GraphRAG entity extraction
    - Extracts entities and relationships using LLM
    - Stores to Neo4j graph database and Qdrant
    - Starts automatically after Phase 1.5 completes
    - Runs in separate background thread, non-blocking
    - Document remains queryable even if this phase fails

Benefits:
- Users can query documents immediately after Phase 1 completes (~5s)
- Metadata available for smart routing after Phase 1.5 (~60s)
- GraphRAG processing doesn't block user operations (~10min background)
- Graceful degradation if any phase fails
- Progressive enhancement: queryable → routable → graph-enhanced
"""

import os
import logging
import threading
from pathlib import Path
from typing import Set, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from .markdown_chunker import chunk_markdown_file, chunk_markdown_with_summaries
from .vectorstore import (
    get_vectorstore,
    delete_documents_by_source,
    delete_documents_by_file_path,
    is_document_indexed,
)

# Import database utilities for checking index status
# Note: These imports may fail if psycopg2 is not installed
try:
    from db.database import get_db_session
    from db.document_repository import DocumentRepository
    DB_AVAILABLE = True
except ImportError as e:
    DB_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Database not available for indexer: {e}")

# Import GraphRAG indexer for entity extraction
try:
    from .graph_rag import index_chunks_sync, GRAPH_RAG_ENABLED
    GRAPHRAG_AVAILABLE = True
except ImportError as e:
    GRAPHRAG_AVAILABLE = False
    GRAPH_RAG_ENABLED = False
    logging.getLogger(__name__).warning(f"GraphRAG not available for indexer: {e}")

logger = logging.getLogger(__name__)

# Output directory - can be set via set_output_dir() or defaults to backend/output
_output_dir: Optional[str] = None


def set_output_dir(output_dir: str):
    """Set the output directory for indexing."""
    global _output_dir
    _output_dir = output_dir
    logger.info(f"Output directory set to: {output_dir}")


def get_output_dir() -> str:
    """Get the output directory, with fallback to default."""
    if _output_dir:
        return _output_dir
    # Fallback to default based on file location
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")


def _is_document_fully_indexed_in_db(source_name: str) -> bool:
    """
    Check if a document is FULLY indexed in the database with granular status checking.
    Falls back to Qdrant check if database is not available.

    This function checks:
    1. Document-level index_status
    2. Granular indexing_details.vector_indexing status
    3. Individual page statuses to ensure no failed pages exist

    Args:
        source_name: The source/document name (filename without extension)

    Returns:
        True if the document is marked as fully indexed AND has no failed pages/chunks
    """
    if DB_AVAILABLE:
        try:
            with get_db_session() as db:
                repo = DocumentRepository(db)
                # Check if document exists and is fully indexed
                doc = _get_document_by_source(db, source_name)
                if not doc:
                    return False

                # First check document-level status
                if not repo.is_fully_indexed(doc):
                    return False

                # Check granular indexing_details for failed pages/chunks
                indexing_details = doc.indexing_details or {}
                vector_indexing = indexing_details.get("vector_indexing", {})

                # Check vector indexing status
                vector_status = vector_indexing.get("status")
                if vector_status not in ["completed", None]:
                    # If status is pending, failed, partial, or processing, not fully indexed
                    if vector_status in ["pending", "failed", "partial", "processing"]:
                        logger.debug(f"Document '{source_name}' has vector_status={vector_status}, not fully indexed")
                        return False

                # Check for any failed pages in granular tracking
                pages = vector_indexing.get("pages", {})
                for page_key, page_info in pages.items():
                    if page_info.get("status") == "failed":
                        logger.debug(f"Document '{source_name}' has failed page: {page_key}")
                        return False

                # Check GraphRAG status (optional, but good to verify)
                graphrag_indexing = indexing_details.get("graphrag_indexing", {})
                graphrag_status = graphrag_indexing.get("status")
                if graphrag_status in ["failed", "partial"]:
                    # Note: We don't block on GraphRAG failures for "fully indexed" check
                    # because vector indexing is the primary requirement
                    logger.debug(f"Document '{source_name}' has graphrag_status={graphrag_status} (not blocking)")

                return True
        except Exception as e:
            logger.warning(f"Database check failed for '{source_name}', falling back to Qdrant: {e}")

    # Fallback to Qdrant check if database is not available
    return is_document_indexed(source_name)


def _get_document_by_source(db, source_name: str):
    """Get document from database by source name."""
    from db.models import Document
    from sqlalchemy import and_

    return db.query(Document).filter(
        and_(
            Document.filename.like(f"{source_name}.%"),
            Document.deleted_at.is_(None)
        )
    ).first()


def _get_pending_pages_for_source(source_name: str) -> list:
    """
    Get list of pages that still need indexing for a document.
    Returns empty list if document is fully indexed or not found.
    """
    if not DB_AVAILABLE:
        return []

    try:
        with get_db_session() as db:
            repo = DocumentRepository(db)
            doc = _get_document_by_source(db, source_name)
            if doc:
                return repo.get_pending_pages(doc)
    except Exception as e:
        logger.warning(f"Could not get pending pages for '{source_name}': {e}")

    return []


def _get_successfully_indexed_page_files(source_name: str) -> Set[str]:
    """
    Get set of page file paths that have been successfully indexed.
    This is used to skip re-indexing of already successful pages.

    Args:
        source_name: The source/document name (filename without extension)

    Returns:
        Set of file paths (relative to doc directory) that are successfully indexed
    """
    if not DB_AVAILABLE:
        return set()

    try:
        with get_db_session() as db:
            doc = _get_document_by_source(db, source_name)
            if not doc:
                return set()

            indexing_details = doc.indexing_details or {}
            vector_indexing = indexing_details.get("vector_indexing", {})
            pages = vector_indexing.get("pages", {})

            successful_files = set()
            for page_key, page_info in pages.items():
                if page_info.get("status") == "success":
                    # Extract file path from page info
                    file_path = page_info.get("file_path")
                    if file_path:
                        successful_files.add(file_path)

            logger.debug(f"Found {len(successful_files)} successfully indexed pages for '{source_name}'")
            return successful_files
    except Exception as e:
        logger.warning(f"Could not get successfully indexed pages for '{source_name}': {e}")
        return set()


def _get_failed_page_files(source_name: str) -> Set[str]:
    """
    Get set of page file paths that have failed indexing.
    These pages need to be re-indexed.

    Args:
        source_name: The source/document name (filename without extension)

    Returns:
        Set of file paths (relative to doc directory) that failed indexing
    """
    if not DB_AVAILABLE:
        return set()

    try:
        with get_db_session() as db:
            doc = _get_document_by_source(db, source_name)
            if not doc:
                return set()

            indexing_details = doc.indexing_details or {}
            vector_indexing = indexing_details.get("vector_indexing", {})
            pages = vector_indexing.get("pages", {})

            failed_files = set()
            for page_key, page_info in pages.items():
                if page_info.get("status") == "failed":
                    # Extract file path from page info
                    file_path = page_info.get("file_path")
                    if file_path:
                        failed_files.add(file_path)

            logger.debug(f"Found {len(failed_files)} failed pages for '{source_name}'")
            return failed_files
    except Exception as e:
        logger.warning(f"Could not get failed pages for '{source_name}': {e}")
        return set()


# Track indexed files to avoid duplicates
_indexed_files: Set[str] = set()
_index_lock = threading.Lock()

# Global observer instance
_observer: Observer = None


class MarkdownFileHandler(FileSystemEventHandler):
    """Handles file system events for markdown files."""

    def on_created(self, event: FileSystemEvent):
        """Handle file creation events."""
        if event.is_directory:
            return
        self._process_file(event.src_path)

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events."""
        if event.is_directory:
            return
        self._process_file(event.src_path)

    def _process_file(self, file_path: str):
        """
        Process a markdown file for two-phase indexing.

        Phase 1: Chunk embedding to Qdrant (makes document queryable)
        Phase 2: GraphRAG entity extraction (runs in background)
        """
        # Only process _nohf.md files (converted markdown without headers/footers)
        if not file_path.endswith("_nohf.md"):
            return

        with _index_lock:
            # Check if already indexed (by modification time)
            file_stat = os.stat(file_path)
            file_key = f"{file_path}:{file_stat.st_mtime}"

            if file_key in _indexed_files:
                logger.debug(f"File already indexed: {file_path}")
                return

        try:
            # Get source name from directory name
            source_name = Path(file_path).parent.name
            logger.info(f"[Two-Phase] Processing file: {file_path} (source: {source_name})")

            # ========== PHASE 1: Chunk embedding to Qdrant ==========
            result = chunk_markdown_with_summaries(file_path, source_name, generate_summaries=True)

            if result.chunks:
                # Add chunks to Qdrant vectorstore
                vectorstore = get_vectorstore()
                vectorstore.add_documents(result.chunks)
                logger.info(f"[Phase 1] Indexed {len(result.chunks)} chunks from {source_name} to Qdrant")

                # Mark as indexed in tracking
                with _index_lock:
                    _indexed_files.add(file_key)

                # ========== PHASE 2: GraphRAG in background ==========
                if GRAPHRAG_AVAILABLE and GRAPH_RAG_ENABLED:
                    graphrag_chunks = [
                        {
                            "id": chunk.metadata.get("chunk_id", f"{source_name}_{i}"),
                            "page_content": chunk.page_content,
                            "metadata": chunk.metadata,
                        }
                        for i, chunk in enumerate(result.chunks)
                    ]
                    # Start GraphRAG in background (non-blocking)
                    _run_graphrag_background(
                        chunks_data=graphrag_chunks,
                        source_name=source_name,
                    )
            else:
                logger.warning(f"No chunks generated from {file_path}")

        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")


def index_existing_documents(output_dir: str = None):
    """
    Index all existing markdown files in the output directory using two-phase approach.
    Called on application startup.

    Phase 1: Chunk embedding to Qdrant (synchronous, makes documents queryable)
    Phase 2: GraphRAG entity extraction (background threads, non-blocking)

    Skips documents that are already FULLY indexed (checks database first, then Qdrant).

    Args:
        output_dir: Optional output directory path. If provided, sets it for future use.
    """
    if output_dir:
        set_output_dir(output_dir)

    current_output_dir = get_output_dir()
    output_path = Path(current_output_dir)

    if not output_path.exists():
        logger.warning(f"Output directory does not exist: {current_output_dir}")
        return 0

    total_chunks = 0
    total_files = 0
    skipped_files = 0

    logger.info(f"[Two-Phase] Starting indexing of existing documents in {current_output_dir}")

    # Track which source names we've already checked
    checked_sources: Set[str] = set()
    # Track chunks per source for Phase 2
    source_chunks_map: dict = {}

    # ========== PHASE 1: Index all chunks to Qdrant ==========
    logger.info("[Phase 1] Starting chunk embedding to Qdrant...")

    # Track successfully indexed pages per source to avoid re-indexing
    source_successful_pages: dict = {}

    for root, _, files in os.walk(output_path):
        for filename in files:
            if not filename.endswith("_nohf.md"):
                continue

            file_path = os.path.join(root, filename)

            try:
                source_name = Path(file_path).parent.name
                file_stat = os.stat(file_path)
                file_key = f"{file_path}:{file_stat.st_mtime}"

                with _index_lock:
                    if file_key in _indexed_files:
                        continue

                # Load successfully indexed pages for this source (once per source)
                if source_name not in checked_sources:
                    checked_sources.add(source_name)

                    # Check if document is fully indexed (with granular status check)
                    if _is_document_fully_indexed_in_db(source_name):
                        logger.info(f"[Phase 1] Source '{source_name}' already fully indexed, skipping all pages")
                        # Mark all files for this source as indexed
                        source_successful_pages[source_name] = "ALL"
                        with _index_lock:
                            _indexed_files.add(file_key)
                        skipped_files += 1
                        continue

                    # Document not fully indexed - get list of successfully indexed pages
                    successful_pages = _get_successfully_indexed_page_files(source_name)
                    source_successful_pages[source_name] = successful_pages

                    if successful_pages:
                        logger.info(f"[Phase 1] Source '{source_name}' has {len(successful_pages)} successfully indexed pages, will skip them")

                # Skip this file if it's already successfully indexed
                if source_name in source_successful_pages:
                    if source_successful_pages[source_name] == "ALL":
                        # All pages are indexed, skip
                        with _index_lock:
                            _indexed_files.add(file_key)
                        skipped_files += 1
                        continue
                    elif file_path in source_successful_pages[source_name]:
                        # This specific page is already successfully indexed, skip it
                        logger.debug(f"[Phase 1] Skipping already indexed page: {file_path}")
                        with _index_lock:
                            _indexed_files.add(file_key)
                        skipped_files += 1
                        continue

                # This page needs indexing - chunk and index to Qdrant
                logger.info(f"[Phase 1] Indexing page: {file_path}")
                result = chunk_markdown_with_summaries(file_path, source_name, generate_summaries=True)

                if result.chunks:
                    vectorstore = get_vectorstore()
                    vectorstore.add_documents(result.chunks)
                    total_chunks += len(result.chunks)
                    total_files += 1

                    # Collect chunks for Phase 2 (grouped by source)
                    if GRAPHRAG_AVAILABLE and GRAPH_RAG_ENABLED:
                        if source_name not in source_chunks_map:
                            source_chunks_map[source_name] = []
                        for i, chunk in enumerate(result.chunks):
                            source_chunks_map[source_name].append({
                                "id": chunk.metadata.get("chunk_id", f"{source_name}_{i}"),
                                "page_content": chunk.page_content,
                                "metadata": chunk.metadata,
                            })

                    with _index_lock:
                        _indexed_files.add(file_key)

                    # Update database status
                    if DB_AVAILABLE:
                        try:
                            _update_db_index_status(source_name, len(result.chunks))
                        except Exception as db_e:
                            logger.warning(f"Could not update database index status: {db_e}")

                    logger.debug(f"[Phase 1] Indexed {len(result.chunks)} chunks from {source_name}")

            except Exception as e:
                logger.error(f"[Phase 1] Error indexing {file_path}: {e}")

    logger.info(
        f"[Phase 1] Complete: {total_files} files indexed, {total_chunks} chunks in Qdrant, "
        f"{skipped_files} files skipped"
    )

    # ========== PHASE 2: Start GraphRAG in background for each source ==========
    if GRAPHRAG_AVAILABLE and GRAPH_RAG_ENABLED and source_chunks_map:
        logger.info(f"[Phase 2] Starting GraphRAG background processing for {len(source_chunks_map)} sources...")
        for source_name, chunks in source_chunks_map.items():
            _run_graphrag_background(
                chunks_data=chunks,
                source_name=source_name,
            )
        logger.info(f"[Phase 2] All GraphRAG background threads started (documents are queryable now)")

    return total_chunks


def _update_db_index_status(source_name: str, chunks_indexed: int):
    """
    Update the database index status for a document after successful indexing.

    Args:
        source_name: The source/document name (filename without extension)
        chunks_indexed: Number of chunks that were indexed
    """
    if not DB_AVAILABLE:
        return

    try:
        from db.models import Document, IndexStatus
        from sqlalchemy import and_

        with get_db_session() as db:
            repo = DocumentRepository(db)
            # Find document by source name (filename starts with source_name.)
            doc = db.query(Document).filter(
                and_(
                    Document.filename.like(f"{source_name}.%"),
                    Document.deleted_at.is_(None)
                )
            ).first()

            if doc:
                repo.update_index_status(
                    doc,
                    IndexStatus.INDEXED,
                    indexed_chunks=chunks_indexed,
                    message=f"Auto-indexed on startup ({chunks_indexed} chunks)"
                )
                logger.debug(f"Updated database index status for '{source_name}'")
    except Exception as e:
        logger.warning(f"Could not update database index status for '{source_name}': {e}")


def start_watching_output(output_dir: str = None):
    """
    Start watching the output directory for new markdown files.
    Uses watchdog to monitor file system changes.

    Args:
        output_dir: Optional output directory path. If provided, sets it for future use.
    """
    global _observer

    if output_dir:
        set_output_dir(output_dir)

    if _observer is not None:
        logger.warning("File watcher already running")
        return

    current_output_dir = get_output_dir()
    output_path = Path(current_output_dir)

    if not output_path.exists():
        logger.warning(f"Creating output directory: {current_output_dir}")
        output_path.mkdir(parents=True, exist_ok=True)

    _observer = Observer()
    event_handler = MarkdownFileHandler()
    _observer.schedule(event_handler, str(output_path), recursive=True)
    _observer.start()

    logger.info(f"Started watching {current_output_dir} for new documents")


def stop_watching():
    """Stop the file watcher."""
    global _observer

    if _observer is not None:
        _observer.stop()
        _observer.join()
        _observer = None
        logger.info("Stopped file watcher")


def get_indexed_count() -> int:
    """Get the number of indexed files."""
    with _index_lock:
        return len(_indexed_files)


def index_document_now(source_name: str, output_dir: str = None, run_graphrag: bool = True, run_metadata: bool = True, filename: str = None) -> int:
    """
    Index a specific document immediately with summarization.
    Deletes any existing embeddings for this source before re-indexing.

    This function performs:
    - Phase 1 (Qdrant embedding) synchronously
    - Phase 1.5 (Metadata extraction) synchronously if run_metadata=True
    - Phase 2 (GraphRAG) in background if run_graphrag=True

    Args:
        source_name: The document source name (folder name under output directory).
        output_dir: Optional output directory path.
        run_graphrag: Whether to run GraphRAG in background after Qdrant indexing (default: True)
        run_metadata: Whether to extract metadata after Qdrant indexing (default: True)
        filename: Original filename with extension for database tracking (optional)

    Returns:
        Number of chunks indexed to Qdrant.
    """
    # Use the shared Phase 1 function
    total_chunks, graphrag_chunks = _index_chunks_to_qdrant(source_name, output_dir, filename)

    # ========== PHASE 1.5: Metadata Extraction ==========
    if run_metadata and graphrag_chunks:
        try:
            logger.info(f"[Phase 1.5] Starting metadata extraction for {source_name}")

            # Import metadata extractor
            from .graph_rag.metadata_extractor import HierarchicalMetadataExtractor

            extractor = HierarchicalMetadataExtractor()

            # Run extraction (handle both sync and async contexts)
            import asyncio
            try:
                # Try to get the running event loop (if we're in an async context)
                loop = asyncio.get_running_loop()
                # We're in an async context, use create_task
                task = loop.create_task(
                    extractor.extract_metadata(
                        chunks=graphrag_chunks,
                        source_name=source_name,
                        batch_size=10,
                        progress_callback=None,
                    )
                )
                # Wait for the task to complete
                import concurrent.futures
                import threading

                # Run in a separate thread to avoid blocking
                result_container = []
                error_container = []

                def run_async():
                    try:
                        # Create a new event loop for this thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            result = new_loop.run_until_complete(
                                extractor.extract_metadata(
                                    chunks=graphrag_chunks,
                                    source_name=source_name,
                                    batch_size=10,
                                    progress_callback=None,
                                )
                            )
                            result_container.append(result)
                        finally:
                            new_loop.close()
                    except Exception as e:
                        error_container.append(e)

                thread = threading.Thread(target=run_async)
                thread.start()
                thread.join()

                if error_container:
                    raise error_container[0]
                metadata = result_container[0]

            except RuntimeError:
                # No running event loop, we're in a sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    metadata = loop.run_until_complete(
                        extractor.extract_metadata(
                            chunks=graphrag_chunks,
                            source_name=source_name,
                            batch_size=10,
                            progress_callback=None,
                        )
                    )
                finally:
                    loop.close()

            # Save metadata to PostgreSQL
            if DB_AVAILABLE:
                try:
                    # Try to find the document by source name (use filename param if available)
                    lookup_filename = filename if filename else f"{source_name}.pdf"
                    with get_db_session() as db:
                        repo = DocumentRepository(db)
                        doc = repo.get_by_filename(lookup_filename)
                        if doc:
                            repo.update_document_metadata(
                                doc,
                                metadata,
                                message=f"Extracted: {metadata.get('document_type', 'unknown')} - {metadata.get('subject_name', 'N/A')}"
                            )
                            # Update granular status: metadata extraction succeeded
                            repo.update_metadata_extraction_status(doc, "completed")
                            logger.info(
                                f"[Phase 1.5] Metadata saved for {source_name}: "
                                f"{metadata.get('document_type')} | "
                                f"Subject: {metadata.get('subject_name')} | "
                                f"Confidence: {metadata.get('confidence', 0):.2f}"
                            )
                        else:
                            logger.warning(f"[Phase 1.5] Document not found in database: {lookup_filename}")
                except Exception as e:
                    logger.error(f"Failed to save metadata to database: {e}", exc_info=True)

            logger.info(f"[Phase 1.5] Complete for {source_name}")

        except Exception as e:
            logger.error(f"[Phase 1.5] Metadata extraction failed for {source_name}: {e}", exc_info=True)

            # Update granular status: metadata extraction failed
            if DB_AVAILABLE:
                try:
                    lookup_filename = filename if filename else f"{source_name}.pdf"
                    with get_db_session() as db:
                        repo = DocumentRepository(db)
                        doc = repo.get_by_filename(lookup_filename)
                        if doc:
                            repo.update_metadata_extraction_status(doc, "failed", error=str(e))
                except Exception as db_error:
                    logger.warning(f"Could not update metadata extraction failed status: {db_error}")
            # Don't fail the entire indexing - metadata is optional

    # Start Phase 2 (GraphRAG) in background if enabled
    if run_graphrag and graphrag_chunks:
        _run_graphrag_background(
            chunks_data=graphrag_chunks,
            source_name=source_name,
            filename=filename,
        )

    logger.info(f"Document indexing complete for {source_name}: {total_chunks} chunks")
    return total_chunks


def reindex_document(source_name: str, output_dir: str = None) -> int:
    """
    Re-index a document by first deleting all existing embeddings for it,
    then creating new embeddings with summarization. Runs in a background thread.

    Args:
        source_name: The document source name (folder name under output directory).
        output_dir: Optional output directory path.

    Returns:
        Number of chunks indexed (0 if running in background).
    """
    import threading

    def _reindex_task():
        try:
            logger.info(f"Starting background re-indexing for: {source_name}")
            # First delete all existing embeddings for this source
            delete_documents_by_source(source_name)
            logger.info(f"Deleted existing embeddings for source: {source_name}")

            # Now index the document
            chunks = index_document_now(source_name, output_dir)
            logger.info(f"Background re-indexing complete for {source_name}: {chunks} chunks")
        except Exception as e:
            logger.error(f"Error in background re-indexing for {source_name}: {e}")

    # Run in background thread
    thread = threading.Thread(target=_reindex_task, daemon=True)
    thread.start()
    logger.info(f"Started background re-indexing thread for: {source_name}")
    return 0


def _run_graphrag_background(
    chunks_data: list,
    source_name: str,
    filename: str = None,
    conversion_id: str = None,
    broadcast_callback=None
):
    """
    Run GraphRAG indexing (entity extraction) in background thread.
    This is Phase 2 of the two-phase indexing process.

    Args:
        chunks_data: List of chunk dicts with 'id', 'content', 'metadata'
        source_name: The document source name
        filename: Original filename for database updates
        conversion_id: The conversion ID for WebSocket notifications
        broadcast_callback: Function for WebSocket broadcasts
    """
    if not GRAPHRAG_AVAILABLE or not GRAPH_RAG_ENABLED:
        logger.debug(f"[GraphRAG] Skipping background indexing - not enabled")
        return

    def _graphrag_task():
        try:
            logger.info(f"[GraphRAG Phase 2] Starting background entity extraction for: {source_name}")

            # Send WebSocket notification that GraphRAG started
            if broadcast_callback and conversion_id:
                broadcast_callback(conversion_id, {
                    "status": "graphrag_indexing",
                    "progress": 100,
                    "message": "Building knowledge graph (background)...",
                    "graphrag_status": "started",
                })

            # Perform GraphRAG indexing
            num_entities, num_rels = index_chunks_sync(chunks_data, source_name, filename_with_ext=filename)

            logger.info(
                f"[GraphRAG Phase 2] Complete for {source_name}: "
                f"{num_entities} entities, {num_rels} relationships"
            )

            # Send WebSocket notification that GraphRAG completed
            if broadcast_callback and conversion_id:
                broadcast_callback(conversion_id, {
                    "status": "graphrag_indexed",
                    "progress": 100,
                    "message": f"Knowledge graph built ({num_entities} entities, {num_rels} relationships)",
                    "graphrag_status": "completed",
                    "entities_extracted": num_entities,
                    "relationships_extracted": num_rels,
                })

        except Exception as e:
            logger.error(f"[GraphRAG Phase 2] Failed for {source_name}: {e}", exc_info=True)

            # Send WebSocket notification for error (non-blocking, document is still queryable)
            if broadcast_callback and conversion_id:
                broadcast_callback(conversion_id, {
                    "status": "graphrag_error",
                    "progress": 100,
                    "message": f"Knowledge graph building failed (document still searchable): {str(e)}",
                    "graphrag_status": "error",
                })

    # Run in background thread
    thread = threading.Thread(target=_graphrag_task, daemon=True, name=f"graphrag-{source_name}")
    thread.start()
    logger.info(f"[GraphRAG Phase 2] Started background thread for: {source_name}")


def _index_chunks_to_qdrant(source_name: str, output_dir: str = None, filename_with_ext: str = None, skip_successful_pages: bool = False) -> tuple:
    """
    Phase 1: Index all chunks to Qdrant vector database.
    This makes the document queryable immediately.

    Now tracks granular indexing status at page and chunk level for selective re-indexing.

    Args:
        source_name: The document source name
        output_dir: Optional output directory path
        filename_with_ext: Original filename with extension for database lookup
        skip_successful_pages: If True, skip pages that are already successfully indexed

    Returns:
        Tuple of (total_chunks, all_graphrag_chunks) where all_graphrag_chunks
        is a list of chunk dicts for GraphRAG Phase 2
    """
    if output_dir:
        set_output_dir(output_dir)

    current_output_dir = get_output_dir()
    doc_dir = Path(current_output_dir) / source_name

    if not doc_dir.exists():
        logger.warning(f"Document directory does not exist: {doc_dir}")
        return 0, []

    # Get document from database for granular status tracking
    doc = None
    repo = None
    if DB_AVAILABLE and filename_with_ext:
        try:
            with get_db_session() as db:
                repo = DocumentRepository(db)
                doc = repo.get_by_filename(filename_with_ext)
                if doc:
                    # Initialize indexing_details structure
                    repo.init_indexing_details(doc)
                    logger.info(f"[Phase 1] Tracking granular status for: {filename_with_ext}")
        except Exception as e:
            logger.warning(f"Could not get document from database: {e}")

    # Get list of successfully indexed pages to skip (if requested)
    successful_pages = set()
    if skip_successful_pages:
        successful_pages = _get_successfully_indexed_page_files(source_name)
        if successful_pages:
            logger.info(f"[Phase 1] Will skip {len(successful_pages)} successfully indexed pages for '{source_name}'")

    total_chunks = 0
    indexed_files = []
    skipped_files = []
    all_graphrag_chunks = []  # Collect all chunks for Phase 2

    # Find all _nohf.md files in the document directory
    for filename in os.listdir(doc_dir):
        if not filename.endswith("_nohf.md"):
            continue

        file_path = str(doc_dir / filename)

        # Skip this page if it's already successfully indexed
        if skip_successful_pages and file_path in successful_pages:
            logger.debug(f"[Phase 1] Skipping successfully indexed page: {file_path}")
            skipped_files.append(filename)
            continue

        # Extract page number from filename (e.g., "doc_page_5_nohf.md" -> 5)
        page_number = None
        try:
            if "_page_" in filename:
                page_str = filename.split("_page_")[1].split("_")[0]
                page_number = int(page_str)
        except (IndexError, ValueError):
            pass

        try:
            # Delete existing embeddings for this file
            delete_documents_by_file_path(file_path)

            # Remove from indexed files tracking
            with _index_lock:
                keys_to_remove = [k for k in _indexed_files if k.startswith(file_path + ":")]
                for k in keys_to_remove:
                    _indexed_files.discard(k)

            # Chunk the markdown file
            result = chunk_markdown_with_summaries(file_path, source_name, generate_summaries=True)

            if result.chunks:
                # Extract chunk IDs for status tracking
                chunk_ids = [chunk.metadata.get("chunk_id", f"{source_name}_{i}") for i, chunk in enumerate(result.chunks)]

                # Phase 1: Add chunks to Qdrant vectorstore
                vectorstore = get_vectorstore()
                vectorstore.add_documents(result.chunks)
                total_chunks += len(result.chunks)
                indexed_files.append(filename)

                # Update granular status: page and chunks succeeded
                if DB_AVAILABLE and filename_with_ext:
                    try:
                        with get_db_session() as db:
                            repo = DocumentRepository(db)
                            doc = repo.get_by_filename(filename_with_ext)
                            if doc:
                                repo.update_vector_indexing_status(
                                    doc,
                                    page_file_path=file_path,
                                    chunk_ids=chunk_ids,
                                    status="success",
                                    page_number=page_number
                                )
                    except Exception as e:
                        logger.warning(f"Could not update vector indexing status: {e}")

                # Collect chunks for GraphRAG Phase 2
                if GRAPHRAG_AVAILABLE and GRAPH_RAG_ENABLED:
                    for i, chunk in enumerate(result.chunks):
                        all_graphrag_chunks.append({
                            "id": chunk.metadata.get("chunk_id", f"{source_name}_{i}"),
                            "page_content": chunk.page_content,
                            "metadata": chunk.metadata,
                        })

                # Update indexed files tracking
                file_stat = os.stat(file_path)
                file_key = f"{file_path}:{file_stat.st_mtime}"
                with _index_lock:
                    _indexed_files.add(file_key)

                logger.info(f"[Phase 1] Indexed {len(result.chunks)} chunks from {filename} to Qdrant")

        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}")

            # Update granular status: page and chunks failed
            if DB_AVAILABLE and filename_with_ext:
                try:
                    # Get chunk IDs even if indexing failed (from chunking result)
                    chunk_ids = []
                    if result and result.chunks:
                        chunk_ids = [chunk.metadata.get("chunk_id", f"{source_name}_{i}") for i, chunk in enumerate(result.chunks)]

                    with get_db_session() as db:
                        repo = DocumentRepository(db)
                        doc = repo.get_by_filename(filename_with_ext)
                        if doc:
                            repo.update_vector_indexing_status(
                                doc,
                                page_file_path=file_path,
                                chunk_ids=chunk_ids,
                                status="failed",
                                error=str(e),
                                page_number=page_number
                            )
                except Exception as db_error:
                    logger.warning(f"Could not update failed vector indexing status: {db_error}")

    if skipped_files:
        logger.info(f"[Phase 1] Complete for {source_name}: {len(indexed_files)} files indexed, {len(skipped_files)} files skipped, {total_chunks} chunks in Qdrant")
    else:
        logger.info(f"[Phase 1] Complete for {source_name}: {len(indexed_files)} files, {total_chunks} chunks in Qdrant")
    return total_chunks, all_graphrag_chunks


def trigger_embedding_for_document(
    source_name: str,
    output_dir: str = None,
    filename: str = None,
    conversion_id: str = None,
    broadcast_callback=None
):
    """
    Trigger two-phase indexing for a document after conversion completes.
    Both phases run in background without blocking the main thread.

    Phase 1: Chunk embedding to Qdrant (makes document queryable)
    Phase 2: GraphRAG entity extraction to Neo4j (runs after Phase 1 completes)

    This approach allows users to query the document immediately after Phase 1,
    even while GraphRAG processing is still running in the background.

    Args:
        source_name: The document source name (folder name under output directory).
        output_dir: Optional output directory path.
        filename: Original filename (with extension) for database updates.
        conversion_id: The conversion ID for WebSocket notifications.
        broadcast_callback: Function to call for WebSocket broadcasts.
                          Signature: broadcast_callback(conversion_id, message_dict)
    """
    # Import IndexStatus here to avoid circular imports
    try:
        from db.models import IndexStatus
    except ImportError:
        IndexStatus = None

    def _two_phase_indexing_task():
        try:
            logger.info(f"[Two-Phase Indexing] Starting for: {source_name}")

            # Update database status to INDEXING
            if DB_AVAILABLE and filename and IndexStatus:
                try:
                    with get_db_session() as db:
                        repo = DocumentRepository(db)
                        doc = repo.get_by_filename(filename)
                        if doc:
                            repo.update_index_status(
                                doc, IndexStatus.INDEXING, 0,
                                message="Phase 1: Indexing chunks to vector database..."
                            )
                except Exception as e:
                    logger.warning(f"Could not update database to INDEXING status: {e}")

            # Send WebSocket notification that Phase 1 started
            if broadcast_callback and conversion_id:
                broadcast_callback(conversion_id, {
                    "status": "indexing",
                    "progress": 100,
                    "message": "Phase 1: Indexing document for search...",
                    "indexing_status": "started",
                    "phase": 1,
                })

            # ========== PHASE 1: Chunk embedding to Qdrant ==========
            chunks, graphrag_chunks = _index_chunks_to_qdrant(source_name, output_dir, filename)
            logger.info(f"[Phase 1] Complete for {source_name}: {chunks} chunks indexed to Qdrant")

            # Update database status to INDEXED (document is now queryable)
            if DB_AVAILABLE and filename and IndexStatus:
                try:
                    with get_db_session() as db:
                        repo = DocumentRepository(db)
                        doc = repo.get_by_filename(filename)
                        if doc:
                            repo.update_index_status(
                                doc, IndexStatus.INDEXED, chunks,
                                message=f"Indexed {chunks} chunks (GraphRAG processing in background)"
                            )
                except Exception as e:
                    logger.warning(f"Could not update database to INDEXED status: {e}")

            # Send WebSocket notification that Phase 1 completed
            if broadcast_callback and conversion_id:
                broadcast_callback(conversion_id, {
                    "status": "indexed",
                    "progress": 100,
                    "message": f"Document indexed successfully ({chunks} chunks). Extracting metadata...",
                    "indexing_status": "completed",
                    "chunks_indexed": chunks,
                    "phase": 1,
                })

            # ========== PHASE 1.5: Metadata Extraction ==========
            # Extract document metadata using hierarchical summarization
            # This runs synchronously to make metadata available ASAP for document routing
            if graphrag_chunks:
                try:
                    logger.info(f"[Phase 1.5] Starting metadata extraction for {source_name}")

                    # Send WebSocket notification
                    if broadcast_callback and conversion_id:
                        broadcast_callback(conversion_id, {
                            "status": "extracting_metadata",
                            "progress": 100,
                            "message": "Extracting document metadata...",
                            "phase": 1.5,
                        })

                    # Import metadata extractor
                    from .graph_rag.metadata_extractor import HierarchicalMetadataExtractor

                    extractor = HierarchicalMetadataExtractor()

                    # Create progress callback for WebSocket updates
                    def metadata_progress(msg: str):
                        if broadcast_callback and conversion_id:
                            broadcast_callback(conversion_id, {
                                "status": "extracting_metadata",
                                "progress": 100,
                                "message": f"Metadata: {msg}",
                                "phase": 1.5,
                            })

                    # Run extraction (handle both sync and async contexts)
                    import asyncio
                    import threading

                    # Run in a separate thread to avoid event loop conflicts
                    result_container = []
                    error_container = []

                    def run_async():
                        try:
                            # Create a new event loop for this thread
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                result = new_loop.run_until_complete(
                                    extractor.extract_metadata(
                                        chunks=graphrag_chunks,
                                        source_name=source_name,
                                        batch_size=10,
                                        progress_callback=metadata_progress,
                                    )
                                )
                                result_container.append(result)
                            finally:
                                new_loop.close()
                        except Exception as e:
                            error_container.append(e)

                    thread = threading.Thread(target=run_async)
                    thread.start()
                    thread.join()

                    if error_container:
                        raise error_container[0]
                    metadata = result_container[0]

                    # Save metadata to PostgreSQL
                    if DB_AVAILABLE and filename:
                        try:
                            with get_db_session() as db:
                                repo = DocumentRepository(db)
                                doc = repo.get_by_filename(filename)
                                if doc:
                                    repo.update_document_metadata(
                                        doc,
                                        metadata,
                                        message=f"Extracted: {metadata.get('document_type', 'unknown')} - {metadata.get('subject_name', 'N/A')}"
                                    )
                                    logger.info(
                                        f"[Phase 1.5] Metadata saved for {source_name}: "
                                        f"{metadata.get('document_type')} | "
                                        f"Subject: {metadata.get('subject_name')} | "
                                        f"Confidence: {metadata.get('confidence', 0):.2f}"
                                    )
                        except Exception as e:
                            logger.error(f"Failed to save metadata to database: {e}", exc_info=True)

                    # Send WebSocket notification that metadata extraction completed
                    if broadcast_callback and conversion_id:
                        broadcast_callback(conversion_id, {
                            "status": "metadata_extracted",
                            "progress": 100,
                            "message": f"Metadata extracted: {metadata.get('document_type', 'unknown')} document",
                            "metadata": {
                                "document_type": metadata.get("document_type"),
                                "subject_name": metadata.get("subject_name"),
                                "confidence": metadata.get("confidence"),
                            },
                            "phase": 1.5,
                        })

                    logger.info(f"[Phase 1.5] Complete for {source_name}")

                except Exception as e:
                    logger.error(f"[Phase 1.5] Metadata extraction failed for {source_name}: {e}", exc_info=True)
                    # Don't fail the entire indexing - metadata is optional
                    if broadcast_callback and conversion_id:
                        broadcast_callback(conversion_id, {
                            "status": "metadata_extraction_failed",
                            "progress": 100,
                            "message": "Metadata extraction failed (continuing with indexing)",
                            "phase": 1.5,
                        })

            # ========== PHASE 2: GraphRAG entity extraction (background) ==========
            # Start GraphRAG in a separate background thread
            # This does NOT block - user can query the document immediately
            if graphrag_chunks:
                _run_graphrag_background(
                    chunks_data=graphrag_chunks,
                    source_name=source_name,
                    filename=filename,
                    conversion_id=conversion_id,
                    broadcast_callback=broadcast_callback,
                )
            else:
                logger.debug(f"[Phase 2] No chunks for GraphRAG or GraphRAG disabled for {source_name}")

        except Exception as e:
            logger.error(f"Error in two-phase indexing for {source_name}: {e}")

            # Update database status to FAILED
            if DB_AVAILABLE and filename and IndexStatus:
                try:
                    with get_db_session() as db:
                        repo = DocumentRepository(db)
                        doc = repo.get_by_filename(filename)
                        if doc:
                            repo.update_index_status(
                                doc, IndexStatus.FAILED, 0,
                                message=f"Indexing failed: {str(e)}"
                            )
                except Exception as db_e:
                    logger.warning(f"Could not update database to FAILED status: {db_e}")

            # Send WebSocket notification for error
            if broadcast_callback and conversion_id:
                broadcast_callback(conversion_id, {
                    "status": "index_error",
                    "progress": 100,
                    "message": f"Indexing failed: {str(e)}",
                    "indexing_status": "error",
                })

    # Run Phase 1 in background thread (Phase 2 will be spawned from within)
    thread = threading.Thread(target=_two_phase_indexing_task, daemon=True, name=f"indexing-{source_name}")
    thread.start()
    logger.info(f"[Two-Phase Indexing] Started background thread for: {source_name}")

