"""
Document indexer with watchdog-based file watching.
Monitors the output folder for new markdown files and indexes them.
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
    add_file_summary,
    delete_file_summary_by_source,
    add_chunk_summaries,
    delete_chunk_summaries_by_source,
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
    Check if a document is FULLY indexed in the database.
    Falls back to Qdrant check if database is not available.

    Args:
        source_name: The source/document name (filename without extension)

    Returns:
        True if the document is marked as fully indexed in the database
    """
    if DB_AVAILABLE:
        try:
            with get_db_session() as db:
                repo = DocumentRepository(db)
                # Check if document exists and is fully indexed
                doc = _get_document_by_source(db, source_name)
                if doc:
                    return repo.is_fully_indexed(doc)
                return False
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
        """Process a markdown file for indexing with summarization."""
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
            logger.info(f"Indexing new file with summaries: {file_path} (source: {source_name})")

            # Step 1 & 2: Chunk the markdown file with LLM summarization
            result = chunk_markdown_with_summaries(file_path, source_name, generate_summaries=True)

            if result.chunks:
                # Add chunks to main vectorstore
                vectorstore = get_vectorstore()
                vectorstore.add_documents(result.chunks)
                logger.info(f"Indexed {len(result.chunks)} chunks from {source_name}")

                # Step 3: Add chunk summaries to chunk summary collection
                if result.chunk_summary_infos:
                    chunk_summary_dicts = [
                        {
                            "chunk_indices": cs.chunk_indices,
                            "chunk_ids": cs.chunk_ids,
                            "summary": cs.summary,
                            "heading_path": cs.heading_path,
                        }
                        for cs in result.chunk_summary_infos
                    ]
                    add_chunk_summaries(source_name, file_path, chunk_summary_dicts)
                    logger.info(f"Added {len(chunk_summary_dicts)} chunk summaries for {source_name}")

                # Step 4: Add file summary to separate collection
                if result.file_summary:
                    add_file_summary(source_name, file_path, result.file_summary)
                    logger.info(f"Added file summary for {source_name}")

                with _index_lock:
                    _indexed_files.add(file_key)
            else:
                logger.warning(f"No chunks generated from {file_path}")

        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")


def index_existing_documents(output_dir: str = None):
    """
    Index all existing markdown files in the output directory.
    Called on application startup.

    Skips documents that are already FULLY indexed (checks database first, then Qdrant).
    For partially indexed documents, only indexes the pending pages.

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

    logger.info(f"Starting indexing of existing documents in {current_output_dir}")

    # Track which source names we've already checked (to avoid duplicate database/Qdrant queries)
    checked_sources: Set[str] = set()

    # Walk through all subdirectories
    for root, dirs, files in os.walk(output_path):
        for filename in files:
            # Only process _nohf.md files
            if not filename.endswith("_nohf.md"):
                continue

            file_path = os.path.join(root, filename)

            try:
                # Get source name from directory name
                source_name = Path(file_path).parent.name

                # Check in-memory cache first (for this session)
                file_stat = os.stat(file_path)
                file_key = f"{file_path}:{file_stat.st_mtime}"

                with _index_lock:
                    if file_key in _indexed_files:
                        continue

                # Check if this source is already FULLY indexed (database first, then Qdrant)
                # Only check once per source to avoid excessive queries
                if source_name not in checked_sources:
                    checked_sources.add(source_name)
                    if _is_document_fully_indexed_in_db(source_name):
                        logger.info(f"Source '{source_name}' already fully indexed, skipping")
                        # Add to in-memory cache so we don't check again
                        with _index_lock:
                            _indexed_files.add(file_key)
                        skipped_files += 1
                        continue
                    else:
                        # Log if partially indexed
                        pending = _get_pending_pages_for_source(source_name)
                        if pending:
                            logger.info(f"Source '{source_name}' has {len(pending)} pending pages to index: {pending}")
                else:
                    # If we already checked this source and it wasn't fully indexed,
                    # we still need to index additional files from the same source
                    pass

                # Chunk and index with summarization
                result = chunk_markdown_with_summaries(file_path, source_name, generate_summaries=True)

                if result.chunks:
                    vectorstore = get_vectorstore()
                    vectorstore.add_documents(result.chunks)
                    total_chunks += len(result.chunks)
                    total_files += 1

                    # Add chunk summaries to chunk summary collection
                    if result.chunk_summary_infos:
                        chunk_summary_dicts = [
                            {
                                "chunk_indices": cs.chunk_indices,
                                "chunk_ids": cs.chunk_ids,
                                "summary": cs.summary,
                                "heading_path": cs.heading_path,
                            }
                            for cs in result.chunk_summary_infos
                        ]
                        add_chunk_summaries(source_name, file_path, chunk_summary_dicts)

                    # Add file summary to separate collection
                    if result.file_summary:
                        add_file_summary(source_name, file_path, result.file_summary)

                    with _index_lock:
                        _indexed_files.add(file_key)

                    # Update database status if available
                    if DB_AVAILABLE:
                        try:
                            _update_db_index_status(source_name, len(result.chunks))
                        except Exception as db_e:
                            logger.warning(f"Could not update database index status for '{source_name}': {db_e}")

                    logger.debug(
                        f"Indexed {len(result.chunks)} chunks from {source_name} with file and chunk summaries"
                    )

            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")

    logger.info(
        f"Startup indexing complete: {total_files} files indexed, {total_chunks} chunks, {skipped_files} files skipped (already indexed)"
    )
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


def index_document_now(source_name: str, output_dir: str = None) -> int:
    """
    Index a specific document immediately with summarization.
    Deletes any existing embeddings for this source before re-indexing.

    Args:
        source_name: The document source name (folder name under output directory).
        output_dir: Optional output directory path.

    Returns:
        Number of chunks indexed.
    """
    if output_dir:
        set_output_dir(output_dir)

    current_output_dir = get_output_dir()
    doc_dir = Path(current_output_dir) / source_name

    if not doc_dir.exists():
        logger.warning(f"Document directory does not exist: {doc_dir}")
        return 0

    total_chunks = 0
    indexed_files = []

    # Delete existing summaries for this source first
    delete_file_summary_by_source(source_name)
    delete_chunk_summaries_by_source(source_name)

    # Find all _nohf.md files in the document directory
    for filename in os.listdir(doc_dir):
        if not filename.endswith("_nohf.md"):
            continue

        file_path = str(doc_dir / filename)

        try:
            # Delete existing embeddings for this file
            delete_documents_by_file_path(file_path)

            # Remove from indexed files tracking
            with _index_lock:
                keys_to_remove = [k for k in _indexed_files if k.startswith(file_path + ":")]
                for k in keys_to_remove:
                    _indexed_files.discard(k)

            # Chunk and index with summarization
            result = chunk_markdown_with_summaries(file_path, source_name, generate_summaries=True)

            if result.chunks:
                vectorstore = get_vectorstore()
                vectorstore.add_documents(result.chunks)
                total_chunks += len(result.chunks)
                indexed_files.append(filename)

                # Add chunk summaries to chunk summary collection
                if result.chunk_summary_infos:
                    chunk_summary_dicts = [
                        {
                            "chunk_indices": cs.chunk_indices,
                            "chunk_ids": cs.chunk_ids,
                            "summary": cs.summary,
                            "heading_path": cs.heading_path,
                        }
                        for cs in result.chunk_summary_infos
                    ]
                    add_chunk_summaries(source_name, file_path, chunk_summary_dicts)

                # Add file summary to separate collection
                if result.file_summary:
                    add_file_summary(source_name, file_path, result.file_summary)

                # Update indexed files tracking
                file_stat = os.stat(file_path)
                file_key = f"{file_path}:{file_stat.st_mtime}"
                with _index_lock:
                    _indexed_files.add(file_key)

                logger.info(f"Indexed {len(result.chunks)} chunks from {filename} with summaries")

        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}")

    logger.info(f"Document indexing complete for {source_name}: {len(indexed_files)} files, {total_chunks} chunks")
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
            # Also delete existing summaries
            delete_file_summary_by_source(source_name)
            delete_chunk_summaries_by_source(source_name)
            logger.info(f"Deleted existing embeddings and summaries for source: {source_name}")

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


def trigger_embedding_for_document(
    source_name: str,
    output_dir: str = None,
    filename: str = None,
    conversion_id: str = None,
    broadcast_callback=None
):
    """
    Trigger embedding for a document after conversion completes.
    This is the main entry point called after conversion.
    Runs in background without blocking.

    Args:
        source_name: The document source name (folder name under output directory).
        output_dir: Optional output directory path.
        filename: Original filename (with extension) for database updates.
        conversion_id: The conversion ID for WebSocket notifications.
        broadcast_callback: Function to call for WebSocket broadcasts.
                          Signature: broadcast_callback(conversion_id, message_dict)
    """
    import threading
    # Import IndexStatus here to avoid circular imports
    try:
        from db.models import IndexStatus
    except ImportError:
        IndexStatus = None

    def _embedding_task():
        try:
            logger.info(f"Starting background embedding for: {source_name}")

            # Update database status to INDEXING
            if DB_AVAILABLE and filename and IndexStatus:
                try:
                    with get_db_session() as db:
                        repo = DocumentRepository(db)
                        doc = repo.get_by_filename(filename)
                        if doc:
                            repo.update_index_status(
                                doc, IndexStatus.INDEXING, 0,
                                message="Indexing started..."
                            )
                except Exception as e:
                    logger.warning(f"Could not update database to INDEXING status: {e}")

            # Send WebSocket notification that indexing started
            if broadcast_callback and conversion_id:
                broadcast_callback(conversion_id, {
                    "status": "indexing",
                    "progress": 100,
                    "message": "Indexing document for search...",
                    "indexing_status": "started",
                })

            # Perform the actual indexing
            chunks = index_document_now(source_name, output_dir)
            logger.info(f"Background embedding complete for {source_name}: {chunks} chunks")

            # Update database status to INDEXED
            if DB_AVAILABLE and filename and IndexStatus:
                try:
                    with get_db_session() as db:
                        repo = DocumentRepository(db)
                        doc = repo.get_by_filename(filename)
                        if doc:
                            repo.update_index_status(
                                doc, IndexStatus.INDEXED, chunks,
                                message=f"Indexed {chunks} chunks"
                            )
                except Exception as e:
                    logger.warning(f"Could not update database to INDEXED status: {e}")

            # Send WebSocket notification that indexing completed
            if broadcast_callback and conversion_id:
                broadcast_callback(conversion_id, {
                    "status": "indexed",
                    "progress": 100,
                    "message": f"Document indexed successfully ({chunks} chunks)",
                    "indexing_status": "completed",
                    "chunks_indexed": chunks,
                })

        except Exception as e:
            logger.error(f"Error in background embedding for {source_name}: {e}")

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

    # Run in background thread to avoid blocking
    thread = threading.Thread(target=_embedding_task, daemon=True)
    thread.start()
    logger.info(f"Started background embedding thread for: {source_name}")

