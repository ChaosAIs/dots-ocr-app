"""
Document indexer with two-phase indexing and watchdog-based file watching.
Monitors the output folder for new markdown files and indexes them.

Two-Phase Indexing Architecture:
================================
Phase 1 (Synchronous/Background): Chunk embedding to Qdrant
    - Chunks markdown files and embeds them to Qdrant vector database
    - Makes documents immediately queryable after completion
    - Runs in background thread to not block main application

Phase 2 (Background/Async): GraphRAG entity extraction
    - Extracts entities and relationships using LLM
    - Stores to Neo4j graph database and Qdrant
    - Starts automatically after Phase 1 completes
    - Runs in separate background thread, non-blocking
    - Document remains queryable even if this phase fails

Benefits:
- Users can query documents immediately after Phase 1 completes
- GraphRAG processing doesn't block user operations
- Graceful degradation if GraphRAG fails
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

                # Check if already fully indexed
                if source_name not in checked_sources:
                    checked_sources.add(source_name)
                    if _is_document_fully_indexed_in_db(source_name):
                        logger.info(f"[Phase 1] Source '{source_name}' already fully indexed, skipping")
                        with _index_lock:
                            _indexed_files.add(file_key)
                        skipped_files += 1
                        continue

                # Chunk and index to Qdrant
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


def index_document_now(source_name: str, output_dir: str = None, run_graphrag: bool = True) -> int:
    """
    Index a specific document immediately with summarization.
    Deletes any existing embeddings for this source before re-indexing.

    This function performs Phase 1 (Qdrant embedding) synchronously.
    If run_graphrag=True, it also starts Phase 2 (GraphRAG) in background.

    Args:
        source_name: The document source name (folder name under output directory).
        output_dir: Optional output directory path.
        run_graphrag: Whether to run GraphRAG in background after Qdrant indexing (default: True)

    Returns:
        Number of chunks indexed to Qdrant.
    """
    # Use the shared Phase 1 function
    total_chunks, graphrag_chunks = _index_chunks_to_qdrant(source_name, output_dir)

    # Start Phase 2 (GraphRAG) in background if enabled
    if run_graphrag and graphrag_chunks:
        _run_graphrag_background(
            chunks_data=graphrag_chunks,
            source_name=source_name,
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
            num_entities, num_rels = index_chunks_sync(chunks_data, source_name)

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


def _index_chunks_to_qdrant(source_name: str, output_dir: str = None) -> tuple:
    """
    Phase 1: Index all chunks to Qdrant vector database.
    This makes the document queryable immediately.

    Args:
        source_name: The document source name
        output_dir: Optional output directory path

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

    total_chunks = 0
    indexed_files = []
    all_graphrag_chunks = []  # Collect all chunks for Phase 2

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

            # Chunk the markdown file
            result = chunk_markdown_with_summaries(file_path, source_name, generate_summaries=True)

            if result.chunks:
                # Phase 1: Add chunks to Qdrant vectorstore
                vectorstore = get_vectorstore()
                vectorstore.add_documents(result.chunks)
                total_chunks += len(result.chunks)
                indexed_files.append(filename)

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
            chunks, graphrag_chunks = _index_chunks_to_qdrant(source_name, output_dir)
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
                    "message": f"Document indexed successfully ({chunks} chunks). Knowledge graph building in progress...",
                    "indexing_status": "completed",
                    "chunks_indexed": chunks,
                    "phase": 1,
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

