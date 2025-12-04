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

from .markdown_chunker import chunk_markdown_file
from .vectorstore import get_vectorstore, delete_documents_by_source, delete_documents_by_file_path

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
        """Process a markdown file for indexing."""
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
            logger.info(f"Indexing new file: {file_path} (source: {source_name})")

            # Chunk the markdown file
            documents = chunk_markdown_file(file_path, source_name)

            if documents:
                # Add to vectorstore
                vectorstore = get_vectorstore()
                vectorstore.add_documents(documents)
                logger.info(f"Indexed {len(documents)} chunks from {source_name}")

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

    logger.info(f"Starting indexing of existing documents in {current_output_dir}")

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

                # Check if already indexed
                file_stat = os.stat(file_path)
                file_key = f"{file_path}:{file_stat.st_mtime}"

                with _index_lock:
                    if file_key in _indexed_files:
                        continue

                # Chunk and index
                documents = chunk_markdown_file(file_path, source_name)

                if documents:
                    vectorstore = get_vectorstore()
                    vectorstore.add_documents(documents)
                    total_chunks += len(documents)
                    total_files += 1

                    with _index_lock:
                        _indexed_files.add(file_key)

                    logger.debug(
                        f"Indexed {len(documents)} chunks from {source_name}"
                    )

            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")

    logger.info(
        f"Startup indexing complete: {total_files} files, {total_chunks} chunks"
    )
    return total_chunks


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
    Index a specific document immediately (non-blocking background task).
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

            # Chunk and index
            documents = chunk_markdown_file(file_path, source_name)

            if documents:
                vectorstore = get_vectorstore()
                vectorstore.add_documents(documents)
                total_chunks += len(documents)
                indexed_files.append(filename)

                # Update indexed files tracking
                file_stat = os.stat(file_path)
                file_key = f"{file_path}:{file_stat.st_mtime}"
                with _index_lock:
                    _indexed_files.add(file_key)

                logger.info(f"Indexed {len(documents)} chunks from {filename}")

        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}")

    logger.info(f"Document indexing complete for {source_name}: {len(indexed_files)} files, {total_chunks} chunks")
    return total_chunks


def reindex_document(source_name: str, output_dir: str = None) -> int:
    """
    Re-index a document by first deleting all existing embeddings for it,
    then creating new embeddings. Runs in a background thread.

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


def trigger_embedding_for_document(source_name: str, output_dir: str = None):
    """
    Trigger embedding for a document after conversion completes.
    This is the main entry point called after conversion.
    Runs in background without blocking.

    Args:
        source_name: The document source name (folder name under output directory).
        output_dir: Optional output directory path.
    """
    import threading

    def _embedding_task():
        try:
            logger.info(f"Starting background embedding for: {source_name}")
            chunks = index_document_now(source_name, output_dir)
            logger.info(f"Background embedding complete for {source_name}: {chunks} chunks")
        except Exception as e:
            logger.error(f"Error in background embedding for {source_name}: {e}")

    # Run in background thread to avoid blocking
    thread = threading.Thread(target=_embedding_task, daemon=True)
    thread.start()
    logger.info(f"Started background embedding thread for: {source_name}")

