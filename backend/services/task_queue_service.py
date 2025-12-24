"""
Task queue service for hierarchical document processing.
"""
import os
import time
import hashlib
import logging
import threading
from datetime import datetime
from typing import Optional, Callable, Any
from uuid import UUID

from sqlalchemy.orm import Session

from db.models import Document, ConvertStatus, IndexStatus
from db.document_repository import DocumentRepository
from db.database import get_db_session

logger = logging.getLogger(__name__)


class TaskQueueService:
    """
    Service for hierarchical task queue management including:
    - OCR page tasks
    - Vector chunk tasks
    - GraphRAG chunk tasks
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        parser: Any,
        task_queue_manager: Any = None,
        task_scheduler: Any = None,
        queue_worker_pool: Any = None,
        document_status_broadcast: Optional[Callable] = None
    ):
        """
        Initialize task queue service.

        Args:
            input_dir: Base directory for input files
            output_dir: Base directory for output files
            parser: DotsOCRParser instance
            task_queue_manager: HierarchicalTaskQueueManager instance
            task_scheduler: TaskScheduler instance
            queue_worker_pool: HierarchicalWorkerPool instance
            document_status_broadcast: Callback for document status broadcasts
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.parser = parser
        self.task_queue_manager = task_queue_manager
        self.task_scheduler = task_scheduler
        self.queue_worker_pool = queue_worker_pool
        self.document_status_broadcast = document_status_broadcast

        # Check if task queue is enabled
        self.task_queue_enabled = os.getenv("TASK_QUEUE_ENABLED", "true").lower() in ("true", "1", "yes")

        # Import GraphRAG settings
        try:
            from rag_service.graph_rag import GRAPH_RAG_INDEX_ENABLED
            self.graph_rag_index_enabled = GRAPH_RAG_INDEX_ENABLED
        except ImportError:
            self.graph_rag_index_enabled = False

    def resolve_file_path(self, relative_or_absolute_path: str, base_dir: str = None) -> str:
        """Resolve a file path to absolute path."""
        if base_dir is None:
            base_dir = self.input_dir

        if os.path.isabs(relative_or_absolute_path):
            if os.path.exists(relative_or_absolute_path):
                return relative_or_absolute_path
            for known_base in [self.input_dir, self.output_dir]:
                if known_base in relative_or_absolute_path:
                    relative_part = relative_or_absolute_path.split(known_base + os.sep)[-1]
                    candidate = os.path.join(base_dir, relative_part)
                    if os.path.exists(candidate):
                        return candidate

        return os.path.join(base_dir, relative_or_absolute_path)

    def to_relative_output_path(self, absolute_path: str) -> str:
        """Convert an absolute output path to a relative path."""
        if not absolute_path:
            return absolute_path

        if not os.path.isabs(absolute_path):
            return absolute_path

        if absolute_path.startswith(self.output_dir + os.sep):
            return absolute_path[len(self.output_dir) + 1:]
        elif absolute_path.startswith(self.output_dir):
            return absolute_path[len(self.output_dir):].lstrip(os.sep)

        return absolute_path

    def get_queue_stats(self) -> dict:
        """Get task queue statistics."""
        if not self.task_queue_manager:
            return {"error": "Task queue system not initialized"}

        return self.task_queue_manager.get_queue_stats()

    def trigger_maintenance(self):
        """Manually trigger queue maintenance."""
        if not self.task_scheduler:
            return {"error": "Task scheduler not initialized"}

        def run_maintenance():
            self.task_scheduler._periodic_maintenance()

        thread = threading.Thread(target=run_maintenance, daemon=True)
        thread.start()

        return {"status": "success", "message": "Queue maintenance triggered"}

    def process_ocr_page_task(self, page_task) -> str:
        """
        Process OCR task for a single page.

        This function checks if the page markdown exists. If not, it triggers the full document
        conversion. After markdown exists, it chunks the content and creates chunk tasks.

        Args:
            page_task: PageTaskData with document_id, page_number, etc.

        Returns:
            str: Path to the generated page markdown file
        """
        from rag_service.markdown_chunker import chunk_markdown_with_summaries
        from queue_service import TaskQueuePage

        try:
            with get_db_session() as db:
                repo = DocumentRepository(db)
                doc = repo.get_by_id(page_task.document_id)

                if not doc:
                    raise ValueError(f"Document not found: {page_task.document_id}")

                filename = doc.filename
                base_name = os.path.splitext(filename)[0]
                logger.info(f"🔄 Processing OCR for page {page_task.page_number} of document: {filename}")

                # Get input file path from database
                db_file_path = doc.file_path
                if not db_file_path:
                    db_file_path = filename

                input_path = self.resolve_file_path(db_file_path)

                if not os.path.exists(input_path):
                    raise FileNotFoundError(f"Input file not found: {input_path}")

                # Determine output directory
                if doc.output_path:
                    page_output_dir = self.resolve_file_path(doc.output_path, self.output_dir)
                else:
                    rel_dir = os.path.dirname(db_file_path)
                    page_output_dir = os.path.join(self.output_dir, rel_dir, base_name) if rel_dir else os.path.join(self.output_dir, base_name)

                # Helper function to find the actual page markdown file
                def find_page_markdown(page_num: int, total_pages: int) -> str:
                    if total_pages == 1:
                        simple_path = os.path.join(page_output_dir, f"{base_name}_nohf.md")
                        if os.path.exists(simple_path):
                            return simple_path

                    page_path = os.path.join(page_output_dir, f"{base_name}_page_{page_num}_nohf.md")
                    if os.path.exists(page_path):
                        return page_path

                    simple_path = os.path.join(page_output_dir, f"{base_name}_nohf.md")
                    if os.path.exists(simple_path):
                        return simple_path

                    return None

                total_pages = doc.total_pages or 1
                page_output_path = find_page_markdown(page_task.page_number, total_pages)

                if not page_output_path:
                    logger.info(f"📄 Page markdown not found, triggering OCR for document: {filename}")

                    lock_file = os.path.join(page_output_dir, f".{base_name}.converting")

                    if os.path.exists(lock_file):
                        logger.info(f"⏳ OCR conversion in progress by another worker, waiting...")
                        for _ in range(120):
                            time.sleep(5)
                            page_output_path = find_page_markdown(page_task.page_number, total_pages)
                            if page_output_path:
                                logger.info(f"✅ Page markdown now available: {page_output_path}")
                                break
                            if not os.path.exists(lock_file):
                                break
                        else:
                            raise TimeoutError(f"Timeout waiting for page {page_task.page_number} OCR")
                    else:
                        try:
                            os.makedirs(os.path.dirname(lock_file), exist_ok=True)
                            with open(lock_file, 'w') as f:
                                f.write(str(page_task.document_id))

                            parser_output_dir = os.path.dirname(page_output_dir)
                            logger.info(f"🚀 Starting OCR conversion for {filename} -> {parser_output_dir}")
                            results = self.parser.parse_file(
                                input_path,
                                output_dir=parser_output_dir,
                                prompt_mode="prompt_layout_all_en",
                            )
                            logger.info(f"✅ OCR conversion completed for {filename}: {len(results)} pages")

                            doc.output_path = self.to_relative_output_path(page_output_dir)
                            db.commit()

                        finally:
                            if os.path.exists(lock_file):
                                os.remove(lock_file)

                    page_output_path = find_page_markdown(page_task.page_number, total_pages)

                if not page_output_path:
                    raise FileNotFoundError(
                        f"Page markdown not found after OCR. Expected: "
                        f"{base_name}_nohf.md or {base_name}_page_{page_task.page_number}_nohf.md"
                    )

                # Read the page content and create chunks
                logger.info(f"📚 Chunking page {page_task.page_number} content...")

                page_record = db.query(TaskQueuePage).filter(
                    TaskQueuePage.id == page_task.id
                ).first()

                if page_record:
                    result = chunk_markdown_with_summaries(
                        page_output_path,
                        source_name=f"{filename}_page_{page_task.page_number}"
                    )
                    chunks = result.chunks

                    if chunks:
                        chunk_data = []
                        for idx, chunk in enumerate(chunks):
                            content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:16]
                            chunk_id = f"{page_task.document_id}_{page_task.page_number}_{idx}_{content_hash}"
                            chunk_data.append((chunk_id, idx))

                        if chunk_data and self.task_queue_manager:
                            self.task_queue_manager.create_chunk_tasks(
                                page_id=page_task.id,
                                document_id=page_task.document_id,
                                chunks=chunk_data,
                                db=db
                            )
                            logger.info(f"📦 Created {len(chunk_data)} chunk tasks for page {page_task.page_number}")

                        page_record.chunk_count = len(chunks)
                        db.commit()

                logger.info(f"✅ OCR completed for page {page_task.page_number} of {filename}")
                return page_output_path

        except Exception as e:
            logger.error(f"❌ OCR page task failed for page {page_task.page_number} of doc {page_task.document_id}: {e}", exc_info=True)
            raise

    def process_vector_chunk_task(self, chunk_task) -> bool:
        """
        Process Vector indexing task for a single chunk.

        Args:
            chunk_task: ChunkTaskData with chunk_id, document_id, page_id, chunk_index, etc.

        Returns:
            bool: True if indexing succeeded
        """
        from rag_service.markdown_chunker import chunk_markdown_with_summaries
        from rag_service.vectorstore import get_vectorstore
        from queue_service import TaskQueuePage

        try:
            logger.info(f"🔄 Processing Vector index for chunk: {chunk_task.chunk_id}")

            with get_db_session() as db:
                page_record = db.query(TaskQueuePage).filter(
                    TaskQueuePage.id == chunk_task.page_id
                ).first()

                if not page_record or not page_record.page_file_path:
                    raise ValueError(f"Page record or file path not found for page_id: {chunk_task.page_id}")

                page_file_path = page_record.page_file_path
                page_number = page_record.page_number

                doc = db.query(Document).filter(Document.id == chunk_task.document_id).first()
                if not doc:
                    raise ValueError(f"Document not found: {chunk_task.document_id}")

                source_name = os.path.splitext(doc.filename)[0]

            result = chunk_markdown_with_summaries(page_file_path, source_name=source_name)
            chunks = result.chunks

            if not chunks or chunk_task.chunk_index >= len(chunks):
                raise ValueError(
                    f"Chunk index {chunk_task.chunk_index} out of range. "
                    f"Page has {len(chunks) if chunks else 0} chunks."
                )

            chunk = chunks[chunk_task.chunk_index]

            # Build ID mapping for parent/child relationships
            old_to_new_id_map = {}
            for idx, c in enumerate(chunks):
                old_id = c.metadata.get("chunk_id", "")
                content_hash = hashlib.md5(c.page_content.encode()).hexdigest()[:16]
                new_id = f"{chunk_task.document_id}_{page_number}_{idx}_{content_hash}"
                old_to_new_id_map[old_id] = new_id

            chunk.metadata["chunk_id"] = chunk_task.chunk_id

            # Update parent/child references
            old_parent_id = chunk.metadata.get("parent_chunk_id")
            if old_parent_id and old_parent_id in old_to_new_id_map:
                chunk.metadata["parent_chunk_id"] = old_to_new_id_map[old_parent_id]

            old_child_ids = chunk.metadata.get("child_chunk_ids", [])
            if old_child_ids:
                chunk.metadata["child_chunk_ids"] = [old_to_new_id_map.get(cid, cid) for cid in old_child_ids]

            chunk.metadata["document_id"] = str(chunk_task.document_id)

            vectorstore = get_vectorstore()
            vectorstore.add_documents([chunk])

            logger.info(f"✅ Vector index completed for chunk: {chunk_task.chunk_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Vector chunk task failed for chunk {chunk_task.chunk_id}: {e}", exc_info=True)
            raise

    def process_graphrag_chunk_task(self, chunk_task) -> tuple:
        """
        Process GraphRAG indexing task for a single chunk.

        Args:
            chunk_task: ChunkTaskData with chunk_id, document_id, page_id, chunk_index, etc.

        Returns:
            tuple: (entities_count, relationships_count)
        """
        from rag_service.markdown_chunker import chunk_markdown_with_summaries
        from queue_service import TaskQueuePage

        try:
            logger.info(f"🔄 Processing GraphRAG index for chunk: {chunk_task.chunk_id}")

            if not self.graph_rag_index_enabled:
                logger.debug(f"GraphRAG indexing disabled, skipping chunk: {chunk_task.chunk_id}")
                return (0, 0)

            with get_db_session() as db:
                page_record = db.query(TaskQueuePage).filter(
                    TaskQueuePage.id == chunk_task.page_id
                ).first()

                if not page_record or not page_record.page_file_path:
                    raise ValueError(f"Page record or file path not found for page_id: {chunk_task.page_id}")

                page_file_path = page_record.page_file_path

                doc = db.query(Document).filter(Document.id == chunk_task.document_id).first()
                if not doc:
                    raise ValueError(f"Document not found: {chunk_task.document_id}")

                source_name = os.path.splitext(doc.filename)[0]

            result = chunk_markdown_with_summaries(page_file_path, source_name=source_name)
            chunks = result.chunks

            if not chunks or chunk_task.chunk_index >= len(chunks):
                raise ValueError(
                    f"Chunk index {chunk_task.chunk_index} out of range. "
                    f"Page has {len(chunks) if chunks else 0} chunks."
                )

            chunk = chunks[chunk_task.chunk_index]

            chunk_data = {
                "id": chunk_task.chunk_id,
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }

            entities_count = 0
            relationships_count = 0

            try:
                from rag_service.graph_rag.graph_indexer import index_chunks_sync

                entities_count, relationships_count = index_chunks_sync(
                    chunks=[chunk_data],
                    source_name=source_name,
                )

            except ImportError:
                logger.warning("GraphRAG indexer not available")
            except Exception as e:
                logger.error(f"GraphRAG indexing failed for chunk {chunk_task.chunk_id}: {e}")

            logger.info(f"✅ GraphRAG index completed for chunk: {chunk_task.chunk_id} (entities={entities_count}, relationships={relationships_count})")
            return (entities_count, relationships_count)

        except Exception as e:
            logger.error(f"❌ GraphRAG chunk task failed for chunk {chunk_task.chunk_id}: {e}", exc_info=True)
            raise

    def resume_incomplete_ocr(self, conversion_manager, worker_pool):
        """
        Resume incomplete OCR conversion tasks on startup.

        Args:
            conversion_manager: ConversionManager for tracking
            worker_pool: WorkerPool for background processing
        """
        try:
            with get_db_session() as db:
                repo = DocumentRepository(db)
                docs_to_convert = []

                for doc in repo.get_all():
                    if doc.deleted_at:
                        continue

                    needs_conversion = False

                    if doc.convert_status in [ConvertStatus.PENDING, ConvertStatus.CONVERTING, ConvertStatus.PARTIAL, ConvertStatus.FAILED]:
                        if doc.convert_status == ConvertStatus.PARTIAL:
                            if doc.total_pages > 0 and doc.converted_pages < doc.total_pages:
                                needs_conversion = True
                                logger.info(f"Document {doc.filename} has partial conversion ({doc.converted_pages}/{doc.total_pages} pages)")
                        else:
                            needs_conversion = True
                            logger.info(f"Document {doc.filename} needs conversion (status: {doc.convert_status.value})")

                    if needs_conversion:
                        docs_to_convert.append(doc)

                if not docs_to_convert:
                    logger.info("No incomplete OCR conversion tasks found")
                    return

                logger.info(f"Found {len(docs_to_convert)} documents with incomplete OCR conversion")

                for doc in docs_to_convert:
                    try:
                        conversion_id = conversion_manager.create_conversion(doc.filename)

                        conversion_manager.update_conversion(
                            conversion_id,
                            status="processing",
                            progress=0,
                            message="Resuming OCR conversion from startup...",
                            started_at=datetime.now().isoformat()
                        )

                        repo.update_convert_status(doc, ConvertStatus.CONVERTING, message="Resuming conversion from startup")

                        # Import the conversion function
                        from services.conversion_service import ConversionService

                        def progress_callback(progress: int, message: str = ""):
                            try:
                                conversion_manager.update_conversion(
                                    conversion_id,
                                    progress=progress,
                                    message=message,
                                )
                            except Exception as e:
                                logger.error(f"Error in progress callback: {e}")

                        success = worker_pool.submit_task(
                            conversion_id=conversion_id,
                            func=self._convert_document_background,
                            args=(doc.filename, "prompt_layout_all_en"),
                            kwargs={
                                "conversion_id": conversion_id,
                                "progress_callback": progress_callback,
                            }
                        )

                        if success:
                            logger.info(f"Triggered background OCR conversion for: {doc.filename}")
                        else:
                            logger.warning(f"Failed to submit OCR conversion task for {doc.filename}")

                    except Exception as e:
                        logger.error(f"Error resuming OCR conversion for {doc.filename}: {e}")

                logger.info(f"Resume OCR conversion complete: {len(docs_to_convert)} documents queued")

        except Exception as e:
            logger.error(f"Error in resume_incomplete_ocr: {e}")

    def _convert_document_background(
        self,
        filename: str,
        prompt_mode: str,
        file_path: str = None,
        conversion_id: str = None,
        progress_callback=None
    ):
        """Background task to convert document using OCR parser."""
        if not file_path:
            relative_path = filename
        else:
            relative_path = file_path

        absolute_file_path = self.resolve_file_path(relative_path)

        logger.info(f"Starting OCR conversion for {filename} at {absolute_file_path}")

        rel_dir = os.path.dirname(relative_path)

        if rel_dir and rel_dir != '.':
            output_dir = os.path.join(self.output_dir, rel_dir)
        else:
            output_dir = self.output_dir

        os.makedirs(output_dir, exist_ok=True)

        results = self.parser.parse_file(
            absolute_file_path,
            output_dir=output_dir,
            prompt_mode=prompt_mode,
            progress_callback=progress_callback,
        )

        logger.info(f"OCR conversion completed successfully for {filename}")
        return results

    def resume_incomplete_indexing(self, conversion_manager, trigger_embedding_func):
        """
        Resume incomplete indexing tasks on startup.

        Args:
            conversion_manager: ConversionManager for tracking
            trigger_embedding_func: Function to trigger embedding
        """
        try:
            with get_db_session() as db:
                repo = DocumentRepository(db)
                docs_to_index = []

                for doc in repo.get_all():
                    if doc.deleted_at:
                        continue

                    if not doc.convert_status or doc.convert_status == ConvertStatus.PENDING:
                        continue

                    if doc.index_status == IndexStatus.INDEXED:
                        indexing_details = doc.indexing_details or {}
                        vector_status = indexing_details.get("vector_indexing", {}).get("status")
                        metadata_status = indexing_details.get("metadata_extraction", {}).get("status")
                        graphrag_status = indexing_details.get("graphrag_indexing", {}).get("status")

                        all_complete = (
                            vector_status in ["completed", "success", None] and
                            metadata_status in ["completed", "success", None] and
                            graphrag_status in ["completed", "success", None]
                        )

                        if all_complete:
                            continue

                    needs_indexing = False
                    indexing_details = doc.indexing_details or {}

                    vector_status = indexing_details.get("vector_indexing", {}).get("status")
                    if vector_status in ["pending", "failed"]:
                        needs_indexing = True

                    metadata_status = indexing_details.get("metadata_extraction", {}).get("status")
                    if metadata_status in ["pending", "failed"]:
                        needs_indexing = True

                    graphrag_status = indexing_details.get("graphrag_indexing", {}).get("status")
                    if graphrag_status in ["pending", "failed"]:
                        needs_indexing = True

                    if not indexing_details and doc.index_status in [IndexStatus.PENDING, IndexStatus.FAILED]:
                        needs_indexing = True

                    if needs_indexing:
                        docs_to_index.append(doc)

                if not docs_to_index:
                    logger.info("No incomplete indexing tasks found")
                    return

                logger.info(f"Found {len(docs_to_index)} documents with incomplete indexing")

                for doc in docs_to_index:
                    try:
                        file_name_without_ext = os.path.splitext(doc.filename)[0]
                        conversion_id = conversion_manager.create_conversion(doc.filename)

                        conversion_manager.update_conversion(
                            conversion_id,
                            status="indexing",
                            progress=0,
                            message="Resuming indexing from startup...",
                            started_at=datetime.now().isoformat()
                        )

                        trigger_embedding_func(
                            source_name=file_name_without_ext,
                            output_dir=self.output_dir,
                            filename=doc.filename,
                            conversion_id=conversion_id,
                            broadcast_callback=None
                        )

                        logger.info(f"Triggered background indexing for: {doc.filename}")

                    except Exception as e:
                        logger.error(f"Error resuming indexing for {doc.filename}: {e}")

                logger.info(f"Resume indexing complete: {len(docs_to_index)} documents queued")

        except Exception as e:
            logger.error(f"Error in resume_incomplete_indexing: {e}")
