"""
Task queue service for hierarchical document processing.
"""
import os
import time
import hashlib
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
from uuid import UUID

from sqlalchemy.orm import Session

from db.models import Document, ConvertStatus, IndexStatus, TaskStatus
from db.document_repository import DocumentRepository
from db.database import get_db_session
from datetime import timezone

logger = logging.getLogger(__name__)


class TaskQueueService:
    """
    Service for hierarchical task queue management including:
    - OCR page tasks
    - Vector chunk tasks
    - GraphRAG chunk tasks
    """

    # File extensions that should use doc_service converter instead of OCR
    DOC_SERVICE_EXTENSIONS = {'.docx', '.doc', '.xlsx', '.xlsm', '.xlsb', '.xls', '.txt', '.csv', '.tsv', '.log', '.text'}

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        parser: Any,
        task_queue_manager: Any = None,
        task_scheduler: Any = None,
        queue_worker_pool: Any = None,
        document_status_broadcast: Optional[Callable] = None,
        doc_converter_manager: Any = None
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
            doc_converter_manager: DocumentConverterManager for Word/Excel/CSV files
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.parser = parser
        self.task_queue_manager = task_queue_manager
        self.task_scheduler = task_scheduler
        self.queue_worker_pool = queue_worker_pool
        self.document_status_broadcast = document_status_broadcast
        self.doc_converter_manager = doc_converter_manager

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
                # ================================================================
                # SECTION 1: DOCUMENT LOOKUP
                # ================================================================
                logger.info("=" * 80)
                logger.info("[OCR Task] ========== SECTION 1: DOCUMENT LOOKUP ==========")
                logger.info("=" * 80)

                repo = DocumentRepository(db)
                doc = repo.get_by_id(page_task.document_id)

                if not doc:
                    raise ValueError(f"Document not found: {page_task.document_id}")

                filename = doc.filename
                base_name = os.path.splitext(filename)[0]

                logger.info(f"[OCR Task] Document ID: {page_task.document_id}")
                logger.info(f"[OCR Task] Filename: {filename}")
                logger.info(f"[OCR Task] Base name: {base_name}")
                logger.info(f"[OCR Task] Page number: {page_task.page_number}")
                logger.info(f"[OCR Task] Total pages: {doc.total_pages}")
                logger.info(f"[OCR Task] Current convert_status: {doc.convert_status}")
                logger.info("-" * 80)

                # ================================================================
                # SECTION 2: FILE PATH RESOLUTION
                # ================================================================
                logger.info("=" * 80)
                logger.info("[OCR Task] ========== SECTION 2: FILE PATH RESOLUTION ==========")
                logger.info("=" * 80)

                # Get input file path from database
                db_file_path = doc.file_path
                if not db_file_path:
                    db_file_path = filename

                input_path = self.resolve_file_path(db_file_path)
                logger.info(f"[OCR Task] DB file_path: {db_file_path}")
                logger.info(f"[OCR Task] Resolved input_path: {input_path}")
                logger.info(f"[OCR Task] Input file exists: {os.path.exists(input_path)}")

                if not os.path.exists(input_path):
                    raise FileNotFoundError(f"Input file not found: {input_path}")

                # Determine output directory
                if doc.output_path:
                    page_output_dir = self.resolve_file_path(doc.output_path, self.output_dir)
                else:
                    rel_dir = os.path.dirname(db_file_path)
                    page_output_dir = os.path.join(self.output_dir, rel_dir, base_name) if rel_dir else os.path.join(self.output_dir, base_name)

                logger.info(f"[OCR Task] Output directory: {page_output_dir}")
                logger.info("-" * 80)

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

                # ================================================================
                # SECTION 3: DOCUMENT CONVERSION
                # ================================================================
                logger.info("=" * 80)
                logger.info("[OCR Task] ========== SECTION 3: DOCUMENT CONVERSION ==========")
                logger.info("=" * 80)
                logger.info(f"[OCR Task] Looking for existing markdown...")
                logger.info(f"[OCR Task] Page markdown found: {page_output_path is not None}")
                if page_output_path:
                    logger.info(f"[OCR Task] Markdown path: {page_output_path}")

                if not page_output_path:
                    logger.info(f"[OCR Task] Page markdown NOT found, triggering conversion...")
                    logger.info(f"[OCR Task] Document: {filename}")

                    lock_file = os.path.join(page_output_dir, f".{base_name}.converting")

                    if os.path.exists(lock_file):
                        logger.info(f"⏳ Conversion in progress by another worker, waiting...")
                        for _ in range(120):
                            time.sleep(5)
                            page_output_path = find_page_markdown(page_task.page_number, total_pages)
                            if page_output_path:
                                logger.info(f"✅ Page markdown now available: {page_output_path}")
                                break
                            if not os.path.exists(lock_file):
                                break
                        else:
                            raise TimeoutError(f"Timeout waiting for page {page_task.page_number} conversion")
                    else:
                        try:
                            os.makedirs(os.path.dirname(lock_file), exist_ok=True)
                            with open(lock_file, 'w') as f:
                                f.write(str(page_task.document_id))

                            # Check if file should use doc_service converter (Excel, Word, CSV, etc.)
                            file_ext = Path(input_path).suffix.lower()
                            logger.info(f"[OCR Task] File extension: {file_ext}")
                            logger.info(f"[OCR Task] DOC_SERVICE_EXTENSIONS: {self.DOC_SERVICE_EXTENSIONS}")

                            if file_ext in self.DOC_SERVICE_EXTENSIONS:
                                # Use doc_service converter for Excel/Word/CSV files (skip OCR)
                                logger.info("-" * 40)
                                logger.info(f"[OCR Task] >>> Using DOC_SERVICE converter (non-OCR)")
                                logger.info(f"[OCR Task] Converter type: PlainTextConverter or similar")
                                logger.info(f"[OCR Task] Input: {input_path}")
                                logger.info(f"[OCR Task] Output dir: {page_output_dir}")
                                logger.info("-" * 40)

                                page_output_path = self._convert_with_doc_service(
                                    input_path=input_path,
                                    output_dir=page_output_dir,
                                    base_name=base_name
                                )

                                logger.info(f"[OCR Task] Doc service conversion COMPLETED")
                                logger.info(f"[OCR Task] Output file: {page_output_path}")
                            else:
                                # Use OCR for PDFs and images
                                parser_output_dir = os.path.dirname(page_output_dir)
                                logger.info("-" * 40)
                                logger.info(f"[OCR Task] >>> Using OCR converter")
                                logger.info(f"[OCR Task] Input: {input_path}")
                                logger.info(f"[OCR Task] Output dir: {parser_output_dir}")
                                logger.info("-" * 40)

                                results = self.parser.parse_file(
                                    input_path,
                                    output_dir=parser_output_dir,
                                    prompt_mode="prompt_layout_all_en",
                                )

                                logger.info(f"[OCR Task] OCR conversion COMPLETED: {len(results)} pages")

                            doc.output_path = self.to_relative_output_path(page_output_dir)
                            db.commit()
                            logger.info(f"[OCR Task] Updated doc.output_path: {doc.output_path}")

                        finally:
                            if os.path.exists(lock_file):
                                os.remove(lock_file)

                    # Only search for markdown if we used OCR (doc_service already set page_output_path)
                    if not page_output_path:
                        page_output_path = find_page_markdown(page_task.page_number, total_pages)

                if not page_output_path:
                    raise FileNotFoundError(
                        f"Page markdown not found after OCR. Expected: "
                        f"{base_name}_nohf.md or {base_name}_page_{page_task.page_number}_nohf.md"
                    )

                logger.info(f"[OCR Task] Conversion complete. Markdown file: {page_output_path}")
                logger.info("-" * 80)

                # ================================================================
                # SECTION 4: DOCUMENT CLASSIFICATION & ROUTING
                # ================================================================
                logger.info("=" * 80)
                logger.info("[OCR Task] ========== SECTION 4: CLASSIFICATION & ROUTING ==========")
                logger.info("=" * 80)
                logger.info("[OCR Task] Determining processing path: TABULAR vs STANDARD")
                logger.info("[OCR Task]   - TABULAR: CSV, Excel, invoices → data extraction")
                logger.info("[OCR Task]   - STANDARD: Other documents → semantic chunking")

                file_ext = Path(doc.filename).suffix.lower() if doc.filename else ''
                logger.info(f"[OCR Task] File extension for routing: {file_ext}")

                # Check if this is a tabular document that should skip chunking
                logger.info("[OCR Task] Calling _check_and_route_tabular_document()...")
                is_tabular_doc = self._check_and_route_tabular_document(
                    document_id=page_task.document_id,
                    filename=filename,
                    file_ext=file_ext,
                    output_dir=page_output_dir,
                    page_output_path=page_output_path,
                    db=db
                )

                logger.info(f"[OCR Task] Routing result: is_tabular={is_tabular_doc}")

                if is_tabular_doc:
                    # Document routed to TABULAR PATH
                    # Data extraction has been triggered, skip chunking
                    logger.info("=" * 80)
                    logger.info("[OCR Task] ========== TABULAR PATH - EXTRACTION TRIGGERED ==========")
                    logger.info("=" * 80)
                    logger.info(f"[OCR Task] Document routed to TABULAR extraction path")
                    logger.info(f"[OCR Task] Chunking SKIPPED for tabular document")
                    logger.info(f"[OCR Task] Page task completed for: {filename}")
                    logger.info("=" * 80)
                    return page_output_path

                # ================================================================
                # SECTION 5: STANDARD PATH - CHUNKING
                # ================================================================
                logger.info("=" * 80)
                logger.info("[OCR Task] ========== SECTION 5: STANDARD PATH - CHUNKING ==========")
                logger.info("=" * 80)
                logger.info(f"[OCR Task] Document routed to STANDARD chunking path")
                logger.info(f"[OCR Task] Page {page_task.page_number} will be chunked for semantic indexing")

                page_record = db.query(TaskQueuePage).filter(
                    TaskQueuePage.id == page_task.id
                ).first()

                if page_record:
                    # Check if this is a single-file document (non-PDF like Excel, Word, CSV)
                    # These produce a single output file, so we should only create chunks once
                    # Note: file_ext already defined above in classification section
                    is_single_file_doc = file_ext in self.DOC_SERVICE_EXTENSIONS

                    if is_single_file_doc:
                        # For single-file documents, check if chunks already exist for this document
                        # If another page already created chunks, skip chunk creation for this page
                        from queue_service import TaskQueueChunk
                        existing_chunks = db.query(TaskQueueChunk).filter(
                            TaskQueueChunk.document_id == page_task.document_id
                        ).first()

                        if existing_chunks:
                            logger.info(f"⏭️ Skipping chunk creation for page {page_task.page_number} - "
                                       f"chunks already exist for single-file document {filename}")
                            page_record.chunk_count = 0
                            db.commit()
                            logger.info(f"✅ OCR completed for page {page_task.page_number} of {filename} (skipped chunking)")
                            return page_output_path

                    # ================================================================
                    # V3.0: CHUNKING START - Page {page_task.page_number}
                    # ================================================================
                    logger.info("=" * 80)
                    logger.info(f"[Chunking] ========== START CHUNKING PAGE {page_task.page_number} ==========")
                    logger.info("=" * 80)
                    logger.info(f"[Chunking] Document: {filename}")
                    logger.info(f"[Chunking] Page: {page_task.page_number}")
                    logger.info(f"[Chunking] Input file: {page_output_path}")

                    result = chunk_markdown_with_summaries(
                        page_output_path,
                        source_name=f"{filename}_page_{page_task.page_number}"
                    )
                    chunks = result.chunks

                    logger.info("-" * 80)
                    logger.info(f"[Chunking] Chunking complete: {len(chunks) if chunks else 0} chunks created")

                    if chunks:
                        chunk_data = []
                        total_content_size = 0
                        for idx, chunk in enumerate(chunks):
                            content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:16]
                            chunk_id = f"{page_task.document_id}_{page_task.page_number}_{idx}_{content_hash}"
                            # V3.0: Store chunk content and metadata to avoid redundant LLM calls
                            chunk_content = chunk.page_content
                            chunk_metadata = chunk.metadata.copy() if chunk.metadata else {}
                            chunk_metadata["source_page"] = page_task.page_number
                            chunk_metadata["chunk_index"] = idx
                            chunk_metadata["total_chunks"] = len(chunks)
                            chunk_data.append((chunk_id, idx, chunk_content, chunk_metadata))
                            total_content_size += len(chunk_content)

                        logger.info(f"[Chunking] Total content stored: {total_content_size} chars across {len(chunk_data)} chunks")

                        if chunk_data and self.task_queue_manager:
                            self.task_queue_manager.create_chunk_tasks(
                                page_id=page_task.id,
                                document_id=page_task.document_id,
                                chunks=chunk_data,
                                db=db
                            )
                            logger.info(f"[Chunking] 📦 Created {len(chunk_data)} chunk tasks with stored content")

                        page_record.chunk_count = len(chunks)
                        db.commit()

                    # ================================================================
                    # V3.0: CHUNKING END - Page {page_task.page_number}
                    # ================================================================
                    logger.info("=" * 80)
                    logger.info(f"[Chunking] ========== END CHUNKING PAGE {page_task.page_number} ==========")
                    logger.info("=" * 80)

                logger.info(f"✅ OCR completed for page {page_task.page_number} of {filename}")
                return page_output_path

        except Exception as e:
            logger.error(f"❌ OCR page task failed for page {page_task.page_number} of doc {page_task.document_id}: {e}", exc_info=True)
            raise

    def _convert_with_doc_service(self, input_path: str, output_dir: str, base_name: str) -> str:
        """
        Convert a document (Excel, Word, CSV, etc.) to markdown using doc_service.

        This method is used for file types that don't need OCR (already text-based).

        Args:
            input_path: Absolute path to the input file
            output_dir: Directory to save the output markdown file
            base_name: Base name for the output file (without extension)

        Returns:
            str: Path to the generated markdown file

        Raises:
            ValueError: If no converter is available for the file type
            Exception: If conversion fails
        """
        if not self.doc_converter_manager:
            raise ValueError("doc_converter_manager not initialized - cannot convert document files")

        file_path_obj = Path(input_path)

        # Find appropriate converter
        converter = self.doc_converter_manager.find_converter_for_file(file_path_obj)
        if not converter:
            raise ValueError(f"No converter found for file type: {file_path_obj.suffix}")

        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filename matching the expected pattern
        output_filename = f"{base_name}_nohf.md"
        output_path = Path(output_dir) / output_filename

        logger.info(f"📝 Converting {file_path_obj.name} to markdown at {output_path}")

        # Convert the file
        success = converter.convert_file(file_path_obj, output_path)

        if not success:
            raise Exception(f"Failed to convert {file_path_obj.name} with {converter.get_converter_info()['name']}")

        logger.info(f"✅ Successfully converted {file_path_obj.name} using {converter.get_converter_info()['name']}")
        return str(output_path)

    def _check_and_route_tabular_document(
        self,
        document_id: UUID,
        filename: str,
        file_ext: str,
        output_dir: str,
        page_output_path: str,
        db: Session
    ) -> bool:
        """
        Check if document should use TABULAR path and trigger extraction if so.

        This implements the CLASSIFICATION & ROUTING step from the Task Queue
        Workflow Redesign Plan. It runs AFTER convert but BEFORE chunking.

        Routing Logic:
        - CSV, Excel (.csv, .xlsx, .xls) → Always TABULAR
        - PDF/images with invoice/receipt/bank_statement → TABULAR
        - All other documents → STANDARD (continue to chunking)

        Args:
            document_id: Document UUID
            filename: Original filename
            file_ext: File extension (lowercase, with dot)
            output_dir: Output directory path
            page_output_path: Path to converted markdown file
            db: Database session

        Returns:
            True if document routed to TABULAR path (skip chunking)
            False if document should continue to STANDARD path (chunking)
        """
        from common.document_type_classifier import TabularDataDetector, DocumentTypeClassifier
        from queue_service import TaskQueueDocument
        from rag_service.graphrag_skip_config import GRAPHRAG_SKIP_DOCUMENT_TYPES

        logger.info("=" * 80)
        logger.info("[Routing] ========== DOCUMENT CLASSIFICATION & ROUTING START ==========")
        logger.info("=" * 80)
        logger.info(f"[Routing] Document ID: {document_id}")
        logger.info(f"[Routing] Filename: {filename}")
        logger.info(f"[Routing] File extension: {file_ext}")
        logger.info(f"[Routing] Output dir: {output_dir}")
        logger.info(f"[Routing] Page output path: {page_output_path}")
        logger.info("-" * 80)

        # ----------------------------------------------------------------
        # STEP 4.1: Get or create task_queue_document record
        # ----------------------------------------------------------------
        logger.info("[Routing] --- STEP 4.1: TaskQueueDocument Record ---")
        tq_doc = db.query(TaskQueueDocument).filter(
            TaskQueueDocument.document_id == document_id
        ).first()

        if not tq_doc:
            logger.info("[Routing] TaskQueueDocument NOT found, creating new record...")
            tq_doc = TaskQueueDocument(
                document_id=document_id,
                convert_status=TaskStatus.COMPLETED,
                convert_completed_at=datetime.now(timezone.utc),
                classification_status=TaskStatus.PROCESSING,
                classification_started_at=datetime.now(timezone.utc)
            )
            db.add(tq_doc)
            db.flush()
            logger.info(f"[Routing] Created TaskQueueDocument with convert_status=COMPLETED")
        else:
            logger.info(f"[Routing] Found existing TaskQueueDocument")
            logger.info(f"[Routing]   - Current convert_status: {tq_doc.convert_status}")
            logger.info(f"[Routing]   - Current processing_path: {tq_doc.processing_path}")
            # Update classification status
            tq_doc.classification_status = TaskStatus.PROCESSING
            tq_doc.classification_started_at = datetime.now(timezone.utc)
            db.flush()
            logger.info(f"[Routing] Updated classification_status to PROCESSING")

        try:
            # ----------------------------------------------------------------
            # STEP 4.2: LLM-based Document Type Classification
            # ----------------------------------------------------------------
            logger.info("-" * 80)
            logger.info("[Routing] --- STEP 4.2: LLM Document Type Classification ---")

            # Read content preview from converted markdown for classification
            content_preview = None
            detected_document_type = None
            document_types = []

            if page_output_path and os.path.exists(page_output_path):
                try:
                    with open(page_output_path, 'r', encoding='utf-8') as f:
                        content_preview = f.read(3000)  # First 3000 chars for classification
                    logger.info(f"[Routing] Loaded content preview: {len(content_preview)} chars")
                except Exception as e:
                    logger.warning(f"[Routing] Could not read content preview: {e}")

            # Use DocumentTypeClassifier for LLM-based classification
            try:
                from rag_service.llm_service import get_llm_service
                llm_service = get_llm_service()
                classifier = DocumentTypeClassifier(llm_client=llm_service)

                classification_result = classifier.classify(
                    filename=filename,
                    metadata={},
                    content_preview=content_preview
                )

                detected_document_type = classification_result.document_type
                document_types = [detected_document_type]

                logger.info(f"[Routing] LLM Classification result:")
                logger.info(f"[Routing]   - document_type: {detected_document_type}")
                logger.info(f"[Routing]   - confidence: {classification_result.confidence:.2f}")
                logger.info(f"[Routing]   - is_extractable: {classification_result.is_extractable}")
                logger.info(f"[Routing]   - schema_type: {classification_result.schema_type}")
                logger.info(f"[Routing]   - reasoning: {classification_result.reasoning}")

            except Exception as e:
                logger.warning(f"[Routing] LLM classification failed, using pattern fallback: {e}")
                # Fallback: use pattern-based classification from filename
                classifier = DocumentTypeClassifier()
                classification_result = classifier.classify(
                    filename=filename,
                    metadata={},
                    content_preview=content_preview
                )
                detected_document_type = classification_result.document_type
                document_types = [detected_document_type]
                logger.info(f"[Routing] Pattern-based classification: {detected_document_type}")

            # ----------------------------------------------------------------
            # STEP 4.3: Tabular Data Detection (using detected document_type)
            # ----------------------------------------------------------------
            logger.info("-" * 80)
            logger.info("[Routing] --- STEP 4.3: Tabular Data Detection ---")
            logger.info(f"[Routing] Calling TabularDataDetector.is_tabular_data()...")
            logger.info(f"[Routing]   - filename: {filename}")
            logger.info(f"[Routing]   - document_type: {detected_document_type}")

            # Check if this is a tabular document
            # Now passes detected_document_type for better classification
            is_tabular, reason = TabularDataDetector.is_tabular_data(
                filename=filename,
                document_type=detected_document_type,  # Use LLM-detected type
                content=content_preview  # Also check content patterns
            )

            logger.info(f"[Routing] Detection result:")
            logger.info(f"[Routing]   - is_tabular: {is_tabular}")
            logger.info(f"[Routing]   - reason: {reason}")

            # Determine if GraphRAG should be skipped based on document type
            should_skip_graphrag = detected_document_type in GRAPHRAG_SKIP_DOCUMENT_TYPES
            skip_graphrag_reason = f"document_type:{detected_document_type}" if should_skip_graphrag else None
            logger.info(f"[Routing] GraphRAG skip decision:")
            logger.info(f"[Routing]   - should_skip_graphrag: {should_skip_graphrag}")
            logger.info(f"[Routing]   - skip_reason: {skip_graphrag_reason}")

            # ----------------------------------------------------------------
            # STEP 4.4: Update Document Record
            # ----------------------------------------------------------------
            logger.info("-" * 80)
            logger.info("[Routing] --- STEP 4.4: Update Document Record ---")
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                logger.info(f"[Routing] Found document record")
                logger.info(f"[Routing]   - Current is_tabular_data: {doc.is_tabular_data}")
                logger.info(f"[Routing]   - Current processing_path: {doc.processing_path}")
                logger.info(f"[Routing]   - Current convert_status: {doc.convert_status}")
                logger.info(f"[Routing]   - Current skip_graphrag: {doc.skip_graphrag}")

                doc.is_tabular_data = is_tabular
                doc.processing_path = 'tabular' if is_tabular else 'standard'

                # Set skip_graphrag flag based on document type classification
                doc.skip_graphrag = should_skip_graphrag
                doc.skip_graphrag_reason = skip_graphrag_reason

                # Store early document_metadata with classification results
                # This ensures metadata is available BEFORE GraphRAG runs
                early_metadata = {
                    "document_types": document_types,  # List of applicable document types
                    "classification_source": "early_routing",
                    "classification_confidence": classification_result.confidence if classification_result else 0.5,
                    "is_extractable": classification_result.is_extractable if classification_result else False,
                    "schema_type": classification_result.schema_type if classification_result else None,
                }
                # Merge with existing metadata if present, or set new
                if doc.document_metadata:
                    doc.document_metadata.update(early_metadata)
                else:
                    doc.document_metadata = early_metadata

                # IMPORTANT: Update convert_status to CONVERTED so eligibility check passes
                # This is needed because the extraction eligibility checker checks document.convert_status
                from db.models import ConvertStatus
                doc.convert_status = ConvertStatus.CONVERTED
                db.flush()

                logger.info(f"[Routing] Updated document record:")
                logger.info(f"[Routing]   - is_tabular_data: {doc.is_tabular_data}")
                logger.info(f"[Routing]   - processing_path: {doc.processing_path}")
                logger.info(f"[Routing]   - convert_status: {doc.convert_status}")
                logger.info(f"[Routing]   - skip_graphrag: {doc.skip_graphrag}")
                logger.info(f"[Routing]   - skip_graphrag_reason: {doc.skip_graphrag_reason}")
                logger.info(f"[Routing]   - document_metadata.document_types: {document_types}")
            else:
                logger.error(f"[Routing] ERROR: Document {document_id} not found!")

            # Update task_queue_document
            tq_doc.processing_path = 'tabular' if is_tabular else 'standard'
            tq_doc.classification_status = TaskStatus.COMPLETED
            tq_doc.classification_completed_at = datetime.now(timezone.utc)
            logger.info(f"[Routing] Updated TaskQueueDocument: processing_path={tq_doc.processing_path}, classification_status=COMPLETED")

            # ----------------------------------------------------------------
            # STEP 4.5: Route to appropriate path
            # ----------------------------------------------------------------
            logger.info("-" * 80)
            logger.info("[Routing] --- STEP 4.5: Execute Routing Decision ---")

            if is_tabular:
                # TABULAR PATH: Trigger data extraction
                logger.info("[Routing] *** DECISION: TABULAR PATH ***")
                logger.info("[Routing] Action: Trigger tabular data extraction")
                logger.info("[Routing] Action: Skip chunking")

                # Update extraction status to pending
                tq_doc.extraction_status = TaskStatus.PENDING
                db.commit()
                logger.info(f"[Routing] Set extraction_status to PENDING")

                # Trigger tabular extraction in background
                source_name = os.path.splitext(filename)[0]
                logger.info(f"[Routing] Calling _trigger_tabular_extraction()...")
                logger.info(f"[Routing]   - source_name: {source_name}")
                logger.info(f"[Routing]   - output_dir: {output_dir}")

                self._trigger_tabular_extraction(
                    document_id=document_id,
                    source_name=source_name,
                    output_dir=output_dir,
                    filename=filename
                )

                logger.info("=" * 80)
                logger.info("[Routing] ========== CLASSIFICATION & ROUTING COMPLETE ==========")
                logger.info("[Routing] Result: TABULAR PATH")
                logger.info("[Routing] Extraction thread started in background")
                logger.info("=" * 80)
                return True
            else:
                # STANDARD PATH: Continue to chunking
                logger.info("[Routing] *** DECISION: STANDARD PATH ***")
                logger.info("[Routing] Action: Continue to semantic chunking")
                logger.info("[Routing] Action: Skip extraction")

                tq_doc.extraction_status = TaskStatus.SKIPPED
                db.commit()
                logger.info(f"[Routing] Set extraction_status to SKIPPED")

                logger.info("=" * 80)
                logger.info("[Routing] ========== CLASSIFICATION & ROUTING COMPLETE ==========")
                logger.info("[Routing] Result: STANDARD PATH")
                logger.info("[Routing] Document will proceed to chunking")
                logger.info("=" * 80)
                return False

        except Exception as e:
            logger.error(f"[Routing] Classification failed: {e}", exc_info=True)
            # On error, default to STANDARD path
            tq_doc.classification_status = TaskStatus.FAILED
            tq_doc.classification_error = str(e)
            tq_doc.processing_path = 'standard'
            db.commit()
            return False

    def _trigger_tabular_extraction(
        self,
        document_id: UUID,
        source_name: str,
        output_dir: str,
        filename: str
    ) -> None:
        """
        Trigger tabular data extraction in background thread.

        Args:
            document_id: Document UUID
            source_name: Document source name (filename without extension)
            output_dir: Output directory path
            filename: Original filename
        """
        logger.info("=" * 80)
        logger.info("[Extraction] ========== TRIGGER TABULAR EXTRACTION ==========")
        logger.info("=" * 80)
        logger.info(f"[Extraction] Document ID: {document_id}")
        logger.info(f"[Extraction] Source name: {source_name}")
        logger.info(f"[Extraction] Output dir: {output_dir}")
        logger.info(f"[Extraction] Filename: {filename}")
        logger.info("-" * 80)

        try:
            from services.tabular_extraction_service import trigger_tabular_extraction

            logger.info(f"[Extraction] Importing trigger_tabular_extraction from tabular_extraction_service...")
            logger.info(f"[Extraction] Calling trigger_tabular_extraction()...")

            trigger_tabular_extraction(
                document_id=document_id,
                source_name=source_name,
                output_dir=output_dir,
                filename=filename,
                conversion_id=None,
                broadcast_callback=self.document_status_broadcast
            )

            logger.info(f"[Extraction] ✅ Tabular extraction TRIGGERED successfully")
            logger.info(f"[Extraction] Background thread started for: {source_name}")
            logger.info("=" * 80)

        except ImportError as e:
            logger.error("=" * 80)
            logger.error(f"[Extraction] ❌ IMPORT ERROR: Tabular extraction service not available")
            logger.error(f"[Extraction] Error: {e}")
            logger.error("=" * 80)
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"[Extraction] ❌ FAILED to trigger extraction")
            logger.error(f"[Extraction] Error: {e}")
            logger.error("=" * 80, exc_info=True)

    def process_vector_chunk_task(self, chunk_task) -> bool:
        """
        Process Vector indexing task for a single chunk.

        V3.0: Now uses stored chunk content from database instead of re-chunking.

        Args:
            chunk_task: ChunkTaskData with chunk_id, document_id, page_id, chunk_index, etc.

        Returns:
            bool: True if indexing succeeded
        """
        from rag_service.markdown_chunker import chunk_markdown_with_summaries
        from rag_service.vectorstore import get_vectorstore
        from queue_service import TaskQueuePage, TaskQueueChunk
        from langchain_core.documents import Document as LangchainDocument

        try:
            # ================================================================
            # V3.0: VECTOR INDEXING START
            # ================================================================
            logger.info("=" * 80)
            logger.info(f"[Vector] ========== START VECTOR INDEXING ==========")
            logger.info("=" * 80)
            logger.info(f"[Vector] Chunk ID: {chunk_task.chunk_id}")
            logger.info(f"[Vector] Chunk Index: {chunk_task.chunk_index}")

            with get_db_session() as db:
                # V3.0: First try to get stored chunk content from database
                chunk_record = db.query(TaskQueueChunk).filter(
                    TaskQueueChunk.id == chunk_task.id
                ).first()

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

                # Check if this is a single-file document type
                file_ext = Path(doc.filename).suffix.lower() if doc.filename else ''
                is_single_file_doc = file_ext in self.DOC_SERVICE_EXTENSIONS
                canonical_page_number = 0 if is_single_file_doc else page_number

                # V3.0: Use stored content if available, otherwise fall back to re-chunking
                if chunk_record and chunk_record.chunk_content:
                    logger.info(f"[Vector] 📦 V3.0 OPTIMIZATION: Using stored chunk content ({len(chunk_record.chunk_content)} chars)")
                    chunk_content = chunk_record.chunk_content
                    chunk_metadata = chunk_record.chunk_metadata or {}
                    chunk_metadata["chunk_id"] = chunk_task.chunk_id
                    chunk_metadata["document_id"] = str(chunk_task.document_id)
                    chunk = LangchainDocument(page_content=chunk_content, metadata=chunk_metadata)
                else:
                    # Fallback: Re-chunk if stored content not available
                    logger.warning(f"[Vector] ⚠️ V3.0 FALLBACK: Stored content not found, re-chunking...")
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
                        new_id = f"{chunk_task.document_id}_{canonical_page_number}_{idx}_{content_hash}"
                        old_to_new_id_map[old_id] = new_id

                    # Use canonical chunk_id for single-file documents to prevent duplicates
                    if is_single_file_doc:
                        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:16]
                        canonical_chunk_id = f"{chunk_task.document_id}_{canonical_page_number}_{chunk_task.chunk_index}_{content_hash}"
                        chunk.metadata["chunk_id"] = canonical_chunk_id
                    else:
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

            # For single-file documents, check if this chunk already exists in Qdrant
            # This prevents duplicate vectors when multiple page tasks process the same content
            if is_single_file_doc:
                canonical_chunk_id = chunk.metadata["chunk_id"]
                try:
                    # Check if point already exists using the Qdrant client
                    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
                    existing = vectorstore.client.scroll(
                        collection_name=vectorstore.collection_name,
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="metadata.chunk_id",
                                    match=MatchValue(value=canonical_chunk_id)
                                )
                            ]
                        ),
                        limit=1
                    )
                    if existing and existing[0]:
                        logger.info(f"⏭️ Skipping duplicate vector for chunk: {canonical_chunk_id}")
                        return True
                except Exception as check_err:
                    logger.debug(f"Could not check for existing chunk (will proceed): {check_err}")

            vectorstore.add_documents([chunk])

            # ================================================================
            # V3.0: VECTOR INDEXING END
            # ================================================================
            logger.info("=" * 80)
            logger.info(f"[Vector] ========== END VECTOR INDEXING ==========")
            logger.info("=" * 80)
            logger.info(f"[Vector] ✅ Successfully indexed chunk: {chunk_task.chunk_id}")
            return True

        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"[Vector] ========== VECTOR INDEXING FAILED ==========")
            logger.error("=" * 80)
            logger.error(f"❌ Vector chunk task failed for chunk {chunk_task.chunk_id}: {e}", exc_info=True)
            raise

    def process_graphrag_chunk_task(self, chunk_task) -> tuple:
        """
        Process GraphRAG indexing task for a single chunk.

        V3.0: Now uses stored chunk content from database instead of re-chunking.

        Args:
            chunk_task: ChunkTaskData with chunk_id, document_id, page_id, chunk_index, etc.

        Returns:
            tuple: (entities_count, relationships_count)
        """
        from rag_service.markdown_chunker import chunk_markdown_with_summaries
        from queue_service import TaskQueuePage, TaskQueueChunk

        try:
            # ================================================================
            # V3.0: GRAPHRAG INDEXING START
            # ================================================================
            logger.info("=" * 80)
            logger.info(f"[GraphRAG] ========== START GRAPHRAG INDEXING ==========")
            logger.info("=" * 80)
            logger.info(f"[GraphRAG] Chunk ID: {chunk_task.chunk_id}")
            logger.info(f"[GraphRAG] Chunk Index: {chunk_task.chunk_index}")

            if not self.graph_rag_index_enabled:
                logger.info(f"[GraphRAG] GraphRAG indexing disabled, skipping chunk: {chunk_task.chunk_id}")
                logger.info("=" * 80)
                return (0, 0)

            with get_db_session() as db:
                # V3.0: First try to get stored chunk content from database
                chunk_record = db.query(TaskQueueChunk).filter(
                    TaskQueueChunk.id == chunk_task.id
                ).first()

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

                # V3.0: Use stored content if available, otherwise fall back to re-chunking
                if chunk_record and chunk_record.chunk_content:
                    logger.info(f"[GraphRAG] 📦 V3.0 OPTIMIZATION: Using stored chunk content ({len(chunk_record.chunk_content)} chars)")
                    chunk_content = chunk_record.chunk_content
                    chunk_metadata = chunk_record.chunk_metadata or {}
                    chunk_metadata["chunk_id"] = chunk_task.chunk_id
                    chunk_metadata["document_id"] = str(chunk_task.document_id)
                else:
                    # Fallback: Re-chunk if stored content not available
                    logger.warning(f"[GraphRAG] ⚠️ V3.0 FALLBACK: Stored content not found, re-chunking...")
                    result = chunk_markdown_with_summaries(page_file_path, source_name=source_name)
                    chunks = result.chunks

                    if not chunks or chunk_task.chunk_index >= len(chunks):
                        raise ValueError(
                            f"Chunk index {chunk_task.chunk_index} out of range. "
                            f"Page has {len(chunks) if chunks else 0} chunks."
                        )

                    chunk = chunks[chunk_task.chunk_index]
                    chunk_content = chunk.page_content
                    chunk_metadata = chunk.metadata

            chunk_data = {
                "id": chunk_task.chunk_id,
                "page_content": chunk_content,
                "metadata": chunk_metadata,
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
                logger.warning("[GraphRAG] GraphRAG indexer not available")
            except Exception as e:
                logger.error(f"[GraphRAG] GraphRAG indexing failed for chunk {chunk_task.chunk_id}: {e}")

            # ================================================================
            # V3.0: GRAPHRAG INDEXING END
            # ================================================================
            logger.info("=" * 80)
            logger.info(f"[GraphRAG] ========== END GRAPHRAG INDEXING ==========")
            logger.info("=" * 80)
            logger.info(f"[GraphRAG] ✅ Chunk: {chunk_task.chunk_id}")
            logger.info(f"[GraphRAG] Entities extracted: {entities_count}")
            logger.info(f"[GraphRAG] Relationships extracted: {relationships_count}")
            return (entities_count, relationships_count)

        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"[GraphRAG] ========== GRAPHRAG INDEXING FAILED ==========")
            logger.error("=" * 80)
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

                        # Check if this is a tabular document - use different processing path
                        if doc.is_tabular_data or doc.processing_path == 'tabular':
                            # Tabular documents need extraction, not standard chunking
                            logger.info(f"[Recovery] Resuming TABULAR extraction for: {doc.filename}")

                            # Get the output directory from the document
                            if doc.output_path:
                                output_dir = self.resolve_file_path(doc.output_path, self.output_dir)
                            else:
                                base_name = os.path.splitext(doc.filename)[0]
                                output_dir = os.path.join(self.output_dir, base_name)

                            self._trigger_tabular_extraction(
                                document_id=doc.id,
                                source_name=file_name_without_ext,
                                output_dir=output_dir,
                                filename=doc.filename
                            )
                        else:
                            # Standard documents use chunking + vector indexing
                            logger.info(f"[Recovery] Resuming STANDARD indexing for: {doc.filename}")
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
