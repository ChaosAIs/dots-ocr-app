"""
Task Queue Integration for Data Extraction

Integrates structured data extraction with the hierarchical task queue.
Extraction is triggered after GraphRAG indexing completes for a document.
"""

import os
import logging
import threading
from typing import Optional
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from db.database import create_db_session
from db.models import Document
from queue_service.models import TaskQueueChunk, TaskStatus

logger = logging.getLogger(__name__)


def check_and_trigger_data_extraction(
    document_id: UUID,
    db: Optional[Session] = None
) -> None:
    """
    Check if all Vector chunks for a document are complete.
    If so, trigger structured data extraction in a background thread.

    This function should be called after each Vector chunk completes.
    Data extraction runs BEFORE GraphRAG indexing to ensure structured
    data is available early in the pipeline.

    Args:
        document_id: Document UUID
        db: Database session (optional, creates new if not provided)
    """
    # Check if extraction is enabled
    if os.getenv("DATA_EXTRACTION_ENABLED", "true").lower() != "true":
        return

    should_close_db = db is None
    if db is None:
        db = create_db_session()

    try:
        # Check if all Vector chunks are complete
        vector_stats = db.query(
            func.count().label("total"),
            func.count().filter(TaskQueueChunk.vector_status == TaskStatus.COMPLETED).label("completed"),
        ).filter(TaskQueueChunk.document_id == document_id).first()

        if not vector_stats or vector_stats.total == 0:
            return

        # Not all chunks complete yet
        if vector_stats.completed < vector_stats.total:
            return

        # Check document extraction status
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return

        # Skip if extraction already done or in progress
        # Note: 'pending' and None are allowed - these indicate extraction should run
        # 'skipped' is now allowed to be retried after metadata is populated
        extraction_status = getattr(doc, 'extraction_status', None)
        if extraction_status in ['completed', 'processing']:
            logger.debug(f"Extraction already {extraction_status} for document {document_id}")
            return

        # All Vector chunks complete - trigger extraction before GraphRAG
        logger.info(f"[Extraction] All Vector chunks complete for doc={document_id}, triggering data extraction")

        # Run extraction in background thread
        def _extraction_task():
            try:
                run_document_extraction(document_id)
            except Exception as e:
                logger.error(f"[Extraction] Failed for doc={document_id}: {e}", exc_info=True)

        extraction_thread = threading.Thread(
            target=_extraction_task,
            daemon=True,
            name=f"extraction-{str(document_id)[:8]}"
        )
        extraction_thread.start()

    except Exception as e:
        logger.error(f"Error checking data extraction trigger: {e}")
    finally:
        if should_close_db:
            db.close()


def run_document_extraction(document_id: UUID) -> bool:
    """
    Run structured data extraction for a document.

    This is called from the background thread after GraphRAG completes.

    Args:
        document_id: Document UUID

    Returns:
        True if extraction succeeded, False otherwise
    """
    from .extraction_service import ExtractionService
    from .eligibility_checker import ExtractionEligibilityChecker

    db = create_db_session()
    try:
        # Get LLM client for eligibility checking and extraction
        llm_client = get_extraction_llm_client()

        # Check eligibility first (with LLM for document type inference)
        checker = ExtractionEligibilityChecker(db, llm_client=llm_client)
        eligible, schema_type, reason = checker.check_eligibility(document_id)

        if not eligible:
            logger.info(f"[Extraction] Document {document_id} not eligible: {reason}")
            checker.update_document_extraction_status(
                document_id, False, None, "skipped"
            )
            return False

        # Update status to eligible
        checker.update_document_extraction_status(
            document_id, True, schema_type, "pending"
        )

        # Create extraction service
        # Note: LLM client would be injected here in production
        llm_client = get_extraction_llm_client()
        service = ExtractionService(db, llm_client=llm_client)

        # Run extraction
        result = service.extract_document(document_id)

        if result:
            logger.info(f"[Extraction] ✅ Completed for document {document_id}")
            return True
        else:
            logger.warning(f"[Extraction] ⚠️ No result for document {document_id}")
            return False

    except Exception as e:
        logger.error(f"[Extraction] Failed for document {document_id}: {e}", exc_info=True)

        # Update status to failed
        try:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.extraction_status = "failed"
                doc.extraction_error = str(e)
                db.commit()
        except Exception as db_error:
            logger.warning(f"[Extraction] Could not update failed status: {db_error}")

        return False
    finally:
        db.close()


def get_extraction_llm_client():
    """
    Get the LLM client for extraction.

    Uses the same LLM backend as the RAG service.
    Wraps the LangChain chat model for simpler extraction interface.

    Returns:
        LLM client wrapper or None
    """
    try:
        from rag_service.llm_service import get_llm_service

        llm_service = get_llm_service()
        if not llm_service.is_available():
            logger.warning("[Extraction] LLM service not available")
            return None

        # Get chat model with extraction-appropriate settings
        chat_model = llm_service.get_chat_model(
            temperature=0.1,  # Low temperature for structured extraction
            num_ctx=8192,     # Context for document content
            num_predict=4096  # Allow for full JSON response
        )

        # Create wrapper that provides simple generate() interface
        class LLMClientWrapper:
            def __init__(self, chat_model):
                self.chat_model = chat_model

            def generate(self, prompt: str) -> str:
                """Generate a response from the prompt."""
                from langchain_core.messages import HumanMessage
                response = self.chat_model.invoke([HumanMessage(content=prompt)])
                return response.content

        return LLMClientWrapper(chat_model)

    except ImportError as e:
        logger.warning(f"[Extraction] Could not import LLM service: {e}")
        return None
    except Exception as e:
        logger.warning(f"[Extraction] Could not create LLM client: {e}")
        return None


def process_pending_extractions(limit: int = 10) -> int:
    """
    Process pending extraction tasks.

    This can be called periodically to handle any extractions that
    were missed or failed.

    Args:
        limit: Maximum number of documents to process

    Returns:
        Number of documents processed
    """
    from .eligibility_checker import ExtractionEligibilityChecker

    db = create_db_session()
    try:
        checker = ExtractionEligibilityChecker(db)
        pending_docs = checker.get_pending_extractions(limit=limit)

        processed = 0
        for doc_id in pending_docs:
            try:
                success = run_document_extraction(doc_id)
                if success:
                    processed += 1
            except Exception as e:
                logger.error(f"Error processing extraction for {doc_id}: {e}")

        logger.info(f"[Extraction] Processed {processed}/{len(pending_docs)} pending documents")
        return processed

    finally:
        db.close()
