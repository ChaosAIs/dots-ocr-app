"""
Document repository for CRUD operations.
"""
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy import and_

from .models import Document, DocumentStatusLog, UploadStatus, ConvertStatus, IndexStatus

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Repository for document operations."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        filename: str,
        original_filename: str,
        file_path: str,
        file_size: int,
        mime_type: Optional[str] = None,
        total_pages: int = 0,
    ) -> Document:
        """Create a new document record."""
        doc = Document(
            filename=filename,
            original_filename=original_filename,
            file_path=file_path,
            file_size=file_size,
            mime_type=mime_type,
            total_pages=total_pages,
            upload_status=UploadStatus.UPLOADED,
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        
        # Log the upload
        self._log_status(doc.id, "upload", None, UploadStatus.UPLOADED.value, "File uploaded successfully")
        
        logger.info(f"Created document record: {filename}")
        return doc

    def get_by_filename(self, filename: str) -> Optional[Document]:
        """Get document by filename."""
        return self.db.query(Document).filter(
            and_(Document.filename == filename, Document.deleted_at.is_(None))
        ).first()

    def get_by_id(self, doc_id: UUID) -> Optional[Document]:
        """Get document by ID."""
        return self.db.query(Document).filter(
            and_(Document.id == doc_id, Document.deleted_at.is_(None))
        ).first()

    def get_all(self) -> List[Document]:
        """Get all active documents."""
        return self.db.query(Document).filter(Document.deleted_at.is_(None)).order_by(Document.created_at.desc()).all()

    def get_all_with_metadata(self) -> List[Document]:
        """Get all active documents that have metadata extracted."""
        return self.db.query(Document).filter(
            and_(
                Document.deleted_at.is_(None),
                Document.document_metadata.isnot(None)
            )
        ).order_by(Document.created_at.desc()).all()

    def update_convert_status(
        self,
        doc: Document,
        status: ConvertStatus,
        converted_pages: int = None,
        output_path: str = None,
        message: str = None,
    ) -> Document:
        """Update document conversion status."""
        old_status = doc.convert_status.value if doc.convert_status else None
        doc.convert_status = status
        if converted_pages is not None:
            doc.converted_pages = converted_pages
        if output_path:
            doc.output_path = output_path
        self.db.commit()
        self.db.refresh(doc)
        
        self._log_status(doc.id, "convert", old_status, status.value, message)
        return doc

    def update_index_status(
        self,
        doc: Document,
        status: IndexStatus,
        indexed_chunks: int = None,
        message: str = None,
        indexed_pages_info: Dict[str, Any] = None,
    ) -> Document:
        """Update document indexing status."""
        old_status = doc.index_status.value if doc.index_status else None
        doc.index_status = status
        if indexed_chunks is not None:
            doc.indexed_chunks = indexed_chunks
        if indexed_pages_info is not None:
            doc.indexed_pages_info = indexed_pages_info
        self.db.commit()
        self.db.refresh(doc)

        self._log_status(doc.id, "index", old_status, status.value, message)
        return doc

    def update_document_metadata(
        self,
        doc: Document,
        metadata: Dict[str, Any],
        message: str = None,
    ) -> Document:
        """Update document metadata after extraction."""
        doc.document_metadata = metadata
        self.db.commit()
        self.db.refresh(doc)

        if message:
            self._log_status(doc.id, "metadata", None, "extracted", message)

        logger.info(f"Updated metadata for document: {doc.filename}")
        return doc

    def mark_page_indexed(self, doc: Document, page_num: int, chunks: int = 0) -> Document:
        """Mark a specific page as indexed."""
        info = doc.indexed_pages_info or {"indexed": [], "pending": [], "failed": []}

        # Remove from pending/failed if present
        if page_num in info.get("pending", []):
            info["pending"].remove(page_num)
        if page_num in info.get("failed", []):
            info["failed"].remove(page_num)

        # Add to indexed if not already there
        if page_num not in info.get("indexed", []):
            info["indexed"] = sorted(info.get("indexed", []) + [page_num])

        doc.indexed_pages_info = info
        doc.indexed_chunks = (doc.indexed_chunks or 0) + chunks

        # Update status based on pages
        if doc.total_pages > 0:
            if len(info["indexed"]) >= doc.total_pages:
                doc.index_status = IndexStatus.INDEXED
            elif len(info["indexed"]) > 0:
                doc.index_status = IndexStatus.PARTIAL

        self.db.commit()
        self.db.refresh(doc)
        return doc

    def mark_page_failed(self, doc: Document, page_num: int, error: str = None) -> Document:
        """Mark a specific page as failed to index."""
        info = doc.indexed_pages_info or {"indexed": [], "pending": [], "failed": []}

        # Remove from pending if present
        if page_num in info.get("pending", []):
            info["pending"].remove(page_num)

        # Add to failed if not already there
        if page_num not in info.get("failed", []):
            info["failed"] = sorted(info.get("failed", []) + [page_num])

        doc.indexed_pages_info = info
        self.db.commit()
        self.db.refresh(doc)

        if error:
            self._log_status(doc.id, "index", None, "page_failed", f"Page {page_num} failed: {error}")

        return doc

    def init_pages_for_indexing(self, doc: Document, page_numbers: List[int]) -> Document:
        """Initialize pages as pending for indexing."""
        info = doc.indexed_pages_info or {"indexed": [], "pending": [], "failed": []}

        # Add new pages to pending (excluding already indexed ones)
        existing_indexed = set(info.get("indexed", []))
        new_pending = [p for p in page_numbers if p not in existing_indexed]
        info["pending"] = sorted(list(set(info.get("pending", []) + new_pending)))

        doc.indexed_pages_info = info
        doc.index_status = IndexStatus.INDEXING
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def get_pending_pages(self, doc: Document) -> List[int]:
        """Get list of pages that still need indexing."""
        info = doc.indexed_pages_info or {"indexed": [], "pending": [], "failed": []}
        return info.get("pending", [])

    def get_indexed_pages(self, doc: Document) -> List[int]:
        """Get list of already indexed pages."""
        info = doc.indexed_pages_info or {"indexed": [], "pending": [], "failed": []}
        return info.get("indexed", [])

    def is_fully_indexed(self, doc: Document) -> bool:
        """Check if all pages are indexed."""
        if doc.index_status == IndexStatus.INDEXED:
            return True
        if doc.total_pages == 0:
            return False
        info = doc.indexed_pages_info or {"indexed": [], "pending": [], "failed": []}
        return len(info.get("indexed", [])) >= doc.total_pages

    def hard_delete(self, doc: Document) -> None:
        """Hard delete a document and its status logs (cascade)."""
        doc_id = doc.id
        filename = doc.filename
        self.db.delete(doc)
        self.db.commit()
        logger.info(f"Hard deleted document {filename} (id: {doc_id}) and associated status logs")

    def get_status_logs(self, doc_id: UUID, limit: int = 50) -> List[DocumentStatusLog]:
        """Get status logs for a document."""
        return self.db.query(DocumentStatusLog).filter(
            DocumentStatusLog.document_id == doc_id
        ).order_by(DocumentStatusLog.created_at.desc()).limit(limit).all()

    def update_total_pages(self, doc: Document, total_pages: int) -> Document:
        """Update total pages count."""
        doc.total_pages = total_pages
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def exists(self, filename: str) -> bool:
        """Check if a document exists."""
        return self.db.query(Document).filter(
            and_(Document.filename == filename, Document.deleted_at.is_(None))
        ).count() > 0

    def is_indexed(self, source_name: str) -> bool:
        """
        Check if a document is indexed based on source name (filename without extension).
        Returns True if index_status is INDEXED.
        """
        # Try exact match first (source_name could be filename without ext)
        # The source_name from markdown is the folder name, which matches filename without extension
        docs = self.db.query(Document).filter(
            and_(
                Document.filename.like(f"{source_name}.%"),
                Document.deleted_at.is_(None)
            )
        ).all()

        for doc in docs:
            if doc.index_status == IndexStatus.INDEXED:
                return True
        return False

    def get_documents_needing_index(self) -> List[Document]:
        """
        Get all documents that need indexing (converted but not indexed).
        """
        return self.db.query(Document).filter(
            and_(
                Document.deleted_at.is_(None),
                Document.convert_status.in_([ConvertStatus.CONVERTED, ConvertStatus.PARTIAL]),
                Document.index_status != IndexStatus.INDEXED,
            )
        ).all()

    def get_or_create(
        self,
        filename: str,
        original_filename: str,
        file_path: str,
        file_size: int,
        mime_type: Optional[str] = None,
        total_pages: int = 0,
    ) -> tuple[Document, bool]:
        """Get existing document or create new one. Returns (document, created)."""
        existing = self.get_by_filename(filename)
        if existing:
            return existing, False
        return self.create(filename, original_filename, file_path, file_size, mime_type, total_pages), True

    # ========== Granular Indexing Status Methods ==========

    def init_indexing_details(self, doc: Document) -> Document:
        """Initialize indexing_details structure for a new document."""
        # Initialize empty dict if None
        if not doc.indexing_details:
            doc.indexing_details = {}

        # Ensure all required fields exist (merge with existing data)
        if "version" not in doc.indexing_details:
            doc.indexing_details["version"] = "1.0"

        if "vector_indexing" not in doc.indexing_details:
            doc.indexing_details["vector_indexing"] = {
                "status": "pending",
                "total_chunks": 0,
                "indexed_chunks": 0,
                "failed_chunks": 0,
                "pages": {},
                "chunks": {}
            }

        if "metadata_extraction" not in doc.indexing_details:
            doc.indexing_details["metadata_extraction"] = {
                "status": "pending",
                "error": None,
                "retry_count": 0
            }

        if "graphrag_indexing" not in doc.indexing_details:
            doc.indexing_details["graphrag_indexing"] = {
                "status": "pending",
                "total_chunks": 0,
                "expected_total_chunks": 0,
                "processed_chunks": 0,
                "failed_chunks": 0,
                "skipped_chunks": 0,
                "entities_extracted": 0,
                "relationships_extracted": 0,
                "chunks": {}
            }

        # Mark as modified for SQLAlchemy to detect changes
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(doc, "indexing_details")

        self.db.commit()
        self.db.refresh(doc)
        return doc

    def start_vector_indexing(self, doc: Document) -> Document:
        """
        Mark vector indexing as started/processing.
        Call this when Phase 1 (vector indexing) begins.
        """
        if not doc.indexing_details:
            self.init_indexing_details(doc)

        details = doc.indexing_details
        vector = details["vector_indexing"]

        # Only update if still pending (don't overwrite partial/completed)
        if vector.get("status") in ["pending", None]:
            vector["status"] = "processing"
            vector["started_at"] = datetime.utcnow().isoformat()

            doc.indexing_details = details
            flag_modified(doc, "indexing_details")
            self.db.commit()
            self.db.refresh(doc)

            logger.info(f"Started vector indexing for {doc.filename}")

        return doc

    def update_vector_indexing_status(
        self,
        doc: Document,
        page_file_path: str,
        chunk_ids: List[str],
        status: str,  # "success" or "failed"
        error: str = None,
        page_number: int = None,
    ) -> Document:
        """
        Update vector indexing status for a page and its chunks.

        Args:
            doc: Document instance
            page_file_path: Path to the _nohf.md file
            chunk_ids: List of chunk UUIDs from this page
            status: "success" or "failed"
            error: Error message if failed
            page_number: Page number (optional)
        """
        if not doc.indexing_details:
            self.init_indexing_details(doc)

        details = doc.indexing_details
        vector = details["vector_indexing"]

        # Extract page identifier from file path
        page_key = page_file_path.split("/")[-1].replace("_nohf.md", "")

        # Update page status
        if page_key not in vector["pages"]:
            vector["pages"][page_key] = {}

        page_info = vector["pages"][page_key]
        page_info["status"] = status
        page_info["file_path"] = page_file_path
        page_info["chunk_count"] = len(chunk_ids)

        if page_number is not None:
            page_info["page"] = page_number

        if status == "success":
            page_info["indexed_at"] = datetime.utcnow().isoformat()
            page_info["error"] = None
        else:
            page_info["failed_at"] = datetime.utcnow().isoformat()
            page_info["error"] = error
            page_info["retry_count"] = page_info.get("retry_count", 0) + 1

        # Update chunk statuses
        for chunk_id in chunk_ids:
            if chunk_id not in vector["chunks"]:
                vector["chunks"][chunk_id] = {}

            chunk_info = vector["chunks"][chunk_id]
            chunk_info["status"] = status
            chunk_info["file_path"] = page_file_path

            if page_number is not None:
                chunk_info["page"] = page_number

            if status == "success":
                chunk_info["indexed_at"] = datetime.utcnow().isoformat()
                chunk_info["error"] = None
            else:
                chunk_info["error"] = error

        # Update totals
        total_chunks = len(vector["chunks"])
        indexed_chunks = sum(1 for c in vector["chunks"].values() if c.get("status") == "success")
        failed_chunks = sum(1 for c in vector["chunks"].values() if c.get("status") == "failed")

        vector["total_chunks"] = total_chunks
        vector["indexed_chunks"] = indexed_chunks
        vector["failed_chunks"] = failed_chunks

        # Update overall status
        if failed_chunks > 0 and indexed_chunks > 0:
            vector["status"] = "partial"
        elif failed_chunks > 0 and indexed_chunks == 0:
            vector["status"] = "failed"
        elif indexed_chunks == total_chunks and total_chunks > 0:
            vector["status"] = "completed"
            vector["completed_at"] = datetime.utcnow().isoformat()
        elif indexed_chunks > 0:
            vector["status"] = "partial"

        doc.indexing_details = details
        # Mark the JSONB field as modified so SQLAlchemy detects the change
        flag_modified(doc, "indexing_details")
        self.db.commit()
        self.db.refresh(doc)

        logger.info(f"Updated vector indexing status for {page_key}: {status} ({len(chunk_ids)} chunks)")
        return doc

    def update_metadata_extraction_status(
        self,
        doc: Document,
        status: str,  # "completed", "failed", "skipped"
        error: str = None,
    ) -> Document:
        """Update metadata extraction status."""
        if not doc.indexing_details:
            self.init_indexing_details(doc)

        details = doc.indexing_details
        metadata = details["metadata_extraction"]

        metadata["status"] = status

        if status == "completed":
            metadata["extracted_at"] = datetime.utcnow().isoformat()
            metadata["error"] = None
        elif status == "failed":
            metadata["error"] = error
            metadata["retry_count"] = metadata.get("retry_count", 0) + 1

        doc.indexing_details = details
        # Mark the JSONB field as modified so SQLAlchemy detects the change
        flag_modified(doc, "indexing_details")
        self.db.commit()
        self.db.refresh(doc)

        logger.info(f"Updated metadata extraction status for {doc.filename}: {status}")
        return doc

    def init_graphrag_total_chunks(
        self,
        doc: Document,
        total_chunks: int,
    ) -> Document:
        """
        Initialize GraphRAG indexing with the total number of chunks to process.
        This should be called before processing starts.

        Args:
            doc: Document instance
            total_chunks: Total number of chunks that will be processed
        """
        if not doc.indexing_details:
            self.init_indexing_details(doc)

        details = doc.indexing_details
        graphrag = details["graphrag_indexing"]

        # Set the expected total chunks (not just the ones we've started processing)
        graphrag["expected_total_chunks"] = total_chunks
        graphrag["status"] = "processing"
        graphrag["started_at"] = datetime.utcnow().isoformat()

        doc.indexing_details = details
        flag_modified(doc, "indexing_details")
        self.db.commit()
        self.db.refresh(doc)

        return doc

    def update_graphrag_chunk_status(
        self,
        doc: Document,
        chunk_id: str,
        status: str,  # "success", "failed", "skipped"
        entities: int = 0,
        relationships: int = 0,
        error: str = None,
        page_number: int = None,
    ) -> Document:
        """
        Update GraphRAG indexing status for a single chunk.

        Args:
            doc: Document instance
            chunk_id: Chunk UUID
            status: "success", "failed", or "skipped"
            entities: Number of entities extracted
            relationships: Number of relationships extracted
            error: Error message if failed
            page_number: Page number (optional)
        """
        if not doc.indexing_details:
            self.init_indexing_details(doc)

        details = doc.indexing_details
        graphrag = details["graphrag_indexing"]

        # Update chunk status
        if chunk_id not in graphrag["chunks"]:
            graphrag["chunks"][chunk_id] = {}

        chunk_info = graphrag["chunks"][chunk_id]
        chunk_info["status"] = status

        if page_number is not None:
            chunk_info["page"] = page_number

        if status == "success":
            chunk_info["processed_at"] = datetime.utcnow().isoformat()
            chunk_info["entities"] = entities
            chunk_info["relationships"] = relationships
            chunk_info["error"] = None
            graphrag["entities_extracted"] = graphrag.get("entities_extracted", 0) + entities
            graphrag["relationships_extracted"] = graphrag.get("relationships_extracted", 0) + relationships
        elif status == "failed":
            chunk_info["error"] = error
            chunk_info["retry_count"] = chunk_info.get("retry_count", 0) + 1
        elif status == "skipped":
            chunk_info["skipped_at"] = datetime.utcnow().isoformat()

        # Update totals
        total_chunks = len(graphrag["chunks"])
        processed_chunks = sum(1 for c in graphrag["chunks"].values() if c.get("status") == "success")
        failed_chunks = sum(1 for c in graphrag["chunks"].values() if c.get("status") == "failed")
        skipped_chunks = sum(1 for c in graphrag["chunks"].values() if c.get("status") == "skipped")

        graphrag["total_chunks"] = total_chunks
        graphrag["processed_chunks"] = processed_chunks
        graphrag["failed_chunks"] = failed_chunks
        graphrag["skipped_chunks"] = skipped_chunks

        # Update overall status
        # Use expected_total_chunks if available, otherwise fall back to total_chunks
        expected_total = graphrag.get("expected_total_chunks", total_chunks)

        if failed_chunks > 0 and processed_chunks > 0:
            graphrag["status"] = "partial"
        elif failed_chunks > 0 and processed_chunks == 0:
            graphrag["status"] = "failed"
        elif processed_chunks + skipped_chunks >= expected_total and expected_total > 0:
            graphrag["status"] = "completed"
            graphrag["completed_at"] = datetime.utcnow().isoformat()
        elif processed_chunks > 0 or skipped_chunks > 0:
            graphrag["status"] = "processing"
        else:
            graphrag["status"] = "pending"

        doc.indexing_details = details
        # Mark the JSONB field as modified so SQLAlchemy detects the change
        flag_modified(doc, "indexing_details")
        self.db.commit()
        self.db.refresh(doc)

        return doc

    def get_failed_pages(self, doc: Document, phase: str = "vector") -> List[Dict[str, Any]]:
        """
        Get list of failed pages for a specific indexing phase.

        Args:
            doc: Document instance
            phase: "vector" or "graphrag"

        Returns:
            List of page info dictionaries with status, file_path, error, etc.
        """
        if not doc.indexing_details:
            return []

        if phase == "vector":
            pages = doc.indexing_details.get("vector_indexing", {}).get("pages", {})
            return [
                {"page_key": key, **info}
                for key, info in pages.items()
                if info.get("status") == "failed"
            ]

        return []

    def get_failed_chunk_ids(self, doc: Document, phase: str = "vector") -> List[str]:
        """
        Get list of failed chunk IDs for a specific indexing phase.

        Args:
            doc: Document instance
            phase: "vector" or "graphrag"

        Returns:
            List of chunk UUIDs that failed
        """
        if not doc.indexing_details:
            return []

        if phase == "vector":
            chunks = doc.indexing_details.get("vector_indexing", {}).get("chunks", {})
        elif phase == "graphrag":
            chunks = doc.indexing_details.get("graphrag_indexing", {}).get("chunks", {})
        else:
            return []

        return [
            chunk_id
            for chunk_id, info in chunks.items()
            if info.get("status") == "failed"
        ]

    def get_indexing_progress(self, doc: Document) -> Dict[str, Any]:
        """
        Get detailed indexing progress for all phases.

        Returns:
            Dictionary with progress info for vector, metadata, and GraphRAG phases
        """
        if not doc.indexing_details:
            return {
                "vector": {"status": "pending", "progress": 0},
                "metadata": {"status": "pending", "progress": 0},
                "graphrag": {"status": "pending", "progress": 0}
            }

        vector = doc.indexing_details.get("vector_indexing", {})
        metadata = doc.indexing_details.get("metadata_extraction", {})
        graphrag = doc.indexing_details.get("graphrag_indexing", {})

        return {
            "vector": {
                "status": vector.get("status", "pending"),
                "progress": int((vector.get("indexed_chunks", 0) / vector.get("total_chunks", 1)) * 100) if vector.get("total_chunks", 0) > 0 else 0,
                "total_chunks": vector.get("total_chunks", 0),
                "indexed_chunks": vector.get("indexed_chunks", 0),
                "failed_chunks": vector.get("failed_chunks", 0),
            },
            "metadata": {
                "status": metadata.get("status", "pending"),
                "progress": 100 if metadata.get("status") == "completed" else 0,
            },
            "graphrag": {
                "status": graphrag.get("status", "pending"),
                "progress": int((graphrag.get("processed_chunks", 0) / graphrag.get("total_chunks", 1)) * 100) if graphrag.get("total_chunks", 0) > 0 else 0,
                "total_chunks": graphrag.get("total_chunks", 0),
                "processed_chunks": graphrag.get("processed_chunks", 0),
                "failed_chunks": graphrag.get("failed_chunks", 0),
                "skipped_chunks": graphrag.get("skipped_chunks", 0),
                "entities_extracted": graphrag.get("entities_extracted", 0),
                "relationships_extracted": graphrag.get("relationships_extracted", 0),
            }
        }

    def _log_status(
        self,
        doc_id: UUID,
        status_type: str,
        old_status: Optional[str],
        new_status: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log a status change."""
        log = DocumentStatusLog(
            document_id=doc_id,
            status_type=status_type,
            old_status=old_status,
            new_status=new_status,
            message=message,
            details=details,
        )
        self.db.add(log)
        self.db.commit()

    # ========== OCR Status Tracking Methods ==========

    def init_ocr_details(self, doc: Document) -> Document:
        """Initialize ocr_details structure for a new document."""
        if not doc.ocr_details:
            doc.ocr_details = {
                "version": "1.0",
                "page_ocr": {
                    "status": "pending",
                    "total_pages": 0,
                    "converted_pages": 0,
                    "failed_pages": 0,
                    "pages": {}
                }
            }
            flag_modified(doc, "ocr_details")
            self.db.commit()
            self.db.refresh(doc)
        return doc

    def set_total_pages_for_ocr(self, doc: Document, total_pages: int) -> Document:
        """Set total pages count for OCR tracking."""
        if not doc.ocr_details:
            self.init_ocr_details(doc)

        doc.ocr_details["page_ocr"]["total_pages"] = total_pages
        doc.ocr_details["page_ocr"]["started_at"] = datetime.utcnow().isoformat()
        flag_modified(doc, "ocr_details")
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def update_page_ocr_status(
        self,
        doc: Document,
        page_number: int,
        page_file_path: str,
        status: str,  # "success" or "failed"
        error: str = None,
        embedded_images_count: int = 0,
    ) -> Document:
        """
        Update OCR status for a single page.

        Args:
            doc: Document instance
            page_number: Page number (0-indexed)
            page_file_path: Path to the _nohf.md file
            status: "success" or "failed"
            error: Error message if failed
            embedded_images_count: Number of embedded images in this page
        """
        if not doc.ocr_details:
            self.init_ocr_details(doc)

        details = doc.ocr_details
        page_ocr = details["page_ocr"]

        # Create page key
        page_key = f"page_{page_number}"

        # Initialize page entry if not exists
        if page_key not in page_ocr["pages"]:
            page_ocr["pages"][page_key] = {}

        page_info = page_ocr["pages"][page_key]

        # Track if this is a new success or new failure
        was_previously_failed = page_info.get("status") == "failed"
        was_previously_success = page_info.get("status") == "success"

        page_info["status"] = status
        page_info["page_number"] = page_number
        page_info["file_path"] = page_file_path
        page_info["embedded_images_count"] = embedded_images_count

        if status == "success":
            page_info["converted_at"] = datetime.utcnow().isoformat()
            page_info["error"] = None

            # Only increment if not previously successful
            if not was_previously_success:
                page_ocr["converted_pages"] = page_ocr.get("converted_pages", 0) + 1

            # Decrement failed count if this was previously failed
            if was_previously_failed:
                page_ocr["failed_pages"] = max(0, page_ocr.get("failed_pages", 0) - 1)
        else:
            page_info["failed_at"] = datetime.utcnow().isoformat()
            page_info["error"] = error
            page_info["retry_count"] = page_info.get("retry_count", 0) + 1

            # Only increment if not previously failed
            if not was_previously_failed:
                page_ocr["failed_pages"] = page_ocr.get("failed_pages", 0) + 1

            # Decrement success count if this was previously successful
            if was_previously_success:
                page_ocr["converted_pages"] = max(0, page_ocr.get("converted_pages", 0) - 1)

        # Update overall status
        total_pages = page_ocr.get("total_pages", 0)
        converted_pages = page_ocr.get("converted_pages", 0)
        failed_pages = page_ocr.get("failed_pages", 0)

        if failed_pages > 0 and converted_pages > 0:
            page_ocr["status"] = "partial"
        elif failed_pages > 0 and converted_pages == 0:
            page_ocr["status"] = "failed"
        elif converted_pages >= total_pages and total_pages > 0:
            page_ocr["status"] = "completed"
            page_ocr["completed_at"] = datetime.utcnow().isoformat()
        elif converted_pages > 0:
            page_ocr["status"] = "processing"

        doc.ocr_details = details
        flag_modified(doc, "ocr_details")
        self.db.commit()
        self.db.refresh(doc)

        return doc

    def update_embedded_image_ocr_status(
        self,
        doc: Document,
        page_number: int,
        image_position: int,
        status: str,  # "success", "failed", "skipped"
        ocr_backend: str = None,  # "gemma3" or "qwen3"
        error: str = None,
        image_size_pixels: int = None,
        skip_reason: str = None,
    ) -> Document:
        """
        Update OCR status for an embedded image within a page.

        Args:
            doc: Document instance
            page_number: Page number containing the image
            image_position: Position of image within the page (0-indexed)
            status: "success", "failed", or "skipped"
            ocr_backend: Backend used (gemma3/qwen3)
            error: Error message if failed
            image_size_pixels: Image size in pixels
            skip_reason: Reason if skipped (e.g., "Image too small")
        """
        if not doc.ocr_details:
            self.init_ocr_details(doc)

        details = doc.ocr_details
        page_key = f"page_{page_number}"

        # Ensure page exists
        if page_key not in details["page_ocr"]["pages"]:
            details["page_ocr"]["pages"][page_key] = {
                "page_number": page_number,
                "embedded_images": {}
            }

        page_info = details["page_ocr"]["pages"][page_key]

        # Initialize embedded_images dict if not exists
        if "embedded_images" not in page_info:
            page_info["embedded_images"] = {}

        # Create image key
        image_key = f"image_{image_position}"

        # Initialize image entry
        if image_key not in page_info["embedded_images"]:
            page_info["embedded_images"][image_key] = {}

        image_info = page_info["embedded_images"][image_key]
        image_info["status"] = status
        image_info["image_position"] = image_position

        if ocr_backend:
            image_info["ocr_backend"] = ocr_backend

        if image_size_pixels is not None:
            image_info["image_size_pixels"] = image_size_pixels

        if status == "success":
            image_info["converted_at"] = datetime.utcnow().isoformat()
            image_info["error"] = None
        elif status == "failed":
            image_info["failed_at"] = datetime.utcnow().isoformat()
            image_info["error"] = error
            image_info["retry_count"] = image_info.get("retry_count", 0) + 1
        elif status == "skipped":
            image_info["skipped_at"] = datetime.utcnow().isoformat()
            image_info["was_skipped"] = True
            image_info["skip_reason"] = skip_reason

        doc.ocr_details = details
        flag_modified(doc, "ocr_details")
        self.db.commit()
        self.db.refresh(doc)

        return doc

    def get_failed_pages(self, doc: Document) -> List[int]:
        """Get list of page numbers that failed OCR."""
        if not doc.ocr_details:
            return []

        pages = doc.ocr_details.get("page_ocr", {}).get("pages", {})
        failed_pages = []

        for page_key, page_info in pages.items():
            if page_info.get("status") == "failed":
                failed_pages.append(page_info.get("page_number"))

        return sorted(failed_pages)

    def get_pages_with_failed_embedded_images(self, doc: Document) -> Dict[int, List[int]]:
        """
        Get pages that have failed embedded image OCR.

        Returns:
            Dict mapping page_number -> list of failed image positions
        """
        if not doc.ocr_details:
            return {}

        pages = doc.ocr_details.get("page_ocr", {}).get("pages", {})
        failed_images_by_page = {}

        for page_key, page_info in pages.items():
            page_number = page_info.get("page_number")
            embedded_images = page_info.get("embedded_images", {})

            failed_images = []
            for image_key, image_info in embedded_images.items():
                if image_info.get("status") == "failed":
                    failed_images.append(image_info.get("image_position"))

            if failed_images:
                failed_images_by_page[page_number] = sorted(failed_images)

        return failed_images_by_page

    def get_ocr_summary(self, doc: Document) -> Dict[str, Any]:
        """
        Get a summary of OCR status for a document.

        Returns:
            Dict with overall status, page counts, and failure details
        """
        if not doc.ocr_details:
            return {
                "status": "not_started",
                "total_pages": 0,
                "converted_pages": 0,
                "failed_pages": 0,
                "failed_page_numbers": [],
                "pages_with_failed_images": {}
            }

        page_ocr = doc.ocr_details.get("page_ocr", {})

        return {
            "status": page_ocr.get("status", "unknown"),
            "total_pages": page_ocr.get("total_pages", 0),
            "converted_pages": page_ocr.get("converted_pages", 0),
            "failed_pages": page_ocr.get("failed_pages", 0),
            "failed_page_numbers": self.get_failed_pages(doc),
            "pages_with_failed_images": self.get_pages_with_failed_embedded_images(doc),
            "started_at": page_ocr.get("started_at"),
            "completed_at": page_ocr.get("completed_at")
        }

