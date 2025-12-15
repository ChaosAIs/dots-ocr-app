"""
Document repository for CRUD operations.
"""
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy.orm import Session
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

