"""
SQLAlchemy models for document management.
"""
import enum
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID

from sqlalchemy import Column, String, Integer, BigInteger, DateTime, Text, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class UploadStatus(str, enum.Enum):
    PENDING = "pending"
    UPLOADED = "uploaded"
    FAILED = "failed"


class ConvertStatus(str, enum.Enum):
    PENDING = "pending"
    CONVERTING = "converting"
    CONVERTED = "converted"
    PARTIAL = "partial"
    FAILED = "failed"


class IndexStatus(str, enum.Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    INDEXED = "indexed"
    PARTIAL = "partial"  # Some pages indexed, some pending
    FAILED = "failed"


class Document(Base):
    """Document model for file management."""
    __tablename__ = "documents"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    filename = Column(String(500), nullable=False, unique=True)
    original_filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    output_path = Column(String(1000), nullable=True)
    file_size = Column(BigInteger, nullable=False)
    mime_type = Column(String(100), nullable=True)
    total_pages = Column(Integer, default=0)
    converted_pages = Column(Integer, default=0)
    indexed_chunks = Column(Integer, default=0)
    # Page-level index tracking: {"indexed": [1,2,3], "pending": [4,5], "failed": []}
    indexed_pages_info = Column(JSONB, nullable=True, default=lambda: {"indexed": [], "pending": [], "failed": []})
    # Document metadata from hierarchical summarization: type, subject, topics, entities, summary
    document_metadata = Column(JSONB, nullable=True, default=None)
    # Use values_callable to ensure SQLAlchemy uses enum values (lowercase) not names (uppercase)
    upload_status = Column(Enum(UploadStatus, name="upload_status", values_callable=lambda x: [e.value for e in x]), default=UploadStatus.PENDING)
    convert_status = Column(Enum(ConvertStatus, name="convert_status", values_callable=lambda x: [e.value for e in x]), default=ConvertStatus.PENDING)
    index_status = Column(Enum(IndexStatus, name="index_status", values_callable=lambda x: [e.value for e in x]), default=IndexStatus.PENDING)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # Relationship to status logs
    status_logs = relationship("DocumentStatusLog", back_populates="document", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_path": self.file_path,
            "output_path": self.output_path,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "total_pages": self.total_pages,
            "converted_pages": self.converted_pages,
            "indexed_chunks": self.indexed_chunks,
            "indexed_pages_info": self.indexed_pages_info,
            "upload_status": self.upload_status.value if self.upload_status else None,
            "convert_status": self.convert_status.value if self.convert_status else None,
            "index_status": self.index_status.value if self.index_status else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DocumentStatusLog(Base):
    """Status log for document audit trail."""
    __tablename__ = "document_status_log"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    document_id = Column(PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    status_type = Column(String(50), nullable=False)
    old_status = Column(String(50), nullable=True)
    new_status = Column(String(50), nullable=False)
    message = Column(Text, nullable=True)
    details = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship to document
    document = relationship("Document", back_populates="status_logs")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "status_type": self.status_type,
            "old_status": self.old_status,
            "new_status": self.new_status,
            "message": self.message,
            "details": self.details,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

