"""
SQLAlchemy models for document management, user management, and chat sessions.
"""
import enum
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID

from sqlalchemy import Column, String, Integer, BigInteger, DateTime, Text, ForeignKey, Enum, Boolean
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


# ===== User Management Enums =====

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    USER = "user"


class UserStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


# ===== Document Management Enums =====

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


class TaskStatus(str, enum.Enum):
    """Task status enumeration for hierarchical task queue phases"""
    PENDING = "pending"      # Waiting to be picked up by a worker
    PROCESSING = "processing"  # Worker is actively processing
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"        # Failed after max retries


# ===== User Management Models =====

class User(Base):
    """User model for authentication and authorization."""
    __tablename__ = "users"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    username = Column(String(100), nullable=False, unique=True, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(Enum(UserRole, values_callable=lambda x: [e.value for e in x]),
                  nullable=False, server_default='user')
    status = Column(Enum(UserStatus, values_callable=lambda x: [e.value for e in x]),
                    nullable=False, server_default='active')

    # Profile information
    profile_data = Column(JSONB, default=dict)

    # Authentication tracking
    last_login_at = Column(DateTime(timezone=True))
    last_login_ip = Column(String(45))
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True))

    # Password management
    password_changed_at = Column(DateTime(timezone=True), server_default=func.now())
    password_reset_token = Column(String(255))
    password_reset_expires = Column(DateTime(timezone=True))

    # Email verification
    email_verified = Column(Boolean, default=False)
    email_verification_token = Column(String(255))
    email_verification_expires = Column(DateTime(timezone=True))

    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"))
    updated_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"))
    deleted_at = Column(DateTime(timezone=True))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (exclude password_hash)."""
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role.value if self.role else None,
            "status": self.status.value if self.status else None,
            "profile_data": self.profile_data,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "email_verified": self.email_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class RefreshToken(Base):
    """Refresh token model for JWT authentication."""
    __tablename__ = "refresh_tokens"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token = Column(String(500), nullable=False, unique=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    revoked_at = Column(DateTime(timezone=True))
    replaced_by_token = Column(String(500))
    user_agent = Column(Text)
    ip_address = Column(String(45))


# ===== Chat Session Models =====

class ChatSession(Base):
    """Chat session model for conversation history."""
    __tablename__ = "chat_sessions"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_name = Column(String(500))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_message_at = Column(DateTime(timezone=True))
    message_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    session_metadata = Column(JSONB, default=dict)
    deleted_at = Column(DateTime(timezone=True))

    # Relationships
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    summaries = relationship("ChatSessionSummary", back_populates="session", cascade="all, delete-orphan")

    def to_dict(self, message_count: Optional[int] = None) -> Dict[str, Any]:
        """Convert to dictionary.

        Args:
            message_count: If provided, use this as the message count (computed at runtime).
                          If None, falls back to the stored message_count column.
        """
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "session_name": self.session_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "message_count": message_count if message_count is not None else self.message_count,
            "is_active": self.is_active,
            "session_metadata": self.session_metadata,
        }


class ChatMessage(Base):
    """Chat message model for individual messages within sessions."""
    __tablename__ = "chat_messages"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    session_id = Column(PGUUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    message_metadata = Column(JSONB, default=dict)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "message_metadata": self.message_metadata,
        }


class ChatSessionSummary(Base):
    """Chat session summary model for hierarchical summaries."""
    __tablename__ = "chat_session_summaries"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    session_id = Column(PGUUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    summary_type = Column(String(50), nullable=False)
    summary_content = Column(Text, nullable=False)
    message_range_start = Column(PGUUID(as_uuid=True))
    message_range_end = Column(PGUUID(as_uuid=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    summary_metadata = Column(JSONB, default=dict)

    # Relationships
    session = relationship("ChatSession", back_populates="summaries")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "summary_type": self.summary_type,
            "summary_content": self.summary_content,
            "message_range_start": str(self.message_range_start) if self.message_range_start else None,
            "message_range_end": str(self.message_range_end) if self.message_range_end else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "summary_metadata": self.summary_metadata,
        }


# ===== Document Management Models =====


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
    # Granular OCR status tracking for selective re-processing (page-level, embedded images)
    ocr_details = Column(JSONB, nullable=True, default=None)
    # Granular indexing status tracking for selective re-indexing (vector, metadata, GraphRAG)
    indexing_details = Column(JSONB, nullable=True, default=None)
    # Use values_callable to ensure SQLAlchemy uses enum values (lowercase) not names (uppercase)
    upload_status = Column(Enum(UploadStatus, name="upload_status", values_callable=lambda x: [e.value for e in x]), default=UploadStatus.PENDING)
    convert_status = Column(Enum(ConvertStatus, name="convert_status", values_callable=lambda x: [e.value for e in x]), default=ConvertStatus.PENDING)
    index_status = Column(Enum(IndexStatus, name="index_status", values_callable=lambda x: [e.value for e in x]), default=IndexStatus.PENDING)

    # Hierarchical task queue status columns (added by migration 012)
    # Phase 1: OCR status
    ocr_status = Column(Enum(TaskStatus, name="task_status", create_type=False, values_callable=lambda x: [e.value for e in x]), default=TaskStatus.PENDING)
    ocr_started_at = Column(DateTime(timezone=True), nullable=True)
    ocr_completed_at = Column(DateTime(timezone=True), nullable=True)
    ocr_error = Column(Text, nullable=True)
    # Phase 2: Vector indexing status
    vector_status = Column(Enum(TaskStatus, name="task_status", create_type=False, values_callable=lambda x: [e.value for e in x]), default=TaskStatus.PENDING)
    vector_started_at = Column(DateTime(timezone=True), nullable=True)
    vector_completed_at = Column(DateTime(timezone=True), nullable=True)
    vector_error = Column(Text, nullable=True)
    # Phase 3: GraphRAG indexing status
    graphrag_status = Column(Enum(TaskStatus, name="task_status", create_type=False, values_callable=lambda x: [e.value for e in x]), default=TaskStatus.PENDING)
    graphrag_started_at = Column(DateTime(timezone=True), nullable=True)
    graphrag_completed_at = Column(DateTime(timezone=True), nullable=True)
    graphrag_error = Column(Text, nullable=True)
    # Worker tracking at document level
    current_worker_id = Column(String(100), nullable=True)
    last_heartbeat = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # Audit fields
    created_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"))
    updated_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id"))

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
            "document_metadata": self.document_metadata,
            "ocr_details": self.ocr_details,
            "indexing_details": self.indexing_details,
            "upload_status": self.upload_status.value if self.upload_status else None,
            "convert_status": self.convert_status.value if self.convert_status else None,
            "index_status": self.index_status.value if self.index_status else None,
            # Hierarchical task queue status fields
            "ocr_status": self.ocr_status.value if self.ocr_status else None,
            "vector_status": self.vector_status.value if self.vector_status else None,
            "graphrag_status": self.graphrag_status.value if self.graphrag_status else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def get_indexing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of indexing progress across all phases.

        Returns:
            Dictionary with progress percentages and status for each phase
        """
        if not self.indexing_details:
            return {
                "vector_progress": 0,
                "metadata_progress": 0,
                "graphrag_progress": 0,
                "overall_status": self.index_status.value if self.index_status else "pending"
            }

        vector = self.indexing_details.get("vector_indexing", {})
        metadata = self.indexing_details.get("metadata_extraction", {})
        graphrag = self.indexing_details.get("graphrag_indexing", {})

        # Calculate vector progress
        vector_total = vector.get("total_chunks", 0)
        vector_indexed = vector.get("indexed_chunks", 0)
        vector_progress = int((vector_indexed / vector_total * 100)) if vector_total > 0 else 0

        # Metadata is binary (done or not)
        metadata_progress = 100 if metadata.get("status") == "completed" else 0

        # Calculate GraphRAG progress
        graphrag_total = graphrag.get("total_chunks", 0)
        graphrag_processed = graphrag.get("processed_chunks", 0)
        graphrag_progress = int((graphrag_processed / graphrag_total * 100)) if graphrag_total > 0 else 0

        return {
            "vector_progress": vector_progress,
            "vector_status": vector.get("status", "pending"),
            "metadata_progress": metadata_progress,
            "metadata_status": metadata.get("status", "pending"),
            "graphrag_progress": graphrag_progress,
            "graphrag_status": graphrag.get("status", "pending"),
            "overall_status": self.index_status.value if self.index_status else "pending"
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

