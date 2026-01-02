"""
SQLAlchemy models for document management, user management, and chat sessions.
"""
import enum
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID

from sqlalchemy import Column, String, Integer, BigInteger, DateTime, Text, ForeignKey, Enum, Boolean, ARRAY
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from sqlalchemy.orm import declarative_base, relationship, backref
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


# ===== Document Permission Enums =====

class DocumentPermission(str, enum.Enum):
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    SHARE = "share"
    FULL = "full"


class PermissionOrigin(str, enum.Enum):
    OWNER = "owner"
    SHARED = "shared"
    ADMIN_GRANTED = "admin_granted"
    PUBLIC = "public"


class DocumentVisibility(str, enum.Enum):
    PRIVATE = "private"
    SHARED = "shared"
    PUBLIC = "public"


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
    SKIPPED = "skipped"      # Intentionally skipped (e.g., tabular data for GraphRAG)


# Statuses that count as "done" for status rollup calculations
# Both COMPLETED and SKIPPED are considered successful completion
DONE_STATUSES = {TaskStatus.COMPLETED, TaskStatus.SKIPPED}


# ===== User Management Models =====

class User(Base):
    """User model for authentication and authorization."""
    __tablename__ = "users"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    username = Column(String(100), nullable=False, unique=True, index=True)
    normalized_username = Column(String(100), nullable=False, unique=True, index=True)  # Sanitized for folder paths
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(Enum(UserRole, values_callable=lambda x: [e.value for e in x]),
                  nullable=False, server_default='user')
    status = Column(Enum(UserStatus, values_callable=lambda x: [e.value for e in x]),
                    nullable=False, server_default='active')

    # Profile information
    profile_data = Column(JSONB, default=dict)

    # User preferences (chat settings, UI preferences, etc.)
    preferences = Column(JSONB, default=dict)

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
            "normalized_username": self.normalized_username,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role.value if self.role else None,
            "status": self.status.value if self.status else None,
            "profile_data": self.profile_data,
            "preferences": self.preferences,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "email_verified": self.email_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ===== Workspace Management Models =====

class Workspace(Base):
    """Workspace model for organizing user documents into folders."""
    __tablename__ = "workspaces"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())

    # Owner reference
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Workspace identity
    name = Column(String(100), nullable=False)  # Display name: "Project Alpha"
    folder_name = Column(String(50), nullable=False)  # Physical folder name: "project_alpha"
    folder_path = Column(String(200), nullable=False)  # Full relative path: "john_doe/project_alpha"

    # Metadata
    description = Column(Text, nullable=True)
    color = Column(String(7), default='#6366f1')  # Hex color for UI
    icon = Column(String(50), default='folder')  # Icon identifier

    # Flags
    is_default = Column(Boolean, default=False)  # User's default workspace
    is_system = Column(Boolean, default=False)  # System workspace (e.g., "Shared With Me")

    # Cached stats
    document_count = Column(Integer, default=0)  # Cached for performance

    # Display
    display_order = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", backref="workspaces")
    documents = relationship("Document", back_populates="workspace")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "name": self.name,
            "folder_name": self.folder_name,
            "folder_path": self.folder_path,
            "description": self.description,
            "color": self.color,
            "icon": self.icon,
            "is_default": self.is_default,
            "is_system": self.is_system,
            "document_count": self.document_count,
            "display_order": self.display_order,
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

    # Workspace and ownership (added for access control)
    workspace_id = Column(PGUUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True)
    owner_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    visibility = Column(String(20), default='private')  # private, shared, public
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

    # GraphRAG skip control (for tabular/structured data that doesn't benefit from entity extraction)
    skip_graphrag = Column(Boolean, default=False, nullable=False)
    skip_graphrag_reason = Column(String(100), nullable=True)  # e.g., "file_type:.xlsx", "document_type:invoice"

    # ==========================================
    # Tabular Data Processing (Optimized Pathway)
    # ==========================================
    # Flag to identify tabular/dataset-style documents (CSV, Excel, invoices, receipts, etc.)
    is_tabular_data = Column(Boolean, default=False, nullable=False)

    # Processing path indicator: "standard" (full chunking), "tabular" (summary-only), "hybrid"
    processing_path = Column(String(50), default="standard", nullable=False)

    # Track summary chunk IDs for tabular documents (1-3 chunks instead of many row chunks)
    # Example: ["doc-uuid_summary", "doc-uuid_schema", "doc-uuid_context"]
    summary_chunk_ids = Column(ARRAY(String), nullable=True)

    # Structured Data Extraction status (for extracting tabular data from spreadsheets, invoices, etc.)
    extraction_eligible = Column(Boolean, nullable=True)
    extraction_status = Column(String(20), default='pending')  # pending, processing, completed, failed, skipped
    extraction_schema_type = Column(String(64), nullable=True)  # invoice, receipt, spreadsheet, etc.
    extraction_started_at = Column(DateTime(timezone=True), nullable=True)
    extraction_completed_at = Column(DateTime(timezone=True), nullable=True)
    extraction_error = Column(Text, nullable=True)

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

    # Workspace and owner relationships
    workspace = relationship("Workspace", back_populates="documents")
    owner = relationship("User", foreign_keys=[owner_id], backref="owned_documents")
    user_permissions = relationship("UserDocument", back_populates="document", cascade="all, delete-orphan")

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
            # Workspace and ownership
            "workspace_id": str(self.workspace_id) if self.workspace_id else None,
            "owner_id": str(self.owner_id) if self.owner_id else None,
            "visibility": self.visibility,
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
            # Tabular data processing fields
            "is_tabular_data": self.is_tabular_data,
            "processing_path": self.processing_path,
            "summary_chunk_ids": self.summary_chunk_ids,
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


# ===== Document Permissions Models =====

class UserDocument(Base):
    """Bridge table for user document permissions and access control."""
    __tablename__ = "user_documents"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())

    # Core references
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)

    # Permission details - using PostgreSQL enum types
    permissions = Column(
        ARRAY(Enum(DocumentPermission, name="document_permission", create_type=False, values_callable=lambda x: [e.value for e in x])),
        nullable=False,
        default=['read']
    )
    origin = Column(
        Enum(PermissionOrigin, name="permission_origin", create_type=False, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default='shared'
    )
    is_owner = Column(Boolean, nullable=False, default=False)

    # Sharing metadata
    shared_by = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    shared_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    share_message = Column(Text, nullable=True)

    # Notification
    is_new = Column(Boolean, default=True)
    viewed_at = Column(DateTime(timezone=True), nullable=True)

    # Access tracking
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    access_count = Column(Integer, default=0)

    # Audit
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", foreign_keys=[user_id], backref="document_permissions")
    document = relationship("Document", back_populates="user_permissions")
    shared_by_user = relationship("User", foreign_keys=[shared_by])

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        if self.is_owner or 'full' in self.permissions:
            return True
        return permission in self.permissions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "document_id": str(self.document_id),
            "permissions": self.permissions,
            "origin": self.origin,
            "is_owner": self.is_owner,
            "shared_by": str(self.shared_by) if self.shared_by else None,
            "shared_at": self.shared_at.isoformat() if self.shared_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "share_message": self.share_message,
            "is_new": self.is_new,
            "viewed_at": self.viewed_at.isoformat() if self.viewed_at else None,
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ===== Data Extraction Enums =====

class ExtractionStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ValidationStatus(str, enum.Enum):
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    PARTIAL = "partial"


class AnalyticsState(str, enum.Enum):
    INITIAL = "INITIAL"
    CLASSIFYING = "CLASSIFYING"
    QUESTIONING = "QUESTIONING"
    PLANNING = "PLANNING"
    REVIEWING = "REVIEWING"
    REFINING = "REFINING"
    EXECUTING = "EXECUTING"
    COMPLETE = "COMPLETE"
    FOLLOW_UP = "FOLLOW_UP"
    ERROR = "ERROR"
    EXPIRED = "EXPIRED"


# ===== Data Extraction Models =====

class DataSchema(Base):
    """Schema definitions for structured data extraction."""
    __tablename__ = "data_schemas"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    schema_type = Column(String(64), nullable=False, unique=True)
    schema_version = Column(String(16), default='1.0')
    domain = Column(String(32), nullable=False)
    display_name = Column(String(128))
    description = Column(Text)

    # Schema definitions (JSON Schema format)
    header_schema = Column(JSONB, nullable=False, default=dict)
    line_items_schema = Column(JSONB)
    summary_schema = Column(JSONB)

    # Extraction configuration
    extraction_prompt = Column(Text)
    field_mappings = Column(JSONB, default=dict)

    # Validation rules
    validation_rules = Column(JSONB, default=dict)

    # Metadata
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "schema_type": self.schema_type,
            "schema_version": self.schema_version,
            "domain": self.domain,
            "display_name": self.display_name,
            "description": self.description,
            "header_schema": self.header_schema,
            "line_items_schema": self.line_items_schema,
            "summary_schema": self.summary_schema,
            "is_active": self.is_active,
        }


class DocumentData(Base):
    """Extracted structured data from documents."""
    __tablename__ = "documents_data"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    document_id = Column(PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, unique=True)

    # Schema Reference
    schema_type = Column(String(64), nullable=False)
    schema_version = Column(String(16), default='1.0')

    # Extracted Data
    header_data = Column(JSONB, nullable=False, default=dict)
    summary_data = Column(JSONB, default=dict)

    # Line items count (actual data stored in documents_data_line_items table)
    # NOTE: line_items and line_items_storage columns removed in migration 022
    # All line items now stored externally in documents_data_line_items table
    line_items_count = Column(Integer, default=0)

    # Validation & Quality
    validation_status = Column(String(20), default='pending')
    overall_confidence = Column(String(10))  # Store as string to avoid decimal issues
    field_confidences = Column(JSONB, default=dict)
    validation_results = Column(JSONB)

    # Extraction Metadata
    extraction_method = Column(String(32))
    extraction_model = Column(String(64))
    extraction_duration_ms = Column(Integer)
    extraction_metadata = Column(JSONB, default=dict)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships - use passive_deletes=True to let database handle CASCADE
    document = relationship("Document", backref=backref("extracted_data", passive_deletes=True))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "schema_type": self.schema_type,
            "schema_version": self.schema_version,
            "header_data": self.header_data,
            "line_items": f"[{self.line_items_count} items in external storage]",
            "summary_data": self.summary_data,
            "line_items_count": self.line_items_count,
            "validation_status": self.validation_status,
            "overall_confidence": self.overall_confidence,
            "extraction_method": self.extraction_method,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DocumentDataLineItem(Base):
    """Overflow storage for large table line items."""
    __tablename__ = "documents_data_line_items"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    documents_data_id = Column(PGUUID(as_uuid=True), ForeignKey("documents_data.id", ondelete="CASCADE"), nullable=False)
    line_number = Column(Integer, nullable=False)
    data = Column(JSONB, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships - use passive_deletes=True to let database handle CASCADE
    documents_data = relationship("DocumentData", backref=backref("overflow_line_items", passive_deletes=True))


class AnalyticsSession(Base):
    """Analytics conversation session for interactive querying."""
    __tablename__ = "analytics_sessions"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())

    # Links to existing infrastructure
    chat_session_id = Column(PGUUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="SET NULL"))
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    workspace_id = Column(PGUUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)

    # State Machine
    state = Column(String(32), nullable=False, default='INITIAL')
    state_entered_at = Column(DateTime(timezone=True), server_default=func.now())

    # Query Context
    original_query = Column(Text, nullable=False)
    intent_classification = Column(JSONB)

    # Gathered Information
    gathered_info = Column(JSONB, default=dict)

    # Plan Management
    current_plan = Column(JSONB)
    plan_version = Column(Integer, default=0)
    plan_history = Column(JSONB, default=list)

    # Execution State
    execution_progress = Column(JSONB)

    # Results Cache
    cached_results = Column(JSONB)
    result_generated_at = Column(DateTime(timezone=True))

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True))

    # Relationships
    chat_session = relationship("ChatSession", backref="analytics_sessions")
    user = relationship("User", backref="analytics_sessions")
    workspace = relationship("Workspace", backref="analytics_sessions")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "chat_session_id": str(self.chat_session_id) if self.chat_session_id else None,
            "user_id": str(self.user_id),
            "workspace_id": str(self.workspace_id),
            "state": self.state,
            "state_entered_at": self.state_entered_at.isoformat() if self.state_entered_at else None,
            "original_query": self.original_query,
            "intent_classification": self.intent_classification,
            "gathered_info": self.gathered_info,
            "current_plan": self.current_plan,
            "plan_version": self.plan_version,
            "execution_progress": self.execution_progress,
            "cached_results": self.cached_results,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }
