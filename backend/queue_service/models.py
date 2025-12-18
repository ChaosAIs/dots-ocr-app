"""
Task Queue Models

Defines the TaskQueue model and related enums for task coordination.
"""

from enum import Enum
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Integer, Text, ForeignKey, Enum as SQLEnum, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID, TIMESTAMP
from db.models import Base


class TaskType(str, Enum):
    """Task type enumeration"""
    OCR = "OCR"
    INDEXING = "INDEXING"


class TaskPriority(str, Enum):
    """Task priority enumeration"""
    HIGH = "HIGH"      # User uploads - process immediately
    NORMAL = "NORMAL"  # Auto-resume, scheduled tasks
    LOW = "LOW"        # Background cleanup


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "PENDING"      # Waiting to be claimed
    CLAIMED = "CLAIMED"      # Being processed by a worker
    COMPLETED = "COMPLETED"  # Successfully completed
    FAILED = "FAILED"        # Failed after max retries
    CANCELLED = "CANCELLED"  # Manually cancelled


class TaskQueue(Base):
    """
    Task Queue model for worker coordination.
    
    This table tracks which tasks need to be processed and which worker is processing them.
    Progress tracking is stored in documents.ocr_details and documents.indexing_details.
    """
    __tablename__ = "task_queue"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    document_id = Column(PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    task_type = Column(SQLEnum(TaskType, name="task_type_enum", create_type=False), nullable=False)
    priority = Column(SQLEnum(TaskPriority, name="task_priority_enum", create_type=False), nullable=False, default=TaskPriority.NORMAL)
    status = Column(SQLEnum(TaskStatus, name="task_status_enum", create_type=False), nullable=False, default=TaskStatus.PENDING)
    
    # Worker coordination
    worker_id = Column(String(100), nullable=True)
    claimed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    last_heartbeat = Column(TIMESTAMP(timezone=True), nullable=True)
    
    # Retry logic
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    completed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "task_type": self.task_type.value if self.task_type else None,
            "priority": self.priority.value if self.priority else None,
            "status": self.status.value if self.status else None,
            "worker_id": self.worker_id,
            "claimed_at": self.claimed_at.isoformat() if self.claimed_at else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

