"""
Hierarchical Task Queue Models

Defines the task queue models for the 3-level hierarchy:
- Document level: uses documents table with ocr_status, vector_status, graphrag_status
- Page level: task_queue_page table
- Chunk level: task_queue_chunk table

Each level has 3 processing phases: OCR → Vector Index → GraphRAG Index
"""

from typing import Dict, Any
from sqlalchemy import Column, String, Integer, Text, ForeignKey, Enum as SQLEnum, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID, TIMESTAMP
from sqlalchemy.orm import relationship
from db.models import Base, TaskStatus


class TaskQueuePage(Base):
    """
    Page-level task queue for OCR, Vector indexing, and GraphRAG indexing.

    Each page has 3 status columns - one for each processing phase.
    Sequential dependencies:
    - vector_status can only start when ocr_status = 'completed'
    - graphrag_status can only start when vector_status = 'completed'
    """
    __tablename__ = "task_queue_page"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    document_id = Column(PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)

    # Page identification
    page_number = Column(Integer, nullable=False)
    page_file_path = Column(String(1000), nullable=True)  # Path to _nohf.md file after OCR

    # === Phase 1: OCR (PDF page → Markdown) ===
    ocr_status = Column(
        SQLEnum(TaskStatus, name="task_status", create_type=False, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=TaskStatus.PENDING
    )
    ocr_worker_id = Column(String(100), nullable=True)
    ocr_started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    ocr_completed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    ocr_error = Column(Text, nullable=True)
    ocr_retry_count = Column(Integer, default=0)
    ocr_last_heartbeat = Column(TIMESTAMP(timezone=True), nullable=True)

    # === Phase 2: Vector Index (Markdown → Qdrant) ===
    vector_status = Column(
        SQLEnum(TaskStatus, name="task_status", create_type=False, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=TaskStatus.PENDING
    )
    vector_worker_id = Column(String(100), nullable=True)
    vector_started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    vector_completed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    vector_error = Column(Text, nullable=True)
    vector_retry_count = Column(Integer, default=0)
    vector_last_heartbeat = Column(TIMESTAMP(timezone=True), nullable=True)

    # === Phase 3: GraphRAG Index (Chunks → Neo4j) ===
    graphrag_status = Column(
        SQLEnum(TaskStatus, name="task_status", create_type=False, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=TaskStatus.PENDING
    )
    graphrag_worker_id = Column(String(100), nullable=True)
    graphrag_started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    graphrag_completed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    graphrag_error = Column(Text, nullable=True)
    graphrag_retry_count = Column(Integer, default=0)
    graphrag_last_heartbeat = Column(TIMESTAMP(timezone=True), nullable=True)

    # Common metadata
    chunk_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    chunks = relationship("TaskQueueChunk", back_populates="page", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "page_number": self.page_number,
            "page_file_path": self.page_file_path,
            # OCR phase
            "ocr_status": self.ocr_status.value if self.ocr_status else None,
            "ocr_worker_id": self.ocr_worker_id,
            "ocr_started_at": self.ocr_started_at.isoformat() if self.ocr_started_at else None,
            "ocr_completed_at": self.ocr_completed_at.isoformat() if self.ocr_completed_at else None,
            "ocr_error": self.ocr_error,
            "ocr_retry_count": self.ocr_retry_count,
            # Vector phase
            "vector_status": self.vector_status.value if self.vector_status else None,
            "vector_worker_id": self.vector_worker_id,
            "vector_started_at": self.vector_started_at.isoformat() if self.vector_started_at else None,
            "vector_completed_at": self.vector_completed_at.isoformat() if self.vector_completed_at else None,
            "vector_error": self.vector_error,
            "vector_retry_count": self.vector_retry_count,
            # GraphRAG phase
            "graphrag_status": self.graphrag_status.value if self.graphrag_status else None,
            "graphrag_worker_id": self.graphrag_worker_id,
            "graphrag_started_at": self.graphrag_started_at.isoformat() if self.graphrag_started_at else None,
            "graphrag_completed_at": self.graphrag_completed_at.isoformat() if self.graphrag_completed_at else None,
            "graphrag_error": self.graphrag_error,
            "graphrag_retry_count": self.graphrag_retry_count,
            # Metadata
            "chunk_count": self.chunk_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class TaskQueueChunk(Base):
    """
    Chunk-level task queue for Vector indexing and GraphRAG indexing.

    Chunks only need 2 status columns (no OCR at chunk level).
    Sequential dependencies:
    - graphrag_status can only start when vector_status = 'completed'
    """
    __tablename__ = "task_queue_chunk"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    page_id = Column(PGUUID(as_uuid=True), ForeignKey("task_queue_page.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)

    # Chunk identification
    chunk_id = Column(String(64), nullable=False)  # External chunk ID (used in Qdrant)
    chunk_index = Column(Integer, nullable=False)   # Position within page (0-indexed)
    chunk_content_hash = Column(String(64), nullable=True)  # For deduplication

    # === Phase 2: Vector Index (Chunk → Qdrant embedding) ===
    vector_status = Column(
        SQLEnum(TaskStatus, name="task_status", create_type=False, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=TaskStatus.PENDING
    )
    vector_worker_id = Column(String(100), nullable=True)
    vector_started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    vector_completed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    vector_error = Column(Text, nullable=True)
    vector_retry_count = Column(Integer, default=0)
    vector_last_heartbeat = Column(TIMESTAMP(timezone=True), nullable=True)

    # === Phase 3: GraphRAG Index (Chunk → Neo4j entities) ===
    graphrag_status = Column(
        SQLEnum(TaskStatus, name="task_status", create_type=False, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=TaskStatus.PENDING
    )
    graphrag_worker_id = Column(String(100), nullable=True)
    graphrag_started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    graphrag_completed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    graphrag_error = Column(Text, nullable=True)
    graphrag_retry_count = Column(Integer, default=0)
    graphrag_last_heartbeat = Column(TIMESTAMP(timezone=True), nullable=True)
    graphrag_skip_reason = Column(String(50), nullable=True)  # Reason for skipping GraphRAG

    # Processing results
    entities_extracted = Column(Integer, default=0)
    relationships_extracted = Column(Integer, default=0)

    # Common metadata
    max_retries = Column(Integer, default=3)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    page = relationship("TaskQueuePage", back_populates="chunks")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "page_id": str(self.page_id),
            "document_id": str(self.document_id),
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "chunk_content_hash": self.chunk_content_hash,
            # Vector phase
            "vector_status": self.vector_status.value if self.vector_status else None,
            "vector_worker_id": self.vector_worker_id,
            "vector_started_at": self.vector_started_at.isoformat() if self.vector_started_at else None,
            "vector_completed_at": self.vector_completed_at.isoformat() if self.vector_completed_at else None,
            "vector_error": self.vector_error,
            "vector_retry_count": self.vector_retry_count,
            # GraphRAG phase
            "graphrag_status": self.graphrag_status.value if self.graphrag_status else None,
            "graphrag_worker_id": self.graphrag_worker_id,
            "graphrag_started_at": self.graphrag_started_at.isoformat() if self.graphrag_started_at else None,
            "graphrag_completed_at": self.graphrag_completed_at.isoformat() if self.graphrag_completed_at else None,
            "graphrag_error": self.graphrag_error,
            "graphrag_retry_count": self.graphrag_retry_count,
            # Results
            "entities_extracted": self.entities_extracted,
            "relationships_extracted": self.relationships_extracted,
            # Metadata
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
