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
from sqlalchemy.dialects.postgresql import UUID as PGUUID, TIMESTAMP, JSONB
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

    # V3.0: Chunk content storage for optimization
    # Stores chunk content after Stage 1 (OCR) to avoid redundant LLM calls in Stages 2 & 3
    chunk_content = Column(Text, nullable=True)  # Actual chunk text content
    chunk_metadata = Column(JSONB, nullable=True)  # Chunk metadata as JSON

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
            # V3.0: Chunk content (not included by default to avoid large payloads)
            "has_chunk_content": self.chunk_content is not None,
            "chunk_metadata": self.chunk_metadata,
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


class TaskQueueDocument(Base):
    """
    Document-level task queue for classification, extraction, and processing path routing.

    This table coordinates document-level tasks that happen BEFORE chunk-level processing:
    - Convert: OCR/document conversion coordination
    - Classification: Document type detection and metadata extraction
    - Extraction: Tabular data extraction (for tabular documents only)

    Processing path determines subsequent workflow:
    - 'standard': Document goes through chunking → vector indexing → GraphRAG
    - 'tabular': Document goes through data extraction → summary chunks → vector indexing
    """
    __tablename__ = "task_queue_document"

    id = Column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    document_id = Column(PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, unique=True)

    # === Phase 1: Convert (OCR/Document Conversion) ===
    convert_status = Column(
        SQLEnum(TaskStatus, name="task_status", create_type=False, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=TaskStatus.PENDING
    )
    convert_worker_id = Column(String(100), nullable=True)
    convert_started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    convert_completed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    convert_error = Column(Text, nullable=True)
    convert_retry_count = Column(Integer, default=0)
    convert_last_heartbeat = Column(TIMESTAMP(timezone=True), nullable=True)

    # === Phase 2: Classification & Metadata Extraction ===
    classification_status = Column(
        SQLEnum(TaskStatus, name="task_status", create_type=False, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=TaskStatus.PENDING
    )
    classification_worker_id = Column(String(100), nullable=True)
    classification_started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    classification_completed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    classification_error = Column(Text, nullable=True)
    classification_retry_count = Column(Integer, default=0)
    classification_last_heartbeat = Column(TIMESTAMP(timezone=True), nullable=True)

    # === Phase 3: Data Extraction (Tabular Documents Only) ===
    extraction_status = Column(
        SQLEnum(TaskStatus, name="task_status", create_type=False, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=TaskStatus.PENDING
    )
    extraction_worker_id = Column(String(100), nullable=True)
    extraction_started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    extraction_completed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    extraction_error = Column(Text, nullable=True)
    extraction_retry_count = Column(Integer, default=0)
    extraction_last_heartbeat = Column(TIMESTAMP(timezone=True), nullable=True)

    # Routing decision after classification
    processing_path = Column(String(20), default='standard')  # 'standard' or 'tabular'

    # Common metadata
    max_retries = Column(Integer, default=3)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            # Convert phase
            "convert_status": self.convert_status.value if self.convert_status else None,
            "convert_worker_id": self.convert_worker_id,
            "convert_started_at": self.convert_started_at.isoformat() if self.convert_started_at else None,
            "convert_completed_at": self.convert_completed_at.isoformat() if self.convert_completed_at else None,
            "convert_error": self.convert_error,
            "convert_retry_count": self.convert_retry_count,
            # Classification phase
            "classification_status": self.classification_status.value if self.classification_status else None,
            "classification_worker_id": self.classification_worker_id,
            "classification_started_at": self.classification_started_at.isoformat() if self.classification_started_at else None,
            "classification_completed_at": self.classification_completed_at.isoformat() if self.classification_completed_at else None,
            "classification_error": self.classification_error,
            "classification_retry_count": self.classification_retry_count,
            # Extraction phase
            "extraction_status": self.extraction_status.value if self.extraction_status else None,
            "extraction_worker_id": self.extraction_worker_id,
            "extraction_started_at": self.extraction_started_at.isoformat() if self.extraction_started_at else None,
            "extraction_completed_at": self.extraction_completed_at.isoformat() if self.extraction_completed_at else None,
            "extraction_error": self.extraction_error,
            "extraction_retry_count": self.extraction_retry_count,
            # Routing
            "processing_path": self.processing_path,
            # Metadata
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
