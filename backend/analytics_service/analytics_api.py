"""
Analytics API endpoints.

Provides REST API for:
- Analytics session management
- Intent classification
- Query planning and execution
- Extracted data querying
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User, DocumentData, DataSchema, AnalyticsSession, AnalyticsState
from auth.dependencies import get_current_active_user

from .analytics_session_manager import AnalyticsSessionManager
from .redis_session_manager import RedisSessionManager
from .intent_classifier import IntentClassifier, QueryIntent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


# ===== Request/Response Models =====

class ClassifyIntentRequest(BaseModel):
    """Request to classify query intent."""
    query: str = Field(..., min_length=1)
    workspace_id: Optional[str] = None


class ClassifyIntentResponse(BaseModel):
    """Response from intent classification."""
    intent: str
    confidence: float
    reasoning: str
    requires_extracted_data: bool
    suggested_schemas: List[str] = []
    detected_entities: List[str] = []
    detected_metrics: List[str] = []
    detected_time_range: Optional[Dict[str, str]] = None


class AnalyticsSessionResponse(BaseModel):
    """Analytics session response."""
    id: str
    chat_session_id: Optional[str]
    state: str
    original_query: str
    intent_classification: Optional[Dict[str, Any]]
    gathered_info: Dict[str, Any]
    current_plan: Optional[Dict[str, Any]]
    plan_version: int
    execution_progress: Optional[Dict[str, Any]]
    cached_results: Optional[Dict[str, Any]]
    created_at: Optional[str]
    updated_at: Optional[str]


class CreateAnalyticsSessionRequest(BaseModel):
    """Request to create analytics session."""
    chat_session_id: str
    workspace_id: str
    query: str = Field(..., min_length=1)


class UpdateSessionStateRequest(BaseModel):
    """Request to update session state."""
    new_state: str
    additional_data: Optional[Dict[str, Any]] = None


class UpdatePlanRequest(BaseModel):
    """Request to update plan."""
    plan: Dict[str, Any]


class AddClarificationRequest(BaseModel):
    """Request to add clarification."""
    question: str
    answer: str


class ExtractedDataResponse(BaseModel):
    """Response for extracted data."""
    id: str
    document_id: str
    schema_type: str
    header_data: Dict[str, Any]
    summary_data: Dict[str, Any]
    line_items_count: int
    validation_status: str
    extraction_method: Optional[str]
    created_at: Optional[str]


class SchemaResponse(BaseModel):
    """Response for schema definition."""
    id: str
    schema_type: str
    domain: str
    display_name: Optional[str]
    description: Optional[str]
    header_schema: Dict[str, Any]
    line_items_schema: Optional[Dict[str, Any]]
    summary_schema: Optional[Dict[str, Any]]


class RedisStatsResponse(BaseModel):
    """Redis connection stats."""
    connected: bool
    host: Optional[str]
    port: Optional[int]
    db: Optional[int]
    active_sessions: Optional[int]
    error: Optional[str]


# ===== Intent Classification =====

@router.post("/classify-intent", response_model=ClassifyIntentResponse)
def classify_intent(
    request: ClassifyIntentRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Classify the intent of a query."""
    classifier = IntentClassifier()

    # Get available schemas
    schemas = db.query(DataSchema.schema_type).filter(
        DataSchema.is_active == True
    ).all()
    available_schemas = [s[0] for s in schemas]

    result = classifier.classify(request.query, available_schemas)

    return ClassifyIntentResponse(
        intent=result.intent.value,
        confidence=result.confidence,
        reasoning=result.reasoning,
        requires_extracted_data=result.requires_extracted_data,
        suggested_schemas=result.suggested_schemas,
        detected_entities=result.detected_entities,
        detected_metrics=result.detected_metrics,
        detected_time_range=result.detected_time_range
    )


# ===== Session Management =====

@router.post("/sessions", response_model=AnalyticsSessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    request: CreateAnalyticsSessionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new analytics session."""
    manager = AnalyticsSessionManager(db)

    session = manager.get_or_create_session(
        chat_session_id=UUID(request.chat_session_id),
        user_id=current_user.id,
        workspace_id=UUID(request.workspace_id),
        query=request.query
    )

    return AnalyticsSessionResponse(**session)


@router.get("/sessions/{session_id}", response_model=AnalyticsSessionResponse)
def get_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get an analytics session."""
    manager = AnalyticsSessionManager(db)
    session = manager.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Verify ownership
    if session.get("user_id") != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    return AnalyticsSessionResponse(**session)


@router.get("/sessions", response_model=List[AnalyticsSessionResponse])
def list_sessions(
    include_completed: bool = False,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List analytics sessions for the current user."""
    manager = AnalyticsSessionManager(db)
    sessions = manager.get_user_sessions(
        user_id=current_user.id,
        include_completed=include_completed,
        limit=limit
    )

    return [AnalyticsSessionResponse(**s) for s in sessions]


@router.patch("/sessions/{session_id}/state", response_model=AnalyticsSessionResponse)
def update_session_state(
    session_id: str,
    request: UpdateSessionStateRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update session state."""
    manager = AnalyticsSessionManager(db)

    # Verify session exists and belongs to user
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("user_id") != str(current_user.id):
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        new_state = AnalyticsState(request.new_state)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid state: {request.new_state}")

    updated = manager.transition_state(session_id, new_state, request.additional_data)
    return AnalyticsSessionResponse(**updated)


@router.patch("/sessions/{session_id}/plan", response_model=AnalyticsSessionResponse)
def update_session_plan(
    session_id: str,
    request: UpdatePlanRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update session plan."""
    manager = AnalyticsSessionManager(db)

    # Verify session exists and belongs to user
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("user_id") != str(current_user.id):
        raise HTTPException(status_code=403, detail="Access denied")

    updated = manager.set_plan(session_id, request.plan)
    return AnalyticsSessionResponse(**updated)


@router.post("/sessions/{session_id}/clarification", response_model=AnalyticsSessionResponse)
def add_clarification(
    session_id: str,
    request: AddClarificationRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Add a clarification Q&A to session."""
    manager = AnalyticsSessionManager(db)

    # Verify session exists and belongs to user
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("user_id") != str(current_user.id):
        raise HTTPException(status_code=403, detail="Access denied")

    updated = manager.add_clarification(session_id, request.question, request.answer)
    return AnalyticsSessionResponse(**updated)


@router.get("/sessions/{session_id}/recover", response_model=AnalyticsSessionResponse)
def recover_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Recover a session after disconnection."""
    manager = AnalyticsSessionManager(db)

    session = manager.recover_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("user_id") != str(current_user.id):
        raise HTTPException(status_code=403, detail="Access denied")

    return AnalyticsSessionResponse(**session)


# ===== Extracted Data =====

@router.get("/data", response_model=List[ExtractedDataResponse])
def list_extracted_data(
    schema_type: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List extracted data for documents the user can access."""
    from db.models import Document

    # Get user's accessible documents
    query = db.query(DocumentData).join(
        Document, DocumentData.document_id == Document.id
    ).filter(
        Document.owner_id == current_user.id
    )

    if schema_type:
        query = query.filter(DocumentData.schema_type == schema_type)

    data = query.limit(limit).all()

    return [
        ExtractedDataResponse(
            id=str(d.id),
            document_id=str(d.document_id),
            schema_type=d.schema_type,
            header_data=d.header_data or {},
            summary_data=d.summary_data or {},
            line_items_count=d.line_items_count or 0,
            validation_status=d.validation_status or "unknown",
            extraction_method=d.extraction_method,
            created_at=d.created_at.isoformat() if d.created_at else None
        )
        for d in data
    ]


@router.get("/data/{document_id}", response_model=ExtractedDataResponse)
def get_extracted_data(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get extracted data for a specific document."""
    from db.models import Document

    # Verify document access
    doc = db.query(Document).filter(Document.id == UUID(document_id)).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    data = db.query(DocumentData).filter(
        DocumentData.document_id == UUID(document_id)
    ).first()

    if not data:
        raise HTTPException(status_code=404, detail="No extracted data for this document")

    return ExtractedDataResponse(
        id=str(data.id),
        document_id=str(data.document_id),
        schema_type=data.schema_type,
        header_data=data.header_data or {},
        summary_data=data.summary_data or {},
        line_items_count=data.line_items_count or 0,
        validation_status=data.validation_status or "unknown",
        extraction_method=data.extraction_method,
        created_at=data.created_at.isoformat() if data.created_at else None
    )


# ===== Schemas =====

@router.get("/schemas", response_model=List[SchemaResponse])
def list_schemas(
    domain: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List available data schemas."""
    query = db.query(DataSchema).filter(DataSchema.is_active == True)

    if domain:
        query = query.filter(DataSchema.domain == domain)

    schemas = query.all()

    return [
        SchemaResponse(
            id=str(s.id),
            schema_type=s.schema_type,
            domain=s.domain,
            display_name=s.display_name,
            description=s.description,
            header_schema=s.header_schema or {},
            line_items_schema=s.line_items_schema,
            summary_schema=s.summary_schema
        )
        for s in schemas
    ]


@router.get("/schemas/{schema_type}", response_model=SchemaResponse)
def get_schema(
    schema_type: str,
    db: Session = Depends(get_db)
):
    """Get a specific data schema."""
    schema = db.query(DataSchema).filter(
        DataSchema.schema_type == schema_type,
        DataSchema.is_active == True
    ).first()

    if not schema:
        raise HTTPException(status_code=404, detail="Schema not found")

    return SchemaResponse(
        id=str(schema.id),
        schema_type=schema.schema_type,
        domain=schema.domain,
        display_name=schema.display_name,
        description=schema.description,
        header_schema=schema.header_schema or {},
        line_items_schema=schema.line_items_schema,
        summary_schema=schema.summary_schema
    )


# ===== Admin/Debug Endpoints =====

@router.get("/redis/stats", response_model=RedisStatsResponse)
def get_redis_stats(
    current_user: User = Depends(get_current_active_user)
):
    """Get Redis connection statistics."""
    redis_manager = RedisSessionManager()
    stats = redis_manager.get_stats()
    return RedisStatsResponse(**stats)


@router.post("/cleanup-expired")
def cleanup_expired_sessions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Clean up expired analytics sessions."""
    manager = AnalyticsSessionManager(db)
    count = manager.cleanup_expired()
    return {"cleaned_up": count}
