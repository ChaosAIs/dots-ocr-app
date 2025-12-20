"""
Chat session API endpoints for managing conversation history.
"""
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from db.database import get_db
from db.models import User, UserRole, UserStatus
from db.user_repository import UserRepository
from auth.dependencies import get_current_active_user, get_optional_user
from chat_service.conversation_manager import ConversationManager
from uuid import uuid4, UUID

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat/sessions", tags=["Chat Sessions"])


# ===== Request/Response Models =====

class CreateSessionRequest(BaseModel):
    """Create chat session request."""
    session_name: Optional[str] = None


class UpdateSessionRequest(BaseModel):
    """Update chat session request."""
    session_name: Optional[str] = None
    is_active: Optional[bool] = None


class UpdateMessageRequest(BaseModel):
    """Update chat message request."""
    content: str = Field(..., min_length=1, description="The new message content")


class SessionResponse(BaseModel):
    """Chat session response."""
    id: str
    user_id: str
    session_name: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    last_message_at: Optional[str]
    message_count: int
    is_active: bool
    session_metadata: dict


class MessageResponse(BaseModel):
    """Chat message response."""
    id: str
    session_id: str
    role: str
    content: str
    created_at: Optional[str]
    message_metadata: dict


class ConversationContextResponse(BaseModel):
    """Conversation context response."""
    short_term_messages: List[dict]
    session_summary: Optional[dict]
    session_metadata: dict


# ===== API Endpoints =====

@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    request: CreateSessionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new chat session for the authenticated user."""
    conv_manager = ConversationManager(db)

    session = conv_manager.create_session(
        user_id=current_user.id,
        session_name=request.session_name
    )

    logger.info(f"Created chat session {session.id} for user {current_user.username} ({current_user.id})")
    return SessionResponse(**session.to_dict())


@router.get("", response_model=List[SessionResponse])
def list_sessions(
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all chat sessions for the authenticated user."""
    conv_manager = ConversationManager(db)

    sessions = conv_manager.get_user_sessions(
        user_id=current_user.id,
        limit=limit
    )

    # Compute message counts at runtime for all sessions in a single query
    session_ids = [session.id for session in sessions]
    message_counts = conv_manager.chat_repo.get_message_counts_for_sessions(session_ids)

    return [
        SessionResponse(**session.to_dict(message_count=message_counts.get(session.id, 0)))
        for session in sessions
    ]


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific chat session."""
    conv_manager = ConversationManager(db)

    session = conv_manager.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Verify ownership
    if str(session.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Compute message count at runtime for consistency
    message_count = conv_manager.chat_repo.get_message_count(UUID(session_id))

    return SessionResponse(**session.to_dict(message_count=message_count))


@router.patch("/{session_id}", response_model=SessionResponse)
def update_session(
    session_id: str,
    request: UpdateSessionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update a chat session."""
    conv_manager = ConversationManager(db)

    # Verify session exists and user owns it
    session = conv_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Verify ownership
    if str(session.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Update session
    updated_session = conv_manager.chat_repo.update_session(
        session_id=session_id,
        session_name=request.session_name,
        is_active=request.is_active
    )

    # Compute message count at runtime for consistency
    message_count = conv_manager.chat_repo.get_message_count(UUID(session_id))

    return SessionResponse(**updated_session.to_dict(message_count=message_count))


@router.delete("/{session_id}")
def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a chat session."""
    conv_manager = ConversationManager(db)

    # Verify session exists and user owns it
    session = conv_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Verify ownership
    if str(session.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Delete session
    success = conv_manager.delete_session(session_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session"
        )

    return {"message": "Session deleted successfully"}


@router.post("/cleanup-empty")
def cleanup_empty_sessions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Clean up empty chat sessions (sessions with 0 messages) for the current user."""
    from db.models import ChatSession
    from datetime import datetime

    # Find all empty sessions for this user
    empty_sessions = db.query(ChatSession).filter(
        ChatSession.user_id == current_user.id,
        ChatSession.message_count == 0,
        ChatSession.deleted_at.is_(None)
    ).all()

    count = len(empty_sessions)

    # Soft delete them
    for session in empty_sessions:
        session.deleted_at = datetime.utcnow()
        session.is_active = False

    db.commit()

    logger.info(f"Cleaned up {count} empty sessions for user {current_user.username}")
    return {"deleted_count": count, "message": f"Deleted {count} empty sessions"}


@router.post("/{session_id}/regenerate-title")
def regenerate_session_title(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Regenerate the title for a chat session using LLM."""
    conv_manager = ConversationManager(db)

    # Verify session exists and user owns it
    session = conv_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Verify ownership
    if str(session.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Regenerate title
    new_title = conv_manager.regenerate_title_if_needed(session_id)

    if new_title:
        logger.info(f"Regenerated title for session {session_id}: {new_title}")
        return {"session_id": session_id, "new_title": new_title, "message": "Title regenerated successfully"}
    else:
        return {"session_id": session_id, "new_title": session.session_name, "message": "Title already exists or session has insufficient messages"}


@router.post("/regenerate-all-titles")
def regenerate_all_titles(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Regenerate titles for all sessions with generic titles (New Chat, empty, etc.)."""
    from db.models import ChatSession

    conv_manager = ConversationManager(db)

    # Find all sessions with generic titles
    sessions = db.query(ChatSession).filter(
        ChatSession.user_id == current_user.id,
        ChatSession.deleted_at.is_(None),
        ChatSession.message_count >= 2  # Only sessions with at least 2 messages
    ).all()

    regenerated_count = 0
    failed_count = 0

    for session in sessions:
        # Check if title needs regeneration
        needs_regeneration = (
            not session.session_name or
            session.session_name.strip() == "" or
            session.session_name == "New Chat" or
            session.session_name.startswith("Chat ")
        )

        if needs_regeneration:
            try:
                new_title = conv_manager.regenerate_title_if_needed(session.id)
                if new_title:
                    regenerated_count += 1
            except Exception as e:
                logger.error(f"Failed to regenerate title for session {session.id}: {e}")
                failed_count += 1

    logger.info(f"Regenerated {regenerated_count} titles for user {current_user.username}")
    return {
        "regenerated_count": regenerated_count,
        "failed_count": failed_count,
        "message": f"Regenerated {regenerated_count} titles, {failed_count} failed"
    }


@router.get("/{session_id}/messages", response_model=List[MessageResponse])
def get_session_messages(
    session_id: str,
    limit: Optional[int] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all messages for a chat session."""
    conv_manager = ConversationManager(db)

    # Verify session exists and user owns it
    session = conv_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Verify ownership
    if str(session.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Get messages
    messages = conv_manager.get_all_messages(session_id, limit=limit)

    return [MessageResponse(**msg.to_dict()) for msg in messages]


@router.delete("/{session_id}/messages/after/{message_id}")
def delete_messages_after(
    session_id: str,
    message_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete all messages after a specific message (including the message itself).
    Used for retry functionality to clean up failed conversation attempts.
    """
    conv_manager = ConversationManager(db)

    # Verify session exists and user owns it
    session = conv_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Verify ownership
    if str(session.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Delete messages
    try:
        deleted_count = conv_manager.delete_messages_after(UUID(session_id), UUID(message_id))
        logger.info(f"Deleted {deleted_count} messages after message {message_id} in session {session_id}")
        return {"deleted_count": deleted_count, "message": f"Deleted {deleted_count} messages"}
    except Exception as e:
        logger.error(f"Error deleting messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete messages: {str(e)}"
        )


@router.patch("/{session_id}/messages/{message_id}", response_model=MessageResponse)
def update_message(
    session_id: str,
    message_id: str,
    request: UpdateMessageRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update a chat message content.
    Used for correcting user queries or AI assistant responses.
    """
    conv_manager = ConversationManager(db)

    # Verify session exists and user owns it
    session = conv_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Verify ownership
    if str(session.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Update the message
    try:
        updated_message = conv_manager.chat_repo.update_message(
            message_id=UUID(message_id),
            content=request.content
        )

        if not updated_message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message not found"
            )

        logger.info(f"Updated message {message_id} in session {session_id}")
        return MessageResponse(**updated_message.to_dict())
    except Exception as e:
        logger.error(f"Error updating message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update message: {str(e)}"
        )


@router.get("/{session_id}/context", response_model=ConversationContextResponse)
def get_conversation_context(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get conversation context for a session (short-term + summary)."""
    conv_manager = ConversationManager(db)

    # Verify session exists and user owns it
    session = conv_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Verify ownership
    if str(session.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Get context
    context = conv_manager.get_conversation_context(session_id)

    return ConversationContextResponse(**context)

