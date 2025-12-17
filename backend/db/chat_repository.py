"""
Chat session repository for database operations.
"""
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import desc

from db.models import ChatSession, ChatMessage, ChatSessionSummary

logger = logging.getLogger(__name__)


class ChatRepository:
    """Repository for chat session database operations."""
    
    def __init__(self, db: Session):
        """Initialize repository with database session."""
        self.db = db
    
    # ===== Chat Session Operations =====
    
    def create_session(
        self,
        user_id: UUID,
        session_name: Optional[str] = None,
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> ChatSession:
        """Create a new chat session."""
        session = ChatSession(
            user_id=user_id,
            session_name=session_name or f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            session_metadata=session_metadata or {}
        )
        
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        
        logger.info(f"Created chat session: {session.id} for user: {user_id}")
        return session
    
    def get_session(self, session_id: UUID) -> Optional[ChatSession]:
        """Get chat session by ID."""
        return self.db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.deleted_at.is_(None)
        ).first()
    
    def get_user_sessions(
        self,
        user_id: UUID,
        include_inactive: bool = False,
        limit: int = 50
    ) -> List[ChatSession]:
        """Get all chat sessions for a user. Only returns sessions with at least one message."""
        query = self.db.query(ChatSession).filter(
            ChatSession.user_id == user_id,
            ChatSession.deleted_at.is_(None),
            ChatSession.message_count > 0  # Only show sessions with messages
        )

        if not include_inactive:
            query = query.filter(ChatSession.is_active == True)

        query = query.order_by(desc(ChatSession.updated_at)).limit(limit)

        return query.all()
    
    def update_session(
        self,
        session_id: UUID,
        session_name: Optional[str] = None,
        session_metadata: Optional[Dict[str, Any]] = None,
        is_active: Optional[bool] = None
    ) -> Optional[ChatSession]:
        """Update chat session."""
        session = self.get_session(session_id)
        
        if not session:
            return None
        
        if session_name is not None:
            session.session_name = session_name
        
        if session_metadata is not None:
            session.session_metadata = session_metadata
        
        if is_active is not None:
            session.is_active = is_active
        
        session.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(session)
        
        return session
    
    def delete_session(self, session_id: UUID) -> bool:
        """Soft delete chat session."""
        session = self.get_session(session_id)
        
        if not session:
            return False
        
        session.deleted_at = datetime.utcnow()
        session.is_active = False
        self.db.commit()
        
        logger.info(f"Deleted chat session: {session_id}")
        return True
    
    # ===== Chat Message Operations =====
    
    def add_message(
        self,
        session_id: UUID,
        role: str,
        content: str,
        message_metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """Add a message to a chat session."""
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            message_metadata=message_metadata or {}
        )
        
        self.db.add(message)
        
        # Update session stats
        session = self.get_session(session_id)
        if session:
            session.message_count += 1
            session.last_message_at = datetime.utcnow()
            session.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(message)
        
        return message
    
    def get_session_messages(
        self,
        session_id: UUID,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[ChatMessage]:
        """Get messages for a chat session."""
        query = self.db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.created_at)
        
        if offset > 0:
            query = query.offset(offset)
        
        if limit:
            query = query.limit(limit)
        
        return query.all()

    def get_recent_messages(
        self,
        session_id: UUID,
        limit: int = 20
    ) -> List[ChatMessage]:
        """Get recent messages for short-term context."""
        return self.db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(desc(ChatMessage.created_at)).limit(limit).all()[::-1]  # Reverse to chronological order

    # ===== Chat Session Summary Operations =====

    def create_summary(
        self,
        session_id: UUID,
        summary_type: str,
        summary_content: str,
        message_range_start: Optional[UUID] = None,
        message_range_end: Optional[UUID] = None,
        summary_metadata: Optional[Dict[str, Any]] = None
    ) -> ChatSessionSummary:
        """Create a session summary."""
        summary = ChatSessionSummary(
            session_id=session_id,
            summary_type=summary_type,
            summary_content=summary_content,
            message_range_start=message_range_start,
            message_range_end=message_range_end,
            summary_metadata=summary_metadata or {}
        )

        self.db.add(summary)
        self.db.commit()
        self.db.refresh(summary)

        return summary

    def get_session_summaries(
        self,
        session_id: UUID,
        summary_type: Optional[str] = None
    ) -> List[ChatSessionSummary]:
        """Get summaries for a chat session."""
        query = self.db.query(ChatSessionSummary).filter(
            ChatSessionSummary.session_id == session_id
        )

        if summary_type:
            query = query.filter(ChatSessionSummary.summary_type == summary_type)

        return query.order_by(ChatSessionSummary.created_at).all()

    def get_latest_summary(
        self,
        session_id: UUID,
        summary_type: str = "session"
    ) -> Optional[ChatSessionSummary]:
        """Get the latest summary for a session."""
        return self.db.query(ChatSessionSummary).filter(
            ChatSessionSummary.session_id == session_id,
            ChatSessionSummary.summary_type == summary_type
        ).order_by(desc(ChatSessionSummary.created_at)).first()

