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

    def get_message(self, message_id: UUID) -> Optional[ChatMessage]:
        """Get a specific message by ID."""
        return self.db.query(ChatMessage).filter(
            ChatMessage.id == message_id
        ).first()

    def update_message(
        self,
        message_id: UUID,
        content: str,
        message_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ChatMessage]:
        """
        Update a chat message content.
        Used for correcting user queries or assistant responses.

        Returns:
            Updated ChatMessage or None if not found
        """
        message = self.get_message(message_id)

        if not message:
            logger.warning(f"Message {message_id} not found for update")
            return None

        message.content = content

        if message_metadata is not None:
            # Merge with existing metadata
            existing_metadata = message.message_metadata or {}
            existing_metadata.update(message_metadata)
            existing_metadata["last_edited_at"] = datetime.utcnow().isoformat()
            message.message_metadata = existing_metadata
        else:
            # Just mark as edited
            existing_metadata = message.message_metadata or {}
            existing_metadata["last_edited_at"] = datetime.utcnow().isoformat()
            message.message_metadata = existing_metadata

        # Update session's updated_at timestamp
        session = self.get_session(message.session_id)
        if session:
            session.updated_at = datetime.utcnow()

        self.db.commit()
        self.db.refresh(message)

        logger.info(f"Updated message {message_id} content")
        return message

    def delete_messages_after(
        self,
        session_id: UUID,
        message_id: UUID
    ) -> int:
        """
        Delete all messages after a specific message (including the message itself).
        Used for retry functionality to clean up failed conversation attempts.

        Returns:
            Number of messages deleted
        """
        # Get the target message to find its timestamp
        target_message = self.db.query(ChatMessage).filter(
            ChatMessage.id == message_id,
            ChatMessage.session_id == session_id
        ).first()

        if not target_message:
            logger.warning(f"Message {message_id} not found in session {session_id}")
            return 0

        target_timestamp = target_message.created_at

        # Get all messages in the session ordered by creation time
        all_messages = self.db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.created_at).all()

        # Find the index of the target message
        target_index = None
        for idx, msg in enumerate(all_messages):
            if msg.id == message_id:
                target_index = idx
                break

        if target_index is None:
            logger.warning(f"Message {message_id} not found in ordered list")
            return 0

        # Delete the target message and all messages after it (by index)
        messages_to_delete = all_messages[target_index:]

        count = len(messages_to_delete)
        # Calculate the new message count directly: messages at indices 0 to target_index-1 remain
        new_message_count = target_index

        logger.info(f"Deleting {count} messages starting from index {target_index} (message {message_id})")
        for msg in messages_to_delete:
            logger.debug(f"  Deleting message {msg.id}: {msg.role} - {msg.content[:50]}...")
            self.db.delete(msg)

        # Flush the deletes to ensure they are executed
        self.db.flush()

        # Update session message count using the calculated value (more reliable than querying)
        session = self.get_session(session_id)
        if session:
            session.message_count = new_message_count
            session.updated_at = datetime.utcnow()
            logger.info(f"Updated session message_count to {new_message_count} (deleted {count} messages)")

        self.db.commit()

        # Refresh the session to ensure the updated message_count is reflected
        if session:
            self.db.refresh(session)

        logger.info(f"Successfully deleted {count} messages after message {message_id} in session {session_id}")
        return count

    def get_message_count(self, session_id: UUID) -> int:
        """
        Get the actual message count for a session by counting messages in the database.
        This is the runtime count and should be used instead of the stored message_count column.
        """
        return self.db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).count()

    def get_message_counts_for_sessions(self, session_ids: List[UUID]) -> Dict[UUID, int]:
        """
        Get the actual message counts for multiple sessions in a single query.
        Returns a dictionary mapping session_id to message count.
        """
        from sqlalchemy import func as sqlfunc

        if not session_ids:
            return {}

        results = self.db.query(
            ChatMessage.session_id,
            sqlfunc.count(ChatMessage.id).label('count')
        ).filter(
            ChatMessage.session_id.in_(session_ids)
        ).group_by(
            ChatMessage.session_id
        ).all()

        # Build dictionary with counts, defaulting to 0 for sessions with no messages
        counts = {session_id: 0 for session_id in session_ids}
        for session_id, count in results:
            counts[session_id] = count

        return counts

    def recalculate_message_count(self, session_id: UUID) -> int:
        """
        Recalculate and update the message count for a session based on actual messages.
        Returns the updated count.
        """
        session = self.get_session(session_id)
        if not session:
            return 0

        actual_count = self.db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).count()

        session.message_count = actual_count
        session.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(session)

        logger.info(f"Recalculated message count for session {session_id}: {actual_count}")
        return actual_count

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

