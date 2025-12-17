"""
Conversation manager for handling chat sessions with short-term and long-term memory.
"""
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from db.chat_repository import ChatRepository
from db.models import ChatSession, ChatMessage
from rag_service.llm_service import get_llm_service
from .query_analyzer import analyze_query_with_llm

logger = logging.getLogger(__name__)


def is_greeting_conversation(user_message: str, assistant_response: str) -> bool:
    """
    Detect if the conversation is just a greeting exchange using LLM analysis.

    Args:
        user_message: The first user message
        assistant_response: The first assistant response

    Returns:
        True if this appears to be a greeting conversation, False otherwise
    """
    try:
        # Use LLM-based query analyzer
        analysis = analyze_query_with_llm(user_message, assistant_response)
        return analysis.is_greeting
    except Exception as e:
        logger.error(f"Error in LLM greeting detection: {e}")
        # Fallback to simple heuristic
        return _fallback_greeting_detection(user_message, assistant_response)


def _fallback_greeting_detection(user_message: str, assistant_response: str) -> bool:
    """Fallback heuristic-based greeting detection if LLM fails."""
    greeting_patterns = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon']
    user_lower = user_message.lower().strip().rstrip('!?.,:;')

    if len(user_lower.split()) <= 5:
        for pattern in greeting_patterns:
            if pattern in user_lower:
                assistant_lower = assistant_response.lower()
                if any(ind in assistant_lower for ind in ['hello', 'hi', 'how can i help', 'how may i help']):
                    logger.info(f"Fallback: Detected greeting - User: '{user_message[:50]}'")
                    return True
    return False


def generate_chat_title_with_llm(user_message: str, assistant_response: str) -> str:
    """
    Generate a concise chat title using LLM to summarize the conversation topic.

    Args:
        user_message: The first user message
        assistant_response: The first assistant response

    Returns:
        A concise title (max 50 characters)
    """
    try:
        llm_service = get_llm_service()

        # Use a lightweight model for quick title generation
        llm = llm_service.get_query_model(
            temperature=0.3,
            num_ctx=1024,
            num_predict=20  # Short output for title
        )

        # Create a prompt for title generation
        prompt = f"""Based on the following conversation, generate a very short and concise title (maximum 6 words) that captures the main topic. Only output the title, nothing else.

User: {user_message[:200]}
Assistant: {assistant_response[:200]}

Title:"""

        # Get LLM response
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])

        # Extract and clean the title
        title = response.content.strip()

        # Remove quotes if present
        title = title.strip('"').strip("'")

        # Truncate to 50 characters
        if len(title) > 50:
            title = title[:47] + "..."

        logger.info(f"Generated title with LLM: {title}")
        return title if title else "New Chat"

    except Exception as e:
        logger.error(f"Error generating title with LLM: {e}")
        # Fallback to simple heuristic
        return generate_chat_title_fallback(user_message)


def generate_chat_title_fallback(user_message: str) -> str:
    """
    Fallback method to generate title using simple heuristics.

    Args:
        user_message: The first user message

    Returns:
        A concise title (max 50 characters)
    """
    # Clean the user message
    message = user_message.strip()

    # Take first few words
    words = message.split()
    title = ' '.join(words[:7])

    # Capitalize first letter
    if title:
        title = title[0].upper() + title[1:]

    # Truncate to 50 characters
    if len(title) > 50:
        title = title[:47] + "..."

    return title if title else "New Chat"


class ConversationManager:
    """
    Manages conversation history with short-term and long-term memory.
    
    Short-term memory: Last 10-20 messages for immediate context
    Long-term memory: Full session history with summaries
    """
    
    # Configuration
    SHORT_TERM_WINDOW = 20  # Number of recent messages to keep in context
    SUMMARY_THRESHOLD = 50  # Create summary after this many messages
    
    def __init__(self, db: Session):
        """Initialize conversation manager."""
        self.db = db
        self.chat_repo = ChatRepository(db)
    
    def create_session(
        self,
        user_id: UUID,
        session_name: Optional[str] = None
    ) -> ChatSession:
        """Create a new chat session."""
        return self.chat_repo.create_session(
            user_id=user_id,
            session_name=session_name
        )
    
    def get_session(self, session_id: UUID) -> Optional[ChatSession]:
        """Get chat session by ID."""
        return self.chat_repo.get_session(session_id)
    
    def get_user_sessions(
        self,
        user_id: UUID,
        limit: int = 50
    ) -> List[ChatSession]:
        """Get all chat sessions for a user."""
        return self.chat_repo.get_user_sessions(user_id, limit=limit)
    
    def add_message(
        self,
        session_id: UUID,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """Add a message to the session."""
        message = self.chat_repo.add_message(
            session_id=session_id,
            role=role,
            content=content,
            message_metadata=metadata
        )

        # Get fresh session data to check message count
        session = self.get_session(session_id)

        # Auto-generate or regenerate title after each assistant response
        if session and role == "assistant":
            # Trigger title generation/regeneration check on every assistant response
            logger.info(f"Triggering title check for session {session_id}, message_count={session.message_count}")
            self._auto_generate_or_update_title(session_id, content)

        # Check if we need to create a summary
        if session and session.message_count >= self.SUMMARY_THRESHOLD:
            self._maybe_create_summary(session_id)

        return message

    def _auto_generate_or_update_title(self, session_id: UUID, assistant_response: str):
        """
        Automatically generate or update the title for the chat session.
        Uses LLM to generate a meaningful title based on the conversation.

        This is triggered on every assistant response to check if title needs generation/regeneration.

        Smart title generation:
        - For first response: checks if it's a greeting, skips if so
        - For subsequent responses: uses latest user message if title is still generic
        - Only generates titles for substantive conversations
        - Checks if current title is generic and needs regeneration
        """
        try:
            # Get the session to check current title and message count
            session = self.get_session(session_id)
            if not session:
                return

            # Check if title needs generation/regeneration
            needs_title_update = (
                not session.session_name or
                session.session_name.strip() == "" or
                session.session_name == "New Chat" or
                session.session_name.startswith("Chat ")  # Default pattern like "Chat 2024-01-01"
            )

            if not needs_title_update:
                # Title already exists and is not generic, skip
                return

            # Get all messages to find the right pair to use for title generation
            all_messages = self.chat_repo.get_session_messages(session_id)
            if len(all_messages) < 2:
                # Not enough messages yet
                return

            # Determine which user message to use for title generation
            if session.message_count == 2:
                # First response - use first user message and check for greeting
                user_message = all_messages[0].content

                # Check if this is just a greeting conversation
                if is_greeting_conversation(user_message, assistant_response):
                    logger.info(f"Skipping title generation for greeting conversation in session {session_id}")
                    # Keep the default "New Chat" or timestamp-based title
                    return
            else:
                # Subsequent responses - find the most recent substantive user message
                # Skip the first exchange if it was a greeting
                first_user_msg = all_messages[0].content
                first_assistant_msg = all_messages[1].content if len(all_messages) > 1 else ""

                # Check if first exchange was a greeting
                first_was_greeting = is_greeting_conversation(first_user_msg, first_assistant_msg)

                if first_was_greeting and len(all_messages) >= 4:
                    # Use the second user message (index 2) for title generation
                    user_message = all_messages[2].content
                    logger.info(f"First exchange was greeting, using second user message for title: '{user_message[:50]}...'")
                else:
                    # Use the first user message
                    user_message = first_user_msg

            # Use LLM to generate a meaningful title
            title = generate_chat_title_with_llm(user_message, assistant_response)

            # Update session name
            self.chat_repo.update_session(
                session_id=session_id,
                session_name=title
            )
            logger.info(f"Auto-generated/updated title for session {session_id}: {title}")
        except Exception as e:
            logger.error(f"Error auto-generating/updating title: {e}")

    def regenerate_title_if_needed(self, session_id: UUID) -> Optional[str]:
        """
        Check if a session has a generic title ("New Chat" or empty) and regenerate it.
        This is useful for existing sessions that were created before LLM title generation.

        Args:
            session_id: The session UUID

        Returns:
            The new title if regenerated, None otherwise
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return None

            # Check if title needs regeneration
            needs_regeneration = (
                not session.session_name or
                session.session_name.strip() == "" or
                session.session_name == "New Chat" or
                session.session_name.startswith("Chat ")  # Default pattern like "Chat 2024-01-01"
            )

            if not needs_regeneration:
                return None

            # Get the first two messages
            messages = self.chat_repo.get_session_messages(session_id, limit=2)
            if len(messages) < 2:
                logger.info(f"Session {session_id} has less than 2 messages, skipping title regeneration")
                return None

            user_message = messages[0].content
            assistant_response = messages[1].content

            # Check if this is just a greeting conversation
            if is_greeting_conversation(user_message, assistant_response):
                logger.info(f"Skipping title regeneration for greeting conversation in session {session_id}")
                # Keep the default title for greeting conversations
                return None

            # Generate new title with LLM
            new_title = generate_chat_title_with_llm(user_message, assistant_response)

            # Update session name
            self.chat_repo.update_session(
                session_id=session_id,
                session_name=new_title
            )

            logger.info(f"Regenerated title for session {session_id}: {new_title}")
            return new_title

        except Exception as e:
            logger.error(f"Error regenerating title for session {session_id}: {e}")
            return None
    
    def get_conversation_context(
        self,
        session_id: UUID,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Get conversation context for the session.
        
        Returns:
            Dictionary with:
            - short_term_messages: Recent messages (last 20)
            - session_summary: Latest session summary (if available)
            - session_metadata: Session metadata for context tracking
        """
        # Get short-term messages (recent 20)
        short_term_messages = self.chat_repo.get_recent_messages(
            session_id=session_id,
            limit=self.SHORT_TERM_WINDOW
        )
        
        context = {
            "short_term_messages": [msg.to_dict() for msg in short_term_messages],
            "session_summary": None,
            "session_metadata": {}
        }
        
        # Get session metadata
        session = self.get_session(session_id)
        if session:
            context["session_metadata"] = session.session_metadata
        
        # Get latest summary if requested
        if include_summary:
            latest_summary = self.chat_repo.get_latest_summary(session_id)
            if latest_summary:
                context["session_summary"] = latest_summary.to_dict()
        
        return context
    
    def get_all_messages(
        self,
        session_id: UUID,
        limit: Optional[int] = None
    ) -> List[ChatMessage]:
        """Get all messages for a session."""
        return self.chat_repo.get_session_messages(
            session_id=session_id,
            limit=limit
        )
    
    def update_session_metadata(
        self,
        session_id: UUID,
        metadata: Dict[str, Any]
    ) -> Optional[ChatSession]:
        """Update session metadata for context tracking."""
        return self.chat_repo.update_session(
            session_id=session_id,
            session_metadata=metadata
        )
    
    def delete_session(self, session_id: UUID) -> bool:
        """Delete a chat session."""
        return self.chat_repo.delete_session(session_id)
    
    def _maybe_create_summary(self, session_id: UUID):
        """
        Create a summary if needed.
        This is a placeholder for future implementation with LLM summarization.
        """
        # TODO: Implement LLM-based summarization
        logger.info(f"Summary creation triggered for session {session_id}")
        pass

