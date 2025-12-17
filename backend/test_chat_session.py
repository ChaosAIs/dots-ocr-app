#!/usr/bin/env python3
"""
Test script to verify chat session functionality.
"""
import sys
import os
from uuid import uuid4

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.database import get_db_session
from chat_service.conversation_manager import ConversationManager

def test_chat_session():
    """Test creating a session and adding messages."""
    print("Testing chat session functionality...")
    
    with get_db_session() as db:
        conv_manager = ConversationManager(db)
        
        # Create a test user ID (in production, this would come from auth)
        test_user_id = uuid4()
        print(f"✓ Using test user ID: {test_user_id}")
        
        # Create a session
        session = conv_manager.create_session(
            user_id=test_user_id,
            session_name="Test Chat Session"
        )
        print(f"✓ Created session: {session.id} - {session.session_name}")
        
        # Add user message
        user_msg = conv_manager.add_message(
            session_id=session.id,
            role="user",
            content="Hello, this is a test message!"
        )
        print(f"✓ Added user message: {user_msg.id}")
        
        # Add assistant message
        assistant_msg = conv_manager.add_message(
            session_id=session.id,
            role="assistant",
            content="Hello! I received your test message."
        )
        print(f"✓ Added assistant message: {assistant_msg.id}")
        
        # Get conversation context
        context = conv_manager.get_conversation_context(session.id)
        print(f"✓ Retrieved conversation context:")
        print(f"  - Short-term messages: {len(context['short_term_messages'])}")
        
        # Get all messages
        all_messages = conv_manager.get_all_messages(session.id)
        print(f"✓ Retrieved all messages: {len(all_messages)}")
        
        for msg in all_messages:
            print(f"  - [{msg.role}] {msg.content[:50]}...")
        
        # Get user sessions
        sessions = conv_manager.get_user_sessions(test_user_id)
        print(f"✓ Retrieved user sessions: {len(sessions)}")
        
        for s in sessions:
            print(f"  - {s.session_name} ({s.message_count} messages)")
        
        print("\n✅ All tests passed!")
        return True

if __name__ == "__main__":
    try:
        test_chat_session()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

