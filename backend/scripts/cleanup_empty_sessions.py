#!/usr/bin/env python3
"""
Script to clean up empty chat sessions (sessions with 0 messages).
"""
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.database import get_db_session
from db.models import ChatSession
from datetime import datetime


def cleanup_empty_sessions():
    """Delete all chat sessions with 0 messages."""
    with get_db_session() as db:
        # Find all sessions with 0 messages
        empty_sessions = db.query(ChatSession).filter(
            ChatSession.message_count == 0,
            ChatSession.deleted_at.is_(None)
        ).all()
        
        count = len(empty_sessions)
        
        if count == 0:
            print("No empty sessions found.")
            return 0
        
        print(f"Found {count} empty sessions. Deleting...")
        
        # Soft delete them
        for session in empty_sessions:
            session.deleted_at = datetime.utcnow()
            session.is_active = False
        
        db.commit()
        print(f"Successfully deleted {count} empty sessions.")
        return count


if __name__ == "__main__":
    try:
        deleted_count = cleanup_empty_sessions()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

