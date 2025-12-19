#!/usr/bin/env python3
"""
Script to fix message counts for all chat sessions.
This recalculates the message_count field based on actual messages in the database.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.models import ChatSession, ChatMessage
from db.database import get_db_session

def fix_message_counts():
    """Fix message counts for all chat sessions."""
    with get_db_session() as db:
        # Get all sessions
        sessions = db.query(ChatSession).all()

        print(f"Found {len(sessions)} chat sessions")

        fixed_count = 0
        for session in sessions:
            # Count actual messages
            actual_count = db.query(ChatMessage).filter(
                ChatMessage.session_id == session.id
            ).count()

            if session.message_count != actual_count:
                print(f"Session {session.id} ({session.session_name}): "
                      f"stored={session.message_count}, actual={actual_count} - FIXING")
                session.message_count = actual_count
                fixed_count += 1
            else:
                print(f"Session {session.id} ({session.session_name}): "
                      f"count={actual_count} - OK")

        print(f"\nFixed {fixed_count} sessions")

if __name__ == "__main__":
    fix_message_counts()

