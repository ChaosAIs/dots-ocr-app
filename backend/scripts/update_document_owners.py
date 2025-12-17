#!/usr/bin/env python3
"""
Script to update existing documents with fyang user ID.
This assigns the fyang user as the creator and updater of all existing documents.
"""
import sys
import os

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.database import get_db_session
from db.models import User, Document
from sqlalchemy import text


def main():
    """Update all documents with fyang user ID."""
    with get_db_session() as db:
        # Find the fyang user
        user = db.query(User).filter(User.username == 'fyang').first()
        
        if not user:
            print("ERROR: User 'fyang' not found in database")
            print("Please create the user first before running this script")
            return 1
        
        print(f"Found user: {user.username} (ID: {user.id})")
        print(f"Email: {user.email}")
        print(f"Full name: {user.full_name}")
        print()
        
        # Count documents that need updating
        count_query = db.execute(
            text('''
                SELECT COUNT(*) 
                FROM documents 
                WHERE created_by IS NULL OR updated_by IS NULL
            ''')
        )
        count = count_query.scalar()
        
        if count == 0:
            print("No documents need updating. All documents already have owners assigned.")
            return 0
        
        print(f"Found {count} documents that need updating")
        print(f"Updating documents to assign user '{user.username}' as creator and updater...")
        
        # Update all documents that don't have created_by or updated_by set
        result = db.execute(
            text('''
                UPDATE documents 
                SET created_by = :user_id, updated_by = :user_id
                WHERE created_by IS NULL OR updated_by IS NULL
            '''),
            {'user_id': user.id}
        )
        
        print(f"✓ Successfully updated {result.rowcount} documents")
        print()
        
        # Verify the update
        verify_query = db.execute(
            text('''
                SELECT COUNT(*) 
                FROM documents 
                WHERE created_by = :user_id AND updated_by = :user_id
            '''),
            {'user_id': user.id}
        )
        verify_count = verify_query.scalar()
        print(f"Verification: {verify_count} documents now have '{user.username}' as owner")
        
        return 0


if __name__ == "__main__":
    sys.exit(main())

