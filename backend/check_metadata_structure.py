#!/usr/bin/env python3
"""Check metadata structure for meal receipts."""
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from db.database import get_db_session
from db.document_repository import DocumentRepository

def main():
    with get_db_session() as db:
        repo = DocumentRepository(db)
        docs = repo.get_all_with_metadata()
        
        # Get one meal receipt to see full metadata structure
        for doc in docs:
            if doc.filename == "Meal-01-15.png":
                print("=" * 80)
                print(f"Metadata structure for: {doc.filename}")
                print("=" * 80)
                print(json.dumps(doc.document_metadata, indent=2))
                break

if __name__ == "__main__":
    main()

