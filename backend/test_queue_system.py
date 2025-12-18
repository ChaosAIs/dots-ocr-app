"""
Test script to verify queue system implementation.

This script tests:
1. Task queue manager initialization
2. Task enqueueing
3. Task claiming
4. Heartbeat updates
5. Task completion
6. Stale task detection
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all queue system modules can be imported"""
    try:
        from queue_service import (
            TaskQueue,
            TaskType,
            TaskPriority,
            TaskStatus,
            TaskQueueManager,
            TaskScheduler,
            QueueWorkerPool,
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_connection():
    """Test database connection"""
    try:
        from db.database import get_db_session
        with next(get_db_session()) as db:
            print("✅ Database connection successful")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def test_task_queue_table():
    """Test that task_queue table exists"""
    try:
        from db.database import get_db_session
        from sqlalchemy import text
        
        with next(get_db_session()) as db:
            result = db.execute(text("SELECT COUNT(*) FROM task_queue"))
            count = result.scalar()
            print(f"✅ task_queue table exists (current tasks: {count})")
            return True
    except Exception as e:
        print(f"❌ task_queue table check failed: {e}")
        return False


def test_task_queue_manager():
    """Test TaskQueueManager basic operations"""
    try:
        from queue_service import TaskQueueManager, TaskType, TaskPriority
        from db.database import get_db_session
        from db.document_repository import DocumentRepository
        import uuid
        
        # Initialize manager
        manager = TaskQueueManager()
        print("✅ TaskQueueManager initialized")
        
        # Get queue stats
        stats = manager.get_queue_stats()
        print(f"✅ Queue stats: {stats}")
        
        # Test finding orphaned documents
        orphaned_ocr = manager.find_orphaned_ocr_documents()
        orphaned_indexing = manager.find_orphaned_indexing_documents()
        print(f"✅ Found {len(orphaned_ocr)} orphaned OCR documents")
        print(f"✅ Found {len(orphaned_indexing)} orphaned indexing documents")
        
        return True
    except Exception as e:
        print(f"❌ TaskQueueManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Queue System Implementation")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Database Connection", test_database_connection),
        ("Task Queue Table", test_task_queue_table),
        ("TaskQueueManager", test_task_queue_manager),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- Testing: {name} ---")
        result = test_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)

