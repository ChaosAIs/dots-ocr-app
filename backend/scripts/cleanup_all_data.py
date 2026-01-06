#!/usr/bin/env python3
"""
Database Cleanup Script

This script removes all data from the application EXCEPT user records.
It cleans up:
1. PostgreSQL: documents, workspaces, chat sessions, task queue, etc.
2. Qdrant Vector Database: document embeddings and metadata
3. Neo4j Graph Database: entities and relationships
4. File System: uploaded files and converted outputs

CAUTION: This is a destructive operation and cannot be undone!
"""

import os
import sys
import shutil
import logging
from typing import Optional

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

from sqlalchemy import text
from sqlalchemy.orm import Session

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cleanup_postgresql(db: Session, dry_run: bool = False) -> dict:
    """
    Clean up all PostgreSQL tables except users.

    Deletion order respects foreign key constraints:
    1. user_documents (bridge table)
    2. document_status_log (CASCADE from documents)
    3. task_queue (CASCADE from documents)
    4. documents
    5. chat_session_summaries (CASCADE from chat_sessions)
    6. chat_messages (CASCADE from chat_sessions)
    7. chat_sessions
    8. workspaces
    9. refresh_tokens

    Returns dict with counts of deleted records.
    """
    results = {}

    try:
        logger.info("Starting PostgreSQL cleanup...")

        # Tables to clean in order (respecting FK constraints)
        # Note: CASCADE tables are automatically cleaned, but we count them first

        # 1. Count records before deletion
        tables_to_count = [
            'user_documents',
            'document_status_log',
            'task_queue',
            'documents',
            'chat_session_summaries',
            'chat_messages',
            'chat_sessions',
            'workspaces',
            'refresh_tokens'
        ]

        for table in tables_to_count:
            try:
                count_result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = count_result.scalar()
                results[table] = {'before': count, 'deleted': 0}
                logger.info(f"  {table}: {count} records")
            except Exception as e:
                logger.warning(f"  Could not count {table}: {e}")
                results[table] = {'before': 0, 'deleted': 0, 'error': str(e)}

        if dry_run:
            logger.info("DRY RUN - No changes made to PostgreSQL")
            return results

        # 2. Delete in correct order
        # user_documents must be deleted before documents
        logger.info("Deleting user_documents...")
        result = db.execute(text("DELETE FROM user_documents"))
        results['user_documents']['deleted'] = result.rowcount

        # document_status_log and task_queue will CASCADE from documents
        # But let's delete them explicitly for accurate counts
        logger.info("Deleting document_status_log...")
        result = db.execute(text("DELETE FROM document_status_log"))
        results['document_status_log']['deleted'] = result.rowcount

        logger.info("Deleting task_queue...")
        result = db.execute(text("DELETE FROM task_queue"))
        results['task_queue']['deleted'] = result.rowcount

        # Now delete documents
        logger.info("Deleting documents...")
        result = db.execute(text("DELETE FROM documents"))
        results['documents']['deleted'] = result.rowcount

        # Chat related tables (CASCADE from chat_sessions)
        logger.info("Deleting chat_session_summaries...")
        result = db.execute(text("DELETE FROM chat_session_summaries"))
        results['chat_session_summaries']['deleted'] = result.rowcount

        logger.info("Deleting chat_messages...")
        result = db.execute(text("DELETE FROM chat_messages"))
        results['chat_messages']['deleted'] = result.rowcount

        logger.info("Deleting chat_sessions...")
        result = db.execute(text("DELETE FROM chat_sessions"))
        results['chat_sessions']['deleted'] = result.rowcount

        # Workspaces (user's workspaces, not users themselves)
        logger.info("Deleting workspaces...")
        result = db.execute(text("DELETE FROM workspaces"))
        results['workspaces']['deleted'] = result.rowcount

        # Refresh tokens (optional - keeps users logged out)
        logger.info("Deleting refresh_tokens...")
        result = db.execute(text("DELETE FROM refresh_tokens"))
        results['refresh_tokens']['deleted'] = result.rowcount

        db.commit()
        logger.info("PostgreSQL cleanup completed successfully")

    except Exception as e:
        db.rollback()
        logger.error(f"PostgreSQL cleanup failed: {e}")
        raise

    return results


def cleanup_qdrant(dry_run: bool = False) -> dict:
    """
    Clean up Qdrant vector database collections.

    Clears:
    - documents collection (document chunks and metadata)
    """
    results = {
        'documents': {'status': 'pending'}
    }

    try:
        logger.info("Starting Qdrant cleanup...")

        from rag_service.vectorstore import (
            clear_collection,
            get_collection_info
        )

        # Get collection info before cleanup
        try:
            main_info = get_collection_info()
            results['documents']['before'] = main_info.get('total_documents', 0)
            logger.info(f"  documents: {results['documents']['before']} documents")
        except Exception as e:
            logger.warning(f"  Could not get main collection info: {e}")
            results['documents']['before'] = 'unknown'

        if dry_run:
            logger.info("DRY RUN - No changes made to Qdrant")
            return results

        # Clear main documents collection
        logger.info("Clearing documents collection...")
        try:
            clear_collection()
            results['documents']['status'] = 'cleared'
            logger.info("  documents cleared successfully")
        except Exception as e:
            results['documents']['status'] = 'error'
            results['documents']['error'] = str(e)
            logger.error(f"  Failed to clear documents: {e}")

        logger.info("Qdrant cleanup completed")

    except ImportError as e:
        logger.warning(f"Qdrant modules not available: {e}")
        results['status'] = 'skipped'
    except Exception as e:
        logger.error(f"Qdrant cleanup failed: {e}")
        results['status'] = 'error'
        results['error'] = str(e)

    return results


def cleanup_neo4j(dry_run: bool = False) -> dict:
    """
    Clean up Neo4j graph database.

    Removes all Entity nodes and RELATES_TO relationships.
    """
    results = {'status': 'pending', 'nodes_deleted': 0}

    try:
        logger.info("Starting Neo4j cleanup...")

        # Check if Neo4j is enabled
        from rag_service.graph_rag import GRAPH_RAG_INDEX_ENABLED

        if not GRAPH_RAG_INDEX_ENABLED:
            logger.info("  Neo4j/GraphRAG is not enabled, skipping...")
            results['status'] = 'skipped'
            results['reason'] = 'GraphRAG not enabled'
            return results

        from rag_service.storage.neo4j_storage import Neo4jStorage

        # Initialize Neo4j storage
        storage = Neo4jStorage()

        if dry_run:
            # Count nodes before deletion
            try:
                import asyncio

                async def count_nodes():
                    query = "MATCH (n:Entity) RETURN count(n) as count"
                    result = await storage._run_query(query)
                    return result[0]['count'] if result else 0

                count = asyncio.run(count_nodes())
                results['before'] = count
                logger.info(f"  Neo4j entities: {count} nodes")
            except Exception as e:
                logger.warning(f"  Could not count Neo4j nodes: {e}")

            logger.info("DRY RUN - No changes made to Neo4j")
            return results

        # Delete all entities and relationships
        logger.info("Deleting all Neo4j entities and relationships...")
        try:
            import asyncio

            async def delete_all():
                # Delete all Entity nodes (DETACH DELETE removes relationships too)
                query = "MATCH (n:Entity) DETACH DELETE n RETURN count(n) as deleted"
                result = await storage._run_query(query)
                return result[0]['deleted'] if result else 0

            deleted = asyncio.run(delete_all())
            results['nodes_deleted'] = deleted
            results['status'] = 'cleared'
            logger.info(f"  Deleted {deleted} Neo4j nodes")

        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            logger.error(f"  Failed to delete Neo4j data: {e}")

        logger.info("Neo4j cleanup completed")

    except ImportError as e:
        logger.warning(f"Neo4j modules not available: {e}")
        results['status'] = 'skipped'
        results['reason'] = str(e)
    except Exception as e:
        logger.error(f"Neo4j cleanup failed: {e}")
        results['status'] = 'error'
        results['error'] = str(e)

    return results


def cleanup_filesystem(dry_run: bool = False) -> dict:
    """
    Clean up filesystem directories.

    Removes:
    - Input files (uploaded documents)
    - Output files (converted markdown, images)
    """
    results = {
        'input_dir': {'status': 'pending', 'files_deleted': 0},
        'output_dir': {'status': 'pending', 'files_deleted': 0}
    }

    try:
        logger.info("Starting filesystem cleanup...")

        # Get directories from environment or use defaults
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_dir = os.environ.get('INPUT_DIR', os.path.join(base_dir, 'input'))
        output_dir = os.environ.get('OUTPUT_DIR', os.path.join(base_dir, 'output'))

        # Count files in input directory
        if os.path.exists(input_dir):
            input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
            results['input_dir']['before'] = len(input_files)
            logger.info(f"  Input directory: {len(input_files)} files")
        else:
            results['input_dir']['before'] = 0
            logger.info(f"  Input directory does not exist: {input_dir}")

        # Count files/folders in output directory
        if os.path.exists(output_dir):
            output_items = os.listdir(output_dir)
            results['output_dir']['before'] = len(output_items)
            logger.info(f"  Output directory: {len(output_items)} items")
        else:
            results['output_dir']['before'] = 0
            logger.info(f"  Output directory does not exist: {output_dir}")

        if dry_run:
            logger.info("DRY RUN - No changes made to filesystem")
            return results

        # Clean input directory
        if os.path.exists(input_dir):
            logger.info(f"Cleaning input directory: {input_dir}")
            for item in os.listdir(input_dir):
                item_path = os.path.join(input_dir, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        results['input_dir']['files_deleted'] += 1
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        results['input_dir']['files_deleted'] += 1
                except Exception as e:
                    logger.warning(f"  Could not delete {item_path}: {e}")
            results['input_dir']['status'] = 'cleared'
            logger.info(f"  Deleted {results['input_dir']['files_deleted']} items from input directory")

        # Clean output directory
        if os.path.exists(output_dir):
            logger.info(f"Cleaning output directory: {output_dir}")
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        results['output_dir']['files_deleted'] += 1
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        results['output_dir']['files_deleted'] += 1
                except Exception as e:
                    logger.warning(f"  Could not delete {item_path}: {e}")
            results['output_dir']['status'] = 'cleared'
            logger.info(f"  Deleted {results['output_dir']['files_deleted']} items from output directory")

        logger.info("Filesystem cleanup completed")

    except Exception as e:
        logger.error(f"Filesystem cleanup failed: {e}")
        results['status'] = 'error'
        results['error'] = str(e)

    return results


def run_cleanup(dry_run: bool = False, skip_filesystem: bool = False) -> dict:
    """
    Run complete cleanup of all data stores.

    Args:
        dry_run: If True, only show what would be deleted without making changes
        skip_filesystem: If True, skip cleaning up files on disk

    Returns:
        Dictionary with cleanup results for each data store
    """
    logger.info("=" * 60)
    logger.info("DATABASE CLEANUP SCRIPT")
    logger.info("=" * 60)

    if dry_run:
        logger.info("MODE: DRY RUN (no changes will be made)")
    else:
        logger.info("MODE: LIVE (data will be permanently deleted!)")

    logger.info("=" * 60)

    results = {
        'postgresql': {},
        'qdrant': {},
        'neo4j': {},
        'filesystem': {}
    }

    # 1. Clean PostgreSQL
    try:
        from db.database import get_db_session

        with get_db_session() as db:
            results['postgresql'] = cleanup_postgresql(db, dry_run)
    except Exception as e:
        logger.error(f"PostgreSQL cleanup error: {e}")
        results['postgresql'] = {'status': 'error', 'error': str(e)}

    # 2. Clean Qdrant Vector Database
    results['qdrant'] = cleanup_qdrant(dry_run)

    # 3. Clean Neo4j Graph Database
    results['neo4j'] = cleanup_neo4j(dry_run)

    # 4. Clean Filesystem (optional)
    if not skip_filesystem:
        results['filesystem'] = cleanup_filesystem(dry_run)
    else:
        results['filesystem'] = {'status': 'skipped', 'reason': 'skip_filesystem flag set'}
        logger.info("Filesystem cleanup skipped")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("CLEANUP SUMMARY")
    logger.info("=" * 60)

    for store, store_results in results.items():
        logger.info(f"\n{store.upper()}:")
        if isinstance(store_results, dict):
            for key, value in store_results.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}: {value}")
                else:
                    logger.info(f"  {key}: {value}")

    if dry_run:
        logger.info("\n" + "=" * 60)
        logger.info("DRY RUN COMPLETE - No changes were made")
        logger.info("Run without --dry-run to actually delete data")
        logger.info("=" * 60)
    else:
        logger.info("\n" + "=" * 60)
        logger.info("CLEANUP COMPLETE - All data except users has been deleted")
        logger.info("=" * 60)

    return results


def main():
    """Main entry point with command line argument handling."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Clean up all application data except user records.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show what would be deleted (dry run)
  python cleanup_all_data.py --dry-run

  # Actually delete all data
  python cleanup_all_data.py

  # Delete all data except files on disk
  python cleanup_all_data.py --skip-filesystem

  # Force deletion without confirmation
  python cleanup_all_data.py --force
        """
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without making changes'
    )

    parser.add_argument(
        '--skip-filesystem',
        action='store_true',
        help='Skip cleaning up files on disk (input/output directories)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompt'
    )

    args = parser.parse_args()

    # Confirmation prompt (unless --force or --dry-run)
    if not args.dry_run and not args.force:
        print("\n" + "!" * 60)
        print("WARNING: This will permanently delete ALL data except users!")
        print("!" * 60)
        print("\nThe following will be deleted:")
        print("  - All documents and their metadata")
        print("  - All workspaces")
        print("  - All chat sessions and messages")
        print("  - All vector embeddings (Qdrant)")
        print("  - All graph data (Neo4j)")
        if not args.skip_filesystem:
            print("  - All uploaded files and converted outputs")
        print("\nUsers will be preserved but logged out (refresh tokens cleared).")
        print("")

        confirm = input("Type 'DELETE ALL' to confirm: ")
        if confirm != 'DELETE ALL':
            print("Aborted. No changes made.")
            sys.exit(0)

    # Run cleanup
    results = run_cleanup(
        dry_run=args.dry_run,
        skip_filesystem=args.skip_filesystem
    )

    # Exit with appropriate code
    has_errors = any(
        isinstance(r, dict) and r.get('status') == 'error'
        for r in results.values()
    )

    sys.exit(1 if has_errors else 0)


if __name__ == '__main__':
    main()
