#!/usr/bin/env python3
"""
Migration script to embed existing document metadata to the vector collection.

This script:
1. Queries all documents with document_metadata from PostgreSQL
2. Generates embeddings for each document's metadata
3. Upserts to the document_metadatas Qdrant collection

Usage:
    cd backend
    python -m scripts.migrate_metadata_embeddings

Options:
    --dry-run    Show what would be migrated without making changes
    --force      Re-embed all documents, even if already in collection
"""

import os
import sys
import argparse
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db_session
from db.document_repository import DocumentRepository
from rag_service.vectorstore import (
    upsert_document_metadata_embedding,
    get_metadata_collection_info,
    ensure_metadata_collection_exists,
    get_qdrant_client,
    METADATA_COLLECTION_NAME
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_existing_document_ids() -> set:
    """Get set of document IDs already in the metadata collection."""
    try:
        client = get_qdrant_client()

        # Scroll through all points to get their IDs
        existing_ids = set()
        offset = None

        while True:
            result, offset = client.scroll(
                collection_name=METADATA_COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=False
            )

            for point in result:
                existing_ids.add(str(point.id))

            if offset is None:
                break

        return existing_ids
    except Exception as e:
        logger.warning(f"Could not get existing document IDs: {e}")
        return set()


def migrate_metadata_embeddings(dry_run: bool = False, force: bool = False):
    """
    Migrate existing document metadata to the vector collection.

    Args:
        dry_run: If True, only show what would be done without making changes
        force: If True, re-embed all documents even if already in collection
    """
    logger.info("=" * 60)
    logger.info("Document Metadata Embedding Migration")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    # Ensure collection exists
    if not dry_run:
        ensure_metadata_collection_exists()

    # Get collection info
    collection_info = get_metadata_collection_info()
    logger.info(f"Collection status: {collection_info}")

    # Get existing document IDs in collection (for skip logic)
    existing_ids = set() if force else get_existing_document_ids()
    logger.info(f"Found {len(existing_ids)} documents already in metadata collection")

    # Get all documents with metadata from PostgreSQL
    with get_db_session() as db:
        repo = DocumentRepository(db)
        docs = repo.get_all_with_metadata()

        total_docs = len(docs)
        docs_with_metadata = [d for d in docs if d.document_metadata]

        logger.info(f"Total documents in database: {total_docs}")
        logger.info(f"Documents with metadata: {len(docs_with_metadata)}")

        # Stats
        migrated = 0
        skipped = 0
        failed = 0

        for i, doc in enumerate(docs_with_metadata, 1):
            doc_id = str(doc.id)
            source_name = doc.filename.rsplit('.', 1)[0] if '.' in doc.filename else doc.filename

            # Check if already exists
            if doc_id in existing_ids and not force:
                logger.debug(f"[{i}/{len(docs_with_metadata)}] Skipping {source_name} (already in collection)")
                skipped += 1
                continue

            logger.info(
                f"[{i}/{len(docs_with_metadata)}] Migrating: {source_name} "
                f"(type={doc.document_metadata.get('document_type', 'unknown')})"
            )

            if dry_run:
                migrated += 1
                continue

            # Upsert to collection
            try:
                success = upsert_document_metadata_embedding(
                    document_id=doc_id,
                    source_name=source_name,
                    filename=doc.filename,
                    metadata=doc.document_metadata
                )

                if success:
                    migrated += 1
                else:
                    failed += 1
                    logger.warning(f"Failed to migrate {source_name}")

            except Exception as e:
                failed += 1
                logger.error(f"Error migrating {source_name}: {e}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Migration Summary")
    logger.info("=" * 60)
    logger.info(f"Total documents with metadata: {len(docs_with_metadata)}")
    logger.info(f"Migrated: {migrated}")
    logger.info(f"Skipped (already exists): {skipped}")
    logger.info(f"Failed: {failed}")

    if dry_run:
        logger.info("")
        logger.info("This was a dry run. Run without --dry-run to apply changes.")

    # Verify final state
    if not dry_run:
        final_info = get_metadata_collection_info()
        logger.info(f"Final collection state: {final_info}")

    return migrated, skipped, failed


def main():
    parser = argparse.ArgumentParser(
        description='Migrate document metadata to vector collection'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be migrated without making changes'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-embed all documents, even if already in collection'
    )

    args = parser.parse_args()

    try:
        migrate_metadata_embeddings(
            dry_run=args.dry_run,
            force=args.force
        )
    except KeyboardInterrupt:
        logger.info("\nMigration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
