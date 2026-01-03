#!/usr/bin/env python3
"""
Script to normalize existing line item field names in the database.

This script:
1. Fetches all line items from documents_data_line_items
2. For each document, gets the schema's field_mappings (with aliases)
3. Applies the FieldNormalizer to transform field names
4. Updates the database records

Usage:
    python scripts/normalize_existing_line_items.py [--dry-run]
"""

import sys
import os
import argparse
import json
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from psycopg2.extras import RealDictCursor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the normalizer
import importlib.util
spec = importlib.util.spec_from_file_location(
    "field_normalizer",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "backend/extraction_service/field_normalizer.py")
)
field_normalizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(field_normalizer_module)
FieldNormalizer = field_normalizer_module.FieldNormalizer


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 6400)),
        dbname=os.getenv('DB_NAME', 'dots_ocr'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'FyUbuntu@2025Ai')
    )


def get_schema_field_mappings(cursor):
    """Fetch all schema field mappings from data_schemas."""
    cursor.execute("""
        SELECT schema_type, field_mappings
        FROM data_schemas
        WHERE field_mappings IS NOT NULL
    """)
    schemas = {}
    for row in cursor.fetchall():
        schemas[row['schema_type']] = row['field_mappings']
    return schemas


def get_documents_with_line_items(cursor):
    """Get all documents with their schema type and line items."""
    cursor.execute("""
        SELECT
            d.id as document_id,
            d.filename,
            dd.id as documents_data_id,
            dd.schema_type,
            dli.id as line_item_id,
            dli.line_number,
            dli.data
        FROM documents d
        JOIN documents_data dd ON dd.document_id = d.id
        JOIN documents_data_line_items dli ON dli.documents_data_id = dd.id
        ORDER BY d.id, dli.line_number
    """)
    return cursor.fetchall()


def needs_normalization(data, line_item_fields):
    """Check if data has non-canonical field names that need normalization."""
    if not data or not line_item_fields:
        return False

    canonical_names = set(line_item_fields.keys())
    data_keys = set(data.keys()) - {'row_number', 'line_number', 'id', '_id'}

    # If any key is not a canonical name, it needs normalization
    for key in data_keys:
        if key not in canonical_names:
            return True
    return False


def normalize_data(data, line_item_fields, normalizer):
    """Normalize a single line item's data."""
    if not data or not line_item_fields:
        return data, {}

    # Wrap in list for normalize_line_items
    normalized_items, mapping = normalizer.normalize_line_items(
        [data],
        line_item_fields,
        use_llm=False  # Use alias matching only for consistency
    )

    return normalized_items[0] if normalized_items else data, mapping


def main():
    parser = argparse.ArgumentParser(description='Normalize existing line item field names')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be updated without making changes')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Starting line item field normalization")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        # Get schema field mappings
        logger.info("Fetching schema field mappings...")
        schema_mappings = get_schema_field_mappings(cursor)
        logger.info(f"Found {len(schema_mappings)} schemas with field mappings")

        # Get all line items
        logger.info("Fetching line items...")
        line_items = get_documents_with_line_items(cursor)
        logger.info(f"Found {len(line_items)} line items to process")

        # Initialize normalizer
        normalizer = FieldNormalizer()

        # Track statistics
        stats = {
            'processed': 0,
            'normalized': 0,
            'skipped_no_schema': 0,
            'skipped_already_normalized': 0,
            'errors': 0
        }

        updates = []
        current_doc = None

        for item in line_items:
            stats['processed'] += 1

            doc_id = item['document_id']
            schema_type = item['schema_type']
            line_item_id = item['line_item_id']
            data = item['data']

            # Log progress per document
            if current_doc != doc_id:
                current_doc = doc_id
                logger.info(f"\nProcessing document: {item['filename']} (schema: {schema_type})")

            # Get schema field mappings
            if schema_type not in schema_mappings:
                stats['skipped_no_schema'] += 1
                continue

            field_mappings = schema_mappings[schema_type]
            line_item_fields = field_mappings.get('line_item_fields', {})

            if not line_item_fields:
                stats['skipped_no_schema'] += 1
                continue

            # Check if normalization is needed
            if not needs_normalization(data, line_item_fields):
                stats['skipped_already_normalized'] += 1
                continue

            # Normalize the data
            try:
                normalized_data, mapping = normalize_data(data, line_item_fields, normalizer)

                if mapping:
                    logger.info(f"  Line {item['line_number']}: {list(data.keys())} -> {list(normalized_data.keys())}")
                    updates.append({
                        'id': line_item_id,
                        'old_data': data,
                        'new_data': normalized_data,
                        'mapping': mapping
                    })
                    stats['normalized'] += 1
                else:
                    stats['skipped_already_normalized'] += 1

            except Exception as e:
                logger.error(f"  Error normalizing line {item['line_number']}: {e}")
                stats['errors'] += 1

        # Apply updates
        logger.info("\n" + "=" * 60)
        logger.info(f"Summary: {len(updates)} line items need normalization")
        logger.info("=" * 60)

        if updates and not args.dry_run:
            logger.info("Applying updates...")
            update_cursor = conn.cursor()

            for i, update in enumerate(updates):
                update_cursor.execute(
                    "UPDATE documents_data_line_items SET data = %s WHERE id = %s",
                    (json.dumps(update['new_data']), str(update['id']))
                )

                if (i + 1) % 100 == 0:
                    logger.info(f"  Updated {i + 1}/{len(updates)} records...")

            conn.commit()
            logger.info(f"Successfully updated {len(updates)} records")
        elif args.dry_run and updates:
            logger.info("DRY RUN - Would update these records:")
            for update in updates[:10]:
                logger.info(f"  ID: {update['id']}")
                logger.info(f"    Old: {update['old_data']}")
                logger.info(f"    New: {update['new_data']}")
            if len(updates) > 10:
                logger.info(f"  ... and {len(updates) - 10} more")

        # Print final statistics
        logger.info("\n" + "=" * 60)
        logger.info("Final Statistics:")
        logger.info("=" * 60)
        logger.info(f"  Total processed: {stats['processed']}")
        logger.info(f"  Normalized: {stats['normalized']}")
        logger.info(f"  Already normalized: {stats['skipped_already_normalized']}")
        logger.info(f"  Skipped (no schema): {stats['skipped_no_schema']}")
        logger.info(f"  Errors: {stats['errors']}")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


if __name__ == '__main__':
    main()
