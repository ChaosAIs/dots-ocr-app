"""
End-to-End Analytics Test with Database

This script tests the complete flow:
1. Retrieve documents with extracted data from the database
2. Analyze field mappings from stored data
3. Execute queries using the LLM-based query analyzer
4. Verify aggregation results
"""

import os
import sys
from urllib.parse import quote_plus

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from uuid import UUID

from db.models import DocumentData, Document, Base
from analytics_service.sql_query_executor import SQLQueryExecutor
from analytics_service.schema_analyzer import analyze_user_query


def get_db_session():
    """Create database session."""
    password = quote_plus("FyUbuntu@2025Ai")
    # PostgreSQL is running in Docker on port 6400 (as per .env configuration)
    DATABASE_URL = f"postgresql://postgres:{password}@localhost:6400/dots_ocr"
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def test_e2e_analytics():
    """Test end-to-end analytics flow."""
    print("\n" + "=" * 70)
    print(" END-TO-END ANALYTICS TEST")
    print("=" * 70)

    db = get_db_session()

    try:
        # Step 1: Get all documents with extracted data
        print("\n1. Retrieving documents with extracted data...")
        doc_data_records = db.query(DocumentData).all()

        if not doc_data_records:
            print("   ✗ No extracted data found in documents_data table")
            return False

        print(f"   ✓ Found {len(doc_data_records)} document data records")

        # Step 2: Check for documents with field mappings
        print("\n2. Checking for documents with field mappings...")
        docs_with_mappings = []
        for doc_data in doc_data_records:
            header_data = doc_data.header_data or {}
            if 'field_mappings' in header_data:
                docs_with_mappings.append(doc_data)
                print(f"   ✓ Document {str(doc_data.document_id)[:8]}... has field mappings")
                # Show sample mappings
                mappings = header_data['field_mappings']
                for col, mapping in list(mappings.items())[:3]:
                    print(f"      - {col}: {mapping.get('semantic_type')} ({mapping.get('data_type')})")
                if len(mappings) > 3:
                    print(f"      ... and {len(mappings) - 3} more fields")

        if not docs_with_mappings:
            print("   ! No documents have field mappings (may need re-extraction)")
            print("   Checking for line_items data instead...")

            # Check if we have line items
            docs_with_items = [d for d in doc_data_records if d.line_items and len(d.line_items) > 0]
            if docs_with_items:
                print(f"   ✓ Found {len(docs_with_items)} documents with line items")
                for doc_data in docs_with_items[:2]:
                    items = doc_data.line_items
                    if items and isinstance(items[0], dict):
                        print(f"   Sample columns: {list(items[0].keys())}")

        # Step 3: Get accessible document IDs
        print("\n3. Getting document IDs for query...")
        doc_ids = [d.document_id for d in doc_data_records]
        print(f"   ✓ {len(doc_ids)} documents available for querying")

        # Step 4: Initialize query executor
        print("\n4. Testing SQL query executor...")
        executor = SQLQueryExecutor(db)

        # Test basic query
        print("\n   4a. Testing basic analytics query (group by category)...")
        result = executor.execute_analytics_query(
            accessible_doc_ids=doc_ids,
            schema_types=None,
            metrics=['total_amount', 'count'],
            group_by='category'
        )

        if result.get('data'):
            print(f"   ✓ Query returned {len(result['data'])} groups")
            for row in result['data'][:5]:
                print(f"      - {row.get('group', 'Unknown')}: ${row.get('total_amount', 0):,.2f}")
            summary = result.get('summary', {})
            print(f"   Summary: Total={summary.get('total_amount', 0):,.2f}, Records={summary.get('total_records', 0)}")
        else:
            print("   ! No data returned from category grouping")

        # Test natural language query
        print("\n   4b. Testing natural language query...")
        intent = {
            'detected_metrics': ['total_amount', 'count'],
            'suggested_schemas': []
        }
        nl_result = executor.execute_natural_language_query(
            accessible_doc_ids=doc_ids,
            intent_classification=intent,
            query="Show me total sales by category and year"
        )

        if nl_result.get('data'):
            print(f"   ✓ NL Query returned {len(nl_result['data'])} results")
            for row in nl_result['data'][:3]:
                if 'sub_groups' in row:
                    print(f"      - {row.get('group')}: ${row.get('total_amount', 0):,.2f}")
                    for sub in row.get('sub_groups', [])[:2]:
                        print(f"        └─ {sub.get('group')}: ${sub.get('total_amount', 0):,.2f}")
                else:
                    print(f"      - {row.get('group', 'Unknown')}: ${row.get('total_amount', 0):,.2f}")
        else:
            print("   ! No data returned from natural language query")

        # Step 5: Test LLM query analysis (heuristic mode)
        print("\n5. Testing query analysis (heuristic mode)...")

        # Get field mappings from a document
        field_mappings = {}
        for doc_data in doc_data_records:
            header_data = doc_data.header_data or {}
            if 'field_mappings' in header_data:
                field_mappings = header_data['field_mappings']
                break

        if field_mappings:
            test_queries = [
                "Total sales by category",
                "Sales breakdown by year and region",
                "Average order value by customer",
                "Monthly revenue trend"
            ]

            for query in test_queries:
                analysis = analyze_user_query(query, field_mappings, llm_client=None)
                print(f"\n   Query: \"{query}\"")
                print(f"   → aggregation: {analysis['aggregation_type']}")
                print(f"   → group_by: {analysis['group_by_fields']}")
                print(f"   → time: {analysis['time_grouping']}")
                print(f"   → explanation: {analysis['explanation']}")
        else:
            print("   ! No field mappings available for query analysis")

        # Step 6: Summary
        print("\n" + "=" * 70)
        print(" TEST SUMMARY")
        print("=" * 70)
        print(f"   Documents with data: {len(doc_data_records)}")
        print(f"   Documents with field mappings: {len(docs_with_mappings)}")
        print(f"   Query execution: {'✓' if result.get('data') else '✗'}")
        print(f"   Natural language query: {'✓' if nl_result.get('data') else '✗'}")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        db.close()


if __name__ == "__main__":
    success = test_e2e_analytics()
    sys.exit(0 if success else 1)
