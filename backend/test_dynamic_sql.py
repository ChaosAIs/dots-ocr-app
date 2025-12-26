"""
Test script for LLM-based Dynamic SQL Generation

This script tests:
1. Dynamic SQL generation based on user query and data schema
2. Execution of generated SQL against the database
3. Hierarchical summary report generation
"""

import os
import sys
import json
from urllib.parse import quote_plus
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import DocumentData, Document
from analytics_service.sql_query_executor import SQLQueryExecutor
from analytics_service.llm_sql_generator import LLMSQLGenerator, generate_analytics_sql


def get_db_session():
    """Create database session."""
    password = quote_plus("FyUbuntu@2025Ai")
    DATABASE_URL = f"postgresql://postgres:{password}@localhost:6400/dots_ocr"
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def decimal_to_float(obj):
    """Convert Decimal to float for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def test_dynamic_sql_generation():
    """Test dynamic SQL generation with various queries."""
    print("\n" + "=" * 80)
    print(" DYNAMIC SQL GENERATION TEST")
    print("=" * 80)

    db = get_db_session()

    try:
        # Get field mappings from documents - specifically look for documents with field_mappings
        print("\n1. Getting field mappings from database...")

        # Use raw SQL to find document with field_mappings
        from sqlalchemy import text
        result = db.execute(text("""
            SELECT dd.document_id, dd.header_data->'field_mappings' as field_mappings
            FROM documents_data dd
            WHERE dd.header_data->'field_mappings' IS NOT NULL
            LIMIT 1
        """))
        row = result.fetchone()

        if not row or not row[1]:
            print("   ✗ No field mappings found")
            return False

        field_mappings = dict(row[1]) if row[1] else {}
        print(f"   ✓ Found {len(field_mappings)} field mappings")

        for field, mapping in list(field_mappings.items())[:5]:
            print(f"      - {field}: {mapping.get('semantic_type')} ({mapping.get('data_type')})")

        # Test queries
        test_queries = [
            "Summarize total sales by category",
            "Show sales by year and category",
            "Total sales grouped by year then by category",
            "Monthly sales breakdown by category",
            "Total quantity and sales by region",
        ]

        print("\n2. Testing SQL generation (heuristic mode - no LLM)...")
        generator = LLMSQLGenerator(llm_client=None)

        for query in test_queries:
            print(f"\n   Query: \"{query}\"")
            print("   " + "-" * 70)

            result = generator.generate_sql(query, field_mappings)

            print(f"   Explanation: {result.explanation}")
            print(f"   Time granularity: {result.time_granularity}")
            print(f"   Grouping fields: {result.grouping_fields}")
            print(f"   Success: {result.success}")
            print(f"\n   Generated SQL:")
            for line in result.sql_query.split('\n'):
                print(f"      {line}")

        # Test execution of generated SQL
        print("\n" + "=" * 80)
        print(" TESTING SQL EXECUTION")
        print("=" * 80)

        # Get all document IDs
        doc_ids = [d.document_id for d in db.query(DocumentData).all()]

        executor = SQLQueryExecutor(db)

        for query in test_queries[:3]:  # Test first 3 queries
            print(f"\n   Query: \"{query}\"")
            print("   " + "-" * 70)

            result = executor.execute_dynamic_sql_query(
                accessible_doc_ids=doc_ids,
                query=query,
                llm_client=None  # Use heuristic mode
            )

            if result.get('data'):
                print(f"   ✓ Returned {len(result['data'])} rows")

                # Show sample data
                print("\n   Sample results:")
                for row in result['data'][:5]:
                    formatted = {}
                    for k, v in row.items():
                        if k == 'total_amount' and v is not None:
                            formatted[k] = f"${float(v):,.2f}"
                        else:
                            formatted[k] = v
                    print(f"      {formatted}")

                # Show summary
                summary = result.get('summary', {})
                print(f"\n   Grand Total: ${summary.get('grand_total', 0):,.2f}")
                print(f"   Total Records: {summary.get('total_records', 0)}")

                # Show hierarchical data if available
                if 'hierarchical_data' in summary:
                    print("\n   Hierarchical Summary:")
                    for group in summary['hierarchical_data'][:3]:
                        print(f"      {group.get('group_name')}: ${group.get('group_total', 0):,.2f}")
                        if 'sub_groups' in group:
                            for sub in group.get('sub_groups', [])[:3]:
                                print(f"         └─ {sub.get('name')}: ${sub.get('total', 0):,.2f}")
            else:
                print(f"   ✗ No data returned")
                if 'error' in result.get('summary', {}):
                    print(f"   Error: {result['summary']['error']}")

            # Show generated SQL
            print(f"\n   Generated SQL:")
            sql = result.get('metadata', {}).get('generated_sql', 'N/A')
            for line in sql.split('\n')[:10]:
                print(f"      {line}")

        print("\n" + "=" * 80)
        print(" TEST COMPLETE")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        db.close()


def test_year_category_aggregation():
    """Test specific year-category aggregation query."""
    print("\n" + "=" * 80)
    print(" YEAR-CATEGORY AGGREGATION TEST")
    print("=" * 80)

    db = get_db_session()

    try:
        # Get field mappings using raw SQL
        from sqlalchemy import text
        result = db.execute(text("""
            SELECT dd.document_id, dd.header_data->'field_mappings' as field_mappings
            FROM documents_data dd
            WHERE dd.header_data->'field_mappings' IS NOT NULL
            LIMIT 1
        """))
        row = result.fetchone()

        if not row or not row[1]:
            print("   ✗ No field mappings found")
            return False

        field_mappings = dict(row[1]) if row[1] else {}
        doc_ids = [d.document_id for d in db.query(DocumentData).all()]

        executor = SQLQueryExecutor(db)

        # Test the specific query from user
        query = "summarize the purchase order amount group by categories and year, list the details amount by year then by categories under year"

        print(f"\n   Query: \"{query}\"")
        print("   " + "-" * 70)

        result = executor.execute_dynamic_sql_query(
            accessible_doc_ids=doc_ids,
            query=query,
            llm_client=None
        )

        if result.get('data'):
            print(f"\n   ✓ Query returned {len(result['data'])} rows")

            # Print full results
            print("\n   Complete Results:")
            print("   " + "-" * 70)

            # Group by year for display
            by_year = {}
            for row in result['data']:
                year = row.get('year', 'Unknown')
                if year is None:
                    year = 'Unknown'
                if year not in by_year:
                    by_year[year] = []
                by_year[year].append(row)

            for year in sorted(by_year.keys(), key=lambda x: str(x)):
                year_rows = by_year[year]
                year_total = sum(float(r.get('total_amount', 0) or 0) for r in year_rows)
                print(f"\n   {year}")
                print(f"   Total Purchase Amount: ${year_total:,.2f}")

                for row in sorted(year_rows, key=lambda x: x.get('category', '')):
                    category = row.get('category', 'Unknown')
                    amount = float(row.get('total_amount', 0) or 0)
                    count = row.get('item_count', 0)
                    print(f"      {category}: ${amount:,.2f} ({count} items)")

            # Summary
            summary = result.get('summary', {})
            print("\n   " + "-" * 70)
            print(f"   GRAND TOTAL: ${summary.get('grand_total', 0):,.2f}")
            print(f"   TOTAL RECORDS: {summary.get('total_records', 0)}")

            # Summary by category
            if 'summary_by_category' in summary:
                print("\n   Summary by Category:")
                for cat, total in sorted(summary['summary_by_category'].items()):
                    print(f"      {cat}: ${total:,.2f}")

            # Summary by year
            if 'summary_by_year' in summary:
                print("\n   Summary by Year:")
                for year, total in sorted(summary['summary_by_year'].items()):
                    print(f"      {year}: ${total:,.2f}")

        else:
            print(f"   ✗ No data returned")
            if result.get('metadata', {}).get('error'):
                print(f"   Error: {result['metadata']['error']}")

        print("\n   Generated SQL:")
        sql = result.get('metadata', {}).get('generated_sql', 'N/A')
        print(f"   {sql}")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        db.close()


if __name__ == "__main__":
    test_dynamic_sql_generation()
    test_year_category_aggregation()
