"""
Test the chat orchestrator analytics flow with dynamic SQL generation.
"""

import os
import sys
from urllib.parse import quote_plus
from uuid import UUID

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from db.models import DocumentData
from chat_service.chat_orchestrator import ChatOrchestrator, IntentClassification


def get_db_session():
    """Create database session."""
    password = quote_plus("FyUbuntu@2025Ai")
    DATABASE_URL = f"postgresql://postgres:{password}@localhost:6400/dots_ocr"
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def test_orchestrator_analytics():
    """Test the orchestrator analytics query execution."""
    print("\n" + "=" * 80)
    print(" ORCHESTRATOR ANALYTICS TEST")
    print("=" * 80)

    db = get_db_session()

    try:
        # Get all document IDs
        doc_ids = [d.document_id for d in db.query(DocumentData).all()]
        print(f"\n1. Found {len(doc_ids)} documents with extracted data")

        # Create orchestrator
        orchestrator = ChatOrchestrator(db)

        # Create a mock classification
        classification = IntentClassification(
            intent="data_analytics",
            confidence=0.9,
            reasoning="User wants to aggregate purchase data",
            requires_extracted_data=True,
            suggested_schemas=["spreadsheet"],
            detected_entities=[],
            detected_metrics=["total_amount"],
            detected_time_range=None  # No time range filter
        )

        # Test query
        query = "summarize the purchase order amount group by categories and year"

        print(f"\n2. Executing analytics query...")
        print(f"   Query: '{query}'")

        result = orchestrator.execute_analytics_query(
            query=query,
            classification=classification,
            accessible_doc_ids=doc_ids
        )

        print(f"\n3. Results:")
        data = result.get('data', [])
        summary = result.get('summary', {})
        metadata = result.get('metadata', {})

        print(f"   - Data rows: {len(data)}")
        print(f"   - Grand total: ${summary.get('grand_total', 0):,.2f}")
        print(f"   - Total records: {summary.get('total_records', 0)}")

        if 'hierarchical_data' in summary:
            print(f"\n   Hierarchical data:")
            for group in summary.get('hierarchical_data', [])[:3]:
                group_name = group.get('group_name', 'Unknown')
                group_total = group.get('group_total', 0) or 0
                print(f"     {group_name}: ${group_total:,.2f}")
                for sub in group.get('sub_groups', [])[:3]:
                    name = sub.get('name', 'Unknown')
                    total = sub.get('total', 0) or 0
                    count = sub.get('count', 0)
                    print(f"       └─ {name}: ${total:,.2f} ({count} items)")

        if 'summary_by_category' in summary:
            print(f"\n   Summary by category:")
            for cat, total in sorted(summary.get('summary_by_category', {}).items()):
                if total is not None:
                    print(f"     - {cat}: ${float(total):,.2f}")

        print(f"\n   Generated SQL explanation: {metadata.get('explanation', 'N/A')}")

        # Test formatting
        print(f"\n4. Formatted response:")
        print("-" * 60)
        formatted = orchestrator.format_analytics_response(
            query=query,
            analytics_result=result,
            classification=classification
        )
        print(formatted[:2000])  # Limit output
        print("-" * 60)

        print(f"\n5. Test PASSED" if len(data) > 0 else "\n5. Test FAILED - No data returned")

        return len(data) > 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        db.close()


if __name__ == "__main__":
    success = test_orchestrator_analytics()
    sys.exit(0 if success else 1)
