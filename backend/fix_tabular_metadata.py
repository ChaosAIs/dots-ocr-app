"""
Fix script to add metadata embeddings and summary chunks for existing tabular documents.
This ensures CSV/Excel files have proper metadata for document routing.
"""
import sys
sys.path.insert(0, '/home/fy/MyWorkPlace/dots-ocr-app/backend')

from dotenv import load_dotenv
load_dotenv('/home/fy/MyWorkPlace/dots-ocr-app/backend/.env')

from db.database import get_db_session
from db.models import Document, DocumentData
from rag_service.vectorstore import get_vectorstore
from langchain_core.documents import Document as LangchainDocument
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_tabular_metadata():
    """Add metadata embeddings for tabular documents that are missing them."""

    with get_db_session() as db:
        # Find all tabular documents
        tabular_docs = db.query(Document).filter(
            Document.is_tabular_data == True,
            Document.index_status == 'indexed'
        ).all()

        logger.info(f"Found {len(tabular_docs)} indexed tabular documents")

        for doc in tabular_docs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {doc.original_filename} (ID: {doc.id})")

            # Get document data
            doc_data = db.query(DocumentData).filter(
                DocumentData.document_id == doc.id
            ).first()

            if not doc_data:
                logger.warning(f"  No document_data found, skipping")
                continue

            # Build metadata with all necessary fields for tabular documents
            source_name = doc.original_filename.rsplit('.', 1)[0] if '.' in doc.original_filename else doc.original_filename
            columns = doc_data.header_data.get("column_headers", []) if doc_data.header_data else []

            # Generate a summary based on the data
            summary = f"Tabular dataset '{source_name}' containing {doc_data.line_items_count} records with columns: {', '.join(columns)}."

            # Infer document types from column content
            inferred_types = _infer_document_types_from_columns(columns)
            if not inferred_types:
                inferred_types = [doc_data.schema_type or "spreadsheet"]

            metadata = {
                "document_types": inferred_types,
                "subject_name": source_name,
                "schema_type": doc_data.schema_type,
                "is_tabular": True,
                "row_count": doc_data.line_items_count,
                "column_count": len(columns),
                "columns": columns,
                "confidence": 0.95,
                "summary": summary,
                "topics": _infer_topics_from_columns(columns),
            }

            if doc.workspace_id:
                metadata["workspace_id"] = str(doc.workspace_id)

            logger.info(f"  Document types: {metadata['document_types']}")
            logger.info(f"  Columns: {columns}")
            logger.info(f"  Row count: {doc_data.line_items_count}")
            logger.info(f"  Topics: {metadata['topics']}")

            # Create comprehensive summary chunk in the documents collection for RAG search
            # NOTE: Separate 'metadatas' collection removed - document routing now uses
            # summary chunks in 'documents' collection directly.
            logger.info(f"  Creating comprehensive summary chunk in documents collection...")
            chunk_ids = _create_summary_chunks(doc, doc_data, source_name, columns, metadata['topics'])
            if chunk_ids:
                logger.info(f"  ✅ Created comprehensive summary chunk")
                # Update document with chunk IDs
                doc.summary_chunk_ids = chunk_ids
                db.commit()
            else:
                logger.warning(f"  ⚠️ Failed to create summary chunks")

        logger.info(f"\n{'='*60}")
        logger.info(f"Done! Processed {len(tabular_docs)} tabular documents")


def _create_summary_chunks(doc: Document, doc_data: DocumentData, source_name: str, columns: list, topics: list) -> list:
    """Create a single comprehensive summary chunk for a tabular document.

    This generates ONE chunk that combines document overview, schema info,
    and context - replacing the previous multi-chunk approach.

    Benefits:
    - Single chunk = no duplicate search results
    - All relevant info in one place for RAG
    - Simpler architecture
    """
    try:
        document_id = str(doc.id)
        workspace_id = str(doc.workspace_id) if doc.workspace_id else None

        # Metadata for the comprehensive chunk
        base_metadata = {
            "document_id": document_id,
            "source": source_name,
            "filename": doc.original_filename,
            "is_tabular": True,
            "schema_type": doc_data.schema_type or "spreadsheet",
            "row_count": doc_data.line_items_count,
            "column_count": len(columns),
            "columns": columns,
            "chunk_type": "tabular_summary",
            "chunk_id": f"{document_id}_summary",
        }
        if workspace_id:
            base_metadata["workspace_id"] = workspace_id

        # Single comprehensive chunk combining summary, schema, and context
        comprehensive_content = f"""# {source_name}

## Document Overview
This is a tabular dataset ({doc_data.schema_type or 'spreadsheet'}) containing {doc_data.line_items_count} records with {len(columns)} columns.

Topics: {', '.join(topics)}

## Data Schema
Column names: {', '.join(columns)}

This dataset contains structured data with the following fields:
{chr(10).join([f'- {col}' for col in columns])}

## Query Capabilities
Data can be queried for:
- Finding maximum and minimum values
- Filtering by category, brand, or other fields
- Aggregating totals and counts
- Searching for specific products or items"""

        chunk = LangchainDocument(
            page_content=comprehensive_content,
            metadata=base_metadata
        )

        # Delete existing chunks for this document to prevent duplicates
        from rag_service.vectorstore import delete_documents_by_document_id
        deleted_count = delete_documents_by_document_id(document_id)
        if deleted_count > 0:
            logger.info(f"  Deleted {deleted_count} existing chunks before re-indexing")

        # Index chunk to Qdrant
        vectorstore = get_vectorstore()
        vectorstore.add_documents([chunk])

        return [chunk.metadata["chunk_id"]]

    except Exception as e:
        logger.error(f"Failed to create summary chunk: {e}", exc_info=True)
        return []


def _infer_topics_from_columns(columns: list) -> list:
    """Infer topics from column names for better semantic matching."""
    topics = []
    column_lower = [c.lower() for c in columns]

    # Inventory/Product related
    if any(kw in col for col in column_lower for kw in ['product', 'item', 'sku', 'inventory', 'stock']):
        topics.extend(['inventory management', 'product data', 'stock tracking'])
    if any(kw in col for col in column_lower for kw in ['price', 'cost', 'amount', 'total']):
        topics.append('pricing data')
    if any(kw in col for col in column_lower for kw in ['quantity', 'qty', 'count', 'stock']):
        topics.append('quantity tracking')
    if any(kw in col for col in column_lower for kw in ['category', 'type', 'brand']):
        topics.append('product categorization')

    # Sales related
    if any(kw in col for col in column_lower for kw in ['sale', 'order', 'transaction', 'revenue']):
        topics.extend(['sales data', 'transactions'])

    # Customer related
    if any(kw in col for col in column_lower for kw in ['customer', 'client', 'buyer']):
        topics.append('customer data')

    # Date related
    if any(kw in col for col in column_lower for kw in ['date', 'time', 'created', 'updated']):
        topics.append('time-series data')

    # Default if no specific topics found
    if not topics:
        topics = ['tabular data', 'spreadsheet']

    return list(set(topics))[:5]  # Unique topics, max 5


def _infer_document_types_from_columns(columns: list) -> list:
    """Infer document types from column names."""
    types = []
    columns_text = " ".join([c.lower() for c in columns])

    # Inventory/Product related
    if any(kw in columns_text for kw in ['product', 'inventory', 'stock', 'sku', 'quantity']):
        types.append('inventory_report')

    # Financial related
    if any(kw in columns_text for kw in ['amount', 'total', 'payment', 'invoice', 'transaction', 'balance']):
        types.append('financial_report')

    # Sales related
    if any(kw in columns_text for kw in ['sale', 'order', 'revenue', 'customer']):
        types.append('sales_report')

    # Always include spreadsheet for tabular data
    types.append('spreadsheet')

    return types


if __name__ == "__main__":
    fix_tabular_metadata()
