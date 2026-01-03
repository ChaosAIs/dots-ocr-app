"""
Tabular Data Extraction Service

Optimized workflow for dataset-style documents that:
1. Skips row-level chunking (no embedding of individual data rows)
2. Extracts structured data directly to PostgreSQL
3. Generates LLM-driven summary/metadata embeddings for document discovery
4. Creates 1-3 summary chunks for Qdrant (instead of many row chunks)

This service handles the "tabular" processing path for:
- CSV, Excel files
- Invoices, receipts with line items
- Bank statements, expense reports
- Any document with structured rows/columns
"""

import os
import json
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
from datetime import datetime

from sqlalchemy.orm import Session as DBSession

from db.database import get_db_session
from db.models import Document, DocumentData, IndexStatus
from extraction_service.extraction_service import ExtractionService
from common.document_type_classifier import TabularDataDetector, DocumentTypeClassifier
from rag_service.vectorstore import get_vectorstore, get_embeddings, upsert_document_metadata_embedding
from rag_service.llm_service import get_llm_service

from langchain_core.documents import Document as LangchainDocument

logger = logging.getLogger(__name__)


class TabularExtractionService:
    """
    Service for processing tabular/dataset-style documents.

    This service provides an optimized pathway that:
    - Skips row-level chunking (table rows don't benefit from semantic search)
    - KEEPS summary/metadata indexing (for document discovery)
    - Skips GraphRAG (entities already structured in rows)
    - Focuses on structured data extraction and schema analysis
    - Uses LLM-driven dynamic summary generation
    """

    def __init__(self, db: DBSession = None, llm_client=None):
        """
        Initialize the tabular extraction service.

        Args:
            db: SQLAlchemy database session
            llm_client: Optional LLM client for summary generation
        """
        self.db = db
        self.llm_client = llm_client
        self._llm_service = None

    def _get_llm(self):
        """Get LLM client for summary generation."""
        if self.llm_client:
            return self.llm_client
        if self._llm_service is None:
            try:
                self._llm_service = get_llm_service()
                logger.debug("[Tabular] LLM service initialized successfully")
            except Exception as e:
                logger.warning(f"[Tabular] LLM service not available: {e}")
                return None
        return self._llm_service

    def process_tabular_document(
        self,
        document_id: UUID,
        source_name: str,
        output_dir: str,
        filename: str,
        conversion_id: str = None,
        broadcast_callback=None
    ) -> Tuple[bool, str]:
        """
        Process a tabular document through the optimized pathway.

        Steps:
        1. Extract structured data (header + rows) to PostgreSQL
        2. Generate field mappings (schema analysis)
        3. Generate document summary using LLM
        4. Create summary/metadata embeddings for Qdrant (1-3 chunks)
        5. Update document status

        Args:
            document_id: UUID of the document
            source_name: Source name (filename without extension)
            output_dir: Output directory path
            filename: Original filename with extension
            conversion_id: For WebSocket notifications
            broadcast_callback: For progress updates

        Returns:
            Tuple of (success, message)
        """
        try:
            logger.info("=" * 80)
            logger.info("[Tabular] ========== TABULAR DOCUMENT PROCESSING START ==========")
            logger.info("=" * 80)
            logger.info(f"[Tabular] Document ID: {document_id}")
            logger.info(f"[Tabular] Filename: {filename}")
            logger.info(f"[Tabular] Source Name: {source_name}")
            logger.info(f"[Tabular] Output Dir: {output_dir}")
            logger.info(f"[Tabular] Conversion ID: {conversion_id}")
            logger.info("-" * 80)

            # Update status
            self._update_extraction_status(document_id, "processing")
            self._broadcast(broadcast_callback, document_id, {
                "status": "extracting",
                "message": "Extracting structured data...",
                "progress": 10
            })

            # ========== STEP 1: Extract structured data to PostgreSQL ==========
            logger.info("=" * 80)
            logger.info("[Tabular] ========== STEP 1: STRUCTURED DATA EXTRACTION ==========")
            logger.info("=" * 80)
            logger.info("[Tabular] Initializing ExtractionService...")

            extraction_service = ExtractionService(self.db, self._get_llm())
            logger.info("[Tabular] Calling extract_document()...")

            document_data = extraction_service.extract_document(
                document_id=document_id,
                force=False
            )

            if not document_data:
                logger.error("[Tabular] Extraction returned no data!")
                raise ValueError("Extraction returned no data")

            logger.info("-" * 80)
            logger.info(f"[Tabular] STEP 1 RESULT: Extraction successful")
            logger.info(f"[Tabular]   - Row count: {document_data.line_items_count}")
            logger.info(f"[Tabular]   - Schema type: {document_data.schema_type}")
            logger.info(f"[Tabular]   - Headers: {document_data.header_data.get('column_headers', [])[:5]}...")
            logger.info("-" * 80)

            self._broadcast(broadcast_callback, document_id, {
                "status": "extracting",
                "message": f"Extracted {document_data.line_items_count} rows",
                "progress": 40
            })

            # ========== STEP 2: Ensure field mappings exist ==========
            logger.info("=" * 80)
            logger.info("[Tabular] ========== STEP 2: FIELD MAPPING ANALYSIS ==========")
            logger.info("=" * 80)

            self._ensure_field_mappings(document_data)

            field_mappings = document_data.extraction_metadata.get("field_mappings", {}) if document_data.extraction_metadata else {}
            logger.info(f"[Tabular] Field mappings count: {len(field_mappings)}")
            logger.info(f"[Tabular] Field mappings type: {type(field_mappings)}")

            # Safely log field mappings (handle both dict and string values)
            for field_name, mapping in list(field_mappings.items())[:5]:
                if isinstance(mapping, dict):
                    logger.info(f"[Tabular]   - {field_name}: type={mapping.get('semantic_type')}, agg={mapping.get('aggregation')}")
                else:
                    # mapping might be a string or other type
                    logger.info(f"[Tabular]   - {field_name}: {mapping}")
            if len(field_mappings) > 5:
                logger.info(f"[Tabular]   ... and {len(field_mappings) - 5} more fields")
            logger.info("-" * 80)

            self._broadcast(broadcast_callback, document_id, {
                "status": "generating_summary",
                "message": "Generating document summary...",
                "progress": 50
            })

            # ========== STEP 3: Generate summary embeddings for discovery ==========
            logger.info("=" * 80)
            logger.info("[Tabular] ========== STEP 3: LLM SUMMARY GENERATION ==========")
            logger.info("=" * 80)
            logger.info("[Tabular] Generating 1-3 summary chunks for document discovery...")
            logger.info("[Tabular] This enables semantic search without row-level chunking")

            summary_chunks = self._generate_summary_chunks(
                document_id=document_id,
                document_data=document_data,
                source_name=source_name,
                filename=filename
            )

            logger.info("-" * 80)
            logger.info(f"[Tabular] STEP 3 RESULT: Generated {len(summary_chunks)} summary chunks")
            for i, chunk in enumerate(summary_chunks):
                chunk_type = chunk.metadata.get("chunk_type", "unknown")
                method = chunk.metadata.get("summary_method") or chunk.metadata.get("schema_method") or chunk.metadata.get("context_method", "unknown")
                content_preview = chunk.page_content[:100].replace('\n', ' ')
                logger.info(f"[Tabular]   Chunk {i+1}: type={chunk_type}, method={method}")
                logger.info(f"[Tabular]            preview: '{content_preview}...'")
            logger.info("-" * 80)

            self._broadcast(broadcast_callback, document_id, {
                "status": "indexing_summary",
                "message": "Indexing document summary...",
                "progress": 70
            })

            # ========== STEP 4: Index summary chunks to Qdrant ==========
            logger.info("=" * 80)
            logger.info("[Tabular] ========== STEP 4: QDRANT VECTOR INDEXING ==========")
            logger.info("=" * 80)
            logger.info(f"[Tabular] Indexing {len(summary_chunks)} summary chunks to Qdrant...")
            logger.info("[Tabular] NOTE: This is ~80% less storage than row-level chunking!")

            chunk_ids = self._index_summary_chunks(summary_chunks, document_id)

            logger.info("-" * 80)
            logger.info(f"[Tabular] STEP 4 RESULT: Indexed {len(chunk_ids)} chunks to Qdrant")
            for chunk_id in chunk_ids:
                logger.info(f"[Tabular]   - {chunk_id}")
            logger.info("-" * 80)

            # ========== STEP 4.5: Create Metadata Embedding for Document Routing ==========
            logger.info("[Tabular] ========== STEP 4.5: METADATA EMBEDDING ==========")
            logger.info("[Tabular] Creating metadata embedding for document routing...")

            try:
                columns = document_data.header_data.get("column_headers", [])

                # Get workspace_id from document
                doc = self.db.query(Document).filter(Document.id == document_id).first() if self.db else None
                workspace_id = str(doc.workspace_id) if doc and doc.workspace_id else None

                # Use LLM-based multi-type classification to determine all applicable document types
                llm_service = get_llm_service()
                classifier = DocumentTypeClassifier(llm_client=llm_service)

                # Get sample data for classification context (from external storage)
                sample_rows = self._fetch_sample_line_items(document_data.id, limit=3)
                data_preview = ""
                if sample_rows:
                    data_preview = "\n".join([str(row) for row in sample_rows])

                multi_type_result = classifier.classify_multi_type(
                    filename=filename,
                    columns=columns,
                    row_count=document_data.line_items_count,
                    data_preview=data_preview,
                    is_tabular=True
                )

                logger.info(f"[Tabular] Multi-type classification result:")
                logger.info(f"[Tabular]   - document_types: {multi_type_result.document_types}")
                logger.info(f"[Tabular]   - primary_type: {multi_type_result.primary_type}")
                logger.info(f"[Tabular]   - confidence: {multi_type_result.confidence}")
                logger.info(f"[Tabular]   - reasoning: {multi_type_result.reasoning}")

                # Build metadata for embedding with document_types list
                metadata = {
                    "document_types": multi_type_result.document_types,
                    "subject_name": source_name,
                    "schema_type": document_data.schema_type,
                    "is_tabular": True,
                    "row_count": document_data.line_items_count,
                    "column_count": len(columns),
                    "columns": columns,
                    "confidence": multi_type_result.confidence,
                }

                # Add workspace_id if available
                if workspace_id:
                    metadata["workspace_id"] = workspace_id

                # Use the first summary chunk content as the base for embedding
                if summary_chunks:
                    metadata["summary"] = summary_chunks[0].page_content[:500]

                success = upsert_document_metadata_embedding(
                    document_id=str(document_id),
                    source_name=source_name,
                    filename=filename,
                    metadata=metadata
                )

                if success:
                    logger.info(f"[Tabular] ✅ Metadata embedding created for document routing")
                    logger.info(f"[Tabular]   - Stored document_types: {multi_type_result.document_types}")
                else:
                    logger.warning(f"[Tabular] ⚠️ Failed to create metadata embedding")
            except Exception as meta_error:
                logger.warning(f"[Tabular] ⚠️ Metadata embedding error: {meta_error}", exc_info=True)
            logger.info("-" * 80)

            self._broadcast(broadcast_callback, document_id, {
                "status": "finalizing",
                "message": "Finalizing...",
                "progress": 90
            })

            # ========== STEP 5: Update final status ==========
            logger.info("=" * 80)
            logger.info("[Tabular] ========== STEP 5: STATUS UPDATE ==========")
            logger.info("=" * 80)

            self._update_extraction_status(document_id, "completed")
            logger.info("[Tabular] Updated extraction_status to 'completed'")

            self._update_index_status(document_id, IndexStatus.INDEXED, len(chunk_ids))
            logger.info(f"[Tabular] Updated index_status to 'INDEXED' with {len(chunk_ids)} chunks")

            self._save_summary_chunk_ids(document_id, chunk_ids)
            logger.info(f"[Tabular] Saved summary_chunk_ids to document record")
            logger.info("-" * 80)

            column_count = len(document_data.header_data.get("column_headers", []))
            self._broadcast(broadcast_callback, document_id, {
                "status": "completed",
                "message": f"Document ready ({document_data.line_items_count} rows, {len(chunk_ids)} index entries)",
                "progress": 100,
                "row_count": document_data.line_items_count,
                "column_count": column_count,
                "summary_chunks": len(chunk_ids)
            })

            logger.info("=" * 80)
            logger.info("[Tabular] ========== TABULAR DOCUMENT PROCESSING COMPLETE ==========")
            logger.info("=" * 80)
            logger.info(f"[Tabular] FINAL SUMMARY:")
            logger.info(f"[Tabular]   - Document: {filename}")
            logger.info(f"[Tabular]   - Rows extracted: {document_data.line_items_count}")
            logger.info(f"[Tabular]   - Columns: {column_count}")
            logger.info(f"[Tabular]   - Summary chunks indexed: {len(chunk_ids)}")
            logger.info(f"[Tabular]   - GraphRAG: SKIPPED (tabular data)")
            logger.info(f"[Tabular]   - Row-level chunking: SKIPPED (optimized pathway)")
            logger.info("=" * 80)

            return True, "Processing successful"

        except Exception as e:
            logger.error("=" * 80)
            logger.error("[Tabular] ========== TABULAR PROCESSING FAILED ==========")
            logger.error("=" * 80)
            logger.error(f"[Tabular] Document ID: {document_id}")
            logger.error(f"[Tabular] Filename: {filename}")
            logger.error(f"[Tabular] Error: {e}")
            logger.error("=" * 80, exc_info=True)

            self._update_extraction_status(document_id, "failed", error=str(e))
            self._broadcast(broadcast_callback, document_id, {
                "status": "error",
                "message": f"Processing failed: {str(e)}",
                "progress": 0
            })
            return False, str(e)

    def _fetch_sample_line_items(self, documents_data_id: UUID, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch sample line items from external storage (documents_data_line_items table).

        As of migration 022, all line items are stored externally.

        Args:
            documents_data_id: UUID of the documents_data record
            limit: Maximum number of rows to fetch (default 10)

        Returns:
            List of line item dictionaries
        """
        try:
            from db.models import DocumentDataLineItem

            if not self.db:
                logger.warning("[Tabular] No database session, cannot fetch line items")
                return []

            line_items = self.db.query(DocumentDataLineItem).filter(
                DocumentDataLineItem.documents_data_id == documents_data_id
            ).order_by(DocumentDataLineItem.line_number).limit(limit).all()

            logger.debug(f"[Tabular] Fetched {len(line_items)} sample line items from external storage")
            return [item.data for item in line_items]

        except Exception as e:
            logger.error(f"[Tabular] Failed to fetch line items from external storage: {e}")
            return []

    def _generate_summary_chunks(
        self,
        document_id: UUID,
        document_data: DocumentData,
        source_name: str,
        filename: str
    ) -> List[LangchainDocument]:
        """
        Generate 1-3 summary chunks for document discovery.

        These chunks enable semantic search to find relevant documents
        without embedding every row of data.

        Returns:
            List of LangchainDocument objects ready for embedding
        """
        logger.debug("[Tabular] _generate_summary_chunks() called")

        chunks = []
        # Get column headers (for spreadsheets) - may be empty for invoices
        column_headers = document_data.header_data.get("column_headers", [])
        row_count = document_data.line_items_count
        field_mappings = document_data.extraction_metadata.get("field_mappings", {}) if document_data.extraction_metadata else {}

        # Extract invoice/document header data (vendor, customer, dates, amounts, etc.)
        # These are the actual extracted values, not column names
        extracted_header_data = {k: v for k, v in document_data.header_data.items()
                                  if k not in ["column_headers", "_field_normalization"] and v is not None}

        # Get summary data (subtotal, total_amount, tax, etc.)
        summary_data = document_data.summary_data or {}

        logger.debug(f"[Tabular] Column headers: {column_headers}")
        logger.debug(f"[Tabular] Extracted header data: {extracted_header_data}")
        logger.debug(f"[Tabular] Summary data: {summary_data}")
        logger.debug(f"[Tabular] Row count: {row_count}")
        logger.debug(f"[Tabular] Field mappings: {len(field_mappings)} fields")

        # Get document for workspace_id
        doc = self.db.query(Document).filter(Document.id == document_id).first() if self.db else None
        workspace_id = str(doc.workspace_id) if doc and doc.workspace_id else None
        logger.debug(f"[Tabular] Workspace ID: {workspace_id}")

        # Combine column headers with extracted field names for comprehensive column list
        all_columns = list(column_headers) + list(extracted_header_data.keys()) + list(summary_data.keys())
        # Remove duplicates while preserving order
        seen = set()
        unique_columns = [x for x in all_columns if not (x in seen or seen.add(x))]

        # Common metadata for all chunks
        base_metadata = {
            "document_id": str(document_id),
            "source": source_name,
            "filename": filename,
            "is_tabular": True,
            "schema_type": document_data.schema_type,
            "row_count": row_count,
            "column_count": len(unique_columns),
            "columns": unique_columns,
            "chunk_type": "tabular_summary",  # Mark as tabular for filtering
        }
        if workspace_id:
            base_metadata["workspace_id"] = workspace_id

        # Add flattened header fields for structured filtering (e.g., filter by vendor_name)
        if extracted_header_data.get("invoice_number"):
            base_metadata["invoice_number"] = extracted_header_data["invoice_number"]
        if extracted_header_data.get("vendor_name"):
            base_metadata["vendor_name"] = extracted_header_data["vendor_name"]
        if extracted_header_data.get("customer_name"):
            base_metadata["customer_name"] = extracted_header_data["customer_name"]
        if extracted_header_data.get("invoice_date"):
            base_metadata["invoice_date"] = extracted_header_data["invoice_date"]
        if extracted_header_data.get("due_date"):
            base_metadata["due_date"] = extracted_header_data["due_date"]
        if extracted_header_data.get("currency"):
            base_metadata["currency"] = extracted_header_data["currency"]

        # Add flattened summary fields for numeric filtering (e.g., filter by total_amount > 100)
        if summary_data.get("total_amount"):
            base_metadata["total_amount"] = summary_data["total_amount"]
        if summary_data.get("subtotal"):
            base_metadata["subtotal"] = summary_data["subtotal"]
        if summary_data.get("tax_amount"):
            base_metadata["tax_amount"] = summary_data["tax_amount"]

        # Extract field type information (safely handle both dict and string values)
        amount_fields = [k for k, v in field_mappings.items()
                        if isinstance(v, dict) and v.get("semantic_type") == "amount"]
        date_fields = [k for k, v in field_mappings.items()
                      if isinstance(v, dict) and v.get("semantic_type") == "date"]
        category_fields = [k for k, v in field_mappings.items()
                         if isinstance(v, dict) and v.get("aggregation") == "group_by"]

        logger.debug(f"[Tabular] Amount fields: {amount_fields}")
        logger.debug(f"[Tabular] Date fields: {date_fields}")
        logger.debug(f"[Tabular] Category fields: {category_fields}")

        # Get sample rows for LLM analysis from external storage
        # NOTE: As of migration 022, line_items are stored in documents_data_line_items table
        sample_rows = self._fetch_sample_line_items(document_data.id, limit=10)
        logger.debug(f"[Tabular] Sample rows for LLM: {len(sample_rows)}")

        # ===== SINGLE COMPREHENSIVE CHUNK =====
        # Combines document summary, schema info, and business context into one chunk
        # Benefits: 1 LLM call, no duplicate search results, simpler architecture
        logger.info("[Tabular] Generating comprehensive summary chunk...")
        summary_text, summary_method = self._generate_comprehensive_summary(
            filename=filename,
            headers=unique_columns,
            row_count=row_count,
            sample_rows=sample_rows,
            field_mappings=field_mappings,
            header_data=extracted_header_data,
            summary_data=summary_data
        )
        logger.info(f"[Tabular] Comprehensive summary generated using method: {summary_method}")

        chunks.append(LangchainDocument(
            page_content=summary_text,
            metadata={
                **base_metadata,
                "chunk_type": "tabular_summary",
                "chunk_id": f"{document_id}_summary",
                "summary_method": summary_method,
                "amount_fields": amount_fields,
                "date_fields": date_fields,
                "category_fields": category_fields,
            }
        ))

        logger.info(f"[Tabular] Total summary chunks generated: {len(chunks)}")
        return chunks

    def _generate_comprehensive_summary(
        self,
        filename: str,
        headers: List[str],
        row_count: int,
        sample_rows: List[Dict],
        field_mappings: Dict,
        header_data: Dict = None,
        summary_data: Dict = None
    ) -> Tuple[str, str]:
        """
        Generate a single comprehensive summary combining document overview,
        schema info, and business context.

        This replaces the previous 3-chunk approach (summary, schema, context)
        with a single LLM call that produces a search-optimized comprehensive chunk.

        Benefits:
        - Single LLM call (faster, cheaper)
        - No duplicate search results
        - Simpler architecture
        - All relevant info in one place for RAG

        Returns:
            Tuple of (summary_text, generation_method)
            generation_method is "llm" or "rule_based"
        """
        logger.debug("[Tabular] _generate_comprehensive_summary() called")
        llm = self._get_llm()
        header_data = header_data or {}
        summary_data = summary_data or {}

        # Build header info string for invoice-like documents
        header_info = ""
        if header_data:
            header_parts = []
            if header_data.get("invoice_number"):
                header_parts.append(f"Invoice Number: {header_data['invoice_number']}")
            if header_data.get("vendor_name"):
                header_parts.append(f"Vendor: {header_data['vendor_name']}")
            if header_data.get("customer_name"):
                header_parts.append(f"Customer: {header_data['customer_name']}")
            if header_data.get("invoice_date"):
                header_parts.append(f"Invoice Date: {header_data['invoice_date']}")
            if header_data.get("due_date"):
                header_parts.append(f"Due Date: {header_data['due_date']}")
            if header_data.get("currency"):
                header_parts.append(f"Currency: {header_data['currency']}")
            # Include any other header fields
            for key, value in header_data.items():
                if key not in ["invoice_number", "vendor_name", "customer_name", "invoice_date", "due_date", "currency"] and value:
                    header_parts.append(f"{key}: {value}")
            if header_parts:
                header_info = "\n- " + "\n- ".join(header_parts)

        # Build summary/totals info string
        summary_info = ""
        if summary_data:
            summary_parts = []
            if summary_data.get("total_amount"):
                summary_parts.append(f"Total Amount: ${summary_data['total_amount']}")
            if summary_data.get("subtotal"):
                summary_parts.append(f"Subtotal: ${summary_data['subtotal']}")
            if summary_data.get("tax_amount"):
                summary_parts.append(f"Tax: ${summary_data['tax_amount']}")
            # Include any other summary fields
            for key, value in summary_data.items():
                if key not in ["total_amount", "subtotal", "tax_amount"] and value:
                    summary_parts.append(f"{key}: {value}")
            if summary_parts:
                summary_info = "\n- " + "\n- ".join(summary_parts)

        # Extract field type information (safely handle both dict and string values)
        amount_fields = [k for k, v in field_mappings.items()
                        if isinstance(v, dict) and v.get("semantic_type") == "amount"]
        date_fields = [k for k, v in field_mappings.items()
                      if isinstance(v, dict) and v.get("semantic_type") == "date"]
        category_fields = [k for k, v in field_mappings.items()
                         if isinstance(v, dict) and v.get("aggregation") == "group_by"]

        # PRIMARY: LLM-based comprehensive summary
        if llm and (sample_rows or header_data):
            try:
                logger.info("[Tabular] Using LLM for comprehensive summary generation...")

                # Get sample values per column for schema context
                sample_values = {}
                for header in headers[:10]:  # Limit to first 10 columns
                    values = [str(row.get(header, "")) for row in sample_rows[:5] if row.get(header)]
                    if values:
                        sample_values[header] = values[:3]

                # Extract unique values from categorical fields for context
                category_values = {}
                for field, mapping in field_mappings.items():
                    if isinstance(mapping, dict) and mapping.get("aggregation") == "group_by":
                        values = set()
                        for row in sample_rows:
                            if field in row and row[field]:
                                values.add(str(row[field]))
                        if values:
                            category_values[field] = list(values)[:5]

                prompt = f"""You are analyzing a tabular dataset to generate a comprehensive search-optimized summary.
This summary will be embedded for vector search, so include ALL key identifying information.

=== DOCUMENT INFORMATION ===
Filename: {filename}
Total Rows: {row_count}
Columns ({len(headers)}): {', '.join(headers[:15])}{'...' if len(headers) > 15 else ''}
{f"Document Header Information:{header_info}" if header_info else ""}
{f"Financial Summary:{summary_info}" if summary_info else ""}

=== SAMPLE DATA ===
{json.dumps(sample_rows[:5], indent=2, default=str) if sample_rows else "No sample rows available"}

=== FIELD ANALYSIS ===
{f"Amount/Numeric Fields: {', '.join(amount_fields)}" if amount_fields else ""}
{f"Date Fields: {', '.join(date_fields)}" if date_fields else ""}
{f"Category/Grouping Fields: {', '.join(category_fields)}" if category_fields else ""}

Generate a comprehensive summary (4-6 sentences) that includes:

1. DOCUMENT IDENTITY: What type of document is this? Include specific identifiers like invoice numbers, vendor names, customer names.

2. KEY VALUES: Include the actual amounts, dates, and totals from the document. Example: "Invoice #AXVGVB-00004 from Augment Code to yangzhenwu@gmail.com for $50.00"

3. DATA STRUCTURE: What columns/fields does this contain? What can be analyzed or aggregated?

4. BUSINESS CONTEXT: What business domain is this from? What questions could this data answer?

CRITICAL REQUIREMENTS:
- Include SPECIFIC values (invoice numbers, vendor names, amounts, dates) - these are essential for search
- Write in natural language optimized for semantic search
- A user searching for "Augment Code invoice" or "invoice $50" should find this document
- Include both the document overview AND the key identifying details

Summary:"""

                logger.debug(f"[Tabular] LLM prompt length: {len(prompt)} chars")
                chat_model = llm.get_chat_model(temperature=0.3, num_predict=500)
                response = chat_model.invoke(prompt)
                summary = response.content.strip() if hasattr(response, 'content') else str(response).strip()

                logger.info(f"[Tabular] LLM comprehensive summary generated ({len(summary)} chars)")
                logger.debug(f"[Tabular] Summary preview: {summary[:300]}...")
                return summary, "llm"

            except Exception as e:
                logger.warning(f"[Tabular] LLM comprehensive summary generation failed: {e}")
                logger.info("[Tabular] Falling back to rule-based summary...")

        else:
            if not llm:
                logger.info("[Tabular] LLM not available, using rule-based summary")
            if not sample_rows and not header_data:
                logger.info("[Tabular] No data available, using rule-based summary")

        # FALLBACK: Rule-based comprehensive summary
        summary = self._generate_fallback_comprehensive_summary(
            filename, headers, row_count, field_mappings, header_data, summary_data
        )
        logger.info(f"[Tabular] Rule-based comprehensive summary generated ({len(summary)} chars)")
        return summary, "rule_based"

    def _generate_fallback_comprehensive_summary(
        self,
        filename: str,
        headers: List[str],
        row_count: int,
        field_mappings: Dict,
        header_data: Dict = None,
        summary_data: Dict = None
    ) -> str:
        """
        Rule-based fallback for comprehensive summary when LLM is unavailable.
        Combines document identity, key values, structure, and context.
        """
        logger.debug("[Tabular] _generate_fallback_comprehensive_summary() called")
        header_data = header_data or {}
        summary_data = summary_data or {}

        parts = []

        # 1. Document identity with specific values
        identity_parts = []
        if header_data.get("invoice_number"):
            identity_parts.append(f"Invoice #{header_data['invoice_number']}")
        if header_data.get("vendor_name"):
            identity_parts.append(f"from {header_data['vendor_name']}")
        if header_data.get("customer_name"):
            identity_parts.append(f"to {header_data['customer_name']}")

        if identity_parts:
            parts.append(" ".join(identity_parts) + ".")
        else:
            # Detect document type from filename
            doc_type = "tabular data"
            filename_lower = filename.lower()
            if any(kw in filename_lower for kw in ["invoice", "receipt", "billing"]):
                doc_type = "invoice/billing document"
            elif any(kw in filename_lower for kw in ["sales", "revenue", "transaction"]):
                doc_type = "sales/transaction data"
            elif any(kw in filename_lower for kw in ["inventory", "stock", "product"]):
                doc_type = "inventory/product data"
            parts.append(f"This is a {doc_type}.")

        # 2. Key values (amounts, dates)
        value_parts = []
        if summary_data.get("total_amount"):
            value_parts.append(f"Total: ${summary_data['total_amount']}")
        if summary_data.get("subtotal"):
            value_parts.append(f"Subtotal: ${summary_data['subtotal']}")
        if header_data.get("invoice_date"):
            value_parts.append(f"Date: {header_data['invoice_date']}")
        if header_data.get("due_date"):
            value_parts.append(f"Due: {header_data['due_date']}")

        if value_parts:
            parts.append(" | ".join(value_parts) + ".")

        # 3. Data structure
        parts.append(f"Contains {row_count} rows with {len(headers)} columns: {', '.join(headers[:8])}{'...' if len(headers) > 8 else ''}.")

        # 4. Field type info (safely handle both dict and string values)
        amount_fields = [k for k, v in field_mappings.items()
                        if isinstance(v, dict) and v.get("semantic_type") == "amount"]
        date_fields = [k for k, v in field_mappings.items()
                      if isinstance(v, dict) and v.get("semantic_type") == "date"]
        category_fields = [k for k, v in field_mappings.items()
                         if isinstance(v, dict) and v.get("aggregation") == "group_by"]

        if amount_fields:
            parts.append(f"Numeric fields for analysis: {', '.join(amount_fields[:4])}.")
        if date_fields:
            parts.append(f"Date tracking: {', '.join(date_fields[:3])}.")
        if category_fields:
            parts.append(f"Grouping fields: {', '.join(category_fields[:3])}.")

        # 5. Source reference
        parts.append(f"Source: '{filename}'.")

        return " ".join(parts)

    def _generate_document_summary(
        self,
        filename: str,
        headers: List[str],
        row_count: int,
        sample_rows: List[Dict],
        field_mappings: Dict,
        header_data: Dict = None,
        summary_data: Dict = None
    ) -> Tuple[str, str]:
        """
        Generate a natural language summary of the document using LLM.

        This is the PRIMARY method - LLM-driven for dynamic adaptation
        to any business dataset.

        Returns:
            Tuple of (summary_text, generation_method)
            generation_method is "llm" or "rule_based"
        """
        logger.debug("[Tabular] _generate_document_summary() called")
        llm = self._get_llm()
        header_data = header_data or {}
        summary_data = summary_data or {}

        # Build header info string for invoice-like documents
        header_info = ""
        if header_data:
            header_parts = []
            if header_data.get("invoice_number"):
                header_parts.append(f"Invoice #: {header_data['invoice_number']}")
            if header_data.get("vendor_name"):
                header_parts.append(f"Vendor: {header_data['vendor_name']}")
            if header_data.get("customer_name"):
                header_parts.append(f"Customer: {header_data['customer_name']}")
            if header_data.get("invoice_date"):
                header_parts.append(f"Date: {header_data['invoice_date']}")
            if header_data.get("due_date"):
                header_parts.append(f"Due: {header_data['due_date']}")
            if header_parts:
                header_info = "\n- " + "\n- ".join(header_parts)

        # Build summary info string
        summary_info = ""
        if summary_data:
            summary_parts = []
            if summary_data.get("total_amount"):
                summary_parts.append(f"Total: ${summary_data['total_amount']}")
            if summary_data.get("subtotal"):
                summary_parts.append(f"Subtotal: ${summary_data['subtotal']}")
            if summary_data.get("tax_amount"):
                summary_parts.append(f"Tax: ${summary_data['tax_amount']}")
            if summary_parts:
                summary_info = "\n- " + "\n- ".join(summary_parts)

        # PRIMARY: LLM-based summary
        if llm and (sample_rows or header_data):
            try:
                logger.info("[Tabular] Using LLM for document summary generation...")
                prompt = f"""You are analyzing a tabular dataset to generate a search-optimized summary.

Dataset Information:
- Filename: {filename}
- Total Rows: {row_count}
- Columns: {', '.join(headers)}
{f"Document Header Information:{header_info}" if header_info else ""}
{f"Financial Summary:{summary_info}" if summary_info else ""}

Sample Data (first {min(len(sample_rows), 5)} rows):
{json.dumps(sample_rows[:5], indent=2, default=str) if sample_rows else "No sample rows available"}

Field Analysis:
{json.dumps(field_mappings, indent=2) if field_mappings else "No field mappings"}

Generate a 2-4 sentence summary that captures:
1. What type of data this is (invoice, receipt, sales, inventory, HR, financial, etc.)
2. Key identifying information (invoice number, vendor, customer, dates)
3. The amounts/totals involved
4. The apparent business purpose

IMPORTANT: Include specific values like invoice numbers, vendor names, and amounts in the summary.
The summary should help users find this document when searching for related data.
Write in natural language, optimized for semantic search.

Summary:"""

                logger.debug(f"[Tabular] LLM prompt length: {len(prompt)} chars")
                chat_model = llm.get_chat_model(temperature=0.3, num_predict=300)
                response = chat_model.invoke(prompt)
                summary = response.content.strip() if hasattr(response, 'content') else str(response).strip()

                logger.info(f"[Tabular] LLM summary generated successfully ({len(summary)} chars)")
                logger.debug(f"[Tabular] Summary preview: {summary[:200]}...")
                return summary, "llm"

            except Exception as e:
                logger.warning(f"[Tabular] LLM summary generation failed: {e}")
                logger.info("[Tabular] Falling back to rule-based summary...")

        else:
            if not llm:
                logger.info("[Tabular] LLM not available, using rule-based summary")
            if not sample_rows:
                logger.info("[Tabular] No sample rows, using rule-based summary")

        # FALLBACK: Rule-based summary
        summary = self._generate_fallback_summary(
            filename, headers, row_count, field_mappings, header_data, summary_data
        )
        logger.info(f"[Tabular] Rule-based summary generated ({len(summary)} chars)")
        return summary, "rule_based"

    def _generate_fallback_summary(
        self,
        filename: str,
        headers: List[str],
        row_count: int,
        field_mappings: Dict,
        header_data: Dict = None,
        summary_data: Dict = None
    ) -> str:
        """Rule-based fallback when LLM is unavailable."""
        logger.debug("[Tabular] _generate_fallback_summary() called")
        header_data = header_data or {}
        summary_data = summary_data or {}

        # Detect document type from filename
        doc_type = "tabular data"
        filename_lower = filename.lower()

        if any(kw in filename_lower for kw in ["sales", "revenue", "transaction", "order"]):
            doc_type = "sales/transaction data"
        elif any(kw in filename_lower for kw in ["inventory", "stock", "product", "sku"]):
            doc_type = "inventory/product data"
        elif any(kw in filename_lower for kw in ["employee", "hr", "payroll", "staff"]):
            doc_type = "HR/employee data"
        elif any(kw in filename_lower for kw in ["invoice", "receipt", "billing"]):
            doc_type = "financial/billing data"
        elif any(kw in filename_lower for kw in ["customer", "client", "contact"]):
            doc_type = "customer/contact data"
        elif any(kw in filename_lower for kw in ["expense", "cost", "budget"]):
            doc_type = "expense/budget data"

        logger.debug(f"[Tabular] Detected doc_type: {doc_type}")

        # Extract field types (safely handle both dict and string values)
        amount_fields = [k for k, v in field_mappings.items()
                        if isinstance(v, dict) and v.get("semantic_type") == "amount"]
        date_fields = [k for k, v in field_mappings.items()
                      if isinstance(v, dict) and v.get("semantic_type") == "date"]
        category_fields = [k for k, v in field_mappings.items()
                         if isinstance(v, dict) and v.get("aggregation") == "group_by"]

        # Build summary with header data (invoice details)
        summary_parts = []

        # Include key document identifiers
        if header_data.get("invoice_number"):
            summary_parts.append(f"Invoice #{header_data['invoice_number']}")
        if header_data.get("vendor_name"):
            summary_parts.append(f"from {header_data['vendor_name']}")
        if header_data.get("customer_name"):
            summary_parts.append(f"to {header_data['customer_name']}")

        if summary_parts:
            summary = " ".join(summary_parts) + ". "
        else:
            summary = ""

        summary += f"'{filename}' containing {row_count} rows of {doc_type}. "

        # Include financial totals
        if summary_data.get("total_amount"):
            summary += f"Total amount: ${summary_data['total_amount']}. "

        summary += f"Columns: {', '.join(headers[:6])}{'...' if len(headers) > 6 else ''}. "

        if amount_fields:
            summary += f"Numeric fields for analysis: {', '.join(amount_fields[:3])}. "
        if date_fields:
            summary += f"Date tracking: {', '.join(date_fields[:2])}. "
        if category_fields:
            summary += f"Grouping fields: {', '.join(category_fields[:3])}."

        return summary

    def _generate_schema_description(
        self,
        filename: str,
        headers: List[str],
        field_mappings: Dict,
        row_count: int,
        sample_rows: List[Dict],
        header_data: Dict = None,
        summary_data: Dict = None
    ) -> Tuple[str, str]:
        """
        Generate a schema description using LLM for dynamic adaptation.

        Returns:
            Tuple of (schema_text, generation_method)
        """
        logger.debug("[Tabular] _generate_schema_description() called")
        llm = self._get_llm()
        header_data = header_data or {}
        summary_data = summary_data or {}

        # Build header info string for invoice-like documents
        header_info = ""
        if header_data:
            header_parts = []
            for key, value in header_data.items():
                if value:
                    header_parts.append(f"{key}: {value}")
            if header_parts:
                header_info = "\n- " + "\n- ".join(header_parts)

        # Build summary info string
        summary_info = ""
        if summary_data:
            summary_parts = []
            for key, value in summary_data.items():
                if value:
                    summary_parts.append(f"{key}: {value}")
            if summary_parts:
                summary_info = "\n- " + "\n- ".join(summary_parts)

        # PRIMARY: LLM-based schema description
        if llm and (sample_rows or header_data):
            try:
                logger.info("[Tabular] Using LLM for schema description generation...")

                # Get sample values per column
                sample_values = {}
                for header in headers:
                    values = [str(row.get(header, "")) for row in sample_rows[:5] if row.get(header)]
                    sample_values[header] = values[:3]  # First 3 non-empty values

                prompt = f"""Analyze this dataset's structure and generate a schema description for search indexing.

Filename: {filename}
Total Rows: {row_count}
Columns: {', '.join(headers)}
{f"Document Header Fields:{header_info}" if header_info else ""}
{f"Summary/Totals Fields:{summary_info}" if summary_info else ""}

Sample Values per Column:
{json.dumps(sample_values, indent=2) if sample_values else "No sample values available"}

Field Semantic Analysis (auto-detected types):
{json.dumps(field_mappings, indent=2) if field_mappings else "No field mappings"}

Generate a concise schema description (2-3 paragraphs) that includes:
1. Overview: Total columns and what the data represents
2. Key document identifiers (invoice number, vendor, customer, dates if applicable)
3. Numeric fields: Which fields contain amounts/quantities that can be summed or averaged
4. Categorical fields: Which fields are good for grouping/filtering
5. Date fields: Which fields track time, and any date ranges you can infer

IMPORTANT: Include specific values like invoice numbers, vendor names, and amounts in the description.
Format as natural language optimized for semantic search.

Schema Description:"""

                logger.debug(f"[Tabular] LLM prompt length: {len(prompt)} chars")
                chat_model = llm.get_chat_model(temperature=0.3, num_predict=400)
                response = chat_model.invoke(prompt)
                schema_text = response.content.strip() if hasattr(response, 'content') else str(response).strip()

                logger.info(f"[Tabular] LLM schema description generated ({len(schema_text)} chars)")
                return schema_text, "llm"

            except Exception as e:
                logger.warning(f"[Tabular] LLM schema generation failed: {e}")
                logger.info("[Tabular] Falling back to rule-based schema...")

        # FALLBACK: Rule-based schema description
        schema_text = self._generate_fallback_schema(headers, field_mappings, row_count)
        logger.info(f"[Tabular] Rule-based schema generated ({len(schema_text)} chars)")
        return schema_text, "rule_based"

    def _generate_fallback_schema(
        self,
        headers: List[str],
        field_mappings: Dict,
        row_count: int
    ) -> str:
        """Rule-based fallback for schema description."""
        logger.debug("[Tabular] _generate_fallback_schema() called")

        lines = [
            f"Tabular dataset with {len(headers)} columns and {row_count} rows.",
            f"Column names: {', '.join(headers)}.",
        ]

        # Group fields by semantic type (safely handle both dict and string values)
        by_type = {}
        for field, mapping in field_mappings.items():
            if isinstance(mapping, dict):
                sem_type = mapping.get("semantic_type", "unknown")
            else:
                sem_type = "unknown"
            if sem_type not in by_type:
                by_type[sem_type] = []
            by_type[sem_type].append(field)

        if "amount" in by_type:
            lines.append(f"Numeric/amount fields (summable): {', '.join(by_type['amount'])}.")
        if "quantity" in by_type:
            lines.append(f"Quantity fields: {', '.join(by_type['quantity'])}.")
        if "date" in by_type:
            lines.append(f"Date fields: {', '.join(by_type['date'])}.")
        if "category" in by_type:
            lines.append(f"Category fields (groupable): {', '.join(by_type['category'])}.")
        if "entity" in by_type:
            lines.append(f"Entity fields: {', '.join(by_type['entity'])}.")

        return " ".join(lines)

    def _generate_business_context(
        self,
        filename: str,
        headers: List[str],
        sample_rows: List[Dict],
        field_mappings: Dict,
        header_data: Dict = None,
        summary_data: Dict = None
    ) -> Tuple[str, str]:
        """
        Generate business context and insights using LLM.

        This enables semantic search based on business domain understanding.

        Returns:
            Tuple of (context_text, generation_method)
        """
        logger.debug("[Tabular] _generate_business_context() called")
        llm = self._get_llm()
        header_data = header_data or {}
        summary_data = summary_data or {}

        # Build header info string for invoice-like documents
        header_info = ""
        if header_data:
            header_parts = []
            for key, value in header_data.items():
                if value:
                    header_parts.append(f"{key}: {value}")
            if header_parts:
                header_info = "\n- " + "\n- ".join(header_parts)

        # Build summary info string
        summary_info = ""
        if summary_data:
            summary_parts = []
            for key, value in summary_data.items():
                if value:
                    summary_parts.append(f"{key}: {value}")
            if summary_parts:
                summary_info = "\n- " + "\n- ".join(summary_parts)

        # PRIMARY: LLM-based business context
        if llm and (sample_rows or header_data):
            try:
                logger.info("[Tabular] Using LLM for business context generation...")

                # Extract unique values from categorical fields for context (safely handle both dict and string values)
                category_values = {}
                for field, mapping in field_mappings.items():
                    if isinstance(mapping, dict) and mapping.get("aggregation") == "group_by":
                        values = set()
                        for row in sample_rows:
                            if field in row and row[field]:
                                values.add(str(row[field]))
                        if values:
                            category_values[field] = list(values)[:10]

                logger.debug(f"[Tabular] Category values for context: {category_values}")

                prompt = f"""Analyze this dataset to identify business context and key insights.

Filename: {filename}
Columns: {', '.join(headers)}
{f"Document Information:{header_info}" if header_info else ""}
{f"Financial Totals:{summary_info}" if summary_info else ""}

Sample Data (first {len(sample_rows) if sample_rows else 0} rows):
{json.dumps(sample_rows[:5], indent=2, default=str) if sample_rows else "No sample rows available"}

Unique Values in Key Fields:
{json.dumps(category_values, indent=2) if category_values else "No category values"}

Identify and describe:
1. Business Domain: What industry or function does this data serve? (retail, healthcare, finance, HR, logistics, manufacturing, etc.)
2. Key Entities: What companies, products, people, or locations are mentioned?
3. Transaction Details: Include specific invoice numbers, vendor/customer names, and amounts
4. Use Cases: What business questions could this data answer?

IMPORTANT: Include specific identifying information (invoice numbers, vendor names, amounts) in your description.
Write 2-3 sentences capturing the business context, optimized for semantic search.

Business Context:"""

                logger.debug(f"[Tabular] LLM prompt length: {len(prompt)} chars")
                chat_model = llm.get_chat_model(temperature=0.3, num_predict=300)
                response = chat_model.invoke(prompt)
                context_text = response.content.strip() if hasattr(response, 'content') else str(response).strip()

                logger.info(f"[Tabular] LLM business context generated ({len(context_text)} chars)")
                return context_text, "llm"

            except Exception as e:
                logger.warning(f"[Tabular] LLM business context generation failed: {e}")
                logger.info("[Tabular] Falling back to rule-based context...")

        # FALLBACK: Rule-based context
        context_text = self._generate_fallback_context(filename, sample_rows, field_mappings)
        logger.info(f"[Tabular] Rule-based context generated ({len(context_text)} chars)")
        return context_text, "rule_based"

    def _generate_fallback_context(
        self,
        filename: str,
        sample_rows: List[Dict],
        field_mappings: Dict
    ) -> str:
        """Rule-based fallback for business context."""
        logger.debug("[Tabular] _generate_fallback_context() called")

        if not sample_rows:
            return ""

        # Extract unique values from categorical fields (safely handle both dict and string values)
        category_values = {}
        for field, mapping in field_mappings.items():
            if isinstance(mapping, dict) and mapping.get("aggregation") == "group_by":
                values = set()
                for row in sample_rows:
                    if field in row and row[field]:
                        values.add(str(row[field]))
                if values:
                    category_values[field] = list(values)[:10]

        if not category_values:
            return f"Business data from '{filename}'."

        lines = [f"Business data from '{filename}' containing:"]
        for field, values in list(category_values.items())[:3]:
            lines.append(f"- {field}: {', '.join(values[:5])}")

        return " ".join(lines)

    def _index_summary_chunks(
        self,
        chunks: List[LangchainDocument],
        document_id: UUID
    ) -> List[str]:
        """
        Index summary chunks to Qdrant vector store.

        IMPORTANT: Deletes existing chunks for this document before adding new ones
        to prevent duplicates when documents are re-uploaded or re-processed.

        Returns:
            List of chunk IDs that were indexed
        """
        logger.debug("[Tabular] _index_summary_chunks() called")

        if not chunks:
            logger.warning("[Tabular] No chunks to index!")
            return []

        try:
            # Delete existing chunks for this document to prevent duplicates
            from rag_service.vectorstore import delete_documents_by_document_id
            deleted_count = delete_documents_by_document_id(str(document_id))
            if deleted_count > 0:
                logger.info(f"[Tabular] Deleted {deleted_count} existing chunks for document {document_id}")

            logger.info(f"[Tabular] Getting Qdrant vectorstore...")
            vectorstore = get_vectorstore()

            logger.info(f"[Tabular] Adding {len(chunks)} documents to Qdrant...")
            vectorstore.add_documents(chunks)

            chunk_ids = [chunk.metadata.get("chunk_id") for chunk in chunks]
            logger.info(f"[Tabular] Successfully indexed {len(chunks)} summary chunks")
            logger.debug(f"[Tabular] Chunk IDs: {chunk_ids}")

            return chunk_ids

        except Exception as e:
            logger.error(f"[Tabular] Failed to index summary chunks: {e}", exc_info=True)
            return []

    def _ensure_field_mappings(self, document_data: DocumentData):
        """
        Ensure field mappings are generated and cached in extraction_metadata.
        """
        logger.debug("[Tabular] _ensure_field_mappings() called")

        if not document_data.extraction_metadata:
            document_data.extraction_metadata = {}
            logger.debug("[Tabular] Initialized empty extraction_metadata")

        if "field_mappings" not in document_data.extraction_metadata:
            # Field mappings should already be in header_data from extraction
            header_mappings = document_data.header_data.get("field_mappings", {})
            if header_mappings:
                document_data.extraction_metadata["field_mappings"] = header_mappings
                logger.info(f"[Tabular] Copied {len(header_mappings)} field mappings from header_data")

                if self.db:
                    self.db.merge(document_data)
                    self.db.commit()
                    logger.debug("[Tabular] Committed field mappings to database")
            else:
                logger.warning("[Tabular] No field mappings found in header_data")
        else:
            logger.debug("[Tabular] Field mappings already exist in extraction_metadata")

    def _save_summary_chunk_ids(self, document_id: UUID, chunk_ids: List[str]):
        """Save the summary chunk IDs to the document record."""
        logger.debug("[Tabular] _save_summary_chunk_ids() called")

        if not self.db:
            logger.warning("[Tabular] No database session, cannot save chunk IDs")
            return

        doc = self.db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.summary_chunk_ids = chunk_ids
            self.db.commit()
            logger.info(f"[Tabular] Saved {len(chunk_ids)} summary chunk IDs to document")
        else:
            logger.warning(f"[Tabular] Document {document_id} not found")

    def _update_extraction_status(
        self,
        document_id: UUID,
        status: str,
        error: Optional[str] = None
    ):
        """Update document extraction status."""
        logger.debug(f"[Tabular] _update_extraction_status() called: status={status}")

        if not self.db:
            logger.warning("[Tabular] No database session, cannot update status")
            return

        doc = self.db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.extraction_status = status
            if error:
                doc.extraction_error = error
                logger.debug(f"[Tabular] Set extraction_error: {error[:100]}...")
            if status == "processing":
                doc.extraction_started_at = datetime.utcnow()
                logger.debug("[Tabular] Set extraction_started_at")
            elif status in ["completed", "failed"]:
                doc.extraction_completed_at = datetime.utcnow()
                logger.debug("[Tabular] Set extraction_completed_at")
            self.db.commit()
            logger.info(f"[Tabular] Updated extraction_status to '{status}'")
        else:
            logger.warning(f"[Tabular] Document {document_id} not found")

    def _update_index_status(
        self,
        document_id: UUID,
        status: IndexStatus,
        indexed_chunks: int = 0
    ):
        """Update document index status."""
        logger.debug(f"[Tabular] _update_index_status() called: status={status}, chunks={indexed_chunks}")

        if not self.db:
            logger.warning("[Tabular] No database session, cannot update index status")
            return

        doc = self.db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.index_status = status
            doc.indexed_chunks = indexed_chunks
            # Update indexing_details to reflect tabular processing
            if not doc.indexing_details:
                doc.indexing_details = {}
            doc.indexing_details["tabular_processing"] = {
                "status": "completed",
                "summary_chunks_indexed": indexed_chunks,
                "completed_at": datetime.utcnow().isoformat()
            }
            # Also update metadata_extraction status to completed
            if "metadata_extraction" not in doc.indexing_details:
                doc.indexing_details["metadata_extraction"] = {}
            doc.indexing_details["metadata_extraction"]["status"] = "completed"
            doc.indexing_details["metadata_extraction"]["updated_at"] = datetime.utcnow().isoformat()
            # Skip GraphRAG for tabular documents
            doc.skip_graphrag = True
            doc.skip_graphrag_reason = "tabular_data_structured_rows"
            self.db.commit()
            logger.info(f"[Tabular] Updated index_status to '{status.value}', skip_graphrag=True")
        else:
            logger.warning(f"[Tabular] Document {document_id} not found")

    def _broadcast(self, callback, document_id, message):
        """
        Send WebSocket broadcast if callback available.

        NOTE: The callback is document_status_manager.broadcast_from_thread which
        expects a single dict argument (not conversion_id, message tuple).

        Args:
            callback: The broadcast function (document_status_manager.broadcast_from_thread)
            document_id: Document UUID for frontend identification
            message: Status message dict with status, message, progress fields
        """
        if callback:
            try:
                # Convert document_id to string for JSON serialization
                doc_id_str = str(document_id) if document_id else None

                # Map internal status to frontend event_type
                status = message.get("status", "")
                if status == "completed":
                    event_type = "extraction_completed"
                elif status == "error":
                    event_type = "extraction_failed"
                else:
                    event_type = "extraction_progress"

                # Build broadcast message with document_id and event_type for frontend
                broadcast_message = {
                    "event_type": event_type,
                    "document_id": doc_id_str,
                    "status": status,
                    "message": message.get("message", ""),
                    "progress": message.get("progress", 0),
                    "timestamp": datetime.now().isoformat(),
                    # Include additional data if present
                    **{k: v for k, v in message.items() if k not in ["status", "message", "progress"]}
                }

                logger.info(f"[Tabular] Broadcasting: event_type={event_type}, document_id={doc_id_str}, status={status}")
                callback(broadcast_message)
            except Exception as e:
                logger.warning(f"[Tabular] Broadcast failed: {e}")


def trigger_tabular_extraction(
    document_id: UUID = None,
    source_name: str = None,
    output_dir: str = None,
    filename: str = None,
    conversion_id: str = None,
    broadcast_callback=None
):
    """
    Trigger tabular extraction in background thread.

    This is the main entry point called from the conversion callback.
    """
    logger.info("=" * 80)
    logger.info("[Tabular] ========== TRIGGER TABULAR EXTRACTION ==========")
    logger.info("=" * 80)
    logger.info(f"[Tabular] Document ID: {document_id}")
    logger.info(f"[Tabular] Source Name: {source_name}")
    logger.info(f"[Tabular] Filename: {filename}")
    logger.info(f"[Tabular] Output Dir: {output_dir}")
    logger.info(f"[Tabular] Conversion ID: {conversion_id}")
    logger.info("[Tabular] Starting background extraction thread...")
    logger.info("-" * 80)

    def _extraction_task():
        try:
            logger.info(f"[Tabular] Background thread started for: {source_name}")

            with get_db_session() as db:
                # Get document_id if not provided
                doc_id = document_id
                if not doc_id and filename:
                    logger.debug(f"[Tabular] Looking up document by filename: {filename}")
                    doc = db.query(Document).filter(
                        Document.filename == filename
                    ).first()
                    doc_id = doc.id if doc else None
                    if doc_id:
                        logger.info(f"[Tabular] Found document ID: {doc_id}")

                if not doc_id:
                    logger.error(f"[Tabular] Could not find document for {filename}")
                    return

                service = TabularExtractionService(db)
                service.process_tabular_document(
                    document_id=doc_id,
                    source_name=source_name,
                    output_dir=output_dir,
                    filename=filename,
                    conversion_id=conversion_id,
                    broadcast_callback=broadcast_callback
                )
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"[Tabular] Background extraction failed: {e}")
            logger.error("=" * 80, exc_info=True)

    thread = threading.Thread(
        target=_extraction_task,
        daemon=True,
        name=f"tabular-extraction-{source_name}"
    )
    thread.start()
    logger.info(f"[Tabular] Background thread '{thread.name}' started successfully")


def is_tabular_document(filename: str = None, document_type: str = None, content: str = None) -> bool:
    """
    Quick check if a document should use the tabular processing path.

    This is a convenience function wrapping TabularDataDetector.

    Args:
        filename: Document filename
        document_type: Document type
        content: Optional content for analysis

    Returns:
        True if document should use tabular path
    """
    is_tabular, reason = TabularDataDetector.is_tabular_data(
        filename=filename,
        document_type=document_type,
        content=content
    )
    logger.debug(f"[Tabular] is_tabular_document({filename}): {is_tabular} ({reason})")
    return is_tabular
