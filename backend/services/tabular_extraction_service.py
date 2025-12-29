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
from common.document_type_classifier import TabularDataDetector
from rag_service.vectorstore import get_vectorstore, get_embeddings
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
            self._broadcast(broadcast_callback, conversion_id, {
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

            self._broadcast(broadcast_callback, conversion_id, {
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
            for field_name, mapping in list(field_mappings.items())[:5]:
                logger.info(f"[Tabular]   - {field_name}: type={mapping.get('semantic_type')}, agg={mapping.get('aggregation')}")
            if len(field_mappings) > 5:
                logger.info(f"[Tabular]   ... and {len(field_mappings) - 5} more fields")
            logger.info("-" * 80)

            self._broadcast(broadcast_callback, conversion_id, {
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

            self._broadcast(broadcast_callback, conversion_id, {
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

            self._broadcast(broadcast_callback, conversion_id, {
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
            self._broadcast(broadcast_callback, conversion_id, {
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
            self._broadcast(broadcast_callback, conversion_id, {
                "status": "error",
                "message": f"Processing failed: {str(e)}",
                "progress": 0
            })
            return False, str(e)

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
        headers = document_data.header_data.get("column_headers", [])
        row_count = document_data.line_items_count
        field_mappings = document_data.extraction_metadata.get("field_mappings", {}) if document_data.extraction_metadata else {}

        logger.debug(f"[Tabular] Headers: {headers}")
        logger.debug(f"[Tabular] Row count: {row_count}")
        logger.debug(f"[Tabular] Field mappings: {len(field_mappings)} fields")

        # Get document for workspace_id
        doc = self.db.query(Document).filter(Document.id == document_id).first() if self.db else None
        workspace_id = str(doc.workspace_id) if doc and doc.workspace_id else None
        logger.debug(f"[Tabular] Workspace ID: {workspace_id}")

        # Common metadata for all chunks
        base_metadata = {
            "document_id": str(document_id),
            "source": source_name,
            "filename": filename,
            "is_tabular": True,
            "schema_type": document_data.schema_type,
            "row_count": row_count,
            "column_count": len(headers),
            "columns": headers,
            "chunk_type": "tabular_summary",  # Mark as tabular for filtering
        }
        if workspace_id:
            base_metadata["workspace_id"] = workspace_id

        # Extract field type information
        amount_fields = [k for k, v in field_mappings.items()
                        if v.get("semantic_type") == "amount"]
        date_fields = [k for k, v in field_mappings.items()
                      if v.get("semantic_type") == "date"]
        category_fields = [k for k, v in field_mappings.items()
                         if v.get("aggregation") == "group_by"]

        logger.debug(f"[Tabular] Amount fields: {amount_fields}")
        logger.debug(f"[Tabular] Date fields: {date_fields}")
        logger.debug(f"[Tabular] Category fields: {category_fields}")

        # Get sample rows for LLM analysis
        sample_rows = document_data.line_items[:10] if document_data.line_items else []
        logger.debug(f"[Tabular] Sample rows for LLM: {len(sample_rows)}")

        # ===== CHUNK 1: Document Summary =====
        logger.info("[Tabular] Generating CHUNK 1: Document Summary...")
        summary_text, summary_method = self._generate_document_summary(
            filename=filename,
            headers=headers,
            row_count=row_count,
            sample_rows=sample_rows,
            field_mappings=field_mappings
        )
        logger.info(f"[Tabular] CHUNK 1 generated using method: {summary_method}")

        chunks.append(LangchainDocument(
            page_content=summary_text,
            metadata={
                **base_metadata,
                "chunk_type": "tabular_summary",
                "chunk_id": f"{document_id}_summary",
                "summary_method": summary_method
            }
        ))

        # ===== CHUNK 2: Schema Description =====
        logger.info("[Tabular] Generating CHUNK 2: Schema Description...")
        schema_text, schema_method = self._generate_schema_description(
            filename=filename,
            headers=headers,
            field_mappings=field_mappings,
            row_count=row_count,
            sample_rows=sample_rows
        )
        logger.info(f"[Tabular] CHUNK 2 generated using method: {schema_method}")

        chunks.append(LangchainDocument(
            page_content=schema_text,
            metadata={
                **base_metadata,
                "chunk_type": "tabular_schema",
                "chunk_id": f"{document_id}_schema",
                "amount_fields": amount_fields,
                "date_fields": date_fields,
                "category_fields": category_fields,
                "schema_method": schema_method
            }
        ))

        # ===== CHUNK 3: Business Context (Optional) =====
        if sample_rows:
            logger.info("[Tabular] Generating CHUNK 3: Business Context...")
            context_text, context_method = self._generate_business_context(
                filename=filename,
                headers=headers,
                sample_rows=sample_rows,
                field_mappings=field_mappings
            )

            if context_text and len(context_text.strip()) > 50:
                logger.info(f"[Tabular] CHUNK 3 generated using method: {context_method}")
                chunks.append(LangchainDocument(
                    page_content=context_text,
                    metadata={
                        **base_metadata,
                        "chunk_type": "tabular_context",
                        "chunk_id": f"{document_id}_context",
                        "context_method": context_method
                    }
                ))
            else:
                logger.info("[Tabular] CHUNK 3 skipped (insufficient content)")
        else:
            logger.info("[Tabular] CHUNK 3 skipped (no sample rows)")

        logger.info(f"[Tabular] Total summary chunks generated: {len(chunks)}")
        return chunks

    def _generate_document_summary(
        self,
        filename: str,
        headers: List[str],
        row_count: int,
        sample_rows: List[Dict],
        field_mappings: Dict
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

        # PRIMARY: LLM-based summary
        if llm and sample_rows:
            try:
                logger.info("[Tabular] Using LLM for document summary generation...")
                prompt = f"""You are analyzing a tabular dataset to generate a search-optimized summary.

Dataset Information:
- Filename: {filename}
- Total Rows: {row_count}
- Columns: {', '.join(headers)}

Sample Data (first {len(sample_rows)} rows):
{json.dumps(sample_rows[:5], indent=2, default=str)}

Field Analysis:
{json.dumps(field_mappings, indent=2)}

Generate a 2-4 sentence summary that captures:
1. What type of data this is (sales, inventory, HR, financial, logistics, etc.)
2. The time period covered (if date fields exist)
3. Key entities or categories present (companies, products, regions, etc.)
4. The apparent business purpose

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
        summary = self._generate_fallback_summary(filename, headers, row_count, field_mappings)
        logger.info(f"[Tabular] Rule-based summary generated ({len(summary)} chars)")
        return summary, "rule_based"

    def _generate_fallback_summary(
        self,
        filename: str,
        headers: List[str],
        row_count: int,
        field_mappings: Dict
    ) -> str:
        """Rule-based fallback when LLM is unavailable."""
        logger.debug("[Tabular] _generate_fallback_summary() called")

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

        # Extract field types
        amount_fields = [k for k, v in field_mappings.items()
                        if v.get("semantic_type") == "amount"]
        date_fields = [k for k, v in field_mappings.items()
                      if v.get("semantic_type") == "date"]
        category_fields = [k for k, v in field_mappings.items()
                         if v.get("aggregation") == "group_by"]

        # Build summary
        summary = f"'{filename}' containing {row_count} rows of {doc_type}. "
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
        sample_rows: List[Dict]
    ) -> Tuple[str, str]:
        """
        Generate a schema description using LLM for dynamic adaptation.

        Returns:
            Tuple of (schema_text, generation_method)
        """
        logger.debug("[Tabular] _generate_schema_description() called")
        llm = self._get_llm()

        # PRIMARY: LLM-based schema description
        if llm and sample_rows:
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

Sample Values per Column:
{json.dumps(sample_values, indent=2)}

Field Semantic Analysis (auto-detected types):
{json.dumps(field_mappings, indent=2)}

Generate a concise schema description (2-3 paragraphs) that includes:
1. Overview: Total columns and what the data represents
2. Numeric fields: Which fields contain amounts/quantities that can be summed or averaged
3. Categorical fields: Which fields are good for grouping/filtering
4. Date fields: Which fields track time, and any date ranges you can infer
5. Relationships: Any apparent relationships between fields

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

        # Group fields by semantic type
        by_type = {}
        for field, mapping in field_mappings.items():
            sem_type = mapping.get("semantic_type", "unknown")
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
        field_mappings: Dict
    ) -> Tuple[str, str]:
        """
        Generate business context and insights using LLM.

        This enables semantic search based on business domain understanding.

        Returns:
            Tuple of (context_text, generation_method)
        """
        logger.debug("[Tabular] _generate_business_context() called")
        llm = self._get_llm()

        # PRIMARY: LLM-based business context
        if llm and sample_rows:
            try:
                logger.info("[Tabular] Using LLM for business context generation...")

                # Extract unique values from categorical fields for context
                category_values = {}
                for field, mapping in field_mappings.items():
                    if mapping.get("aggregation") == "group_by":
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

Sample Data (first {len(sample_rows)} rows):
{json.dumps(sample_rows[:5], indent=2, default=str)}

Unique Values in Key Fields:
{json.dumps(category_values, indent=2)}

Identify and describe:
1. Business Domain: What industry or function does this data serve? (retail, healthcare, finance, HR, logistics, manufacturing, etc.)
2. Key Entities: What companies, products, people, or locations are mentioned?
3. Data Patterns: What categories, types, or segments exist in the data?
4. Use Cases: What business questions could this data answer?

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

        # Extract unique values from categorical fields
        category_values = {}
        for field, mapping in field_mappings.items():
            if mapping.get("aggregation") == "group_by":
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

        Returns:
            List of chunk IDs that were indexed
        """
        logger.debug("[Tabular] _index_summary_chunks() called")

        if not chunks:
            logger.warning("[Tabular] No chunks to index!")
            return []

        try:
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
            # Skip GraphRAG for tabular documents
            doc.skip_graphrag = True
            doc.skip_graphrag_reason = "tabular_data_structured_rows"
            self.db.commit()
            logger.info(f"[Tabular] Updated index_status to '{status.value}', skip_graphrag=True")
        else:
            logger.warning(f"[Tabular] Document {document_id} not found")

    def _broadcast(self, callback, conversion_id, message):
        """Send WebSocket broadcast if callback available."""
        if callback and conversion_id:
            try:
                logger.debug(f"[Tabular] Broadcasting: {message.get('status')} - {message.get('message')}")
                callback(conversion_id, message)
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
