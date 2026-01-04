"""
Extraction Service

Main service for extracting structured data from documents.

Key Features:
- LLM-driven schema analysis at each stage
- Dynamic field mapping inference for unknown document types
- Schema caching in extraction_metadata for efficient queries
- Support for both formal schemas (DataSchema table) and dynamic schemas
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from uuid import UUID
from datetime import datetime

from sqlalchemy.orm import Session as DBSession

from db.models import Document, DocumentData, DocumentDataLineItem, DataSchema
from .extraction_config import (
    get_extraction_prompt,
    get_schema_for_document_type,
)
from .eligibility_checker import ExtractionEligibilityChecker
from .field_normalizer import FieldNormalizer
from rag_service.entity_extractor import (
    extract_all_entities,
    create_normalized_metadata,
)

logger = logging.getLogger(__name__)


class ExtractionService:
    """
    Service for extracting structured data from documents.

    Extraction Approach:
    - Direct parsing for all tabular data (CSV, Excel, markdown tables)
    - LLM used only for document classification and field mapping inference
    - All extracted row data stored in documents_data_line_items table (external storage only)

    Schema Management:
    - Looks up formal schemas from DataSchema table
    - Falls back to LLM-driven schema inference
    - Caches field_mappings in extraction_metadata for efficient queries
    """

    def __init__(self, db: DBSession, llm_client=None):
        """
        Initialize the extraction service.

        Args:
            db: SQLAlchemy database session
            llm_client: LLM client for extraction
        """
        self.db = db
        self.llm_client = llm_client
        self.eligibility_checker = ExtractionEligibilityChecker(db)
        self.min_confidence = float(os.getenv("EXTRACTION_MIN_CONFIDENCE", "0.85"))

    def _get_schema_and_prompt(
        self,
        schema_type: str
    ) -> Tuple[Optional[DataSchema], str, Dict[str, Any]]:
        """
        Get schema and extraction prompt using LLM-driven approach.

        Priority:
        1. DataSchema table (formal schema)
        2. Hardcoded extraction_config (legacy)
        3. LLM-generated prompt (dynamic)

        Args:
            schema_type: The schema type to lookup

        Returns:
            Tuple of (DataSchema or None, extraction_prompt, field_mappings)
        """
        # Priority 1: Check DataSchema table
        formal_schema = self.db.query(DataSchema).filter(
            DataSchema.schema_type == schema_type,
            DataSchema.is_active == True
        ).first()

        if formal_schema:
            logger.info(f"[Schema] Found formal schema for {schema_type}")
            return (
                formal_schema,
                formal_schema.extraction_prompt or get_extraction_prompt(schema_type),
                formal_schema.field_mappings or {}
            )

        # Priority 2: Use hardcoded extraction_config (legacy support)
        prompt = get_extraction_prompt(schema_type)
        if prompt:
            logger.info(f"[Schema] Using legacy extraction config for {schema_type}")
            return (None, prompt, {})

        # Priority 3: LLM-generated prompt (for unknown types)
        if self.llm_client:
            logger.info(f"[Schema] Generating dynamic prompt for {schema_type}")
            prompt = self._generate_dynamic_prompt(schema_type)
            return (None, prompt, {})

        # Fallback
        logger.warning(f"[Schema] No schema found for {schema_type}, using generic prompt")
        return (None, self._get_generic_extraction_prompt(), {})

    def _generate_dynamic_prompt(self, schema_type: str) -> str:
        """
        Use LLM to generate an extraction prompt for an unknown schema type.

        Args:
            schema_type: The schema type to generate prompt for

        Returns:
            Generated extraction prompt
        """
        generation_prompt = f"""Generate an extraction prompt for documents of type "{schema_type}".

The prompt should instruct an LLM to:
1. Identify and extract all header-level fields (document metadata like date, vendor, customer, etc.)
2. Identify and extract all line items (individual records/rows in the document)
3. Calculate summary fields (totals, counts, etc.)

Return a JSON object with:
{{
    "extraction_prompt": "The complete prompt text...",
    "expected_header_fields": ["field1", "field2"],
    "expected_line_item_fields": ["field1", "field2"],
    "expected_summary_fields": ["field1", "field2"]
}}

Generate for schema type: {schema_type}"""

        try:
            response = self.llm_client.generate(generation_prompt)
            result = self._parse_llm_response(response)
            if result and 'extraction_prompt' in result:
                return result['extraction_prompt']
        except Exception as e:
            logger.warning(f"[Schema] Dynamic prompt generation failed: {e}")

        return self._get_generic_extraction_prompt()

    def _get_generic_extraction_prompt(self) -> str:
        """Get a generic extraction prompt for unknown document types."""
        return """Extract structured data from this document.

Return a JSON object with:
{
    "header_data": {
        // Document-level fields (date, vendor, customer, document number, etc.)
        // Use null for any field not explicitly visible
    },
    "line_items": [
        // Array of item records/rows in the document
        // Each item should have consistent fields
    ],
    "summary_data": {
        // Summary fields (totals, counts, etc.)
    }
}

IMPORTANT:
- Extract ALL visible fields
- Use null for missing values
- Ensure numbers are actual numbers, not strings
- Dates should be in YYYY-MM-DD format"""

    def extract_document(
        self,
        document_id: UUID,
        force: bool = False
    ) -> Optional[DocumentData]:
        """
        Extract structured data from a document using LLM-driven schema analysis.

        This method:
        1. Checks extraction eligibility
        2. Looks up formal schema or generates dynamic schema
        3. Uses schema-aware extraction prompt
        4. Caches field_mappings in extraction_metadata

        Args:
            document_id: Document UUID
            force: Force re-extraction even if data exists

        Returns:
            DocumentData model or None if extraction failed
        """
        logger.info("=" * 80)
        logger.info("[Extraction] ========== DOCUMENT EXTRACTION START ==========")
        logger.info("=" * 80)
        logger.info(f"[Extraction] Document ID: {document_id}")
        logger.info(f"[Extraction] Force re-extraction: {force}")
        logger.info("-" * 80)

        start_time = time.time()

        # Check if already extracted
        logger.info("[Extraction] STEP 1: Checking for existing extraction...")
        existing = self.db.query(DocumentData).filter(
            DocumentData.document_id == document_id
        ).first()

        if existing and not force:
            logger.info(f"[Extraction] Found existing extraction:")
            logger.info(f"[Extraction]   - Schema type: {existing.schema_type}")
            logger.info(f"[Extraction]   - Line items count: {existing.line_items_count}")
            logger.info(f"[Extraction]   - Extraction method: {existing.extraction_method}")
            logger.info("[Extraction] Skipping re-extraction (use force=True to override)")
            logger.info("=" * 80)
            # Ensure status is marked as completed if data exists
            self._update_extraction_status(document_id, "completed")
            return existing

        # Check eligibility
        logger.info("-" * 80)
        logger.info("[Extraction] STEP 2: Checking extraction eligibility...")
        eligible, schema_type, reason = self.eligibility_checker.check_eligibility(document_id)
        logger.info(f"[Extraction] Eligibility result:")
        logger.info(f"[Extraction]   - Eligible: {eligible}")
        logger.info(f"[Extraction]   - Schema type: {schema_type}")
        logger.info(f"[Extraction]   - Reason: {reason}")

        if not eligible:
            logger.info("[Extraction] Document NOT eligible for extraction")
            logger.info("=" * 80)
            self._update_extraction_status(document_id, "skipped", error=reason)
            return None

        # Determine strategy (now always returns "direct_parsing")
        logger.info("-" * 80)
        logger.info("[Extraction] STEP 3: Determining extraction strategy...")
        strategy, strategy_metadata = self.eligibility_checker.determine_strategy(document_id)
        logger.info(f"[Extraction] Strategy selected: {strategy}")
        if strategy_metadata:
            logger.info(f"[Extraction] Strategy metadata: {strategy_metadata}")

        logger.info("-" * 80)
        logger.info("[Extraction] STEP 4: Starting extraction process...")
        self._update_extraction_status(document_id, "processing")

        try:
            # Get document content
            logger.info("[Extraction] Loading document content...")
            document = self.db.query(Document).filter(Document.id == document_id).first()
            logger.info(f"[Extraction] Document filename: {document.filename if document else 'N/A'}")
            content = self._get_document_content(document)

            if not content:
                raise ValueError("Could not retrieve document content")

            content_length = len(content)
            logger.info(f"[Extraction] Content loaded: {content_length} characters")

            # Get schema and extraction prompt (LLM-driven)
            logger.info("-" * 80)
            logger.info("[Extraction] STEP 5: Getting schema and extraction prompt...")
            formal_schema, extraction_prompt, schema_field_mappings = self._get_schema_and_prompt(schema_type)
            logger.info(f"[Extraction] Schema source: {'formal' if formal_schema else 'dynamic/legacy'}")
            if schema_field_mappings:
                logger.info(f"[Extraction] Pre-defined field mappings: {len(schema_field_mappings)} fields")

            # Execute extraction using direct parsing (unified strategy since migration 022)
            logger.info("-" * 80)
            logger.info("[Extraction] STEP 6: Executing extraction...")
            logger.info(f"[Extraction] Using strategy: {strategy}")

            # All extraction now uses direct parsing approach
            # For spreadsheets: parse tables directly
            # For other documents: use LLM for metadata, parse for row data
            logger.info("[Extraction] Running direct_parsing extraction...")
            result = self._extract_direct_parsing(content, schema_type, extraction_prompt, document)

            # Validate and save
            if result:
                duration_ms = int((time.time() - start_time) * 1000)
                line_items_count = len(result.get("line_items", []))
                header_fields = len(result.get("header_data", {}).keys())

                logger.info("-" * 80)
                logger.info("[Extraction] STEP 7: Extraction result summary:")
                logger.info(f"[Extraction]   - Header fields: {header_fields}")
                logger.info(f"[Extraction]   - Line items: {line_items_count}")
                logger.info(f"[Extraction]   - Duration: {duration_ms}ms")

                # Build field_mappings for caching in extraction_metadata
                logger.info("-" * 80)
                logger.info("[Extraction] STEP 8: Building field mappings...")
                final_field_mappings = self._build_field_mappings(
                    result=result,
                    schema_field_mappings=schema_field_mappings,
                    formal_schema=formal_schema
                )
                logger.info(f"[Extraction] Field mappings generated: {len(final_field_mappings)} categories")

                # STEP 8.5: Normalize field names to canonical schema names
                logger.info("-" * 80)
                logger.info("[Extraction] STEP 8.5: Normalizing field names to canonical schema...")
                result = self._normalize_extraction_result(
                    result=result,
                    field_mappings=final_field_mappings,
                    schema_type=schema_type
                )

                # STEP 8.6: Enhance with NER-based entity extraction and normalization
                logger.info("-" * 80)
                logger.info("[Extraction] STEP 8.6: Enhancing with NER entity extraction...")
                result = self._enhance_with_entity_extraction(result, content)

                # Add field_mappings to metadata for caching
                enhanced_metadata = strategy_metadata.copy() if strategy_metadata else {}
                enhanced_metadata['schema_source'] = 'formal' if formal_schema else 'dynamic'
                enhanced_metadata['schema_version'] = formal_schema.schema_version if formal_schema else 'inferred'

                logger.info("-" * 80)
                logger.info("[Extraction] STEP 9: Saving extraction result to database...")
                document_data = self._save_extraction_result(
                    document_id=document_id,
                    schema_type=schema_type,
                    result=result,
                    strategy=strategy,
                    duration_ms=duration_ms,
                    metadata=enhanced_metadata,
                    field_mappings=final_field_mappings
                )

                self._update_extraction_status(document_id, "completed")

                logger.info("=" * 80)
                logger.info("[Extraction] ========== DOCUMENT EXTRACTION COMPLETE ==========")
                logger.info("=" * 80)
                logger.info(f"[Extraction] FINAL SUMMARY:")
                logger.info(f"[Extraction]   - Document ID: {document_id}")
                logger.info(f"[Extraction]   - Schema type: {schema_type}")
                logger.info(f"[Extraction]   - Strategy: {strategy}")
                logger.info(f"[Extraction]   - Line items extracted: {line_items_count}")
                logger.info(f"[Extraction]   - Total duration: {duration_ms}ms")
                logger.info(f"[Extraction]   - Status: COMPLETED")
                logger.info("=" * 80)

                return document_data
            else:
                raise ValueError("Extraction returned empty result")

        except Exception as e:
            logger.error("=" * 80)
            logger.error("[Extraction] ========== DOCUMENT EXTRACTION FAILED ==========")
            logger.error("=" * 80)
            logger.error(f"[Extraction] Document ID: {document_id}")
            logger.error(f"[Extraction] Error: {e}")
            logger.error("=" * 80)
            self._update_extraction_status(document_id, "failed", error=str(e))
            return None

    def _build_field_mappings(
        self,
        result: Dict[str, Any],
        schema_field_mappings: Dict[str, Any],
        formal_schema: Optional[DataSchema]
    ) -> Dict[str, Any]:
        """
        Build comprehensive field mappings for caching in extraction_metadata.

        Priority:
        1. Formal schema field_mappings
        2. Field_mappings from extraction result (dynamic)
        3. LLM-inferred field_mappings

        Args:
            result: Extraction result
            schema_field_mappings: Field mappings from schema lookup
            formal_schema: Formal DataSchema if found

        Returns:
            Complete field_mappings with header_fields and line_item_fields
        """
        # Start with formal schema mappings if available
        if schema_field_mappings:
            logger.info("[Field Mappings] Using formal schema field_mappings")
            return schema_field_mappings

        # Check if extraction result contains field_mappings (from _parse_markdown_table)
        header_data = result.get("header_data", {})
        if "field_mappings" in header_data:
            logger.info("[Field Mappings] Using field_mappings from extraction result")
            # Convert to standard format with source markers
            extracted_mappings = header_data["field_mappings"]
            return self._normalize_field_mappings(extracted_mappings)

        # LLM-driven inference as last resort
        logger.info("[Field Mappings] Inferring field_mappings from extracted data")
        return self._infer_field_mappings_from_result(result)

    def _get_document_content(self, document: Document) -> Optional[str]:
        """
        Get the content of a document for extraction.

        Args:
            document: Document model

        Returns:
            Document content as string
        """
        if not document:
            return None

        # Get the output directory path - resolve to absolute path
        output_dir = os.getenv("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "..", "output"))
        # If relative path, resolve relative to backend directory (where .env is loaded from)
        if not os.path.isabs(output_dir):
            backend_dir = os.path.dirname(os.path.dirname(__file__))  # extraction_service/../ = backend/
            output_dir = os.path.join(backend_dir, output_dir.lstrip("./"))
        output_dir = os.path.abspath(output_dir)

        try:
            # Get filename without extension
            filename_base = os.path.splitext(document.filename)[0]

            # Determine the document output folder
            if document.output_path:
                # output_path is typically a relative path like "user/workspace/filename"
                doc_output_dir = os.path.join(output_dir, document.output_path)
            else:
                # Fallback: construct from file_path
                if document.file_path:
                    rel_dir = os.path.dirname(document.file_path)
                    doc_output_dir = os.path.join(output_dir, rel_dir, filename_base) if rel_dir else os.path.join(output_dir, filename_base)
                else:
                    doc_output_dir = os.path.join(output_dir, filename_base)

            logger.debug(f"Looking for markdown content in: {doc_output_dir}")

            # Import image cleanup utility to remove base64 images before LLM processing
            # This reduces token usage and improves extraction accuracy
            from rag_service.markdown_chunker import clean_markdown_images

            def read_and_clean_markdown(file_path: str) -> str:
                """Read markdown file and clean embedded base64 images."""
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                return clean_markdown_images(raw_content)

            # Check if it's a directory (most common case)
            if os.path.isdir(doc_output_dir):
                # Look for markdown files inside the directory
                # Priority order: _nohf.md, .md, _page_N_nohf.md
                md_patterns = [
                    f"{filename_base}_nohf.md",
                    f"{filename_base}.md",
                ]

                for pattern in md_patterns:
                    md_path = os.path.join(doc_output_dir, pattern)
                    if os.path.exists(md_path):
                        logger.debug(f"Found markdown file: {md_path}")
                        return read_and_clean_markdown(md_path)

                # Try to find any page-based markdown files and combine them
                import glob
                page_files = sorted(glob.glob(os.path.join(doc_output_dir, f"{filename_base}_page_*_nohf.md")))
                if page_files:
                    content_parts = []
                    for page_file in page_files:
                        content_parts.append(read_and_clean_markdown(page_file))
                    if content_parts:
                        logger.debug(f"Combined {len(page_files)} page markdown files")
                        return "\n\n---\n\n".join(content_parts)

                # Try any .md file in the directory
                md_files = glob.glob(os.path.join(doc_output_dir, "*.md"))
                if md_files:
                    # Prefer _nohf.md files
                    nohf_files = [f for f in md_files if '_nohf.md' in f]
                    if nohf_files:
                        return read_and_clean_markdown(nohf_files[0])
                    return read_and_clean_markdown(md_files[0])

            # If output_path points directly to a file
            elif os.path.isfile(doc_output_dir):
                return read_and_clean_markdown(doc_output_dir)

            # Try with .md extension
            if os.path.isfile(doc_output_dir + '.md'):
                return read_and_clean_markdown(doc_output_dir + '.md')

            logger.warning(f"No markdown content found for document {document.id} at {doc_output_dir}")

        except Exception as e:
            logger.error(f"Error reading document content for {document.id}: {e}")

        return None

    def _normalize_field_mappings(
        self,
        mappings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Normalize field mappings to standard format with source markers.

        Args:
            mappings: Raw field mappings from extraction

        Returns:
            Normalized field mappings with header_fields and line_item_fields
        """
        header_fields = {}
        line_item_fields = {}

        for field_name, field_info in mappings.items():
            if isinstance(field_info, dict):
                source = field_info.get('source', 'line_item')
                if source == 'header' or source == 'summary':
                    header_fields[field_name] = {**field_info, 'source': source}
                else:
                    line_item_fields[field_name] = {**field_info, 'source': 'line_item'}
            else:
                # Simple mapping, assume line_item
                line_item_fields[field_name] = {
                    'semantic_type': 'unknown',
                    'data_type': 'string',
                    'source': 'line_item'
                }

        return {
            'header_fields': header_fields,
            'line_item_fields': line_item_fields
        }

    def _infer_field_mappings_from_result(
        self,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Infer field mappings from extraction result using LLM or heuristics.

        Args:
            result: Extraction result with header_data, line_items, summary_data

        Returns:
            Inferred field mappings
        """
        header_fields = {}
        line_item_fields = {}

        # Process header_data
        header_data = result.get("header_data", {})
        for field in header_data.keys():
            if field not in ['field_mappings', 'column_headers']:
                header_fields[field] = self._classify_field(field, 'header')

        # Process line_items (first item as template)
        line_items = result.get("line_items", [])
        if line_items and isinstance(line_items[0], dict):
            for field in line_items[0].keys():
                if field != 'row_number':
                    line_item_fields[field] = self._classify_field(field, 'line_item')

        # Process summary_data
        summary_data = result.get("summary_data", {})
        for field in summary_data.keys():
            if field not in header_fields:
                header_fields[field] = self._classify_field(field, 'summary')

        return {
            'header_fields': header_fields,
            'line_item_fields': line_item_fields
        }

    def _classify_field(self, field_name: str, source: str) -> Dict[str, Any]:
        """
        Classify a field using semantic pattern matching.

        Args:
            field_name: The field name to classify
            source: The source (header, line_item, summary)

        Returns:
            Field classification with semantic_type, data_type, aggregation
        """
        field_lower = field_name.lower().replace('_', ' ').replace('-', ' ')

        # Semantic patterns (more specific first)
        patterns = [
            ('date', ['date', 'time', 'created', 'updated', 'timestamp'], 'datetime', None),
            ('amount', ['total', 'amount', 'price', 'cost', 'revenue', 'sales', 'subtotal', 'sum', 'fee', 'tax'], 'number', 'sum'),
            ('quantity', ['quantity', 'qty', 'count', 'units', 'items', 'number of'], 'number', 'sum'),
            ('entity', ['customer', 'vendor', 'supplier', 'client', 'company', 'store', 'merchant'], 'string', 'group_by'),
            ('category', ['category', 'type', 'class', 'group', 'segment'], 'string', 'group_by'),
            ('product', ['product', 'item', 'description', 'name', 'sku'], 'string', 'group_by'),
            ('region', ['region', 'city', 'country', 'location', 'address', 'area'], 'string', 'group_by'),
            ('person', ['sales rep', 'representative', 'agent', 'manager', 'assignee'], 'string', 'group_by'),
            ('method', ['method', 'payment', 'shipping', 'channel'], 'string', 'group_by'),
            ('identifier', ['id', 'number', 'code', 'reference', 'invoice', 'receipt', 'order'], 'string', None),
        ]

        for sem_type, keywords, data_type, aggregation in patterns:
            if any(kw in field_lower for kw in keywords):
                return {
                    'semantic_type': sem_type,
                    'data_type': data_type,
                    'source': source,
                    'aggregation': aggregation,
                    'original_name': field_name
                }

        return {
            'semantic_type': 'unknown',
            'data_type': 'string',
            'source': source,
            'aggregation': None,
            'original_name': field_name
        }

    def _extract_llm_direct(
        self,
        content: str,
        schema_type: str,
        extraction_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract using a single LLM call with schema-aware prompt.

        Args:
            content: Document content
            schema_type: Target schema type
            extraction_prompt: Schema-specific extraction prompt (from formal schema or dynamic)

        Returns:
            Extracted data dictionary
        """
        # For spreadsheet types with markdown tables, try direct parsing first
        # This is more reliable than LLM extraction for tabular data
        if schema_type == "spreadsheet":
            parsed_result = self._parse_markdown_table(content)
            if parsed_result and parsed_result.get("line_items"):
                logger.info(f"Successfully parsed {len(parsed_result['line_items'])} rows from markdown table")
                return parsed_result
            logger.info("Markdown table parsing returned no data, falling back to LLM extraction")

        if not self.llm_client:
            logger.warning("LLM client not configured, using mock extraction")
            return self._mock_extraction(content, schema_type)

        # Use provided extraction_prompt or fallback to legacy config
        prompt = extraction_prompt or get_extraction_prompt(schema_type)
        full_prompt = f"{prompt}\n\nDocument content:\n{content[:50000]}"  # Limit content size

        try:
            response = self.llm_client.generate(full_prompt)
            result = self._parse_llm_response(response)

            # Validate spreadsheet extraction - if LLM only returned headers, try parsing
            if schema_type == "spreadsheet":
                line_items = result.get("line_items", [])
                # Check if line_items is just a list of strings (column names) instead of dicts
                if line_items and all(isinstance(item, str) for item in line_items):
                    logger.warning("LLM returned column headers instead of row data, falling back to markdown parsing")
                    parsed_result = self._parse_markdown_table(content)
                    if parsed_result and parsed_result.get("line_items"):
                        return parsed_result
                # Check if line_items is empty but we have content with tables
                elif not line_items and '|' in content:
                    logger.warning("LLM returned empty line_items, falling back to markdown parsing")
                    parsed_result = self._parse_markdown_table(content)
                    if parsed_result and parsed_result.get("line_items"):
                        return parsed_result

            # Post-process: ensure monetary flags are set for invoice/receipt types
            if schema_type in ("invoice", "receipt"):
                result = self._ensure_monetary_flags(result, content)

            return result
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            raise

    def _extract_direct_parsing(
        self,
        content: str,
        schema_type: str,
        extraction_prompt: Optional[str] = None,
        document: Optional[Document] = None
    ) -> Dict[str, Any]:
        """
        Unified extraction using direct parsing approach.

        This is the primary extraction method as of migration 022.
        All row data is extracted via direct parsing, LLM is only used
        for header/metadata extraction when needed.

        Strategy:
        1. For spreadsheets: Direct table parsing (no LLM for row data)
        2. For invoices/receipts: Use LLM extraction (tables often lack proper headers)
        3. For other tabular documents: Parse tables first, use LLM for metadata
        4. Fallback: LLM extraction for complex/unstructured content

        Args:
            content: Document content
            schema_type: Target schema type
            extraction_prompt: Schema-specific extraction prompt
            document: Document model for file access

        Returns:
            Extracted data dictionary with header_data, line_items, summary_data
        """
        # For spreadsheet types, always try direct parsing first
        if schema_type == "spreadsheet":
            # Try parsing spreadsheet file directly
            if document:
                file_result = self._parse_spreadsheet_file(document)
                if file_result and file_result.get("line_items"):
                    logger.info(f"[DirectParsing] Parsed {len(file_result['line_items'])} rows from spreadsheet file")
                    return file_result

            # Try parsing markdown tables in content
            parsed_result = self._parse_markdown_table(content)
            if parsed_result and parsed_result.get("line_items"):
                logger.info(f"[DirectParsing] Parsed {len(parsed_result['line_items'])} rows from markdown table")
                return parsed_result

            logger.info("[DirectParsing] Direct parsing found no data, falling back to LLM")

        # For invoice/receipt types, use LLM extraction directly
        # These document types often have tables without proper column headers
        # (e.g., label-value format where first column is description, not a header)
        if schema_type in ("invoice", "receipt"):
            logger.info(f"[DirectParsing] Using LLM extraction for {schema_type} (tables may lack proper headers)")
            return self._extract_llm_direct(content, schema_type, extraction_prompt)

        # For other non-spreadsheet documents with tables, try hybrid approach
        if '|' in content:
            # Content has tables, try to parse them
            parsed_result = self._parse_markdown_table(content)
            if parsed_result and parsed_result.get("line_items"):
                # Check if headers look valid (not just data values)
                headers = parsed_result.get("header_data", {}).get("column_headers", [])
                if self._has_valid_table_headers(headers):
                    # We have valid table headers, get metadata via LLM if available
                    if self.llm_client and extraction_prompt:
                        try:
                            # Extract only header/metadata from first section
                            first_section = content[:10000]
                            prompt = extraction_prompt or get_extraction_prompt(schema_type)
                            metadata_prompt = f"""{prompt}

IMPORTANT: Focus only on extracting the header_data (document metadata) and summary_data.
The line_items have already been extracted separately.

Document content:
{first_section}"""
                            response = self.llm_client.generate(metadata_prompt)
                            llm_result = self._parse_llm_response(response)
                            # Merge LLM metadata with parsed line items
                            parsed_result["header_data"] = llm_result.get("header_data", {})
                            parsed_result["summary_data"] = llm_result.get("summary_data", {})
                            logger.info(f"[DirectParsing] Merged LLM metadata with {len(parsed_result['line_items'])} parsed rows")
                        except Exception as e:
                            logger.warning(f"[DirectParsing] LLM metadata extraction failed: {e}")
                    return parsed_result
                else:
                    logger.info(f"[DirectParsing] Table headers look like data values, falling back to LLM: {headers[:3]}")

        # Fallback: Use LLM extraction for complex/unstructured content
        # This is only for non-tabular documents where parsing doesn't work
        return self._extract_llm_direct(content, schema_type, extraction_prompt)

    def _has_valid_table_headers(self, headers: List[str]) -> bool:
        """
        Check if table headers look like actual column names rather than data values.

        Valid headers typically:
        - Have descriptive names (Description, Qty, Amount, Date, etc.)
        - Are relatively short
        - Don't look like data values (numbers, prices, long sentences)

        Args:
            headers: List of potential header strings

        Returns:
            True if headers look valid, False if they look like data values
        """
        import re

        if not headers:
            return False

        # Common valid header keywords (case-insensitive)
        valid_header_keywords = [
            'description', 'item', 'product', 'service', 'name',
            'qty', 'quantity', 'units', 'count',
            'price', 'rate', 'cost', 'amount', 'total', 'value',
            'date', 'time', 'period',
            'category', 'type', 'status', 'reference', 'code', 'sku',
            'tax', 'subtotal', 'discount', 'fee',
            'debit', 'credit', 'balance',
        ]

        valid_count = 0
        for header in headers:
            if not header or not header.strip():
                # Empty header is suspicious
                continue

            header_lower = header.lower().strip()

            # Check if header contains valid keywords
            if any(keyword in header_lower for keyword in valid_header_keywords):
                valid_count += 1
                continue

            # Check if header looks like a data value (numeric, price pattern, etc.)
            # Patterns that indicate data rather than headers:
            # - Pure numbers or numbers with units: "600", "465.00", "600 x 1"
            # - Price patterns: "$50.00", "50.00 User Messages"
            # - Very long strings (> 40 chars) are likely sentences/descriptions
            if len(header) > 40:
                # Long string - likely a description, not a header
                continue

            # Check for numeric patterns
            if re.match(r'^[\d.,\s\-+$€£¥%x×]+$', header):
                # Pure numeric/currency - definitely data
                continue

            if re.match(r'^\d+(\.\d+)?\s*(x|\×)\s*\d+', header_lower):
                # Pattern like "600 x 1" - quantity pattern, likely data
                continue

            if re.match(r'^\d+(\.\d+)?\s+\w+', header_lower):
                # Pattern like "600 User Messages" - data with unit
                continue

            # Short, non-numeric headers might be valid
            if len(header) < 20 and not re.search(r'\d', header):
                valid_count += 1

        # Consider headers valid if at least 30% of non-empty headers look valid
        non_empty_headers = [h for h in headers if h and h.strip()]
        if not non_empty_headers:
            return False

        validity_ratio = valid_count / len(non_empty_headers)
        logger.debug(f"[DirectParsing] Header validity: {valid_count}/{len(non_empty_headers)} = {validity_ratio:.2f}")

        return validity_ratio >= 0.3

    def _extract_llm_chunked(
        self,
        content: str,
        schema_type: str,
        metadata: Dict[str, Any],
        extraction_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract using parallel chunked LLM calls with schema-aware prompts.

        Args:
            content: Document content
            schema_type: Target schema type
            metadata: Extraction metadata
            extraction_prompt: Schema-specific extraction prompt

        Returns:
            Merged extracted data dictionary
        """
        # Split content into chunks
        chunks = self._split_content_into_chunks(content)

        if len(chunks) == 1:
            return self._extract_llm_direct(content, schema_type, extraction_prompt)

        # First chunk: Extract headers
        header_result = self._extract_llm_direct(chunks[0], schema_type, extraction_prompt)

        # Remaining chunks: Extract line items only
        all_line_items = header_result.get("line_items", [])

        for chunk in chunks[1:]:
            line_items_prompt = f"""Extract only the line items/data rows from this document section.
Return a JSON array of line items matching this structure:
{json.dumps(header_result.get("line_items", [{}])[:1], indent=2)}

Document section:
{chunk}"""

            try:
                if self.llm_client:
                    response = self.llm_client.generate(line_items_prompt)
                    chunk_items = self._parse_llm_response(response)
                    if isinstance(chunk_items, list):
                        all_line_items.extend(chunk_items)
                    elif isinstance(chunk_items, dict) and "line_items" in chunk_items:
                        all_line_items.extend(chunk_items["line_items"])
            except Exception as e:
                logger.warning(f"Chunk extraction failed: {e}")
                continue

        # Deduplicate line items
        all_line_items = self._deduplicate_line_items(all_line_items)

        # Merge results
        header_result["line_items"] = all_line_items
        return header_result

    def _extract_hybrid(
        self,
        content: str,
        schema_type: str,
        metadata: Dict[str, Any],
        extraction_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract using LLM for headers and rules for data.

        Args:
            content: Document content
            schema_type: Target schema type
            metadata: Extraction metadata
            extraction_prompt: Schema-specific extraction prompt

        Returns:
            Extracted data dictionary
        """
        # Extract first part with LLM to get structure
        first_section = content[:10000]  # First ~10k chars for header detection
        header_result = self._extract_llm_direct(first_section, schema_type, extraction_prompt)

        # If spreadsheet-like, try to parse the rest with rules
        if schema_type == "spreadsheet" and header_result.get("header_data", {}).get("column_headers"):
            column_headers = header_result["header_data"]["column_headers"]
            line_items = self._parse_tabular_content(content, column_headers)
            header_result["line_items"] = line_items
        else:
            # Fall back to chunked extraction
            return self._extract_llm_chunked(content, schema_type, metadata, extraction_prompt)

        return header_result

    def _extract_parsed(
        self,
        content: str,
        schema_type: str,
        document: Document
    ) -> Dict[str, Any]:
        """
        Extract using pure pattern matching for large documents.

        Args:
            content: Document content
            schema_type: Target schema type
            document: Document model

        Returns:
            Extracted data dictionary
        """
        # For spreadsheets, try direct parsing
        if schema_type == "spreadsheet":
            return self._parse_spreadsheet_file(document)

        # For other types, try to detect table structure
        lines = content.split('\n')
        header_lines = lines[:50]  # First 50 lines for header
        data_lines = lines[50:]

        # Try to detect table markers
        table_start = None
        headers = []

        for i, line in enumerate(lines):
            if '|' in line and len(line.split('|')) > 2:
                table_start = i
                # Parse header
                headers = [h.strip() for h in line.split('|') if h.strip()]
                break

        if table_start is not None and headers:
            # Parse table rows
            line_items = []
            for line in lines[table_start + 2:]:  # Skip header and separator
                if '|' in line:
                    values = [v.strip() for v in line.split('|') if v.strip()]
                    if len(values) == len(headers):
                        row = dict(zip(headers, values))
                        line_items.append(row)

            return {
                "header_data": {
                    "column_headers": headers,
                    "total_rows": len(line_items)
                },
                "line_items": line_items,
                "summary_data": {"row_count": len(line_items)}
            }

        # Fall back to hybrid extraction for small datasets
        return self._extract_hybrid(content, schema_type, {})

    def _parse_spreadsheet_file(self, document: Document) -> Dict[str, Any]:
        """
        Parse spreadsheet file directly.

        Args:
            document: Document model

        Returns:
            Extracted data dictionary
        """
        try:
            import pandas as pd

            # Get the input directory path from env (same pattern as OUTPUT_DIR)
            input_dir = os.getenv("INPUT_DIR", os.path.join(os.path.dirname(__file__), "..", "input"))

            # Resolve relative file path to absolute path
            file_path = document.file_path
            if not os.path.isabs(file_path):
                file_path = os.path.join(input_dir, file_path)

            logger.debug(f"[Extraction] Parsing spreadsheet file: {file_path}")

            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return {}

            # Convert to structured data
            columns = df.columns.tolist()
            line_items = df.to_dict('records')

            # Convert NaN to None and Timestamps to ISO strings for JSON serialization
            for item in line_items:
                for key, value in item.items():
                    if pd.isna(value):
                        item[key] = None
                    elif isinstance(value, (pd.Timestamp, datetime)):
                        # Convert Timestamp/datetime to ISO string for JSON serialization
                        item[key] = value.isoformat()
                    elif hasattr(value, 'item'):
                        # Convert numpy types (int64, float64, etc.) to Python native types
                        item[key] = value.item()

            return {
                "header_data": {
                    "column_headers": columns,
                    "total_rows": len(line_items),
                    "total_columns": len(columns)
                },
                "line_items": line_items,
                "summary_data": {
                    "row_count": len(line_items),
                    "column_count": len(columns)
                }
            }

        except Exception as e:
            logger.error(f"Spreadsheet parsing failed: {e}")
            return {}

    def _split_content_into_chunks(
        self,
        content: str,
        chunk_size: int = 8000,
        overlap: int = 500
    ) -> List[str]:
        """
        Split content into overlapping chunks, preserving table structure.

        For tabular content (markdown tables), splits by rows while preserving headers.
        For non-tabular content, splits at natural boundaries.

        Args:
            content: Document content
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks

        Returns:
            List of content chunks
        """
        # Check if content contains significant table data
        table_lines = [l for l in content.split('\n') if '|' in l and len(l.split('|')) > 2]
        total_lines = len(content.split('\n'))

        # If >50% of lines are table rows, use table-aware chunking
        if len(table_lines) > total_lines * 0.5:
            return self._split_table_content(content, chunk_size)

        # Standard chunking for non-tabular content
        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size

            # Try to break at a natural boundary
            if end < len(content):
                # Look for paragraph or line break
                for boundary in ['\n\n', '\n', '. ', ', ']:
                    boundary_pos = content.rfind(boundary, start, end)
                    if boundary_pos > start + chunk_size // 2:
                        end = boundary_pos + len(boundary)
                        break

            chunks.append(content[start:end])
            start = end - overlap

        return chunks

    def _split_table_content(
        self,
        content: str,
        chunk_size: int = 8000
    ) -> List[str]:
        """
        Split table content by rows while preserving headers.

        Args:
            content: Document content with tables
            chunk_size: Target chunk size in characters

        Returns:
            List of content chunks with table headers preserved
        """
        import re

        lines = content.split('\n')
        chunks = []

        # Find table header (first row with pipes and following separator)
        header_lines = []
        data_lines = []
        in_table = False
        pre_table_content = []

        for i, line in enumerate(lines):
            is_table_row = '|' in line and len(line.split('|')) > 2
            is_separator = bool(re.match(r'^\s*\|[\s\-:|]+\|\s*$', line))

            if not in_table:
                if is_table_row and not is_separator:
                    # Found table header
                    in_table = True
                    header_lines.append(line)
                    # Check if next line is separator
                    if i + 1 < len(lines) and re.match(r'^\s*\|[\s\-:|]+\|\s*$', lines[i + 1]):
                        header_lines.append(lines[i + 1])
                else:
                    pre_table_content.append(line)
            else:
                if is_separator and len(header_lines) == 1:
                    # This is the separator line after header
                    header_lines.append(line)
                elif is_table_row:
                    data_lines.append(line)
                # Ignore non-table content within table section

        # If no table found, fall back to standard chunking
        if not header_lines or not data_lines:
            return self._split_non_table_content(content, chunk_size)

        # Prepare header
        header = '\n'.join(header_lines)
        header_overhead = len(header) + 2  # +2 for newlines

        # Include pre-table content in first chunk if small enough
        pre_table = '\n'.join(pre_table_content).strip()
        if pre_table:
            header_overhead += len(pre_table) + 2

        available_size = chunk_size - header_overhead

        # Build chunks by rows
        current_rows = []
        current_size = 0

        for row in data_lines:
            row_size = len(row) + 1  # +1 for newline

            # Check if adding this row would exceed limit
            if current_rows and current_size + row_size > available_size:
                # Build chunk with header
                chunk_content = header + '\n' + '\n'.join(current_rows)
                if pre_table and not chunks:  # Add pre-table content to first chunk
                    chunk_content = pre_table + '\n\n' + chunk_content
                chunks.append(chunk_content)
                current_rows = []
                current_size = 0

            current_rows.append(row)
            current_size += row_size

        # Add remaining rows
        if current_rows:
            chunk_content = header + '\n' + '\n'.join(current_rows)
            if pre_table and not chunks:  # Add pre-table content to first chunk
                chunk_content = pre_table + '\n\n' + chunk_content
            chunks.append(chunk_content)

        return chunks if chunks else [content]

    def _split_non_table_content(
        self,
        content: str,
        chunk_size: int = 8000,
        overlap: int = 500
    ) -> List[str]:
        """
        Split non-table content at natural boundaries.

        Args:
            content: Document content
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks

        Returns:
            List of content chunks
        """
        if len(content) <= chunk_size:
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size

            # Try to break at a natural boundary
            if end < len(content):
                for boundary in ['\n\n', '\n', '. ', ', ']:
                    boundary_pos = content.rfind(boundary, start, end)
                    if boundary_pos > start + chunk_size // 2:
                        end = boundary_pos + len(boundary)
                        break

            chunks.append(content[start:end])
            start = end - overlap

        return chunks

    def _deduplicate_line_items(self, items: List[Dict]) -> List[Dict]:
        """
        Remove duplicate line items.

        Args:
            items: List of line items

        Returns:
            Deduplicated list
        """
        seen = set()
        unique_items = []

        for item in items:
            # Create a hashable key from item
            key = json.dumps(item, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        return unique_items

    def _parse_tabular_content(
        self,
        content: str,
        headers: List[str]
    ) -> List[Dict]:
        """
        Parse tabular content with known headers.

        Args:
            content: Document content
            headers: Column headers

        Returns:
            List of row dictionaries
        """
        # Simple table parsing - looks for markdown tables
        lines = content.split('\n')
        rows = []

        for line in lines:
            if '|' in line:
                values = [v.strip() for v in line.split('|') if v.strip()]
                # Skip header and separator lines
                if values and not all(c in '-:| ' for c in line):
                    if len(values) == len(headers):
                        rows.append(dict(zip(headers, values)))
                    elif len(values) > len(headers):
                        # Take first N values
                        rows.append(dict(zip(headers, values[:len(headers)])))

        return rows

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract JSON.

        Args:
            response: LLM response string

        Returns:
            Parsed dictionary
        """
        # Try to find JSON in response
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith('```'):
            lines = response.split('\n')
            # Remove first and last lines (code block markers)
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response = '\n'.join(lines)

        # Try to parse directly
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in response
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(response[start:end + 1])
            except json.JSONDecodeError:
                pass

        # Try to find JSON array
        start = response.find('[')
        end = response.rfind(']')
        if start != -1 and end != -1:
            try:
                return {"line_items": json.loads(response[start:end + 1])}
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse LLM response as JSON: {response[:200]}")

    def _mock_extraction(
        self,
        content: str,
        schema_type: str
    ) -> Dict[str, Any]:
        """
        Mock extraction for testing without LLM.

        Args:
            content: Document content
            schema_type: Target schema type

        Returns:
            Mock extracted data
        """
        return {
            "header_data": {
                "extracted_at": datetime.utcnow().isoformat(),
                "schema_type": schema_type,
                "mock": True
            },
            "line_items": [],
            "summary_data": {}
        }

    def _parse_markdown_table(self, content: str) -> Dict[str, Any]:
        """
        Parse markdown table format to extract structured data.

        This method dynamically analyzes column headers and sample data to generate
        semantic field mappings, making it adaptable to any spreadsheet structure.

        Args:
            content: Document content with markdown tables

        Returns:
            Extracted data dictionary with header_data, line_items, summary_data,
            including dynamic field_mappings for query processing
        """
        import re

        lines = content.split('\n')
        headers = []
        line_items = []
        table_start = None

        # Find the table header (first line with multiple pipe-separated columns)
        for i, line in enumerate(lines):
            # Skip lines that are clearly not table rows
            if not line.strip() or '|' not in line:
                continue

            # Check if this looks like a table header (has multiple columns)
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 2:
                # Check if next line is a separator line (contains --- or :--)
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if re.match(r'^\s*\|[\s\-:|]+\|\s*$', next_line) or all(c in '-:| ' for c in next_line.strip()):
                        table_start = i
                        headers = parts
                        break

        if not headers or table_start is None:
            logger.warning("No markdown table found in content")
            return {}

        logger.info(f"Found table with headers: {headers}")

        # Collect sample values for each column (for type inference)
        sample_values = {h: [] for h in headers}

        # Parse data rows (skip header and separator)
        row_number = 0
        for line in lines[table_start + 2:]:  # +2 to skip header and separator
            line = line.strip()
            if not line or '|' not in line:
                continue

            # Skip separator lines
            if all(c in '-:| ' for c in line):
                continue

            values = [v.strip() for v in line.split('|') if v.strip()]

            # Only process rows with matching column count
            if len(values) == len(headers):
                row_number += 1
                row = {"row_number": row_number}

                for header, value in zip(headers, values):
                    header_lower = header.lower()
                    # Detect if this is a monetary column (amount, price, total, cost, etc.)
                    is_monetary_column = any(kw in header_lower for kw in [
                        'amount', 'price', 'total', 'cost', 'subtotal', 'tax',
                        'unit_price', 'unit price', 'rate', 'fee', 'charge'
                    ])

                    if is_monetary_column:
                        # Parse with monetary flag detection
                        parsed_value, is_monetary = self._parse_cell_value(value, return_monetary_flag=True)
                        row[header] = parsed_value
                        # Store the is_currency flag for the row
                        row['is_currency'] = is_monetary
                    else:
                        # Regular parsing for non-monetary columns
                        parsed_value = self._parse_cell_value(value)
                        row[header] = parsed_value

                    # Collect samples for type inference (first 10 non-null values)
                    if len(sample_values[header]) < 10 and parsed_value is not None:
                        sample_values[header].append(parsed_value)

                line_items.append(row)
            elif len(values) > 0:
                # Handle rows with different column counts (take what we can)
                logger.debug(f"Row has {len(values)} values but expected {len(headers)}, skipping")

        logger.info(f"Parsed {len(line_items)} data rows from markdown table")

        # Generate dynamic field mappings using LLM or heuristics
        field_mappings = self._generate_field_mappings_with_llm(headers, line_items[:10])
        if not field_mappings:
            # Fallback to heuristic-based inference
            field_mappings = self._infer_field_mappings(headers, sample_values)
        logger.info(f"Generated field mappings: {field_mappings}")

        return {
            "header_data": {
                "column_headers": headers,
                "total_rows": len(line_items),
                "total_columns": len(headers),
                "field_mappings": field_mappings  # Store semantic mappings for queries
            },
            "line_items": line_items,
            "summary_data": {
                "row_count": len(line_items),
                "column_count": len(headers)
            }
        }

    def _generate_field_mappings_with_llm(
        self,
        headers: List[str],
        sample_data: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Use LLM to analyze column headers and sample data to generate field mappings.

        This provides more accurate semantic understanding than heuristic-based inference.

        Args:
            headers: List of column header names
            sample_data: List of sample row dictionaries (first 10 rows)

        Returns:
            Dictionary mapping column names to semantic field info, or None if LLM unavailable
        """
        if not self.llm_client:
            logger.info("LLM client not available, skipping LLM-based field mapping")
            return None

        try:
            from analytics_service.schema_analyzer import analyze_spreadsheet_schema
            field_mappings = analyze_spreadsheet_schema(headers, sample_data, self.llm_client)
            logger.info(f"[LLM] Successfully generated field mappings for {len(field_mappings)} columns")
            return field_mappings
        except ImportError:
            logger.warning("Schema analyzer module not available")
            return None
        except Exception as e:
            logger.warning(f"LLM field mapping generation failed: {e}")
            return None

    def _infer_field_mappings(
        self,
        headers: List[str],
        sample_values: Dict[str, List]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Infer semantic field mappings from column headers and sample values.

        This creates a dynamic schema that maps each column to a semantic type
        (date, amount, category, entity, etc.) for use in queries.

        Args:
            headers: List of column header names
            sample_values: Dict mapping headers to sample values

        Returns:
            Dict mapping each column to its semantic type and properties
        """
        import re

        # Semantic type patterns (column name keywords)
        # Order matters - more specific patterns should come first
        # Use word boundaries or multi-word patterns to avoid false matches
        semantic_patterns = [
            # Date fields (high priority)
            ("date", {
                "keywords": ["date", "time", "created", "updated", "timestamp", "day", "when", "period"],
                "data_type": "datetime",
                "aggregation": None
            }),
            # Person/Representative fields (before entity to avoid "name" false match)
            ("person", {
                "keywords": ["sales rep", "sales_rep", "representative", "agent", "salesperson", "manager", "assignee", "owner", "rep "],
                "data_type": "string",
                "aggregation": "group_by"
            }),
            # Payment method (before amount to catch "payment method" specifically)
            ("method", {
                "keywords": ["payment method", "payment_method", "pay method", "method of payment", "channel", "source", "medium"],
                "data_type": "string",
                "aggregation": "group_by"
            }),
            # Amount/Money fields
            ("amount", {
                "keywords": ["total sales", "total_sales", "total amount", "total_amount", "grand total", "amount", "price", "cost", "revenue", "fee", "tax", "subtotal"],
                "data_type": "number",
                "aggregation": "sum"
            }),
            # Unit price (specific, not for aggregation typically)
            ("unit_price", {
                "keywords": ["unit price", "unit_price", "unit cost", "price each", "rate"],
                "data_type": "number",
                "aggregation": None  # Usually not summed directly
            }),
            # Quantity fields
            ("quantity", {
                "keywords": ["quantity", "qty", "count", "units", "items", "number of"],
                "data_type": "number",
                "aggregation": "sum"
            }),
            # Category fields
            ("category", {
                "keywords": ["category", "type", "class", "group", "classification", "segment", "kind"],
                "data_type": "string",
                "aggregation": "group_by"
            }),
            # Status fields
            ("status", {
                "keywords": ["status", "state", "condition", "stage", "phase"],
                "data_type": "string",
                "aggregation": "group_by"
            }),
            # Entity/Customer fields
            ("entity", {
                "keywords": ["customer name", "customer_name", "vendor name", "vendor_name", "supplier", "client", "company", "merchant", "buyer", "seller"],
                "data_type": "string",
                "aggregation": "group_by"
            }),
            # Product fields
            ("product", {
                "keywords": ["product", "item", "sku", "goods", "service"],
                "data_type": "string",
                "aggregation": "group_by"
            }),
            # Region fields
            ("region", {
                "keywords": ["region", "area", "location", "zone", "territory", "country", "city", "state"],
                "data_type": "string",
                "aggregation": "group_by"
            }),
            # Identifier fields (low priority)
            ("identifier", {
                "keywords": ["order id", "order_id", "id", "number", "code", "reference", "invoice", "receipt", "transaction"],
                "data_type": "string",
                "aggregation": None
            }),
        ]

        field_mappings = {}

        for header in headers:
            header_lower = header.lower().replace('_', ' ').replace('-', ' ')
            samples = sample_values.get(header, [])

            # Determine semantic type from column name
            # Patterns are ordered by priority - more specific patterns first
            semantic_type = "unknown"
            data_type = "string"
            aggregation = None

            for sem_type, config in semantic_patterns:
                if any(kw in header_lower for kw in config["keywords"]):
                    semantic_type = sem_type
                    data_type = config["data_type"]
                    aggregation = config["aggregation"]
                    break

            # If not detected by name, infer from sample values
            if semantic_type == "unknown" and samples:
                inferred_type = self._infer_type_from_samples(samples)
                if inferred_type == "number":
                    # Check if it looks like an amount (has decimals, larger values)
                    if any(isinstance(v, float) and v != int(v) for v in samples if isinstance(v, (int, float))):
                        semantic_type = "amount"
                        aggregation = "sum"
                    else:
                        semantic_type = "quantity"
                        aggregation = "sum"
                    data_type = "number"
                elif inferred_type == "date":
                    semantic_type = "date"
                    data_type = "datetime"
                else:
                    # Check cardinality for potential grouping fields
                    unique_values = len(set(str(v) for v in samples))
                    if unique_values < len(samples) * 0.5:  # Low cardinality = good for grouping
                        semantic_type = "category"
                        aggregation = "group_by"

            field_mappings[header] = {
                "semantic_type": semantic_type,
                "data_type": data_type,
                "aggregation": aggregation,
                "original_name": header
            }

        return field_mappings

    def _infer_type_from_samples(self, samples: List) -> str:
        """
        Infer data type from sample values.

        Args:
            samples: List of sample values

        Returns:
            Inferred type: 'number', 'date', or 'string'
        """
        import re

        if not samples:
            return "string"

        num_count = 0
        date_count = 0

        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY or DD/MM/YYYY
            r'^\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY or DD-MM-YYYY
        ]

        for val in samples:
            if isinstance(val, (int, float)):
                num_count += 1
            elif isinstance(val, str):
                for pattern in date_patterns:
                    if re.match(pattern, val):
                        date_count += 1
                        break

        total = len(samples)
        if num_count / total > 0.7:
            return "number"
        if date_count / total > 0.7:
            return "date"
        return "string"

    def _parse_cell_value(self, value: str, return_monetary_flag: bool = False):
        """
        Parse a cell value, converting to appropriate type.

        Args:
            value: Cell value as string
            return_monetary_flag: If True, return tuple (value, is_monetary)

        Returns:
            If return_monetary_flag is False: Parsed value (number, date string, or original string)
            If return_monetary_flag is True: Tuple of (parsed_value, is_monetary)
        """
        import re

        if not value or value.lower() in ('null', 'none', 'n/a', '-'):
            if return_monetary_flag:
                return None, False
            return None

        # Check if value contains currency symbols BEFORE stripping them
        value_stripped = value.strip()
        has_currency_symbol = bool(re.search(r'[$€£¥]', value_stripped))

        # Try to parse as number (handle currency symbols and commas)
        cleaned = re.sub(r'[$€£¥,]', '', value_stripped)
        try:
            # Try float first
            num = float(cleaned)
            # Return int if it's a whole number
            if num == int(num) and '.' not in cleaned:
                parsed = int(num)
            else:
                parsed = num

            if return_monetary_flag:
                return parsed, has_currency_symbol
            return parsed
        except ValueError:
            pass

        # Return as string (dates will stay as strings in YYYY-MM-DD format)
        if return_monetary_flag:
            return value, False
        return value

    def _ensure_monetary_flags(
        self,
        result: Dict[str, Any],
        content: str
    ) -> Dict[str, Any]:
        """
        Post-process extraction result to ensure is_currency boolean field
        is set on line items. This is a lightweight fallback if the LLM
        didn't generate it properly.

        The primary fix is in the LLM prompt (extraction_config.py) which uses
        the is_currency boolean field (completely separate name from amount/unit_price
        to avoid LLM confusion).

        Args:
            result: Extraction result with line_items
            content: Original document content for context lookup

        Returns:
            Result with is_currency field ensured on all line items
        """
        import re

        line_items = result.get("line_items", [])
        if not line_items:
            return result

        def has_currency_symbol(amount_value: float) -> bool:
            """
            Check if the amount value appears with a currency symbol in the content.
            Returns True if has currency symbol, False otherwise.
            """
            if amount_value is None:
                return True  # Default to currency for safety

            # Format the amount value for searching
            amount_str = str(amount_value)
            if amount_value == int(amount_value):
                amount_int_str = str(int(amount_value))
            else:
                amount_int_str = None

            # Look for currency symbol immediately before the number
            currency_patterns = [
                rf'\$\s*{re.escape(amount_str)}',
                rf'€\s*{re.escape(amount_str)}',
                rf'£\s*{re.escape(amount_str)}',
                rf'¥\s*{re.escape(amount_str)}',
            ]
            if amount_int_str:
                currency_patterns.extend([
                    rf'\$\s*{re.escape(amount_int_str)}(?:\.\d+)?',
                    rf'€\s*{re.escape(amount_int_str)}(?:\.\d+)?',
                    rf'£\s*{re.escape(amount_int_str)}(?:\.\d+)?',
                    rf'¥\s*{re.escape(amount_int_str)}(?:\.\d+)?',
                ])

            for pattern in currency_patterns:
                if re.search(pattern, content):
                    return True

            return False

        # Process each line item
        updated_items = []
        for item in line_items:
            amount = item.get('amount')
            unit_price = item.get('unit_price')
            description = item.get('description', '')

            # Handle LLM confusion: if amount is a string like "money", fix it
            if isinstance(amount, str) and amount in ('money', 'count'):
                logger.warning(f"[Monetary] LLM put string '{amount}' in amount field for '{description}'")
                item['is_currency'] = (amount == 'money')
                # Try to recover amount from quantity
                quantity = item.get('quantity')
                if quantity is not None and isinstance(quantity, (int, float)):
                    item['amount'] = float(quantity)
                else:
                    item['amount'] = None

            # Handle LLM confusion: if amount is a boolean, fix it
            if isinstance(amount, bool):
                logger.warning(f"[Monetary] LLM put boolean in amount field for '{description}'")
                item['is_currency'] = amount
                quantity = item.get('quantity')
                if quantity is not None and isinstance(quantity, (int, float)):
                    item['amount'] = float(quantity)
                else:
                    item['amount'] = None

            # Handle LLM confusion: if unit_price is a string like "money", fix it
            if isinstance(unit_price, str) and unit_price in ('money', 'none'):
                logger.warning(f"[Monetary] LLM put string '{unit_price}' in unit_price field for '{description}'")
                item['unit_price'] = None

            # Handle LLM confusion: if unit_price is a boolean, fix it
            if isinstance(unit_price, bool):
                logger.warning(f"[Monetary] LLM put boolean in unit_price field for '{description}'")
                item['unit_price'] = None

            # Refresh amount after potential fixes
            amount = item.get('amount')

            # Ensure is_currency exists
            if 'is_currency' not in item and amount is not None:
                item['is_currency'] = has_currency_symbol(amount)

            updated_items.append(item)

        result["line_items"] = updated_items
        logger.debug(f"[Monetary] Processed {len(updated_items)} line items for is_currency field")
        return result

    def _normalize_extraction_result(
        self,
        result: Dict[str, Any],
        field_mappings: Dict[str, Any],
        schema_type: str = "invoice"
    ) -> Dict[str, Any]:
        """
        Normalize field names in extraction result to canonical schema names.

        Uses the FieldNormalizer to transform field names like "Item", "Qty", "Amount"
        to canonical names like "description", "quantity", "amount" based on schema aliases.

        Supports two formats for field_mappings:
        1. NEW GROUPED FORMAT (preferred):
           {
               "header_mappings": [...],
               "line_item_mappings": [...],
               "summary_mappings": [...]
           }
        2. LEGACY FLAT FORMAT (backwards compatible):
           {
               "line_item_fields": {...},
               "header_fields": {...}
           }

        Args:
            result: Extraction result with header_data, line_items, summary_data
            field_mappings: Field mappings (grouped or legacy format)
            schema_type: Schema type for default mappings fallback

        Returns:
            Result with normalized field names
        """
        # Initialize normalizer with LLM client
        normalizer = FieldNormalizer(llm_client=self.llm_client)

        # Determine if we have grouped mappings (new format) or legacy format
        has_grouped_mappings = any(
            field_mappings.get(k) for k in ['header_mappings', 'line_item_mappings', 'summary_mappings']
        ) if field_mappings else False

        if has_grouped_mappings:
            # Use new grouped normalization
            return self._normalize_with_grouped_mappings(
                result, field_mappings, schema_type, normalizer
            )
        else:
            # Fall back to legacy normalization for backwards compatibility
            return self._normalize_with_legacy_mappings(
                result, field_mappings, normalizer
            )

    def _normalize_with_grouped_mappings(
        self,
        result: Dict[str, Any],
        grouped_mappings: Dict[str, Any],
        schema_type: str,
        normalizer: FieldNormalizer
    ) -> Dict[str, Any]:
        """
        Normalize extraction result using new grouped field mappings.

        Normalizes header_data, line_items, and summary_data separately using
        their respective mapping rules. This provides better separation of concerns
        and prevents bilingual labels from being misclassified.

        Args:
            result: Extraction result with header_data, line_items, summary_data
            grouped_mappings: Dict with header_mappings, line_item_mappings, summary_mappings
            schema_type: Schema type for default mappings fallback
            normalizer: FieldNormalizer instance

        Returns:
            Result with normalized field names
        """
        try:
            header_data = result.get("header_data", {})
            line_items = result.get("line_items", [])
            summary_data = result.get("summary_data", {})

            logger.info(f"[Extraction] Using GROUPED normalization for schema: {schema_type}")
            logger.info(f"[Extraction] Input: header={len(header_data)} fields, line_items={len(line_items)}, summary={len(summary_data)} fields")

            # Use the new grouped normalization method
            normalized_header, normalized_items, normalized_summary, all_mappings = normalizer.normalize_with_grouped_mappings(
                header_data=header_data,
                line_items=line_items,
                summary_data=summary_data,
                grouped_mappings=grouped_mappings,
                schema_type=schema_type,
                use_llm=True
            )

            # Update result with normalized data
            result["header_data"] = normalized_header
            result["line_items"] = normalized_items
            result["summary_data"] = normalized_summary

            # Store the mappings used for debugging
            result["header_data"]["_field_normalization"] = {
                "format": "grouped",
                "mappings": all_mappings
            }

            # Log normalization summary
            total_mappings = sum(len(m) for m in all_mappings.values())
            if total_mappings > 0:
                logger.info(f"[Extraction] Grouped normalization applied: {total_mappings} field mappings")
                for group, mappings in all_mappings.items():
                    if mappings:
                        logger.info(f"[Extraction]   {group}: {mappings}")
            else:
                logger.debug("[Extraction] No grouped normalization needed")

        except Exception as e:
            logger.error(f"[Extraction] Grouped field normalization failed: {e}", exc_info=True)
            # Return original result on error

        return result

    def _normalize_with_legacy_mappings(
        self,
        result: Dict[str, Any],
        field_mappings: Dict[str, Any],
        normalizer: FieldNormalizer
    ) -> Dict[str, Any]:
        """
        Normalize extraction result using legacy flat field mappings.

        This is the backwards-compatible normalization path for schemas that
        haven't been migrated to the new grouped format.

        Args:
            result: Extraction result with header_data, line_items, summary_data
            field_mappings: Legacy format with line_item_fields, header_fields
            normalizer: FieldNormalizer instance

        Returns:
            Result with normalized field names
        """
        if not field_mappings:
            logger.debug("[Extraction] No field mappings available for normalization")
            return result

        line_items = result.get("line_items", [])
        if not line_items:
            logger.debug("[Extraction] No line items to normalize")
            return result

        # Get line_item_fields from field_mappings
        line_item_fields = field_mappings.get("line_item_fields", {})
        if not line_item_fields:
            logger.debug("[Extraction] No line_item_fields in field_mappings")
            return result

        logger.info("[Extraction] Using LEGACY normalization (consider migrating to grouped format)")

        # Normalize line items using legacy method
        try:
            normalized_items, mapping_used = normalizer.normalize_line_items(
                line_items=line_items,
                schema_field_mappings=line_item_fields,
                use_llm=True
            )

            if mapping_used:
                logger.info(f"[Extraction] Legacy field normalization applied: {mapping_used}")
                # Update result with normalized items
                result["line_items"] = normalized_items

                # Store the mapping used in result for reference
                if "header_data" not in result:
                    result["header_data"] = {}
                result["header_data"]["_field_normalization"] = {
                    "format": "legacy",
                    "line_item": mapping_used
                }
            else:
                logger.debug("[Extraction] No legacy field normalization needed")

        except Exception as e:
            logger.error(f"[Extraction] Legacy field normalization failed: {e}")
            # Return original result on error

        return result

    def _enhance_with_entity_extraction(
        self,
        result: Dict[str, Any],
        content: str
    ) -> Dict[str, Any]:
        """
        Enhance extraction result with NER-based entity extraction and normalization.

        This adds normalized entity fields to header_data for better filtering:
        - vendor_normalized: Normalized vendor name for exact matching
        - customer_normalized: Normalized customer name for exact matching
        - all_entities_normalized: List of all normalized entity names

        Args:
            result: Extraction result with header_data, line_items, summary_data
            content: Document text content for NER processing

        Returns:
            Result with enhanced entity fields in header_data
        """
        try:
            header_data = result.get("header_data", {})

            # Extract entities using all available methods
            entity_result = extract_all_entities(
                text=content,
                header_data=header_data,  # Use structured data as highest priority
                use_spacy=True,
                use_regex=True,
            )

            # Create normalized metadata
            normalized_metadata = create_normalized_metadata(entity_result)

            # Add normalized fields to header_data
            if normalized_metadata.get("vendor_normalized"):
                header_data["vendor_normalized"] = normalized_metadata["vendor_normalized"]
                # Also set vendor_name if not already present
                if not header_data.get("vendor_name") and normalized_metadata.get("vendor_name"):
                    header_data["vendor_name"] = normalized_metadata["vendor_name"]

            if normalized_metadata.get("customer_normalized"):
                header_data["customer_normalized"] = normalized_metadata["customer_normalized"]
                # Also set customer_name if not already present
                if not header_data.get("customer_name") and normalized_metadata.get("customer_name"):
                    header_data["customer_name"] = normalized_metadata["customer_name"]

            # Store all normalized entities for array-based filtering
            if normalized_metadata.get("all_entities_normalized"):
                header_data["all_entities_normalized"] = normalized_metadata["all_entities_normalized"]

            # Store organization and person lists separately
            if normalized_metadata.get("organizations_normalized"):
                header_data["organizations_normalized"] = normalized_metadata["organizations_normalized"]
            if normalized_metadata.get("persons_normalized"):
                header_data["persons_normalized"] = normalized_metadata["persons_normalized"]

            result["header_data"] = header_data

            logger.info(
                f"[Extraction] Entity enhancement: "
                f"vendor={normalized_metadata.get('vendor_name')}, "
                f"customer={normalized_metadata.get('customer_name')}, "
                f"total_entities={len(normalized_metadata.get('all_entities_normalized', []))}"
            )

        except Exception as e:
            logger.error(f"[Extraction] Entity extraction enhancement failed: {e}")
            # Return original result on error

        return result

    def _save_extraction_result(
        self,
        document_id: UUID,
        schema_type: str,
        result: Dict[str, Any],
        strategy: str,
        duration_ms: int,
        metadata: Dict[str, Any],
        field_mappings: Optional[Dict[str, Any]] = None
    ) -> DocumentData:
        """
        Save extraction result to database with field_mappings cached in extraction_metadata.

        NOTE: As of migration 022, strategy is always "direct_parsing" (string).
        The ExtractionStrategy enum has been removed.

        Args:
            document_id: Document UUID
            schema_type: Schema type
            result: Extraction result
            strategy: Strategy string (always "direct_parsing")
            duration_ms: Extraction duration
            metadata: Additional metadata
            field_mappings: Field mappings to cache in extraction_metadata

        Returns:
            DocumentData model instance
        """
        header_data = result.get("header_data", {})
        line_items = result.get("line_items", [])
        summary_data = result.get("summary_data", {})

        # Remove field_mappings from header_data if present (will be stored in extraction_metadata)
        if "field_mappings" in header_data:
            header_data = {k: v for k, v in header_data.items() if k != "field_mappings"}

        # Validate currency field - LLMs often assume USD when no currency is explicit
        # Only allow currency if it's a valid 3-letter code, otherwise set to null
        if "currency" in header_data:
            currency_val = header_data.get("currency")
            if currency_val:
                # Check if it's a valid 3-letter currency code
                valid_codes = {"CAD", "USD", "EUR", "GBP", "AUD", "NZD", "JPY", "CNY", "HKD", "SGD", "CHF", "INR", "MXN", "BRL"}
                if not (isinstance(currency_val, str) and currency_val.upper() in valid_codes):
                    # Not a valid code, set to null
                    header_data["currency"] = None
                    logger.debug(f"Cleared invalid currency value: {currency_val}")

        # All line items stored externally (external-only storage since migration 022)
        line_items_count = len(line_items)

        # Build extraction_metadata with field_mappings cache
        extraction_metadata = metadata.copy() if metadata else {}
        if field_mappings:
            extraction_metadata['field_mappings'] = field_mappings
        extraction_metadata['cached_at'] = datetime.utcnow().isoformat()

        # Determine extraction method string (for tracking purposes)
        extraction_method = "direct_parsing"
        if strategy:
            extraction_method = strategy if isinstance(strategy, str) else getattr(strategy, 'value', 'direct_parsing')

        # Create or update DocumentData
        existing = self.db.query(DocumentData).filter(
            DocumentData.document_id == document_id
        ).first()

        if existing:
            existing.schema_type = schema_type
            existing.header_data = header_data
            existing.summary_data = summary_data
            existing.line_items_count = line_items_count
            existing.extraction_method = extraction_method
            existing.extraction_duration_ms = duration_ms
            existing.extraction_metadata = extraction_metadata
            existing.validation_status = "pending"
            document_data = existing
        else:
            document_data = DocumentData(
                document_id=document_id,
                schema_type=schema_type,
                header_data=header_data,
                summary_data=summary_data,
                line_items_count=line_items_count,
                extraction_method=extraction_method,
                extraction_duration_ms=duration_ms,
                extraction_metadata=extraction_metadata,
                validation_status="pending"
            )
            self.db.add(document_data)

        self.db.commit()
        self.db.refresh(document_data)

        logger.info(f"[Extraction] Saved document_data with field_mappings cached in extraction_metadata")

        # Store ALL line items in external table (external-only storage)
        if line_items:
            # Delete existing line items
            self.db.query(DocumentDataLineItem).filter(
                DocumentDataLineItem.documents_data_id == document_data.id
            ).delete()

            # Insert all line items to external table
            for i, item in enumerate(line_items):
                line_item = DocumentDataLineItem(
                    documents_data_id=document_data.id,
                    line_number=i,
                    data=item
                )
                self.db.add(line_item)

            self.db.commit()
            logger.info(f"[Extraction] Stored {line_items_count} line items in external table")

        return document_data

    def _update_extraction_status(
        self,
        document_id: UUID,
        status: str,
        error: Optional[str] = None
    ):
        """
        Update document extraction status.

        Args:
            document_id: Document UUID
            status: New status
            error: Error message if failed
        """
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.extraction_status = status
            if error:
                document.extraction_error = error
            if status == "processing":
                document.extraction_started_at = datetime.utcnow()
            elif status in ["completed", "failed", "skipped"]:
                document.extraction_completed_at = datetime.utcnow()
            self.db.commit()
