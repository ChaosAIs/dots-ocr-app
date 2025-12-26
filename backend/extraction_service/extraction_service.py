"""
Extraction Service

Main service for extracting structured data from documents.
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List
from uuid import UUID
from datetime import datetime

from sqlalchemy.orm import Session as DBSession

from db.models import Document, DocumentData, DocumentDataLineItem, DataSchema
from .extraction_config import (
    ExtractionStrategy,
    get_extraction_prompt,
    get_schema_for_document_type,
    CHUNKED_LLM_MAX_ROWS,
)
from .eligibility_checker import ExtractionEligibilityChecker

logger = logging.getLogger(__name__)


class ExtractionService:
    """
    Service for extracting structured data from documents.

    Supports multiple extraction strategies:
    - LLM Direct: Single LLM call for small documents
    - LLM Chunked: Parallel chunked extraction for medium documents
    - Hybrid: LLM for headers, rules for data
    - Parsed: Pure pattern matching for large documents
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

    def extract_document(
        self,
        document_id: UUID,
        force: bool = False
    ) -> Optional[DocumentData]:
        """
        Extract structured data from a document.

        Args:
            document_id: Document UUID
            force: Force re-extraction even if data exists

        Returns:
            DocumentData model or None if extraction failed
        """
        start_time = time.time()

        # Check if already extracted
        existing = self.db.query(DocumentData).filter(
            DocumentData.document_id == document_id
        ).first()

        if existing and not force:
            logger.info(f"Document {document_id} already extracted, skipping")
            return existing

        # Check eligibility
        eligible, schema_type, reason = self.eligibility_checker.check_eligibility(document_id)
        if not eligible:
            logger.info(f"Document {document_id} not eligible: {reason}")
            self._update_extraction_status(document_id, "skipped", error=reason)
            return None

        # Determine strategy
        strategy, strategy_metadata = self.eligibility_checker.determine_strategy(document_id)

        logger.info(f"Extracting document {document_id} with strategy {strategy.value}")
        self._update_extraction_status(document_id, "processing")

        try:
            # Get document content
            document = self.db.query(Document).filter(Document.id == document_id).first()
            content = self._get_document_content(document)

            if not content:
                raise ValueError("Could not retrieve document content")

            # Execute extraction based on strategy
            if strategy == ExtractionStrategy.LLM_DIRECT:
                result = self._extract_llm_direct(content, schema_type)
            elif strategy == ExtractionStrategy.LLM_CHUNKED:
                result = self._extract_llm_chunked(content, schema_type, strategy_metadata)
            elif strategy == ExtractionStrategy.HYBRID:
                result = self._extract_hybrid(content, schema_type, strategy_metadata)
            elif strategy == ExtractionStrategy.PARSED:
                result = self._extract_parsed(content, schema_type, document)
            else:
                result = self._extract_llm_direct(content, schema_type)

            # Validate and save
            if result:
                duration_ms = int((time.time() - start_time) * 1000)
                document_data = self._save_extraction_result(
                    document_id=document_id,
                    schema_type=schema_type,
                    result=result,
                    strategy=strategy,
                    duration_ms=duration_ms,
                    metadata=strategy_metadata
                )

                self._update_extraction_status(document_id, "completed")
                logger.info(f"Document {document_id} extracted successfully in {duration_ms}ms")
                return document_data
            else:
                raise ValueError("Extraction returned empty result")

        except Exception as e:
            logger.error(f"Extraction failed for document {document_id}: {e}")
            self._update_extraction_status(document_id, "failed", error=str(e))
            return None

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

        # Get the output directory path
        output_dir = os.getenv("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "..", "output"))

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
                        with open(md_path, 'r', encoding='utf-8') as f:
                            return f.read()

                # Try to find any page-based markdown files and combine them
                import glob
                page_files = sorted(glob.glob(os.path.join(doc_output_dir, f"{filename_base}_page_*_nohf.md")))
                if page_files:
                    content_parts = []
                    for page_file in page_files:
                        with open(page_file, 'r', encoding='utf-8') as f:
                            content_parts.append(f.read())
                    if content_parts:
                        logger.debug(f"Combined {len(page_files)} page markdown files")
                        return "\n\n---\n\n".join(content_parts)

                # Try any .md file in the directory
                md_files = glob.glob(os.path.join(doc_output_dir, "*.md"))
                if md_files:
                    # Prefer _nohf.md files
                    nohf_files = [f for f in md_files if '_nohf.md' in f]
                    if nohf_files:
                        with open(nohf_files[0], 'r', encoding='utf-8') as f:
                            return f.read()
                    with open(md_files[0], 'r', encoding='utf-8') as f:
                        return f.read()

            # If output_path points directly to a file
            elif os.path.isfile(doc_output_dir):
                with open(doc_output_dir, 'r', encoding='utf-8') as f:
                    return f.read()

            # Try with .md extension
            if os.path.isfile(doc_output_dir + '.md'):
                with open(doc_output_dir + '.md', 'r', encoding='utf-8') as f:
                    return f.read()

            logger.warning(f"No markdown content found for document {document.id} at {doc_output_dir}")

        except Exception as e:
            logger.error(f"Error reading document content for {document.id}: {e}")

        return None

    def _extract_llm_direct(
        self,
        content: str,
        schema_type: str
    ) -> Dict[str, Any]:
        """
        Extract using a single LLM call.

        Args:
            content: Document content
            schema_type: Target schema type

        Returns:
            Extracted data dictionary
        """
        if not self.llm_client:
            logger.warning("LLM client not configured, using mock extraction")
            return self._mock_extraction(content, schema_type)

        prompt = get_extraction_prompt(schema_type)
        full_prompt = f"{prompt}\n\nDocument content:\n{content[:50000]}"  # Limit content size

        try:
            response = self.llm_client.generate(full_prompt)
            result = self._parse_llm_response(response)
            return result
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            raise

    def _extract_llm_chunked(
        self,
        content: str,
        schema_type: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract using parallel chunked LLM calls.

        Args:
            content: Document content
            schema_type: Target schema type
            metadata: Extraction metadata

        Returns:
            Merged extracted data dictionary
        """
        # Split content into chunks
        chunks = self._split_content_into_chunks(content)

        if len(chunks) == 1:
            return self._extract_llm_direct(content, schema_type)

        # First chunk: Extract headers
        header_result = self._extract_llm_direct(chunks[0], schema_type)

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
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract using LLM for headers and rules for data.

        Args:
            content: Document content
            schema_type: Target schema type
            metadata: Extraction metadata

        Returns:
            Extracted data dictionary
        """
        # Extract first part with LLM to get structure
        first_section = content[:10000]  # First ~10k chars for header detection
        header_result = self._extract_llm_direct(first_section, schema_type)

        # If spreadsheet-like, try to parse the rest with rules
        if schema_type == "spreadsheet" and header_result.get("header_data", {}).get("column_headers"):
            column_headers = header_result["header_data"]["column_headers"]
            line_items = self._parse_tabular_content(content, column_headers)
            header_result["line_items"] = line_items
        else:
            # Fall back to chunked extraction
            return self._extract_llm_chunked(content, schema_type, metadata)

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

            file_path = document.file_path
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return {}

            # Convert to structured data
            columns = df.columns.tolist()
            line_items = df.to_dict('records')

            # Convert NaN to None
            for item in line_items:
                for key, value in item.items():
                    if pd.isna(value):
                        item[key] = None

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

    def _save_extraction_result(
        self,
        document_id: UUID,
        schema_type: str,
        result: Dict[str, Any],
        strategy: ExtractionStrategy,
        duration_ms: int,
        metadata: Dict[str, Any]
    ) -> DocumentData:
        """
        Save extraction result to database.

        Args:
            document_id: Document UUID
            schema_type: Schema type
            result: Extraction result
            strategy: Strategy used
            duration_ms: Extraction duration
            metadata: Additional metadata

        Returns:
            DocumentData model instance
        """
        header_data = result.get("header_data", {})
        line_items = result.get("line_items", [])
        summary_data = result.get("summary_data", {})

        # Determine storage method
        line_items_count = len(line_items)
        if line_items_count > CHUNKED_LLM_MAX_ROWS:
            storage = "external"
            inline_items = []  # Will store in overflow table
        else:
            storage = "inline"
            inline_items = line_items

        # Create or update DocumentData
        existing = self.db.query(DocumentData).filter(
            DocumentData.document_id == document_id
        ).first()

        if existing:
            existing.schema_type = schema_type
            existing.header_data = header_data
            existing.line_items = inline_items
            existing.summary_data = summary_data
            existing.line_items_storage = storage
            existing.line_items_count = line_items_count
            existing.extraction_method = strategy.value
            existing.extraction_duration_ms = duration_ms
            existing.extraction_metadata = metadata
            existing.validation_status = "pending"
            document_data = existing
        else:
            document_data = DocumentData(
                document_id=document_id,
                schema_type=schema_type,
                header_data=header_data,
                line_items=inline_items,
                summary_data=summary_data,
                line_items_storage=storage,
                line_items_count=line_items_count,
                extraction_method=strategy.value,
                extraction_duration_ms=duration_ms,
                extraction_metadata=metadata,
                validation_status="pending"
            )
            self.db.add(document_data)

        self.db.commit()
        self.db.refresh(document_data)

        # Store overflow line items if needed
        if storage == "external":
            # Delete existing overflow items
            self.db.query(DocumentDataLineItem).filter(
                DocumentDataLineItem.documents_data_id == document_data.id
            ).delete()

            # Insert new overflow items
            for i, item in enumerate(line_items):
                overflow_item = DocumentDataLineItem(
                    documents_data_id=document_data.id,
                    line_number=i,
                    data=item
                )
                self.db.add(overflow_item)

            self.db.commit()

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
