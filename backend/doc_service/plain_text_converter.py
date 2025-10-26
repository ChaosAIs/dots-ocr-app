#!/usr/bin/env python3
"""
Plain Text Document to Markdown Converter Service

This service handles conversion of plain text files (txt, csv, etc.) to Markdown format.
For CSV files, it creates proper table layouts. For plain text, it can use AI analysis
to improve structure and readability when AI features are enabled.
"""

import sys
import os
import csv
import io
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Required packages not installed. Please install them using:")
    print("pip install -r requirements.txt")
    print(f"Missing: {e}")
    sys.exit(1)

from .base_converter import BaseDocumentConverter
from .ai_services import ai_service_factory, AIServiceUnavailableError
from .text_chunking_utils import TokenEstimator, create_chunker


class PlainTextConverter(BaseDocumentConverter):
    """Converter for plain text files (txt, csv, etc.) to Markdown format."""

    def __init__(self, ai_service_type: Optional[str] = None):
        """
        Initialize the plain text converter.
        
        Args:
            ai_service_type: Type of AI service to use ('openai', 'ollama', or None for auto-detection)
        """
        super().__init__()
        
        # Load environment variables
        load_dotenv()
        
        # Initialize AI service for text analysis (optional)
        self.ai_service = None
        self.ai_enabled = False
        self.max_tokens = 3000  # Default conservative limit

        try:
            self.ai_service = ai_service_factory.create_service(ai_service_type)
            if self.ai_service and self.ai_service.is_available():
                self.ai_enabled = True
                # Get appropriate token limit for this service
                service_name = self.ai_service.get_service_name().lower()
                model_name = self.ai_service.get_model_name()
                self.max_tokens = TokenEstimator.get_max_tokens(service_name, model_name)
                self.logger.info(f"AI service enabled: {self.ai_service.get_service_name()} ({model_name}) - Max tokens: {self.max_tokens}")
            else:
                self.logger.info("AI service not available, using fallback text processing")
        except Exception as e:
            self.logger.warning(f"Could not initialize AI service: {e}")
            self.logger.info("Using fallback text processing")

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return ['.txt', '.csv', '.tsv', '.log', '.md', '.text']

    def can_convert(self, file_path: Path) -> bool:
        """Check if this converter can handle the given file."""
        if not file_path.exists():
            return False
        
        # Check file extension
        if file_path.suffix.lower() in self.get_supported_extensions():
            return True
        
        # Check if it's a text file by trying to read it
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)  # Try to read first 1KB
            return True
        except (UnicodeDecodeError, PermissionError):
            return False

    def _convert_document_to_markdown(self, doc_path: Path) -> str:
        """
        Convert a plain text document to Markdown format.
        
        Args:
            doc_path: Path to the document to convert
            
        Returns:
            Markdown content as string
        """
        try:
            # Read the file content
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return f"# {doc_path.stem}\n\n*This file appears to be empty.*\n"
            
            # Detect file type and process accordingly
            file_extension = doc_path.suffix.lower()
            
            if file_extension in ['.csv', '.tsv']:
                return self._convert_csv_to_markdown(content, doc_path, file_extension)
            else:
                return self._convert_text_to_markdown(content, doc_path)
                
        except Exception as e:
            self.logger.error(f"Error converting {doc_path}: {str(e)}")
            return f"# Error Converting {doc_path.name}\n\nFailed to convert file: {str(e)}\n"

    def _convert_csv_to_markdown(self, content: str, doc_path: Path, file_extension: str) -> str:
        """
        Convert CSV content to Markdown table format with chunking support.

        Args:
            content: CSV content as string
            doc_path: Path to the original file
            file_extension: File extension (.csv or .tsv)

        Returns:
            Markdown content with table formatting
        """
        try:
            # Determine delimiter
            delimiter = '\t' if file_extension == '.tsv' else ','

            # Parse CSV content to check if it's valid
            csv_reader = csv.reader(io.StringIO(content), delimiter=delimiter)
            rows = list(csv_reader)

            if not rows:
                return f"# {doc_path.stem}\n\n*This CSV file appears to be empty.*\n"

            # If AI is enabled, use it with chunking support
            if self.ai_enabled:
                try:
                    return self._process_content_with_chunking(content, "csv", doc_path, delimiter)
                except Exception as e:
                    self.logger.warning(f"AI analysis failed, using fallback: {e}")

            # Fallback: Basic CSV to markdown table conversion
            markdown_content = f"# {doc_path.stem}\n\n"

            # Create table header
            if len(rows) > 0:
                headers = rows[0]
                markdown_content += "| " + " | ".join(str(cell).strip() for cell in headers) + " |\n"
                markdown_content += "| " + " | ".join("---" for _ in headers) + " |\n"

                # Add data rows
                for row in rows[1:]:
                    # Pad row to match header length
                    padded_row = row + [''] * (len(headers) - len(row))
                    markdown_content += "| " + " | ".join(str(cell).strip() for cell in padded_row[:len(headers)]) + " |\n"

            return markdown_content

        except Exception as e:
            self.logger.error(f"Error converting CSV {doc_path}: {str(e)}")
            return f"# {doc_path.stem}\n\n## Error Processing CSV\n\n{str(e)}\n\n## Raw Content\n\n```\n{content}\n```\n"

    def _convert_text_to_markdown(self, content: str, doc_path: Path) -> str:
        """
        Convert plain text content to Markdown format with chunking support.

        Args:
            content: Text content as string
            doc_path: Path to the original file

        Returns:
            Markdown content with improved structure
        """
        try:
            # If AI is enabled, use it with chunking support
            if self.ai_enabled:
                try:
                    return self._process_content_with_chunking(content, "txt", doc_path)
                except Exception as e:
                    self.logger.warning(f"AI analysis failed, using fallback: {e}")

            # Fallback: Basic text to markdown conversion
            markdown_content = f"# {doc_path.stem}\n\n"

            # Split content into paragraphs
            paragraphs = content.split('\n\n')

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                # Check if it looks like a heading (short line, possibly with special chars)
                lines = paragraph.split('\n')
                if len(lines) == 1 and len(paragraph) < 80 and any(char in paragraph for char in [':', '-', '=']):
                    markdown_content += f"## {paragraph}\n\n"
                else:
                    # Regular paragraph
                    # Replace single newlines with spaces, keep double newlines as paragraph breaks
                    formatted_paragraph = ' '.join(line.strip() for line in lines if line.strip())
                    markdown_content += f"{formatted_paragraph}\n\n"

            return markdown_content

        except Exception as e:
            self.logger.error(f"Error converting text {doc_path}: {str(e)}")
            return f"# {doc_path.stem}\n\n## Error Processing Text\n\n{str(e)}\n\n## Raw Content\n\n```\n{content}\n```\n"

    def _process_content_with_chunking(self, content: str, file_type: str, doc_path: Path, delimiter: str = None) -> str:
        """
        Process content using AI with intelligent chunking for large files.

        Args:
            content: Content to process
            file_type: Type of content ('txt', 'csv')
            doc_path: Path to the original file
            delimiter: CSV delimiter (for CSV files)

        Returns:
            Combined markdown content from all chunks
        """
        try:
            # Check if content needs chunking
            estimated_tokens = TokenEstimator.estimate_tokens(content)

            if estimated_tokens <= self.max_tokens:
                # Content fits in single request
                self.logger.info(f"Processing {file_type} content in single chunk ({estimated_tokens} tokens)")
                return self.ai_service.analyze_text_content(content, file_type)

            # Content needs chunking
            self.logger.info(f"Processing large {file_type} content with chunking ({estimated_tokens} tokens, max: {self.max_tokens})")

            # Create appropriate chunker
            if file_type == "csv":
                chunker = create_chunker("csv", self.max_tokens)
                chunks = chunker.chunk_csv(content, delimiter or ',')
            else:
                chunker = create_chunker("text", self.max_tokens)
                chunks = chunker.chunk_text(content)

            if len(chunks) == 1:
                # Only one chunk after all, process normally
                return self.ai_service.analyze_text_content(chunks[0].content, file_type)

            # Process multiple chunks
            self.logger.info(f"Split content into {len(chunks)} chunks")
            processed_chunks = []

            for chunk in chunks:
                try:
                    self.logger.info(f"Processing chunk {chunk.chunk_index + 1}/{chunk.total_chunks} ({chunk.estimated_tokens} tokens)")

                    # Use chunk-aware analysis
                    result = self.ai_service.analyze_text_chunk(
                        chunk.content,
                        file_type,
                        chunk.chunk_index,
                        chunk.total_chunks,
                        chunk.metadata
                    )
                    processed_chunks.append(result)

                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk.chunk_index + 1}: {e}")
                    # Add fallback content for failed chunk
                    processed_chunks.append(f"# Error Processing Chunk {chunk.chunk_index + 1}\n\nFailed to process this section: {str(e)}\n\n```\n{chunk.content[:500]}...\n```\n")

            # Combine processed chunks
            return self._combine_processed_chunks(processed_chunks, file_type, doc_path)

        except Exception as e:
            self.logger.error(f"Error in chunked processing: {e}")
            # Fallback to original content
            return f"# {doc_path.stem}\n\n## Error in Chunked Processing\n\n{str(e)}\n\n## Original Content\n\n```\n{content[:1000]}...\n```\n"

    def _combine_processed_chunks(self, processed_chunks: List[str], file_type: str, doc_path: Path) -> str:
        """
        Combine processed chunks into a cohesive markdown document.

        Args:
            processed_chunks: List of processed chunk results
            file_type: Type of content ('txt', 'csv')
            doc_path: Path to the original file

        Returns:
            Combined markdown content
        """
        if not processed_chunks:
            return f"# {doc_path.stem}\n\n*No content was processed.*\n"

        if file_type == "csv":
            # For CSV, combine tables intelligently
            combined_content = f"# {doc_path.stem}\n\n"

            for i, chunk_result in enumerate(processed_chunks):
                if i == 0:
                    # First chunk - include everything
                    combined_content += chunk_result + "\n\n"
                else:
                    # Subsequent chunks - extract table rows only
                    lines = chunk_result.split('\n')
                    table_lines = []
                    in_table = False

                    for line in lines:
                        if line.strip().startswith('|') and '|' in line.strip()[1:]:
                            in_table = True
                            # Skip header separator lines in subsequent chunks
                            if not line.strip().replace('|', '').replace('-', '').replace(' ', ''):
                                continue
                            table_lines.append(line)
                        elif in_table and line.strip() == '':
                            break

                    if table_lines:
                        combined_content += '\n'.join(table_lines) + "\n\n"

            return combined_content

        else:
            # For text, combine with section breaks
            combined_content = ""

            for i, chunk_result in enumerate(processed_chunks):
                if i == 0:
                    # First chunk - include everything
                    combined_content += chunk_result
                else:
                    # Subsequent chunks - remove duplicate titles and combine
                    lines = chunk_result.split('\n')
                    # Skip the first line if it looks like a title
                    start_idx = 0
                    if lines and lines[0].startswith('# '):
                        start_idx = 1
                        # Also skip empty line after title
                        if len(lines) > 1 and lines[1].strip() == '':
                            start_idx = 2

                    chunk_content = '\n'.join(lines[start_idx:])
                    if chunk_content.strip():
                        combined_content += "\n\n" + chunk_content

            return combined_content

    def get_converter_info(self) -> Dict[str, Any]:
        """Get information about this converter."""
        info = super().get_converter_info()
        info.update({
            "ai_enabled": self.ai_enabled,
            "ai_service": self.ai_service.get_service_name() if self.ai_service else None,
            "ai_model": self.ai_service.get_model_name() if self.ai_service else None,
            "max_tokens": self.max_tokens,
            "supports_csv_tables": True,
            "supports_ai_analysis": self.ai_enabled,
            "supports_chunking": True
        })
        return info
