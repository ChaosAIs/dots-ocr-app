#!/usr/bin/env python3
"""
PDF Document to Markdown Converter Service

This service handles conversion of PDF documents to Markdown format.
Uses PyMuPDF (fitz) for PDF text extraction and processing.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import re

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF is not installed. Please install it using:")
    print("pip install PyMuPDF")
    sys.exit(1)

from .base_converter import BaseDocumentConverter


class PDFDocumentConverter(BaseDocumentConverter):
    """Converts PDF documents to Markdown format with full content preservation."""

    def __init__(self):
        """Initialize PDF converter."""
        super().__init__()
        self.font_size_threshold = 12  # Minimum font size to consider for headings
        self.heading_font_sizes = {}  # Track font sizes to determine heading levels
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported PDF document extensions."""
        return ['.pdf']
    
    def can_convert(self, file_path: Path) -> bool:
        """Check if this converter can handle PDF documents."""
        return file_path.suffix.lower() in self.get_supported_extensions()
    
    def _analyze_font_sizes(self, doc: fitz.Document) -> Dict[float, int]:
        """
        Analyze font sizes in the document to determine heading levels.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            Dictionary mapping font sizes to heading levels
        """
        font_sizes = set()
        
        # Collect all font sizes from the document
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_size = span["size"]
                            if font_size > self.font_size_threshold:
                                font_sizes.add(font_size)
        
        # Sort font sizes in descending order and assign heading levels
        sorted_sizes = sorted(font_sizes, reverse=True)
        font_to_level = {}
        
        for i, size in enumerate(sorted_sizes[:6]):  # Max 6 heading levels
            font_to_level[size] = i + 1
        
        self.logger.info(f"Detected font sizes for headings: {font_to_level}")
        return font_to_level
    
    def _is_heading_text(self, text: str) -> bool:
        """
        Check if text content suggests it's a heading (conservative approach).

        Args:
            text: Text content to analyze

        Returns:
            True if text appears to be a heading
        """
        text = text.strip()

        # Only detect clear heading patterns to preserve original content
        # Check for explicit section numbering patterns
        section_patterns = [
            r'^\d+\.?\s+[A-Z]',  # "1. Introduction" or "1 Introduction"
            r'^\d+\.\d+\.?\s+[A-Z]',  # "1.1. Overview" or "1.1 Overview"
            r'^[A-Z][A-Z\s&/]{3,}$',  # All caps text like "INTRODUCTION", "BUSINESS REQUIREMENTS"
        ]

        for pattern in section_patterns:
            if re.match(pattern, text):
                return True

        # Check for title case patterns that are clearly headings
        title_patterns = [
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]*)*:?$',  # Title case like "Business Requirements"
            r'^Abstract$',  # Common academic paper sections
            r'^Conclusion$',
            r'^References$',
            r'^Acknowledgments?$',
        ]

        for pattern in title_patterns:
            if re.match(pattern, text):
                return True

        return False
    
    def _extract_text_with_formatting(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """
        Extract text with formatting information from PDF.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            List of text blocks with formatting information
        """
        text_blocks = []
        font_to_level = self._analyze_font_sizes(doc)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:  # Text block
                    block_text = ""
                    block_font_size = 0
                    is_bold = False
                    
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            span_text = span["text"]
                            font_size = span["size"]
                            font_flags = span["flags"]
                            
                            # Check if text is bold (font flags & 16)
                            if font_flags & 16:
                                is_bold = True
                            
                            # Track the largest font size in the block
                            if font_size > block_font_size:
                                block_font_size = font_size
                            
                            line_text += span_text
                        
                        block_text += line_text + "\n"
                    
                    block_text = block_text.strip()
                    if block_text:
                        # Determine if this is a heading
                        is_heading = (
                            block_font_size in font_to_level or
                            (is_bold and self._is_heading_text(block_text)) or
                            self._is_heading_text(block_text)
                        )
                        
                        heading_level = font_to_level.get(block_font_size, 2) if is_heading else 0
                        
                        text_blocks.append({
                            'text': block_text,
                            'font_size': block_font_size,
                            'is_bold': is_bold,
                            'is_heading': is_heading,
                            'heading_level': heading_level,
                            'page': page_num + 1
                        })
        
        return text_blocks
    
    def _convert_text_blocks_to_markdown(self, text_blocks: List[Dict[str, Any]], page_images: Optional[Dict[int, List[Dict[str, Any]]]] = None) -> str:
        """
        Convert extracted text blocks to Markdown format.

        Args:
            text_blocks: List of text blocks with formatting information
            page_images: Dictionary mapping page numbers to image info lists

        Returns:
            Markdown content as string
        """
        markdown_content = ""
        current_page = 0

        for block in text_blocks:
            text = block['text']
            block_page = block.get('page', 1)

            # If we've moved to a new page, insert any images from the previous page
            if page_images and block_page > current_page:
                for page_num in range(current_page, block_page):
                    if page_num in page_images and page_images[page_num]:
                        for image_info in page_images[page_num]:
                            image_markdown = self._convert_image_to_markdown(image_info['path'])
                            markdown_content += image_markdown
                current_page = block_page

            if block['is_heading']:
                # Convert to markdown heading
                heading_level = block['heading_level']
                heading_prefix = "#" * heading_level
                section_number = self._update_section_counter(heading_level)
                markdown_content += f"{heading_prefix} {section_number}{text}\n\n"
            else:
                # Regular paragraph
                # Handle potential table-like content
                if self._looks_like_table(text):
                    markdown_content += self._convert_table_like_text(text)
                else:
                    # Apply bold formatting if the entire block was bold
                    if block['is_bold'] and not block['is_heading']:
                        # Clean up the text and ensure proper bold formatting
                        text = text.strip()
                        if text:
                            text = f"**{text}**"

                    markdown_content += f"{text}\n\n"

        # Add any remaining images from the last page
        if page_images:
            for page_num in range(current_page, len(page_images)):
                if page_num in page_images and page_images[page_num]:
                    for image_info in page_images[page_num]:
                        image_markdown = self._convert_image_to_markdown(image_info['path'])
                        markdown_content += image_markdown

        return markdown_content
    
    def _looks_like_table(self, text: str) -> bool:
        """
        Check if text looks like tabular data.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text appears to be tabular
        """
        lines = text.split('\n')
        if len(lines) < 2:
            return False
        
        # Check for consistent column separators
        separators = ['\t', '  ', ' | ', '|']
        for sep in separators:
            if all(sep in line for line in lines if line.strip()):
                return True
        
        return False
    
    def _convert_table_like_text(self, text: str) -> str:
        """
        Convert table-like text to Azure DevOps Wiki-compatible Markdown table format.

        Args:
            text: Table-like text

        Returns:
            Markdown table format
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return text + "\n\n"

        # Try to detect separator
        separator = None
        for sep in ['\t', ' | ', '|', '  ']:
            if sep in lines[0]:
                separator = sep
                break

        if not separator:
            return text + "\n\n"

        # Extract table data
        table_data = []
        for line in lines:
            cells = [cell.strip() for cell in line.split(separator)]
            table_data.append(cells)

        # Use the Azure DevOps compatible table creation method
        return self._create_azure_devops_table(table_data)

    def _extract_images_from_pdf_document(self, doc: fitz.Document) -> Dict[int, List[Dict[str, Any]]]:
        """
        Extract embedded images from PDF document and save them as temporary files.

        Args:
            doc: PyMuPDF document object

        Returns:
            Dictionary mapping page numbers to lists of image info dictionaries
        """
        page_images: Dict[int, List[Dict[str, Any]]] = {}

        if not self.extract_images:
            return page_images

        try:
            # Create temp directory for images
            temp_dir = None

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_images[page_num] = []

                # Get list of images on this page
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    try:
                        # Get image xref (reference number)
                        xref = img[0]

                        # Extract image data
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Create temp directory if needed
                        if not temp_dir:
                            temp_dir = self._create_temp_image_dir()

                        # Create filename
                        img_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                        temp_img_path = temp_dir / img_filename

                        # Write image to temp file
                        with open(temp_img_path, 'wb') as f:
                            f.write(image_bytes)

                        # Store image info with position
                        image_info = {
                            'path': temp_img_path,
                            'filename': img_filename,
                            'bbox': img[1:5] if len(img) > 4 else None,  # Bounding box if available
                            'page': page_num + 1
                        }
                        page_images[page_num].append(image_info)

                        self.logger.info(f"Extracted image: {img_filename} from page {page_num + 1}")

                    except Exception as e:
                        self.logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {str(e)}")

            total_images = sum(len(images) for images in page_images.values())
            self.logger.info(f"Extracted {total_images} images from PDF document")

        except Exception as e:
            self.logger.error(f"Failed to extract images from PDF: {str(e)}")

        return page_images

    def _convert_document_to_markdown(self, doc_path: Path) -> str:
        """Convert a PDF document to Markdown format."""
        try:
            # Reset section counters for new document
            self._reset_section_counters()

            self.logger.info(f"Opening PDF document: {doc_path}")
            doc = fitz.open(doc_path)

            # Extract embedded images first (organized by page)
            page_images = self._extract_images_from_pdf_document(doc)

            # Extract text with formatting
            text_blocks = self._extract_text_with_formatting(doc)

            # Convert to markdown with inline images
            markdown_content = self._convert_text_blocks_to_markdown(text_blocks, page_images)

            doc.close()

            return markdown_content

        except Exception as e:
            self.logger.error(f"Error converting PDF document {doc_path}: {str(e)}")
            return f"# Error Converting Document\n\nFailed to convert {doc_path.name}: {str(e)}\n"
