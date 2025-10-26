#!/usr/bin/env python3
"""
Base Document Converter Interface

This module defines the abstract base class for all document converters.
All specific converters (Word, PDF, etc.) should inherit from this base class.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import tempfile
import os


class BaseDocumentConverter(ABC):
    """Abstract base class for document converters."""

    def __init__(self):
        """Initialize the base converter."""
        self.section_counters = [0] * 6  # Support up to 6 heading levels

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Image processing settings
        self.extract_images = True  # Enable image extraction by default
        self.temp_image_dir = None  # Will be set when needed
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of file extensions supported by this converter.
        
        Returns:
            List of supported file extensions (e.g., ['.docx', '.doc'])
        """
        pass
    
    @abstractmethod
    def can_convert(self, file_path: Path) -> bool:
        """
        Check if this converter can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if this converter can handle the file, False otherwise
        """
        pass
    
    @abstractmethod
    def _convert_document_to_markdown(self, doc_path: Path) -> str:
        """
        Convert a document to Markdown format.
        
        Args:
            doc_path: Path to the document to convert
            
        Returns:
            Markdown content as string
        """
        pass
    
    def _reset_section_counters(self) -> None:
        """Reset section counters for a new document."""
        self.section_counters = [0] * 6
    
    def _update_section_counter(self, level: int) -> str:
        """
        Update section counter for the given level and return section number.

        Args:
            level: Heading level (1-6)

        Returns:
            Section number string (e.g., "1.2.3 ")
        """
        # Increment counter for current level
        self.section_counters[level - 1] += 1

        # Reset counters for deeper levels
        for i in range(level, 6):
            self.section_counters[i] = 0

        # Build section number string
        section_parts = []
        for i in range(level):
            if self.section_counters[i] > 0:
                section_parts.append(str(self.section_counters[i]))

        return ".".join(section_parts) + " " if section_parts else ""

    def _create_azure_devops_table(self, data: List[List[str]]) -> str:
        """Create an Azure DevOps Wiki-compatible Markdown table from 2D data array."""
        if not data:
            return ""

        # Remove completely empty rows
        filtered_data = []
        for row in data:
            if any(str(cell).strip() for cell in row):  # Keep row if any cell has content
                filtered_data.append(row)

        if not filtered_data:
            return ""

        # Find the maximum number of columns
        max_cols = max(len(row) for row in filtered_data)

        # Pad all rows to have the same number of columns
        for row in filtered_data:
            while len(row) < max_cols:
                row.append("")

        # Sanitize cell content for Azure DevOps Wiki
        for row in filtered_data:
            for i, cell in enumerate(row):
                row[i] = self._sanitize_cell_for_azure_devops(str(cell))

        # Create the table with Azure DevOps Wiki formatting
        markdown_table = ""

        # Add header row (first row)
        if filtered_data:
            markdown_table += "| " + " | ".join(filtered_data[0]) + " |\n"
            # Add separator row with proper Azure DevOps formatting
            separators = []
            for _ in range(max_cols):
                separators.append("-" * 8)  # Use longer separators for better compatibility
            markdown_table += "|" + "|".join(separators) + "|\n"

            # Add data rows
            for row in filtered_data[1:]:
                markdown_table += "| " + " | ".join(row) + " |\n"

        return markdown_table + "\n"

    def _sanitize_cell_for_azure_devops(self, content: str) -> str:
        """Sanitize cell content for Azure DevOps Wiki Markdown table."""
        if not content:
            return ""

        # Convert to string and handle special characters
        content = str(content).strip()

        # Replace problematic characters for Azure DevOps Wiki
        content = content.replace("|", "&#124;")  # Use HTML entity for pipe characters
        content = content.replace("\n", " ")      # Replace newlines with spaces
        content = content.replace("\r", "")       # Remove carriage returns
        content = content.replace("\t", " ")      # Replace tabs with spaces

        # Clean up multiple consecutive spaces
        content = " ".join(content.split())

        # Handle special markdown characters that might break tables
        content = content.replace("*", "&#42;")   # Escape asterisks
        content = content.replace("_", "&#95;")   # Escape underscores
        content = content.replace("`", "&#96;")   # Escape backticks

        return content

    def _create_azure_devops_mermaid(self, diagram_definition: str, title: str = "") -> str:
        """Create an Azure DevOps Wiki-compatible Mermaid diagram."""
        # Azure DevOps Wiki supports Mermaid with ::: mermaid syntax
        mermaid_block = f":::mermaid\n{diagram_definition.strip()}\n:::\n"

        if title:
            return f"\n### {title}\n\n{mermaid_block}\n"
        else:
            return f"\n{mermaid_block}\n"
    
    def convert_file(self, input_file: Path, output_file: Path) -> bool:
        """
        Convert a single document to Markdown.

        Args:
            input_file: Path to the input document
            output_file: Path where the Markdown file should be saved

        Returns:
            True if conversion successful, False otherwise
        """
        try:
            self.logger.info(f"Converting: {input_file.name}")

            # Check if we can convert this file
            if not self.can_convert(input_file):
                self.logger.error(f"Cannot convert file type: {input_file.suffix}")
                return False

            # Convert document content
            markdown_content = self._convert_document_to_markdown(input_file)

            # Write Markdown file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            self.logger.info(f"Successfully converted: {input_file.name} -> {output_file.name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to convert {input_file.name}: {str(e)}")
            return False
        finally:
            # Always cleanup temporary images
            self._cleanup_temp_images()
    
    def get_converter_info(self) -> Dict[str, Any]:
        """
        Get information about this converter.

        Returns:
            Dictionary with converter information
        """
        return {
            "name": self.__class__.__name__,
            "supported_extensions": self.get_supported_extensions(),
            "supports_section_numbering": True,
            "section_numbering_enabled": True,
            "supports_image_extraction": True
        }

    def _create_temp_image_dir(self) -> Path:
        """
        Create a temporary directory for extracted images.
        Uses solution directory instead of system temp to avoid permission issues.

        Returns:
            Path to the temporary directory
        """
        if not self.temp_image_dir:
            # Create temp directory in current working directory instead of system temp
            import random
            import string

            # Generate a random suffix for the temp directory
            random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            temp_dir_name = f"doc_images_{random_suffix}"

            # Create temp directory in current working directory
            current_dir = Path.cwd()
            self.temp_image_dir = current_dir / "temp" / temp_dir_name

            # Create the directory (including parent 'temp' directory if needed)
            self.temp_image_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Created temporary image directory: {self.temp_image_dir}")
        return self.temp_image_dir

    def _cleanup_temp_images(self) -> None:
        """Clean up temporary image files and directory."""
        if self.temp_image_dir and self.temp_image_dir.exists():
            try:
                # Remove all files in the temp directory
                for file_path in self.temp_image_dir.iterdir():
                    if file_path.is_file():
                        file_path.unlink()

                # Remove the directory
                self.temp_image_dir.rmdir()
                self.logger.info(f"Cleaned up temporary image directory: {self.temp_image_dir}")
                self.temp_image_dir = None
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp images: {str(e)}")

    def _convert_image_to_markdown(self, image_path: Path) -> str:
        """
        Convert an extracted image to markdown using the image converter.

        Args:
            image_path: Path to the image file

        Returns:
            Markdown content extracted from the image, or empty string if extraction failed
        """
        try:
            # Import here to avoid circular imports
            from .image_converter import ImageDocumentConverter

            # Create image converter instance
            image_converter = ImageDocumentConverter()

            # Check if image converter can handle this file
            if not image_converter.can_convert(image_path):
                self.logger.warning(f"Image converter cannot handle file: {image_path}")
                return ""  # Return empty string instead of error comment

            # Convert image to markdown
            image_markdown = image_converter._extract_text_with_ai_vision(image_path)

            # Check if extraction was successful
            if self._is_failed_image_extraction(image_markdown, image_path.name):
                self.logger.info(f"Skipping failed image extraction for: {image_path.name}")
                return ""  # Return empty string for failed extractions

            # Add minimal context comment and place content inline
            context_comment = f"\n<!-- Content extracted from image: {image_path.name} -->\n"

            return context_comment + image_markdown + "\n"

        except Exception as e:
            self.logger.error(f"Failed to convert image {image_path}: {str(e)}")
            return ""  # Return empty string instead of error comment

    def _is_failed_image_extraction(self, content: str, filename: str) -> bool:
        """
        Check if image extraction failed or returned unusable content.

        Args:
            content: Extracted content from image
            filename: Name of the image file

        Returns:
            True if extraction failed or content is unusable
        """
        if not content or not content.strip():
            return True

        # Convert to lowercase for case-insensitive checking
        content_lower = content.lower().strip()

        # List of phrases that indicate failed extraction
        failed_indicators = [
            "[unclear]",
            "i'm unable to extract text",
            "i'm sorry, i can't assist",
            "no text content could be extracted",
            "failed to extract text",
            "error extracting text",
            "error processing image",
            "i cannot read",
            "i can't read",
            "unable to read",
            "cannot extract",
            "no readable text",
            "image appears to be",
            "this image contains",
            "i don't see any text",
            "there is no text",
            "no visible text"
        ]

        # List of phrases that indicate unwanted analysis steps in output
        analysis_indicators = [
            "step 1: determine the image type",
            "step 2: extract content based on type",
            "for regular text:",
            "for flowcharts/process diagrams:",
            "for tables/forms:",
            "- this is a logo",
            "- this is a regular image",
            "- extracted text:"
        ]

        # Check if content contains analysis steps that should be filtered out
        for indicator in analysis_indicators:
            if indicator in content_lower:
                return True

        # Check if content is only failed indicators
        for indicator in failed_indicators:
            if indicator in content_lower:
                # If the content is mostly just the failed indicator, consider it failed
                if len(content_lower.replace(indicator, "").strip()) < 10:
                    return True

        # Check if content is too short to be meaningful (less than 5 characters)
        if len(content.strip()) < 5:
            return True

        # Check if content is just a simple logo text (common patterns)
        simple_logo_patterns = [
            r'^[a-z]{2,5}$',  # Simple 2-5 letter logos like "BDO", "IBM", etc.
            r'^\*\*[a-z]{2,5}\*\*$',  # Bold logos like "**BDO**"
        ]

        import re
        content_clean = content.strip()
        for pattern in simple_logo_patterns:
            if re.match(pattern, content_clean, re.IGNORECASE):
                return True

        return False
