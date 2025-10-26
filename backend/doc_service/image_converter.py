#!/usr/bin/env python3
"""
Image Document to Markdown Converter Service

This service handles conversion of image files to Markdown format using AI vision capabilities.
Supports common image formats: jpg, jpeg, png, gif, bmp, tiff, webp
Uses pluggable AI services (OpenAI, OLLAMA, etc.) for image-to-text conversion.
"""

import sys
import os
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Required packages not installed. Please install them using:")
    print("pip install -r requirements.txt")
    print(f"Missing: {e}")
    sys.exit(1)

from .base_converter import BaseDocumentConverter
from .ai_services import ai_service_factory, AIServiceUnavailableError


class ImageDocumentConverter(BaseDocumentConverter):
    """Converts image files to Markdown format using AI vision capabilities."""

    def __init__(self, ai_service_type: Optional[str] = None):
        """
        Initialize the image converter with AI vision capabilities.

        Args:
            ai_service_type: Type of AI service to use ('openai', 'ollama', or None for auto-detection)
        """
        super().__init__()

        # Load environment variables
        load_dotenv()

        # Initialize AI service
        self.ai_service = None
        try:
            self.ai_service = ai_service_factory.create_service(ai_service_type)
            self.logger.info(f"AI service initialized: {self.ai_service.get_service_name()} with model {self.ai_service.get_model_name()}")
        except AIServiceUnavailableError as e:
            self.logger.warning(f"AI service not available: {e}")
        except Exception as e:
            self.logger.error(f"Error initializing AI service: {e}")

        # Configuration from environment variables (for backward compatibility)
        self.max_image_size_mb = int(os.getenv('IMAGE_MAX_SIZE_MB', '20'))
        self.image_quality = int(os.getenv('IMAGE_QUALITY_COMPRESSION', '85'))

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported image file extensions."""
        return ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp']

    def can_convert(self, file_path: Path) -> bool:
        """Check if this converter can handle image files."""
        if not self.ai_service:
            self.logger.error("AI service not initialized. Check your AI service configuration.")
            return False
        return file_path.suffix.lower() in self.get_supported_extensions()



    def _extract_text_with_ai_vision(self, image_path: Path) -> str:
        """
        Extract text content from image using AI vision API.

        Args:
            image_path: Path to the image file

        Returns:
            Extracted text content formatted as markdown
        """
        if not self.ai_service:
            return f"# Error: AI Vision Not Available\n\nAI service not configured for image: {image_path.name}\n"

        # Create the vision prompt - enhanced for flowchart detection and ASCII conversion
        prompt = """Extract content from this image based on what type of content it contains. Return ONLY the extracted content without any explanations or analysis steps.

If this is a FLOWCHART/PROCESS DIAGRAM/WORKFLOW:
- Convert to ASCII flow diagram using text characters
- Use arrows (→, ↓, ←, ↑) and boxes made with characters like ┌─┐│└─┘
- Show the flow direction clearly
- Include all text labels from boxes/nodes
- Represent decision points with diamond shapes
- Example format:
```
┌─────────────┐
│   Start     │
└──────┬──────┘
       ↓
┌─────────────┐
│  Process A  │
└──────┬──────┘
       ↓
    /\\     /\\
   /  \\   /  \\
  /    \\ /    \\
 / Decision?  \\
 \\            /
  \\          /
   \\        /
    \\      /
     \\    /
      \\  /
       \\/
    Yes ↓  No →
```

If this is a TABLE/FORM:
- Format as Markdown table
- Preserve all data and structure

If this is REGULAR TEXT/DOCUMENT:
- Extract ALL visible text exactly as written
- Preserve structure using appropriate Markdown formatting
- Read left to right, top to bottom for multi-column content

REQUIREMENTS:
- Return ONLY the actual content - no explanations, no analysis steps, no descriptions
- Use [unclear] for illegible text
- For logos or simple images with minimal text, return only the text content
- Skip images that contain no meaningful text content"""

        # Use the AI service to extract text
        return self.ai_service.extract_text_from_image(image_path, prompt)

    def _convert_document_to_markdown(self, doc_path: Path) -> str:
        """Convert an image file to Markdown format using AI vision."""
        try:
            self.logger.info(f"Converting image to markdown: {doc_path}")

            # Extract text using AI vision
            markdown_content = self._extract_text_with_ai_vision(doc_path)

            # Add metadata header
            metadata_header = f"<!-- Converted from image: {doc_path.name} -->\n"
            if self.ai_service:
                service_info = f"{self.ai_service.get_service_name()} {self.ai_service.get_model_name()}"
            else:
                service_info = "AI Vision (service not available)"
            metadata_header += f"<!-- Conversion method: AI Vision ({service_info}) -->\n"
            metadata_header += f"<!-- Original file: {doc_path.absolute()} -->\n\n"

            return metadata_header + markdown_content

        except Exception as e:
            self.logger.error(f"Error converting image document {doc_path}: {str(e)}")
            return f"# Error Converting Image\n\nFailed to convert {doc_path.name}: {str(e)}\n"
