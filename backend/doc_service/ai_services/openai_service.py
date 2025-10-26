#!/usr/bin/env python3
"""
OpenAI AI Service Implementation

This module provides OpenAI-specific implementation of the AI service interface.
"""

from pathlib import Path
from typing import Dict, Any
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .base_ai_service import BaseAIService, AIServiceUnavailableError, AIServiceConfigurationError


class OpenAIService(BaseAIService):
    """OpenAI implementation of the AI service."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI service.
        
        Args:
            config: Configuration dictionary with OpenAI settings
        """
        super().__init__(config)

        # Service identification
        self.service_name = "OpenAI"

        # Extract OpenAI-specific configuration
        self.api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        self.model = config.get('model', 'gpt-4o')
        self.max_tokens = config.get('max_tokens', 4096)
        self.temperature = config.get('temperature', 0.1)
        self.base_url = config.get('base_url') or os.getenv('OPENAI_BASE_URL')
        
        # Initialize client
        self.client = None
        if self.api_key and OpenAI:
            try:
                client_kwargs = {'api_key': self.api_key}
                if self.base_url:
                    client_kwargs['base_url'] = self.base_url
                    
                self.client = OpenAI(**client_kwargs)
                self.logger.info("OpenAI client initialized successfully")
            except Exception as e:
                self.logger.error(f"Error initializing OpenAI client: {e}")
                self.client = None
    
    def is_available(self) -> bool:
        """Check if OpenAI service is available."""
        if not OpenAI:
            self.logger.error("OpenAI library not installed")
            return False
            
        if not self.api_key:
            self.logger.error("OpenAI API key not configured")
            return False
            
        if not self.client:
            self.logger.error("OpenAI client not initialized")
            return False
            
        return True
    
    def get_service_name(self) -> str:
        """Get the service name."""
        return "OpenAI"
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model
    
    def extract_text_from_image(self, image_path: Path, prompt: str) -> str:
        """
        Extract text from image using OpenAI Vision API.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt to guide the extraction
            
        Returns:
            Extracted text content formatted as markdown
        """
        if not self.is_available():
            raise AIServiceUnavailableError("OpenAI service is not available")
        
        try:
            # Apply delay before API request to avoid connection issues
            self._apply_request_delay()

            # Prepare image for API
            image_base64 = self._prepare_image_for_processing(image_path)
            if not image_base64:
                return f"# Error Processing Image\n\nFailed to prepare image: {image_path.name}\n"

            # Make API call to OpenAI Vision
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Extract the response content
            if response.choices and response.choices[0].message.content:
                extracted_content = response.choices[0].message.content.strip()
                self.logger.info(f"Successfully extracted {len(extracted_content)} characters from {image_path.name}")
                return extracted_content
            else:
                self.logger.warning(f"No content extracted from {image_path.name}")
                return f"# {image_path.stem}\n\nNo text content could be extracted from this image.\n"
                
        except Exception as e:
            # Enhanced error logging with more details
            error_details = []
            error_details.append(f"Error Type: {type(e).__name__}")
            error_details.append(f"Error Message: {str(e)}")

            # Check for specific OpenAI error attributes
            if hasattr(e, 'response'):
                if hasattr(e.response, 'status_code'):
                    error_details.append(f"HTTP Status: {e.response.status_code}")
                if hasattr(e.response, 'text'):
                    error_details.append(f"Response Text: {e.response.text}")

            if hasattr(e, 'code'):
                error_details.append(f"Error Code: {e.code}")

            if hasattr(e, 'body'):
                error_details.append(f"Error Body: {e.body}")

            # Log detailed error information
            detailed_error = " | ".join(error_details)
            self.logger.error(f"Error extracting text from image {image_path}: {detailed_error}")

            return f"# Error Extracting Text\n\nFailed to extract text from {image_path.name}: {str(e)}\n"

    def analyze_text_content(self, text_content: str, file_type: str = "txt") -> str:
        """
        Analyze plain text content and suggest markdown structure improvements.

        Args:
            text_content: The plain text content to analyze
            file_type: Type of the source file (txt, csv, etc.)

        Returns:
            Improved markdown content with proper structure, sections, and formatting
        """
        if not self.is_available():
            raise AIServiceUnavailableError("OpenAI service is not available")

        try:
            # Create a prompt based on file type
            if file_type.lower() == "csv":
                prompt = """Analyze the following CSV content and convert it to a well-formatted markdown table.
Include proper headers, alignment, and any necessary formatting. If there are multiple tables or sections,
organize them appropriately with headers.

CSV Content:
"""
            else:
                prompt = """Analyze the following plain text content and convert it to well-structured markdown.
Apply proper sections, headings, formatting, and organization to make it more readable and professional.
Add a summary if appropriate, organize content into logical sections, and improve the overall structure
while preserving all original information.

Text Content:
"""

            # Make API call to OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt + text_content
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            # Extract the response content
            if response.choices and response.choices[0].message.content:
                analyzed_content = response.choices[0].message.content.strip()
                self.logger.info(f"Successfully analyzed {len(text_content)} characters of {file_type} content")
                return analyzed_content
            else:
                self.logger.warning(f"No analysis result for {file_type} content")
                return f"# Content Analysis Failed\n\nCould not analyze the provided {file_type} content.\n\n```\n{text_content}\n```\n"

        except Exception as e:
            self.logger.error(f"Error analyzing {file_type} content: {str(e)}")
            return f"# Error Analyzing Content\n\nFailed to analyze {file_type} content: {str(e)}\n\n```\n{text_content}\n```\n"

    def analyze_text_chunk(self, text_content: str, file_type: str, chunk_index: int,
                          total_chunks: int, chunk_metadata: dict = None) -> str:
        """
        Analyze a chunk of text content with awareness of its position in the larger document.

        Args:
            text_content: The text chunk content to analyze
            file_type: Type of the source file (txt, csv, etc.)
            chunk_index: Index of this chunk (0-based)
            total_chunks: Total number of chunks in the document
            chunk_metadata: Additional metadata about the chunk

        Returns:
            Improved markdown content for this chunk
        """
        if not self.is_available():
            raise AIServiceUnavailableError("OpenAI service is not available")

        try:
            # Create chunk-aware prompts
            if file_type.lower() == "csv":
                if chunk_index == 0 and total_chunks == 1:
                    # Single chunk CSV
                    prompt = """Analyze the following CSV content and convert it to a well-formatted markdown table.
Include proper headers, alignment, and any necessary formatting. If there are multiple tables or sections,
organize them appropriately with headers.

CSV Content:
"""
                elif chunk_index == 0:
                    # First chunk of multi-chunk CSV
                    prompt = f"""This is the first part (chunk 1 of {total_chunks}) of a large CSV dataset.
Convert this chunk to a well-formatted markdown table with proper headers and alignment.
Since this is part of a larger dataset, focus on creating a clean table structure that can be combined with other parts.

CSV Content (Part 1 of {total_chunks}):
"""
                elif chunk_index == total_chunks - 1:
                    # Last chunk of multi-chunk CSV
                    prompt = f"""This is the final part (chunk {chunk_index + 1} of {total_chunks}) of a large CSV dataset.
Convert this chunk to a well-formatted markdown table. This continues from previous parts, so maintain consistent formatting.
Do not repeat headers - just provide the data rows in table format.

CSV Content (Part {chunk_index + 1} of {total_chunks}):
"""
                else:
                    # Middle chunk of multi-chunk CSV
                    prompt = f"""This is part {chunk_index + 1} of {total_chunks} of a large CSV dataset.
Convert this chunk to a well-formatted markdown table. This continues from previous parts, so maintain consistent formatting.
Do not repeat headers - just provide the data rows in table format.

CSV Content (Part {chunk_index + 1} of {total_chunks}):
"""
            else:
                # Text content chunking
                if chunk_index == 0 and total_chunks == 1:
                    # Single chunk text
                    prompt = """Analyze the following plain text content and convert it to well-structured markdown.
Apply proper sections, headings, formatting, and organization to make it more readable and professional.
Add a summary if appropriate, organize content into logical sections, and improve the overall structure
while preserving all original information.

Text Content:
"""
                elif chunk_index == 0:
                    # First chunk of multi-chunk text
                    prompt = f"""This is the beginning (part 1 of {total_chunks}) of a large document.
Analyze this content and convert it to well-structured markdown. Create appropriate headings, sections, and formatting.
Since this is the first part, include a document title and introduction if appropriate. The content continues in subsequent parts.

Text Content (Part 1 of {total_chunks}):
"""
                elif chunk_index == total_chunks - 1:
                    # Last chunk of multi-chunk text
                    prompt = f"""This is the final part (part {chunk_index + 1} of {total_chunks}) of a large document.
Convert this content to well-structured markdown, maintaining consistency with the previous parts.
Add a conclusion or summary if appropriate since this is the end of the document.

Text Content (Part {chunk_index + 1} of {total_chunks}):
"""
                else:
                    # Middle chunk of multi-chunk text
                    prompt = f"""This is part {chunk_index + 1} of {total_chunks} of a large document.
Convert this content to well-structured markdown, maintaining consistency with previous parts.
Focus on clear section organization and proper formatting.

Text Content (Part {chunk_index + 1} of {total_chunks}):
"""

            # Add metadata information if available
            if chunk_metadata:
                if file_type.lower() == "csv" and 'row_start' in chunk_metadata:
                    prompt += f"\n\nNote: This chunk contains rows {chunk_metadata['row_start']} to {chunk_metadata['row_end']} of {chunk_metadata['total_rows']} total rows."

            # Make API call to OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt + text_content
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            # Extract the response content
            if response.choices and response.choices[0].message.content:
                analyzed_content = response.choices[0].message.content.strip()
                self.logger.info(f"Successfully analyzed chunk {chunk_index + 1}/{total_chunks} of {file_type} content ({len(text_content)} characters)")
                return analyzed_content
            else:
                self.logger.warning(f"No analysis result for chunk {chunk_index + 1}/{total_chunks} of {file_type} content")
                return f"# Content Analysis Failed (Chunk {chunk_index + 1}/{total_chunks})\n\nCould not analyze this part of the {file_type} content.\n\n```\n{text_content}\n```\n"

        except Exception as e:
            self.logger.error(f"Error analyzing chunk {chunk_index + 1}/{total_chunks} of {file_type} content: {str(e)}")
            return f"# Error Analyzing Content (Chunk {chunk_index + 1}/{total_chunks})\n\nFailed to analyze this part of {file_type} content: {str(e)}\n\n```\n{text_content}\n```\n"
