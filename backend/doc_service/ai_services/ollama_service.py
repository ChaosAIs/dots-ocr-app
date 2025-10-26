#!/usr/bin/env python3
"""
OLLAMA AI Service Implementation

This module provides OLLAMA-specific implementation of the AI service interface.
OLLAMA is a local AI inference server that can run models like LLaVA for vision tasks.
"""

from pathlib import Path
from typing import Dict, Any
import os
import json

try:
    import requests
except ImportError:
    requests = None

from .base_ai_service import BaseAIService, AIServiceUnavailableError, AIServiceConfigurationError


class OllamaService(BaseAIService):
    """OLLAMA implementation of the AI service."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OLLAMA service.
        
        Args:
            config: Configuration dictionary with OLLAMA settings
        """
        super().__init__(config)
        
        # Extract OLLAMA-specific configuration
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model = config.get('model', 'llava:latest')
        self.timeout = config.get('timeout', 120)  # OLLAMA can be slower than cloud APIs
        self.temperature = config.get('temperature', 0.1)
        
        # Ensure base_url doesn't end with slash
        self.base_url = self.base_url.rstrip('/')
        
        # API endpoints
        self.generate_endpoint = f"{self.base_url}/api/generate"
        self.tags_endpoint = f"{self.base_url}/api/tags"
        
    def is_available(self) -> bool:
        """Check if OLLAMA service is available."""
        if not requests:
            self.logger.error("requests library not installed")
            return False
        
        try:
            # Check if OLLAMA server is running
            response = requests.get(self.tags_endpoint, timeout=5)
            if response.status_code != 200:
                self.logger.error(f"OLLAMA server not responding: {response.status_code}")
                return False
            
            # Check if the specified model is available
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            if self.model not in available_models:
                self.logger.error(f"Model '{self.model}' not found in OLLAMA. Available models: {available_models}")
                self.logger.info("To install the model, run: ollama pull llava:latest")
                return False
            
            self.logger.info(f"OLLAMA service available with model: {self.model}")
            return True
            
        except requests.exceptions.ConnectionError:
            self.logger.error("Cannot connect to OLLAMA server. Make sure OLLAMA is running.")
            self.logger.info("Start OLLAMA with: ollama serve")
            return False
        except Exception as e:
            self.logger.error(f"Error checking OLLAMA availability: {e}")
            return False
    
    def get_service_name(self) -> str:
        """Get the service name."""
        return "OLLAMA"
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model
    
    def extract_text_from_image(self, image_path: Path, prompt: str) -> str:
        """
        Extract text from image using OLLAMA Vision API.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt to guide the extraction
            
        Returns:
            Extracted text content formatted as markdown
        """
        if not self.is_available():
            raise AIServiceUnavailableError("OLLAMA service is not available")
        
        try:
            # Prepare image for API
            image_base64 = self._prepare_image_for_processing(image_path)
            if not image_base64:
                return f"# Error Processing Image\n\nFailed to prepare image: {image_path.name}\n"
            
            # Prepare the request payload for OLLAMA
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": self.temperature
                }
            }
            
            # Make API call to OLLAMA
            self.logger.info(f"Sending request to OLLAMA for image: {image_path.name}")
            response = requests.post(
                self.generate_endpoint,
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code != 200:
                error_msg = f"OLLAMA API error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return f"# Error Extracting Text\n\n{error_msg}\n"
            
            # Parse the response
            response_data = response.json()
            extracted_content = response_data.get('response', '').strip()
            
            if extracted_content:
                self.logger.info(f"Successfully extracted {len(extracted_content)} characters from {image_path.name}")
                return extracted_content
            else:
                self.logger.warning(f"No content extracted from {image_path.name}")
                return f"# {image_path.stem}\n\nNo text content could be extracted from this image.\n"
                
        except requests.exceptions.Timeout:
            error_msg = f"OLLAMA request timed out after {self.timeout} seconds"
            self.logger.error(error_msg)
            return f"# Error Extracting Text\n\n{error_msg}\n"
        except Exception as e:
            self.logger.error(f"Error extracting text from image {image_path}: {str(e)}")
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
            raise AIServiceUnavailableError("OLLAMA service is not available")

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

            # Prepare the request payload for OLLAMA
            payload = {
                "model": self.model,
                "prompt": prompt + text_content,
                "stream": False,
                "options": {
                    "temperature": self.temperature
                }
            }

            # Make API call to OLLAMA
            self.logger.info(f"Sending text analysis request to OLLAMA for {file_type} content")
            response = requests.post(
                self.generate_endpoint,
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code != 200:
                error_msg = f"OLLAMA API error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return f"# Error Analyzing Content\n\n{error_msg}\n\n```\n{text_content}\n```\n"

            # Parse the response
            response_data = response.json()
            analyzed_content = response_data.get('response', '').strip()

            if analyzed_content:
                self.logger.info(f"Successfully analyzed {len(text_content)} characters of {file_type} content")
                return analyzed_content
            else:
                self.logger.warning(f"No analysis result for {file_type} content")
                return f"# Content Analysis Failed\n\nCould not analyze the provided {file_type} content.\n\n```\n{text_content}\n```\n"

        except requests.exceptions.Timeout:
            error_msg = f"OLLAMA request timed out after {self.timeout} seconds"
            self.logger.error(error_msg)
            return f"# Error Analyzing Content\n\n{error_msg}\n\n```\n{text_content}\n```\n"
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
            raise AIServiceUnavailableError("OLLAMA service is not available")

        try:
            # Create chunk-aware prompts (same logic as OpenAI)
            if file_type.lower() == "csv":
                if chunk_index == 0 and total_chunks == 1:
                    prompt = """Analyze the following CSV content and convert it to a well-formatted markdown table.
Include proper headers, alignment, and any necessary formatting.

CSV Content:
"""
                elif chunk_index == 0:
                    prompt = f"""This is the first part (chunk 1 of {total_chunks}) of a large CSV dataset.
Convert this chunk to a well-formatted markdown table with proper headers and alignment.

CSV Content (Part 1 of {total_chunks}):
"""
                elif chunk_index == total_chunks - 1:
                    prompt = f"""This is the final part (chunk {chunk_index + 1} of {total_chunks}) of a large CSV dataset.
Convert this chunk to a markdown table. Do not repeat headers - just provide the data rows.

CSV Content (Part {chunk_index + 1} of {total_chunks}):
"""
                else:
                    prompt = f"""This is part {chunk_index + 1} of {total_chunks} of a large CSV dataset.
Convert this chunk to a markdown table. Do not repeat headers - just provide the data rows.

CSV Content (Part {chunk_index + 1} of {total_chunks}):
"""
            else:
                if chunk_index == 0 and total_chunks == 1:
                    prompt = """Analyze the following plain text content and convert it to well-structured markdown.
Apply proper sections, headings, and formatting while preserving all original information.

Text Content:
"""
                elif chunk_index == 0:
                    prompt = f"""This is the beginning (part 1 of {total_chunks}) of a large document.
Convert to well-structured markdown with appropriate headings and sections.

Text Content (Part 1 of {total_chunks}):
"""
                elif chunk_index == total_chunks - 1:
                    prompt = f"""This is the final part (part {chunk_index + 1} of {total_chunks}) of a large document.
Convert to well-structured markdown, maintaining consistency with previous parts.

Text Content (Part {chunk_index + 1} of {total_chunks}):
"""
                else:
                    prompt = f"""This is part {chunk_index + 1} of {total_chunks} of a large document.
Convert to well-structured markdown, maintaining consistency with previous parts.

Text Content (Part {chunk_index + 1} of {total_chunks}):
"""

            # Prepare the request payload for OLLAMA
            payload = {
                "model": self.model,
                "prompt": prompt + text_content,
                "stream": False,
                "options": {
                    "temperature": self.temperature
                }
            }

            # Make API call to OLLAMA
            self.logger.info(f"Sending chunk {chunk_index + 1}/{total_chunks} analysis request to OLLAMA for {file_type} content")
            response = requests.post(
                self.generate_endpoint,
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code != 200:
                error_msg = f"OLLAMA API error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return f"# Error Analyzing Content (Chunk {chunk_index + 1}/{total_chunks})\n\n{error_msg}\n\n```\n{text_content}\n```\n"

            # Parse the response
            response_data = response.json()
            analyzed_content = response_data.get('response', '').strip()

            if analyzed_content:
                self.logger.info(f"Successfully analyzed chunk {chunk_index + 1}/{total_chunks} of {file_type} content ({len(text_content)} characters)")
                return analyzed_content
            else:
                self.logger.warning(f"No analysis result for chunk {chunk_index + 1}/{total_chunks} of {file_type} content")
                return f"# Content Analysis Failed (Chunk {chunk_index + 1}/{total_chunks})\n\nCould not analyze this part of the {file_type} content.\n\n```\n{text_content}\n```\n"

        except requests.exceptions.Timeout:
            error_msg = f"OLLAMA request timed out after {self.timeout} seconds"
            self.logger.error(error_msg)
            return f"# Error Analyzing Content (Chunk {chunk_index + 1}/{total_chunks})\n\n{error_msg}\n\n```\n{text_content}\n```\n"
        except Exception as e:
            self.logger.error(f"Error analyzing chunk {chunk_index + 1}/{total_chunks} of {file_type} content: {str(e)}")
            return f"# Error Analyzing Content (Chunk {chunk_index + 1}/{total_chunks})\n\nFailed to analyze this part of {file_type} content: {str(e)}\n\n```\n{text_content}\n```\n"
