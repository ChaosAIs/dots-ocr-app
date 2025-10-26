#!/usr/bin/env python3
"""
Base AI Service Interface

This module defines the abstract base class for AI services used in document conversion.
It provides a common interface for different AI providers (OpenAI, OLLAMA, etc.).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import time
import os
from .global_delay_tracker import global_delay_tracker


class BaseAIService(ABC):
    """Abstract base class for AI services."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AI service with configuration.

        Args:
            config: Configuration dictionary containing service-specific settings
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Note: Delay tracking is now handled by global_delay_tracker

    def _apply_request_delay(self):
        """Apply delay between API requests to avoid connection issues."""
        print(f"🔧 _apply_request_delay called for {self.__class__.__name__}")
        service_name = getattr(self, 'service_name', self.__class__.__name__)
        try:
            print(f"🔍 Global delay tracker setting: {global_delay_tracker.get_delay_setting()}s")
            global_delay_tracker.apply_delay(service_name)
            print(f"✅ Delay application completed")
        except Exception as e:
            print(f"❌ Error applying delay: {e}")
            import traceback
            traceback.print_exc()

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the AI service is available and properly configured.
        
        Returns:
            True if the service is available, False otherwise
        """
        pass
    
    @abstractmethod
    def extract_text_from_image(self, image_path: Path, prompt: str) -> str:
        """
        Extract text content from an image using AI vision capabilities.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt to guide the extraction
            
        Returns:
            Extracted text content formatted as markdown
        """
        pass
    
    @abstractmethod
    def get_service_name(self) -> str:
        """
        Get the name of the AI service.
        
        Returns:
            Service name (e.g., "OpenAI", "OLLAMA")
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the model being used.

        Returns:
            Model name (e.g., "gpt-4o", "llava:latest")
        """
        pass

    @abstractmethod
    def analyze_text_content(self, text_content: str, file_type: str = "txt") -> str:
        """
        Analyze plain text content and suggest markdown structure improvements.

        Args:
            text_content: The plain text content to analyze
            file_type: Type of the source file (txt, csv, etc.)

        Returns:
            Improved markdown content with proper structure, sections, and formatting
        """
        pass

    @abstractmethod
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
        pass
    
    def _prepare_image_for_processing(self, image_path: Path) -> Optional[str]:
        """
        Prepare image for processing (base64 encoding, resizing, etc.).
        This is a common utility that can be overridden by specific implementations.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Prepared image data (usually base64 encoded) or None if failed
        """
        try:
            import base64
            from PIL import Image
            import io
            
            # Open and process the image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large (max 2048x2048 for most AI services)
                max_size = self.config.get('max_image_size', 2048)
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    self.logger.info(f"Resized image to {img.size}")
                
                # Convert to base64
                img_byte_arr = io.BytesIO()
                quality = self.config.get('image_quality', 85)
                img.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
                img_byte_arr.seek(0)
                img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                
                self.logger.info(f"Image prepared for processing: {len(img_base64)} characters")
                return img_base64
                
        except Exception as e:
            self.logger.error(f"Error preparing image {image_path}: {str(e)}")
            return None


class AIServiceError(Exception):
    """Custom exception for AI service errors."""
    pass


class AIServiceUnavailableError(AIServiceError):
    """Exception raised when AI service is not available."""
    pass


class AIServiceConfigurationError(AIServiceError):
    """Exception raised when AI service configuration is invalid."""
    pass
