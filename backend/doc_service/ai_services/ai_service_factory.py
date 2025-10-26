#!/usr/bin/env python3
"""
AI Service Factory

This module provides a factory for creating and managing AI service instances.
It handles configuration loading and service selection based on user preferences.
"""

import os
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv

from .base_ai_service import BaseAIService, AIServiceUnavailableError
from .openai_service import OpenAIService
from .ollama_service import OllamaService


class AIServiceFactory:
    """Factory for creating AI service instances."""
    
    def __init__(self):
        """Initialize the AI service factory."""
        self.logger = logging.getLogger(self.__class__.__name__)
        load_dotenv()  # Load environment variables
        
        # Registry of available services
        self._services = {
            'openai': OpenAIService,
            'ollama': OllamaService
        }
    
    def create_service(self, service_type: Optional[str] = None) -> BaseAIService:
        """
        Create an AI service instance.
        
        Args:
            service_type: Type of service to create ('openai', 'ollama', or None for auto-detection)
            
        Returns:
            Configured AI service instance
            
        Raises:
            AIServiceUnavailableError: If no suitable service is available
        """
        # Auto-detect service type if not specified
        if service_type is None:
            service_type = self._detect_preferred_service()
        
        # Normalize service type
        service_type = service_type.lower()
        
        if service_type not in self._services:
            available_services = ', '.join(self._services.keys())
            raise ValueError(f"Unknown service type: {service_type}. Available: {available_services}")
        
        # Get configuration for the service
        config = self._get_service_config(service_type)
        
        # Create the service instance
        service_class = self._services[service_type]
        service = service_class(config)
        
        # Verify the service is available
        if not service.is_available():
            raise AIServiceUnavailableError(f"{service_type.upper()} service is not available")
        
        self.logger.info(f"Created {service_type.upper()} service with model: {service.get_model_name()}")
        return service
    
    def _detect_preferred_service(self) -> str:
        """
        Auto-detect the preferred AI service based on environment configuration.
        
        Returns:
            Preferred service type
        """
        # Check for explicit preference
        ai_service = os.getenv('AI_SERVICE', '').lower()
        if ai_service in self._services:
            self.logger.info(f"Using explicitly configured AI service: {ai_service}")
            return ai_service
        
        # Try OLLAMA first (local AI preference)
        try:
            ollama_config = self._get_service_config('ollama')
            ollama_service = OllamaService(ollama_config)
            if ollama_service.is_available():
                self.logger.info("Auto-detected OLLAMA service as available")
                return 'ollama'
        except Exception as e:
            self.logger.debug(f"OLLAMA not available: {e}")
        
        # Fall back to OpenAI
        try:
            openai_config = self._get_service_config('openai')
            openai_service = OpenAIService(openai_config)
            if openai_service.is_available():
                self.logger.info("Auto-detected OpenAI service as available")
                return 'openai'
        except Exception as e:
            self.logger.debug(f"OpenAI not available: {e}")
        
        # Default to OpenAI if nothing else is available
        self.logger.warning("No AI service auto-detected, defaulting to OpenAI")
        return 'openai'
    
    def _get_service_config(self, service_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific service type.
        
        Args:
            service_type: Type of service ('openai' or 'ollama')
            
        Returns:
            Configuration dictionary
        """
        if service_type == 'openai':
            return {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'model': os.getenv('OPENAI_MODEL', 'gpt-4o'),
                'max_tokens': int(os.getenv('OPENAI_MAX_TOKENS', '4096')),
                'temperature': float(os.getenv('OPENAI_TEMPERATURE', '0.1')),
                'base_url': os.getenv('OPENAI_BASE_URL'),
                'max_image_size': int(os.getenv('IMAGE_MAX_SIZE_PIXELS', '2048')),
                'image_quality': int(os.getenv('IMAGE_QUALITY_COMPRESSION', '85'))
            }
        elif service_type == 'ollama':
            return {
                'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                'model': os.getenv('OLLAMA_MODEL', 'llava:latest'),
                'timeout': int(os.getenv('OLLAMA_TIMEOUT', '120')),
                'temperature': float(os.getenv('OLLAMA_TEMPERATURE', '0.1')),
                'max_image_size': int(os.getenv('IMAGE_MAX_SIZE_PIXELS', '2048')),
                'image_quality': int(os.getenv('IMAGE_QUALITY_COMPRESSION', '85'))
            }
        else:
            raise ValueError(f"Unknown service type: {service_type}")
    
    def list_available_services(self) -> Dict[str, bool]:
        """
        List all available services and their availability status.
        
        Returns:
            Dictionary mapping service names to availability status
        """
        availability = {}
        
        for service_type in self._services:
            try:
                config = self._get_service_config(service_type)
                service = self._services[service_type](config)
                availability[service_type] = service.is_available()
            except Exception as e:
                self.logger.debug(f"Error checking {service_type}: {e}")
                availability[service_type] = False
        
        return availability


# Global factory instance
ai_service_factory = AIServiceFactory()
