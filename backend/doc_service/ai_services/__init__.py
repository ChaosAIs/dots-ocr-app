#!/usr/bin/env python3
"""
AI Services Package

This package provides a unified interface for different AI services used in document conversion.
It supports both cloud-based services (OpenAI) and local AI services (OLLAMA).
"""

from .base_ai_service import BaseAIService, AIServiceError, AIServiceUnavailableError, AIServiceConfigurationError
from .openai_service import OpenAIService
from .ollama_service import OllamaService
from .ai_service_factory import AIServiceFactory, ai_service_factory

__all__ = [
    'BaseAIService',
    'OpenAIService', 
    'OllamaService',
    'AIServiceFactory',
    'ai_service_factory',
    'AIServiceError',
    'AIServiceUnavailableError',
    'AIServiceConfigurationError'
]
