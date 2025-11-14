"""Gemma3 OCR service package.

This package provides helpers for calling a local Gemma3 model (via Ollama)
for image analysis and markdown generation.
"""

from .gemma3_ocr_converter import Gemma3OCRConverter

__all__ = ["Gemma3OCRConverter"]

