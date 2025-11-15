"""Qwen3 OCR service package.

Helpers for calling a local Qwen3-VL vision model (via Ollama) for image
analysis and markdown generation.
"""

from .qwen3_ocr_converter import Qwen3OCRConverter

__all__ = ["Qwen3OCRConverter"]

