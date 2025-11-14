#!/usr/bin/env python3
"""Basic tests for Gemma3 OCR image analysis integration.

These tests are lightweight and focus on verifying that:
- The Gemma3OCRConverter can be instantiated.
- The DotsOCRParser can call the Gemma3 analysis helper on markdown content.

They do **not** perform real network calls to an Ollama instance; instead, we
patch the converter method to return a deterministic string.
"""

import os
from pathlib import Path
from typing import Any

from dots_ocr_service.parser import DotsOCRParser
from gemma_ocr_service.gemma3_ocr_converter import Gemma3OCRConverter


class DummyGemmaConverter(Gemma3OCRConverter):
    """Test double that avoids real HTTP calls."""

    def __init__(self) -> None:  # pragma: no cover - trivial
        # Do not call super().__init__ to avoid reading env / network setup
        pass

    def convert_image_base64_to_markdown(self, image_base64: str, prompt: Any | None = None) -> str:  # type: ignore[override]
        # Return a deterministic text so we can assert on it
        return "This is a dummy analysis for testing."


def test_gemma3_converter_initialization() -> None:
    """Gemma3OCRConverter should be instantiable with default configuration."""
    converter = Gemma3OCRConverter()
    assert converter is not None


def test_add_image_analysis_to_markdown(monkeypatch) -> None:
    """DotsOCRParser._add_image_analysis_to_markdown should inject analysis."""

    parser = DotsOCRParser(model_path=None, ollama=None)

    # Patch in the dummy converter to avoid real network calls
    parser.gemma3_converter = DummyGemmaConverter()

    # Simple markdown with a single base64 image
    fake_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAUA"  # truncated; content not used by dummy
    md_input = (
        "Some text before image.\n\n"
        f"![](data:image/png;base64,{fake_base64})\n\n"
        "Some text after image.\n"
    )

    md_output = parser._add_image_analysis_to_markdown(md_input)

    # The output should contain the analysis section inserted before the image
    assert "### Image Analysis" in md_output
    assert "This is a dummy analysis for testing." in md_output
    # Ensure the original image tag is still present
    assert "![](data:image/png;base64," in md_output

