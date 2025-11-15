#!/usr/bin/env python3
"""Tests for Qwen3 OCR image analysis integration."""

from typing import Any

from dots_ocr_service.parser import DotsOCRParser
from qwen_ocr_service.qwen3_ocr_converter import Qwen3OCRConverter


class DummyQwenConverter(Qwen3OCRConverter):
    """Test double that avoids real HTTP calls."""

    def __init__(self) -> None:  # pragma: no cover - trivial
        # Do not call super().__init__ to avoid reading env / network setup
        pass

    def convert_image_base64_to_markdown(
        self,
        image_base64: str,
        prompt: Any | None = None,
    ) -> str:  # type: ignore[override]
        # Return a deterministic text so we can assert on it
        return "This is a dummy Qwen3 analysis for testing."


def test_qwen3_converter_initialization() -> None:
    """Qwen3OCRConverter should be instantiable with default configuration."""
    converter = Qwen3OCRConverter()
    assert converter is not None


def test_add_image_analysis_to_markdown_with_qwen(monkeypatch) -> None:
    """DotsOCRParser._add_image_analysis_to_markdown should inject Qwen3 analysis."""

    # Force parser to use Qwen3 backend
    monkeypatch.setenv("IMAGE_ANALYSIS_BACKEND", "qwen3")

    parser = DotsOCRParser()

    # Patch in the dummy converter to avoid real network calls
    parser.qwen3_converter = DummyQwenConverter()

    # Simple markdown with a single base64 image
    fake_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAUA"  # truncated; content not used by dummy
    md_input = (
        "Some text before image.\n\n"
        f"![](data:image/png;base64,{fake_base64})\n\n"
        "Some text after image.\n"
    )

    md_output = parser._add_image_analysis_to_markdown(md_input)

    # The output should contain the analysis section inserted before the image.
    expected_heading = "#### This is a dummy Qwen3 analysis for testing."
    assert expected_heading in md_output

    # Ensure the analysis block appears before the image tag in the output
    assert md_output.index(expected_heading) < md_output.index("![](data:image/png;base64,")

