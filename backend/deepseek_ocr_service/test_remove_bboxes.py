"""Small sanity tests for _remove_bounding_boxes.

These tests do NOT call the DeepSeek API. They only exercise the
bounding-box / grounding-tag cleanup logic so we can quickly check
that we never end up with an empty markdown string.

Run:
    cd backend
    python -m deepseek_ocr_service.test_remove_bboxes
"""

from pathlib import Path

from deepseek_ocr_service.deepseek_ocr_converter import DeepSeekOCRConverter


def _print_case(title: str, input_text: str, output_text: str) -> None:
    print("=" * 80)
    print(title)
    print("- INPUT:")
    print(repr(input_text))
    print("- OUTPUT:")
    print(repr(output_text))


def main() -> None:
    converter = DeepSeekOCRConverter()

    # Case 1: Normal text with one grounding tag in the middle
    case1 = (
        "Document Title\n"
        "<|ref|>title<|/ref|><|det|>[[100, 200, 300, 400]]<|/det|>\n"
        "Body text line 1.\n"
        "Body text line 2."
    )
    out1 = converter._remove_bounding_boxes(case1)
    _print_case("CASE 1 (text + one tag)", case1, out1)

    # Case 2: Only a grounding tag (pathological case). We should NOT return empty
    # string here; the helper should fall back to the original content.
    case2 = "<|ref|>title<|/ref|><|det|>[[100, 200, 300, 400]]<|/det|>"
    out2 = converter._remove_bounding_boxes(case2)
    _print_case("CASE 2 (only tag; should not become empty)", case2, out2)

    if not out2.strip():
        print("[ERROR] _remove_bounding_boxes returned empty output for CASE 2!")
    else:
        print("[OK] _remove_bounding_boxes kept non-empty output for CASE 2.")


if __name__ == "__main__":
    main()

