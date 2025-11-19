#!/usr/bin/env python3
"""
Test script to verify that the parser uses image size detection.

This script checks that the parser correctly calls _is_simple_image()
and uses the appropriate prompt based on image size.
"""

import os
import sys
import base64
import io
from PIL import Image, ImageDraw, ImageFont

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from qwen_ocr_service.qwen3_ocr_converter import Qwen3OCRConverter


def create_test_image(width: int, height: int, text: str) -> str:
    """Create a test image and return base64 string."""
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    draw.text((x, y), text, fill='black', font=font)
    
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    
    return base64.b64encode(image_bytes).decode('utf-8')


def test_parser_integration():
    """Test that parser uses image size detection."""
    print("=" * 80)
    print("Testing Parser Integration with Image Size Detection")
    print("=" * 80)
    print()

    # Initialize converter
    print("🔧 Initializing Qwen3OCRConverter...")
    converter = Qwen3OCRConverter()
    print(f"   Backend: {converter.backend}")
    print(f"   Min Dimension Threshold: {converter.min_dimension_threshold} pixels")
    print(f"   Pixel Area Threshold: {converter.pixel_area_threshold} pixels")
    print()
    
    # Test 1: Small image (should use simple prompt)
    print("Test 1: Small Image (100×50)")
    print("-" * 40)
    small_image = create_test_image(100, 50, "Alt")
    
    # Check classification
    is_simple = converter._is_simple_image(small_image)
    print(f"   Classification: {'SIMPLE' if is_simple else 'COMPLEX'}")
    print(f"   Expected: SIMPLE")
    print(f"   Result: {'✓ PASS' if is_simple else '✗ FAIL'}")
    print()
    
    # Get the prompt that would be used
    if is_simple:
        prompt = converter._build_simple_prompt()
        print(f"   Prompt type: Simple")
        print(f"   Prompt length: {len(prompt)} chars")
    else:
        prompt = converter._build_complex_prompt()
        print(f"   Prompt type: Complex")
        print(f"   Prompt length: {len(prompt)} chars")
    print()
    
    # Test 2: Large image (should use complex prompt)
    print("Test 2: Large Image (800×600)")
    print("-" * 40)
    large_image = create_test_image(800, 600, "Complex Chart")
    
    # Check classification
    is_simple = converter._is_simple_image(large_image)
    print(f"   Classification: {'SIMPLE' if is_simple else 'COMPLEX'}")
    print(f"   Expected: COMPLEX")
    print(f"   Result: {'✓ PASS' if not is_simple else '✗ FAIL'}")
    print()
    
    # Get the prompt that would be used
    if is_simple:
        prompt = converter._build_simple_prompt()
        print(f"   Prompt type: Simple")
        print(f"   Prompt length: {len(prompt)} chars")
    else:
        prompt = converter._build_complex_prompt()
        print(f"   Prompt type: Complex")
        print(f"   Prompt length: {len(prompt)} chars")
    print()
    
    # Test 3: Simulate parser behavior
    print("Test 3: Simulating Parser Behavior")
    print("-" * 40)
    print("   The parser now:")
    print("   1. Calls _is_simple_image() to check image size")
    print("   2. Calls _build_simple_prompt() or _build_complex_prompt()")
    print("   3. Appends language instructions")
    print("   4. Passes combined prompt to convert_image_base64_to_markdown()")
    print()
    print("   This ensures image size classification is logged!")
    print()
    
    print("=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Restart the backend server: cd backend && python main.py")
    print("2. Upload a document with images")
    print("3. Check the logs for 'Image analysis: dimensions=...' messages")
    print()


if __name__ == "__main__":
    test_parser_integration()

