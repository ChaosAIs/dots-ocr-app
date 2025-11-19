#!/usr/bin/env python3
"""Test script to verify prompt classification happens before image resize."""

import os
import sys
import base64
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PIL import Image
except ImportError:
    print("❌ PIL/Pillow not installed. Please install: pip install Pillow")
    sys.exit(1)

from qwen_ocr_service.qwen3_ocr_converter import Qwen3OCRConverter


def create_test_image(width: int, height: int) -> str:
    """Create a test image with specified dimensions and return base64."""
    img = Image.new('RGB', (width, height), color='white')
    
    # Add some visual content
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            r = int(255 * x / width)
            g = int(255 * y / height)
            b = 128
            pixels[x, y] = (r, g, b)
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def test_classification_before_resize():
    """Test that classification happens on original image, not resized."""
    print("=" * 80)
    print("Testing Prompt Classification Logic (Before vs After Resize)")
    print("=" * 80)
    
    # Set aggressive resize to make the issue obvious
    os.environ["QWEN_TRANSFORMERS_MAX_IMAGE_DIMENSION"] = "150"
    os.environ["QWEN_IMAGE_MIN_DIMENSION_THRESHOLD"] = "200"
    os.environ["QWEN_IMAGE_PIXEL_AREA_THRESHOLD"] = "1000"
    
    # Create converter (we'll use internal methods to test classification)
    converter = Qwen3OCRConverter()
    
    # Test Case 1: Small icon - should be SIMPLE (before and after resize)
    print("\n📝 Test 1: Small icon (80x80)")
    small_icon = create_test_image(80, 80)
    is_simple = converter._is_simple_image(small_icon)
    print(f"   Classification: {'SIMPLE' if is_simple else 'COMPLEX'}")
    print(f"   Expected: SIMPLE")
    print(f"   ✅ PASS" if is_simple else "   ❌ FAIL")
    
    # Test Case 2: Tall image - should be COMPLEX (original), would be SIMPLE if resized first
    print("\n📝 Test 2: Tall image (250x2000) - Critical test case")
    print("   Original: 250x2000 (500K pixels)")
    print("   After resize (max=150): ~19x150 (2.8K pixels)")
    tall_img = create_test_image(250, 2000)
    is_simple = converter._is_simple_image(tall_img)
    print(f"   Classification: {'SIMPLE' if is_simple else 'COMPLEX'}")
    print(f"   Expected: COMPLEX (based on original dimensions)")
    if not is_simple:
        print(f"   ✅ PASS - Correctly classified as COMPLEX")
    else:
        print(f"   ❌ FAIL - Incorrectly classified as SIMPLE (resize happened first!)")
    
    # Test Case 3: Wide banner - should be COMPLEX (original), would be SIMPLE if resized first
    print("\n📝 Test 3: Wide banner (3000x200)")
    print("   Original: 3000x200 (600K pixels)")
    print("   After resize (max=150): 150x10 (1.5K pixels)")
    wide_img = create_test_image(3000, 200)
    is_simple = converter._is_simple_image(wide_img)
    print(f"   Classification: {'SIMPLE' if is_simple else 'COMPLEX'}")
    print(f"   Expected: COMPLEX (based on original dimensions)")
    if not is_simple:
        print(f"   ✅ PASS - Correctly classified as COMPLEX")
    else:
        print(f"   ❌ FAIL - Incorrectly classified as SIMPLE (resize happened first!)")
    
    # Test Case 4: Large image - should be COMPLEX (before and after resize)
    print("\n📝 Test 4: Large image (3068x3835)")
    print("   Original: 3068x3835 (11.7M pixels)")
    print("   After resize (max=150): ~120x150 (18K pixels)")
    large_img = create_test_image(3068, 3835)
    is_simple = converter._is_simple_image(large_img)
    print(f"   Classification: {'SIMPLE' if is_simple else 'COMPLEX'}")
    print(f"   Expected: COMPLEX (based on original dimensions)")
    if not is_simple:
        print(f"   ✅ PASS - Correctly classified as COMPLEX")
    else:
        print(f"   ❌ FAIL - Incorrectly classified as SIMPLE (resize happened first!)")
    
    print("\n" + "=" * 80)
    print("✅ Classification test completed!")
    print("=" * 80)
    print("\nNote: If Test 2, 3, or 4 failed, it means resize is happening BEFORE")
    print("classification, which causes incorrect prompt selection.")


if __name__ == "__main__":
    test_classification_before_resize()

