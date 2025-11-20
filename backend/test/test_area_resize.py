#!/usr/bin/env python3
"""Test script to verify area-based image resize functionality."""

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

from utils.image_utils import resize_image_if_needed


def create_test_image(width, height):
    """Create a test image with given dimensions and return as base64."""
    img = Image.new('RGB', (width, height), color='red')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def get_image_dimensions(image_base64):
    """Get dimensions of a base64-encoded image."""
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(BytesIO(image_bytes))
    return img.size


def test_area_resize():
    """Test the area-based resize functionality."""
    print("=" * 80)
    print("Testing Area-Based Image Resize Functionality")
    print("=" * 80)

    # Test 1: Image smaller than area threshold (should not resize)
    print("\n📝 Test 1: Small image (1000x800 = 800K pixels) - should NOT resize with max_area=2.56M")
    small_img = create_test_image(1000, 800)
    original_size = get_image_dimensions(small_img)
    original_area = original_size[0] * original_size[1]
    print(f"   Original: {original_size[0]}x{original_size[1]} = {original_area:,} pixels")

    resized_img = resize_image_if_needed(small_img, max_area=2560000)
    new_size = get_image_dimensions(resized_img)
    new_area = new_size[0] * new_size[1]
    print(f"   After resize: {new_size[0]}x{new_size[1]} = {new_area:,} pixels")

    if original_size == new_size:
        print("   ✅ PASS: Image was not resized (as expected)")
    else:
        print("   ❌ FAIL: Image was resized when it shouldn't be")

    # Test 2: Image larger than area threshold (should resize)
    print("\n📝 Test 2: Large image (3068x3835 = 11.8M pixels) - should resize to ~2.56M pixels")
    large_img = create_test_image(3068, 3835)
    original_size = get_image_dimensions(large_img)
    original_area = original_size[0] * original_size[1]
    print(f"   Original: {original_size[0]}x{original_size[1]} = {original_area:,} pixels")

    max_area = 2560000
    resized_img = resize_image_if_needed(large_img, max_area=max_area)
    new_size = get_image_dimensions(resized_img)
    new_area = new_size[0] * new_size[1]
    print(f"   After resize: {new_size[0]}x{new_size[1]} = {new_area:,} pixels")
    print(f"   Target area: {max_area:,} pixels")

    # Check if new area is close to max_area (within 5% tolerance)
    area_ratio = new_area / max_area
    if 0.95 <= area_ratio <= 1.05:
        print(f"   ✅ PASS: Resized area is within 5% of target (ratio: {area_ratio:.3f})")
    else:
        print(f"   ❌ FAIL: Resized area is not close to target (ratio: {area_ratio:.3f})")

    # Check aspect ratio is maintained
    original_aspect = original_size[0] / original_size[1]
    new_aspect = new_size[0] / new_size[1]
    aspect_diff = abs(original_aspect - new_aspect) / original_aspect
    if aspect_diff < 0.01:  # Less than 1% difference
        print(f"   ✅ PASS: Aspect ratio maintained ({original_aspect:.3f} → {new_aspect:.3f})")
    else:
        print(f"   ❌ FAIL: Aspect ratio changed significantly ({original_aspect:.3f} → {new_aspect:.3f})")

    # Test 3: Square image (should maintain square aspect)
    print("\n📝 Test 3: Square image (2000x2000 = 4M pixels) - should resize to ~1600x1600")
    square_img = create_test_image(2000, 2000)
    original_size = get_image_dimensions(square_img)
    original_area = original_size[0] * original_size[1]
    print(f"   Original: {original_size[0]}x{original_size[1]} = {original_area:,} pixels")

    resized_img = resize_image_if_needed(square_img, max_area=2560000)
    new_size = get_image_dimensions(resized_img)
    new_area = new_size[0] * new_size[1]
    print(f"   After resize: {new_size[0]}x{new_size[1]} = {new_area:,} pixels")

    # Check if it's still square
    if abs(new_size[0] - new_size[1]) <= 1:  # Allow 1 pixel difference due to rounding
        print(f"   ✅ PASS: Image is still square")
    else:
        print(f"   ❌ FAIL: Image is no longer square")

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_area_resize()

