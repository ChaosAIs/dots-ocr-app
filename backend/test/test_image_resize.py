#!/usr/bin/env python3
"""Test script to verify image resize functionality."""

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


def create_test_image(width: int, height: int) -> str:
    """Create a test image with specified dimensions and return base64."""
    # Create a simple test image with gradient
    img = Image.new('RGB', (width, height), color='white')
    
    # Add some visual content (gradient)
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


def get_image_dimensions(image_base64: str) -> tuple:
    """Get dimensions from base64 image."""
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(BytesIO(image_bytes))
    return img.size


def test_resize():
    """Test the resize functionality."""
    print("=" * 80)
    print("Testing Image Resize Functionality")
    print("=" * 80)

    # Test 1: Image smaller than threshold (should not resize)
    print("\n📝 Test 1: Small image (1000x800) - should NOT resize")
    small_img = create_test_image(1000, 800)
    original_size = get_image_dimensions(small_img)
    print(f"   Original size: {original_size[0]}x{original_size[1]}")

    resized_img = resize_image_if_needed(small_img, max_dimension=2000)
    new_size = get_image_dimensions(resized_img)
    print(f"   After resize: {new_size[0]}x{new_size[1]}")

    if original_size == new_size:
        print("   ✅ PASS: Image was not resized (as expected)")
    else:
        print("   ❌ FAIL: Image was resized when it shouldn't be")

    # Test 2: Image with width > threshold (should resize maintaining aspect ratio)
    print("\n📝 Test 2: Wide image (3068x1000) - should resize to fit max_dimension")
    wide_img = create_test_image(3068, 1000)
    original_size = get_image_dimensions(wide_img)
    print(f"   Original size: {original_size[0]}x{original_size[1]}")

    max_dim = 2000
    resized_img = resize_image_if_needed(wide_img, max_dimension=max_dim)
    new_size = get_image_dimensions(resized_img)
    print(f"   After resize: {new_size[0]}x{new_size[1]}")

    # Calculate expected size: scale so largest dimension = max_dim
    scale = max_dim / max(original_size)
    expected_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    if new_size == expected_size:
        print(f"   ✅ PASS: Image resized to {new_size[0]}x{new_size[1]} (expected {expected_size[0]}x{expected_size[1]})")
    else:
        print(f"   ❌ FAIL: Expected {expected_size[0]}x{expected_size[1]}, got {new_size[0]}x{new_size[1]}")
    
    # Test 3: Image with height > threshold (should resize maintaining aspect ratio)
    print("\n📝 Test 3: Tall image (1000x3835) - should resize to fit max_dimension")
    tall_img = create_test_image(1000, 3835)
    original_size = get_image_dimensions(tall_img)
    print(f"   Original size: {original_size[0]}x{original_size[1]}")

    max_dim = 2000
    resized_img = resize_image_if_needed(tall_img, max_dimension=max_dim)
    new_size = get_image_dimensions(resized_img)
    print(f"   After resize: {new_size[0]}x{new_size[1]}")

    # Calculate expected size: scale so largest dimension = max_dim
    scale = max_dim / max(original_size)
    expected_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    if new_size == expected_size:
        print(f"   ✅ PASS: Image resized to {new_size[0]}x{new_size[1]} (expected {expected_size[0]}x{expected_size[1]})")
    else:
        print(f"   ❌ FAIL: Expected {expected_size[0]}x{expected_size[1]}, got {new_size[0]}x{new_size[1]}")

    # Test 4: Very large image (both dimensions > threshold)
    print("\n📝 Test 4: Large image (3068x3835) - should resize to fit max_dimension")
    large_img = create_test_image(3068, 3835)
    original_size = get_image_dimensions(large_img)
    print(f"   Original size: {original_size[0]}x{original_size[1]}")

    max_dim = 2000
    resized_img = resize_image_if_needed(large_img, max_dimension=max_dim)
    new_size = get_image_dimensions(resized_img)
    print(f"   After resize: {new_size[0]}x{new_size[1]}")

    # Calculate expected size: scale so largest dimension = max_dim
    scale = max_dim / max(original_size)
    expected_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    if new_size == expected_size:
        print(f"   ✅ PASS: Image resized to {new_size[0]}x{new_size[1]} (expected {expected_size[0]}x{expected_size[1]})")
    else:
        print(f"   ❌ FAIL: Expected {expected_size[0]}x{expected_size[1]}, got {new_size[0]}x{new_size[1]}")
    
    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_resize()

