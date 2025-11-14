#!/usr/bin/env python3
"""
Test script to verify image resizing functionality.
"""

import os
import sys
from PIL import Image

# Add parent directory to path to import main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import _resize_image_if_needed

def test_resize_functionality():
    """Test the image resize function with the large PNG image."""
    
    # Test with the large PNG image
    test_image_path = "input/PP-Overall-Flow.png"
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    # Get original image info
    with Image.open(test_image_path) as img:
        original_width, original_height = img.size
        original_pixels = original_width * original_height
    
    # Get max_pixels from environment
    max_pixels_env = os.getenv('MAX_PIXELS', '8000000')
    if max_pixels_env.lower() == 'none':
        max_pixels = 11289600
    else:
        max_pixels = int(max_pixels_env)
    safe_max_pixels = int(max_pixels * 0.9)

    print(f"\n📊 Original Image Info:")
    print(f"   Path: {test_image_path}")
    print(f"   Size: {original_width} x {original_height}")
    print(f"   Total Pixels: {original_pixels:,}")
    print(f"   Max Allowed (from env): {max_pixels:,}")
    print(f"   Safe Max (90%): {safe_max_pixels:,}")
    print(f"   Exceeds Limit: {'Yes' if original_pixels > safe_max_pixels else 'No'}")
    
    # Create a copy for testing
    test_copy_path = "input/PP-Overall-Flow_test_copy.png"
    with Image.open(test_image_path) as img:
        img.save(test_copy_path)
    
    print(f"\n🔄 Testing resize function...")
    
    # Test resize
    result = _resize_image_if_needed(test_copy_path)
    
    print(f"\n📋 Resize Result:")
    print(f"   Resized: {result['resized']}")
    print(f"   Message: {result['message']}")
    
    if result['resized']:
        print(f"   Original Size: {result['original_size']}")
        print(f"   New Size: {result['new_size']}")
        
        # Verify the resized image
        with Image.open(test_copy_path) as img:
            new_width, new_height = img.size
            new_pixels = new_width * new_height
        
        print(f"\n✅ Verification:")
        print(f"   New dimensions: {new_width} x {new_height}")
        print(f"   New total pixels: {new_pixels:,}")
        print(f"   Within safe limit: {'Yes' if new_pixels <= safe_max_pixels else 'No'}")
        print(f"   Within max limit: {'Yes' if new_pixels <= max_pixels else 'No'}")
        print(f"   Divisible by 28: Width={new_width % 28 == 0}, Height={new_height % 28 == 0}")
        
        # Calculate reduction percentage
        reduction = ((original_pixels - new_pixels) / original_pixels) * 100
        print(f"   Size reduction: {reduction:.1f}%")
        
        # Clean up test copy
        os.remove(test_copy_path)
        print(f"\n🧹 Cleaned up test copy")
        
        return True
    else:
        print(f"\n⚠️  Image was not resized (might already be within limits)")
        os.remove(test_copy_path)
        return True

if __name__ == "__main__":
    print("=" * 60)
    print("Image Resize Functionality Test")
    print("=" * 60)
    
    success = test_resize_functionality()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")
    print("=" * 60)

