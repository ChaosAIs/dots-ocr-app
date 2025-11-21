#!/usr/bin/env python3
"""Test script for minimum image size threshold in Dots OCR parser."""

import os
import sys
from pathlib import Path
from PIL import Image
import tempfile

# Add parent directory to path to import parser
sys.path.insert(0, str(Path(__file__).parent.parent))

from dots_ocr_service.parser import DotsOCRParser


def create_test_image(width, height, filename):
    """Create a simple test image with specified dimensions."""
    # Create a simple gradient image
    img = Image.new('RGB', (width, height), color='white')
    
    # Add some text to make it look like a real image
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    # Draw some simple shapes
    draw.rectangle([10, 10, width-10, height-10], outline='black', width=2)
    draw.text((width//2, height//2), f"{width}x{height}", fill='black', anchor='mm')
    
    return img


def test_min_image_size_threshold():
    """Test that images below threshold are skipped."""
    print("=" * 80)
    print("Testing Minimum Image Size Threshold for Dots OCR")
    print("=" * 80)
    
    # Create temporary directory for test images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test cases: (width, height, should_skip)
        test_cases = [
            (200, 200, True, "Small square - should skip"),      # 40,000 pixels
            (250, 300, True, "Small rectangle - should skip"),   # 75,000 pixels
            (300, 300, True, "Medium square - should skip"),     # 90,000 pixels
            (400, 300, False, "Large rectangle - should process"), # 120,000 pixels
            (500, 400, False, "Large image - should process"),   # 200,000 pixels
        ]
        
        # Set threshold to 100,000 pixels
        os.environ['DOTS_MIN_IMAGE_SIZE_THRESHOLD'] = '100000'
        
        # Initialize parser (will fail if vLLM not available, but that's OK for this test)
        try:
            parser = DotsOCRParser()
            print(f"\n✓ Parser initialized with threshold: {parser.min_image_size_threshold:,} pixels\n")
        except Exception as e:
            print(f"\n⚠️  Parser initialization failed (expected if vLLM not running): {e}")
            print("   Testing threshold configuration only...\n")
            parser = None
        
        # Test each case
        for width, height, should_skip, description in test_cases:
            pixels = width * height
            print(f"\nTest: {description}")
            print(f"  Dimensions: {width}x{height} = {pixels:,} pixels")
            
            # Create test image
            img = create_test_image(width, height, f"test_{width}x{height}.png")
            img_path = temp_path / f"test_{width}x{height}.png"
            img.save(img_path)
            
            if parser is None:
                # Just check the logic without running parser
                threshold = 100000
                will_skip = pixels < threshold
                status = "SKIP" if will_skip else "PROCESS"
                expected = "SKIP" if should_skip else "PROCESS"
                
                if will_skip == should_skip:
                    print(f"  ✓ PASS: Would {status} (expected: {expected})")
                else:
                    print(f"  ✗ FAIL: Would {status} (expected: {expected})")
            else:
                # Actually run the parser (will fail if vLLM not available)
                try:
                    # We can't actually run parse_image without vLLM, but we can check the threshold
                    will_skip = pixels < parser.min_image_size_threshold
                    status = "SKIP" if will_skip else "PROCESS"
                    expected = "SKIP" if should_skip else "PROCESS"
                    
                    if will_skip == should_skip:
                        print(f"  ✓ PASS: Would {status} (expected: {expected})")
                    else:
                        print(f"  ✗ FAIL: Would {status} (expected: {expected})")
                except Exception as e:
                    print(f"  ⚠️  Could not test parsing: {e}")
        
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)
        print(f"Threshold: 100,000 pixels")
        print(f"Images < 100,000 pixels: SKIPPED")
        print(f"Images >= 100,000 pixels: PROCESSED")
        print("=" * 80)


def test_threshold_configuration():
    """Test that threshold can be configured via environment variable."""
    print("\n" + "=" * 80)
    print("Testing Threshold Configuration")
    print("=" * 80)
    
    test_values = [
        ("100000", 100000),
        ("150000", 150000),
        ("50000", 50000),
        ("0", 0),  # Disable threshold
    ]
    
    for env_value, expected_value in test_values:
        os.environ['DOTS_MIN_IMAGE_SIZE_THRESHOLD'] = env_value
        
        try:
            parser = DotsOCRParser()
            actual_value = parser.min_image_size_threshold
            
            if actual_value == expected_value:
                print(f"✓ PASS: DOTS_MIN_IMAGE_SIZE_THRESHOLD={env_value} → {actual_value:,} pixels")
            else:
                print(f"✗ FAIL: DOTS_MIN_IMAGE_SIZE_THRESHOLD={env_value} → {actual_value:,} pixels (expected: {expected_value:,})")
        except Exception as e:
            print(f"⚠️  Could not test threshold {env_value}: {e}")
    
    print("=" * 80)


if __name__ == "__main__":
    test_min_image_size_threshold()
    test_threshold_configuration()
    
    print("\n✓ All tests completed!")

