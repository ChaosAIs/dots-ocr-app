#!/usr/bin/env python3
"""Full test for Qwen3 transformers backend with real image."""

import os
import sys
import base64
from pathlib import Path
from io import BytesIO

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment variables for transformers backend
os.environ['IMAGE_ANALYSIS_BACKEND'] = 'transformers'
os.environ['QWEN_TRANSFORMERS_ATTN_IMPL'] = 'eager'
os.environ['QWEN_TRANSFORMERS_DTYPE'] = 'bfloat16'
os.environ['QWEN_TRANSFORMERS_GPU_DEVICES'] = '4,5,6,7'

from qwen_ocr_service.qwen3_ocr_converter import Qwen3OCRConverter


def create_test_image_base64():
    """Create a simple test image and return base64 string."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("⚠️  PIL not installed, using minimal test image")
        # Return a minimal 1x1 red pixel PNG
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    
    # Create a simple image with text
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some text
    text = "Test OCR Document\nLine 1: Hello World\nLine 2: Testing 123"
    draw.text((10, 10), text, fill='black')
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return img_base64


def test_converter_initialization():
    """Test 1: Check if converter initializes properly."""
    print("\n" + "=" * 80)
    print("TEST 1: Converter Initialization")
    print("=" * 80)
    
    try:
        converter = Qwen3OCRConverter()
        print(f"✅ Converter created")
        print(f"   Backend: {converter.backend}")
        print(f"   Model loaded: {converter.model is not None}")
        print(f"   Processor loaded: {converter.processor is not None}")
        
        if converter.model is None or converter.processor is None:
            print("\n❌ FAILED: Model or processor not loaded")
            print("\nDEBUG INFO:")
            print(f"   QWEN_BACKEND: {os.environ.get('QWEN_BACKEND')}")

            # Check if default cache directory exists
            cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
            print(f"   Default cache directory exists: {os.path.exists(cache_dir)}")
            if os.path.exists(cache_dir):
                model_dir = os.path.join(cache_dir, 'models--Qwen--Qwen3-VL-8B-Instruct')
                print(f"   Model directory exists: {os.path.exists(model_dir)}")
            
            return False
        
        print("✅ TEST 1 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_conversion():
    """Test 2: Test actual image conversion."""
    print("\n" + "=" * 80)
    print("TEST 2: Image Conversion")
    print("=" * 80)
    
    try:
        converter = Qwen3OCRConverter()
        
        if converter.model is None or converter.processor is None:
            print("❌ Skipping test - model/processor not loaded")
            return False
        
        # Create test image
        print("📸 Creating test image...")
        img_base64 = create_test_image_base64()
        print(f"   Image base64 length: {len(img_base64)} characters")
        
        # Convert image to markdown
        print("\n🔄 Converting image to markdown...")
        result = converter.convert_image_base64_to_markdown(img_base64)
        
        print("\n📝 Conversion Result:")
        print("-" * 80)
        print(result)
        print("-" * 80)
        
        if "error" in result.lower():
            print("\n❌ TEST 2 FAILED: Conversion returned error")
            return False
        
        print("\n✅ TEST 2 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_markdown_embedded_image():
    """Test 3: Test with markdown embedded image (data URL format)."""
    print("\n" + "=" * 80)
    print("TEST 3: Markdown Embedded Image")
    print("=" * 80)
    
    try:
        converter = Qwen3OCRConverter()
        
        if converter.model is None or converter.processor is None:
            print("❌ Skipping test - model/processor not loaded")
            return False
        
        # Create test image
        img_base64 = create_test_image_base64()
        
        # Test with data URL format (as it appears in markdown)
        print("📸 Testing with data URL format...")
        data_url = f"data:image/png;base64,{img_base64}"
        print(f"   Data URL length: {len(data_url)} characters")
        
        # Extract base64 from data URL (simulating what the parser does)
        if data_url.startswith("data:"):
            # Extract base64 part after the comma
            base64_part = data_url.split(",", 1)[1] if "," in data_url else data_url
            print(f"   Extracted base64 length: {len(base64_part)} characters")
        else:
            base64_part = img_base64
        
        # Convert
        print("\n🔄 Converting markdown embedded image...")
        result = converter.convert_image_base64_to_markdown(base64_part)
        
        print("\n📝 Conversion Result:")
        print("-" * 80)
        print(result)
        print("-" * 80)
        
        if "error" in result.lower():
            print("\n❌ TEST 3 FAILED: Conversion returned error")
            return False
        
        print("\n✅ TEST 3 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("Qwen3 Transformers Backend - Full Test Suite")
    print("=" * 80)
    print("\n📋 Environment Configuration:")
    print(f"   QWEN_BACKEND: {os.environ.get('QWEN_BACKEND')}")
    print(f"   QWEN_TRANSFORMERS_GPU_DEVICES: {os.environ.get('QWEN_TRANSFORMERS_GPU_DEVICES')}")
    print(f"   QWEN_TRANSFORMERS_DTYPE: {os.environ.get('QWEN_TRANSFORMERS_DTYPE')}")
    print(f"   QWEN_TRANSFORMERS_ATTN_IMPL: {os.environ.get('QWEN_TRANSFORMERS_ATTN_IMPL')}")
    
    # Run tests
    results = []
    results.append(("Converter Initialization", test_converter_initialization()))
    
    if results[0][1]:  # Only run other tests if initialization passed
        results.append(("Image Conversion", test_image_conversion()))
        results.append(("Markdown Embedded Image", test_markdown_embedded_image()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)

