#!/usr/bin/env python3
"""
Test script for DeepSeek OCR converter

This script tests the DeepSeek OCR converter by converting a sample image.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from deepseek_ocr_service.deepseek_ocr_converter import DeepSeekOCRConverter

# Load environment variables
load_dotenv()

def test_deepseek_ocr():
    """Test DeepSeek OCR converter"""
    
    # Get configuration from environment
    deepseek_ocr_ip = os.getenv("DEEPSEEK_OCR_IP", "localhost")
    deepseek_ocr_port = os.getenv("DEEPSEEK_OCR_PORT", "8005")
    deepseek_ocr_model = os.getenv("DEEPSEEK_OCR_MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
    deepseek_ocr_temperature = float(os.getenv("DEEPSEEK_OCR_TEMPERATURE", "0.0"))
    deepseek_ocr_max_tokens = int(os.getenv("DEEPSEEK_OCR_MAX_TOKENS", "8192"))
    deepseek_ocr_ngram_size = int(os.getenv("DEEPSEEK_OCR_NGRAM_SIZE", "30"))
    deepseek_ocr_window_size = int(os.getenv("DEEPSEEK_OCR_WINDOW_SIZE", "90"))
    deepseek_ocr_whitelist_str = os.getenv("DEEPSEEK_OCR_WHITELIST_TOKEN_IDS", "128821,128822")
    deepseek_ocr_whitelist = [int(x.strip()) for x in deepseek_ocr_whitelist_str.split(",") if x.strip()]
    
    print("=" * 80)
    print("DeepSeek OCR Converter Test")
    print("=" * 80)
    print(f"API Base: http://{deepseek_ocr_ip}:{deepseek_ocr_port}/v1")
    print(f"Model: {deepseek_ocr_model}")
    print(f"Temperature: {deepseek_ocr_temperature}")
    print(f"Max Tokens: {deepseek_ocr_max_tokens}")
    print("=" * 80)
    
    # Initialize converter
    converter = DeepSeekOCRConverter(
        api_base=f"http://{deepseek_ocr_ip}:{deepseek_ocr_port}/v1",
        model_name=deepseek_ocr_model,
        temperature=deepseek_ocr_temperature,
        max_tokens=deepseek_ocr_max_tokens,
        ngram_size=deepseek_ocr_ngram_size,
        window_size=deepseek_ocr_window_size,
        whitelist_token_ids=deepseek_ocr_whitelist,
    )
    
    # Find a test image in the input directory
    input_dir = Path(__file__).parent / "input"
    test_images = []
    
    for ext in converter.get_supported_extensions():
        test_images.extend(list(input_dir.glob(f"*{ext}")))
    
    if not test_images:
        print("\n❌ No test images found in input directory!")
        print(f"   Please add an image file to: {input_dir}")
        print(f"   Supported formats: {', '.join(converter.get_supported_extensions())}")
        return False
    
    # Use the first image found
    test_image = test_images[0]
    print(f"\n📄 Test Image: {test_image.name}")
    print(f"   Size: {test_image.stat().st_size / 1024:.2f} KB")
    
    # Create output directory
    output_dir = Path(__file__).parent / "output" / test_image.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{test_image.stem}_deepseek.md"
    
    print(f"\n🔄 Converting image to markdown...")
    print(f"   Output: {output_file}")
    
    try:
        # Test 1: Convert with default prompt
        print(f"\n📋 Test 1: Converting with default prompt...")
        success = converter.convert_file(test_image, output_file)

        if success:
            print(f"\n✅ Conversion successful!")
            print(f"   Output file: {output_file}")
            print(f"   Output size: {output_file.stat().st_size / 1024:.2f} KB")

            # Show first few lines of output
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')[:10]
                print(f"\n📝 First 10 lines of output:")
                print("-" * 80)
                for line in lines:
                    print(line)
                if len(content.split('\n')) > 10:
                    print("...")
                print("-" * 80)
        else:
            print(f"\n❌ Conversion failed!")
            return False

        # Test 2: Convert with custom prompt (Free OCR)
        print(f"\n📋 Test 2: Converting with custom prompt (Free OCR)...")
        output_file_custom = output_dir / f"{test_image.stem}_free_ocr.md"
        success_custom = converter.convert_file(
            test_image,
            output_file_custom,
            prompt="Free OCR."
        )

        if success_custom:
            print(f"\n✅ Custom prompt conversion successful!")
            print(f"   Output file: {output_file_custom}")
            print(f"   Output size: {output_file_custom.stat().st_size / 1024:.2f} KB")
        else:
            print(f"\n⚠️  Custom prompt conversion failed (non-critical)")

        return True

    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_deepseek_ocr()
    sys.exit(0 if success else 1)

