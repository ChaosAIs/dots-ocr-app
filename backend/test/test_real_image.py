#!/usr/bin/env python3
"""Test Qwen3 transformers with a real image file."""

import os
import sys
import base64
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure transformers backend
os.environ['QWEN_BACKEND'] = 'transformers'
os.environ['QWEN_TRANSFORMERS_ATTN_IMPL'] = 'eager'
os.environ['QWEN_TRANSFORMERS_DTYPE'] = 'bfloat16'
os.environ['QWEN_TRANSFORMERS_GPU_DEVICES'] = '4,5,6,7'

from qwen_ocr_service.qwen3_ocr_converter import Qwen3OCRConverter


def encode_image_to_base64(image_path):
    """Read image file and encode to base64."""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode('utf-8')


def test_image_file(image_path):
    """Test OCR on a real image file."""
    print("=" * 80)
    print("Qwen3 Transformers - Real Image Test")
    print("=" * 80)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        print("\nPlease provide a valid image path as argument:")
        print(f"   python {sys.argv[0]} <image_path>")
        return False
    
    print(f"\n📸 Image: {image_path}")
    print(f"   Size: {os.path.getsize(image_path) / 1024:.2f} KB")
    
    # Encode image
    print("\n🔄 Encoding image to base64...")
    try:
        img_base64 = encode_image_to_base64(image_path)
        print(f"   Base64 length: {len(img_base64)} characters")
    except Exception as e:
        print(f"❌ Error encoding image: {e}")
        return False
    
    # Initialize converter
    print("\n🔧 Initializing Qwen3 converter...")
    try:
        converter = Qwen3OCRConverter()
        print(f"   Backend: {converter.backend}")
        print(f"   Model loaded: {converter.model is not None}")
        print(f"   Processor loaded: {converter.processor is not None}")
        
        if converter.model is None or converter.processor is None:
            print("\n❌ Error: Model or processor not loaded")
            print("\nTroubleshooting:")
            print("1. Make sure model is downloaded to default cache:")
            print("   ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/")
            print("2. Check if model files exist:")
            print("   python scripts/download_qwen3_model.py")
            return False
    except Exception as e:
        print(f"❌ Error initializing converter: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Convert image
    print("\n🔄 Converting image to markdown...")
    try:
        result = converter.convert_image_base64_to_markdown(img_base64)
        
        print("\n" + "=" * 80)
        print("📝 OCR RESULT")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
        if "error" in result.lower():
            print("\n⚠️  Warning: Result contains error message")
            return False
        
        # Save result to file
        output_path = Path(image_path).with_suffix('.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\n✅ Result saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_real_image.py <image_path>")
        print("\nExample:")
        print("   python test_real_image.py input/sample.png")
        print("   python test_real_image.py ../input/document.jpg")
        
        # Try to find a sample image
        sample_paths = [
            "input/One-Pager-Business-Monthly-Financial-Report.png",
            "../input/One-Pager-Business-Monthly-Financial-Report.png",
        ]
        
        for sample_path in sample_paths:
            if os.path.exists(sample_path):
                print(f"\n📸 Found sample image: {sample_path}")
                user_input = input("Do you want to test with this image? [y/N]: ")
                if user_input.lower() == 'y':
                    success = test_image_file(sample_path)
                    sys.exit(0 if success else 1)
        
        sys.exit(1)
    
    image_path = sys.argv[1]
    success = test_image_file(image_path)
    sys.exit(0 if success else 1)

