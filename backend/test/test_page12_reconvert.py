#!/usr/bin/env python3
"""
Test script to re-convert page 12 of GSC MA System Blueprint v6.0.pdf
using different OCR services to diagnose the empty table issue.
"""

import sys
import os
from pathlib import Path
import base64

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepseek_ocr_service.deepseek_ocr_converter import DeepSeekOCRConverter
from qwen_ocr_service.qwen3_ocr_converter import Qwen3OCRConverter
from dots_ocr_service.parser import DotsOCRParser


def encode_image_to_base64(image_path: Path) -> str:
    """Encode image to base64 string."""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')


def test_with_deepseek(image_path: Path, output_dir: Path):
    """Test conversion with DeepSeek OCR."""
    print("\n" + "=" * 80)
    print("Testing with DeepSeek OCR")
    print("=" * 80)
    
    try:
        converter = DeepSeekOCRConverter(
            api_base="http://localhost:8005/v1",
            model_name="deepseek-ai/DeepSeek-OCR",
            temperature=0.0,
            max_tokens=8192
        )
        
        output_file = output_dir / "page_12_deepseek.md"
        
        print(f"Converting {image_path.name}...")
        success = converter.convert_file(image_path, output_file)
        
        if success:
            print(f"✅ DeepSeek conversion successful!")
            print(f"   Output: {output_file}")
            
            # Show content
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"   Size: {len(content)} characters")
                print(f"\n📝 First 500 characters:")
                print("-" * 80)
                print(content[:500])
                if len(content) > 500:
                    print("...")
                print("-" * 80)
            return True
        else:
            print(f"❌ DeepSeek conversion failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error with DeepSeek: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_qwen3(image_path: Path, output_dir: Path):
    """Test conversion with Qwen3 OCR."""
    print("\n" + "=" * 80)
    print("Testing with Qwen3 OCR (Transformers)")
    print("=" * 80)
    
    try:
        # Set environment for transformers backend
        os.environ['QWEN_BACKEND'] = 'transformers'
        
        converter = Qwen3OCRConverter()
        
        output_file = output_dir / "page_12_qwen3.md"
        
        print(f"Converting {image_path.name}...")
        
        # Encode image to base64
        base64_image = encode_image_to_base64(image_path)
        
        # Convert
        markdown = converter.convert_image_base64_to_markdown(
            f"data:image/jpeg;base64,{base64_image}"
        )
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        print(f"✅ Qwen3 conversion successful!")
        print(f"   Output: {output_file}")
        print(f"   Size: {len(markdown)} characters")
        print(f"\n📝 First 500 characters:")
        print("-" * 80)
        print(markdown[:500])
        if len(markdown) > 500:
            print("...")
        print("-" * 80)
        return True
            
    except Exception as e:
        print(f"❌ Error with Qwen3: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_dots_ocr(image_path: Path, output_dir: Path):
    """Test conversion with Dots OCR (original method)."""
    print("\n" + "=" * 80)
    print("Testing with Dots OCR (Original)")
    print("=" * 80)
    
    try:
        parser = DotsOCRParser()
        
        output_file = output_dir / "page_12_dots_ocr.md"
        
        print(f"Converting {image_path.name}...")
        
        # Parse single image
        results = parser.parse_file(
            str(image_path),
            prompt_mode="prompt_layout_all_en"
        )
        
        if results and len(results) > 0:
            result = results[0]
            markdown = result.get('markdown', '')
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown)
            
            print(f"✅ Dots OCR conversion successful!")
            print(f"   Output: {output_file}")
            print(f"   Size: {len(markdown)} characters")
            print(f"\n📝 First 500 characters:")
            print("-" * 80)
            print(markdown[:500])
            if len(markdown) > 500:
                print("...")
            print("-" * 80)
            return True
        else:
            print(f"❌ Dots OCR conversion failed - no results!")
            return False
            
    except Exception as e:
        print(f"❌ Error with Dots OCR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    # Paths
    image_path = Path("backend/output/GSC MA System Blueprint v6.0/GSC MA System Blueprint v6.0_page_12.jpg")
    output_dir = Path("backend/test/page12_reconvert_results")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Page 12 Re-conversion Test")
    print("=" * 80)
    print(f"Image: {image_path}")
    print(f"Output directory: {output_dir}")
    
    if not image_path.exists():
        print(f"\n❌ Error: Image file not found: {image_path}")
        return False
    
    # Test with all three methods
    results = {}
    
    # 1. DeepSeek OCR
    results['deepseek'] = test_with_deepseek(image_path, output_dir)
    
    # 2. Qwen3 OCR
    results['qwen3'] = test_with_qwen3(image_path, output_dir)
    
    # 3. Dots OCR (original)
    results['dots_ocr'] = test_with_dots_ocr(image_path, output_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for method, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{method:15s}: {status}")
    
    print(f"\nAll results saved to: {output_dir}")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

