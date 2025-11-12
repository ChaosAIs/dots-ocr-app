#!/usr/bin/env python3
"""
Test script for enhanced DeepSeek OCR with intelligent image analysis.

This script demonstrates the improved OCR capabilities that go beyond simple
text extraction to provide context, insights, and explanations.

Usage:
    python test_enhanced_ocr.py <image_path>
    
Example:
    python test_enhanced_ocr.py input/receipt1.png
    python test_enhanced_ocr.py input/One-Pager-Business-Monthly-Financial-Report.png
"""

import sys
import logging
from pathlib import Path
from deepseek_ocr_service.deepseek_ocr_converter import DeepSeekOCRConverter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_enhanced_ocr(image_path: str):
    """
    Test the enhanced OCR with intelligent analysis.
    
    Args:
        image_path: Path to the image file to analyze
    """
    image_path_obj = Path(image_path)
    
    if not image_path_obj.exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    print("=" * 80)
    print("Enhanced DeepSeek OCR - Intelligent Image Analysis")
    print("=" * 80)
    print(f"\nImage: {image_path_obj.name}")
    print(f"Path: {image_path_obj.absolute()}")
    
    # Initialize converter
    converter = DeepSeekOCRConverter(
        api_base="http://localhost:8005/v1",
        model_name="deepseek-ai/DeepSeek-OCR",
        temperature=0.0,
        max_tokens=8192
    )
    
    # Detect image type
    image_type = converter._detect_image_type(image_path_obj)
    print(f"\n🔍 Detected Image Type: {image_type.upper()}")
    
    # Get the optimized prompt
    prompt = converter._get_optimized_prompt(image_type)
    print(f"\n📝 Intelligent Prompt (first 200 chars):")
    print(f"   {prompt[:200]}...")
    
    # Convert with auto-detection enabled (default)
    print(f"\n⚙️  Converting with intelligent analysis...")
    print(f"   This will extract text AND provide context, insights, and explanations.")
    
    try:
        markdown_content = converter.convert_image_to_markdown(
            image_path_obj,
            auto_detect_type=True  # Enable intelligent analysis
        )
        
        print(f"\n✅ Conversion successful!")
        print(f"\n📄 Result Preview (first 500 characters):")
        print("-" * 80)
        print(markdown_content[:500])
        if len(markdown_content) > 500:
            print("...")
        print("-" * 80)
        
        # Save to output file
        output_dir = Path("output") / "enhanced_ocr_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{image_path_obj.stem}_enhanced.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Enhanced OCR Analysis\n\n")
            f.write(f"**Image**: {image_path_obj.name}\n\n")
            f.write(f"**Detected Type**: {image_type}\n\n")
            f.write(f"**Analysis Mode**: Intelligent (Context + Insights + Extraction)\n\n")
            f.write("---\n\n")
            f.write(markdown_content)
        
        print(f"\n💾 Full result saved to: {output_file}")
        print(f"   Total length: {len(markdown_content)} characters")
        
        # Show statistics
        lines = markdown_content.split('\n')
        print(f"\n📊 Statistics:")
        print(f"   - Lines: {len(lines)}")
        print(f"   - Characters: {len(markdown_content)}")
        print(f"   - Words: {len(markdown_content.split())}")
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python test_enhanced_ocr.py <image_path>")
        print("\nExample:")
        print("  python test_enhanced_ocr.py input/receipt1.png")
        print("  python test_enhanced_ocr.py input/One-Pager-Business-Monthly-Financial-Report.png")
        print("\nThis will use intelligent analysis to:")
        print("  - Understand the image context and purpose")
        print("  - Extract all text and data")
        print("  - Provide insights and explanations")
        print("  - Generate structured markdown output")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_enhanced_ocr(image_path)

if __name__ == "__main__":
    main()

