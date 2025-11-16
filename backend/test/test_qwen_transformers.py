#!/usr/bin/env python3
"""Test script for Qwen3 OCR with transformers backend."""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen_ocr_service.qwen3_ocr_converter import Qwen3OCRConverter


def test_transformers_backend():
    """Test Qwen3OCRConverter with transformers backend."""
    
    print("=" * 80)
    print("Testing Qwen3 OCR with Transformers Backend")
    print("=" * 80)
    
    # Set environment to use transformers backend
    os.environ["QWEN_BACKEND"] = "transformers"
    
    # Optional: Configure GPU devices (default is 4,5,6,7)
    # os.environ["QWEN_TRANSFORMERS_GPU_DEVICES"] = "4,5,6,7"
    
    # Optional: Configure model (default is Qwen/Qwen3-VL-8B-Instruct)
    # os.environ["QWEN_TRANSFORMERS_MODEL"] = "Qwen/Qwen3-VL-8B-Instruct"
    
    print("\n📋 Configuration:")
    print(f"   Backend: {os.environ.get('QWEN_BACKEND')}")
    print(f"   Model: {os.environ.get('QWEN_TRANSFORMERS_MODEL', 'Qwen/Qwen3-VL-8B-Instruct')}")
    print(f"   GPU Devices: {os.environ.get('QWEN_TRANSFORMERS_GPU_DEVICES', '4,5,6,7')}")
    
    try:
        print("\n🔧 Initializing Qwen3OCRConverter with transformers backend...")
        converter = Qwen3OCRConverter()
        
        if converter.model is None or converter.processor is None:
            print("\n⚠️  Model or processor not loaded. This is expected if:")
            print("   - transformers library is not installed")
            print("   - torch is not installed")
            print("   - Model files are not downloaded")
            print("   - GPUs are not available")
            print("\nTo install dependencies:")
            print("   pip install transformers torch")
            print("   pip install flash-attn --no-build-isolation  # optional, for better performance")
            return False
        
        print("✅ Converter initialized successfully!")
        print(f"   Backend: {converter.backend}")
        print(f"   Model loaded: {converter.model is not None}")
        print(f"   Processor loaded: {converter.processor is not None}")
        
        # Test with a simple base64 image (1x1 red pixel PNG)
        # This is just to test the pipeline, not for actual OCR
        test_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )
        
        print("\n🧪 Testing image analysis with a simple test image...")
        result = converter.convert_image_base64_to_markdown(test_base64)
        
        print("\n📝 Analysis Result:")
        print("-" * 80)
        print(result)
        print("-" * 80)
        
        if "error" in result.lower():
            print("\n⚠️  Analysis returned an error message")
            return False
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backend_switching():
    """Test that backend switching works correctly."""
    
    print("\n" + "=" * 80)
    print("Testing Backend Switching")
    print("=" * 80)
    
    backends = ["ollama", "vllm", "transformers"]
    
    for backend in backends:
        print(f"\n🔄 Testing backend: {backend}")
        os.environ["QWEN_BACKEND"] = backend
        
        try:
            converter = Qwen3OCRConverter()
            print(f"   ✅ Converter initialized with backend: {converter.backend}")
            
            # Check that the correct backend was selected
            assert converter.backend == backend, f"Expected {backend}, got {converter.backend}"
            
        except Exception as e:
            print(f"   ⚠️  Error initializing {backend} backend: {e}")
    
    print("\n✅ Backend switching test completed!")


if __name__ == "__main__":
    print("\nNote: This test requires:")
    print("  - transformers library installed")
    print("  - torch installed")
    print("  - Qwen3-VL-8B-Instruct model downloaded (or internet connection to download)")
    print("  - GPUs 4,5,6,7 available (or modify QWEN_TRANSFORMERS_GPU_DEVICES)")
    print()
    
    # Test backend switching first (doesn't require model loading)
    test_backend_switching()
    
    # Test transformers backend (requires model loading)
    print("\n" + "=" * 80)
    user_input = input("\nDo you want to test the transformers backend? (requires model download) [y/N]: ")
    if user_input.lower() == 'y':
        success = test_transformers_backend()
        sys.exit(0 if success else 1)
    else:
        print("\nSkipping transformers backend test.")
        sys.exit(0)

