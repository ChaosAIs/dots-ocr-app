#!/usr/bin/env python3
"""Download Qwen3-VL model for transformers backend."""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_model():
    """Download the Qwen3-VL model to the default HuggingFace cache directory."""

    print("=" * 80)
    print("Qwen3-VL Model Download Script")
    print("=" * 80)

    # Get configuration from environment
    model_name = os.getenv("QWEN_TRANSFORMERS_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    cache_dir = os.path.expanduser("~/.cache/huggingface")

    print(f"\n📋 Configuration:")
    print(f"   Model: {model_name}")
    print(f"   Cache Directory: {cache_dir}")

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    print(f"\n✅ Cache directory created/verified: {cache_dir}")
    
    # Check if transformers is installed
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        import torch
        print("✅ transformers and torch libraries are installed")
    except ImportError as e:
        print(f"\n❌ Error: Required libraries not installed: {e}")
        print("\nPlease install required libraries:")
        print("   pip install transformers torch")
        print("   pip install flash-attn --no-build-isolation  # optional, for better performance")
        return False
    
    # Download model
    try:
        print(f"\n📥 Downloading model: {model_name}")
        print("   This may take a while (model is ~16GB)...")
        print("   Progress will be shown below:\n")
        
        # Download model
        print("📦 Downloading model weights...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
        )
        print("✅ Model weights downloaded successfully!")

        # Download processor
        print("\n📦 Downloading processor/tokenizer...")
        processor = AutoProcessor.from_pretrained(
            model_name,
        )
        print("✅ Processor/tokenizer downloaded successfully!")
        
        # Verify download
        print(f"\n✅ Model downloaded successfully to: {cache_dir}")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Processor type: {type(processor).__name__}")
        
        # Show cache directory size
        cache_size = sum(f.stat().st_size for f in Path(cache_dir).rglob('*') if f.is_file())
        cache_size_gb = cache_size / (1024 ** 3)
        print(f"   Cache directory size: {cache_size_gb:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n⚠️  Note: This script will download ~16GB of model files.")
    print("   Make sure you have enough disk space and a stable internet connection.\n")
    
    # Ask for confirmation
    user_input = input("Do you want to proceed with the download? [y/N]: ")
    if user_input.lower() != 'y':
        print("\n❌ Download cancelled.")
        sys.exit(0)
    
    success = download_model()
    
    if success:
        print("\n" + "=" * 80)
        print("✅ Download Complete!")
        print("=" * 80)
        print("\nYou can now use the transformers backend by setting in .env:")
        print("   IMAGE_ANALYSIS_BACKEND=transformers")
        print("\nOr test it with:")
        print("   cd backend && python test/test_qwen_transformers.py")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("❌ Download Failed")
        print("=" * 80)
        sys.exit(1)

