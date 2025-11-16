# Qwen3-VL Transformers Backend Setup Guide

This guide explains how to set up and use the transformers backend for Qwen3-VL OCR.

## Prerequisites

1. **Install required libraries:**
   ```bash
   pip install transformers torch
   # Optional for better performance:
   pip install flash-attn --no-build-isolation
   ```

2. **GPU Requirements:**
   - Default configuration uses GPUs 4, 5, 6, 7
   - Requires CUDA-capable GPUs with sufficient VRAM (~16GB for Qwen3-VL-8B)

## Configuration

### Environment Variables

Add these to your `backend/.env` file:

```bash
# Select transformers backend
QWEN_BACKEND=transformers

# Model configuration
QWEN_TRANSFORMERS_MODEL=Qwen/Qwen3-VL-8B-Instruct
QWEN_TRANSFORMERS_CACHE_DIR=~/huggingface_cache
QWEN_TRANSFORMERS_GPU_DEVICES=4,5,6,7
QWEN_TRANSFORMERS_DTYPE=bfloat16
QWEN_TRANSFORMERS_ATTN_IMPL=eager

# Generation parameters
QWEN_TRANSFORMERS_MAX_NEW_TOKENS=16384
QWEN_TRANSFORMERS_TOP_P=0.8
QWEN_TRANSFORMERS_TOP_K=20
QWEN_TRANSFORMERS_REPETITION_PENALTY=1.0
```

### Important: HF_HOME Environment Variable

**CRITICAL:** You must set `HF_HOME` environment variable **BEFORE** importing any HuggingFace libraries to avoid permission issues with the default cache directory.

#### Option 1: Set in shell (recommended for testing)
```bash
export HF_HOME=~/huggingface_cache
python your_script.py
```

#### Option 2: Set in Python script (before imports)
```python
import os
os.environ['HF_HOME'] = os.path.expanduser('~/huggingface_cache')

# Now import HuggingFace libraries
from qwen_ocr_service.qwen3_ocr_converter import Qwen3OCRConverter
```

#### Option 3: Set in systemd service file
```ini
[Service]
Environment="HF_HOME=/home/user/huggingface_cache"
```

## Download Model

### Option 1: Using the download script
```bash
cd backend
export HF_HOME=~/huggingface_cache
python scripts/download_qwen3_model.py
```

### Option 2: Manual download
```bash
export HF_HOME=~/huggingface_cache
python -c "
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os

cache_dir = os.path.expanduser('~/huggingface_cache')
model = Qwen3VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen3-VL-8B-Instruct',
    cache_dir=cache_dir
)
processor = AutoProcessor.from_pretrained(
    'Qwen/Qwen3-VL-8B-Instruct',
    cache_dir=cache_dir
)
print('✅ Model and processor downloaded successfully!')
"
```

## Usage

### Basic Usage
```python
import os
# IMPORTANT: Set HF_HOME before importing
os.environ['HF_HOME'] = os.path.expanduser('~/huggingface_cache')
os.environ['QWEN_BACKEND'] = 'transformers'

from qwen_ocr_service.qwen3_ocr_converter import Qwen3OCRConverter

converter = Qwen3OCRConverter()
result = converter.convert_image_base64_to_markdown(image_base64)
print(result)
```

### Running the Backend Service
```bash
cd backend
export HF_HOME=~/huggingface_cache
export QWEN_BACKEND=transformers
python main.py
```

## Troubleshooting

### Error: "PermissionError at /home/user/.cache/huggingface/hub"
**Solution:** Set `HF_HOME` environment variable before starting the application:
```bash
export HF_HOME=~/huggingface_cache
```

### Error: "Model not found" or "local_files_only"
**Solution:** Download the model first using the download script or manual method above.

### Error: "CUDA out of memory"
**Solutions:**
1. Use fewer GPUs: `QWEN_TRANSFORMERS_GPU_DEVICES=4`
2. Use float16 instead of bfloat16: `QWEN_TRANSFORMERS_DTYPE=float16`
3. Reduce max tokens: `QWEN_TRANSFORMERS_MAX_NEW_TOKENS=8192`

### Warning: "`torch_dtype` is deprecated"
This is a warning from the transformers library itself, not our code. It can be safely ignored.

## Performance Tips

1. **Use Flash Attention 2** (if available):
   ```bash
   pip install flash-attn --no-build-isolation
   QWEN_TRANSFORMERS_ATTN_IMPL=flash_attention_2
   ```

2. **Use multiple GPUs** for faster inference:
   ```bash
   QWEN_TRANSFORMERS_GPU_DEVICES=4,5,6,7
   ```

3. **Use bfloat16** for better performance and memory efficiency:
   ```bash
   QWEN_TRANSFORMERS_DTYPE=bfloat16
   ```

## Comparison with Other Backends

| Backend | Pros | Cons |
|---------|------|------|
| **ollama** | Easy setup, good for development | Requires Ollama server running |
| **vllm** | Fast inference, OpenAI-compatible API | Requires vLLM server setup |
| **transformers** | Direct model access, no server needed | Higher memory usage, slower startup |

Choose the backend that best fits your deployment scenario.

