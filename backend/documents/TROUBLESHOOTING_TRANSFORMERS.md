# Troubleshooting: "transformers model/processor is not available"

## Problem
You're getting the error:
```
Image analysis error: the transformers model/processor is not available, 
so Qwen3 analysis cannot be performed.
```

## Root Cause
The HuggingFace `transformers` library needs the `HF_HOME` environment variable to be set **BEFORE** Python imports any HuggingFace libraries. If it's not set early enough, the library will try to use the default cache directory (`~/.cache/huggingface/`) which may have permission issues.

## Solution

### ✅ Option 1: Use the Startup Script (Recommended)
```bash
cd backend
./start_with_transformers.sh
```

This script sets `HF_HOME` before starting Python.

### ✅ Option 2: Set HF_HOME in Shell Before Running
```bash
export HF_HOME=~/huggingface_cache
cd backend
python main.py
```

### ✅ Option 3: Use systemd Service (for production)
If running as a systemd service, add to your service file:

```ini
[Service]
Environment="HF_HOME=/home/user/huggingface_cache"
Environment="QWEN_BACKEND=transformers"
ExecStart=/path/to/python /path/to/backend/main.py
```

### ✅ Option 4: Set in Docker/Container
If using Docker, add to your Dockerfile or docker-compose.yml:

```dockerfile
ENV HF_HOME=/app/huggingface_cache
ENV QWEN_BACKEND=transformers
```

## Verification Steps

### Step 1: Verify Model Files Exist
```bash
ls -la ~/huggingface_cache/models--Qwen--Qwen3-VL-8B-Instruct/
```

You should see:
- `blobs/` directory
- `snapshots/` directory
- `refs/` directory

### Step 2: Verify Processor Files Exist
```bash
ls -la ~/huggingface_cache/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/
```

You should see files like:
- `config.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `preprocessor_config.json`
- `model-*.safetensors`

If processor files are missing, download them:
```bash
export HF_HOME=~/huggingface_cache
python -c "
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    'Qwen/Qwen3-VL-8B-Instruct',
    cache_dir='~/huggingface_cache'
)
print('✅ Processor downloaded')
"
```

### Step 3: Test with Test Script
```bash
cd backend
python test/test_qwen_transformers_full.py
```

Expected output:
```
✅ Converter created
   Backend: transformers
   Model loaded: True
   Processor loaded: True
✅ TEST 1 PASSED
```

### Step 4: Test with Real Image
```bash
cd backend
python test/test_real_image.py input/your_image.png
```

## Common Issues

### Issue 1: "PermissionError at /home/user/.cache/huggingface"
**Cause:** HF_HOME not set before Python starts
**Solution:** Use one of the options above to set HF_HOME before running

### Issue 2: "Model not found" or "local_files_only"
**Cause:** Model files not downloaded
**Solution:** Run the download script:
```bash
export HF_HOME=~/huggingface_cache
cd backend
python scripts/download_qwen3_model.py
```

### Issue 3: "CUDA out of memory"
**Cause:** Not enough GPU memory
**Solutions:**
1. Use fewer GPUs:
   ```bash
   export QWEN_TRANSFORMERS_GPU_DEVICES=4
   ```

2. Use float16 instead of bfloat16:
   ```bash
   export QWEN_TRANSFORMERS_DTYPE=float16
   ```

3. Clear GPU memory:
   ```bash
   # Kill other processes using GPU
   nvidia-smi
   # Find process IDs and kill them
   kill <PID>
   ```

### Issue 4: Model loads but conversion fails
**Cause:** Various reasons - check logs
**Solution:** Run with debug logging:
```bash
export HF_HOME=~/huggingface_cache
cd backend
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)

import os
os.environ['QWEN_BACKEND'] = 'transformers'

from qwen_ocr_service.qwen3_ocr_converter import Qwen3OCRConverter
converter = Qwen3OCRConverter()

# Test conversion
import base64
test_img = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='
result = converter.convert_image_base64_to_markdown(test_img)
print(result)
"
```

## Quick Diagnostic Script

Run this to check your setup:

```bash
cd backend
python -c "
import os
print('Environment Check:')
print(f'  HF_HOME: {os.environ.get(\"HF_HOME\", \"NOT SET\")}')
print(f'  QWEN_BACKEND: {os.environ.get(\"QWEN_BACKEND\", \"NOT SET\")}')
print(f'  Cache dir exists: {os.path.exists(os.path.expanduser(\"~/huggingface_cache\"))}')

cache_dir = os.path.expanduser('~/huggingface_cache/models--Qwen--Qwen3-VL-8B-Instruct')
print(f'  Model dir exists: {os.path.exists(cache_dir)}')

if os.path.exists(cache_dir):
    import glob
    snapshots = glob.glob(f'{cache_dir}/snapshots/*')
    if snapshots:
        files = os.listdir(snapshots[0])
        print(f'  Model files: {len(files)} files')
        print(f'  Has tokenizer: {\"tokenizer_config.json\" in files}')
        print(f'  Has processor: {\"preprocessor_config.json\" in files}')
"
```

## Still Not Working?

If you've tried all the above and it's still not working:

1. **Check the logs** - Look for specific error messages
2. **Verify GPU availability**:
   ```bash
   nvidia-smi
   ```
3. **Try a different backend** temporarily:
   ```bash
   export QWEN_BACKEND=ollama  # or vllm
   ```
4. **Contact support** with:
   - Full error message
   - Output of diagnostic script above
   - GPU information (`nvidia-smi`)
   - Python version (`python --version`)

