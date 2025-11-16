# Fix: Model Name Logging for Transformers Backend

## Issue
The debug logs were showing the wrong model name when using the transformers backend:
- **Expected:** `model=Qwen/Qwen3-VL-8B-Instruct` (from `.env` file)
- **Actual:** `model=qwen3-vl:32b` (Ollama default model name)

## Root Cause
The `_convert_with_transformers()` method was logging `self.model_name`, which is the model name for the **Ollama/vLLM backends**, not the transformers backend.

The transformers backend uses a separate environment variable `QWEN_TRANSFORMERS_MODEL` and stores it in a local variable `transformers_model`, but this wasn't being saved as an instance variable for later use in logging.

## Solution
1. **Added instance variable** `self.transformers_model_name` to store the transformers model name
2. **Updated initialization** to save the transformers model name when loading the model
3. **Updated logging** in `_convert_with_transformers()` to use `self.transformers_model_name` instead of `self.model_name`

## Changes Made

### File: `backend/qwen_ocr_service/qwen3_ocr_converter.py`

**Change 1: Initialize transformers_model_name (lines 56-80)**
```python
# Transformers model name (set later if using transformers backend)
self.transformers_model_name = None
```

**Change 2: Store transformers model name during initialization (lines 131-137)**
```python
# Get model path/name from environment
# Default to Qwen/Qwen3-VL-8B-Instruct as specified in the task
transformers_model = os.getenv("QWEN_TRANSFORMERS_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
# Store for later use in logging
self.transformers_model_name = transformers_model
```

**Change 3: Use correct model name in logging (lines 552-556)**
```python
self.logger.info(
    "Sending image analysis request to Qwen3 via transformers: model=%s",
    self.transformers_model_name,  # Changed from self.model_name
)
```

## Verification

### Before Fix:
```
INFO:Qwen3OCRConverter:Sending image analysis request to Qwen3 via transformers: model=qwen3-vl:32b
```

### After Fix:
```
INFO:Qwen3OCRConverter:Sending image analysis request to Qwen3 via transformers: model=Qwen/Qwen3-VL-8B-Instruct
```

## Impact
- ✅ **No breaking changes** - only affects log messages
- ✅ **Correct model name** now displayed in logs for transformers backend
- ✅ **Ollama/vLLM backends** unaffected - still use `self.model_name`
- ✅ **All functionality** remains the same

## Testing
Tested with:
```bash
cd backend
python -c "
import os
os.environ['QWEN_BACKEND'] = 'transformers'
from qwen_ocr_service.qwen3_ocr_converter import Qwen3OCRConverter
converter = Qwen3OCRConverter()
print(f'Transformers model: {converter.transformers_model_name}')
print(f'Ollama/vLLM model: {converter.model_name}')
"
```

Output:
```
Transformers model: Qwen/Qwen3-VL-8B-Instruct
Ollama/vLLM model: qwen3-vl:32b
```

## Summary
The logging now correctly shows which model is being used for each backend:
- **Transformers backend:** Shows `Qwen/Qwen3-VL-8B-Instruct` (HuggingFace model)
- **Ollama backend:** Shows `qwen3-vl:32b` (Ollama model tag)
- **vLLM backend:** Shows the model from `QWEN_MODEL` environment variable

This makes debugging and monitoring much clearer! 🎉

