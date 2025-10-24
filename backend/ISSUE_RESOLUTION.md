# Issue Resolution: Inference Process Errors

## Issue Summary
**Date**: 2024-10-24  
**Error**: `ERROR:worker_pool:Error in worker 0: Connection error.`  
**Symptom**: Document conversions failing during "Running model inference" phase with HTTP 500 errors from the vLLM inference server

## Root Cause Analysis

### What Happened
1. User uploaded a large image file: `PP-Overall-Flow.png` (3068 x 3835 pixels = **11,765,780 pixels**)
2. The `.env` configuration had `MAX_PIXELS=None`, which disabled image resizing
3. The full-size image was sent to the vLLM inference server
4. The inference server ran out of memory and returned HTTP 500 errors
5. The OpenAI client retried but eventually failed with "Connection error"

### Why It Wasn't a Worker Pool Issue
The error message mentioned "worker 0", which might suggest too many workers. However:
- The diagnostic showed the inference server was running and healthy
- Simple inference tests passed
- The issue only occurred with large images
- The error was "Connection error" from the inference API, not a worker pool error

## Changes Made

### 1. Updated `.env` Configuration

**File**: `backend/.env`

**Changes**:
```env
# Before:
MAX_PIXELS=None
# (No worker limit)

# After:
MAX_PIXELS=8000000
NUM_WORKERS=1
```

**Rationale**:
- `MAX_PIXELS=8000000`: Automatically resize images larger than 8 million pixels (about 2828 x 2828)
  - This is below the default maximum of 11,289,600 pixels
  - Provides a safety margin for the inference server
  - Still maintains good quality for most documents
  
- `NUM_WORKERS=1`: Temporarily reduced to minimize load on inference server
  - Can be increased back to 4 once stability is confirmed
  - Reduces concurrent memory usage on the inference server

### 2. Improved Error Handling

**File**: `backend/ocr_service/model/inference.py`

**Changes**:
```python
# Before:
except requests.exceptions.RequestException as e:
    print(f"request error: {e}")
    return None

# After:
except requests.exceptions.RequestException as e:
    print(f"Request error: {e}")
    raise Exception(f"Inference API request failed: {e}")
except Exception as e:
    print(f"Inference error: {e}")
    raise Exception(f"Inference failed: {e}")
```

**Rationale**:
- Previously, errors were silently caught and `None` was returned
- Now, errors are properly raised with descriptive messages
- This allows the worker pool to properly report errors to the frontend
- Better debugging and error tracking

### 3. Added Diagnostic Tools

**New Files**:
1. `backend/check_inference_server.py` - Diagnostic script to check vLLM server health
2. `backend/TROUBLESHOOTING.md` - Comprehensive troubleshooting guide

## How Image Resizing Works

The `smart_resize()` function in `backend/ocr_service/utils/image_utils.py`:

1. **Checks aspect ratio**: Rejects images with aspect ratio > 200:1
2. **Applies factor constraint**: Ensures dimensions are divisible by 28 (IMAGE_FACTOR)
3. **Enforces pixel limits**:
   - If image > MAX_PIXELS: Scales down to fit within MAX_PIXELS
   - If image < MIN_PIXELS: Scales up to meet MIN_PIXELS
4. **Maintains aspect ratio**: Preserves the original image proportions

**Example for PP-Overall-Flow.png**:
- Original: 3068 x 3835 = 11,765,780 pixels
- With MAX_PIXELS=8000000:
  - Scale factor: sqrt(8000000 / 11765780) ≈ 0.824
  - New size: ~2528 x 3160 = ~7,988,480 pixels
  - Adjusted to factor 28: 2520 x 3164 = 7,973,280 pixels

## Testing the Fix

### 1. Restart the Backend Server
```bash
cd backend
python main.py
```

### 2. Test with the Problematic Image
Try converting `PP-Overall-Flow.png` again through the UI or API.

### 3. Monitor the Logs
Watch for:
- ✓ No HTTP 500 errors
- ✓ Successful image resizing messages
- ✓ Conversion completes successfully

### 4. Check Inference Server Health
```bash
cd backend
python check_inference_server.py
```

## Recommendations

### Short-term (Immediate)
1. ✅ Keep `NUM_WORKERS=1` until stability is confirmed
2. ✅ Keep `MAX_PIXELS=8000000` to prevent memory issues
3. Monitor conversion success rate for the next few days

### Medium-term (Next Week)
1. Test with various image sizes to find optimal MAX_PIXELS value
2. Gradually increase NUM_WORKERS (try 2, then 3, then 4)
3. Monitor inference server memory usage during peak load
4. Consider adding automatic image size validation in the upload endpoint

### Long-term (Future Improvements)
1. Add image size validation in the `/upload` endpoint
   - Warn users about very large images
   - Suggest optimal image sizes
   
2. Implement adaptive worker pool sizing
   - Automatically reduce workers if inference server is struggling
   - Increase workers when server has capacity
   
3. Add monitoring and alerting
   - Track inference server health
   - Alert on repeated failures
   - Monitor memory usage trends

4. Consider image preprocessing pipeline
   - Automatically optimize images on upload
   - Store both original and optimized versions
   - Use optimized version for inference

## Configuration Guidelines

### For Development/Testing (Limited Resources)
```env
NUM_WORKERS=1
MAX_PIXELS=4000000
DPI=150
NUM_THREAD=16
```

### For Production (Dedicated GPU Server)
```env
NUM_WORKERS=4
MAX_PIXELS=8000000
DPI=200
NUM_THREAD=64
```

### For High-Volume Production (Multiple GPUs)
```env
NUM_WORKERS=8
MAX_PIXELS=11289600
DPI=200
NUM_THREAD=128
```

## Verification Checklist

- [x] Identified root cause (image too large, no resizing)
- [x] Updated MAX_PIXELS configuration
- [x] Reduced NUM_WORKERS temporarily
- [x] Improved error handling
- [x] Created diagnostic tools
- [x] Documented the issue and resolution
- [ ] Tested with problematic image (user to verify)
- [ ] Monitored for 24 hours (user to verify)
- [ ] Gradually increased workers if stable (future)

## Related Files

- `backend/.env` - Configuration file
- `backend/ocr_service/model/inference.py` - Inference function
- `backend/ocr_service/utils/image_utils.py` - Image resizing logic
- `backend/check_inference_server.py` - Diagnostic script
- `backend/TROUBLESHOOTING.md` - Troubleshooting guide

## Contact

If issues persist after these changes:
1. Run `python check_inference_server.py` and share the output
2. Check vLLM server logs for specific error messages
3. Share the image dimensions and file size that's causing issues
4. Monitor system resources (GPU/CPU/RAM) during conversion

