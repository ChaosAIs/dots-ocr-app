# OCR Performance Optimization Guide

## Overview
This document explains the performance optimization settings for the Dots OCR application, focusing on balancing processing speed with OCR quality.

## Key Performance Settings

### 1. PDF Rendering DPI (`.env` line 25)

**Current Setting:** `DPI=150`

**Impact on Performance:**
- **150 DPI** (Recommended): 
  - Image size: ~1281x1659 pixels (~2.1M pixels per page)
  - Processing time: ~10-12 seconds per page
  - Quality: Excellent for most documents, industry standard for OCR
  
- **200 DPI** (Previous setting):
  - Image size: ~1708x2212 pixels (~3.78M pixels per page)
  - Processing time: ~13-15 seconds per page
  - Quality: Very high, but diminishing returns for OCR

- **100 DPI** (Fast mode):
  - Image size: ~854x1106 pixels (~0.94M pixels per page)
  - Processing time: ~7-9 seconds per page
  - Quality: Good for simple documents, may lose detail in small text

**Recommendation:** Keep at 150 DPI for optimal balance.

### 2. Parallel Processing Threads (`.env` line 23)

**Current Setting:** `NUM_THREAD=4`

**Impact on Performance:**
- **4 threads**: Process 4 pages simultaneously (if GPU memory allows)
- **2 threads**: Process 2 pages simultaneously (safer for GPU memory)

**GPU Memory Considerations:**
- vLLM shows GPU KV cache usage at ~23-40% during processing
- With 4 threads, expect higher GPU memory usage but faster overall processing
- If you see GPU OOM errors, reduce back to 2

**Recommendation:** Try 4 threads; reduce to 2 if GPU memory issues occur.

### 3. Image Resizing for Qwen3 Analysis (`.env` line 111)

**Current Setting:** `QWEN_TRANSFORMERS_MAX_IMAGE_AREA=250000`

**Impact:**
- Layout detection crops are automatically resized to max 250K pixels
- Prevents CUDA timeouts on display-attached GPUs
- Reduces memory usage for Qwen3 transformers inference

**Recommendation:** Keep at 250000 pixels (approximately 500x500).

## Expected Performance Improvements

### Before Optimization (DPI=200, NUM_THREAD=2):
- **62-page PDF**: ~13-15 minutes total
- **Per page**: ~13-15 seconds
- **Image size**: 1708x2212 pixels (3.78M pixels)

### After Optimization (DPI=150, NUM_THREAD=4):
- **62-page PDF**: ~6-8 minutes total (50% faster)
- **Per page**: ~6-8 seconds (with 4 parallel threads)
- **Image size**: 1281x1659 pixels (2.1M pixels)

## How to Apply Changes

### For New Documents:
1. **Restart the backend server** to pick up the new `.env` settings:
   ```bash
   # Stop the current server (Ctrl+C)
   # Restart with:
   cd backend
   python main.py
   ```

2. Upload and convert new documents - they will use the new settings automatically.

### For Currently Processing Documents:
- The current document (GSC MA System Blueprint v6.0.pdf) will continue with the old settings
- Let it finish, then restart the server for new conversions

## Monitoring Performance

### Check vLLM Status:
```bash
docker logs --tail 50 vllm_dots_ocr-vllm_llm_dots_ocr-1
```

### Check Backend Processing:
- Watch the terminal output for timing information
- Look for: `Processing PDF pages: XX%|████| X/62 [MM:SS<MM:SS, XX.XXs/it]`

### GPU Memory Usage:
- vLLM logs show: `GPU KV cache usage: XX.X%`
- If consistently > 80%, reduce NUM_THREAD to 2

## Quality Assurance

**150 DPI is sufficient for:**
- ✅ Regular text documents
- ✅ Business reports
- ✅ Forms and invoices
- ✅ Technical diagrams
- ✅ Tables and charts

**Consider 200 DPI for:**
- 📄 Very small fonts (< 8pt)
- 📄 High-detail engineering drawings
- 📄 Scanned documents with poor quality

## Troubleshooting

### If processing is still slow:
1. Check vLLM logs for retries or errors
2. Verify GPU is not being used by other processes
3. Consider reducing DPI to 100 for testing

### If OCR quality degrades:
1. Increase DPI back to 200
2. Check sample outputs in `backend/output/`
3. Compare markdown quality before/after changes

## Configuration File Location

All settings are in: `backend/.env`

Key lines:
- Line 23: `NUM_THREAD=4`
- Line 25: `DPI=150`
- Line 111: `QWEN_TRANSFORMERS_MAX_IMAGE_AREA=250000`

