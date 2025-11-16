# Image Size-Based Analysis Selection

## Overview

The Qwen3OCRConverter now automatically detects image size and applies appropriate analysis depth:

- **Simple Analysis**: For small images (icons, labels, single words)
- **Complex Analysis**: For large images (charts, tables, diagrams, documents)

## How It Works

### 1. Image Size Detection

When `convert_image_base64_to_markdown()` is called without a custom prompt, the system:

1. Decodes the base64 image
2. Extracts image dimensions (width × height)
3. Calculates total pixel area
4. Classifies as SIMPLE or COMPLEX based on thresholds

### 2. Classification Thresholds

**SIMPLE Images** (any of these conditions):

- Width < 200 pixels, OR
- Height < 200 pixels, OR
- Total area < 50,000 pixels

**COMPLEX Images** (all of these conditions):

- Width ≥ 200 pixels, AND
- Height ≥ 200 pixels, AND
- Total area ≥ 50,000 pixels

### 3. Prompt Selection

Based on classification:

**Simple Prompt** (for small images):

- Direct extraction without sections
- Concise output
- Example: Image shows "Alt" → Output: "Alt"

**Complex Prompt** (for large images):

- Structured output with sections:
  - Document Analysis
  - Content
  - Key Information
  - Insights
- Detailed analysis with markdown formatting

## Examples

### Simple Images

| Dimensions | Area   | Classification        | Output Style        |
| ---------- | ------ | --------------------- | ------------------- |
| 100×50     | 5,000  | SIMPLE                | Direct text: "Alt"  |
| 150×100    | 15,000 | SIMPLE                | Direct text: "Logo" |
| 180×180    | 32,400 | SIMPLE                | Direct text: "Icon" |
| 300×200    | 60,000 | SIMPLE (by dimension) | Brief description   |

### Complex Images

| Dimensions | Area    | Classification | Output Style             |
| ---------- | ------- | -------------- | ------------------------ |
| 500×500    | 250,000 | COMPLEX        | Full structured analysis |
| 800×600    | 480,000 | COMPLEX        | Full structured analysis |
| 1024×768   | 786,432 | COMPLEX        | Full structured analysis |

## Usage

### Automatic Selection (Recommended)

```python
from qwen_ocr_service.qwen3_ocr_converter import Qwen3OCRConverter

converter = Qwen3OCRConverter()

# Automatically selects simple or complex prompt based on image size
markdown = converter.convert_image_base64_to_markdown(image_base64)
```

### Custom Prompt (Override)

```python
# Use custom prompt regardless of image size
custom_prompt = "Extract only the title from this image."
markdown = converter.convert_image_base64_to_markdown(
    image_base64,
    prompt=custom_prompt
)
```

### Manual Classification Check

```python
# Check if an image would be classified as simple
is_simple = converter._is_simple_image(image_base64)

if is_simple:
    print("Will use simple analysis")
else:
    print("Will use complex analysis")
```

## Benefits

1. **Efficiency**: Avoids over-analyzing simple images
2. **Quality**: Provides detailed analysis only when needed
3. **Cleaner Output**: Simple images get concise responses
4. **Automatic**: No manual configuration required

## Testing

Run the test script to see the classification in action:

```bash
cd backend/qwen_ocr_service
python test_image_size_detection.py
```

This will create test images of various sizes and show how they're classified.

## Configuration

The thresholds are configurable via environment variables in `backend/.env`:

```bash
# Image Size Detection Thresholds for Qwen3 OCR
QWEN_IMAGE_MIN_DIMENSION_THRESHOLD=100
QWEN_IMAGE_PIXEL_AREA_THRESHOLD=1000
```

**Default values:**

- `QWEN_IMAGE_MIN_DIMENSION_THRESHOLD`: 100 pixels
- `QWEN_IMAGE_PIXEL_AREA_THRESHOLD`: 1000 pixels

**To adjust the thresholds:**

1. Edit `backend/.env`
2. Change the values of `QWEN_IMAGE_MIN_DIMENSION_THRESHOLD` and/or `QWEN_IMAGE_PIXEL_AREA_THRESHOLD`
3. Restart the backend server

## Logging

The system logs classification decisions with threshold information:

```
INFO: Image analysis: dimensions=100x50, area=5000, classification=SIMPLE (thresholds: dimension=100, area=1000)
INFO: Image analysis: dimensions=800x600, area=480000, classification=COMPLEX (thresholds: dimension=100, area=1000)
```

This helps debug and understand which analysis mode is being used and what thresholds are active.

## Backward Compatibility

- Existing code continues to work without changes
- Custom prompts override automatic selection
- `_build_default_prompt()` still works (uses complex prompt)
- All three backends (ollama, vllm, transformers) supported
