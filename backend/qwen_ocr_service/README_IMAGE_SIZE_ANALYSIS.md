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

### 2. Classification Threshold

**SIMPLE Images**:

- Total pixel area < 40,000 pixels (approximately 200×200)

**COMPLEX Images**:

- Total pixel area ≥ 40,000 pixels

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

| Dimensions | Area   | Classification | Output Style        |
| ---------- | ------ | -------------- | ------------------- |
| 100×50     | 5,000  | SIMPLE         | Direct text: "Alt"  |
| 150×100    | 15,000 | SIMPLE         | Direct text: "Logo" |
| 180×180    | 32,400 | SIMPLE         | Direct text: "Icon" |
| 200×150    | 30,000 | SIMPLE         | Brief description   |

### Complex Images

| Dimensions | Area    | Classification | Output Style             |
| ---------- | ------- | -------------- | ------------------------ |
| 200×200    | 40,000  | COMPLEX        | Full structured analysis |
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

The threshold is configurable via environment variable in `backend/.env`:

```bash
# Image Size Detection Threshold for Qwen3 OCR
QWEN_IMAGE_PIXEL_AREA_THRESHOLD=40000
```

**Default value:**

- `QWEN_IMAGE_PIXEL_AREA_THRESHOLD`: 40,000 pixels (approximately 200×200)

**To adjust the threshold:**

1. Edit `backend/.env`
2. Change the value of `QWEN_IMAGE_PIXEL_AREA_THRESHOLD`
3. Restart the backend server

**Example values:**

- `10000` (100×100): Very aggressive - treats most images as complex
- `40000` (200×200): Default - balanced classification
- `100000` (316×316): Conservative - treats more images as simple

## Logging

The system logs classification decisions with area threshold information:

```
INFO: Image analysis: dimensions=100x50, area=5,000, classification=SIMPLE (area_threshold=40,000)
INFO: Image analysis: dimensions=800x600, area=480,000, classification=COMPLEX (area_threshold=40,000)
```

This helps debug and understand which analysis mode is being used and what threshold is active.

## Backward Compatibility

- Existing code continues to work without changes
- Custom prompts override automatic selection
- `_build_default_prompt()` still works (uses complex prompt)
- All three backends (ollama, vllm, transformers) supported
