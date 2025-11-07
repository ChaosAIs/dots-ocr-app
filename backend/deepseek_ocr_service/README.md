# DeepSeek OCR Service

This module provides functionality to convert images to markdown using the DeepSeek OCR API.

## Overview

The DeepSeek OCR service is a high-quality OCR solution that uses the DeepSeek-OCR model served via vLLM. It provides superior OCR and markdown generation capabilities specifically for image files.

## Features

- **Image-only conversion**: Supports PNG, JPG, JPEG, GIF, BMP, TIFF, and WEBP formats
- **High-quality OCR**: Uses DeepSeek-OCR model for accurate text extraction
- **Markdown generation**: Converts images directly to well-formatted markdown
- **Non-blocking API**: Integrates with the worker pool for concurrent processing
- **Real-time progress**: WebSocket support for live conversion progress updates

## Architecture

### Components

1. **DeepSeekOCRConverter** (`deepseek_ocr_converter.py`)

   - Main converter class that interfaces with the DeepSeek OCR API
   - Uses OpenAI-compatible API client to communicate with vLLM server
   - Handles image encoding, API calls, and markdown file generation

2. **Backend Endpoint** (`/convert-deepseek` in `main.py`)

   - FastAPI endpoint for initiating DeepSeek OCR conversions
   - Validates image file types
   - Submits conversion tasks to the worker pool
   - Returns conversion ID for progress tracking

3. **Frontend Integration**
   - `documentService.js`: Added `convertDocumentWithDeepSeekOCR()` method
   - `documentList.jsx`: Dropdown selector for choosing converter type
   - Translation support for English and French

## Reference Implementation

This implementation is based on the DeepSeek-OCR reference code:

- **Source**: `DeekSeek-OCR--Dockerized-API/` folder
- **Key Files**:
  - `start_server.py`: vLLM FastAPI server implementation
  - `custom_run_dpsk_ocr_image.py`: Image processing example
  - `custom_config.py`: Configuration and prompt templates
  - `custom_deepseek_ocr.py`: Model implementation

## Prompt Format

### Important Note on API Differences

This implementation uses the **OpenAI-compatible API** (`/v1/chat/completions`) provided by vLLM, which differs from the direct vLLM API used in the reference implementation.

**Key Difference**:

- **OpenAI-compatible API**: Do NOT include `<image>` token in prompts. The image is sent separately via `image_url` field.
- **Direct vLLM API** (reference): Requires `<image>` token in prompts for image placement.

### Default Prompt (OpenAI-compatible API)

```
<|grounding|>Convert the document to markdown.
```

### Common Prompt Variations (OpenAI-compatible API)

1. **Document Conversion** (Default):

   ```
   <|grounding|>Convert the document to markdown.
   ```

2. **General OCR**:

   ```
   <|grounding|>OCR this image.
   ```

3. **Free OCR** (No Layout):

   ```
   Free OCR.
   ```

4. **Figure Parsing**:

   ```
   Parse the figure.
   ```

5. **General Description**:

   ```
   Describe this image in detail.
   ```

6. **Element Localization**:

   ```
   Locate <|ref|>specific text<|/ref|> in the image.
   ```

**Note**: If you accidentally include `<image>` token in your prompt, it will be automatically removed by the converter.

## Configuration

The DeepSeek OCR service is configured via environment variables in `backend/.env`:

```env
# DeepSeek OCR Service Configuration
DEEPSEEK_OCR_IP=localhost
DEEPSEEK_OCR_PORT=8005
DEEPSEEK_OCR_MODEL_NAME=deepseek-ai/DeepSeek-OCR
DEEPSEEK_OCR_TEMPERATURE=0.0
DEEPSEEK_OCR_MAX_TOKENS=8192
DEEPSEEK_OCR_NGRAM_SIZE=30
DEEPSEEK_OCR_WINDOW_SIZE=90
DEEPSEEK_OCR_WHITELIST_TOKEN_IDS=128821,128822
```

### Configuration Parameters

- **DEEPSEEK_OCR_IP**: IP address of the vLLM server (default: localhost)
- **DEEPSEEK_OCR_PORT**: Port of the vLLM server (default: 8005)
- **DEEPSEEK_OCR_MODEL_NAME**: Model name (default: deepseek-ai/DeepSeek-OCR)
- **DEEPSEEK_OCR_TEMPERATURE**: Sampling temperature (0.0 for deterministic)
- **DEEPSEEK_OCR_MAX_TOKENS**: Maximum tokens to generate (default: 8192)
- **DEEPSEEK_OCR_NGRAM_SIZE**: N-gram size for logits processor (default: 30)
- **DEEPSEEK_OCR_WINDOW_SIZE**: Window size for logits processor (default: 90)
- **DEEPSEEK_OCR_WHITELIST_TOKEN_IDS**: Comma-separated token IDs to whitelist (default: 128821,128822 for <td>, </td>)

## Usage

### Backend API

**Endpoint**: `POST /convert-deepseek`

**Parameters**:

- `filename` (form data): Name of the image file to convert

**Response**:

```json
{
  "status": "accepted",
  "conversion_id": "uuid-string",
  "filename": "image.png",
  "message": "DeepSeek OCR conversion task started. Use WebSocket to track progress.",
  "converter_type": "deepseek_ocr",
  "queue_size": 0,
  "active_tasks": 1
}
```

**WebSocket Progress Tracking**:
Connect to `ws://localhost:8080/ws/conversion/{conversion_id}` to receive real-time progress updates.

### Frontend Usage

Users can select the converter type from a dropdown in the document list:

1. **Auto**: Automatically selects the best converter based on file type
2. **DeepSeek OCR**: Uses DeepSeek OCR for image files (images only)
3. **OCR Service**: Uses the existing OCR parser (for PDFs and images)
4. **Doc Service**: Uses document converters (for Word, Excel, TXT files)

The dropdown only shows relevant options based on the file type.

### Programmatic Usage

```python
from deepseek_ocr_service.deepseek_ocr_converter import DeepSeekOCRConverter
from pathlib import Path

# Initialize converter
converter = DeepSeekOCRConverter(
    api_base="http://localhost:8005/v1",
    model_name="deepseek-ai/DeepSeek-OCR",
    temperature=0.0,
    max_tokens=8192
)

# Convert an image with default prompt
input_path = Path("input/image.png")
output_path = Path("output/image.md")

success = converter.convert_file(input_path, output_path)
if success:
    print(f"Conversion successful: {output_path}")

# Convert with custom prompt (no <image> token needed)
success = converter.convert_file(
    input_path,
    output_path,
    prompt="Free OCR."
)

# Or get markdown content directly
markdown = converter.convert_image_to_markdown(
    input_path,
    prompt="<|grounding|>OCR this image."
)
print(markdown)
```

## Testing

A test script is provided to verify the DeepSeek OCR converter:

```bash
cd backend
python test_deepseek_ocr.py
```

This script will:

1. Find an image file in the `input` directory
2. Convert it to markdown using DeepSeek OCR
3. Save the output to the `output` directory
4. Display the first 10 lines of the generated markdown

## Supported File Types

The DeepSeek OCR converter supports the following image formats:

- PNG (.png)
- JPEG (.jpg, .jpeg)
- GIF (.gif)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## Output Format

The converter generates markdown files in the same format as other converters:

- Output directory: `output/{filename_without_ext}/`
- Output file: `{filename_without_ext}_nohf.md`

This ensures compatibility with the existing frontend markdown viewer.

## Integration with Worker Pool

The DeepSeek OCR converter integrates seamlessly with the existing worker pool:

- Supports concurrent conversions (configurable via `NUM_WORKERS`)
- Non-blocking API calls
- Real-time progress updates via WebSocket
- Automatic error handling and recovery

## Comparison with Other Converters

| Feature          | DeepSeek OCR           | OCR Service           | Doc Service         |
| ---------------- | ---------------------- | --------------------- | ------------------- |
| File Types       | Images only            | PDF, Images           | Word, Excel, TXT    |
| OCR Quality      | Highest                | High                  | N/A                 |
| Speed            | Fast                   | Medium                | Fast                |
| Layout Detection | Yes                    | Yes                   | N/A                 |
| Table Support    | Excellent              | Good                  | Excellent           |
| Use Case         | High-quality image OCR | General PDF/image OCR | Document conversion |

## Troubleshooting

### Connection Errors

If you see connection errors, ensure the vLLM server is running:

```bash
docker ps | grep deepseek-ocr
```

The DeepSeek OCR container should be running on port 8005.

### Model Loading Issues

Check the vLLM server logs:

```bash
docker logs deepseek-ocr-deepseek-ocr-1
```

### API Timeout

If conversions timeout, increase the timeout in the converter initialization:

```python
converter = DeepSeekOCRConverter(
    api_base="http://localhost:8005/v1",
    timeout=7200  # 2 hours
)
```

## Future Enhancements

Potential improvements for the DeepSeek OCR service:

- Batch processing support for multiple images
- Custom prompt templates for different OCR tasks
- Image preprocessing options (rotation, cropping, etc.)
- Support for multi-page TIFF files
- OCR confidence scores and quality metrics
