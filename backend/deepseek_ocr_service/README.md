# DeepSeek OCR Service

This module provides functionality to convert images to markdown using the DeepSeek OCR API.

## Overview

The DeepSeek OCR service is a high-quality OCR solution that uses the DeepSeek-OCR model served via vLLM. It provides superior OCR and markdown generation capabilities specifically for image files.

## Features

- **Image-only conversion**: Supports PNG, JPG, JPEG, GIF, BMP, TIFF, and WEBP formats
- **High-quality OCR**: Uses DeepSeek-OCR model for accurate text extraction
- **Intelligent image analysis**: Goes beyond text extraction to understand context and meaning
- **Smart image type detection**: Automatically detects 9+ image types including receipts, forms, charts, diagrams, infographics, and more
- **Context-aware prompts**: Uses specialized prompts that encourage explanation and insight, not just text extraction
- **Multi-dimensional analysis**: Extracts text, explains purpose, identifies patterns, and provides insights
- **Business document intelligence**: Special handling for receipts, invoices, and forms with transaction/field analysis
- **Data visualization understanding**: Analyzes charts and graphs to explain trends and key insights
- **Technical diagram comprehension**: Understands architecture diagrams, flowcharts, and system designs
- **Automatic table fixing**: Post-processes markdown to fix incomplete table formatting
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

### Important Note on API Differences and Prompt Requirements

This implementation uses the **OpenAI-compatible API** (`/v1/chat/completions`) provided by vLLM, which differs from the direct vLLM API used in the reference implementation.

**Critical Findings**:

1. **Do NOT use `<image>` token**: The image is sent separately via `image_url` field in the OpenAI-compatible API
2. **Do NOT use `<|grounding|>` prefix**: This special token causes the model to return empty responses or only grounding coordinates
3. **Keep prompts SHORT and simple**: Long, complex prompts (>100 chars) often cause empty responses or repetitive generic output
4. **Use plain English prompts**: Simple, direct instructions work best
5. **Avoid multi-step instructions**: Complex numbered lists cause the model to output repetitive text instead of extracting data
6. **Focus on extraction, not analysis**: Ask to "extract" and "convert", not "analyze" or "provide insights"

**IMPORTANT LESSON LEARNED**: Initial attempts to use detailed multi-step prompts (e.g., "1. Analyze... 2. Extract... 3. Provide insights...") caused the model to output repetitive generic text like "Use bullet points to summarize key findings and insights" repeated 15 times. **Simple, direct prompts work dramatically better**. See `PROMPT_LESSONS_LEARNED.md` for details.

### Intelligent Auto-Detection Feature (ENHANCED)

The converter now automatically detects the type of image based on filename keywords and applies **intelligent, context-aware prompts** that encourage understanding and explanation, not just text extraction.

#### Supported Image Types and Intelligence Features

1. **Receipts & Invoices** (`receipt`, `invoice`, `bill`, `payment`)

   - Summarizes the transaction (merchant, date, purpose)
   - Extracts all key information (items, prices, totals, payment method)
   - Identifies anomalies or notable observations
   - Provides structured markdown output

2. **Forms** (`form`, `application`, `survey`, `questionnaire`)

   - Identifies the form's purpose
   - Extracts all fields with labels and values
   - Distinguishes required vs optional fields
   - Notes instructions and important information

3. **Flowcharts** (`flow`, `flowchart`, `workflow`, `process`)

   - Explains what process the flowchart represents
   - Extracts all text from shapes and connectors
   - Describes flow logic, decision points, and branches
   - Identifies start/end points and explains decision criteria
   - Converts to ASCII flow diagram when possible

4. **Technical Diagrams** (`diagram`, `architecture`, `schema`, `blueprint`, `uml`)

   - Identifies what the diagram represents (architecture, system design, etc.)
   - Explains main components and their relationships
   - Extracts all labels and annotations
   - Provides insights about the overall design

5. **Charts & Graphs** (`chart`, `graph`, `plot`, `bar`, `pie`, `line`)

   - Identifies chart type
   - Explains the main message or insight from the data
   - Extracts all labels, axes, legend, and data values
   - Describes trends, patterns, and notable data points
   - Draws conclusions from the visualization

6. **Infographics** (`infographic`, `visual`, `poster`, `presentation`, `report`, `summary`, `dashboard`)

   - Identifies the main topic and message
   - Extracts key sections and their purposes
   - Captures all statistics, facts, and data points
   - Explains the narrative flow and information organization
   - Highlights key insights and takeaways

7. **Tables** (`table`, `grid`, `spreadsheet`)

   - Explains the table's purpose
   - Identifies column headers and their meanings
   - Notes totals, subtotals, and calculated fields
   - Highlights patterns or outliers
   - Provides summary of key insights

8. **Screenshots** (`screenshot`, `screen`, `capture`, `snap`)

   - Identifies the application or interface
   - Explains what the user is doing or what state is shown
   - Extracts all visible text and UI elements
   - Describes layout and main components
   - Notes error messages or important status indicators

9. **Documents** (default for all other files)
   - Identifies document type (letter, article, manual, etc.)
   - Preserves structure (headings, lists, emphasis, tables)
   - Notes special elements (headers, footers, watermarks)
   - Describes embedded images or diagrams
   - Provides complete markdown conversion

This feature is **enabled by default** and works automatically when you upload files with descriptive names.

### Example: Enhanced Intelligence in Action

**Old approach** (simple text extraction):

```
"Convert the document to markdown."
→ Returns: Just the text from the image
```

**New approach** (intelligent analysis):

```
For a receipt: "Analyze this receipt and provide:
1. Summary of the transaction
2. Extract all key information
3. Notable observations
4. Structured markdown format"
→ Returns: Context + Data + Insights + Markdown
```

### Manual Prompt Variations

You can also provide custom prompts to override auto-detection. **Important**: Keep prompts short and simple for best results.

1. **General OCR**:

   ```
   Extract all text from this image.
   ```

2. **Free OCR** (No Layout):

   ```
   Free OCR.
   ```

3. **Figure Parsing**:

   ```
   Parse the figure.
   ```

4. **General Description**:

   ```
   Describe this image in detail.
   ```

5. **Flowchart Analysis**:
   ```
   Extract all text from this flowchart and describe the flow.
   ```

**Note**: If you accidentally include `<image>` token in your prompt, it will be automatically removed by the converter.

## Post-Processing Features

### Automatic Markdown Table Fixing

The DeepSeek OCR converter includes automatic post-processing to fix incomplete markdown tables. This feature addresses a common issue where the OCR model generates table headers without the required separator line.

**Problem**: DeepSeek OCR sometimes generates tables like this:

```markdown
| Column 1 | Column 2 | Column 3 |
| Data 1 | Data 2 | Data 3 |
```

This is invalid markdown because it's missing the separator line after the header.

**Solution**: The converter automatically detects and fixes these tables:

```markdown
| Column 1 | Column 2 | Column 3 |
| -------- | -------- | -------- |
| Data 1   | Data 2   | Data 3   |
```

**How it works**:

1. Scans the markdown output for table rows (lines starting with `|`)
2. Detects table headers (first row after a heading or empty line)
3. Checks if the separator line is missing
4. Automatically inserts the separator line with the correct number of columns

**Benefits**:

- Tables render correctly in markdown viewers
- No manual editing required
- Preserves all original content
- Works automatically for all conversions

This feature is enabled by default and requires no configuration.

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

# Convert a flowchart with auto-detection (recommended)
flowchart_path = Path("input/process_flowchart.png")
output_path = Path("output/flowchart.md")

success = converter.convert_file(flowchart_path, output_path)
if success:
    print(f"Conversion successful: {output_path}")
# Auto-detects 'flowchart' from filename and uses optimized prompt

# Convert a document with auto-detection disabled
document_path = Path("input/document.png")
success = converter.convert_file(
    document_path,
    output_path,
    auto_detect_type=False  # Use default document prompt
)

# Convert with custom prompt (no <image> token needed)
success = converter.convert_file(
    document_path,
    output_path,
    prompt="Free OCR."  # Custom prompt overrides auto-detection
)

# Or get markdown content directly
markdown = converter.convert_image_to_markdown(
    flowchart_path,
    auto_detect_type=True  # Enable auto-detection (default)
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
