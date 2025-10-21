# Dots OCR API

FastAPI-based REST API for document OCR parsing with layout detection.

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `backend/.env` to customize settings:

```env
# VLLM Server Configuration
VLLM_IP=localhost
VLLM_PORT=8001
VLLM_MODEL_NAME=dots_ocr

# Inference Configuration
TEMPERATURE=0.1
TOP_P=1.0
MAX_COMPLETION_TOKENS=16384

# Processing Configuration
NUM_THREAD=64
DPI=200
OUTPUT_DIR=./backend/output

# FastAPI Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

## Running the API

### Start the Server

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### Interactive API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Root Endpoint
```
GET /
```
Returns API information and available endpoints.

### 2. Health Check
```
GET /health
```
Returns service health status.

### 3. Get Configuration
```
GET /config
```
Returns current parser configuration.

### 4. Parse Single File
```
POST /parse
```

**Parameters:**
- `file` (required): PDF or image file to parse
- `prompt_mode` (optional): Prompt mode for parsing
  - `prompt_layout_all_en` (default)
  - `prompt_layout_only_en`
  - `prompt_grounding_ocr`
  - Other available modes
- `bbox` (optional): Bounding box for grounding OCR as JSON string `[x1, y1, x2, y2]`

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/parse" \
  -F "file=@/path/to/document.pdf" \
  -F "prompt_mode=prompt_layout_all_en"
```

**Example with Python:**
```python
import requests

with open("document.pdf", "rb") as f:
    files = {"file": f}
    data = {"prompt_mode": "prompt_layout_all_en"}
    response = requests.post("http://localhost:8000/parse", files=files, data=data)
    print(response.json())
```

### 5. Parse Multiple Files (Batch)
```
POST /parse-batch
```

**Parameters:**
- `files` (required): List of PDF or image files to parse
- `prompt_mode` (optional): Prompt mode for parsing

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/parse-batch" \
  -F "files=@/path/to/document1.pdf" \
  -F "files=@/path/to/document2.pdf" \
  -F "prompt_mode=prompt_layout_all_en"
```

**Example with Python:**
```python
import requests

files = [
    ("files", open("document1.pdf", "rb")),
    ("files", open("document2.pdf", "rb")),
]
data = {"prompt_mode": "prompt_layout_all_en"}
response = requests.post("http://localhost:8000/parse-batch", files=files, data=data)
print(response.json())
```

## Response Format

### Successful Parse Response
```json
{
  "status": "success",
  "filename": "document.pdf",
  "prompt_mode": "prompt_layout_all_en",
  "results": [
    {
      "page_no": 0,
      "input_height": 1024,
      "input_width": 768,
      "layout_info_path": "/path/to/output/document/document.json",
      "layout_image_path": "/path/to/output/document/document.jpg",
      "md_content_path": "/path/to/output/document/document.md",
      "file_path": "/path/to/input/document.pdf"
    }
  ]
}
```

### Batch Response
```json
{
  "status": "completed",
  "total_files": 2,
  "results": [
    {
      "filename": "document1.pdf",
      "status": "success",
      "results": [...]
    },
    {
      "filename": "document2.pdf",
      "status": "success",
      "results": [...]
    }
  ]
}
```

## Output Files

Parsed results are saved to the directory specified in `OUTPUT_DIR` (.env):

```
output/
├── document_name/
│   ├── document_name.json          # Layout information
│   ├── document_name.jpg           # Image with layout overlay
│   ├── document_name.md            # Extracted markdown content
│   ├── document_name_nohf.md       # Markdown without page headers/footers
│   └── document_name_page_*.json   # Per-page layout (for PDFs)
└── document_name.jsonl             # Summary results
```

## Configuration Options

### VLLM Server
- `VLLM_IP`: IP address of vLLM server (default: localhost)
- `VLLM_PORT`: Port of vLLM server (default: 8001)
- `VLLM_MODEL_NAME`: Model name to use (default: dots_ocr)

### Inference
- `TEMPERATURE`: Model temperature (default: 0.1)
- `TOP_P`: Top-p sampling parameter (default: 1.0)
- `MAX_COMPLETION_TOKENS`: Maximum tokens in response (default: 16384)

### Processing
- `NUM_THREAD`: Number of threads for parallel processing (default: 64)
- `DPI`: DPI for PDF rendering (default: 200)
- `OUTPUT_DIR`: Directory for output files (default: ./backend/output)

### Image Processing
- `MIN_PIXELS`: Minimum image pixels (default: None)
- `MAX_PIXELS`: Maximum image pixels (default: None)

### API Server
- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8000)

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Successful request
- `400`: Bad request (invalid parameters)
- `500`: Server error

Error responses include a detail message:
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Performance Tips

1. **Adjust NUM_THREAD**: Increase for faster PDF processing, decrease if memory is limited
2. **Adjust DPI**: Lower DPI (e.g., 150) for faster processing, higher DPI (e.g., 300) for better quality
3. **Batch Processing**: Use `/parse-batch` for multiple files to optimize resource usage
4. **vLLM Server**: Ensure vLLM server is running and accessible at configured IP:PORT

## Troubleshooting

### Connection Error to vLLM Server
- Check that vLLM server is running at the configured IP and port
- Verify `VLLM_IP` and `VLLM_PORT` in `.env`

### Out of Memory
- Reduce `NUM_THREAD` in `.env`
- Reduce `DPI` for PDF rendering
- Process files individually instead of batch

### Slow Processing
- Increase `NUM_THREAD` if system has available resources
- Increase `DPI` for faster processing (lower quality)
- Ensure vLLM server has sufficient GPU memory

