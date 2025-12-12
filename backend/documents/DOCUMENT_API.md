# Document Upload and Conversion API

This document describes the new API endpoints for document upload and conversion functionality.

## Endpoints

### 1. Upload Document

**Endpoint:** `POST /upload`

Upload a document file to the server.

**Request:**

- Content-Type: `multipart/form-data`
- Body:
  - `file`: The file to upload (required)

**Response:**

```json
{
  "status": "success",
  "filename": "document.pdf",
  "file_path": "/path/to/input/document.pdf",
  "file_size": 1024000,
  "upload_time": "2024-01-15T10:30:00.000000"
}
```

**Error Response:**

```json
{
  "detail": "Error message"
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/upload \
  -F "file=@document.pdf"
```

---

### 2. List Documents

**Endpoint:** `GET /documents`

Get a list of all uploaded documents with their conversion status.

**Response:**

```json
{
  "status": "success",
  "documents": [
    {
      "filename": "document.pdf",
      "file_path": "/path/to/input/document.pdf",
      "file_size": 1024000,
      "upload_time": "2024-01-15T10:30:00.000000",
      "markdown_exists": true,
      "markdown_path": "/path/to/output/document/document_nohf.md"
    }
  ],
  "total": 1
}
```

**Example:**

```bash
curl http://localhost:8080/documents
```

---

### 3. Convert Document

**Endpoint:** `POST /convert`

Convert an uploaded document to markdown format.

**Request:**

- Content-Type: `application/x-www-form-urlencoded`
- Body:
  - `filename`: The name of the file to convert (required)
  - `prompt_mode`: The prompt mode to use (optional, default: "prompt_layout_all_en")

**Response:**

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
      "md_content_nohf_path": "/path/to/output/document/document_nohf.md",
      "file_path": "/path/to/input/document.pdf"
    }
  ]
}
```

**Error Response:**

```json
{
  "detail": "File not found: filename.pdf"
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/convert \
  -d "filename=document.pdf&prompt_mode=prompt_layout_all_en"
```

---

### 4. Get Markdown Content

**Endpoint:** `GET /markdown/{filename}`

Get the markdown content of a converted document.

**Parameters:**

- `filename`: The filename without extension (path parameter)

**Response:**

```json
{
  "status": "success",
  "filename": "document",
  "content": "# Document Title\n\nMarkdown content here..."
}
```

**Error Response:**

```json
{
  "detail": "Markdown file not found for: filename"
}
```

**Example:**

```bash
curl http://localhost:8080/markdown/document
```

---

## File Structure

### Input Directory

- Location: `backend/input/`
- Contains: Uploaded document files
- Naming: Original filename preserved

### Output Directory

- Location: `backend/output/`
- Structure:
  ```
  output/
  ├── document1/
  │   ├── document1.json          (Layout information)
  │   ├── document1.jpg           (Layout visualization)
  │   ├── document1.md            (Markdown with page headers)
  │   └── document1_nohf.md       (Markdown without page headers)
  ├── document2/
  │   └── ...
  └── document1.jsonl             (Results metadata)
  ```

## Supported File Types

- **PDF**: `.pdf`
- **Images**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`
- **Documents**: `.doc`, `.docx`
- **Spreadsheets**: `.xls`, `.xlsx`

## Constraints

- Maximum file size: 50MB
- File path validation: Prevents directory traversal attacks
- Filename validation: Prevents special characters in paths

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid input)
- `404`: Not found (file not found)
- `500`: Server error

## Configuration

The API uses the following environment variables:

- `API_HOST`: API host (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000)
- `DOTS_OCR_VLLM_HOST`: Dots OCR vLLM server host (default: localhost)
- `DOTS_OCR_VLLM_PORT`: Dots OCR vLLM server port (default: 8001)
- `DOTS_OCR_VLLM_MODEL`: Dots OCR model name (default: dots_ocr)
- `OUTPUT_DIR`: Output directory (default: ./output)

## Usage Example

### Complete Workflow

1. **Upload a document:**

```bash
curl -X POST http://localhost:8080/upload \
  -F "file=@myfile.pdf"
```

2. **List documents:**

```bash
curl http://localhost:8080/documents
```

3. **Convert document (if not already converted):**

```bash
curl -X POST http://localhost:8080/convert \
  -d "filename=myfile.pdf"
```

4. **Get markdown content:**

```bash
curl http://localhost:8080/markdown/myfile
```

## Notes

- The conversion process is time-consuming and may take several minutes for large documents
- The markdown file is saved with the `_nohf` suffix (no page headers) for cleaner output
- Multiple pages in PDFs are processed in parallel using thread pools
- All file operations are validated to prevent security issues
