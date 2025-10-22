# API Changes for Multi-Page PDF Support

## Updated Endpoints

### 1. GET /documents
**Changes:** Added `is_multipage` field to response

**Response Example:**
```json
{
  "status": "success",
  "documents": [
    {
      "filename": "graph_r1.pdf",
      "file_path": "/path/to/graph_r1.pdf",
      "file_size": 4855751,
      "upload_time": "2025-10-13T16:45:58.809568",
      "markdown_exists": true,
      "markdown_path": "/path/to/output/graph_r1/graph_r1_page_0_nohf.md",
      "is_multipage": true
    },
    {
      "filename": "test4.png",
      "file_path": "/path/to/test4.png",
      "file_size": 274508,
      "upload_time": "2025-10-16T23:18:04.866389",
      "markdown_exists": true,
      "markdown_path": "/path/to/output/test4/test4_nohf.md",
      "is_multipage": false
    }
  ],
  "total": 2
}
```

### 2. GET /markdown/{filename}
**Changes:** Added optional `page_no` query parameter

**Endpoints:**
- `GET /markdown/graph_r1` - Returns first page (backward compatible)
- `GET /markdown/graph_r1?page_no=5` - Returns page 5

**Response:**
```json
{
  "status": "success",
  "filename": "graph_r1",
  "page_no": 5,
  "content": "# Page 5 Content\n..."
}
```

### 3. GET /markdown-files/{filename} (NEW)
**Purpose:** List all markdown files for a document

**Response:**
```json
{
  "status": "success",
  "filename": "graph_r1",
  "markdown_files": [
    {
      "filename": "graph_r1_page_0_nohf.md",
      "path": "/path/to/output/graph_r1/graph_r1_page_0_nohf.md",
      "page_no": 0,
      "is_multipage": true
    },
    {
      "filename": "graph_r1_page_1_nohf.md",
      "path": "/path/to/output/graph_r1/graph_r1_page_1_nohf.md",
      "page_no": 1,
      "is_multipage": true
    },
    ...
    {
      "filename": "graph_r1_page_19_nohf.md",
      "path": "/path/to/output/graph_r1/graph_r1_page_19_nohf.md",
      "page_no": 19,
      "is_multipage": true
    }
  ],
  "total": 20
}
```

## Frontend Service Changes

### DocumentService Methods

#### New Method: getMarkdownFiles()
```javascript
async getMarkdownFiles(filename) {
  const response = await http.get(
    `${this.apiDomain}/markdown-files/${filename}`
  );
  return response.data;
}
```

#### Updated Method: getMarkdownContent()
```javascript
async getMarkdownContent(filename, pageNo = null) {
  let url = `${this.apiDomain}/markdown/${filename}`;
  if (pageNo !== null) {
    url += `?page_no=${pageNo}`;
  }
  const response = await http.get(url);
  return response.data;
}
```

## Usage Examples

### JavaScript/Frontend
```javascript
// Get list of markdown files
const files = await documentService.getMarkdownFiles('graph_r1');
console.log(`Total pages: ${files.total}`);

// Load specific page
const page5 = await documentService.getMarkdownContent('graph_r1', 5);
console.log(page5.content);

// Load first page (backward compatible)
const firstPage = await documentService.getMarkdownContent('graph_r1');
```

### cURL
```bash
# Get documents list
curl http://localhost:8080/documents

# Get markdown files list
curl http://localhost:8080/markdown-files/graph_r1

# Get specific page
curl "http://localhost:8080/markdown/graph_r1?page_no=5"

# Get first page (backward compatible)
curl http://localhost:8080/markdown/graph_r1
```

## Error Handling

### 404 Not Found
- Document not found
- Markdown files not found
- Specific page not found

### 400 Bad Request
- Invalid filename (contains "..", "/", or "\\")
- Missing filename parameter

### 500 Server Error
- File system errors
- Unexpected exceptions

## Backward Compatibility

✅ All existing API calls continue to work
✅ `page_no` parameter is optional
✅ Single-page documents work as before
✅ No breaking changes to response structure

