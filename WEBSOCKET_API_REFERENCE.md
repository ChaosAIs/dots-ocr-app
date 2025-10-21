# WebSocket Conversion API Reference

## Endpoints

### 1. POST /convert (Non-blocking)

**Purpose**: Trigger document conversion without waiting for completion

**Request**:
```
POST /convert
Content-Type: multipart/form-data

Parameters:
- filename (string, required): Name of file in input folder
- prompt_mode (string, optional): OCR prompt mode (default: "prompt_layout_all_en")
```

**Response** (Immediate):
```json
{
  "status": "accepted",
  "conversion_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "message": "Conversion task started. Use WebSocket to track progress."
}
```

**Status Codes**:
- `200 OK` - Conversion task accepted
- `400 Bad Request` - Invalid filename
- `404 Not Found` - File not found
- `500 Internal Server Error` - Server error

---

### 2. GET /conversion-status/{conversion_id}

**Purpose**: Poll conversion status (alternative to WebSocket)

**Request**:
```
GET /conversion-status/550e8400-e29b-41d4-a716-446655440000
```

**Response**:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "status": "processing",
  "progress": 45,
  "message": "Converting document...",
  "created_at": "2024-01-15T10:30:00",
  "started_at": "2024-01-15T10:30:05",
  "completed_at": null,
  "error": null
}
```

**Status Values**:
- `pending` - Queued, not started
- `processing` - Currently converting
- `completed` - Successfully converted
- `error` - Conversion failed

---

### 3. WebSocket /ws/conversion/{conversion_id}

**Purpose**: Real-time progress updates via WebSocket

**Connection**:
```javascript
// Automatic protocol selection
const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
const ws = new WebSocket(
  `${protocol}//${window.location.host}/ws/conversion/550e8400-e29b-41d4-a716-446655440000`
);
```

**Messages Received**:

**Initial Status** (on connect):
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "status": "pending",
  "progress": 0,
  "message": "Conversion queued",
  "created_at": "2024-01-15T10:30:00",
  "started_at": null,
  "completed_at": null,
  "error": null
}
```

**Progress Update**:
```json
{
  "status": "processing",
  "progress": 50,
  "message": "Processing page 5 of 10..."
}
```

**Completion**:
```json
{
  "status": "completed",
  "progress": 100,
  "message": "Conversion completed successfully",
  "completed_at": "2024-01-15T10:35:00",
  "results": {
    "pages_processed": 10,
    "output_files": ["document_nohf.md", "document.md"]
  }
}
```

**Error**:
```json
{
  "status": "error",
  "progress": 0,
  "message": "Conversion failed: Invalid PDF format",
  "error": "Invalid PDF format",
  "completed_at": "2024-01-15T10:30:30"
}
```

---

## Frontend Usage

### Using DocumentService

```javascript
import documentService from "@/services/documentService";

// 1. Start conversion
const response = await documentService.convertDocument("document.pdf");
const conversionId = response.conversion_id;

// 2. Connect to WebSocket
const ws = documentService.connectToConversionProgress(
  conversionId,
  (progressData) => {
    // Handle progress update
    console.log(`Progress: ${progressData.progress}%`);
    
    if (progressData.status === "completed") {
      console.log("Conversion complete!");
      // Refresh document list
    }
    
    if (progressData.status === "error") {
      console.error("Conversion failed:", progressData.error);
    }
  },
  (error) => {
    console.error("WebSocket error:", error);
  }
);

// 3. Cleanup (automatic in component unmount)
ws.close();
```

### In React Component

```javascript
const [conversionProgress, setConversionProgress] = useState({});

const handleConvert = async (filename) => {
  const response = await documentService.convertDocument(filename);
  const conversionId = response.conversion_id;
  
  const ws = documentService.connectToConversionProgress(
    conversionId,
    (data) => {
      setConversionProgress(prev => ({
        ...prev,
        [conversionId]: data.progress
      }));
      
      if (data.status === "completed") {
        // Refresh list
      }
    }
  );
};
```

---

## Status Transitions

```
pending → processing → completed
                    ↘ error
```

- **pending**: Task created, waiting to start
- **processing**: Actively converting document
- **completed**: Successfully converted, markdown files created
- **error**: Conversion failed, check error message

---

## Progress Values

- `0%` - Not started or error
- `10%` - Conversion started
- `50%` - Mid-conversion
- `100%` - Completed

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| File not found | Filename doesn't exist | Check filename spelling |
| Invalid filename | Contains "..", "/", "\\" | Use valid filename |
| Invalid PDF format | Corrupted or unsupported file | Try different file |
| WebSocket connection failed | Network issue | Check connection, retry |

### Retry Logic

```javascript
async function convertWithRetry(filename, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await documentService.convertDocument(filename);
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await new Promise(r => setTimeout(r, 1000 * (i + 1)));
    }
  }
}
```

---

## Performance Notes

- Conversion runs in background thread (non-blocking)
- Multiple conversions can run concurrently
- WebSocket updates sent every status change
- Progress bar updates in real-time
- No timeout issues with long conversions

---

## Troubleshooting

**WebSocket not connecting?**
- Check browser console for errors
- Verify WebSocket protocol (ws:// vs wss://)
- Check CORS configuration
- Ensure conversion_id is valid

**Progress not updating?**
- Check WebSocket connection status
- Verify backend is running
- Check browser network tab
- Try polling endpoint instead

**Conversion stuck?**
- Check backend logs
- Verify file is valid
- Try different prompt mode
- Restart backend service

