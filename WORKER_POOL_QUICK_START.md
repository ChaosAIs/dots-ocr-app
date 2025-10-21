# Worker Pool Quick Start Guide

## What Changed?

The backend now supports **concurrent document conversions** using a worker pool system. Multiple PDFs can be converted simultaneously without blocking the API.

### Key Benefits
✅ API returns immediately (< 100ms)  
✅ Multiple conversions run in parallel  
✅ Real-time progress updates via WebSocket  
✅ Queue monitoring available  
✅ Fully backward compatible  

## Starting the Backend

```bash
cd backend
python main.py --port 8080
```

Or with custom worker count:
```bash
NUM_WORKERS=8 python main.py --port 8080
```

## Basic Usage

### 1. Start a Conversion

```bash
curl -X POST http://localhost:8080/convert \
  -F "filename=document.pdf" \
  -F "prompt_mode=prompt_layout_all_en"
```

**Response (immediate):**
```json
{
  "status": "accepted",
  "conversion_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "message": "Conversion task started. Use WebSocket to track progress.",
  "queue_size": 1,
  "active_tasks": 1
}
```

### 2. Monitor Progress via WebSocket

```javascript
const conversionId = "550e8400-e29b-41d4-a716-446655440000";
const ws = new WebSocket(`ws://localhost:8080/ws/conversion/${conversionId}`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${data.progress}% - ${data.message}`);
  
  if (data.status === "completed") {
    console.log("Conversion complete!");
  }
};

ws.onerror = (error) => {
  console.error("WebSocket error:", error);
};
```

### 3. Check Queue Status

```bash
curl http://localhost:8080/worker-pool-status
```

**Response:**
```json
{
  "status": "ok",
  "queue_size": 2,
  "active_tasks": 1,
  "num_workers": 4
}
```

## Advanced Usage

### Start Multiple Conversions

```bash
# Terminal 1: Start first conversion
curl -X POST http://localhost:8080/convert -F "filename=doc1.pdf"

# Terminal 2: Start second conversion (while first is running)
curl -X POST http://localhost:8080/convert -F "filename=doc2.pdf"

# Terminal 3: Start third conversion
curl -X POST http://localhost:8080/convert -F "filename=doc3.pdf"

# Terminal 4: Monitor queue
watch -n 1 'curl -s http://localhost:8080/worker-pool-status | jq'
```

### Run Automated Test

```bash
cd backend
python test_worker_pool.py
```

This will:
1. Start multiple conversions concurrently
2. Monitor each via WebSocket
3. Display real-time progress
4. Show final summary

## Configuration

### Environment Variables

```bash
# Number of worker threads (default: 4)
export NUM_WORKERS=4

# Backend host and port
export API_HOST=0.0.0.0
export API_PORT=8080

# Start backend
python main.py
```

### Recommended Settings

| Scenario | NUM_WORKERS | Notes |
|----------|-------------|-------|
| Development | 2 | Low resource usage |
| Standard | 4 | Good balance |
| High-throughput | 8 | More concurrent conversions |
| Very high-throughput | 16 | For powerful servers |

## API Reference

### POST /convert

Start a new conversion task.

**Parameters:**
- `filename` (required): Name of file in input folder
- `prompt_mode` (optional): OCR prompt mode (default: "prompt_layout_all_en")

**Response:**
```json
{
  "status": "accepted",
  "conversion_id": "uuid",
  "filename": "document.pdf",
  "message": "Conversion task started...",
  "queue_size": 0,
  "active_tasks": 1
}
```

### GET /conversion-status/{conversion_id}

Get status of a specific conversion.

**Response:**
```json
{
  "id": "uuid",
  "filename": "document.pdf",
  "status": "processing",
  "progress": 45,
  "message": "Processing page 9/20",
  "created_at": "2024-01-01T12:00:00",
  "started_at": "2024-01-01T12:00:01",
  "completed_at": null,
  "error": null
}
```

### GET /worker-pool-status

Get worker pool status.

**Response:**
```json
{
  "status": "ok",
  "queue_size": 2,
  "active_tasks": 1,
  "num_workers": 4
}
```

### WebSocket /ws/conversion/{conversion_id}

Real-time progress updates.

**Messages:**
```json
{
  "status": "processing",
  "progress": 45,
  "message": "Processing page 9/20",
  "step": "page_processing",
  "timestamp": "2024-01-01T12:00:30"
}
```

## Frontend Integration

### React Example

```javascript
import { useState, useEffect } from 'react';

function ConversionMonitor({ filename }) {
  const [conversionId, setConversionId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('idle');

  // Start conversion
  const handleConvert = async () => {
    const formData = new FormData();
    formData.append('filename', filename);
    
    const response = await fetch('http://localhost:8080/convert', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    setConversionId(data.conversion_id);
    setStatus('processing');
  };

  // Monitor via WebSocket
  useEffect(() => {
    if (!conversionId) return;

    const ws = new WebSocket(
      `ws://localhost:8080/ws/conversion/${conversionId}`
    );

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setProgress(data.progress);
      setStatus(data.status);
    };

    return () => ws.close();
  }, [conversionId]);

  return (
    <div>
      <button onClick={handleConvert}>Convert {filename}</button>
      <div>Status: {status}</div>
      <div>Progress: {progress}%</div>
      <progress value={progress} max={100} />
    </div>
  );
}
```

## Troubleshooting

### Q: Conversion not starting
**A:** Check if file exists in input folder and backend is running.
```bash
curl http://localhost:8080/health
```

### Q: WebSocket connection fails
**A:** Ensure WebSocket URL is correct and backend is accessible.
```bash
# Test WebSocket connection
wscat -c ws://localhost:8080/ws/conversion/test-id
```

### Q: High memory usage
**A:** Reduce NUM_WORKERS or check for memory leaks.
```bash
NUM_WORKERS=2 python main.py
```

### Q: Conversions running slowly
**A:** Check queue size and increase workers if needed.
```bash
curl http://localhost:8080/worker-pool-status | jq
```

## Performance Tips

1. **Adjust worker count** based on CPU cores and memory
2. **Monitor queue size** to detect bottlenecks
3. **Use WebSocket** for real-time progress (don't poll)
4. **Batch conversions** for better throughput
5. **Check logs** for errors and performance issues

## Next Steps

1. ✅ Start backend with worker pool
2. ✅ Test with single conversion
3. ✅ Test with multiple concurrent conversions
4. ✅ Monitor queue status
5. ✅ Integrate with frontend
6. ✅ Tune worker count for your environment

## Support

For issues or questions:
1. Check logs: `tail -f backend.log`
2. Monitor queue: `curl http://localhost:8080/worker-pool-status`
3. Review documentation: `backend/WORKER_POOL_IMPLEMENTATION.md`
4. Run tests: `python backend/test_worker_pool.py`

