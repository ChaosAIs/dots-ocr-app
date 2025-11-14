# Worker Pool Implementation for Concurrent Document Conversion

## Overview

This document describes the implementation of a multi-threaded worker pool system that enables the backend to handle multiple concurrent document conversions independently, without blocking the API or other conversions.

## Problem Statement

**Original Issue:**
- When a PDF conversion started, the entire backend process would block until completion
- Only one conversion could run at a time
- Large PDFs (20+ pages) would take 10+ minutes, blocking all other requests
- The `/convert` endpoint would not return until the entire conversion finished

**Example from logs:**
```
INFO:     127.0.0.1:52080 - "POST /convert HTTP/1.1" 200 OK
loading pdf: /home/fy/MyWorkPlace/dots-ocr-app/backend/input/graph_r1.pdf
Parsing PDF with 20 pages using 20 threads...
Processing PDF pages: 100%|██████████████████████████████████| 20/20 [11:46<00:00, 35.32s/it]
```

## Solution Architecture

### Components

#### 1. **WorkerPool** (`backend/worker_pool.py`)
- Thread pool-based task queue system
- Configurable number of worker threads (default: 4)
- Manages concurrent conversion tasks independently
- Features:
  - Task submission and queuing
  - Progress tracking per task
  - Graceful shutdown
  - Queue status monitoring

#### 2. **ConversionWorker** (`backend/worker_pool.py`)
- Individual worker thread that processes tasks from the queue
- Executes conversion functions independently
- Handles errors and reports completion status
- Supports daemon mode for clean shutdown

#### 3. **ProgressTracker** (`backend/progress_tracker.py`)
- Tracks progress of individual conversions
- Provides callbacks for progress updates
- Broadcasts updates via WebSocket
- Supports step-based progress reporting

#### 4. **Updated Main API** (`backend/main.py`)
- Integrated worker pool initialization
- Modified `/convert` endpoint to submit tasks to worker pool
- New `/worker-pool-status` endpoint for monitoring
- Improved error handling and logging

## How It Works

### Conversion Flow

```
1. User calls POST /convert with filename
   ↓
2. Backend validates file and creates conversion task
   ↓
3. Backend submits task to worker pool (returns immediately)
   ↓
4. API returns conversion_id + queue status (< 100ms)
   ↓
5. User connects to WebSocket /ws/conversion/{conversion_id}
   ↓
6. Worker thread picks up task from queue
   ↓
7. Worker executes conversion in background
   ↓
8. Worker broadcasts progress updates via WebSocket
   ↓
9. When complete, worker broadcasts completion message
   ↓
10. User receives completion notification
```

### Key Improvements

1. **Non-blocking API**: `/convert` returns immediately with conversion_id
2. **Concurrent Processing**: Multiple conversions run in parallel
3. **Real-time Progress**: WebSocket updates during conversion
4. **Queue Management**: Monitor queue size and active tasks
5. **Independent Execution**: Each conversion runs independently
6. **Error Handling**: Errors in one conversion don't affect others

## Configuration

### Environment Variables

```bash
# Number of worker threads (default: 4)
NUM_WORKERS=4

# Other existing variables
API_HOST=0.0.0.0
API_PORT=8080
```

### Adjusting Worker Count

```python
# In backend/main.py
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
```

**Recommendations:**
- **4 workers**: Good for most use cases, balances concurrency and resource usage
- **2 workers**: For low-resource environments
- **8+ workers**: For high-throughput scenarios with many concurrent conversions

## API Endpoints

### POST /convert (Updated)

**Request:**
```bash
curl -X POST http://localhost:8080/convert \
  -F "filename=document.pdf" \
  -F "prompt_mode=prompt_layout_all_en"
```

**Response (Immediate):**
```json
{
  "status": "accepted",
  "conversion_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "message": "Conversion task started. Use WebSocket to track progress.",
  "queue_size": 2,
  "active_tasks": 1
}
```

### GET /worker-pool-status (New)

**Request:**
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

### WebSocket /ws/conversion/{conversion_id}

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8080/ws/conversion/550e8400-e29b-41d4-a716-446655440000');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${data.progress}% - ${data.message}`);
};
```

**Messages:**
```json
{
  "status": "processing",
  "progress": 45,
  "message": "Processing page 9/20",
  "step": "page_processing"
}
```

## Testing

### Run Test Script

```bash
cd backend
python test_worker_pool.py
```

This script:
1. Starts multiple conversions concurrently
2. Monitors each conversion via WebSocket
3. Displays real-time progress
4. Shows final summary with completion status

### Manual Testing

```bash
# Terminal 1: Start backend
cd backend
python main.py --port 8080

# Terminal 2: Start first conversion
curl -X POST http://localhost:8080/convert \
  -F "filename=graph_r1.pdf"

# Terminal 3: Start second conversion (while first is running)
curl -X POST http://localhost:8080/convert \
  -F "filename=test3.pdf"

# Terminal 4: Monitor queue status
watch -n 1 'curl -s http://localhost:8080/worker-pool-status | jq'
```

## Performance Characteristics

### Before Implementation
- Single conversion: ~11 minutes (20 pages)
- Two conversions: ~22 minutes (sequential)
- API blocked during conversion

### After Implementation
- Single conversion: ~11 minutes (same)
- Two conversions: ~11 minutes (parallel)
- API returns immediately (< 100ms)
- Multiple conversions can run concurrently

### Resource Usage
- Memory: Minimal overhead per worker
- CPU: Scales with number of workers
- I/O: Efficient with thread pool

## Files Modified/Created

### New Files
- `backend/worker_pool.py` - Worker pool implementation
- `backend/progress_tracker.py` - Progress tracking utilities
- `backend/test_worker_pool.py` - Test script

### Modified Files
- `backend/main.py` - Integrated worker pool, updated endpoints

## Future Enhancements

1. **Persistent Task Queue**: Store tasks in database for recovery
2. **Task Prioritization**: Priority queue for urgent conversions
3. **Metrics Collection**: Prometheus metrics for monitoring
4. **Rate Limiting**: Limit concurrent conversions per user
5. **Task Cancellation**: Allow users to cancel running conversions
6. **Distributed Workers**: Scale to multiple backend instances

## Troubleshooting

### Issue: Conversions not starting
- Check worker pool status: `curl http://localhost:8080/worker-pool-status`
- Check backend logs for errors
- Verify file exists in input directory

### Issue: WebSocket connection fails
- Ensure WebSocket URL is correct
- Check browser console for connection errors
- Verify backend is running and accessible

### Issue: High memory usage
- Reduce NUM_WORKERS
- Check for memory leaks in conversion process
- Monitor with: `watch -n 1 'ps aux | grep python'`

## References

- [Python Threading Documentation](https://docs.python.org/3/library/threading.html)
- [FastAPI WebSocket Documentation](https://fastapi.tiangolo.com/advanced/websockets/)
- [Queue Module Documentation](https://docs.python.org/3/library/queue.html)

