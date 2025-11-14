# Troubleshooting Guide

## Inference Server Connection Errors

### Symptoms
- Error message: `ERROR:worker_pool:Error in worker 0: Connection error.`
- HTTP 500 errors from inference API
- Conversions fail during "Running model inference" phase
- OpenAI client retry attempts visible in logs

### Root Cause
The vLLM inference server (running on port 8001 by default) is either:
1. Not running
2. Overloaded or out of memory
3. Crashed or in an error state
4. Unable to process the request (image too large, etc.)

### Diagnostic Steps

#### 1. Check if inference server is running
```bash
# Check if the server is listening on port 8001
curl http://localhost:8001/health

# Or check available models
curl http://localhost:8001/v1/models
```

#### 2. Run the diagnostic script
```bash
cd backend
python check_inference_server.py
```

This will test:
- Server reachability
- Available models
- Simple inference test

#### 3. Check inference server logs
Look for error messages in the vLLM server logs, such as:
- Out of memory errors
- CUDA errors (if using GPU)
- Model loading errors
- Timeout errors

#### 4. Check system resources
```bash
# Check GPU memory (if using GPU)
nvidia-smi

# Check CPU and RAM usage
top
# or
htop

# Check disk space
df -h
```

### Solutions

#### Solution 1: Restart the inference server
If the server is running but in an error state, restart it:
```bash
# Stop the vLLM server (method depends on how you started it)
# Then restart it with appropriate configuration
```

#### Solution 2: Reduce worker count
Edit `backend/.env` and reduce the number of concurrent workers:
```env
NUM_WORKERS=1
```

This reduces the load on the inference server by processing one document at a time.

#### Solution 3: Reduce image size/quality
Edit `backend/.env` to reduce DPI or set max pixels:
```env
DPI=150
MAX_PIXELS=1024000
```

Lower DPI means smaller images sent to the inference server.

#### Solution 4: Increase inference server resources
If using GPU:
- Reduce batch size in vLLM configuration
- Use a smaller model
- Add more GPU memory

If using CPU:
- Increase available RAM
- Reduce number of threads

#### Solution 5: Add timeout and retry configuration
The OpenAI client has built-in retry logic, but you can configure it:

Edit `backend/ocr_service/model/inference.py` to add timeout:
```python
client = OpenAI(
    api_key="{}".format(os.environ.get("API_KEY", "0")),
    base_url=addr,
    timeout=300.0,  # 5 minutes timeout
    max_retries=2,
)
```

### Common Error Messages

#### "HTTP Request: POST http://localhost:8001/v1/chat/completions HTTP/1.1 500 Internal Server Error"
- **Cause**: Inference server encountered an error processing the request
- **Solution**: Check inference server logs, restart server, reduce image size

#### "Connection error"
- **Cause**: Cannot connect to inference server or connection was lost
- **Solution**: Check if server is running, check network connectivity

#### "Retrying request to /chat/completions in X seconds"
- **Cause**: Request failed, OpenAI client is retrying
- **Solution**: This is normal for transient errors. If it keeps retrying, check server health

### Prevention

1. **Monitor server health**: Set up monitoring for the vLLM server
2. **Set resource limits**: Configure appropriate memory and timeout limits
3. **Use worker pool wisely**: Don't set NUM_WORKERS higher than your server can handle
4. **Test with small files first**: Before processing large documents, test with small images
5. **Implement health checks**: Add periodic health checks to detect server issues early

### Configuration Recommendations

For **development/testing** (limited resources):
```env
NUM_WORKERS=1
DPI=150
MAX_PIXELS=1024000
NUM_THREAD=16
```

For **production** (dedicated GPU server):
```env
NUM_WORKERS=4
DPI=200
MAX_PIXELS=None
NUM_THREAD=64
```

### Getting Help

If the issue persists:
1. Collect diagnostic information:
   - Output from `check_inference_server.py`
   - vLLM server logs
   - System resource usage (GPU/CPU/RAM)
   - Sample image that causes the error

2. Check vLLM documentation:
   - https://docs.vllm.ai/

3. Review the error logs carefully for specific error messages

