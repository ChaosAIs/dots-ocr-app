import os
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
from dotenv import load_dotenv
from datetime import datetime
import asyncio
import threading
from typing import Dict, Set
import uuid
import logging

from ocr_service.parser import DotsOCRParser
from worker_pool import WorkerPool

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Dots OCR API",
    description="API for document OCR parsing with layout detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize parser
parser = DotsOCRParser()

# Define input and output directories
INPUT_DIR = os.path.join(os.path.dirname(__file__), "input")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Create directories if they don't exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get number of worker threads from environment (default: 4)
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))


# Conversion status tracking
class ConversionManager:
    """Manages document conversion tasks and their status"""

    def __init__(self):
        self.conversions: Dict[str, dict] = {}  # conversion_id -> status info
        self.lock = threading.Lock()

    def create_conversion(self, filename: str) -> str:
        """Create a new conversion task and return its ID"""
        conversion_id = str(uuid.uuid4())
        with self.lock:
            self.conversions[conversion_id] = {
                "id": conversion_id,
                "filename": filename,
                "status": "pending",
                "progress": 0,
                "message": "Conversion queued",
                "created_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "error": None,
            }
        return conversion_id

    def update_conversion(self, conversion_id: str, **kwargs):
        """Update conversion status"""
        with self.lock:
            if conversion_id in self.conversions:
                self.conversions[conversion_id].update(kwargs)

    def get_conversion(self, conversion_id: str) -> dict:
        """Get conversion status"""
        with self.lock:
            return self.conversions.get(conversion_id, {})

    def delete_conversion(self, conversion_id: str):
        """Delete conversion record"""
        with self.lock:
            if conversion_id in self.conversions:
                del self.conversions[conversion_id]


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for progress updates"""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}  # conversion_id -> set of websockets
        self.lock = threading.Lock()

    async def connect(self, conversion_id: str, websocket: WebSocket):
        """Register a new WebSocket connection"""
        await websocket.accept()
        with self.lock:
            if conversion_id not in self.active_connections:
                self.active_connections[conversion_id] = set()
            self.active_connections[conversion_id].add(websocket)

    async def disconnect(self, conversion_id: str, websocket: WebSocket):
        """Unregister a WebSocket connection"""
        with self.lock:
            if conversion_id in self.active_connections:
                self.active_connections[conversion_id].discard(websocket)
                if not self.active_connections[conversion_id]:
                    del self.active_connections[conversion_id]

    async def broadcast(self, conversion_id: str, message: dict):
        """Send message to all connected clients for a conversion"""
        with self.lock:
            connections = self.active_connections.get(conversion_id, set()).copy()

        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending message: {e}")
                await self.disconnect(conversion_id, connection)


conversion_manager = ConversionManager()
connection_manager = ConnectionManager()


def _worker_progress_callback(conversion_id: str, status: str, result=None, error=None):
    """Callback for worker pool progress updates"""
    try:
        if status == "completed":
            conversion_manager.update_conversion(
                conversion_id,
                status="completed",
                progress=100,
                message="Conversion completed successfully",
                completed_at=datetime.now().isoformat(),
            )
            asyncio.run(connection_manager.broadcast(conversion_id, {
                "status": "completed",
                "progress": 100,
                "message": "Conversion completed successfully",
                "results": result,
            }))
        elif status == "error":
            conversion_manager.update_conversion(
                conversion_id,
                status="error",
                progress=0,
                message=f"Conversion failed: {error}",
                error=error,
                completed_at=datetime.now().isoformat(),
            )
            asyncio.run(connection_manager.broadcast(conversion_id, {
                "status": "error",
                "progress": 0,
                "message": f"Conversion failed: {error}",
                "error": error,
            }))
    except Exception as e:
        logger.error(f"Error in progress callback: {str(e)}")


# Initialize worker pool
worker_pool = WorkerPool(num_workers=NUM_WORKERS, progress_callback=_worker_progress_callback)


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Dots OCR API",
        "version": "1.0.0",
        "description": "Document OCR parsing with layout detection",
        "endpoints": {
            "parse_file": "/parse",
            "health": "/health",
            "config": "/config"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Dots OCR API"
    }


@app.get("/config")
async def get_config():
    """Get current configuration for both backend and frontend"""
    # Get frontend configuration from environment variables
    app_domain = os.getenv("APP_DOMAIN", "http://localhost:3000")
    base_path = os.getenv("BASE_PATH", "")
    iam_domain = os.getenv("IAM_DOMAIN", "http://localhost:5000")
    client_id = os.getenv("CLIENT_ID", "dots-ocr-app")
    iam_scope = os.getenv("IAM_SCOPE", "openid profile email")
    api_domain = os.getenv("API_DOMAIN", "http://localhost:8080")

    return {
        # Frontend configuration
        "basePath": base_path,
        "appDomain": app_domain,
        "apiDomain": api_domain,
        "iamDomain": iam_domain,
        "clientId": client_id,
        "iamScope": iam_scope,

        # Backend OCR configuration
        "vllm_ip": parser.ip,
        "vllm_port": parser.port,
        "model_name": parser.model_name,
        "temperature": parser.temperature,
        "top_p": parser.top_p,
        "max_completion_tokens": parser.max_completion_tokens,
        "num_thread": parser.num_thread,
        "dpi": parser.dpi,
        "output_dir": parser.output_dir,
        "min_pixels": parser.min_pixels,
        "max_pixels": parser.max_pixels,
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a document file (PDF, image, DOC, EXCEL) to the input folder.

    Parameters:
    - file: The file to upload

    Returns:
    - JSON with upload status and file information
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Save uploaded file to input directory
        file_path = os.path.join(INPUT_DIR, file.filename)

        # Prevent directory traversal attacks
        if not os.path.abspath(file_path).startswith(os.path.abspath(INPUT_DIR)):
            raise HTTPException(status_code=400, detail="Invalid file path")

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        file_size = os.path.getsize(file_path)
        upload_time = datetime.now().isoformat()

        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "file_path": file_path,
            "file_size": file_size,
            "upload_time": upload_time,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.get("/documents")
async def list_documents():
    """
    List all uploaded documents with their conversion status.

    Returns:
    - JSON with list of documents and their markdown conversion status
    """
    try:
        documents = []

        # List all files in input directory
        if os.path.exists(INPUT_DIR):
            for filename in os.listdir(INPUT_DIR):
                file_path = os.path.join(INPUT_DIR, filename)

                # Skip directories
                if not os.path.isfile(file_path):
                    continue

                # Get file info
                file_size = os.path.getsize(file_path)
                file_stat = os.stat(file_path)
                upload_time = datetime.fromtimestamp(file_stat.st_mtime).isoformat()

                # Check if markdown file exists
                file_name_without_ext = os.path.splitext(filename)[0]
                markdown_path = os.path.join(OUTPUT_DIR, file_name_without_ext, f"{file_name_without_ext}_nohf.md")
                markdown_exists = os.path.exists(markdown_path)

                documents.append({
                    "filename": filename,
                    "file_path": file_path,
                    "file_size": file_size,
                    "upload_time": upload_time,
                    "markdown_exists": markdown_exists,
                    "markdown_path": markdown_path if markdown_exists else None,
                })

        return JSONResponse(content={
            "status": "success",
            "documents": documents,
            "total": len(documents),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


def _convert_document_background(filename: str, prompt_mode: str):
    """Background task to convert document - executed by worker pool"""
    file_path = os.path.join(INPUT_DIR, filename)

    logger.info(f"Starting conversion for {filename}")

    # Parse the file using the OCR parser
    results = parser.parse_file(
        file_path,
        output_dir=OUTPUT_DIR,
        prompt_mode=prompt_mode,
    )

    logger.info(f"Conversion completed successfully for {filename}")
    return results


@app.post("/convert")
async def convert_document(filename: str = Form(...), prompt_mode: str = Form("prompt_layout_all_en")):
    """
    Trigger document conversion (non-blocking).

    Returns immediately with a conversion ID. Use WebSocket to track progress.
    Multiple conversions can run concurrently using the worker pool.

    Parameters:
    - filename: The name of the file to convert (must be in input folder)
    - prompt_mode: The prompt mode to use (default: prompt_layout_all_en)

    Returns:
    - JSON with conversion_id for tracking progress
    """
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Validate filename to prevent directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        file_path = os.path.join(INPUT_DIR, filename)

        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        # Create conversion task
        conversion_id = conversion_manager.create_conversion(filename)

        # Update status to processing
        conversion_manager.update_conversion(
            conversion_id,
            status="processing",
            started_at=datetime.now().isoformat(),
            message="Conversion queued, waiting for worker..."
        )

        # Broadcast initial status
        await connection_manager.broadcast(conversion_id, {
            "status": "processing",
            "progress": 5,
            "message": "Conversion queued, waiting for worker...",
        })

        # Submit task to worker pool
        success = worker_pool.submit_task(
            conversion_id=conversion_id,
            func=_convert_document_background,
            args=(filename, prompt_mode),
        )

        if not success:
            raise HTTPException(status_code=409, detail=f"Conversion already in progress for: {filename}")

        # Return immediately with conversion ID
        return JSONResponse(content={
            "status": "accepted",
            "conversion_id": conversion_id,
            "filename": filename,
            "message": "Conversion task started. Use WebSocket to track progress.",
            "queue_size": worker_pool.get_queue_size(),
            "active_tasks": worker_pool.get_active_tasks_count(),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting conversion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting conversion: {str(e)}")


@app.get("/conversion-status/{conversion_id}")
async def get_conversion_status(conversion_id: str):
    """
    Get the status of a conversion task.

    Parameters:
    - conversion_id: The conversion task ID

    Returns:
    - JSON with conversion status and progress
    """
    try:
        conversion = conversion_manager.get_conversion(conversion_id)
        if not conversion:
            raise HTTPException(status_code=404, detail=f"Conversion not found: {conversion_id}")

        return JSONResponse(content=conversion)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting conversion status: {str(e)}")


@app.get("/worker-pool-status")
async def get_worker_pool_status():
    """
    Get the status of the worker pool.

    Returns:
    - JSON with queue size and active tasks count
    """
    try:
        return JSONResponse(content={
            "status": "ok",
            "queue_size": worker_pool.get_queue_size(),
            "active_tasks": worker_pool.get_active_tasks_count(),
            "num_workers": NUM_WORKERS,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting worker pool status: {str(e)}")


@app.websocket("/ws/conversion/{conversion_id}")
async def websocket_conversion_progress(websocket: WebSocket, conversion_id: str):
    """
    WebSocket endpoint for real-time conversion progress updates.

    Parameters:
    - conversion_id: The conversion task ID

    Sends:
    - JSON messages with status, progress, and message
    """
    try:
        # Connect the WebSocket
        await connection_manager.connect(conversion_id, websocket)

        # Send initial status
        conversion = conversion_manager.get_conversion(conversion_id)
        if conversion:
            await websocket.send_json(conversion)

        # Keep connection alive and listen for disconnection
        while True:
            # This will raise WebSocketDisconnect when client disconnects
            await websocket.receive_text()
            # Optional: handle incoming messages if needed

    except WebSocketDisconnect:
        await connection_manager.disconnect(conversion_id, websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await connection_manager.disconnect(conversion_id, websocket)


@app.get("/markdown/{filename}")
async def get_markdown_content(filename: str):
    """
    Get the markdown content of a converted document.

    Parameters:
    - filename: The name of the file (without extension)

    Returns:
    - The markdown file content
    """
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Validate filename to prevent directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Try to find the markdown file
        markdown_path = os.path.join(OUTPUT_DIR, filename, f"{filename}_nohf.md")

        if not os.path.exists(markdown_path):
            raise HTTPException(status_code=404, detail=f"Markdown file not found for: {filename}")

        # Read and return the markdown content
        with open(markdown_path, "r", encoding="utf-8") as f:
            content = f.read()

        return JSONResponse(content={
            "status": "success",
            "filename": filename,
            "content": content,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading markdown file: {str(e)}")


@app.post("/parse")
async def parse_file(
    file: UploadFile = File(...),
    prompt_mode: str = Form("prompt_layout_all_en"),
    bbox: str = Form(None),
):
    """
    Parse a PDF or image file for document layout and text extraction.
    
    Parameters:
    - file: The PDF or image file to parse
    - prompt_mode: The prompt mode to use (default: prompt_layout_all_en)
    - bbox: Optional bounding box as JSON string [x1, y1, x2, y2] for grounding OCR
    
    Returns:
    - JSON with parsing results including layout info, images, and markdown content
    """
    temp_dir = None
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Create temporary directory for input
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Parse bbox if provided
        parsed_bbox = None
        if bbox:
            try:
                parsed_bbox = json.loads(bbox)
                if not isinstance(parsed_bbox, list) or len(parsed_bbox) != 4:
                    raise ValueError("bbox must be a list of 4 numbers [x1, y1, x2, y2]")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid bbox format. Must be valid JSON")
        
        # Parse the file
        results = parser.parse_file(
            temp_file_path,
            prompt_mode=prompt_mode,
            bbox=parsed_bbox,
        )
        
        # Prepare response
        response_data = {
            "status": "success",
            "filename": file.filename,
            "prompt_mode": prompt_mode,
            "results": results,
        }
        
        return JSONResponse(content=response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@app.post("/parse-batch")
async def parse_batch(
    files: list[UploadFile] = File(...),
    prompt_mode: str = Form("prompt_layout_all_en"),
):
    """
    Parse multiple PDF or image files in batch.
    
    Parameters:
    - files: List of PDF or image files to parse
    - prompt_mode: The prompt mode to use (default: prompt_layout_all_en)
    
    Returns:
    - JSON with parsing results for all files
    """
    temp_dir = None
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Create temporary directory for input
        temp_dir = tempfile.mkdtemp()
        
        batch_results = []
        
        for file in files:
            if not file.filename:
                continue
            
            temp_file_path = os.path.join(temp_dir, file.filename)
            
            # Save uploaded file
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            try:
                # Parse the file
                results = parser.parse_file(
                    temp_file_path,
                    prompt_mode=prompt_mode,
                )
                
                batch_results.append({
                    "filename": file.filename,
                    "status": "success",
                    "results": results,
                })
            except Exception as e:
                batch_results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e),
                })
        
        return JSONResponse(content={
            "status": "completed",
            "total_files": len(files),
            "results": batch_results,
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")
    
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import uvicorn
    import sys

    # Get host and port from environment or use defaults
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))

    # Check for command-line arguments
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:], 1):
            if arg == "--port" and i < len(sys.argv) - 1:
                port = int(sys.argv[i + 1])
            elif arg == "--host" and i < len(sys.argv) - 1:
                host = sys.argv[i + 1]

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

