import os
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, FileResponse
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

from dots_ocr_service.parser import DotsOCRParser
from worker_pool import WorkerPool
from doc_service.document_converter_manager import DocumentConverterManager
from deepseek_ocr_service.deepseek_ocr_converter import DeepSeekOCRConverter
from pathlib import Path

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

# Initialize document converter manager for Word/Excel/TXT files
doc_converter_manager = DocumentConverterManager(
    input_folder=INPUT_DIR,
    output_folder=OUTPUT_DIR,
    add_section_numbers=True
)

# Initialize DeepSeek OCR converter for image files
deepseek_ocr_ip = os.getenv("DEEPSEEK_OCR_IP", "localhost")
deepseek_ocr_port = os.getenv("DEEPSEEK_OCR_PORT", "8005")
deepseek_ocr_model = os.getenv("DEEPSEEK_OCR_MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
deepseek_ocr_temperature = float(os.getenv("DEEPSEEK_OCR_TEMPERATURE", "0.0"))
deepseek_ocr_max_tokens = int(os.getenv("DEEPSEEK_OCR_MAX_TOKENS", "8192"))
deepseek_ocr_ngram_size = int(os.getenv("DEEPSEEK_OCR_NGRAM_SIZE", "30"))
deepseek_ocr_window_size = int(os.getenv("DEEPSEEK_OCR_WINDOW_SIZE", "90"))
deepseek_ocr_whitelist_str = os.getenv("DEEPSEEK_OCR_WHITELIST_TOKEN_IDS", "128821,128822")
deepseek_ocr_whitelist = [int(x.strip()) for x in deepseek_ocr_whitelist_str.split(",") if x.strip()]

deepseek_ocr_converter = DeepSeekOCRConverter(
    api_base=f"http://{deepseek_ocr_ip}:{deepseek_ocr_port}/v1",
    model_name=deepseek_ocr_model,
    temperature=deepseek_ocr_temperature,
    max_tokens=deepseek_ocr_max_tokens,
    ngram_size=deepseek_ocr_ngram_size,
    window_size=deepseek_ocr_window_size,
    whitelist_token_ids=deepseek_ocr_whitelist,
)

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


def _create_parser_progress_callback(conversion_id: str):
    """Create a progress callback for the parser"""
    def progress_callback(progress: int, message: str = ""):
        """Callback for parser progress updates"""
        try:
            conversion_manager.update_conversion(
                conversion_id,
                progress=progress,
                message=message,
            )
            asyncio.run(connection_manager.broadcast(conversion_id, {
                "status": "processing",
                "progress": progress,
                "message": message,
            }))
            logger.info(f"Conversion {conversion_id}: {progress}% - {message}")
        except Exception as e:
            logger.error(f"Error in parser progress callback: {str(e)}")

    return progress_callback


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


def _check_markdown_exists(file_name_without_ext: str) -> tuple:
    """
    Check if markdown file exists for a document.
    Handles both single markdown files and multi-page markdown files.
    Both OCR and doc_service converters use _nohf.md format.

    Returns:
    - (markdown_exists: bool, markdown_path: str or None, is_multipage: bool)
    """
    output_dir = os.path.join(OUTPUT_DIR, file_name_without_ext)

    # Check for single markdown file (for single-page documents, images, Word, Excel, TXT)
    markdown_path_nohf = os.path.join(output_dir, f"{file_name_without_ext}_nohf.md")
    if os.path.exists(markdown_path_nohf):
        return True, markdown_path_nohf, False

    # Check for multi-page markdown files (for PDFs with multiple pages)
    if os.path.exists(output_dir):
        # Look for page-specific markdown files
        page_files = []
        for file in os.listdir(output_dir):
            if file.endswith("_nohf.md") and "_page_" in file:
                page_files.append(file)

        if page_files:
            # Sort by page number
            page_files.sort(key=lambda x: int(x.split("_page_")[1].split("_")[0]))
            return True, os.path.join(output_dir, page_files[0]), True

    return False, None, False


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

                # Check if markdown file exists (handles both single and multi-page)
                file_name_without_ext = os.path.splitext(filename)[0]
                markdown_exists, markdown_path, is_multipage = _check_markdown_exists(file_name_without_ext)

                documents.append({
                    "filename": filename,
                    "file_path": file_path,
                    "file_size": file_size,
                    "upload_time": upload_time,
                    "markdown_exists": markdown_exists,
                    "markdown_path": markdown_path if markdown_exists else None,
                    "is_multipage": is_multipage,
                })

        return JSONResponse(content={
            "status": "success",
            "documents": documents,
            "total": len(documents),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


def _convert_with_doc_service(filename: str, conversion_id: str = None, progress_callback=None):
    """Background task to convert document using doc_service - executed by worker pool"""
    file_path = os.path.join(INPUT_DIR, filename)
    file_path_obj = Path(file_path)

    logger.info(f"Starting doc_service conversion for {filename}")

    # Send progress updates
    if progress_callback:
        progress_callback(10, "Starting document conversion...")

    # Find the appropriate converter
    converter = doc_converter_manager.find_converter_for_file(file_path_obj)
    if not converter:
        raise Exception(f"No converter found for file: {filename}")

    if progress_callback:
        progress_callback(30, "Converting document to markdown...")

    # Create output directory structure matching OCR parser format
    # output/{filename_without_ext}/{filename_without_ext}_nohf.md
    filename_without_ext = file_path_obj.stem
    output_subdir = Path(OUTPUT_DIR) / filename_without_ext
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Use _nohf.md suffix to match OCR format for frontend grid compatibility
    output_filename = filename_without_ext + "_nohf.md"
    output_path = output_subdir / output_filename

    # Convert the file
    success = converter.convert_file(file_path_obj, output_path)

    if not success:
        raise Exception(f"Failed to convert {filename}")

    if progress_callback:
        progress_callback(90, "Conversion complete, finalizing...")

    # Return results in a format similar to OCR parser
    results = [{
        "page_no": 0,
        "md_content_path": str(output_path),
        "file_path": file_path,
        "converter_type": "doc_service",
        "converter_name": converter.get_converter_info()["name"]
    }]

    logger.info(f"Doc_service conversion completed successfully for {filename}")
    return results


def _convert_document_background(filename: str, prompt_mode: str, conversion_id: str = None, progress_callback=None):
    """Background task to convert document using OCR parser - executed by worker pool"""
    file_path = os.path.join(INPUT_DIR, filename)

    logger.info(f"Starting OCR conversion for {filename}")

    # Parse the file using the OCR parser with progress callback
    results = parser.parse_file(
        file_path,
        output_dir=OUTPUT_DIR,
        prompt_mode=prompt_mode,
        progress_callback=progress_callback,
    )

    logger.info(f"OCR conversion completed successfully for {filename}")
    return results


def _convert_with_deepseek_ocr(filename: str, conversion_id: str = None, progress_callback=None):
    """Background task to convert image using DeepSeek OCR - executed by worker pool"""
    file_path = os.path.join(INPUT_DIR, filename)
    file_path_obj = Path(file_path)

    logger.info(f"Starting DeepSeek OCR conversion for {filename}")

    # Send progress updates
    if progress_callback:
        progress_callback(10, "Starting DeepSeek OCR conversion...")

    # Validate that this is an image file
    if not deepseek_ocr_converter.is_supported_file(file_path_obj):
        raise Exception(f"File type not supported by DeepSeek OCR: {filename}")

    if progress_callback:
        progress_callback(30, "Converting image to markdown with DeepSeek OCR...")

    # Create output directory structure matching OCR parser format
    # output/{filename_without_ext}/{filename_without_ext}_nohf.md
    filename_without_ext = file_path_obj.stem
    output_subdir = Path(OUTPUT_DIR) / filename_without_ext
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Use _nohf.md suffix to match OCR format for frontend grid compatibility
    output_filename = filename_without_ext + "_nohf.md"
    output_path = output_subdir / output_filename

    if progress_callback:
        progress_callback(50, "Calling DeepSeek OCR API...")

    # Convert the file
    success = deepseek_ocr_converter.convert_file(file_path_obj, output_path)

    if not success:
        raise Exception(f"Failed to convert {filename} with DeepSeek OCR")

    if progress_callback:
        progress_callback(90, "Conversion complete, finalizing...")

    # Return results in a format similar to OCR parser
    results = [{
        "page_no": 0,
        "md_content_path": str(output_path),
        "file_path": str(file_path),
        "converter_type": "deepseek_ocr",
        "converter_name": "DeepSeek OCR"
    }]

    logger.info(f"DeepSeek OCR conversion completed successfully for {filename}")
    return results


@app.post("/convert-doc")
async def convert_document_with_doc_service(filename: str = Form(...)):
    """
    Convert Word/Excel/TXT documents using doc_service converters (non-blocking).

    This endpoint is specifically for structured documents (Word, Excel, TXT files)
    that don't require OCR processing.

    Returns immediately with a conversion ID. Use WebSocket to track progress.
    Multiple conversions can run concurrently using the worker pool.

    Parameters:
    - filename: The name of the file to convert (must be in input folder)

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
        file_path_obj = Path(file_path)

        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        # Validate that this is a doc_service supported file
        doc_service_extensions = ['.docx', '.doc', '.xlsx', '.xlsm', '.xlsb', '.xls', '.txt', '.csv', '.tsv', '.log', '.text']
        file_extension = file_path_obj.suffix.lower()

        if file_extension not in doc_service_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported by doc_service. Use /convert endpoint for PDF/images."
            )

        # Create conversion task
        conversion_id = conversion_manager.create_conversion(filename)

        # Update status to processing
        conversion_manager.update_conversion(
            conversion_id,
            status="processing",
            started_at=datetime.now().isoformat(),
            message="Document conversion queued..."
        )

        # Broadcast initial status
        await connection_manager.broadcast(conversion_id, {
            "status": "processing",
            "progress": 5,
            "message": "Document conversion queued...",
        })

        # Create progress callback for this conversion
        progress_callback = _create_parser_progress_callback(conversion_id)

        # Submit task to worker pool (using doc_service converter)
        success = worker_pool.submit_task(
            conversion_id=conversion_id,
            func=_convert_with_doc_service,
            args=(filename,),
            kwargs={
                "conversion_id": conversion_id,
                "progress_callback": progress_callback,
            }
        )

        if not success:
            raise HTTPException(status_code=409, detail=f"Conversion already in progress for: {filename}")

        # Return immediately with conversion ID
        return JSONResponse(content={
            "status": "accepted",
            "conversion_id": conversion_id,
            "filename": filename,
            "message": "Document conversion task started. Use WebSocket to track progress.",
            "converter_type": "doc_service",
            "queue_size": worker_pool.get_queue_size(),
            "active_tasks": worker_pool.get_active_tasks_count(),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting doc_service conversion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting conversion: {str(e)}")


@app.post("/convert-deepseek")
async def convert_document_with_deepseek_ocr(filename: str = Form(...)):
    """
    Convert images using DeepSeek OCR service (non-blocking).

    This endpoint is specifically for image files that will be converted using
    the DeepSeek OCR model for high-quality OCR and markdown generation.

    Returns immediately with a conversion ID. Use WebSocket to track progress.
    Multiple conversions can run concurrently using the worker pool.

    Parameters:
    - filename: The name of the image file to convert (must be in input folder)

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
        file_path_obj = Path(file_path)

        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        # Validate that this is a DeepSeek OCR supported file (images only)
        deepseek_ocr_extensions = deepseek_ocr_converter.get_supported_extensions()
        file_extension = file_path_obj.suffix.lower()

        if file_extension not in deepseek_ocr_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported by DeepSeek OCR. Supported types: {', '.join(deepseek_ocr_extensions)}"
            )

        # Create conversion task
        conversion_id = conversion_manager.create_conversion(filename)

        # Update status to processing
        conversion_manager.update_conversion(
            conversion_id,
            status="processing",
            started_at=datetime.now().isoformat(),
            message="DeepSeek OCR conversion queued..."
        )

        # Broadcast initial status
        await connection_manager.broadcast(conversion_id, {
            "status": "processing",
            "progress": 5,
            "message": "DeepSeek OCR conversion queued...",
        })

        # Create progress callback for this conversion
        progress_callback = _create_parser_progress_callback(conversion_id)

        # Submit task to worker pool (using DeepSeek OCR converter)
        success = worker_pool.submit_task(
            conversion_id=conversion_id,
            func=_convert_with_deepseek_ocr,
            args=(filename,),
            kwargs={
                "conversion_id": conversion_id,
                "progress_callback": progress_callback,
            }
        )

        if not success:
            raise HTTPException(status_code=409, detail=f"Conversion already in progress for: {filename}")

        # Return immediately with conversion ID
        return JSONResponse(content={
            "status": "accepted",
            "conversion_id": conversion_id,
            "filename": filename,
            "message": "DeepSeek OCR conversion task started. Use WebSocket to track progress.",
            "converter_type": "deepseek_ocr",
            "queue_size": worker_pool.get_queue_size(),
            "active_tasks": worker_pool.get_active_tasks_count(),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting DeepSeek OCR conversion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting conversion: {str(e)}")


@app.post("/convert")
async def convert_document(filename: str = Form(...), prompt_mode: str = Form("prompt_layout_all_en")):
    """
    Trigger document conversion using OCR parser (non-blocking).

    This endpoint is for PDF and image files that require OCR processing.

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

        # Create progress callback for this conversion
        progress_callback = _create_parser_progress_callback(conversion_id)

        # Submit task to worker pool
        success = worker_pool.submit_task(
            conversion_id=conversion_id,
            func=_convert_document_background,
            args=(filename, prompt_mode),
            kwargs={
                "conversion_id": conversion_id,
                "progress_callback": progress_callback,
            }
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


@app.get("/markdown-files/{filename}")
async def list_markdown_files(filename: str):
    """
    List all markdown files associated with a document.
    Handles both single markdown files and multi-page markdown files.

    Parameters:
    - filename: The name of the file (without extension)

    Returns:
    - List of markdown files with their page numbers
    """
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Validate filename to prevent directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        output_dir = os.path.join(OUTPUT_DIR, filename)

        if not os.path.exists(output_dir):
            raise HTTPException(status_code=404, detail=f"No output directory for: {filename}")

        markdown_files = []

        # Check for single markdown file
        single_md_path = os.path.join(output_dir, f"{filename}_nohf.md")
        if os.path.exists(single_md_path):
            markdown_files.append({
                "filename": f"{filename}_nohf.md",
                "path": single_md_path,
                "page_no": None,
                "is_multipage": False,
            })

        # Check for multi-page markdown files
        page_files = []
        for file in os.listdir(output_dir):
            if file.endswith("_nohf.md") and "_page_" in file:
                page_no = int(file.split("_page_")[1].split("_")[0])
                page_files.append({
                    "filename": file,
                    "path": os.path.join(output_dir, file),
                    "page_no": page_no,
                    "is_multipage": True,
                })

        # Sort by page number
        page_files.sort(key=lambda x: x["page_no"])
        markdown_files.extend(page_files)

        if not markdown_files:
            raise HTTPException(status_code=404, detail=f"No markdown files found for: {filename}")

        return JSONResponse(content={
            "status": "success",
            "filename": filename,
            "markdown_files": markdown_files,
            "total": len(markdown_files),
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing markdown files: {str(e)}")


@app.get("/markdown/{filename}")
async def get_markdown_content(filename: str, page_no: int = None):
    """
    Get the markdown content of a converted document.
    Supports both single markdown files and page-specific markdown files.
    Both OCR and doc_service converters use _nohf.md format.

    Parameters:
    - filename: The name of the file (without extension)
    - page_no: Optional page number for multi-page documents

    Returns:
    - The markdown file content
    """
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Validate filename to prevent directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Determine which markdown file to read
        if page_no is not None:
            # Read page-specific markdown file (multi-page PDFs)
            markdown_path = os.path.join(OUTPUT_DIR, filename, f"{filename}_page_{page_no}_nohf.md")
        else:
            # Read single markdown file (single-page documents, images, Word, Excel, TXT)
            markdown_path = os.path.join(OUTPUT_DIR, filename, f"{filename}_nohf.md")

        if not os.path.exists(markdown_path):
            raise HTTPException(status_code=404, detail=f"Markdown file not found for: {filename}")

        # Read and return the markdown content
        with open(markdown_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Normalize any single-line tables in the content
        from dots_ocr_service.utils.format_transformer import normalize_markdown_table, is_markdown_table

        # Process the content line by line to find and normalize tables
        lines = content.split('\n')
        normalized_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Check if this line contains a markdown table
            if is_markdown_table(line):
                # Normalize the table
                normalized_table = normalize_markdown_table(line)
                normalized_lines.append(normalized_table)
            else:
                normalized_lines.append(line)
            i += 1

        content = '\n'.join(normalized_lines)

        return JSONResponse(content={
            "status": "success",
            "filename": filename,
            "page_no": page_no,
            "content": content,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading markdown file: {str(e)}")


@app.put("/markdown/{filename}")
async def update_markdown_content(filename: str, request: Request, page_no: int = None):
    """
    Update the markdown content of a converted document.
    Supports both single markdown files and page-specific markdown files.
    Both OCR and doc_service converters use _nohf.md format.

    Parameters:
    - filename: The name of the file (without extension)
    - page_no: Optional page number for multi-page documents
    - request body: JSON with 'content' field containing the new markdown content

    Returns:
    - Success status
    """
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Validate filename to prevent directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Get the content from request body
        body = await request.json()
        content = body.get("content")

        if content is None:
            raise HTTPException(status_code=400, detail="No content provided")

        # Determine which markdown file to write
        if page_no is not None:
            # Write to page-specific markdown file (multi-page PDFs)
            markdown_path = os.path.join(OUTPUT_DIR, filename, f"{filename}_page_{page_no}_nohf.md")
        else:
            # Write to single markdown file (single-page documents, images, Word, Excel, TXT)
            markdown_path = os.path.join(OUTPUT_DIR, filename, f"{filename}_nohf.md")

        if not os.path.exists(markdown_path):
            raise HTTPException(status_code=404, detail=f"Markdown file not found for: {filename}")

        # Write the updated content
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(content)

        return JSONResponse(content={
            "status": "success",
            "filename": filename,
            "page_no": page_no,
            "message": "Markdown content updated successfully",
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating markdown file: {str(e)}")


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete a document and all its associated files.
    Removes the uploaded file from input folder and all associated markdown/output files.

    Parameters:
    - filename: The name of the file to delete (with extension)

    Returns:
    - Success status with details of deleted files
    """
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Validate filename to prevent directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        deleted_files = []
        errors = []

        # Delete the uploaded file from input folder
        input_file_path = os.path.join(INPUT_DIR, filename)
        if os.path.exists(input_file_path):
            try:
                os.remove(input_file_path)
                deleted_files.append(input_file_path)
                logger.info(f"Deleted input file: {input_file_path}")
            except Exception as e:
                error_msg = f"Failed to delete input file {input_file_path}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        else:
            logger.warning(f"Input file not found: {input_file_path}")

        # Delete the output folder and all its contents
        file_name_without_ext = os.path.splitext(filename)[0]
        output_folder_path = os.path.join(OUTPUT_DIR, file_name_without_ext)

        if os.path.exists(output_folder_path) and os.path.isdir(output_folder_path):
            try:
                import shutil
                shutil.rmtree(output_folder_path)
                deleted_files.append(output_folder_path)
                logger.info(f"Deleted output folder: {output_folder_path}")
            except Exception as e:
                error_msg = f"Failed to delete output folder {output_folder_path}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        else:
            logger.warning(f"Output folder not found: {output_folder_path}")

        # Delete the JSONL results file if it exists
        jsonl_file_path = os.path.join(OUTPUT_DIR, f"{file_name_without_ext}.jsonl")
        if os.path.exists(jsonl_file_path):
            try:
                os.remove(jsonl_file_path)
                deleted_files.append(jsonl_file_path)
                logger.info(f"Deleted JSONL file: {jsonl_file_path}")
            except Exception as e:
                error_msg = f"Failed to delete JSONL file {jsonl_file_path}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

        # If no files were deleted and no errors occurred, the file didn't exist
        if not deleted_files and not errors:
            raise HTTPException(status_code=404, detail=f"Document not found: {filename}")

        # Return success even if some deletions failed
        response_data = {
            "status": "success" if not errors else "partial_success",
            "filename": filename,
            "deleted_files": deleted_files,
            "message": f"Deleted {len(deleted_files)} file(s)/folder(s)",
        }

        if errors:
            response_data["errors"] = errors
            response_data["message"] += f" with {len(errors)} error(s)"

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@app.get("/image/{filename}")
async def get_image(filename: str, page_no: int = None):
    """
    Get the JPG image of a converted document.
    Supports both single image files and page-specific image files.

    Parameters:
    - filename: The name of the file (without extension)
    - page_no: Optional page number for multi-page documents

    Returns:
    - The JPG image file
    """
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Validate filename to prevent directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Determine which image file to serve
        if page_no is not None:
            # Serve page-specific image file
            image_path = os.path.join(OUTPUT_DIR, filename, f"{filename}_page_{page_no}.jpg")
        else:
            # Try to find the single image file
            image_path = os.path.join(OUTPUT_DIR, filename, f"{filename}.jpg")

        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image file not found for: {filename}")

        # Return the image file
        return FileResponse(
            image_path,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Content-Disposition": f'inline; filename="{os.path.basename(image_path)}"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving image file: {str(e)}")


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

