# CRITICAL: Set HF_HOME before ANY imports that might load transformers
# This must happen before qwen3_ocr_converter.py imports the transformers library
import os

# CRITICAL: Load .env file FIRST before any other imports
# This ensures JWT_SECRET_KEY and other env vars are available when modules are imported
# Use explicit path to .env file so it works regardless of working directory
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(_env_path)

# Load .env file manually to get HF_HOME before other imports
# (load_dotenv above handles most vars, but HF_HOME needs special handling for ~ expansion)
if os.path.exists(_env_path):
    with open(_env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Only set HF_HOME if not already set in environment
                if key == 'HF_HOME' and key not in os.environ:
                    # Expand ~ to user home directory
                    os.environ[key] = os.path.expanduser(value)
                    break

# Ensure HF_HOME is set (fallback to default if not in .env)
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = os.path.expanduser('~/huggingface_cache')

import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect, Request, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
from datetime import datetime
import asyncio
import threading
from typing import Dict, Set, Optional
import uuid
from uuid import UUID
import logging

from dots_ocr_service.parser import DotsOCRParser
from worker_pool import WorkerPool
from doc_service.document_converter_manager import DocumentConverterManager
from deepseek_ocr_service.deepseek_ocr_converter import DeepSeekOCRConverter
from gemma_ocr_service.gemma3_ocr_converter import Gemma3OCRConverter
from qwen_ocr_service.qwen3_ocr_converter import Qwen3OCRConverter
from pathlib import Path
import base64

# Import RAG service for chatbot functionality
from rag_service.chat_api import router as chat_router
from rag_service.indexer import (
    trigger_embedding_for_document,
    reindex_document,
    index_document_now,
)
from rag_service.vectorstore import delete_documents_by_source, get_collection_info, clear_collection, is_document_indexed

# Import GraphRAG for source-level deletion and Neo4j initialization
try:
    from rag_service.graph_rag import delete_graphrag_by_source_sync, GRAPH_RAG_INDEX_ENABLED, GRAPH_RAG_QUERY_ENABLED
    GRAPHRAG_DELETE_AVAILABLE = True
except ImportError:
    GRAPHRAG_DELETE_AVAILABLE = False
    GRAPH_RAG_INDEX_ENABLED = False
    GRAPH_RAG_QUERY_ENABLED = False

# Import database services
from db.database import init_db, get_db_session
from db.document_repository import DocumentRepository
from db.models import Document, UploadStatus, ConvertStatus, IndexStatus

# Import authentication
from auth.auth_api import router as auth_router

# Import chat session management
from chat_service.chat_session_api import router as chat_session_router

# Import task queue system
from queue_service import TaskQueueManager, TaskScheduler, QueueWorkerPool, TaskType, TaskPriority

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # ===== STARTUP =====
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Continue without database - will fall back to file-based status

    # Set up the connection managers with the current event loop
    loop = asyncio.get_event_loop()
    connection_manager.set_event_loop(loop)
    document_status_manager.set_event_loop(loop)
    # Start the background broadcast workers
    asyncio.create_task(connection_manager.start_broadcast_worker())
    asyncio.create_task(document_status_manager.start_broadcast_worker())
    logger.info("WebSocket broadcast workers started")

    # Initialize Neo4j indexes (including vector indexes) if GraphRAG indexing or querying is enabled
    if GRAPH_RAG_INDEX_ENABLED or GRAPH_RAG_QUERY_ENABLED:
        try:
            from rag_service.storage import Neo4jStorage
            neo4j_storage = Neo4jStorage()
            await neo4j_storage.ensure_indexes()
            logger.info("Neo4j indexes initialized (including vector indexes)")
        except Exception as e:
            logger.warning(f"Failed to initialize Neo4j indexes: {e}")

    # Initialize task queue system if enabled
    global task_queue_manager, task_scheduler, queue_worker_pool

    if TASK_QUEUE_ENABLED:
        logger.info("🚀 Initializing task queue system...")

        # Initialize task queue manager
        task_queue_manager = TaskQueueManager(
            stale_timeout_seconds=STALE_TASK_TIMEOUT,
            heartbeat_interval_seconds=HEARTBEAT_INTERVAL
        )

        # Initialize queue worker pool
        queue_worker_pool = QueueWorkerPool(
            num_workers=NUM_WORKERS,
            task_queue_manager=task_queue_manager,
            ocr_processor=_process_ocr_task,
            indexing_processor=_process_indexing_task,
            poll_interval=WORKER_POLL_INTERVAL,
            heartbeat_interval=HEARTBEAT_INTERVAL
        )
        queue_worker_pool.start()

        # Initialize and start scheduler
        task_scheduler = TaskScheduler(
            task_queue_manager=task_queue_manager,
            check_interval_seconds=TASK_QUEUE_CHECK_INTERVAL
        )
        task_scheduler.start()

        logger.info("✅ Task queue system initialized successfully")
    else:
        logger.info("Task queue system disabled (set TASK_QUEUE_ENABLED=true to enable)")

        # Fall back to old resume logic if queue is disabled
        def _init_services():
            # Sync existing files to database
            _sync_files_to_database()

            # Resume incomplete OCR conversion for documents with pending/converting/partial/failed status
            auto_resume_ocr = os.getenv("AUTO_RESUME_OCR", "true").lower() in ("true", "1", "yes")
            if auto_resume_ocr:
                try:
                    logger.info("Checking for incomplete OCR conversion tasks...")
                    _resume_incomplete_ocr()
                except Exception as e:
                    logger.error(f"Error resuming incomplete OCR conversion: {e}")
            else:
                logger.info("Auto-resume OCR disabled (set AUTO_RESUME_OCR=true to enable)")

            # Resume incomplete indexing for documents with pending/failed status
            auto_resume_indexing = os.getenv("AUTO_RESUME_INDEXING", "true").lower() in ("true", "1", "yes")
            if auto_resume_indexing:
                try:
                    logger.info("Checking for incomplete indexing tasks...")
                    _resume_incomplete_indexing()
                except Exception as e:
                    logger.error(f"Error resuming incomplete indexing: {e}")
            else:
                logger.info("Auto-resume indexing disabled (set AUTO_RESUME_INDEXING=true to enable)")

        import threading
        init_thread = threading.Thread(target=_init_services, daemon=True)
        init_thread.start()
        logger.info("Services initialization started in background thread")

    yield  # Application runs here

    # ===== SHUTDOWN =====
    logger.info("Application shutting down...")

    # Stop task queue system if enabled
    if TASK_QUEUE_ENABLED and task_scheduler and queue_worker_pool:
        logger.info("Stopping task queue system...")
        task_scheduler.stop()
        queue_worker_pool.stop(wait=True)
        logger.info("Task queue system stopped")


# Initialize FastAPI app with lifespan handler
app = FastAPI(
    title="Dots OCR API",
    description="API for document OCR parsing with layout detection",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)  # Authentication endpoints
app.include_router(chat_session_router)  # Chat session management endpoints
app.include_router(chat_router)  # RAG chatbot endpoints

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

# Reuse converters from DotsOCRParser to avoid duplicate model loading (CUDA OOM)
# The parser already initializes Gemma3 and Qwen3 converters during __init__
gemma3_ocr_converter = parser.gemma3_converter
qwen3_ocr_converter = parser.qwen3_converter

# Create fallback converters only if the parser's converters are None
if gemma3_ocr_converter is None:
    logger.warning("Parser's gemma3_converter is None, creating standalone Gemma3OCRConverter")
    gemma3_ocr_converter = Gemma3OCRConverter()
if qwen3_ocr_converter is None:
    logger.warning("Parser's qwen3_converter is None, creating standalone Qwen3OCRConverter")
    qwen3_ocr_converter = Qwen3OCRConverter()

# Get number of worker threads from environment (default: 4)
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))

# Task queue configuration
TASK_QUEUE_ENABLED = os.getenv("TASK_QUEUE_ENABLED", "true").lower() in ("true", "1", "yes")
TASK_QUEUE_CHECK_INTERVAL = int(os.getenv("TASK_QUEUE_CHECK_INTERVAL", "300"))  # 5 minutes
WORKER_POLL_INTERVAL = int(os.getenv("WORKER_POLL_INTERVAL", "5"))  # 5 seconds
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "30"))  # 30 seconds
STALE_TASK_TIMEOUT = int(os.getenv("STALE_TASK_TIMEOUT", "300"))  # 5 minutes

# Global task queue instances (initialized in lifespan)
task_queue_manager: Optional[TaskQueueManager] = None
task_scheduler: Optional[TaskScheduler] = None
queue_worker_pool: Optional[QueueWorkerPool] = None


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
        """Get conversion status - returns a copy to avoid race conditions"""
        with self.lock:
            conversion = self.conversions.get(conversion_id, {})
            # Return a copy to avoid race conditions with concurrent updates
            return dict(conversion) if conversion else {}

    def delete_conversion(self, conversion_id: str):
        """Delete conversion record"""
        with self.lock:
            if conversion_id in self.conversions:
                del self.conversions[conversion_id]


# WebSocket connection manager for document status updates
class DocumentStatusManager:
    """Manages centralized WebSocket connections for all document status updates with subscription support"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()  # All connected clients
        self.subscriptions: Dict[WebSocket, Set[str]] = {}  # websocket -> set of document_ids
        self.lock = threading.Lock()
        self._message_queue: asyncio.Queue = None
        self._main_loop = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the main event loop for broadcasting messages from worker threads"""
        self._main_loop = loop
        self._message_queue = asyncio.Queue()

    async def start_broadcast_worker(self):
        """Start the background worker that processes broadcast messages"""
        logger.info("📡 Document status broadcast worker started")
        while True:
            try:
                message = await self._message_queue.get()
                logger.info(f"📨 Broadcast worker received message: {message.get('event_type')} for {message.get('filename', message.get('document_id'))}")
                await self._do_broadcast(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in document status broadcast worker: {e}")

    async def connect(self, websocket: WebSocket):
        """Register a new WebSocket connection"""
        await websocket.accept()
        with self.lock:
            self.active_connections.add(websocket)
            self.subscriptions[websocket] = set()  # Initialize empty subscription set
        logger.info(f"Document status WebSocket connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        """Unregister a WebSocket connection"""
        with self.lock:
            self.active_connections.discard(websocket)
            self.subscriptions.pop(websocket, None)  # Remove subscriptions
        logger.info(f"Document status WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def subscribe(self, websocket: WebSocket, document_ids: list):
        """Subscribe a WebSocket to specific document updates"""
        with self.lock:
            if websocket not in self.subscriptions:
                self.subscriptions[websocket] = set()
            self.subscriptions[websocket].update(document_ids)
        logger.info(f"WebSocket subscribed to {len(document_ids)} documents. Total subscriptions: {len(self.subscriptions[websocket])}")

    async def unsubscribe(self, websocket: WebSocket, document_ids: list):
        """Unsubscribe a WebSocket from specific document updates"""
        with self.lock:
            if websocket in self.subscriptions:
                self.subscriptions[websocket].difference_update(document_ids)
        logger.info(f"WebSocket unsubscribed from {len(document_ids)} documents")

    async def _do_broadcast(self, message: dict):
        """Send message only to clients subscribed to the document"""
        document_id = message.get("document_id")

        with self.lock:
            connections = self.active_connections.copy()
            subscriptions = self.subscriptions.copy()

        logger.info(f"📡 Broadcasting to {len(connections)} connections, document_id={document_id}")
        logger.info(f"📋 Subscriptions: {[(id(conn), list(subs)) for conn, subs in subscriptions.items()]}")

        disconnected = []
        sent_count = 0

        for connection in connections:
            try:
                # Get subscriptions for this connection
                subscribed_docs = subscriptions.get(connection, set())

                logger.info(f"🔍 Checking connection {id(connection)}: subscribed_docs={list(subscribed_docs)}, document_id={document_id}")

                # Send if:
                # 1. Client is subscribed to this specific document, OR
                # 2. Client has no subscriptions (backward compatibility - receives all)
                if not subscribed_docs or (document_id and document_id in subscribed_docs):
                    logger.info(f"✅ Sending message to connection {id(connection)}")
                    await connection.send_json(message)
                    sent_count += 1
                else:
                    logger.info(f"⏭️ Skipping connection {id(connection)} - not subscribed to {document_id}")

            except Exception as e:
                logger.error(f"Error sending document status message: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        if disconnected:
            with self.lock:
                for conn in disconnected:
                    self.active_connections.discard(conn)
                    self.subscriptions.pop(conn, None)

        logger.info(f"📤 Broadcast sent to {sent_count}/{len(connections)} clients for document {document_id}")

    async def broadcast(self, message: dict):
        """Send message to all connected clients (async context)"""
        await self._do_broadcast(message)

    def broadcast_from_thread(self, message: dict):
        """Thread-safe method to broadcast message from worker threads"""
        if self._main_loop and self._message_queue:
            logger.info(f"📤 Broadcasting document status: {message.get('event_type')} for {message.get('filename', message.get('document_id'))}")
            asyncio.run_coroutine_threadsafe(
                self._message_queue.put(message),
                self._main_loop
            )
        else:
            logger.error(f"❌ Cannot broadcast document status - event loop not initialized! main_loop={self._main_loop is not None}, message_queue={self._message_queue is not None}")


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for progress updates"""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}  # conversion_id -> set of websockets
        self.lock = threading.Lock()
        self._message_queue: asyncio.Queue = None
        self._broadcast_task = None
        self._main_loop = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the main event loop for broadcasting messages from worker threads"""
        self._main_loop = loop
        self._message_queue = asyncio.Queue()

    async def start_broadcast_worker(self):
        """Start the background worker that processes broadcast messages"""
        while True:
            try:
                conversion_id, message = await self._message_queue.get()
                await self._do_broadcast(conversion_id, message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in broadcast worker: {e}")

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

    async def _do_broadcast(self, conversion_id: str, message: dict):
        """Actually send message to all connected clients for a conversion"""
        with self.lock:
            connections = self.active_connections.get(conversion_id, set()).copy()

        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                # Don't disconnect here to avoid async issues

    async def broadcast(self, conversion_id: str, message: dict):
        """Send message to all connected clients for a conversion (async context)"""
        await self._do_broadcast(conversion_id, message)

    def broadcast_from_thread(self, conversion_id: str, message: dict):
        """Thread-safe method to broadcast message from worker threads"""
        if self._main_loop and self._message_queue:
            # Use call_soon_threadsafe to schedule the coroutine on the main loop
            asyncio.run_coroutine_threadsafe(
                self._message_queue.put((conversion_id, message)),
                self._main_loop
            )


conversion_manager = ConversionManager()
connection_manager = ConnectionManager()
document_status_manager = DocumentStatusManager()


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
            # Use thread-safe broadcast method
            connection_manager.broadcast_from_thread(conversion_id, {
                "status": "processing",
                "progress": progress,
                "message": message,
            })
            logger.info(f"Conversion {conversion_id}: {progress}% - {message}")
        except Exception as e:
            logger.error(f"Error in parser progress callback: {str(e)}")

    return progress_callback


def _worker_progress_callback(conversion_id: str, status: str, result=None, error=None):
    """Callback for worker pool progress updates"""
    try:
        if status == "completed":
            # Check if the result indicates the file was skipped
            is_skipped = False
            skip_reason = None
            source_name = None  # For embedding trigger
            filename = None  # For database update

            if result and isinstance(result, list) and len(result) > 0:
                # Check if any result has 'skipped' flag
                first_result = result[0]
                if isinstance(first_result, dict) and first_result.get('skipped'):
                    is_skipped = True
                    skip_reason = first_result.get('skip_reason', 'Image was skipped')

                # Extract source_name for embedding (folder name under output)
                if isinstance(first_result, dict):
                    md_path = first_result.get('md_content_path') or first_result.get('md_content_nohf_path')
                    if md_path:
                        # Get the parent folder name as source_name
                        source_name = Path(md_path).parent.name
                    # Get original filename from file_path
                    file_path = first_result.get('file_path')
                    if file_path:
                        filename = Path(file_path).name

            if is_skipped:
                # Send warning status instead of completed
                conversion_manager.update_conversion(
                    conversion_id,
                    status="warning",
                    progress=100,
                    message=f"⚠️ {skip_reason}",
                    completed_at=datetime.now().isoformat(),
                )
                # Use thread-safe broadcast method
                connection_manager.broadcast_from_thread(conversion_id, {
                    "status": "warning",
                    "progress": 100,
                    "message": f"⚠️ {skip_reason}",
                    "results": result,
                })
            else:
                # Normal completion
                conversion_manager.update_conversion(
                    conversion_id,
                    status="completed",
                    progress=100,
                    message="Conversion completed successfully",
                    completed_at=datetime.now().isoformat(),
                )
                # Use thread-safe broadcast method
                connection_manager.broadcast_from_thread(conversion_id, {
                    "status": "completed",
                    "progress": 100,
                    "message": "Conversion completed successfully",
                    "results": result,
                })

                # Update database status
                if filename:
                    try:
                        with get_db_session() as db:
                            repo = DocumentRepository(db)
                            doc = repo.get_by_filename(filename)
                            if doc:
                                output_path = os.path.join(OUTPUT_DIR, source_name) if source_name else None
                                converted_pages = len(result) if result else 1
                                repo.update_convert_status(
                                    doc, ConvertStatus.CONVERTED, converted_pages, output_path,
                                    message="Conversion completed successfully"
                                )
                    except Exception as e:
                        logger.warning(f"Could not update database status: {e}")

                # Trigger embedding in background after successful conversion
                if source_name:
                    logger.info(f"Triggering embedding for document: {source_name}")

                    # Get document_id for broadcasting
                    doc_id = None
                    if filename:
                        try:
                            with get_db_session() as db:
                                repo = DocumentRepository(db)
                                doc = repo.get_by_filename(filename)
                                if doc:
                                    doc_id = str(doc.id)
                        except Exception as e:
                            logger.warning(f"Could not get document_id: {e}")

                    # Create a wrapper callback that broadcasts to both WebSocket managers
                    def dual_broadcast_callback(conv_id: str, message: dict):
                        """Broadcast to both conversion WebSocket and document status WebSocket"""
                        # Send to conversion WebSocket (for progress bars)
                        connection_manager.broadcast_from_thread(conv_id, message)

                        # Also send to document status WebSocket when indexing completes
                        if doc_id and filename:
                            status = message.get("status", "")
                            # Completion statuses: vector_indexed (Phase 1), metadata_extracted (Phase 1.5), graphrag_indexed (Phase 2)
                            if status in ["vector_indexed", "metadata_extracted", "graphrag_indexed", "index_error"]:
                                event_type = "indexing_completed" if status in ["vector_indexed", "metadata_extracted", "graphrag_indexed"] else "indexing_failed"
                                logger.info(f"📤 Broadcasting document status: {event_type} for {filename} (status={status})")
                                document_status_manager.broadcast_from_thread({
                                    "event_type": event_type,
                                    "document_id": doc_id,
                                    "filename": filename,
                                    "index_status": "indexed" if status in ["vector_indexed", "metadata_extracted", "graphrag_indexed"] else "failed",
                                    "progress": 100,
                                    "message": message.get("message", "Indexing completed"),
                                    "timestamp": datetime.now().isoformat()
                                })

                    trigger_embedding_for_document(
                        source_name,
                        OUTPUT_DIR,
                        filename=filename,
                        conversion_id=conversion_id,
                        broadcast_callback=dual_broadcast_callback
                    )

        elif status == "error":
            conversion_manager.update_conversion(
                conversion_id,
                status="error",
                progress=0,
                message=f"Conversion failed: {error}",
                error=error,
                completed_at=datetime.now().isoformat(),
            )
            # Use thread-safe broadcast method
            connection_manager.broadcast_from_thread(conversion_id, {
                "status": "error",
                "progress": 0,
                "message": f"Conversion failed: {error}",
                "error": error,
            })

            # Update database status for error
            # Try to get filename from conversion manager
            try:
                conversion_info = conversion_manager.get_conversion(conversion_id)
                if conversion_info:
                    filename = conversion_info.get('filename')
                    if filename:
                        with get_db_session() as db:
                            repo = DocumentRepository(db)
                            doc = repo.get_by_filename(filename)
                            if doc:
                                repo.update_convert_status(
                                    doc, ConvertStatus.FAILED, 0, None,
                                    message=f"Conversion failed: {error}"
                                )
            except Exception as e:
                logger.warning(f"Could not update database status for error: {e}")
    except Exception as e:
        logger.error(f"Error in progress callback: {str(e)}")


# Initialize worker pool
worker_pool = WorkerPool(num_workers=NUM_WORKERS, progress_callback=_worker_progress_callback)


def _sync_files_to_database():
    """Synchronize existing files in input directory with the database."""
    try:
        with get_db_session() as db:
            repo = DocumentRepository(db)
            synced_count = 0

            if os.path.exists(INPUT_DIR):
                for filename in os.listdir(INPUT_DIR):
                    file_path = os.path.join(INPUT_DIR, filename)

                    if not os.path.isfile(file_path):
                        continue

                    # Check if already in database
                    if repo.exists(filename):
                        continue

                    # Get file info
                    file_size = os.path.getsize(file_path)
                    file_name_without_ext = os.path.splitext(filename)[0]

                    # Check conversion status
                    markdown_exists, _, _, converted_pages = _check_markdown_exists(file_name_without_ext)
                    total_pages = _get_pdf_page_count(file_path)
                    if total_pages == 0 and markdown_exists:
                        total_pages = 1

                    # Determine convert status
                    if not markdown_exists:
                        convert_status = ConvertStatus.PENDING
                    elif total_pages > 0 and converted_pages < total_pages:
                        convert_status = ConvertStatus.PARTIAL
                    else:
                        convert_status = ConvertStatus.CONVERTED

                    # Check index status
                    indexed = is_document_indexed(file_name_without_ext) if markdown_exists else False
                    index_status = IndexStatus.INDEXED if indexed else IndexStatus.PENDING

                    # Create document record
                    doc = repo.create(
                        filename=filename,
                        original_filename=filename,
                        file_path=file_path,
                        file_size=file_size,
                        total_pages=total_pages,
                    )

                    # Update statuses
                    if convert_status != ConvertStatus.PENDING:
                        output_path = os.path.join(OUTPUT_DIR, file_name_without_ext)
                        repo.update_convert_status(doc, convert_status, converted_pages, output_path, "Synced from existing files")

                    if index_status == IndexStatus.INDEXED:
                        repo.update_index_status(doc, index_status, message="Synced from existing index")

                    synced_count += 1

            logger.info(f"Database sync complete: {synced_count} new documents synced")
    except Exception as e:
        logger.error(f"Error syncing files to database: {e}")


# ===== Task Queue Processor Functions =====

def _process_ocr_task(document_id: UUID) -> dict:
    """
    Process OCR task for a document (called by queue workers).

    This function:
    1. Checks document status to determine if resume is needed
    2. Processes document with checkpoint-based resume
    3. Updates progress in documents.ocr_details
    4. Broadcasts status updates via centralized WebSocket

    Args:
        document_id: Document UUID

    Returns:
        dict with result information
    """
    try:
        with get_db_session() as db:
            repo = DocumentRepository(db)
            doc = repo.get_by_id(document_id)

            if not doc:
                raise ValueError(f"Document not found: {document_id}")

            filename = doc.filename
            logger.info(f"🔄 Processing OCR task for document: {filename} (id={document_id})")

            # Create conversion ID for tracking
            conversion_id = conversion_manager.create_conversion(filename)

            # Update status to CONVERTING
            repo.update_convert_status(doc, ConvertStatus.CONVERTING, message="Processing OCR task from queue")

            # Store conversion_id in ocr_details
            if doc.ocr_details is None:
                doc.ocr_details = {}
            doc.ocr_details["conversion_id"] = conversion_id
            db.commit()

            # Broadcast OCR started event
            document_status_manager.broadcast_from_thread({
                "event_type": "ocr_started",
                "document_id": str(document_id),
                "filename": filename,
                "convert_status": "converting",
                "progress": 0,
                "message": "OCR processing started",
                "timestamp": datetime.now().isoformat()
            })

            # Create progress callback
            progress_callback = _create_parser_progress_callback(conversion_id)

            # Process document (this will resume from checkpoint if needed)
            result = _convert_document_background(
                filename=filename,
                prompt_mode="prompt_layout_all_en",
                conversion_id=conversion_id,
                progress_callback=progress_callback
            )

            # Update document status to CONVERTED
            repo.update_convert_status(doc, ConvertStatus.CONVERTED, message="OCR processing completed")
            db.commit()

            # Broadcast OCR completed event
            document_status_manager.broadcast_from_thread({
                "event_type": "ocr_completed",
                "document_id": str(document_id),
                "filename": filename,
                "convert_status": "converted",
                "progress": 100,
                "message": "OCR processing completed",
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"✅ OCR task completed for document: {filename}")

            # Auto-enqueue indexing task if queue system is enabled
            if TASK_QUEUE_ENABLED and task_queue_manager:
                indexing_task_id = task_queue_manager.enqueue_task(
                    document_id=document_id,
                    task_type=TaskType.INDEXING,
                    priority=TaskPriority.NORMAL,
                    db=db
                )
                if indexing_task_id:
                    logger.info(f"✅ Auto-enqueued indexing task for {filename} (task_id={indexing_task_id})")
                else:
                    logger.info(f"Indexing task already exists for {filename}")

            return {"status": "success", "filename": filename, "result": result}

    except Exception as e:
        logger.error(f"❌ OCR task failed for document {document_id}: {e}", exc_info=True)

        # Broadcast OCR failed event
        try:
            document_status_manager.broadcast_from_thread({
                "event_type": "ocr_failed",
                "document_id": str(document_id),
                "convert_status": "failed",
                "progress": 0,
                "message": f"OCR processing failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass

        raise


def _process_indexing_task(document_id: UUID) -> dict:
    """
    Process indexing task for a document (called by queue workers).

    This function:
    1. Checks document status to determine if resume is needed
    2. Processes indexing with checkpoint-based resume (vector, metadata, GraphRAG)
    3. Updates progress in documents.indexing_details
    4. Broadcasts status updates via centralized WebSocket

    Args:
        document_id: Document UUID

    Returns:
        dict with result information
    """
    try:
        with get_db_session() as db:
            repo = DocumentRepository(db)
            doc = repo.get_by_id(document_id)

            if not doc:
                raise ValueError(f"Document not found: {document_id}")

            filename = doc.filename
            logger.info(f"🔄 Processing indexing task for document: {filename} (id={document_id})")

            # Check if document is converted
            if doc.convert_status != ConvertStatus.CONVERTED:
                raise ValueError(f"Document not converted yet: {filename} (status={doc.convert_status.value})")

            # Create conversion ID for WebSocket tracking
            conversion_id = str(uuid.uuid4())

            # Update status to INDEXING
            repo.update_index_status(doc, IndexStatus.INDEXING, message="Processing indexing task from queue")

            # Store conversion_id in indexing_details
            if doc.indexing_details is None:
                doc.indexing_details = {}
            doc.indexing_details["conversion_id"] = conversion_id
            db.commit()

            # Broadcast indexing started event
            document_status_manager.broadcast_from_thread({
                "event_type": "indexing_started",
                "document_id": str(document_id),
                "filename": filename,
                "index_status": "indexing",
                "progress": 0,
                "message": "Indexing started",
                "timestamp": datetime.now().isoformat()
            })

            # Get source name (filename without extension)
            file_name_without_ext = os.path.splitext(filename)[0]

            # Create a wrapper callback that broadcasts to both WebSocket managers
            def dual_broadcast_callback(conv_id: str, message: dict):
                """Broadcast to both conversion WebSocket and document status WebSocket"""
                # Send to conversion WebSocket (for progress bars)
                connection_manager.broadcast_from_thread(conv_id, message)

                # Also send to document status WebSocket when indexing completes
                # Map the indexer's status messages to document status events
                status = message.get("status", "")

                # Completion statuses: vector_indexed (Phase 1), metadata_extracted (Phase 1.5), graphrag_indexed (Phase 2)
                if status in ["vector_indexed", "metadata_extracted", "graphrag_indexed", "index_error"]:
                    event_type = "indexing_completed" if status in ["vector_indexed", "metadata_extracted", "graphrag_indexed"] else "indexing_failed"
                    logger.info(f"📤 Broadcasting document status: {event_type} for {filename} (status={status})")
                    document_status_manager.broadcast_from_thread({
                        "event_type": event_type,
                        "document_id": str(document_id),
                        "filename": filename,
                        "index_status": "indexed" if status in ["vector_indexed", "metadata_extracted", "graphrag_indexed"] else "failed",
                        "progress": 100,
                        "message": message.get("message", "Indexing completed"),
                        "timestamp": datetime.now().isoformat()
                    })

            # Trigger indexing (this will resume from checkpoint if needed)
            # This runs in background and updates the database status when complete
            # The background thread will initialize indexing_details structure
            trigger_embedding_for_document(
                source_name=file_name_without_ext,
                output_dir=OUTPUT_DIR,
                filename=filename,
                conversion_id=conversion_id,
                broadcast_callback=dual_broadcast_callback
            )

            # Note: Status update to INDEXED is handled by trigger_embedding_for_document
            # after Phase 1 completes. We don't update it here to avoid race conditions.

            logger.info(f"✅ Indexing task enqueued for document: {filename}")
            return {"status": "success", "filename": filename}

    except Exception as e:
        logger.error(f"❌ Indexing task failed for document {document_id}: {e}", exc_info=True)

        # Broadcast indexing failed event
        try:
            document_status_manager.broadcast_from_thread({
                "event_type": "indexing_failed",
                "document_id": str(document_id),
                "index_status": "failed",
                "progress": 0,
                "message": f"Indexing failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass

        raise


def _resume_incomplete_ocr():
    """
    Resume incomplete OCR conversion tasks on startup.
    Finds documents with pending, converting, partial, or failed OCR status and triggers conversion in background.
    """
    try:
        with get_db_session() as db:
            repo = DocumentRepository(db)

            # Find all documents that need OCR conversion
            docs_to_convert = []

            for doc in repo.get_all():
                # Skip deleted documents
                if doc.deleted_at:
                    continue

                # Check if document needs OCR conversion
                needs_conversion = False

                # Check convert_status
                if doc.convert_status in [ConvertStatus.PENDING, ConvertStatus.CONVERTING, ConvertStatus.PARTIAL, ConvertStatus.FAILED]:
                    # For PARTIAL: check if there are unconverted pages
                    if doc.convert_status == ConvertStatus.PARTIAL:
                        if doc.total_pages > 0 and doc.converted_pages < doc.total_pages:
                            needs_conversion = True
                            logger.info(f"Document {doc.filename} has partial conversion ({doc.converted_pages}/{doc.total_pages} pages)")
                    else:
                        needs_conversion = True
                        logger.info(f"Document {doc.filename} needs conversion (status: {doc.convert_status.value})")

                if needs_conversion:
                    docs_to_convert.append(doc)

            if not docs_to_convert:
                logger.info("No incomplete OCR conversion tasks found")
                return

            logger.info(f"Found {len(docs_to_convert)} documents with incomplete OCR conversion")

            # Trigger conversion for each document in background
            for doc in docs_to_convert:
                try:
                    # Create conversion task and get conversion_id
                    conversion_id = conversion_manager.create_conversion(doc.filename)

                    logger.info(f"Resuming OCR conversion for: {doc.filename} (conversion_id: {conversion_id})")

                    # Update status to processing
                    conversion_manager.update_conversion(
                        conversion_id,
                        status="processing",
                        progress=0,
                        message="Resuming OCR conversion from startup...",
                        started_at=datetime.now().isoformat()
                    )

                    # Update database status to CONVERTING
                    repo.update_convert_status(doc, ConvertStatus.CONVERTING, message="Resuming conversion from startup")

                    # Create progress callback for this conversion
                    progress_callback = _create_parser_progress_callback(conversion_id)

                    # Submit task to worker pool using the standard conversion function
                    success = worker_pool.submit_task(
                        conversion_id=conversion_id,
                        func=_convert_document_background,
                        args=(doc.filename, "prompt_layout_all_en"),
                        kwargs={
                            "conversion_id": conversion_id,
                            "progress_callback": progress_callback,
                        }
                    )

                    if success:
                        logger.info(f"Triggered background OCR conversion for: {doc.filename}")
                    else:
                        logger.warning(f"Failed to submit OCR conversion task for {doc.filename} - already in progress or queue full")
                        conversion_manager.update_conversion(
                            conversion_id,
                            status="failed",
                            message="Failed to submit task - already in progress or queue full"
                        )

                except Exception as e:
                    logger.error(f"Error resuming OCR conversion for {doc.filename}: {e}")

            logger.info(f"Resume OCR conversion complete: {len(docs_to_convert)} documents queued for background conversion")

    except Exception as e:
        logger.error(f"Error in _resume_incomplete_ocr: {e}")


def _resume_incomplete_indexing():
    """
    Resume incomplete indexing tasks on startup.
    Finds documents with pending or failed indexing status and triggers indexing in background.

    Only resumes indexing for documents that:
    1. Have been successfully converted (convert_status = CONVERTED)
    2. Have incomplete indexing phases (vector/metadata/graphrag)
    3. Are not already fully indexed (index_status != INDEXED)
    """
    try:
        with get_db_session() as db:
            repo = DocumentRepository(db)

            # Find all documents that need indexing
            docs_to_index = []

            for doc in repo.get_all():
                # Skip deleted documents
                if doc.deleted_at:
                    continue

                # Skip documents that haven't been converted yet
                if not doc.convert_status or doc.convert_status == ConvertStatus.PENDING:
                    continue

                # IMPORTANT: Skip documents that are already fully indexed
                # This prevents duplicate indexing on every startup
                if doc.index_status == IndexStatus.INDEXED:
                    # Check if all phases are complete
                    indexing_details = doc.indexing_details or {}
                    vector_status = indexing_details.get("vector_indexing", {}).get("status")
                    metadata_status = indexing_details.get("metadata_extraction", {}).get("status")
                    graphrag_status = indexing_details.get("graphrag_indexing", {}).get("status")

                    # If all phases are complete, skip this document
                    # Valid completed statuses: "completed", "success", or None (for legacy/optional phases)
                    all_complete = (
                        vector_status in ["completed", "success", None] and
                        metadata_status in ["completed", "success", None] and
                        graphrag_status in ["completed", "success", None]
                    )

                    if all_complete:
                        logger.debug(f"Document {doc.filename} is already fully indexed, skipping")
                        continue
                    else:
                        logger.debug(f"Document {doc.filename} has index_status=INDEXED but incomplete phases: vector={vector_status}, metadata={metadata_status}, graphrag={graphrag_status}")

                # Check if document needs any indexing phase
                needs_indexing = False
                indexing_details = doc.indexing_details or {}

                # Check vector indexing status (only if failed or explicitly pending)
                vector_status = indexing_details.get("vector_indexing", {}).get("status")
                if vector_status in ["pending", "failed"]:
                    needs_indexing = True
                    logger.info(f"Document {doc.filename} needs vector indexing (status: {vector_status})")

                # Check metadata extraction status (only if failed or explicitly pending)
                metadata_status = indexing_details.get("metadata_extraction", {}).get("status")
                if metadata_status in ["pending", "failed"]:
                    needs_indexing = True
                    logger.info(f"Document {doc.filename} needs metadata extraction (status: {metadata_status})")

                # Check GraphRAG indexing status (only if failed or explicitly pending)
                graphrag_status = indexing_details.get("graphrag_indexing", {}).get("status")
                if graphrag_status in ["pending", "failed"]:
                    needs_indexing = True
                    logger.info(f"Document {doc.filename} needs GraphRAG indexing (status: {graphrag_status})")

                # If no indexing_details at all, check overall index_status
                # Only resume if status is PENDING or FAILED (not PARTIAL or INDEXING)
                if not indexing_details and doc.index_status in [IndexStatus.PENDING, IndexStatus.FAILED]:
                    needs_indexing = True
                    logger.info(f"Document {doc.filename} has no indexing details (overall status: {doc.index_status})")

                if needs_indexing:
                    docs_to_index.append(doc)

            if not docs_to_index:
                logger.info("No incomplete indexing tasks found")
                return

            logger.info(f"Found {len(docs_to_index)} documents with incomplete indexing")

            # Trigger indexing for each document in background
            for doc in docs_to_index:
                try:
                    file_name_without_ext = os.path.splitext(doc.filename)[0]

                    # Create conversion task and get conversion_id
                    conversion_id = conversion_manager.create_conversion(doc.filename)

                    logger.info(f"Resuming indexing for: {doc.filename} (conversion_id: {conversion_id})")

                    # Update status to indexing
                    conversion_manager.update_conversion(
                        conversion_id,
                        status="indexing",
                        progress=0,
                        message="Resuming indexing from startup...",
                        started_at=datetime.now().isoformat()
                    )

                    # Create a wrapper callback that broadcasts to both WebSocket managers
                    doc_id = str(doc.id)
                    doc_filename = doc.filename

                    def dual_broadcast_callback(conv_id: str, message: dict):
                        """Broadcast to both conversion WebSocket and document status WebSocket"""
                        # Send to conversion WebSocket (for progress bars)
                        connection_manager.broadcast_from_thread(conv_id, message)

                        # Also send to document status WebSocket when indexing completes
                        status = message.get("status", "")
                        # Completion statuses: vector_indexed (Phase 1), metadata_extracted (Phase 1.5), graphrag_indexed (Phase 2)
                        if status in ["vector_indexed", "metadata_extracted", "graphrag_indexed", "index_error"]:
                            event_type = "indexing_completed" if status in ["vector_indexed", "metadata_extracted", "graphrag_indexed"] else "indexing_failed"
                            logger.info(f"📤 Broadcasting document status: {event_type} for {doc_filename} (status={status})")
                            document_status_manager.broadcast_from_thread({
                                "event_type": event_type,
                                "document_id": doc_id,
                                "filename": doc_filename,
                                "index_status": "indexed" if status in ["vector_indexed", "metadata_extracted", "graphrag_indexed"] else "failed",
                                "progress": 100,
                                "message": message.get("message", "Indexing completed"),
                                "timestamp": datetime.now().isoformat()
                            })

                    # Trigger indexing in background (non-blocking)
                    # This will handle all three phases: vector, metadata, GraphRAG
                    trigger_embedding_for_document(
                        source_name=file_name_without_ext,
                        output_dir=OUTPUT_DIR,
                        filename=doc.filename,
                        conversion_id=conversion_id,
                        broadcast_callback=dual_broadcast_callback
                    )

                    logger.info(f"Triggered background indexing for: {doc.filename}")

                except Exception as e:
                    logger.error(f"Error resuming indexing for {doc.filename}: {e}")

            logger.info(f"Resume indexing complete: {len(docs_to_index)} documents queued for background indexing")

    except Exception as e:
        logger.error(f"Error in _resume_incomplete_indexing: {e}")


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


@app.get("/queue/stats")
async def get_queue_stats():
    """
    Get task queue statistics.

    Returns:
        Queue statistics including pending, claimed, completed, and failed tasks
    """
    if not task_queue_manager:
        raise HTTPException(status_code=503, detail="Task queue system not initialized")

    stats = task_queue_manager.get_queue_stats()
    return stats


@app.post("/queue/maintenance")
async def trigger_queue_maintenance():
    """
    Manually trigger queue maintenance.

    This will:
    1. Release stale tasks (workers that died)
    2. Find and enqueue orphaned documents
    3. Return updated queue statistics
    """
    if not task_scheduler:
        raise HTTPException(status_code=503, detail="Task scheduler not initialized")

    # Run maintenance in background thread to avoid blocking
    def run_maintenance():
        task_scheduler._periodic_maintenance()

    thread = threading.Thread(target=run_maintenance, daemon=True)
    thread.start()

    return {
        "status": "success",
        "message": "Queue maintenance triggered"
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


def _resize_image_if_needed(file_path: str, max_pixels: int = None) -> dict:
    """
    Resize image if it exceeds maximum pixel dimensions.

    Args:
        file_path: Path to the image file
        max_pixels: Maximum number of pixels allowed. If None, reads from MAX_PIXELS env var.
                   Default fallback is 8000000 (safer for vLLM servers with limited memory)

    Returns:
        dict with keys:
            - resized: bool indicating if image was resized
            - original_size: tuple of (width, height) before resize
            - new_size: tuple of (width, height) after resize (None if not resized)
            - message: str describing what happened
    """
    from PIL import Image
    import math

    # Get max_pixels from environment variable if not specified
    if max_pixels is None:
        max_pixels_env = os.getenv('MAX_PIXELS', '8000000')
        if max_pixels_env.lower() == 'none':
            max_pixels = 11289600  # Use absolute maximum if env is None
        else:
            max_pixels = int(max_pixels_env)

    # Apply a safety margin (90% of max_pixels) to prevent edge cases
    # This ensures the image is comfortably within limits after base64 encoding
    safe_max_pixels = int(max_pixels * 0.9)

    # Check if file is an image
    file_ext = os.path.splitext(file_path)[1].lower()
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}

    if file_ext not in image_extensions:
        return {
            "resized": False,
            "original_size": None,
            "new_size": None,
            "message": "Not an image file"
        }

    try:
        # Open image and get dimensions
        with Image.open(file_path) as img:
            original_width, original_height = img.size
            original_pixels = original_width * original_height

            # Check if resize is needed
            if original_pixels <= safe_max_pixels:
                return {
                    "resized": False,
                    "original_size": (original_width, original_height),
                    "new_size": None,
                    "message": f"Image size OK ({original_width}x{original_height} = {original_pixels:,} pixels, limit: {safe_max_pixels:,})"
                }

            # Calculate new dimensions maintaining aspect ratio
            scale_factor = math.sqrt(safe_max_pixels / original_pixels)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            # Ensure dimensions are divisible by 28 (IMAGE_FACTOR) for better compatibility
            new_width = (new_width // 28) * 28
            new_height = (new_height // 28) * 28

            # Ensure minimum size
            if new_width < 28:
                new_width = 28
            if new_height < 28:
                new_height = 28

            # Convert to RGB if necessary (for PNG with transparency, etc.)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize image using high-quality LANCZOS resampling
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save resized image (overwrite original)
            # Use high quality settings but optimize for size
            if file_ext in {'.jpg', '.jpeg'}:
                resized_img.save(file_path, 'JPEG', quality=90, optimize=True)
            elif file_ext == '.png':
                # For PNG, convert to JPEG to reduce file size significantly
                # This helps with base64 encoding and network transfer
                new_file_path = os.path.splitext(file_path)[0] + '.jpg'
                resized_img.save(new_file_path, 'JPEG', quality=90, optimize=True)
                # Remove original PNG and rename JPG to original name
                if new_file_path != file_path:
                    os.remove(file_path)
                    os.rename(new_file_path, file_path)
            else:
                # For other formats, save as JPEG
                resized_img.save(file_path, 'JPEG', quality=90, optimize=True)

            new_pixels = new_width * new_height
            return {
                "resized": True,
                "original_size": (original_width, original_height),
                "new_size": (new_width, new_height),
                "message": f"Image resized from {original_width}x{original_height} ({original_pixels:,} pixels) to {new_width}x{new_height} ({new_pixels:,} pixels, limit: {safe_max_pixels:,})"
            }

    except Exception as e:
        logger.error(f"Error resizing image {file_path}: {e}")
        return {
            "resized": False,
            "original_size": None,
            "new_size": None,
            "message": f"Error resizing image: {str(e)}"
        }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a document file (PDF, image, DOC, EXCEL) to the input folder.
    Automatically resizes images that are too large to prevent OCR errors.

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

        # Resize image if needed
        resize_info = _resize_image_if_needed(file_path)

        file_size = os.path.getsize(file_path)
        upload_time = datetime.now().isoformat()

        # Create database record
        doc_id = None
        try:
            with get_db_session() as db:
                repo = DocumentRepository(db)
                doc, created = repo.get_or_create(
                    filename=file.filename,
                    original_filename=file.filename,
                    file_path=file_path,
                    file_size=file_size,
                )
                doc_id = str(doc.id)
                if not created:
                    logger.info(f"Document record already exists: {file.filename}")

                # Auto-enqueue OCR task if queue system is enabled
                if TASK_QUEUE_ENABLED and task_queue_manager:
                    task_id = task_queue_manager.enqueue_task(
                        document_id=doc.id,
                        task_type=TaskType.OCR,
                        priority=TaskPriority.HIGH,  # User uploads get high priority
                        db=db
                    )
                    if task_id:
                        logger.info(f"✅ Auto-enqueued OCR task for {file.filename} (task_id={task_id})")
                    else:
                        logger.info(f"OCR task already exists for {file.filename}")
        except Exception as e:
            logger.warning(f"Could not create database record: {e}")

        response_data = {
            "status": "success",
            "filename": file.filename,
            "file_path": file_path,
            "file_size": file_size,
            "upload_time": upload_time,
            "document_id": doc_id,
        }

        # Add resize information if image was resized
        if resize_info["resized"]:
            response_data["resized"] = True
            response_data["original_size"] = resize_info["original_size"]
            response_data["new_size"] = resize_info["new_size"]
            response_data["resize_message"] = resize_info["message"]
            logger.info(f"📏 Image resized: {resize_info['message']}")
            logger.info(f"   Original: {resize_info['original_size'][0]}x{resize_info['original_size'][1]} = {resize_info['original_size'][0] * resize_info['original_size'][1]:,} pixels")
            logger.info(f"   New: {resize_info['new_size'][0]}x{resize_info['new_size'][1]} = {resize_info['new_size'][0] * resize_info['new_size'][1]:,} pixels")
            logger.info(f"   File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
        elif resize_info['original_size'] is not None:
            # Image file that didn't need resizing
            orig_w, orig_h = resize_info['original_size']
            logger.info(f"📏 Image size OK: {orig_w}x{orig_h} = {orig_w * orig_h:,} pixels")
            logger.info(f"   File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
        else:
            # Non-image file (PDF, Excel, Word, etc.)
            logger.info(f"📄 File uploaded: {file.filename}")
            logger.info(f"   File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")

        return JSONResponse(content=response_data)

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
    - (markdown_exists: bool, markdown_path: str or None, is_multipage: bool, converted_pages: int)
    """
    output_dir = os.path.join(OUTPUT_DIR, file_name_without_ext)

    # Check for single markdown file (for single-page documents, images, Word, Excel, TXT)
    markdown_path_nohf = os.path.join(output_dir, f"{file_name_without_ext}_nohf.md")
    if os.path.exists(markdown_path_nohf):
        return True, markdown_path_nohf, False, 1

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
            return True, os.path.join(output_dir, page_files[0]), True, len(page_files)

    return False, None, False, 0


def _get_pdf_page_count(file_path: str) -> int:
    """
    Get the total number of pages in a PDF file.

    Returns:
    - Total page count, or 0 if not a PDF or error occurs
    """
    try:
        if not file_path.lower().endswith('.pdf'):
            return 0

        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        page_count = len(doc)
        doc.close()
        return page_count
    except Exception as e:
        logger.error(f"Error getting PDF page count for {file_path}: {str(e)}")
        return 0


@app.get("/documents")
async def list_documents():
    """
    List all uploaded documents with their conversion status.

    Returns:
    - JSON with list of documents and their markdown conversion status
    - Includes total_pages (for PDFs) and converted_pages to track partial conversions
    - Includes database info (document_id, upload_status, convert_status, index_status)
    """
    try:
        documents = []

        # Get database records for faster lookup
        db_docs = {}
        try:
            with get_db_session() as db:
                repo = DocumentRepository(db)
                for doc in repo.get_all():
                    db_docs[doc.filename] = doc.to_dict()
        except Exception as e:
            logger.warning(f"Could not fetch database records: {e}")

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
                markdown_exists, markdown_path, is_multipage, converted_pages = _check_markdown_exists(file_name_without_ext)

                # Get total page count for PDFs
                total_pages = _get_pdf_page_count(file_path)

                # For non-PDF files, set total_pages to 1 if converted, 0 otherwise
                if total_pages == 0 and markdown_exists:
                    total_pages = 1

                # Check if document is indexed in vector database
                indexed = is_document_indexed(file_name_without_ext) if markdown_exists else False

                # Get database info if available
                db_info = db_docs.get(filename, {})

                documents.append({
                    "filename": filename,
                    "file_path": file_path,
                    "file_size": file_size,
                    "upload_time": db_info.get("created_at") or upload_time,
                    "markdown_exists": markdown_exists,
                    "markdown_path": markdown_path if markdown_exists else None,
                    "is_multipage": is_multipage,
                    "total_pages": total_pages,
                    "converted_pages": converted_pages,
                    "indexed": indexed,
                    # Database fields
                    "document_id": db_info.get("id"),
                    "upload_status": db_info.get("upload_status", "uploaded" if os.path.exists(file_path) else "pending"),
                    "convert_status": db_info.get("convert_status", "converted" if markdown_exists else "pending"),
                    "index_status": db_info.get("index_status", "indexed" if indexed else "pending"),
                    "indexed_chunks": db_info.get("indexed_chunks", 0),
                    "indexing_details": db_info.get("indexing_details"),  # Include granular indexing status
                    "ocr_details": db_info.get("ocr_details"),  # Include granular OCR status
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


@app.post("/retry-ocr")
async def retry_failed_ocr(
    filename: str = Form(...),
    retry_type: str = Form("all"),  # "all", "pages", "images"
    page_numbers: Optional[str] = Form(None)  # Comma-separated page numbers
):
    """
    Retry OCR for failed pages or embedded images only.

    This endpoint allows selective re-processing of only the failed components
    without redoing the entire document conversion.

    Parameters:
    - filename: Document filename
    - retry_type: Type of retry - "all" (retry all failures), "pages" (retry failed pages only),
                  "images" (retry failed embedded images only)
    - page_numbers: Optional comma-separated list of specific page numbers to retry (e.g., "0,5,10")

    Returns:
    - JSON with OCR summary and retry plan
    """
    try:
        # Parse page numbers if provided
        specific_pages = None
        if page_numbers:
            try:
                specific_pages = [int(p.strip()) for p in page_numbers.split(",")]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid page_numbers format. Use comma-separated integers.")

        with get_db_session() as db:
            repo = DocumentRepository(db)
            doc = repo.get_by_filename(filename)

            if not doc:
                raise HTTPException(status_code=404, detail=f"Document not found: {filename}")

            # Get OCR summary
            ocr_summary = repo.get_ocr_summary(doc)

            # Determine what needs to be retried
            retry_plan = {
                "filename": filename,
                "retry_type": retry_type,
                "ocr_summary": ocr_summary,
                "pages_to_retry": [],
                "images_to_retry": {}
            }

            if retry_type in ["pages", "all"]:
                failed_pages = repo.get_failed_pages(doc)
                if specific_pages:
                    failed_pages = [p for p in failed_pages if p in specific_pages]
                retry_plan["pages_to_retry"] = failed_pages

            if retry_type in ["images", "all"]:
                failed_images = repo.get_pages_with_failed_embedded_images(doc)
                if specific_pages:
                    failed_images = {p: imgs for p, imgs in failed_images.items() if p in specific_pages}
                retry_plan["images_to_retry"] = failed_images

            # Check if there's anything to retry
            if not retry_plan["pages_to_retry"] and not retry_plan["images_to_retry"]:
                return JSONResponse(content={
                    "status": "nothing_to_retry",
                    "message": "No failed pages or images found to retry",
                    "retry_plan": retry_plan
                })

            # TODO: Implement actual retry logic
            # For now, just return the retry plan
            return JSONResponse(content={
                "status": "retry_planned",
                "message": f"Retry plan created. Found {len(retry_plan['pages_to_retry'])} failed pages and "
                          f"{sum(len(imgs) for imgs in retry_plan['images_to_retry'].values())} failed images.",
                "retry_plan": retry_plan,
                "note": "Actual retry implementation is pending. This endpoint currently only analyzes what needs to be retried."
            })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in retry-ocr endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing retry request: {str(e)}")


@app.get("/ocr-status/{filename}")
async def get_ocr_status(filename: str):
    """
    Get detailed OCR status for a document.

    Returns granular OCR status including page-level and embedded image-level tracking.

    Parameters:
    - filename: Document filename

    Returns:
    - JSON with detailed OCR status
    """
    try:
        with get_db_session() as db:
            repo = DocumentRepository(db)
            doc = repo.get_by_filename(filename)

            if not doc:
                raise HTTPException(status_code=404, detail=f"Document not found: {filename}")

            ocr_summary = repo.get_ocr_summary(doc)

            return JSONResponse(content={
                "filename": filename,
                "ocr_summary": ocr_summary,
                "ocr_details": doc.ocr_details
            })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ocr-status endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting OCR status: {str(e)}")


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


@app.websocket("/ws/document-status")
async def websocket_document_status(websocket: WebSocket):
    """
    Centralized WebSocket endpoint for all document status updates with subscription support.

    Clients can subscribe to specific documents to receive only relevant updates.
    This eliminates the need for polling and reduces unnecessary network traffic.

    Client -> Server messages:
    - {"action": "subscribe", "document_ids": ["uuid1", "uuid2", ...]}
    - {"action": "unsubscribe", "document_ids": ["uuid1", "uuid2", ...]}
    - {"action": "ping"} (keepalive)

    Server -> Client messages:
    - document_id: UUID of the document
    - filename: Name of the document
    - event_type: "ocr_started", "ocr_progress", "ocr_completed", "ocr_failed",
                  "indexing_started", "indexing_progress", "indexing_completed", "indexing_failed"
    - convert_status: Current OCR status
    - index_status: Current indexing status
    - progress: Progress percentage (0-100)
    - message: Status message
    - timestamp: ISO timestamp
    """
    try:
        # Connect the WebSocket
        await document_status_manager.connect(websocket)

        # Send initial connection confirmation
        await websocket.send_json({
            "event_type": "connected",
            "message": "Connected to document status updates",
            "timestamp": datetime.now().isoformat()
        })

        # Keep connection alive and listen for client messages
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                action = message.get("action")

                if action == "subscribe":
                    # Subscribe to specific documents
                    document_ids = message.get("document_ids", [])
                    if document_ids:
                        await document_status_manager.subscribe(websocket, document_ids)
                        await websocket.send_json({
                            "event_type": "subscribed",
                            "document_ids": document_ids,
                            "count": len(document_ids),
                            "timestamp": datetime.now().isoformat()
                        })

                elif action == "unsubscribe":
                    # Unsubscribe from specific documents
                    document_ids = message.get("document_ids", [])
                    if document_ids:
                        await document_status_manager.unsubscribe(websocket, document_ids)
                        await websocket.send_json({
                            "event_type": "unsubscribed",
                            "document_ids": document_ids,
                            "count": len(document_ids),
                            "timestamp": datetime.now().isoformat()
                        })

                elif action == "ping":
                    # Respond to keepalive ping
                    await websocket.send_json({
                        "event_type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })

                else:
                    logger.warning(f"Unknown WebSocket action: {action}")

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received from WebSocket: {data}")
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")

    except WebSocketDisconnect:
        await document_status_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Document status WebSocket error: {e}")
        await document_status_manager.disconnect(websocket)


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

        # Trigger re-indexing in background after markdown update
        logger.info(f"Triggering re-indexing for document after markdown update: {filename}")
        reindex_document(filename, OUTPUT_DIR)

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


@app.get("/documents/in-progress")
async def get_in_progress_documents():
    """
    Get all documents that are currently being processed (converting or indexing).

    This endpoint is used by the frontend to:
    1. Check for in-progress operations on page load/refresh
    2. Resume polling/monitoring for those operations

    Returns:
    - JSON with list of documents that have:
      - convert_status = 'converting' OR
      - index_status = 'indexing'
    """
    try:
        in_progress_docs = []

        with get_db_session() as db:
            repo = DocumentRepository(db)
            all_docs = repo.get_all()

            for doc in all_docs:
                # Check if document is in progress (converting or indexing)
                is_converting = doc.convert_status == ConvertStatus.CONVERTING
                is_indexing = doc.index_status == IndexStatus.INDEXING

                # Also check granular indexing details for background processing
                is_granular_indexing = False
                all_phases_complete = False
                if doc.indexing_details:
                    vector_status = doc.indexing_details.get("vector_indexing", {}).get("status")
                    metadata_status = doc.indexing_details.get("metadata_extraction", {}).get("status")
                    graphrag_status = doc.indexing_details.get("graphrag_indexing", {}).get("status")

                    # Consider "processing" or "pending" as in-progress
                    is_granular_indexing = (
                        vector_status in ["processing", "pending"] or
                        metadata_status in ["processing", "pending"] or
                        graphrag_status in ["processing", "pending"]
                    )

                    # Check if all phases are complete
                    all_phases_complete = (
                        vector_status == "completed" and
                        metadata_status == "completed" and
                        graphrag_status == "completed"
                    )

                # Fix data inconsistency: if all phases are complete but index_status is still "indexing",
                # update it to "indexed"
                if all_phases_complete and is_indexing:
                    try:
                        indexed_chunks = doc.indexed_chunks or 0
                        repo.update_index_status(
                            doc, IndexStatus.INDEXED, indexed_chunks,
                            message="All indexing phases completed"
                        )
                        logger.info(f"Fixed index_status for {doc.filename}: all phases complete, updated to INDEXED")
                        is_indexing = False  # No longer in progress
                    except Exception as e:
                        logger.warning(f"Could not fix index_status for {doc.filename}: {e}")

                if is_converting or is_indexing or is_granular_indexing:
                    doc_dict = doc.to_dict()

                    # Add additional computed fields for frontend
                    doc_dict["is_converting"] = is_converting
                    doc_dict["is_indexing"] = is_indexing or is_granular_indexing

                    in_progress_docs.append(doc_dict)

        logger.info(f"Found {len(in_progress_docs)} in-progress documents")

        return JSONResponse(content={
            "status": "success",
            "documents": in_progress_docs,
            "total": len(in_progress_docs),
        })

    except Exception as e:
        logger.error(f"Error getting in-progress documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting in-progress documents: {str(e)}")


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

        # Delete vector embeddings from Qdrant (all collections)
        try:
            delete_documents_by_source(file_name_without_ext)
            logger.info(f"Deleted vector embeddings for: {file_name_without_ext}")
        except Exception as e:
            error_msg = f"Failed to delete vector embeddings: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)

        # Delete GraphRAG data (Neo4j, PostgreSQL GraphRAG tables, Qdrant entity/edge vectors)
        if GRAPHRAG_DELETE_AVAILABLE and GRAPH_RAG_INDEX_ENABLED:
            try:
                delete_graphrag_by_source_sync(file_name_without_ext)
                logger.info(f"Deleted GraphRAG data for: {file_name_without_ext}")
            except Exception as e:
                error_msg = f"Failed to delete GraphRAG data: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)

        # Hard delete from database (removes document and status logs via cascade)
        try:
            with get_db_session() as db:
                repo = DocumentRepository(db)
                doc = repo.get_by_filename(filename)
                if doc:
                    repo.hard_delete(doc)
                    logger.info(f"Hard deleted database record for: {filename}")
        except Exception as e:
            error_msg = f"Failed to delete database record: {str(e)}"
            errors.append(error_msg)
            logger.warning(error_msg)

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


# ===== Indexing Status Tracking =====
# Persistent status tracker for batch indexing (survives page refresh)
_batch_index_status = {
    "status": "idle",  # idle, running, completed, error
    "total_documents": 0,
    "indexed_documents": 0,
    "current_document": None,
    "started_at": None,
    "completed_at": None,
    "errors": [],
    "message": None,
}
_batch_index_lock = threading.Lock()


@app.post("/documents/{filename}/index")
async def index_single_document(filename: str):
    """
    Index a single document's markdown files into the vector database.
    Re-indexes if already indexed.

    Now supports WebSocket progress updates for real-time status tracking.

    Parameters:
    - filename: The name of the file to index (with extension)

    Returns:
    - Accepted status with conversion_id for WebSocket tracking
    """
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Validate filename to prevent directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        file_name_without_ext = os.path.splitext(filename)[0]
        doc_dir = os.path.join(OUTPUT_DIR, file_name_without_ext)

        # Check if document directory exists
        if not os.path.exists(doc_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Document output folder not found: {file_name_without_ext}"
            )

        # Register the indexing task in conversion manager
        # create_conversion() generates and returns the conversion_id
        conversion_id = conversion_manager.create_conversion(filename)

        # Update status to indexing
        conversion_manager.update_conversion(
            conversion_id,
            status="indexing",
            progress=0,
            message="Starting indexing process..."
        )

        # Delete existing embeddings and re-index
        try:
            delete_documents_by_source(file_name_without_ext)
            logger.info(f"Deleted existing embeddings for: {file_name_without_ext}")
        except Exception as e:
            logger.warning(f"Error deleting existing embeddings (may not exist): {e}")

        # Broadcast initial status
        await connection_manager.broadcast(conversion_id, {
            "status": "indexing",
            "progress": 10,
            "message": "Preparing to index document...",
        })

        # Get document_id for broadcasting
        doc_id = None
        try:
            with get_db_session() as db:
                repo = DocumentRepository(db)
                doc = repo.get_by_filename(filename)
                if doc:
                    doc_id = str(doc.id)
        except Exception as e:
            logger.warning(f"Could not get document_id: {e}")

        # Create a wrapper callback that broadcasts to both WebSocket managers
        def dual_broadcast_callback(conv_id: str, message: dict):
            """Broadcast to both conversion WebSocket and document status WebSocket"""
            # Send to conversion WebSocket (for progress bars)
            connection_manager.broadcast_from_thread(conv_id, message)

            # Also send to document status WebSocket when indexing completes
            if doc_id:
                status = message.get("status", "")
                # Completion statuses: vector_indexed (Phase 1), metadata_extracted (Phase 1.5), graphrag_indexed (Phase 2)
                if status in ["vector_indexed", "metadata_extracted", "graphrag_indexed", "index_error"]:
                    event_type = "indexing_completed" if status in ["vector_indexed", "metadata_extracted", "graphrag_indexed"] else "indexing_failed"
                    logger.info(f"📤 Broadcasting document status: {event_type} for {filename} (status={status})")
                    document_status_manager.broadcast_from_thread({
                        "event_type": event_type,
                        "document_id": doc_id,
                        "filename": filename,
                        "index_status": "indexed" if status in ["vector_indexed", "metadata_extracted", "graphrag_indexed"] else "failed",
                        "progress": 100,
                        "message": message.get("message", "Indexing completed"),
                        "timestamp": datetime.now().isoformat()
                    })

        # Trigger two-phase indexing with WebSocket support
        # This runs in background and sends progress updates via WebSocket
        trigger_embedding_for_document(
            source_name=file_name_without_ext,
            output_dir=OUTPUT_DIR,
            filename=filename,
            conversion_id=conversion_id,
            broadcast_callback=dual_broadcast_callback
        )

        return JSONResponse(content={
            "status": "accepted",
            "conversion_id": conversion_id,
            "filename": filename,
            "source_name": file_name_without_ext,
            "message": "Indexing started in background. Connect to WebSocket for progress updates.",
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting indexing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting indexing: {str(e)}")


@app.get("/documents/{filename}/status-logs")
async def get_document_status_logs(filename: str):
    """
    Get status change logs for a document (audit trail).

    Parameters:
    - filename: The name of the file

    Returns:
    - List of status change logs with timestamps
    """
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        with get_db_session() as db:
            repo = DocumentRepository(db)
            doc = repo.get_by_filename(filename)
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document not found: {filename}")

            logs = repo.get_status_logs(doc.id)
            return JSONResponse(content={
                "status": "success",
                "filename": filename,
                "document_id": str(doc.id),
                "logs": [log.to_dict() for log in logs],
                "total": len(logs),
            })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting status logs: {str(e)}")


@app.get("/documents/{filename}/indexing-status")
async def get_document_indexing_status(filename: str):
    """
    Get detailed indexing status for all phases (vector, metadata, GraphRAG).

    Parameters:
    - filename: The name of the file

    Returns:
    - Detailed indexing progress and status for each phase
    """
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        with get_db_session() as db:
            repo = DocumentRepository(db)
            doc = repo.get_by_filename(filename)
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document not found: {filename}")

            # Get indexing progress
            progress = repo.get_indexing_progress(doc)

            return JSONResponse(content={
                "status": "success",
                "filename": filename,
                "document_id": str(doc.id),
                "overall_status": doc.index_status.value if doc.index_status else "pending",
                "indexing_progress": progress,
                "indexing_details": doc.indexing_details,
            })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting indexing status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting indexing status: {str(e)}")


@app.post("/documents/{filename}/reindex-failed")
async def reindex_failed_chunks(
    filename: str,
    phases: str = Query("vector,graphrag,metadata", description="Comma-separated phases to reindex")
):
    """
    Re-index only failed chunks/pages for specified phases.

    This endpoint implements selective re-indexing:
    - Vector failures: Re-index failed pages (re-read from markdown)
    - GraphRAG failures: Re-index failed chunks (retrieve from Qdrant)
    - Metadata failures: Re-extract metadata (use all chunks)

    Parameters:
    - filename: The name of the file
    - phases: Comma-separated list of phases (vector, graphrag, metadata)

    Returns:
    - Re-indexing results for each phase
    """
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Validate filename
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        with get_db_session() as db:
            repo = DocumentRepository(db)
            doc = repo.get_by_filename(filename)
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document not found: {filename}")

        file_name_without_ext = os.path.splitext(filename)[0]
        phases_list = [p.strip() for p in phases.split(",")]

        results = {}

        # Import selective re-indexing functions
        from rag_service.selective_reindexer import (
            reindex_failed_vector_pages,
            reindex_failed_graphrag_chunks,
            reindex_metadata
        )

        # Re-index vector failures (page-level)
        if "vector" in phases_list:
            logger.info(f"[Selective Re-index] Starting vector re-indexing for {filename}")
            with get_db_session() as db:
                repo = DocumentRepository(db)
                doc = repo.get_by_filename(filename)
                if doc:
                    results["vector"] = reindex_failed_vector_pages(doc, file_name_without_ext, OUTPUT_DIR)

        # Re-index GraphRAG failures (chunk-level)
        if "graphrag" in phases_list:
            logger.info(f"[Selective Re-index] Starting GraphRAG re-indexing for {filename}")
            with get_db_session() as db:
                repo = DocumentRepository(db)
                doc = repo.get_by_filename(filename)
                if doc:
                    results["graphrag"] = reindex_failed_graphrag_chunks(doc, file_name_without_ext, OUTPUT_DIR)

        # Re-extract metadata
        if "metadata" in phases_list:
            logger.info(f"[Selective Re-index] Starting metadata re-extraction for {filename}")
            with get_db_session() as db:
                repo = DocumentRepository(db)
                doc = repo.get_by_filename(filename)
                if doc:
                    results["metadata"] = reindex_metadata(doc, file_name_without_ext, OUTPUT_DIR)

        return JSONResponse(content={
            "status": "success",
            "message": "Selective re-indexing completed",
            "filename": filename,
            "phases": phases_list,
            "results": results
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in selective re-indexing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in selective re-indexing: {str(e)}")


@app.post("/documents/index-all")
async def index_all_documents():
    """
    Start batch indexing of all documents in the background.
    Clears all existing embeddings and re-indexes all documents.

    Returns:
    - Accepted status with message
    """
    global _batch_index_status

    with _batch_index_lock:
        if _batch_index_status["status"] == "running":
            return JSONResponse(
                status_code=409,
                content={
                    "status": "conflict",
                    "message": "Batch indexing is already in progress",
                    "current_status": _batch_index_status,
                }
            )

    def _batch_index_task():
        global _batch_index_status

        try:
            with _batch_index_lock:
                _batch_index_status = {
                    "status": "running",
                    "total_documents": 0,
                    "indexed_documents": 0,
                    "current_index": 0,  # 0-based index of document currently being processed
                    "current_document": None,
                    "started_at": datetime.now().isoformat(),
                    "completed_at": None,
                    "errors": [],
                    "message": "Initializing batch indexing...",
                }

            # Clear all existing embeddings
            logger.info("Clearing all vector embeddings for re-indexing...")
            with _batch_index_lock:
                _batch_index_status["message"] = "Clearing existing embeddings..."

            try:
                clear_collection()
                logger.info("Cleared all vector embeddings")
            except Exception as e:
                logger.error(f"Error clearing collection: {e}")

            # Find all document directories
            output_path = Path(OUTPUT_DIR)
            doc_dirs = [d for d in output_path.iterdir() if d.is_dir()]

            with _batch_index_lock:
                _batch_index_status["total_documents"] = len(doc_dirs)
                _batch_index_status["message"] = f"Found {len(doc_dirs)} documents to index"

            logger.info(f"Found {len(doc_dirs)} documents to index")

            total_chunks = 0
            for i, doc_dir in enumerate(doc_dirs):
                source_name = doc_dir.name

                with _batch_index_lock:
                    _batch_index_status["current_document"] = source_name
                    _batch_index_status["current_index"] = i  # 0-based index of current document
                    _batch_index_status["message"] = f"Indexing: {source_name} ({i+1}/{len(doc_dirs)})"

                try:
                    chunks = index_document_now(source_name, OUTPUT_DIR)
                    total_chunks += chunks
                    logger.info(f"Indexed {source_name}: {chunks} chunks")

                    # Update to reflect completed count after successful indexing
                    with _batch_index_lock:
                        _batch_index_status["indexed_documents"] = i + 1

                except Exception as e:
                    error_msg = f"Error indexing {source_name}: {str(e)}"
                    logger.error(error_msg)
                    with _batch_index_lock:
                        _batch_index_status["errors"].append(error_msg)
                        # Still move forward even on error
                        _batch_index_status["indexed_documents"] = i + 1

            with _batch_index_lock:
                _batch_index_status["status"] = "completed"
                _batch_index_status["completed_at"] = datetime.now().isoformat()
                _batch_index_status["current_document"] = None
                _batch_index_status["message"] = f"Completed: indexed {total_chunks} chunks from {len(doc_dirs)} documents"

            logger.info(f"Batch indexing completed: {total_chunks} chunks from {len(doc_dirs)} documents")

        except Exception as e:
            logger.error(f"Batch indexing error: {str(e)}")
            with _batch_index_lock:
                _batch_index_status["status"] = "error"
                _batch_index_status["completed_at"] = datetime.now().isoformat()
                _batch_index_status["message"] = f"Error: {str(e)}"

    # Start background thread
    thread = threading.Thread(target=_batch_index_task, daemon=True)
    thread.start()

    return JSONResponse(content={
        "status": "accepted",
        "message": "Batch indexing started in background",
    })


@app.get("/documents/index-status")
async def get_index_status():
    """
    Get the current status of batch indexing.

    Returns:
    - Current indexing status
    """
    global _batch_index_status

    with _batch_index_lock:
        return JSONResponse(content=_batch_index_status.copy())


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


@app.get("/uncompleted-pages/{filename}")
async def get_uncompleted_pages(filename: str):
    """
    Get list of uncompleted pages for a document.

    An uncompleted page is one that has a JPG file but no corresponding _nohf.md file.
    This is useful for finding pages that failed during initial conversion.

    Parameters:
    - filename: The document filename (without extension) or with extension

    Returns:
    - JSON with list of page numbers that need re-conversion
    """
    try:
        # Remove extension if provided
        file_name_without_ext = os.path.splitext(filename)[0]

        output_dir = os.path.join(OUTPUT_DIR, file_name_without_ext)

        if not os.path.exists(output_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Output directory not found for: {filename}"
            )

        # Find all JPG files and their corresponding _nohf.md files
        uncompleted_pages = []
        all_pages = []

        for file in os.listdir(output_dir):
            # Match page JPG files like "document_page_11.jpg"
            if file.endswith(".jpg") and "_page_" in file:
                # Extract page number from filename
                try:
                    page_part = file.split("_page_")[1]
                    page_no = int(page_part.split(".")[0])
                    all_pages.append(page_no)

                    # Check if corresponding _nohf.md file exists
                    md_filename = f"{file_name_without_ext}_page_{page_no}_nohf.md"
                    md_path = os.path.join(output_dir, md_filename)

                    if not os.path.exists(md_path):
                        uncompleted_pages.append({
                            "page_no": page_no,
                            "jpg_file": file,
                            "expected_md_file": md_filename
                        })
                except (ValueError, IndexError):
                    # Skip files that don't match the expected pattern
                    continue

        # Sort pages by page number
        all_pages.sort()
        uncompleted_pages.sort(key=lambda x: x["page_no"])

        return JSONResponse(content={
            "status": "success",
            "filename": file_name_without_ext,
            "total_pages": len(all_pages),
            "completed_pages": len(all_pages) - len(uncompleted_pages),
            "uncompleted_count": len(uncompleted_pages),
            "uncompleted_pages": uncompleted_pages
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting uncompleted pages: {str(e)}"
        )


def _get_image_analysis_backend():
    """
    Get the configured image analysis backend.
    Returns tuple of (backend_name, converter_instance)
    """
    backend = (os.getenv("IMAGE_ANALYSIS_BACKEND", "qwen3") or "").strip().lower()

    if backend == "deepseek":
        return "deepseek", deepseek_ocr_converter
    elif backend == "gemma3" or backend == "gemma":
        return "gemma3", gemma3_ocr_converter
    else:  # Default to qwen3
        return "qwen3", qwen3_ocr_converter


def _convert_page_directly(
    jpg_path: Path,
    output_path: Path,
    backend_name: str,
    converter
) -> str:
    """
    Convert a page image directly to markdown using the specified converter.

    Args:
        jpg_path: Path to the JPG image file
        output_path: Path to save the markdown file
        backend_name: Name of the backend (qwen3, gemma3, deepseek)
        converter: The converter instance to use

    Returns:
        The markdown content
    """
    logger.info(f"Converting {jpg_path} using {backend_name} backend")

    if backend_name == "deepseek":
        # DeepSeek has convert_file method
        success = converter.convert_file(jpg_path, output_path)
        if not success:
            raise Exception(f"DeepSeek conversion failed for {jpg_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # Qwen3 and Gemma3 use base64 image input
        with open(jpg_path, 'rb') as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        markdown_content = converter.convert_image_base64_to_markdown(image_base64)

        # Save markdown to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        return markdown_content


@app.post("/reconvert-page")
async def reconvert_page(
    filename: str = Form(...),
    page_no: int = Form(...)
):
    """
    Re-convert a specific page directly using the configured IMAGE_ANALYSIS_BACKEND.

    This endpoint skips dots_ocr_service and directly uses qwen3, gemma, or deepseek
    based on the IMAGE_ANALYSIS_BACKEND environment variable.

    The page's JPG file (e.g., "document_page_11.jpg") is read and converted
    directly to markdown, saved as "_nohf.md" format.

    Parameters:
    - filename: The document filename (without extension) or with extension
    - page_no: The page number to re-convert (0-indexed)

    Returns:
    - JSON with conversion result
    """
    try:
        # Remove extension if provided
        file_name_without_ext = os.path.splitext(filename)[0]

        output_dir = Path(OUTPUT_DIR) / file_name_without_ext

        if not output_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Output directory not found for: {filename}"
            )

        # Find the JPG file for this page
        jpg_filename = f"{file_name_without_ext}_page_{page_no}.jpg"
        jpg_path = output_dir / jpg_filename

        if not jpg_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"JPG file not found: {jpg_filename}"
            )

        # Get the configured backend
        backend_name, converter = _get_image_analysis_backend()

        # Define output markdown file
        md_filename = f"{file_name_without_ext}_page_{page_no}_nohf.md"
        md_path = output_dir / md_filename

        logger.info(
            f"Re-converting page {page_no} of {file_name_without_ext} "
            f"using {backend_name} backend (skipping dots_ocr_service)"
        )

        # Convert the page directly
        markdown_content = _convert_page_directly(
            jpg_path, md_path, backend_name, converter
        )

        # Trigger re-indexing in background after reconversion
        logger.info(f"Triggering re-indexing for document after page reconvert: {file_name_without_ext}")
        reindex_document(file_name_without_ext, OUTPUT_DIR)

        return JSONResponse(content={
            "status": "success",
            "filename": file_name_without_ext,
            "page_no": page_no,
            "backend_used": backend_name,
            "output_file": str(md_path),
            "content_length": len(markdown_content),
            "message": f"Page {page_no} re-converted successfully using {backend_name}"
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error re-converting page: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error re-converting page: {str(e)}"
        )


@app.post("/reconvert-uncompleted")
async def reconvert_all_uncompleted(
    filename: str = Form(...)
):
    """
    Re-convert all uncompleted pages for a document.

    This endpoint finds all pages that have JPG but no _nohf.md file,
    and converts them directly using the configured IMAGE_ANALYSIS_BACKEND.

    Parameters:
    - filename: The document filename (without extension) or with extension

    Returns:
    - JSON with list of re-converted pages and their status
    """
    try:
        # Remove extension if provided
        file_name_without_ext = os.path.splitext(filename)[0]

        output_dir = Path(OUTPUT_DIR) / file_name_without_ext

        if not output_dir.exists():
            # Document hasn't been converted yet - trigger full conversion
            # Find the input file with any extension
            input_file = None
            for f in os.listdir(INPUT_DIR):
                if os.path.splitext(f)[0] == file_name_without_ext:
                    input_file = f
                    break

            if not input_file:
                raise HTTPException(
                    status_code=404,
                    detail=f"Input file not found for: {filename}"
                )

            # Trigger conversion using the same logic as /convert endpoint
            conversion_id = conversion_manager.create_conversion(input_file)

            conversion_manager.update_conversion(
                conversion_id,
                status="processing",
                started_at=datetime.now().isoformat(),
                message="Conversion queued, waiting for worker..."
            )

            await connection_manager.broadcast(conversion_id, {
                "status": "processing",
                "progress": 5,
                "message": "Conversion queued, waiting for worker...",
            })

            progress_callback = _create_parser_progress_callback(conversion_id)

            success = worker_pool.submit_task(
                conversion_id=conversion_id,
                func=_convert_document_background,
                args=(input_file, "prompt_layout_all_en"),
                kwargs={
                    "conversion_id": conversion_id,
                    "progress_callback": progress_callback,
                }
            )

            if not success:
                raise HTTPException(
                    status_code=409,
                    detail=f"Conversion already in progress for: {filename}"
                )

            logger.info(f"Triggered full conversion for new file: {input_file}")

            return JSONResponse(content={
                "status": "accepted",
                "conversion_id": conversion_id,
                "filename": file_name_without_ext,
                "message": "New file - full conversion started. Use WebSocket to track progress.",
                "queue_size": worker_pool.get_queue_size(),
                "active_tasks": worker_pool.get_active_tasks_count(),
            })

        # Find uncompleted pages
        uncompleted_pages = []
        for file in os.listdir(output_dir):
            if file.endswith(".jpg") and "_page_" in file:
                try:
                    page_part = file.split("_page_")[1]
                    page_no = int(page_part.split(".")[0])

                    md_filename = f"{file_name_without_ext}_page_{page_no}_nohf.md"
                    md_path = output_dir / md_filename

                    if not md_path.exists():
                        uncompleted_pages.append(page_no)
                except (ValueError, IndexError):
                    continue

        uncompleted_pages.sort()

        if not uncompleted_pages:
            return JSONResponse(content={
                "status": "success",
                "filename": file_name_without_ext,
                "message": "No uncompleted pages found",
                "converted_count": 0,
                "results": []
            })

        # Get the configured backend
        backend_name, converter = _get_image_analysis_backend()

        logger.info(
            f"Re-converting {len(uncompleted_pages)} uncompleted pages of "
            f"{file_name_without_ext} using {backend_name} backend"
        )

        results = []
        for page_no in uncompleted_pages:
            jpg_filename = f"{file_name_without_ext}_page_{page_no}.jpg"
            jpg_path = output_dir / jpg_filename
            md_filename = f"{file_name_without_ext}_page_{page_no}_nohf.md"
            md_path = output_dir / md_filename

            try:
                markdown_content = _convert_page_directly(
                    jpg_path, md_path, backend_name, converter
                )
                results.append({
                    "page_no": page_no,
                    "status": "success",
                    "content_length": len(markdown_content)
                })
                logger.info(f"Successfully re-converted page {page_no}")
            except Exception as e:
                results.append({
                    "page_no": page_no,
                    "status": "error",
                    "error": str(e)
                })
                logger.error(f"Failed to re-convert page {page_no}: {str(e)}")

        success_count = sum(1 for r in results if r["status"] == "success")

        # Trigger re-indexing in background after reconversion (if any pages were converted)
        if success_count > 0:
            logger.info(f"Triggering re-indexing for document after uncompleted reconvert: {file_name_without_ext}")
            reindex_document(file_name_without_ext, OUTPUT_DIR)

        return JSONResponse(content={
            "status": "success" if success_count == len(results) else "partial",
            "filename": file_name_without_ext,
            "backend_used": backend_name,
            "total_uncompleted": len(uncompleted_pages),
            "converted_count": success_count,
            "failed_count": len(results) - success_count,
            "results": results
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error re-converting uncompleted pages: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error re-converting uncompleted pages: {str(e)}"
        )


@app.post("/admin/update-document-owners")
async def update_document_owners(username: str):
    """
    Admin endpoint to update all documents with a specific user as creator/updater.
    This is useful for assigning ownership to existing documents.
    NOTE: This endpoint is temporarily unprotected for migration purposes.
    """
    try:
        from db.user_repository import UserRepository
        from sqlalchemy import text

        with get_db_session() as db:
            user_repo = UserRepository(db)

            # Find the target user
            user = user_repo.get_user_by_username(username)
            if not user:
                raise HTTPException(
                    status_code=404,
                    detail=f"User '{username}' not found"
                )

            # Count documents that need updating
            count_query = db.execute(
                text('''
                    SELECT COUNT(*)
                    FROM documents
                    WHERE created_by IS NULL OR updated_by IS NULL
                ''')
            )
            count = count_query.scalar()

            if count == 0:
                return {
                    "status": "success",
                    "message": "No documents need updating",
                    "updated_count": 0,
                    "user": {
                        "username": user.username,
                        "id": str(user.id)
                    }
                }

            # Update all documents that don't have created_by or updated_by set
            result = db.execute(
                text('''
                    UPDATE documents
                    SET created_by = :user_id, updated_by = :user_id
                    WHERE created_by IS NULL OR updated_by IS NULL
                '''),
                {'user_id': user.id}
            )

            logger.info(f"Updated {result.rowcount} documents with user '{username}' (ID: {user.id})")

            return {
                "status": "success",
                "message": f"Successfully updated {result.rowcount} documents",
                "updated_count": result.rowcount,
                "user": {
                    "username": user.username,
                    "id": str(user.id),
                    "full_name": user.full_name
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document owners: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating document owners: {str(e)}"
        )


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

