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
import time
import shutil
import tempfile
import threading
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

import logging

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect, Request, Query, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Import OCR and conversion components
from dots_ocr_service.parser import DotsOCRParser
from worker_pool import WorkerPool
from doc_service.document_converter_manager import DocumentConverterManager
from deepseek_ocr_service.deepseek_ocr_converter import DeepSeekOCRConverter
from gemma_ocr_service.gemma3_ocr_converter import Gemma3OCRConverter
from qwen_ocr_service.qwen3_ocr_converter import Qwen3OCRConverter

# Import RAG service for chatbot functionality
from rag_service.chat_api import router as chat_router
from rag_service.indexer import trigger_embedding_for_document, reindex_document

# Import GraphRAG for source-level deletion and Neo4j initialization
try:
    from rag_service.graph_rag import delete_graphrag_by_source_sync, GRAPH_RAG_INDEX_ENABLED, GRAPH_RAG_QUERY_ENABLED
    GRAPHRAG_DELETE_AVAILABLE = True
except ImportError:
    GRAPHRAG_DELETE_AVAILABLE = False
    GRAPH_RAG_INDEX_ENABLED = False
    GRAPH_RAG_QUERY_ENABLED = False

# Import database services
from db.database import init_db, get_db_session, get_db
from db.document_repository import DocumentRepository
from db.workspace_repository import WorkspaceRepository
from db.models import Document, UploadStatus, ConvertStatus, IndexStatus, User

# Import authentication
from auth.auth_api import router as auth_router
from auth.dependencies import get_current_active_user

# Import chat session management
from chat_service.chat_session_api import router as chat_session_router

# Import workspace and sharing APIs
from services.workspace_api import router as workspace_router
from services.sharing_api import router as sharing_router

# Import analytics API (deferred logging since logger not yet defined)
try:
    from analytics_service.analytics_api import router as analytics_router
    ANALYTICS_AVAILABLE = True
    _analytics_import_error = None
except ImportError as e:
    ANALYTICS_AVAILABLE = False
    _analytics_import_error = str(e)

# Import task queue system (hierarchical)
from queue_service import (
    HierarchicalTaskQueueManager,
    TaskScheduler,
    HierarchicalWorkerPool,
    TaskStatus,
    TaskQueuePage,
)

# Import services
from services import (
    DocumentService,
    ConversionService,
    ConversionManager,
    ConnectionManager,
    DocumentStatusManager,
    IndexingService,
    TaskQueueService,
    PermissionService,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log deferred analytics import error if any
if _analytics_import_error:
    logger.warning(f"Analytics service not available: {_analytics_import_error}")

# Suppress noisy asyncio "Event loop is closed" errors during cleanup
_original_exception_handler = None


def _custom_exception_handler(loop, context):
    """Custom exception handler that filters out event loop closed errors."""
    exception = context.get("exception")
    if exception and isinstance(exception, RuntimeError) and "Event loop is closed" in str(exception):
        return
    if _original_exception_handler:
        _original_exception_handler(loop, context)
    else:
        loop.default_exception_handler(context)


# Define input and output directories
INPUT_DIR = os.path.join(os.path.dirname(__file__), "input")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Create directories if they don't exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get number of worker threads from environment (default: 4)
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))

# Task queue configuration
TASK_QUEUE_ENABLED = os.getenv("TASK_QUEUE_ENABLED", "true").lower() in ("true", "1", "yes")
TASK_QUEUE_CHECK_INTERVAL = int(os.getenv("TASK_QUEUE_CHECK_INTERVAL", "300"))
WORKER_POLL_INTERVAL = int(os.getenv("WORKER_POLL_INTERVAL", "5"))
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "30"))
STALE_TASK_TIMEOUT = int(os.getenv("STALE_TASK_TIMEOUT", "300"))

# Initialize parser
parser = DotsOCRParser()

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
gemma3_ocr_converter = parser.gemma3_converter
qwen3_ocr_converter = parser.qwen3_converter

if gemma3_ocr_converter is None:
    logger.warning("Parser's gemma3_converter is None, creating standalone Gemma3OCRConverter")
    gemma3_ocr_converter = Gemma3OCRConverter()
if qwen3_ocr_converter is None:
    logger.warning("Parser's qwen3_converter is None, creating standalone Qwen3OCRConverter")
    qwen3_ocr_converter = Qwen3OCRConverter()

# Initialize services
document_service = DocumentService(INPUT_DIR, OUTPUT_DIR)
conversion_manager = ConversionManager()
connection_manager = ConnectionManager()
document_status_manager = DocumentStatusManager()
indexing_service = IndexingService(OUTPUT_DIR, broadcast_callback=None)

# Initialize worker pool with progress callback
def _worker_progress_callback(conversion_id: str, status: str, result=None, error=None):
    """Callback for worker pool progress updates."""
    try:
        if status == "completed":
            is_skipped = False
            skip_reason = None
            source_name = None
            filename = None

            if result and isinstance(result, list) and len(result) > 0:
                first_result = result[0]
                if isinstance(first_result, dict) and first_result.get('skipped'):
                    is_skipped = True
                    skip_reason = first_result.get('skip_reason', 'Image was skipped')

                if isinstance(first_result, dict):
                    md_path = first_result.get('md_content_path') or first_result.get('md_content_nohf_path')
                    if md_path:
                        source_name = Path(md_path).parent.name
                    file_path = first_result.get('file_path')
                    if file_path:
                        filename = Path(file_path).name

            if is_skipped:
                conversion_manager.update_conversion(
                    conversion_id,
                    status="warning",
                    progress=100,
                    message=f"⚠️ {skip_reason}",
                    completed_at=datetime.now().isoformat(),
                )
                connection_manager.broadcast_from_thread(conversion_id, {
                    "status": "warning",
                    "progress": 100,
                    "message": f"⚠️ {skip_reason}",
                    "results": result,
                })
            else:
                conversion_manager.update_conversion(
                    conversion_id,
                    status="completed",
                    progress=100,
                    message="Conversion completed successfully",
                    completed_at=datetime.now().isoformat(),
                )
                connection_manager.broadcast_from_thread(conversion_id, {
                    "status": "completed",
                    "progress": 100,
                    "message": "Conversion completed successfully",
                    "results": result,
                })

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

                if source_name:
                    logger.info(f"Triggering embedding for document: {source_name}")
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

                    def dual_broadcast_callback(conv_id: str, message: dict):
                        connection_manager.broadcast_from_thread(conv_id, message)
                        if doc_id and filename:
                            msg_status = message.get("status", "")
                            if msg_status in ["vector_indexed", "metadata_extracted", "graphrag_indexed", "index_error"]:
                                event_type = "indexing_completed" if msg_status != "index_error" else "indexing_failed"
                                document_status_manager.broadcast_from_thread({
                                    "event_type": event_type,
                                    "document_id": doc_id,
                                    "filename": filename,
                                    "index_status": "indexed" if msg_status != "index_error" else "failed",
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
            connection_manager.broadcast_from_thread(conversion_id, {
                "status": "error",
                "progress": 0,
                "message": f"Conversion failed: {error}",
                "error": error,
            })

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


worker_pool = WorkerPool(num_workers=NUM_WORKERS, progress_callback=_worker_progress_callback)

# Initialize conversion service
conversion_service = ConversionService(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    parser=parser,
    doc_converter_manager=doc_converter_manager,
    deepseek_ocr_converter=deepseek_ocr_converter,
    gemma3_ocr_converter=gemma3_ocr_converter,
    qwen3_ocr_converter=qwen3_ocr_converter,
    worker_pool=worker_pool,
    conversion_manager=conversion_manager,
    broadcast_callback=lambda cid, msg: connection_manager.broadcast_from_thread(cid, msg)
)

# Global task queue instances (initialized in lifespan)
task_queue_manager: Optional[HierarchicalTaskQueueManager] = None
task_scheduler: Optional[TaskScheduler] = None
queue_worker_pool: Optional[HierarchicalWorkerPool] = None
task_queue_service: Optional[TaskQueueService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global task_queue_manager, task_scheduler, queue_worker_pool, task_queue_service

    # ===== STARTUP =====
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    try:
        from rag_service.vectorstore import ensure_metadata_collection_exists
        ensure_metadata_collection_exists()
        logger.info("Document metadata vector collection initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize metadata vector collection: {e}")

    loop = asyncio.get_event_loop()

    global _original_exception_handler
    _original_exception_handler = loop.get_exception_handler()
    loop.set_exception_handler(_custom_exception_handler)

    connection_manager.set_event_loop(loop)
    document_status_manager.set_event_loop(loop)
    asyncio.create_task(connection_manager.start_broadcast_worker())
    asyncio.create_task(document_status_manager.start_broadcast_worker())
    logger.info("WebSocket broadcast workers started")

    if GRAPH_RAG_INDEX_ENABLED or GRAPH_RAG_QUERY_ENABLED:
        try:
            from rag_service.storage import Neo4jStorage
            neo4j_storage = Neo4jStorage()
            await neo4j_storage.ensure_indexes()
            logger.info("Neo4j indexes initialized (including vector indexes)")
        except Exception as e:
            logger.warning(f"Failed to initialize Neo4j indexes: {e}")

    if TASK_QUEUE_ENABLED:
        logger.info("🚀 Initializing hierarchical task queue system...")

        task_queue_manager = HierarchicalTaskQueueManager(
            stale_timeout_seconds=STALE_TASK_TIMEOUT,
            max_retries=int(os.getenv("MAX_TASK_RETRIES", "3"))
        )

        task_queue_service = TaskQueueService(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            parser=parser,
            task_queue_manager=task_queue_manager,
            document_status_broadcast=document_status_manager.broadcast_from_thread,
            doc_converter_manager=doc_converter_manager
        )

        queue_worker_pool = HierarchicalWorkerPool(
            num_workers=NUM_WORKERS,
            task_manager=task_queue_manager,
            ocr_processor=task_queue_service.process_ocr_page_task,
            vector_processor=task_queue_service.process_vector_chunk_task,
            graphrag_processor=task_queue_service.process_graphrag_chunk_task,
            poll_interval=WORKER_POLL_INTERVAL,
            heartbeat_interval=HEARTBEAT_INTERVAL,
            status_broadcast_callback=document_status_manager.broadcast_from_thread
        )
        queue_worker_pool.start()

        task_scheduler = TaskScheduler(
            task_manager=task_queue_manager,
            check_interval_seconds=TASK_QUEUE_CHECK_INTERVAL
        )
        task_scheduler.start()

        task_queue_service.task_scheduler = task_scheduler
        task_queue_service.queue_worker_pool = queue_worker_pool

        logger.info("✅ Hierarchical task queue system initialized successfully")
    else:
        logger.info("Task queue system disabled (set TASK_QUEUE_ENABLED=true to enable)")

        def _init_services():
            _sync_files_to_database()
            auto_resume_ocr = os.getenv("AUTO_RESUME_OCR", "true").lower() in ("true", "1", "yes")
            if auto_resume_ocr:
                try:
                    logger.info("Checking for incomplete OCR conversion tasks...")
                    _resume_incomplete_ocr()
                except Exception as e:
                    logger.error(f"Error resuming incomplete OCR conversion: {e}")

            auto_resume_indexing = os.getenv("AUTO_RESUME_INDEXING", "true").lower() in ("true", "1", "yes")
            if auto_resume_indexing:
                try:
                    logger.info("Checking for incomplete indexing tasks...")
                    _resume_incomplete_indexing()
                except Exception as e:
                    logger.error(f"Error resuming incomplete indexing: {e}")

        import threading
        init_thread = threading.Thread(target=_init_services, daemon=True)
        init_thread.start()
        logger.info("Services initialization started in background thread")

    yield

    # ===== SHUTDOWN =====
    logger.info("Application shutting down...")

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
app.include_router(auth_router)
app.include_router(chat_session_router)
app.include_router(chat_router)
app.include_router(workspace_router)
app.include_router(sharing_router)

# Include analytics router if available
if ANALYTICS_AVAILABLE:
    app.include_router(analytics_router)
    logger.info("Analytics API enabled")


# ===== Helper Functions =====

def _sync_files_to_database():
    """Synchronize existing files in input directory with the database."""
    try:
        with get_db_session() as db:
            repo = DocumentRepository(db)
            synced_count = 0

            if not os.path.exists(INPUT_DIR):
                logger.info("Input directory does not exist, skipping sync")
                return

            for root, dirs, files in os.walk(INPUT_DIR):
                for filename in files:
                    file_path = os.path.join(root, filename)

                    if not os.path.isfile(file_path):
                        continue

                    if filename.startswith('.'):
                        continue

                    relative_path = os.path.relpath(file_path, INPUT_DIR)

                    existing_doc = repo.get_by_file_path(relative_path)
                    if existing_doc:
                        continue

                    if repo.exists(filename):
                        continue

                    file_size = os.path.getsize(file_path)
                    file_name_without_ext = os.path.splitext(filename)[0]

                    rel_dir = os.path.dirname(relative_path)
                    if rel_dir:
                        output_base = os.path.join(OUTPUT_DIR, rel_dir, file_name_without_ext)
                    else:
                        output_base = os.path.join(OUTPUT_DIR, file_name_without_ext)

                    markdown_exists, _, _, converted_pages = document_service.check_markdown_exists_at_path(output_base)
                    total_pages = document_service.get_pdf_page_count(file_path)
                    if total_pages == 0 and markdown_exists:
                        total_pages = 1

                    if not markdown_exists:
                        convert_status = ConvertStatus.PENDING
                    elif total_pages > 0 and converted_pages < total_pages:
                        convert_status = ConvertStatus.PARTIAL
                    else:
                        convert_status = ConvertStatus.CONVERTED

                    indexed = indexing_service.is_document_indexed(file_name_without_ext) if markdown_exists else False
                    index_status = IndexStatus.INDEXED if indexed else IndexStatus.PENDING

                    doc = repo.create(
                        filename=filename,
                        original_filename=filename,
                        file_path=relative_path,
                        file_size=file_size,
                        total_pages=total_pages,
                    )

                    if convert_status != ConvertStatus.PENDING:
                        repo.update_convert_status(doc, convert_status, converted_pages, output_base, "Synced from existing files")

                    if index_status == IndexStatus.INDEXED:
                        repo.update_index_status(doc, index_status, message="Synced from existing index")

                    synced_count += 1

            logger.info(f"Database sync complete: {synced_count} new documents synced")
    except Exception as e:
        logger.error(f"Error syncing files to database: {e}")


def _resume_incomplete_ocr():
    """Resume incomplete OCR conversion tasks on startup."""
    if task_queue_service:
        task_queue_service.resume_incomplete_ocr(conversion_manager, worker_pool)


def _resume_incomplete_indexing():
    """Resume incomplete indexing tasks on startup."""
    if task_queue_service:
        task_queue_service.resume_incomplete_indexing(conversion_manager, trigger_embedding_for_document)


def _create_parser_progress_callback(conversion_id: str):
    """Create a progress callback for the parser."""
    return conversion_service.create_progress_callback(conversion_id)


# ===== API Endpoints =====

@app.get("/")
async def root():
    """Root endpoint - API information."""
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
    """Health check endpoint."""
    return {"status": "healthy", "service": "Dots OCR API"}


@app.get("/queue/stats")
async def get_queue_stats():
    """Get task queue statistics."""
    if not task_queue_manager:
        raise HTTPException(status_code=503, detail="Task queue system not initialized")
    return task_queue_manager.get_queue_stats()


@app.post("/queue/maintenance")
async def trigger_queue_maintenance():
    """Manually trigger queue maintenance."""
    if not task_queue_service:
        raise HTTPException(status_code=503, detail="Task queue service not initialized")
    return task_queue_service.trigger_maintenance()


@app.get("/config")
async def get_config():
    """Get current configuration for both backend and frontend."""
    return {
        "basePath": os.getenv("BASE_PATH", ""),
        "appDomain": os.getenv("APP_DOMAIN", "http://localhost:3000"),
        "apiDomain": os.getenv("API_DOMAIN", "http://localhost:8080"),
        "iamDomain": os.getenv("IAM_DOMAIN", "http://localhost:5000"),
        "clientId": os.getenv("CLIENT_ID", "dots-ocr-app"),
        "iamScope": os.getenv("IAM_SCOPE", "openid profile email"),
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
async def upload_file(
    file: UploadFile = File(...),
    workspace_id: Optional[str] = Form(default=None),
    request: Request = None
):
    """Upload a document file (PDF, image, DOC, EXCEL) to the input folder."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        logger.info(f"📁 Upload: {file.filename}, workspace_id={workspace_id}")
        auth_header = request.headers.get("Authorization") if request else None

        with get_db_session() as db:
            user_id = None
            owner_id = None
            resolved_workspace_id = None
            workspace_name = None
            relative_path = file.filename
            absolute_path = os.path.join(INPUT_DIR, file.filename)

            if auth_header and auth_header.startswith("Bearer "):
                try:
                    from auth.jwt_utils import JWTUtils
                    from db.user_repository import UserRepository

                    token = auth_header.split(" ")[1]
                    payload = JWTUtils.verify_access_token(token)
                    if payload:
                        user_id = UUID(payload["sub"])
                        user_repo = UserRepository(db)
                        user = user_repo.get_user_by_id(user_id)
                        if user:
                            owner_id = user.id
                            username = user.username

                            if workspace_id:
                                try:
                                    ws_uuid = UUID(workspace_id)
                                    ws_repo = WorkspaceRepository(db)
                                    workspace = ws_repo.get_workspace_by_id(ws_uuid)
                                    if workspace and workspace.user_id == user_id:
                                        resolved_workspace_id = workspace.id
                                        workspace_name = workspace.name
                                        folder_path = workspace.folder_path
                                        relative_path = os.path.join(folder_path, file.filename)
                                        absolute_path = os.path.join(INPUT_DIR, folder_path, file.filename)
                                except Exception as e:
                                    logger.warning(f"Invalid workspace_id: {e}")

                            if not resolved_workspace_id:
                                ws_repo = WorkspaceRepository(db)
                                workspace = ws_repo.get_or_create_default_workspace(user_id, user.normalized_username)
                                resolved_workspace_id = workspace.id
                                workspace_name = workspace.name
                                folder_path = workspace.folder_path
                                relative_path = os.path.join(folder_path, file.filename)
                                absolute_path = os.path.join(INPUT_DIR, folder_path, file.filename)
                except Exception as e:
                    logger.warning(f"Auth error: {e}")

            if not os.path.abspath(absolute_path).startswith(os.path.abspath(INPUT_DIR)):
                raise HTTPException(status_code=400, detail="Invalid file path")

            os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
            with open(absolute_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            resize_info = document_service.resize_image_if_needed(absolute_path)
            file_size = os.path.getsize(absolute_path)
            upload_time = datetime.now().isoformat()

            repo = DocumentRepository(db)
            doc, created = repo.get_or_create(
                filename=file.filename,
                original_filename=file.filename,
                file_path=relative_path,
                file_size=file_size,
            )
            doc_id = str(doc.id)

            if resolved_workspace_id:
                doc.workspace_id = resolved_workspace_id
            if owner_id:
                doc.owner_id = owner_id
                doc.visibility = 'private'

            # Determine total_pages based on file type
            # For non-PDF files (Excel, Word, CSV, etc.), always use 1 page since
            # doc_service converter produces a single output file
            file_ext = os.path.splitext(file.filename)[1].lower()
            doc_service_extensions = {'.docx', '.doc', '.xlsx', '.xlsm', '.xlsb', '.xls', '.txt', '.csv', '.tsv', '.log', '.text'}

            if file_ext in doc_service_extensions:
                # Non-PDF files: always 1 page (single output file from doc_service)
                total_pages = 1
                logger.info(f"📄 Non-PDF file {file.filename}: setting total_pages=1 for doc_service processing")
            else:
                # PDF files: get actual page count from fitz
                try:
                    import fitz
                    pdf_doc = fitz.open(absolute_path)
                    total_pages = len(pdf_doc)
                    pdf_doc.close()
                except Exception:
                    total_pages = 1

            doc.total_pages = total_pages

            # Set skip_graphrag flag for file types that don't benefit from entity extraction
            from rag_service.graphrag_skip_config import should_skip_graphrag_for_file
            should_skip, skip_reason = should_skip_graphrag_for_file(file.filename)
            if should_skip:
                doc.skip_graphrag = True
                doc.skip_graphrag_reason = skip_reason
                logger.info(f"📊 Document {file.filename} will skip GraphRAG indexing: {skip_reason}")

            db.commit()

            if owner_id:
                perm_service = PermissionService(db)
                perm_service.grant_owner_permission(user_id=owner_id, document_id=doc.id)

            if TASK_QUEUE_ENABLED and task_queue_manager:
                task_created = task_queue_manager.create_document_task(
                    document_id=doc.id,
                    total_pages=total_pages,
                    db=db
                )
                if task_created:
                    document_status_manager.broadcast_from_thread({
                        "event_type": "document_uploaded",
                        "type": "document_status",
                        "document_id": doc_id,
                        "filename": file.filename,
                        "ocr_status": "pending",
                        "vector_status": "pending",
                        "graphrag_status": "pending",
                        "total_pages": total_pages
                    })

        response_data = {
            "status": "success",
            "id": doc_id,
            "document_id": doc_id,
            "filename": file.filename,
            "file_path": relative_path,
            "file_size": file_size,
            "upload_time": upload_time,
        }

        if resolved_workspace_id:
            response_data["workspace_id"] = str(resolved_workspace_id)
            response_data["workspace_name"] = workspace_name
        if owner_id:
            response_data["owner_id"] = str(owner_id)

        if resize_info["resized"]:
            response_data["resized"] = True
            response_data["original_size"] = resize_info["original_size"]
            response_data["new_size"] = resize_info["new_size"]
            response_data["resize_message"] = resize_info["message"]

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.get("/documents")
async def list_documents(
    workspace_id: str = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List documents in a workspace with their conversion status."""
    try:
        if not workspace_id:
            raise HTTPException(status_code=400, detail="workspace_id is required")

        try:
            ws_uuid = UUID(workspace_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid workspace_id format")

        workspace_repo = WorkspaceRepository(db)
        workspace = workspace_repo.get_workspace_by_id(ws_uuid)

        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")

        if workspace.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        doc_repo = DocumentRepository(db)
        all_docs = doc_repo.get_by_workspace(ws_uuid)

        documents = []
        for doc in all_docs:
            db_info = doc.to_dict()
            filename = doc.filename
            relative_file_path = db_info.get("file_path", filename)
            absolute_file_path = document_service.resolve_file_path(relative_file_path)

            if os.path.isfile(absolute_file_path):
                file_size = os.path.getsize(absolute_file_path)
            else:
                file_size = db_info.get("file_size", 0)

            file_name_without_ext = os.path.splitext(filename)[0]

            db_output_path = db_info.get("output_path")
            if db_output_path:
                output_dir = document_service.resolve_output_path(db_output_path)
            else:
                rel_dir = os.path.dirname(relative_file_path)
                output_dir = os.path.join(OUTPUT_DIR, rel_dir, file_name_without_ext) if rel_dir else os.path.join(OUTPUT_DIR, file_name_without_ext)

            markdown_exists, markdown_path, is_multipage, converted_pages = document_service.check_markdown_exists_at_path(output_dir)

            total_pages = db_info.get("total_pages", 0)
            if total_pages == 0 and os.path.isfile(absolute_file_path):
                total_pages = document_service.get_pdf_page_count(absolute_file_path)
            if total_pages == 0 and markdown_exists:
                total_pages = 1

            db_index_status = db_info.get("index_status", "pending")
            db_convert_status = db_info.get("convert_status", "pending")
            indexing_details = db_info.get("indexing_details")

            indexed = False
            if indexing_details:
                vector_status = indexing_details.get("vector_indexing", {}).get("status")
                metadata_status = indexing_details.get("metadata_extraction", {}).get("status")
                graphrag_status = indexing_details.get("graphrag_indexing", {}).get("status")
                indexed = (vector_status == "completed" and
                          metadata_status == "completed" and
                          graphrag_status == "completed")
            elif db_index_status == "indexed":
                indexed = True

            documents.append({
                "id": db_info.get("id"),
                "document_id": db_info.get("id"),
                "filename": filename,
                "file_path": relative_file_path,
                "file_size": file_size,
                "upload_time": db_info.get("created_at"),
                "markdown_exists": markdown_exists,
                "markdown_path": markdown_path if markdown_exists else None,
                "is_multipage": is_multipage,
                "total_pages": total_pages,
                "converted_pages": converted_pages,
                "indexed": indexed,
                "upload_status": db_info.get("upload_status", "pending"),
                "convert_status": db_convert_status,
                "index_status": db_index_status,
                "indexed_chunks": db_info.get("indexed_chunks", 0),
                "indexing_details": indexing_details,
                "ocr_details": db_info.get("ocr_details"),
                "ocr_status": db_info.get("ocr_status"),
                "vector_status": db_info.get("vector_status"),
                "graphrag_status": db_info.get("graphrag_status"),
                "workspace_id": db_info.get("workspace_id"),
            })

        return JSONResponse(content={
            "status": "success",
            "documents": documents,
            "total": len(documents),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@app.post("/convert")
async def convert_document(filename: str = Form(...), prompt_mode: str = Form("prompt_layout_all_en")):
    """Trigger document conversion using OCR parser (non-blocking)."""
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        with get_db_session() as db:
            repo = DocumentRepository(db)
            doc = repo.get_by_filename(filename)
            file_path = doc.file_path if doc and doc.file_path else os.path.join(INPUT_DIR, filename)

        if not os.path.exists(document_service.resolve_file_path(file_path)):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        success, conversion_id = conversion_service.submit_ocr_task(filename, file_path, prompt_mode)

        if not success:
            raise HTTPException(status_code=409, detail=f"Conversion already in progress for: {filename}")

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


@app.post("/convert-doc")
async def convert_document_with_doc_service(filename: str = Form(...)):
    """Convert Word/Excel/TXT documents using doc_service converters (non-blocking)."""
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        with get_db_session() as db:
            repo = DocumentRepository(db)
            doc = repo.get_by_filename(filename)
            file_path = doc.file_path if doc and doc.file_path else os.path.join(INPUT_DIR, filename)

        file_path_obj = Path(document_service.resolve_file_path(file_path))

        if not file_path_obj.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        doc_service_extensions = ['.docx', '.doc', '.xlsx', '.xlsm', '.xlsb', '.xls', '.txt', '.csv', '.tsv', '.log', '.text']
        if file_path_obj.suffix.lower() not in doc_service_extensions:
            raise HTTPException(status_code=400, detail=f"File type not supported by doc_service")

        success, conversion_id = conversion_service.submit_doc_service_task(filename, file_path)

        if not success:
            raise HTTPException(status_code=409, detail=f"Conversion already in progress for: {filename}")

        return JSONResponse(content={
            "status": "accepted",
            "conversion_id": conversion_id,
            "filename": filename,
            "message": "Document conversion task started.",
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
    """Convert images using DeepSeek OCR service (non-blocking)."""
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        with get_db_session() as db:
            repo = DocumentRepository(db)
            doc = repo.get_by_filename(filename)
            file_path = doc.file_path if doc and doc.file_path else os.path.join(INPUT_DIR, filename)

        file_path_obj = Path(document_service.resolve_file_path(file_path))

        if not file_path_obj.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        deepseek_ocr_extensions = deepseek_ocr_converter.get_supported_extensions()
        if file_path_obj.suffix.lower() not in deepseek_ocr_extensions:
            raise HTTPException(status_code=400, detail=f"File type not supported by DeepSeek OCR")

        success, conversion_id = conversion_service.submit_deepseek_task(filename, file_path)

        if not success:
            raise HTTPException(status_code=409, detail=f"Conversion already in progress for: {filename}")

        return JSONResponse(content={
            "status": "accepted",
            "conversion_id": conversion_id,
            "filename": filename,
            "message": "DeepSeek OCR conversion task started.",
            "converter_type": "deepseek_ocr",
            "queue_size": worker_pool.get_queue_size(),
            "active_tasks": worker_pool.get_active_tasks_count(),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting DeepSeek OCR conversion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting conversion: {str(e)}")


@app.get("/conversion-status/{conversion_id}")
async def get_conversion_status(conversion_id: str):
    """Get the status of a conversion task."""
    conversion = conversion_manager.get_conversion(conversion_id)
    if not conversion:
        raise HTTPException(status_code=404, detail=f"Conversion not found: {conversion_id}")
    return JSONResponse(content=conversion)


@app.get("/worker-pool-status")
async def get_worker_pool_status():
    """Get the status of the worker pool."""
    return JSONResponse(content={
        "status": "ok",
        "queue_size": worker_pool.get_queue_size(),
        "active_tasks": worker_pool.get_active_tasks_count(),
        "num_workers": NUM_WORKERS,
    })


@app.websocket("/ws/document-status")
async def websocket_document_status(websocket: WebSocket):
    """Centralized WebSocket endpoint for all document status updates."""
    try:
        await document_status_manager.connect(websocket)
        await websocket.send_json({
            "event_type": "connected",
            "message": "Connected to document status updates",
            "timestamp": datetime.now().isoformat()
        })

        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                action = message.get("action")

                if action == "subscribe":
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
                    await websocket.send_json({
                        "event_type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received from WebSocket")
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")

    except WebSocketDisconnect:
        await document_status_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Document status WebSocket error: {e}")
        await document_status_manager.disconnect(websocket)


@app.websocket("/ws/conversion/{conversion_id}")
async def websocket_conversion_progress(websocket: WebSocket, conversion_id: str):
    """WebSocket endpoint for real-time conversion progress updates."""
    try:
        await connection_manager.connect(conversion_id, websocket)
        conversion = conversion_manager.get_conversion(conversion_id)
        if conversion:
            await websocket.send_json(conversion)

        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        await connection_manager.disconnect(conversion_id, websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await connection_manager.disconnect(conversion_id, websocket)


@app.get("/markdown/{filename}")
async def get_markdown_content(filename: str, page_no: int = None):
    """Get the markdown content of a converted document."""
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        with get_db_session() as db:
            content, actual_page_no = document_service.get_markdown_content(filename, page_no, db)

        return JSONResponse(content={
            "status": "success",
            "filename": filename,
            "page_no": actual_page_no,
            "content": content,
        })

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Markdown file not found for: {filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading markdown file: {str(e)}")


@app.put("/markdown/{filename}")
async def update_markdown_content(filename: str, request: Request, page_no: int = None):
    """Update the markdown content of a converted document."""
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        body = await request.json()
        content = body.get("content")

        if content is None:
            raise HTTPException(status_code=400, detail="No content provided")

        document_service.update_markdown_content(filename, content, page_no)
        indexing_service.reindex_document(filename)

        return JSONResponse(content={
            "status": "success",
            "filename": filename,
            "page_no": page_no,
            "message": "Markdown content updated successfully",
        })

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Markdown file not found for: {filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating markdown file: {str(e)}")


@app.get("/markdown-files/{filename}")
async def list_markdown_files(filename: str):
    """List all markdown files associated with a document."""
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        with get_db_session() as db:
            markdown_files = document_service.list_markdown_files(filename, db)

        if not markdown_files:
            raise HTTPException(status_code=404, detail=f"No markdown files found for: {filename}")

        return JSONResponse(content={
            "status": "success",
            "filename": filename,
            "markdown_files": markdown_files,
            "total": len(markdown_files),
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing markdown files: {str(e)}")


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a document and all its associated files by document ID."""
    try:
        if not document_id:
            raise HTTPException(status_code=400, detail="No document ID provided")

        # Parse and validate document ID
        try:
            doc_uuid = UUID(document_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid document ID format")

        # Get document from database
        repo = DocumentRepository(db)
        doc = repo.get_by_id(doc_uuid)

        if not doc:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

        # Check permission - user must own the document or have delete permission
        if doc.owner_id and doc.owner_id != current_user.id:
            perm_service = PermissionService(db)
            if not perm_service.has_permission(current_user.id, doc.id, 'delete'):
                raise HTTPException(status_code=403, detail="You don't have permission to delete this document")

        filename = doc.filename
        file_path = doc.file_path  # This contains the relative path like "username/workspace/filename"
        output_path = doc.output_path
        file_name_without_ext = os.path.splitext(filename)[0]

        deleted_files = []
        errors = []

        # Delete input file using the actual file_path from database
        if file_path:
            input_file_path = os.path.join(INPUT_DIR, file_path)
        else:
            input_file_path = os.path.join(INPUT_DIR, filename)

        if os.path.exists(input_file_path):
            try:
                os.remove(input_file_path)
                deleted_files.append(input_file_path)
                logger.info(f"Deleted input file: {input_file_path}")
            except Exception as e:
                errors.append(f"Failed to delete input file: {str(e)}")

        # Delete output folder - use output_path from database if available
        if output_path:
            output_folder_path = document_service.resolve_output_path(output_path)
        else:
            # Fallback: construct from file_path
            if file_path:
                rel_dir = os.path.dirname(file_path)
                output_folder_path = os.path.join(OUTPUT_DIR, rel_dir, file_name_without_ext) if rel_dir else os.path.join(OUTPUT_DIR, file_name_without_ext)
            else:
                output_folder_path = os.path.join(OUTPUT_DIR, file_name_without_ext)

        if os.path.exists(output_folder_path) and os.path.isdir(output_folder_path):
            try:
                shutil.rmtree(output_folder_path)
                deleted_files.append(output_folder_path)
                logger.info(f"Deleted output folder: {output_folder_path}")
            except Exception as e:
                errors.append(f"Failed to delete output folder: {str(e)}")

        # Delete JSONL file
        jsonl_file_path = os.path.join(os.path.dirname(output_folder_path), f"{file_name_without_ext}.jsonl")
        if os.path.exists(jsonl_file_path):
            try:
                os.remove(jsonl_file_path)
                deleted_files.append(jsonl_file_path)
                logger.info(f"Deleted JSONL file: {jsonl_file_path}")
            except Exception as e:
                errors.append(f"Failed to delete JSONL file: {str(e)}")

        # Delete embeddings using document ID
        try:
            indexing_service.delete_document_embeddings(filename, document_id)
            logger.info(f"Deleted embeddings for document: {document_id}")
        except Exception as e:
            errors.append(f"Failed to delete embeddings: {str(e)}")

        # Delete GraphRAG data if available
        if GRAPHRAG_DELETE_AVAILABLE:
            try:
                # Use the source name (folder name) for GraphRAG deletion
                source_name = file_name_without_ext
                delete_graphrag_by_source_sync(source_name)
                logger.info(f"Deleted GraphRAG data for source: {source_name}")
            except Exception as e:
                errors.append(f"Failed to delete GraphRAG data: {str(e)}")

        # Delete extracted data (documents_data) - explicitly delete before document
        # Note: This would also cascade delete via FK, but explicit is clearer
        try:
            from db.models import DocumentData
            deleted_count = db.query(DocumentData).filter(
                DocumentData.document_id == doc_uuid
            ).delete()
            if deleted_count > 0:
                db.commit()
                logger.info(f"Deleted {deleted_count} extracted data record(s) for document: {document_id}")
        except Exception as e:
            errors.append(f"Failed to delete extracted data: {str(e)}")

        # Delete document from database
        try:
            repo.hard_delete(doc)
            logger.info(f"Deleted document from database: {document_id}")
        except Exception as e:
            errors.append(f"Failed to delete database record: {str(e)}")

        response_data = {
            "status": "success" if not errors else "partial_success",
            "document_id": document_id,
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
        logger.error(f"Error deleting document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@app.get("/documents/{document_id}/status-logs")
async def get_document_status_logs(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get status logs for a document by document ID."""
    try:
        if not document_id:
            raise HTTPException(status_code=400, detail="No document ID provided")

        # Parse and validate document ID
        try:
            doc_uuid = UUID(document_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid document ID format")

        # Get document from database
        repo = DocumentRepository(db)
        doc = repo.get_by_id(doc_uuid)

        if not doc:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

        # Check permission - user must own the document or have read permission
        if doc.owner_id and doc.owner_id != current_user.id:
            perm_service = PermissionService(db)
            if not perm_service.has_permission(current_user.id, doc.id, 'read'):
                raise HTTPException(status_code=403, detail="You don't have permission to view this document")

        # Get status logs
        status_logs = repo.get_status_logs(doc_uuid)

        logs_data = []
        for log in status_logs:
            logs_data.append({
                "id": str(log.id),
                "status_type": log.status_type,
                "old_status": log.old_status,
                "new_status": log.new_status,
                "message": log.message,
                "details": log.details,
                "created_at": log.created_at.isoformat() if log.created_at else None,
            })

        return JSONResponse(content={
            "status": "success",
            "document_id": document_id,
            "filename": doc.filename,
            "logs": logs_data,
            "total": len(logs_data),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status logs: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting status logs: {str(e)}")


@app.post("/documents/{filename}/index")
async def index_single_document(filename: str):
    """Index a single document's markdown files into the vector database."""
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        file_name_without_ext = os.path.splitext(filename)[0]

        with get_db_session() as db:
            repo = DocumentRepository(db)
            doc = repo.get_by_filename(filename)

            if doc and doc.output_path:
                doc_dir = document_service.resolve_output_path(doc.output_path)
            elif doc and doc.file_path:
                rel_dir = os.path.dirname(doc.file_path)
                doc_dir = os.path.join(OUTPUT_DIR, rel_dir, file_name_without_ext) if rel_dir else os.path.join(OUTPUT_DIR, file_name_without_ext)
            else:
                doc_dir = os.path.join(OUTPUT_DIR, file_name_without_ext)

        if not os.path.exists(doc_dir):
            raise HTTPException(status_code=404, detail=f"Document output folder not found")

        conversion_id = conversion_manager.create_conversion(filename)
        conversion_manager.update_conversion(conversion_id, status="indexing", progress=0, message="Starting indexing...")

        doc_id = None
        try:
            with get_db_session() as db:
                repo = DocumentRepository(db)
                doc = repo.get_by_filename(filename)
                if doc:
                    doc_id = str(doc.id)
        except Exception:
            pass

        def dual_broadcast_callback(conv_id: str, message: dict):
            connection_manager.broadcast_from_thread(conv_id, message)
            if doc_id:
                msg_status = message.get("status", "")
                if msg_status in ["vector_indexed", "metadata_extracted", "graphrag_indexed", "index_error"]:
                    event_type = "indexing_completed" if msg_status != "index_error" else "indexing_failed"
                    document_status_manager.broadcast_from_thread({
                        "event_type": event_type,
                        "document_id": doc_id,
                        "filename": filename,
                        "index_status": "indexed" if msg_status != "index_error" else "failed",
                        "progress": 100,
                        "message": message.get("message", "Indexing completed"),
                        "timestamp": datetime.now().isoformat()
                    })

        indexing_service.index_single_document(filename, doc_dir, conversion_id, dual_broadcast_callback)

        return JSONResponse(content={
            "status": "accepted",
            "conversion_id": conversion_id,
            "filename": filename,
            "source_name": file_name_without_ext,
            "message": "Indexing started in background.",
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting indexing: {str(e)}")


@app.post("/documents/index-all")
async def index_all_documents():
    """Start batch indexing of all documents in the background."""
    if not indexing_service.start_batch_indexing():
        return JSONResponse(
            status_code=409,
            content={
                "status": "conflict",
                "message": "Batch indexing is already in progress",
                "current_status": indexing_service.get_batch_index_status(),
            }
        )

    return JSONResponse(content={
        "status": "accepted",
        "message": "Batch indexing started in background",
    })


@app.get("/documents/index-status")
async def get_index_status():
    """Get the current status of batch indexing."""
    return JSONResponse(content=indexing_service.get_batch_index_status())


@app.get("/documents/in-progress")
async def get_in_progress_documents():
    """Get all documents that are not fully indexed.

    A document is considered "not fully indexed" if:
    - convert_status is not "converted" (pending, converting, partial, failed)
    - index_status is not "indexed" (pending, indexing, partial, failed)
    - Any indexing phase is not "completed"
    """
    try:
        in_progress_docs = []

        with get_db_session() as db:
            repo = DocumentRepository(db)
            all_docs = repo.get_all()

            for doc in all_docs:
                # Check conversion status - only "converted" is complete
                is_conversion_complete = doc.convert_status == ConvertStatus.CONVERTED
                is_converting = doc.convert_status in [ConvertStatus.PENDING, ConvertStatus.CONVERTING]

                # Check indexing status
                is_index_complete = doc.index_status == IndexStatus.INDEXED
                is_indexing = doc.index_status in [IndexStatus.PENDING, IndexStatus.INDEXING]

                # Check granular indexing phases
                all_phases_complete = False
                is_granular_indexing = False
                if doc.indexing_details:
                    vector_status = doc.indexing_details.get("vector_indexing", {}).get("status")
                    metadata_status = doc.indexing_details.get("metadata_extraction", {}).get("status")
                    graphrag_status = doc.indexing_details.get("graphrag_indexing", {}).get("status")

                    is_granular_indexing = (
                        vector_status in ["processing", "pending"] or
                        metadata_status in ["processing", "pending"] or
                        graphrag_status in ["processing", "pending"]
                    )

                    all_phases_complete = (
                        vector_status == "completed" and
                        metadata_status == "completed" and
                        graphrag_status == "completed"
                    )

                # Auto-fix index status if all phases are complete
                if all_phases_complete and doc.index_status == IndexStatus.INDEXING:
                    try:
                        indexed_chunks = doc.indexed_chunks or 0
                        repo.update_index_status(doc, IndexStatus.INDEXED, indexed_chunks, message="All indexing phases completed")
                        is_index_complete = True
                        is_indexing = False
                    except Exception:
                        pass

                # Document is fully indexed only when both conversion and indexing are complete
                # If indexing_details is empty/None, document is not fully indexed
                is_fully_indexed = (
                    is_conversion_complete and
                    is_index_complete and
                    doc.indexing_details and  # Must have indexing details
                    all_phases_complete
                )

                # Include document if it's not fully indexed
                if not is_fully_indexed:
                    doc_dict = doc.to_dict()
                    doc_dict["is_converting"] = is_converting
                    doc_dict["is_indexing"] = is_indexing or is_granular_indexing
                    doc_dict["is_fully_indexed"] = False
                    in_progress_docs.append(doc_dict)

        return JSONResponse(content={
            "status": "success",
            "documents": in_progress_docs,
            "total": len(in_progress_docs),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting in-progress documents: {str(e)}")


@app.get("/image/{filename}")
async def get_image(filename: str, page_no: int = None):
    """Get the JPG image of a converted document."""
    try:
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Look up document in database to get the correct output path
        output_dir = None
        with get_db_session() as db:
            repo = DocumentRepository(db)
            doc = repo.get_by_filename(filename)
            if not doc:
                # Try common extensions
                for ext in ['.pdf', '.docx', '.xlsx', '.png', '.jpg', '.jpeg']:
                    doc = repo.get_by_filename(filename + ext)
                    if doc:
                        break

            if doc and doc.output_path:
                output_dir = document_service.resolve_output_path(doc.output_path)
            elif doc and doc.file_path:
                rel_dir = os.path.dirname(doc.file_path)
                file_name_without_ext = os.path.splitext(doc.filename)[0]
                output_dir = os.path.join(OUTPUT_DIR, rel_dir, file_name_without_ext) if rel_dir else os.path.join(OUTPUT_DIR, file_name_without_ext)

        # Fallback to legacy path
        if not output_dir:
            output_dir = os.path.join(OUTPUT_DIR, filename)

        if page_no is not None:
            image_path = os.path.join(output_dir, f"{filename}_page_{page_no}.jpg")
        else:
            image_path = os.path.join(output_dir, f"{filename}.jpg")

        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image file not found for: {filename}")

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
    """Parse a PDF or image file for document layout and text extraction."""
    temp_dir = None
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)

        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        parsed_bbox = None
        if bbox:
            try:
                parsed_bbox = json.loads(bbox)
                if not isinstance(parsed_bbox, list) or len(parsed_bbox) != 4:
                    raise ValueError("bbox must be a list of 4 numbers")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid bbox format")

        results = parser.parse_file(
            temp_file_path,
            prompt_mode=prompt_mode,
            bbox=parsed_bbox,
        )

        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "prompt_mode": prompt_mode,
            "results": results,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@app.get("/uncompleted-pages/{filename}")
async def get_uncompleted_pages(filename: str):
    """Get list of uncompleted pages for a document."""
    try:
        file_name_without_ext = os.path.splitext(filename)[0]
        uncompleted = document_service.get_uncompleted_pages(file_name_without_ext)

        output_dir = os.path.join(OUTPUT_DIR, file_name_without_ext)
        all_pages = []
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.endswith(".jpg") and "_page_" in file:
                    try:
                        page_part = file.split("_page_")[1]
                        page_no = int(page_part.split(".")[0])
                        all_pages.append(page_no)
                    except (ValueError, IndexError):
                        continue
        all_pages.sort()

        return JSONResponse(content={
            "status": "success",
            "filename": file_name_without_ext,
            "total_pages": len(all_pages),
            "completed_pages": len(all_pages) - len(uncompleted),
            "uncompleted_count": len(uncompleted),
            "uncompleted_pages": uncompleted
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting uncompleted pages: {str(e)}")


@app.post("/reconvert-page")
async def reconvert_page(filename: str = Form(...), page_no: int = Form(...)):
    """Re-convert a specific page directly using the configured IMAGE_ANALYSIS_BACKEND."""
    try:
        file_name_without_ext = os.path.splitext(filename)[0]
        output_dir = Path(OUTPUT_DIR) / file_name_without_ext

        if not output_dir.exists():
            raise HTTPException(status_code=404, detail=f"Output directory not found")

        jpg_filename = f"{file_name_without_ext}_page_{page_no}.jpg"
        jpg_path = output_dir / jpg_filename

        if not jpg_path.exists():
            raise HTTPException(status_code=404, detail=f"JPG file not found: {jpg_filename}")

        backend_name, converter = conversion_service.get_image_analysis_backend()

        md_filename = f"{file_name_without_ext}_page_{page_no}_nohf.md"
        md_path = output_dir / md_filename

        markdown_content = conversion_service.convert_page_directly(jpg_path, md_path, backend_name, converter)

        indexing_service.reindex_document(file_name_without_ext)

        return JSONResponse(content={
            "status": "success",
            "filename": file_name_without_ext,
            "page_no": page_no,
            "backend_used": backend_name,
            "output_file": str(md_path),
            "content_length": len(markdown_content),
            "message": f"Page {page_no} re-converted successfully"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error re-converting page: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import sys

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))

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
