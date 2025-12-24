"""
Conversion service for document OCR and format conversion.
"""
import os
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Callable, Any

from sqlalchemy.orm import Session

from db.models import Document, ConvertStatus
from db.document_repository import DocumentRepository
from db.database import get_db_session
from worker_pool import WorkerPool

logger = logging.getLogger(__name__)


class ConversionManager:
    """Manages document conversion tasks and their status tracking."""

    def __init__(self):
        self.conversions: Dict[str, dict] = {}  # conversion_id -> status info
        self.lock = threading.Lock()

    def create_conversion(self, filename: str) -> str:
        """Create a new conversion task and return its ID."""
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
        """Update conversion status."""
        with self.lock:
            if conversion_id in self.conversions:
                self.conversions[conversion_id].update(kwargs)

    def get_conversion(self, conversion_id: str) -> dict:
        """Get conversion status - returns a copy to avoid race conditions."""
        with self.lock:
            conversion = self.conversions.get(conversion_id, {})
            return dict(conversion) if conversion else {}

    def delete_conversion(self, conversion_id: str):
        """Delete conversion record."""
        with self.lock:
            if conversion_id in self.conversions:
                del self.conversions[conversion_id]


class ConversionService:
    """
    Service for document conversion operations including:
    - OCR conversion using DotsOCRParser
    - Document conversion using doc_service (Word/Excel/TXT)
    - DeepSeek OCR for images
    - Qwen3/Gemma3 OCR backends
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        parser: Any,
        doc_converter_manager: Any,
        deepseek_ocr_converter: Any,
        gemma3_ocr_converter: Any,
        qwen3_ocr_converter: Any,
        worker_pool: WorkerPool,
        conversion_manager: ConversionManager,
        broadcast_callback: Optional[Callable] = None
    ):
        """
        Initialize conversion service.

        Args:
            input_dir: Base directory for input files
            output_dir: Base directory for output files
            parser: DotsOCRParser instance
            doc_converter_manager: DocumentConverterManager for Word/Excel/TXT
            deepseek_ocr_converter: DeepSeekOCRConverter instance
            gemma3_ocr_converter: Gemma3OCRConverter instance
            qwen3_ocr_converter: Qwen3OCRConverter instance
            worker_pool: WorkerPool for background processing
            conversion_manager: ConversionManager for status tracking
            broadcast_callback: Callback for WebSocket broadcasts
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.parser = parser
        self.doc_converter_manager = doc_converter_manager
        self.deepseek_ocr_converter = deepseek_ocr_converter
        self.gemma3_ocr_converter = gemma3_ocr_converter
        self.qwen3_ocr_converter = qwen3_ocr_converter
        self.worker_pool = worker_pool
        self.conversion_manager = conversion_manager
        self.broadcast_callback = broadcast_callback

    def resolve_file_path(self, relative_or_absolute_path: str, base_dir: str = None) -> str:
        """Resolve a file path to absolute path."""
        if base_dir is None:
            base_dir = self.input_dir

        if os.path.isabs(relative_or_absolute_path):
            if os.path.exists(relative_or_absolute_path):
                return relative_or_absolute_path
            for known_base in [self.input_dir, self.output_dir]:
                if known_base in relative_or_absolute_path:
                    relative_part = relative_or_absolute_path.split(known_base + os.sep)[-1]
                    candidate = os.path.join(base_dir, relative_part)
                    if os.path.exists(candidate):
                        return candidate

        return os.path.join(base_dir, relative_or_absolute_path)

    def create_progress_callback(self, conversion_id: str) -> Callable:
        """Create a progress callback for the parser."""
        def progress_callback(progress: int, message: str = ""):
            try:
                self.conversion_manager.update_conversion(
                    conversion_id,
                    progress=progress,
                    message=message,
                )
                if self.broadcast_callback:
                    self.broadcast_callback(conversion_id, {
                        "status": "processing",
                        "progress": progress,
                        "message": message,
                    })
                logger.info(f"Conversion {conversion_id}: {progress}% - {message}")
            except Exception as e:
                logger.error(f"Error in parser progress callback: {str(e)}")

        return progress_callback

    def convert_with_doc_service(
        self,
        filename: str,
        file_path: str = None,
        conversion_id: str = None,
        progress_callback: Callable = None
    ) -> list:
        """
        Convert document using doc_service (Word/Excel/TXT).

        Args:
            filename: Original filename
            file_path: Relative file path from database
            conversion_id: Conversion tracking ID
            progress_callback: Progress callback function

        Returns:
            List of result dicts
        """
        if not file_path:
            relative_path = filename
        else:
            relative_path = file_path

        absolute_file_path = self.resolve_file_path(relative_path)
        file_path_obj = Path(absolute_file_path)

        logger.info(f"Starting doc_service conversion for {filename} at {absolute_file_path}")

        if progress_callback:
            progress_callback(10, "Starting document conversion...")

        converter = self.doc_converter_manager.find_converter_for_file(file_path_obj)
        if not converter:
            raise Exception(f"No converter found for file: {filename}")

        if progress_callback:
            progress_callback(30, "Converting document to markdown...")

        # Create output directory structure
        rel_dir = os.path.dirname(relative_path)
        filename_without_ext = file_path_obj.stem

        if rel_dir and rel_dir != '.':
            output_subdir = Path(self.output_dir) / rel_dir / filename_without_ext
        else:
            output_subdir = Path(self.output_dir) / filename_without_ext

        output_subdir.mkdir(parents=True, exist_ok=True)

        output_filename = filename_without_ext + "_nohf.md"
        output_path = output_subdir / output_filename

        success = converter.convert_file(file_path_obj, output_path)

        if not success:
            raise Exception(f"Failed to convert {filename}")

        if progress_callback:
            progress_callback(90, "Conversion complete, finalizing...")

        results = [{
            "page_no": 0,
            "md_content_path": str(output_path),
            "file_path": absolute_file_path,
            "converter_type": "doc_service",
            "converter_name": converter.get_converter_info()["name"]
        }]

        logger.info(f"Doc_service conversion completed successfully for {filename}")
        return results

    def convert_document_background(
        self,
        filename: str,
        prompt_mode: str,
        file_path: str = None,
        conversion_id: str = None,
        progress_callback: Callable = None
    ) -> list:
        """
        Convert document using OCR parser.

        Args:
            filename: Original filename
            prompt_mode: OCR prompt mode
            file_path: Relative file path from database
            conversion_id: Conversion tracking ID
            progress_callback: Progress callback function

        Returns:
            List of result dicts
        """
        if not file_path:
            relative_path = filename
        else:
            relative_path = file_path

        absolute_file_path = self.resolve_file_path(relative_path)

        logger.info(f"Starting OCR conversion for {filename} at {absolute_file_path}")

        rel_dir = os.path.dirname(relative_path)

        if rel_dir and rel_dir != '.':
            output_dir = os.path.join(self.output_dir, rel_dir)
        else:
            output_dir = self.output_dir

        os.makedirs(output_dir, exist_ok=True)

        results = self.parser.parse_file(
            absolute_file_path,
            output_dir=output_dir,
            prompt_mode=prompt_mode,
            progress_callback=progress_callback,
        )

        logger.info(f"OCR conversion completed successfully for {filename}")
        return results

    def convert_with_deepseek_ocr(
        self,
        filename: str,
        file_path: str = None,
        conversion_id: str = None,
        progress_callback: Callable = None
    ) -> list:
        """
        Convert image using DeepSeek OCR.

        Args:
            filename: Original filename
            file_path: Relative file path from database
            conversion_id: Conversion tracking ID
            progress_callback: Progress callback function

        Returns:
            List of result dicts
        """
        if not file_path:
            relative_path = filename
        else:
            relative_path = file_path

        absolute_file_path = self.resolve_file_path(relative_path)
        file_path_obj = Path(absolute_file_path)

        logger.info(f"Starting DeepSeek OCR conversion for {filename}")

        if progress_callback:
            progress_callback(10, "Starting DeepSeek OCR conversion...")

        if not self.deepseek_ocr_converter.is_supported_file(file_path_obj):
            raise Exception(f"File type not supported by DeepSeek OCR: {filename}")

        if progress_callback:
            progress_callback(30, "Converting image to markdown with DeepSeek OCR...")

        rel_dir = os.path.dirname(relative_path)
        filename_without_ext = file_path_obj.stem

        if rel_dir and rel_dir != '.':
            output_subdir = Path(self.output_dir) / rel_dir / filename_without_ext
        else:
            output_subdir = Path(self.output_dir) / filename_without_ext

        output_subdir.mkdir(parents=True, exist_ok=True)

        output_filename = filename_without_ext + "_nohf.md"
        output_path = output_subdir / output_filename

        if progress_callback:
            progress_callback(50, "Calling DeepSeek OCR API...")

        success = self.deepseek_ocr_converter.convert_file(file_path_obj, output_path)

        if not success:
            raise Exception(f"Failed to convert {filename} with DeepSeek OCR")

        if progress_callback:
            progress_callback(90, "Conversion complete, finalizing...")

        results = [{
            "page_no": 0,
            "md_content_path": str(output_path),
            "file_path": str(file_path),
            "converter_type": "deepseek_ocr",
            "converter_name": "DeepSeek OCR"
        }]

        logger.info(f"DeepSeek OCR conversion completed successfully for {filename}")
        return results

    def get_image_analysis_backend(self) -> tuple:
        """
        Get the configured image analysis backend.

        Returns:
            Tuple of (backend_name, converter_instance)
        """
        backend = (os.getenv("IMAGE_ANALYSIS_BACKEND", "qwen3") or "").strip().lower()

        if backend == "deepseek":
            return "deepseek", self.deepseek_ocr_converter
        elif backend == "gemma3" or backend == "gemma":
            return "gemma3", self.gemma3_ocr_converter
        else:  # Default to qwen3
            return "qwen3", self.qwen3_ocr_converter

    def convert_page_directly(
        self,
        jpg_path: Path,
        output_path: Path,
        backend_name: str,
        converter: Any
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
        import base64

        logger.info(f"Converting {jpg_path} using {backend_name} backend")

        if backend_name == "deepseek":
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

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            return markdown_content

    def submit_ocr_task(
        self,
        filename: str,
        file_path: str,
        prompt_mode: str = "prompt_layout_all_en"
    ) -> tuple:
        """
        Submit an OCR conversion task to the worker pool.

        Args:
            filename: Original filename
            file_path: File path (relative or absolute)
            prompt_mode: OCR prompt mode

        Returns:
            Tuple of (success, conversion_id)
        """
        conversion_id = self.conversion_manager.create_conversion(filename)

        self.conversion_manager.update_conversion(
            conversion_id,
            status="processing",
            started_at=datetime.now().isoformat(),
            message="Conversion queued, waiting for worker..."
        )

        progress_callback = self.create_progress_callback(conversion_id)

        success = self.worker_pool.submit_task(
            conversion_id=conversion_id,
            func=self.convert_document_background,
            args=(filename, prompt_mode),
            kwargs={
                "file_path": file_path,
                "conversion_id": conversion_id,
                "progress_callback": progress_callback,
            }
        )

        return success, conversion_id

    def submit_doc_service_task(self, filename: str, file_path: str) -> tuple:
        """
        Submit a doc_service conversion task to the worker pool.

        Args:
            filename: Original filename
            file_path: File path (relative or absolute)

        Returns:
            Tuple of (success, conversion_id)
        """
        conversion_id = self.conversion_manager.create_conversion(filename)

        self.conversion_manager.update_conversion(
            conversion_id,
            status="processing",
            started_at=datetime.now().isoformat(),
            message="Document conversion queued..."
        )

        progress_callback = self.create_progress_callback(conversion_id)

        success = self.worker_pool.submit_task(
            conversion_id=conversion_id,
            func=self.convert_with_doc_service,
            args=(filename,),
            kwargs={
                "file_path": file_path,
                "conversion_id": conversion_id,
                "progress_callback": progress_callback,
            }
        )

        return success, conversion_id

    def submit_deepseek_task(self, filename: str, file_path: str) -> tuple:
        """
        Submit a DeepSeek OCR conversion task to the worker pool.

        Args:
            filename: Original filename
            file_path: File path (relative or absolute)

        Returns:
            Tuple of (success, conversion_id)
        """
        conversion_id = self.conversion_manager.create_conversion(filename)

        self.conversion_manager.update_conversion(
            conversion_id,
            status="processing",
            started_at=datetime.now().isoformat(),
            message="DeepSeek OCR conversion queued..."
        )

        progress_callback = self.create_progress_callback(conversion_id)

        success = self.worker_pool.submit_task(
            conversion_id=conversion_id,
            func=self.convert_with_deepseek_ocr,
            args=(filename,),
            kwargs={
                "file_path": file_path,
                "conversion_id": conversion_id,
                "progress_callback": progress_callback,
            }
        )

        return success, conversion_id

    def get_queue_status(self) -> dict:
        """Get worker pool status."""
        return {
            "queue_size": self.worker_pool.get_queue_size(),
            "active_tasks": self.worker_pool.get_active_tasks_count(),
        }
