"""
Progress tracking for document conversion tasks.
Provides callbacks and utilities for tracking conversion progress.
"""

import asyncio
from typing import Callable, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks progress of a conversion task and broadcasts updates"""

    def __init__(self, conversion_id: str, broadcast_callback: Optional[Callable] = None):
        """
        Initialize progress tracker.
        
        Args:
            conversion_id: Unique ID for this conversion
            broadcast_callback: Async callback to broadcast progress updates
        """
        self.conversion_id = conversion_id
        self.broadcast_callback = broadcast_callback
        self.current_progress = 0
        self.total_steps = 100
        self.current_message = ""

    async def update(self, progress: int, message: str = "", step_name: str = ""):
        """
        Update progress and broadcast if callback is set.
        
        Args:
            progress: Progress percentage (0-100)
            message: Status message
            step_name: Name of current step
        """
        self.current_progress = min(100, max(0, progress))
        self.current_message = message or step_name

        if self.broadcast_callback:
            try:
                await self.broadcast_callback(self.conversion_id, {
                    "status": "processing",
                    "progress": self.current_progress,
                    "message": self.current_message,
                    "step": step_name,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                logger.error(f"Error broadcasting progress: {str(e)}")

    def get_progress(self) -> dict:
        """Get current progress state"""
        return {
            "progress": self.current_progress,
            "message": self.current_message,
        }


class ConversionProgressCallback:
    """Callback handler for conversion progress updates"""

    def __init__(self, conversion_id: str, broadcast_callback: Optional[Callable] = None):
        """
        Initialize callback handler.
        
        Args:
            conversion_id: Unique ID for this conversion
            broadcast_callback: Async callback to broadcast updates
        """
        self.conversion_id = conversion_id
        self.broadcast_callback = broadcast_callback
        self.tracker = ProgressTracker(conversion_id, broadcast_callback)

    async def on_pdf_load_start(self, filename: str, total_pages: int):
        """Called when PDF loading starts"""
        await self.tracker.update(
            progress=5,
            message=f"Loading PDF: {filename} ({total_pages} pages)",
            step_name="pdf_load"
        )

    async def on_pdf_load_complete(self, total_pages: int):
        """Called when PDF loading completes"""
        await self.tracker.update(
            progress=10,
            message=f"PDF loaded: {total_pages} pages ready for processing",
            step_name="pdf_load_complete"
        )

    async def on_page_processing_start(self, page_num: int, total_pages: int):
        """Called when page processing starts"""
        progress = 10 + int((page_num / total_pages) * 80)
        await self.tracker.update(
            progress=progress,
            message=f"Processing page {page_num + 1}/{total_pages}",
            step_name="page_processing"
        )

    async def on_page_processing_complete(self, page_num: int, total_pages: int):
        """Called when page processing completes"""
        progress = 10 + int(((page_num + 1) / total_pages) * 80)
        await self.tracker.update(
            progress=progress,
            message=f"Completed page {page_num + 1}/{total_pages}",
            step_name="page_complete"
        )

    async def on_markdown_generation_start(self):
        """Called when markdown generation starts"""
        await self.tracker.update(
            progress=90,
            message="Generating markdown output",
            step_name="markdown_generation"
        )

    async def on_markdown_generation_complete(self):
        """Called when markdown generation completes"""
        await self.tracker.update(
            progress=95,
            message="Markdown generation complete",
            step_name="markdown_complete"
        )

    async def on_finalization_start(self):
        """Called when finalization starts"""
        await self.tracker.update(
            progress=97,
            message="Finalizing conversion",
            step_name="finalization"
        )

    def get_progress(self) -> dict:
        """Get current progress"""
        return self.tracker.get_progress()

