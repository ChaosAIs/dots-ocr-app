"""
Worker pool for managing concurrent document conversion tasks.
Implements a thread pool-based worker queue with progress tracking.
"""

import threading
import queue
import asyncio
from typing import Callable, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ConversionWorker(threading.Thread):
    """Worker thread that processes conversion tasks from a queue"""

    def __init__(self, worker_id: int, task_queue: queue.Queue, progress_callback: Callable):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.progress_callback = progress_callback
        self.running = True

    def run(self):
        """Main worker loop - processes tasks from queue"""
        while self.running:
            try:
                # Get task with timeout to allow graceful shutdown
                task = self.task_queue.get(timeout=1)
                if task is None:  # Poison pill for shutdown
                    break

                conversion_id = task["conversion_id"]
                task_func = task["func"]
                task_args = task["args"]
                task_kwargs = task["kwargs"]

                logger.info(f"Worker {self.worker_id} processing conversion {conversion_id}")

                try:
                    # Execute the task
                    result = task_func(*task_args, **task_kwargs)
                    
                    # Notify completion
                    self.progress_callback(
                        conversion_id=conversion_id,
                        status="completed",
                        result=result
                    )
                except Exception as e:
                    logger.error(f"Error in worker {self.worker_id}: {str(e)}")
                    self.progress_callback(
                        conversion_id=conversion_id,
                        status="error",
                        error=str(e)
                    )
                finally:
                    self.task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {str(e)}")

    def stop(self):
        """Stop the worker gracefully"""
        self.running = False


class WorkerPool:
    """Thread pool for managing concurrent conversion tasks"""

    def __init__(self, num_workers: int = 4, progress_callback: Optional[Callable] = None):
        """
        Initialize the worker pool.
        
        Args:
            num_workers: Number of worker threads to create
            progress_callback: Callback function for progress updates
        """
        self.num_workers = num_workers
        self.task_queue: queue.Queue = queue.Queue()
        self.workers: list[ConversionWorker] = []
        self.progress_callback = progress_callback or self._default_callback
        self.active_tasks: Dict[str, dict] = {}
        self.lock = threading.Lock()

        # Start worker threads
        for i in range(num_workers):
            worker = ConversionWorker(i, self.task_queue, self._handle_progress)
            worker.start()
            self.workers.append(worker)

        logger.info(f"WorkerPool initialized with {num_workers} workers")

    def submit_task(
        self,
        conversion_id: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None
    ) -> bool:
        """
        Submit a task to the worker pool.
        
        Args:
            conversion_id: Unique ID for this conversion
            func: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            
        Returns:
            True if task was submitted, False otherwise
        """
        if kwargs is None:
            kwargs = {}

        with self.lock:
            if conversion_id in self.active_tasks:
                logger.warning(f"Conversion {conversion_id} already in progress")
                return False

            self.active_tasks[conversion_id] = {
                "status": "queued",
                "submitted_at": datetime.now().isoformat(),
            }

        task = {
            "conversion_id": conversion_id,
            "func": func,
            "args": args,
            "kwargs": kwargs,
        }

        self.task_queue.put(task)
        logger.info(f"Task {conversion_id} submitted to worker pool")
        return True

    def _handle_progress(self, conversion_id: str, status: str, result: Any = None, error: str = None):
        """Handle progress updates from workers"""
        with self.lock:
            if conversion_id in self.active_tasks:
                self.active_tasks[conversion_id]["status"] = status
                if result is not None:
                    self.active_tasks[conversion_id]["result"] = result
                if error is not None:
                    self.active_tasks[conversion_id]["error"] = error

        # Call the user-provided callback
        if self.progress_callback:
            try:
                self.progress_callback(
                    conversion_id=conversion_id,
                    status=status,
                    result=result,
                    error=error
                )
            except Exception as e:
                logger.error(f"Error in progress callback: {str(e)}")

    def _default_callback(self, conversion_id: str, status: str, result: Any = None, error: str = None):
        """Default progress callback"""
        logger.info(f"Conversion {conversion_id}: {status}")

    def get_task_status(self, conversion_id: str) -> Optional[dict]:
        """Get the status of a task"""
        with self.lock:
            return self.active_tasks.get(conversion_id)

    def shutdown(self, wait: bool = True):
        """Shutdown the worker pool"""
        logger.info("Shutting down worker pool...")
        
        # Send poison pills to stop workers
        for _ in self.workers:
            self.task_queue.put(None)

        if wait:
            # Wait for all workers to finish
            for worker in self.workers:
                worker.join(timeout=5)

        logger.info("Worker pool shutdown complete")

    def get_queue_size(self) -> int:
        """Get the number of tasks in the queue"""
        return self.task_queue.qsize()

    def get_active_tasks_count(self) -> int:
        """Get the number of active tasks"""
        with self.lock:
            return len(self.active_tasks)

