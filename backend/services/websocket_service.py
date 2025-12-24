"""
WebSocket service for real-time communication.
"""
import asyncio
import logging
import threading
from datetime import datetime
from typing import Dict, Set, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class DocumentStatusManager:
    """
    Manages centralized WebSocket connections for all document status updates
    with subscription support.
    """

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[WebSocket, Set[str]] = {}  # websocket -> set of document_ids
        self.lock = threading.Lock()
        self._message_queue: asyncio.Queue = None
        self._main_loop = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the main event loop for broadcasting messages from worker threads."""
        self._main_loop = loop
        self._message_queue = asyncio.Queue()

    async def start_broadcast_worker(self):
        """Start the background worker that processes broadcast messages."""
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
        """Register a new WebSocket connection."""
        await websocket.accept()
        with self.lock:
            self.active_connections.add(websocket)
            self.subscriptions[websocket] = set()
        logger.info(f"Document status WebSocket connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        """Unregister a WebSocket connection."""
        with self.lock:
            self.active_connections.discard(websocket)
            self.subscriptions.pop(websocket, None)
        logger.info(f"Document status WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def subscribe(self, websocket: WebSocket, document_ids: list):
        """Subscribe a WebSocket to specific document updates."""
        valid_ids = [doc_id for doc_id in document_ids if doc_id]
        if not valid_ids:
            logger.warning(f"WebSocket subscribe called with no valid document IDs (received: {document_ids})")
            return

        with self.lock:
            if websocket not in self.subscriptions:
                self.subscriptions[websocket] = set()
            self.subscriptions[websocket].update(valid_ids)
        logger.info(f"WebSocket subscribed to {len(valid_ids)} documents. Total subscriptions: {len(self.subscriptions[websocket])}")

    async def unsubscribe(self, websocket: WebSocket, document_ids: list):
        """Unsubscribe a WebSocket from specific document updates."""
        valid_ids = [doc_id for doc_id in document_ids if doc_id]
        if not valid_ids:
            logger.warning(f"WebSocket unsubscribe called with no valid document IDs (received: {document_ids})")
            return

        with self.lock:
            if websocket in self.subscriptions:
                self.subscriptions[websocket].difference_update(valid_ids)
        logger.info(f"WebSocket unsubscribed from {len(valid_ids)} documents")

    async def _do_broadcast(self, message: dict):
        """Send message only to clients subscribed to the document."""
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
        """Send message to all connected clients (async context)."""
        await self._do_broadcast(message)

    def broadcast_from_thread(self, message: dict):
        """Thread-safe method to broadcast message from worker threads."""
        if self._main_loop and self._message_queue:
            logger.info(f"📤 Broadcasting document status: {message.get('event_type')} for {message.get('filename', message.get('document_id'))}")
            asyncio.run_coroutine_threadsafe(
                self._message_queue.put(message),
                self._main_loop
            )
        else:
            logger.error(f"❌ Cannot broadcast document status - event loop not initialized! main_loop={self._main_loop is not None}, message_queue={self._message_queue is not None}")


class ConnectionManager:
    """Manages WebSocket connections for conversion progress updates."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}  # conversion_id -> set of websockets
        self.lock = threading.Lock()
        self._message_queue: asyncio.Queue = None
        self._broadcast_task = None
        self._main_loop = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the main event loop for broadcasting messages from worker threads."""
        self._main_loop = loop
        self._message_queue = asyncio.Queue()

    async def start_broadcast_worker(self):
        """Start the background worker that processes broadcast messages."""
        while True:
            try:
                conversion_id, message = await self._message_queue.get()
                await self._do_broadcast(conversion_id, message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in broadcast worker: {e}")

    async def connect(self, conversion_id: str, websocket: WebSocket):
        """Register a new WebSocket connection."""
        await websocket.accept()
        with self.lock:
            if conversion_id not in self.active_connections:
                self.active_connections[conversion_id] = set()
            self.active_connections[conversion_id].add(websocket)

    async def disconnect(self, conversion_id: str, websocket: WebSocket):
        """Unregister a WebSocket connection."""
        with self.lock:
            if conversion_id in self.active_connections:
                self.active_connections[conversion_id].discard(websocket)
                if not self.active_connections[conversion_id]:
                    del self.active_connections[conversion_id]

    async def _do_broadcast(self, conversion_id: str, message: dict):
        """Actually send message to all connected clients for a conversion."""
        with self.lock:
            connections = self.active_connections.get(conversion_id, set()).copy()

        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message: {e}")

    async def broadcast(self, conversion_id: str, message: dict):
        """Send message to all connected clients for a conversion (async context)."""
        await self._do_broadcast(conversion_id, message)

    def broadcast_from_thread(self, conversion_id: str, message: dict):
        """Thread-safe method to broadcast message from worker threads."""
        if self._main_loop and self._message_queue:
            asyncio.run_coroutine_threadsafe(
                self._message_queue.put((conversion_id, message)),
                self._main_loop
            )


class WebSocketService:
    """
    Unified service for WebSocket management.
    Combines ConnectionManager and DocumentStatusManager functionality.
    """

    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.document_status_manager = DocumentStatusManager()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop for both managers."""
        self.connection_manager.set_event_loop(loop)
        self.document_status_manager.set_event_loop(loop)

    async def start_broadcast_workers(self):
        """Start background broadcast workers for both managers."""
        await asyncio.gather(
            self.connection_manager.start_broadcast_worker(),
            self.document_status_manager.start_broadcast_worker()
        )

    def create_dual_broadcast_callback(
        self,
        doc_id: str,
        filename: str
    ):
        """
        Create a callback that broadcasts to both WebSocket managers.

        Args:
            doc_id: Document ID for document status broadcasts
            filename: Filename for logging

        Returns:
            Callable that takes (conversion_id, message)
        """
        def dual_broadcast_callback(conv_id: str, message: dict):
            """Broadcast to both conversion WebSocket and document status WebSocket."""
            # Send to conversion WebSocket (for progress bars)
            self.connection_manager.broadcast_from_thread(conv_id, message)

            # Also send to document status WebSocket when indexing completes
            if doc_id:
                status = message.get("status", "")
                # Completion statuses: vector_indexed, metadata_extracted, graphrag_indexed
                if status in ["vector_indexed", "metadata_extracted", "graphrag_indexed", "index_error"]:
                    event_type = "indexing_completed" if status in ["vector_indexed", "metadata_extracted", "graphrag_indexed"] else "indexing_failed"
                    logger.info(f"📤 Broadcasting document status: {event_type} for {filename} (status={status})")
                    self.document_status_manager.broadcast_from_thread({
                        "event_type": event_type,
                        "document_id": doc_id,
                        "filename": filename,
                        "index_status": "indexed" if status in ["vector_indexed", "metadata_extracted", "graphrag_indexed"] else "failed",
                        "progress": 100,
                        "message": message.get("message", "Indexing completed"),
                        "timestamp": datetime.now().isoformat()
                    })

        return dual_broadcast_callback
