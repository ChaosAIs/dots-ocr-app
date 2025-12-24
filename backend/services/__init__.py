"""
Services package for the Dots OCR application.

This package contains service classes that encapsulate business logic:
- DocumentService: File path resolution, markdown operations, document management
- ConversionService: OCR and document format conversion
- ConversionManager: Conversion task status tracking
- WebSocketService: Real-time WebSocket communication
- ConnectionManager: WebSocket connections for conversion progress
- DocumentStatusManager: WebSocket connections for document status updates
- IndexingService: Vector and GraphRAG indexing operations
- TaskQueueService: Hierarchical task queue management
- WorkspaceService: Workspace management with file system sync
- PermissionService: Document access control
"""

from services.document_service import DocumentService
from services.conversion_service import ConversionService, ConversionManager
from services.websocket_service import (
    WebSocketService,
    ConnectionManager,
    DocumentStatusManager
)
from services.indexing_service import IndexingService
from services.task_queue_service import TaskQueueService
from services.workspace_service import WorkspaceService
from services.permission_service import PermissionService

__all__ = [
    # Document operations
    'DocumentService',

    # Conversion operations
    'ConversionService',
    'ConversionManager',

    # WebSocket management
    'WebSocketService',
    'ConnectionManager',
    'DocumentStatusManager',

    # Indexing operations
    'IndexingService',

    # Task queue management
    'TaskQueueService',

    # Workspace management
    'WorkspaceService',

    # Permission management
    'PermissionService',
]
