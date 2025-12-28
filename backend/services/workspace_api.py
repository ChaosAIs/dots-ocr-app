"""
Workspace API endpoints.
"""
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User, Workspace, Document
from auth.dependencies import get_current_active_user
from services.workspace_service import WorkspaceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workspaces", tags=["Workspaces"])


# ===== Request/Response Models =====

class WorkspaceCreate(BaseModel):
    """Request model for creating a workspace."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    color: str = Field(default='#6366f1', pattern=r'^#[0-9A-Fa-f]{6}$')
    icon: str = Field(default='folder', max_length=50)


class WorkspaceUpdate(BaseModel):
    """Request model for updating a workspace."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    color: Optional[str] = Field(None, pattern=r'^#[0-9A-Fa-f]{6}$')
    icon: Optional[str] = Field(None, max_length=50)


class WorkspaceRename(BaseModel):
    """Request model for renaming a workspace."""
    name: str = Field(..., min_length=1, max_length=100)


class WorkspaceOrder(BaseModel):
    """Request model for a workspace order item."""
    workspace_id: UUID
    order: int


class WorkspaceOrderUpdate(BaseModel):
    """Request model for updating workspace display order."""
    orders: List[WorkspaceOrder]


class WorkspaceResponse(BaseModel):
    """Response model for a workspace."""
    id: UUID
    user_id: UUID
    name: str
    folder_name: str
    folder_path: str
    description: Optional[str]
    color: str
    icon: str
    is_default: bool
    is_system: bool
    document_count: int
    display_order: int
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class DocumentBrief(BaseModel):
    """Brief document info for workspace listing."""
    id: UUID
    filename: str
    original_filename: str
    file_size: int
    mime_type: Optional[str]
    upload_status: Optional[str]
    convert_status: Optional[str]
    index_status: Optional[str]
    created_at: Optional[str]

    class Config:
        from_attributes = True


class WorkspaceWithDocuments(BaseModel):
    """Workspace response with documents."""
    workspace: WorkspaceResponse
    documents: List[DocumentBrief]
    total_documents: int


class MoveDocumentRequest(BaseModel):
    """Request to move a document to another workspace."""
    document_id: UUID
    target_workspace_id: UUID


class MessageResponse(BaseModel):
    """Simple message response."""
    message: str
    success: bool = True


# ===== Helper Functions =====

def workspace_to_response(workspace: Workspace) -> WorkspaceResponse:
    """Convert Workspace model to response."""
    return WorkspaceResponse(
        id=workspace.id,
        user_id=workspace.user_id,
        name=workspace.name,
        folder_name=workspace.folder_name,
        folder_path=workspace.folder_path,
        description=workspace.description,
        color=workspace.color,
        icon=workspace.icon,
        is_default=workspace.is_default,
        is_system=workspace.is_system,
        document_count=workspace.document_count,
        display_order=workspace.display_order,
        created_at=workspace.created_at.isoformat() if workspace.created_at else None,
        updated_at=workspace.updated_at.isoformat() if workspace.updated_at else None,
    )


def document_to_brief(doc: Document) -> DocumentBrief:
    """Convert Document model to brief response."""
    return DocumentBrief(
        id=doc.id,
        filename=doc.filename,
        original_filename=doc.original_filename,
        file_size=doc.file_size,
        mime_type=doc.mime_type,
        upload_status=doc.upload_status.value if doc.upload_status else None,
        convert_status=doc.convert_status.value if doc.convert_status else None,
        index_status=doc.index_status.value if doc.index_status else None,
        created_at=doc.created_at.isoformat() if doc.created_at else None,
    )


# ===== API Endpoints =====

@router.get("", response_model=List[WorkspaceResponse])
def list_workspaces(
    include_system: bool = True,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List all workspaces for the current user.

    Args:
        include_system: Include system workspaces like "Shared With Me"
    """
    service = WorkspaceService(db)
    workspaces = service.get_user_workspaces(
        user_id=current_user.id,
        include_system=include_system
    )

    # Recalculate document counts to ensure accuracy
    for ws in workspaces:
        if not ws.is_system:
            service.workspace_repo.recalculate_document_count(ws.id)

    return [workspace_to_response(ws) for ws in workspaces]


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
def create_workspace(
    request: WorkspaceCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new workspace.

    Creates both the database record and physical folder.
    """
    service = WorkspaceService(db)

    # Check if workspace with same name exists
    existing = service.workspace_repo.get_workspace_by_name(
        user_id=current_user.id,
        name=request.name
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Workspace with name '{request.name}' already exists"
        )

    workspace = service.create_workspace(
        user=current_user,
        name=request.name,
        description=request.description,
        color=request.color,
        icon=request.icon
    )

    logger.info(f"Created workspace '{request.name}' for user {current_user.username}")
    return workspace_to_response(workspace)


@router.get("/default", response_model=WorkspaceResponse)
def get_default_workspace(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get or create the user's default workspace.
    """
    service = WorkspaceService(db)
    workspace = service.get_or_create_default_workspace(current_user)
    return workspace_to_response(workspace)


@router.get("/{workspace_id}", response_model=WorkspaceWithDocuments)
def get_workspace(
    workspace_id: UUID,
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get a workspace with its documents.

    Args:
        workspace_id: Workspace ID
        limit: Maximum number of documents to return
        offset: Number of documents to skip
    """
    service = WorkspaceService(db)
    workspace = service.get_workspace(workspace_id)

    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )

    # Check ownership
    if workspace.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    documents = service.get_workspace_documents(
        workspace_id=workspace_id,
        limit=limit,
        offset=offset
    )

    return WorkspaceWithDocuments(
        workspace=workspace_to_response(workspace),
        documents=[document_to_brief(doc) for doc in documents],
        total_documents=workspace.document_count
    )


@router.put("/{workspace_id}", response_model=WorkspaceResponse)
def update_workspace(
    workspace_id: UUID,
    request: WorkspaceUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update workspace metadata.

    Does not change the folder name. Use rename endpoint for that.
    """
    service = WorkspaceService(db)
    workspace = service.get_workspace(workspace_id)

    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )

    if workspace.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    if workspace.is_system:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot modify system workspace"
        )

    updated = service.update_workspace(
        workspace_id=workspace_id,
        name=request.name,
        description=request.description,
        color=request.color,
        icon=request.icon
    )

    return workspace_to_response(updated)


@router.post("/{workspace_id}/rename", response_model=WorkspaceResponse)
def rename_workspace(
    workspace_id: UUID,
    request: WorkspaceRename,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Rename a workspace.

    This also renames the physical folder and updates document paths.
    """
    service = WorkspaceService(db)
    workspace = service.get_workspace(workspace_id)

    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )

    if workspace.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    if workspace.is_system:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot rename system workspace"
        )

    updated = service.rename_workspace(
        workspace_id=workspace_id,
        new_name=request.name,
        user=current_user
    )

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rename workspace"
        )

    logger.info(f"Renamed workspace {workspace_id} to '{request.name}'")
    return workspace_to_response(updated)


@router.delete("/{workspace_id}", response_model=MessageResponse)
def delete_workspace(
    workspace_id: UUID,
    delete_documents: bool = False,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete a workspace.

    Args:
        workspace_id: Workspace ID
        delete_documents: If True, delete all documents. If False, move to default workspace.

    Cannot delete default or system workspaces.
    """
    service = WorkspaceService(db)
    workspace = service.get_workspace(workspace_id)

    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )

    if workspace.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    if workspace.is_default:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete default workspace"
        )

    if workspace.is_system:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete system workspace"
        )

    # Clean up workspace from user preferences before deleting
    try:
        from db.user_repository import UserRepository
        user_repo = UserRepository(db)
        cleaned_count = user_repo.remove_workspace_from_all_preferences(workspace_id)
        if cleaned_count > 0:
            logger.info(f"Cleaned workspace {workspace_id} from {cleaned_count} user(s)' preferences")
    except Exception as e:
        logger.warning(f"Failed to clean workspace from preferences: {e}")

    success = service.delete_workspace(
        workspace_id=workspace_id,
        delete_documents=delete_documents,
        move_to_default=not delete_documents,
        user=current_user
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete workspace"
        )

    logger.info(f"Deleted workspace {workspace_id}")
    return MessageResponse(message="Workspace deleted successfully")


@router.post("/{workspace_id}/set-default", response_model=MessageResponse)
def set_default_workspace(
    workspace_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Set a workspace as the user's default.
    """
    service = WorkspaceService(db)
    workspace = service.get_workspace(workspace_id)

    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )

    if workspace.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    if workspace.is_system:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot set system workspace as default"
        )

    success = service.set_default_workspace(
        user_id=current_user.id,
        workspace_id=workspace_id
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to set default workspace"
        )

    return MessageResponse(message="Default workspace updated")


@router.post("/reorder", response_model=MessageResponse)
def update_workspace_order(
    request: WorkspaceOrderUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update display order for workspaces.
    """
    service = WorkspaceService(db)

    orders = [
        {"workspace_id": item.workspace_id, "order": item.order}
        for item in request.orders
    ]

    success = service.update_display_order(
        user_id=current_user.id,
        workspace_orders=orders
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update workspace order"
        )

    return MessageResponse(message="Workspace order updated")


@router.post("/move-document", response_model=MessageResponse)
def move_document(
    request: MoveDocumentRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Move a document to another workspace.
    """
    service = WorkspaceService(db)

    # Get the document
    from db.document_repository import DocumentRepository
    doc_repo = DocumentRepository(db)
    document = doc_repo.get_by_id(request.document_id)

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Check document ownership (via owner_id or workspace ownership)
    if document.owner_id and document.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    result = service.move_document_to_workspace(
        document=document,
        target_workspace_id=request.target_workspace_id,
        user=current_user
    )

    if not result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to move document"
        )

    logger.info(f"Moved document {request.document_id} to workspace {request.target_workspace_id}")
    return MessageResponse(message="Document moved successfully")


@router.post("/sync", response_model=dict)
def sync_workspace_folders(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Synchronize workspace folders with database.

    Creates missing folders and reports orphaned folders.
    """
    service = WorkspaceService(db)
    results = service.sync_workspace_folders(current_user)

    return {
        "success": True,
        "created_folders": results["created_folders"],
        "orphaned_folders": results["orphaned_folders"],
        "errors": results["errors"]
    }
