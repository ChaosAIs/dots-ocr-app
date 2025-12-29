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


class SaveMarkdownRequest(BaseModel):
    """Request model for saving markdown content to workspace."""
    content: str = Field(..., min_length=1)
    filename: str = Field(..., min_length=1, max_length=255)
    workspace_id: UUID


class SaveMarkdownResponse(BaseModel):
    """Response model for save markdown operation."""
    success: bool
    message: str
    document_id: Optional[UUID] = None
    filename: str


@router.post("/save-markdown", response_model=SaveMarkdownResponse)
def save_markdown_to_workspace(
    request: SaveMarkdownRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Save markdown content as a file to a workspace.

    This endpoint:
    1. Creates markdown file directly in the output folder (skipping OCR)
    2. Creates proper folder structure: output/{username}/{workspace}/{doc_name}/
    3. Creates both {name}.md and {name}_nohf.md files
    4. Registers document in database with OCR status = SKIPPED
    5. Creates task queue page entry with OCR already completed (SKIPPED)
    6. Triggers indexing process (vector embedding, GraphRAG, data extraction)
    """
    import os
    import re
    import threading
    from datetime import datetime, timezone
    from db.document_repository import DocumentRepository
    from services.permission_service import PermissionService
    from db.models import UploadStatus, ConvertStatus, IndexStatus, TaskStatus

    # Get the workspace
    service = WorkspaceService(db)
    workspace = service.get_workspace(request.workspace_id)

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

    if workspace.is_system:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot save files to system workspace"
        )

    # Sanitize filename - remove extension if present, we'll add .md
    import re
    base_name = re.sub(r'[<>:"/\\|?*]', '_', request.filename)
    # Remove .md extension if present (we'll add it properly)
    if base_name.endswith('.md'):
        base_name = base_name[:-3]

    # Get directory paths
    input_dir = os.environ.get('INPUT_DIR', './input')
    output_dir = os.environ.get('OUTPUT_DIR', './output')

    # Build paths following the pattern: output/{username}/{workspace_folder}/{doc_name}/
    # workspace.folder_path is like "username/workspace_folder"
    workspace_output_path = os.path.join(output_dir, workspace.folder_path)

    # Create unique document folder name
    doc_folder_name = base_name
    doc_output_path = os.path.join(workspace_output_path, doc_folder_name)

    # Make folder name unique if it already exists
    if os.path.exists(doc_output_path):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        doc_folder_name = f"{base_name}_{timestamp}"
        doc_output_path = os.path.join(workspace_output_path, doc_folder_name)

    # Create the output folder
    os.makedirs(doc_output_path, exist_ok=True)

    # Validate path to prevent directory traversal
    if not os.path.abspath(doc_output_path).startswith(os.path.abspath(output_dir)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path"
        )

    # File paths
    md_filename = f"{doc_folder_name}.md"
    nohf_filename = f"{doc_folder_name}_nohf.md"
    md_path = os.path.join(doc_output_path, md_filename)
    nohf_path = os.path.join(doc_output_path, nohf_filename)

    # Also save to input folder for consistency with upload flow
    input_workspace_path = os.path.join(input_dir, workspace.folder_path)
    os.makedirs(input_workspace_path, exist_ok=True)
    input_file_path = os.path.join(input_workspace_path, md_filename)

    try:
        # Write markdown content to all locations
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(request.content)

        with open(nohf_path, 'w', encoding='utf-8') as f:
            f.write(request.content)

        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(request.content)

        file_size = os.path.getsize(md_path)
        relative_input_path = os.path.join(workspace.folder_path, md_filename)
        relative_output_path = os.path.join(workspace.folder_path, doc_folder_name)

        # Create document record in database
        doc_repo = DocumentRepository(db)
        doc, created = doc_repo.get_or_create(
            filename=md_filename,
            original_filename=md_filename,
            file_path=relative_input_path,
            file_size=file_size,
        )

        # Set document properties - mark as already converted (skip OCR)
        doc.workspace_id = workspace.id
        doc.owner_id = current_user.id
        doc.visibility = 'private'
        doc.total_pages = 1
        doc.mime_type = 'text/markdown'
        doc.output_path = relative_output_path
        doc.upload_status = UploadStatus.UPLOADED
        doc.convert_status = ConvertStatus.CONVERTED
        doc.index_status = IndexStatus.PENDING  # Ready for indexing
        # Set OCR status to SKIPPED since markdown files bypass OCR
        doc.ocr_status = TaskStatus.SKIPPED
        doc.ocr_completed_at = datetime.now(timezone.utc)

        db.commit()

        # Grant owner permission
        perm_service = PermissionService(db)
        perm_service.grant_owner_permission(user_id=current_user.id, document_id=doc.id)

        # Update workspace document count
        service.workspace_repo.recalculate_document_count(workspace.id)

        # Create task queue page entry with OCR already completed (SKIPPED)
        # This allows vector/graphrag workers to pick up the document
        try:
            from queue_service.models import TaskQueuePage

            # Relative path for page_file_path (relative to output_dir)
            relative_nohf_path = os.path.join(relative_output_path, nohf_filename)

            # Check if page already exists
            existing_page = db.query(TaskQueuePage).filter(
                TaskQueuePage.document_id == doc.id,
                TaskQueuePage.page_number == 0
            ).first()

            if not existing_page:
                page_task = TaskQueuePage(
                    document_id=doc.id,
                    page_number=0,
                    page_file_path=relative_nohf_path,
                    # OCR is SKIPPED (intentionally bypassed for markdown files)
                    ocr_status=TaskStatus.SKIPPED,
                    ocr_completed_at=datetime.now(timezone.utc),
                    # Vector and GraphRAG are pending, ready to be picked up by workers
                    vector_status=TaskStatus.PENDING,
                    graphrag_status=TaskStatus.PENDING,
                )
                db.add(page_task)
                db.commit()
                logger.info(f"Created task queue page entry for markdown document: {md_filename} (OCR=SKIPPED)")
            else:
                logger.info(f"Task queue page entry already exists for: {md_filename}")
        except Exception as e:
            logger.warning(f"Could not create task queue page entry: {e}")
            # Continue - the document is still saved, just may not be picked up by queue workers

        logger.info(f"Saved markdown file '{md_filename}' to workspace '{workspace.name}' for user {current_user.username}")
        logger.info(f"Output path: {doc_output_path}")

        # Trigger indexing in background thread
        doc_id = doc.id
        def trigger_indexing():
            try:
                # Import here to avoid circular imports
                from rag_service.indexer import trigger_embedding_for_document

                logger.info(f"Triggering indexing for markdown document: {md_filename}")
                # source_name must be the full relative path from output_dir
                # e.g., "fyang/test_copy_markdown/Meal Receipt Details for 2025"
                trigger_embedding_for_document(
                    source_name=relative_output_path,
                    output_dir=output_dir,
                    filename=md_filename,
                    conversion_id=f"md-{doc_id}",
                    broadcast_callback=None
                )
                logger.info(f"Indexing triggered successfully for: {md_filename}")
            except Exception as e:
                logger.error(f"Failed to trigger indexing for {md_filename}: {e}")

        # Start indexing in background
        indexing_thread = threading.Thread(target=trigger_indexing, daemon=True)
        indexing_thread.start()

        return SaveMarkdownResponse(
            success=True,
            message="Markdown saved successfully. Indexing started.",
            document_id=doc.id,
            filename=md_filename
        )

    except Exception as e:
        logger.error(f"Failed to save markdown to workspace: {e}")
        # Clean up files if they were created
        for path in [md_path, nohf_path, input_file_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        # Clean up folder if empty
        if os.path.exists(doc_output_path):
            try:
                os.rmdir(doc_output_path)
            except:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save markdown: {str(e)}"
        )
