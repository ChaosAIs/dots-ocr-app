"""
Document sharing API endpoints.
"""
import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User, UserDocument, Document, UserRole
from db.user_repository import UserRepository
from auth.dependencies import get_current_active_user
from services.permission_service import PermissionService, PermissionError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sharing", tags=["Document Sharing"])


# ===== Request/Response Models =====

class ShareDocumentRequest(BaseModel):
    """Request to share a document with users."""
    document_id: UUID
    user_ids: List[UUID] = Field(..., min_length=1)
    permissions: List[str] = Field(default=['read'])
    message: Optional[str] = None
    expires_at: Optional[datetime] = None


class ShareDocumentByUsernameRequest(BaseModel):
    """Request to share a document by username/email."""
    document_id: UUID
    usernames: List[str] = Field(..., min_length=1)  # Can be usernames or emails
    permissions: List[str] = Field(default=['read'])
    message: Optional[str] = None
    expires_at: Optional[datetime] = None


class UpdatePermissionsRequest(BaseModel):
    """Request to update user permissions."""
    document_id: UUID
    user_id: UUID
    permissions: List[str]


class RevokeAccessRequest(BaseModel):
    """Request to revoke user access."""
    document_id: UUID
    user_id: UUID


class TransferOwnershipRequest(BaseModel):
    """Request to transfer document ownership."""
    document_id: UUID
    new_owner_id: UUID


class UserDocumentResponse(BaseModel):
    """Response model for a user document permission."""
    id: UUID
    user_id: UUID
    document_id: UUID
    permissions: List[str]
    origin: str
    is_owner: bool
    shared_by: Optional[UUID]
    shared_at: Optional[str]
    expires_at: Optional[str]
    share_message: Optional[str]
    is_new: bool
    last_accessed_at: Optional[str]
    access_count: int
    created_at: Optional[str]

    # Include user info if available
    user_name: Optional[str] = None
    user_email: Optional[str] = None

    class Config:
        from_attributes = True


class SharedDocumentResponse(BaseModel):
    """Response for a shared document with document info."""
    permission: UserDocumentResponse
    document: dict  # Document brief info


class DocumentSharesResponse(BaseModel):
    """Response for document shares info."""
    document_id: UUID
    owner: Optional[UserDocumentResponse]
    shares: List[UserDocumentResponse]
    total_shares: int


class NewSharesResponse(BaseModel):
    """Response for new shares notification."""
    count: int
    shares: List[SharedDocumentResponse]


class MessageResponse(BaseModel):
    """Simple message response."""
    message: str
    success: bool = True


# ===== Helper Functions =====

def user_document_to_response(
    user_doc: UserDocument,
    user: Optional[User] = None
) -> UserDocumentResponse:
    """Convert UserDocument model to response."""
    return UserDocumentResponse(
        id=user_doc.id,
        user_id=user_doc.user_id,
        document_id=user_doc.document_id,
        permissions=user_doc.permissions or [],
        origin=user_doc.origin,
        is_owner=user_doc.is_owner,
        shared_by=user_doc.shared_by,
        shared_at=user_doc.shared_at.isoformat() if user_doc.shared_at else None,
        expires_at=user_doc.expires_at.isoformat() if user_doc.expires_at else None,
        share_message=user_doc.share_message,
        is_new=user_doc.is_new,
        last_accessed_at=user_doc.last_accessed_at.isoformat() if user_doc.last_accessed_at else None,
        access_count=user_doc.access_count,
        created_at=user_doc.created_at.isoformat() if user_doc.created_at else None,
        user_name=user.username if user else None,
        user_email=user.email if user else None,
    )


# ===== API Endpoints =====

@router.post("/share", response_model=MessageResponse)
def share_document(
    request: ShareDocumentRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Share a document with one or more users.

    Permissions can include: read, update, delete, share, full
    """
    service = PermissionService(db)

    # Validate permissions
    valid_permissions = {'read', 'update', 'delete', 'share', 'full'}
    invalid = set(request.permissions) - valid_permissions
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid permissions: {invalid}"
        )

    try:
        results = service.share_with_multiple_users(
            document_id=request.document_id,
            owner=current_user,
            target_user_ids=request.user_ids,
            permissions=request.permissions,
            message=request.message,
            expires_at=request.expires_at
        )

        logger.info(
            f"User {current_user.username} shared document {request.document_id} "
            f"with {len(results)} users"
        )

        return MessageResponse(
            message=f"Document shared with {len(results)} user(s)",
            success=True
        )

    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@router.post("/share-by-username", response_model=MessageResponse)
def share_document_by_username(
    request: ShareDocumentByUsernameRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Share a document with users by username or email.
    """
    service = PermissionService(db)
    user_repo = UserRepository(db)

    # Validate permissions
    valid_permissions = {'read', 'update', 'delete', 'share', 'full'}
    invalid = set(request.permissions) - valid_permissions
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid permissions: {invalid}"
        )

    # Resolve usernames to user IDs
    user_ids = []
    not_found = []
    for identifier in request.usernames:
        user = user_repo.get_user_by_username_or_email(identifier)
        if user:
            user_ids.append(user.id)
        else:
            not_found.append(identifier)

    if not user_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No users found: {not_found}"
        )

    try:
        results = service.share_with_multiple_users(
            document_id=request.document_id,
            owner=current_user,
            target_user_ids=user_ids,
            permissions=request.permissions,
            message=request.message,
            expires_at=request.expires_at
        )

        message = f"Document shared with {len(results)} user(s)"
        if not_found:
            message += f". Users not found: {not_found}"

        return MessageResponse(message=message, success=True)

    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@router.get("/shared-with-me", response_model=List[SharedDocumentResponse])
def get_shared_with_me(
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get documents shared with the current user.
    """
    service = PermissionService(db)

    shares = service.get_shared_with_me(
        user=current_user,
        limit=limit,
        offset=offset
    )

    results = []
    for share in shares:
        doc_dict = share.document.to_dict() if share.document else {}
        results.append(SharedDocumentResponse(
            permission=user_document_to_response(share),
            document=doc_dict
        ))

    return results


@router.get("/new-shares", response_model=NewSharesResponse)
def get_new_shares(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get new (unviewed) shared documents.
    """
    service = PermissionService(db)

    count = service.get_new_shares_count(current_user)
    shares = service.get_new_shares(current_user)

    results = []
    for share in shares:
        doc_dict = share.document.to_dict() if share.document else {}
        results.append(SharedDocumentResponse(
            permission=user_document_to_response(share),
            document=doc_dict
        ))

    return NewSharesResponse(count=count, shares=results)


@router.post("/mark-viewed/{document_id}", response_model=MessageResponse)
def mark_as_viewed(
    document_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Mark a shared document as viewed.
    """
    service = PermissionService(db)
    result = service.mark_as_viewed(current_user, document_id)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document permission not found"
        )

    return MessageResponse(message="Document marked as viewed")


@router.post("/mark-all-viewed", response_model=MessageResponse)
def mark_all_as_viewed(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Mark all shared documents as viewed.
    """
    service = PermissionService(db)
    count = service.mark_all_as_viewed(current_user)

    return MessageResponse(message=f"Marked {count} document(s) as viewed")


@router.get("/document/{document_id}/shares", response_model=DocumentSharesResponse)
def get_document_shares(
    document_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get all users with access to a document.

    Only the document owner or admin can view this.
    """
    service = PermissionService(db)
    user_repo = UserRepository(db)

    try:
        permissions = service.get_document_users(
            document_id=document_id,
            requesting_user=current_user
        )

        owner = None
        shares = []

        for perm in permissions:
            user = user_repo.get_user_by_id(perm.user_id)
            response = user_document_to_response(perm, user)
            if perm.is_owner:
                owner = response
            else:
                shares.append(response)

        return DocumentSharesResponse(
            document_id=document_id,
            owner=owner,
            shares=shares,
            total_shares=len(shares)
        )

    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@router.put("/update-permissions", response_model=UserDocumentResponse)
def update_permissions(
    request: UpdatePermissionsRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update a user's permissions on a document.

    Only the document owner or admin can do this.
    """
    service = PermissionService(db)

    # Validate permissions
    valid_permissions = {'read', 'update', 'delete', 'share', 'full'}
    invalid = set(request.permissions) - valid_permissions
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid permissions: {invalid}"
        )

    try:
        result = service.update_permissions(
            document_id=request.document_id,
            owner=current_user,
            target_user_id=request.user_id,
            permissions=request.permissions
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Permission record not found"
            )

        return user_document_to_response(result)

    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@router.post("/revoke", response_model=MessageResponse)
def revoke_access(
    request: RevokeAccessRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Revoke a user's access to a document.

    Only the document owner or admin can do this.
    Cannot revoke owner's access.
    """
    service = PermissionService(db)

    try:
        success = service.revoke_access(
            document_id=request.document_id,
            owner=current_user,
            target_user_id=request.user_id
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to revoke access. Cannot revoke owner's access."
            )

        logger.info(
            f"User {current_user.username} revoked access for user {request.user_id} "
            f"on document {request.document_id}"
        )

        return MessageResponse(message="Access revoked successfully")

    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@router.post("/transfer-ownership", response_model=MessageResponse)
def transfer_ownership(
    request: TransferOwnershipRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Transfer document ownership to another user.

    Current owner becomes a shared user with full access.
    """
    service = PermissionService(db)

    try:
        success = service.transfer_ownership(
            document_id=request.document_id,
            current_owner=current_user,
            new_owner_id=request.new_owner_id
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to transfer ownership"
            )

        logger.info(
            f"User {current_user.username} transferred ownership of document "
            f"{request.document_id} to user {request.new_owner_id}"
        )

        return MessageResponse(message="Ownership transferred successfully")

    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@router.get("/check-access/{document_id}", response_model=dict)
def check_access(
    document_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Check current user's access to a document.

    Returns permissions and access level.
    """
    service = PermissionService(db)

    has_access = service.has_any_access(current_user, document_id)

    return {
        "has_access": has_access,
        "is_admin": current_user.role == UserRole.ADMIN,
        "is_owner": service.is_owner(current_user, document_id) if has_access else False,
        "can_read": service.can_read(current_user, document_id),
        "can_update": service.can_update(current_user, document_id),
        "can_delete": service.can_delete(current_user, document_id),
        "can_share": service.can_share(current_user, document_id),
    }


# ===== Admin Endpoints =====

@router.post("/admin/grant-access", response_model=UserDocumentResponse)
def admin_grant_access(
    document_id: UUID,
    user_id: UUID,
    permissions: List[str] = ['read'],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Admin grants access to a document.

    Admin-only endpoint.
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    service = PermissionService(db)

    # Validate permissions
    valid_permissions = {'read', 'update', 'delete', 'share', 'full'}
    invalid = set(permissions) - valid_permissions
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid permissions: {invalid}"
        )

    result = service.admin_grant_access(
        document_id=document_id,
        admin=current_user,
        target_user_id=user_id,
        permissions=permissions
    )

    logger.info(
        f"Admin {current_user.username} granted access to document "
        f"{document_id} for user {user_id}"
    )

    return user_document_to_response(result)


@router.post("/admin/cleanup-expired", response_model=MessageResponse)
def cleanup_expired_permissions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Clean up expired permission records.

    Admin-only endpoint.
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    service = PermissionService(db)
    count = service.cleanup_expired_permissions()

    return MessageResponse(message=f"Cleaned up {count} expired permission(s)")
