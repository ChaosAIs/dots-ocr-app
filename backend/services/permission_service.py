"""
Permission service for document access control.
"""
import logging
from datetime import datetime
from typing import Optional, List, Set
from uuid import UUID

from sqlalchemy.orm import Session

from db.models import (
    User, UserRole, Document, UserDocument,
    DocumentPermission, PermissionOrigin
)
from db.user_document_repository import UserDocumentRepository
from db.workspace_repository import WorkspaceRepository

logger = logging.getLogger(__name__)


class PermissionError(Exception):
    """Exception raised for permission-related errors."""
    pass


class PermissionService:
    """
    Service for managing document permissions and access control.

    Permission Hierarchy:
    1. Admin role - full access to all documents
    2. Document owner - full access to owned documents
    3. Shared permissions - specific permissions granted by owner
    4. Public visibility - read access for all users

    Permission Types:
    - read: View document content
    - update: Edit document (metadata, etc.)
    - delete: Delete document
    - share: Share document with others
    - full: All permissions
    """

    def __init__(self, db: Session):
        """Initialize permission service."""
        self.db = db
        self.user_doc_repo = UserDocumentRepository(db)
        self.workspace_repo = WorkspaceRepository(db)

    def check_permission(
        self,
        user: User,
        document_id: UUID,
        required_permission: str
    ) -> bool:
        """
        Check if a user has a specific permission on a document.

        Args:
            user: User requesting access
            document_id: Document ID
            required_permission: Permission to check (read, update, delete, share, full)

        Returns:
            True if user has permission
        """
        # Admin bypass
        if user.role == UserRole.ADMIN:
            return True

        return self.user_doc_repo.check_permission(
            user_id=user.id,
            document_id=document_id,
            required_permission=required_permission,
            user_role=user.role
        )

    def require_permission(
        self,
        user: User,
        document_id: UUID,
        required_permission: str
    ) -> None:
        """
        Require a permission, raising PermissionError if not granted.

        Args:
            user: User requesting access
            document_id: Document ID
            required_permission: Permission required

        Raises:
            PermissionError: If user doesn't have permission
        """
        if not self.check_permission(user, document_id, required_permission):
            raise PermissionError(
                f"User {user.id} does not have '{required_permission}' permission on document {document_id}"
            )

    def can_read(self, user: User, document_id: UUID) -> bool:
        """Check if user can read a document."""
        return self.check_permission(user, document_id, 'read')

    def can_update(self, user: User, document_id: UUID) -> bool:
        """Check if user can update a document."""
        return self.check_permission(user, document_id, 'update')

    def can_delete(self, user: User, document_id: UUID) -> bool:
        """Check if user can delete a document."""
        return self.check_permission(user, document_id, 'delete')

    def can_share(self, user: User, document_id: UUID) -> bool:
        """Check if user can share a document."""
        return self.check_permission(user, document_id, 'share')

    def has_any_access(self, user: User, document_id: UUID) -> bool:
        """Check if user has any access to a document."""
        return self.user_doc_repo.has_any_access(
            user_id=user.id,
            document_id=document_id,
            user_role=user.role
        )

    def is_owner(self, user: User, document_id: UUID) -> bool:
        """Check if user is the owner of a document."""
        if user.role == UserRole.ADMIN:
            return False  # Admin is not considered owner unless they actually are

        user_doc = self.user_doc_repo.get_user_document(user.id, document_id)
        return user_doc is not None and user_doc.is_owner

    def grant_owner_permission(
        self,
        user_id: UUID,
        document_id: UUID
    ) -> UserDocument:
        """
        Grant owner permission when a user uploads a document.

        Args:
            user_id: User ID (the uploader)
            document_id: Document ID

        Returns:
            UserDocument permission record
        """
        return self.user_doc_repo.create_owner_permission(
            user_id=user_id,
            document_id=document_id
        )

    def share_document(
        self,
        document_id: UUID,
        owner: User,
        target_user_id: UUID,
        permissions: List[str] = None,
        message: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> UserDocument:
        """
        Share a document with another user.

        Args:
            document_id: Document to share
            owner: User sharing the document
            target_user_id: User to share with
            permissions: Permissions to grant
            message: Optional message
            expires_at: Optional expiration

        Returns:
            UserDocument permission record

        Raises:
            PermissionError: If user cannot share the document
        """
        # Check if user can share
        if not self.can_share(owner, document_id):
            raise PermissionError(
                f"User {owner.id} cannot share document {document_id}"
            )

        # Cannot share with yourself
        if target_user_id == owner.id:
            raise PermissionError("Cannot share document with yourself")

        return self.user_doc_repo.share_document(
            document_id=document_id,
            target_user_id=target_user_id,
            shared_by_user_id=owner.id,
            permissions=permissions or ['read'],
            message=message,
            expires_at=expires_at
        )

    def share_with_multiple_users(
        self,
        document_id: UUID,
        owner: User,
        target_user_ids: List[UUID],
        permissions: List[str] = None,
        message: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> List[UserDocument]:
        """
        Share a document with multiple users.

        Returns:
            List of UserDocument permission records
        """
        results = []
        for user_id in target_user_ids:
            if user_id != owner.id:
                try:
                    result = self.share_document(
                        document_id=document_id,
                        owner=owner,
                        target_user_id=user_id,
                        permissions=permissions,
                        message=message,
                        expires_at=expires_at
                    )
                    results.append(result)
                except PermissionError as e:
                    logger.warning(f"Failed to share with user {user_id}: {e}")
        return results

    def revoke_access(
        self,
        document_id: UUID,
        owner: User,
        target_user_id: UUID
    ) -> bool:
        """
        Revoke a user's access to a document.

        Args:
            document_id: Document ID
            owner: User revoking access (must be owner or admin)
            target_user_id: User to revoke access from

        Returns:
            True if successful

        Raises:
            PermissionError: If user cannot revoke access
        """
        # Check if user can manage permissions
        if not (self.is_owner(owner, document_id) or owner.role == UserRole.ADMIN):
            raise PermissionError(
                f"User {owner.id} cannot revoke access on document {document_id}"
            )

        return self.user_doc_repo.revoke_permission(
            user_id=target_user_id,
            document_id=document_id,
            revoked_by=owner.id
        )

    def update_permissions(
        self,
        document_id: UUID,
        owner: User,
        target_user_id: UUID,
        permissions: List[str]
    ) -> Optional[UserDocument]:
        """
        Update a user's permissions on a document.

        Args:
            document_id: Document ID
            owner: User updating permissions
            target_user_id: User whose permissions to update
            permissions: New permissions list

        Returns:
            Updated UserDocument or None

        Raises:
            PermissionError: If user cannot update permissions
        """
        if not (self.is_owner(owner, document_id) or owner.role == UserRole.ADMIN):
            raise PermissionError(
                f"User {owner.id} cannot update permissions on document {document_id}"
            )

        return self.user_doc_repo.update_permissions(
            user_id=target_user_id,
            document_id=document_id,
            permissions=permissions
        )

    def get_document_users(
        self,
        document_id: UUID,
        requesting_user: User
    ) -> List[UserDocument]:
        """
        Get all users with access to a document.

        Args:
            document_id: Document ID
            requesting_user: User making the request

        Returns:
            List of UserDocument records

        Raises:
            PermissionError: If user cannot view permissions
        """
        if not (self.is_owner(requesting_user, document_id) or
                requesting_user.role == UserRole.ADMIN):
            raise PermissionError(
                f"User {requesting_user.id} cannot view permissions for document {document_id}"
            )

        return self.user_doc_repo.get_document_permissions(document_id)

    def get_user_accessible_document_ids(
        self,
        user: User,
        permission: Optional[str] = None
    ) -> Set[UUID]:
        """
        Get all document IDs a user can access.
        For admin users, returns empty set (admin can access all).

        Args:
            user: User
            permission: Optional specific permission to filter by

        Returns:
            Set of document IDs
        """
        if user.role == UserRole.ADMIN:
            # Admin can access all - return empty set to indicate no filtering needed
            return set()

        return self.user_doc_repo.get_user_accessible_document_ids(
            user_id=user.id,
            permission=permission
        )

    def get_shared_with_me(
        self,
        user: User,
        limit: int = 100,
        offset: int = 0
    ) -> List[UserDocument]:
        """Get documents shared with a user."""
        return self.user_doc_repo.get_shared_documents(
            user_id=user.id,
            limit=limit,
            offset=offset
        )

    def get_new_shares(self, user: User) -> List[UserDocument]:
        """Get new (unviewed) shared documents."""
        return self.user_doc_repo.get_new_shares(user.id)

    def get_new_shares_count(self, user: User) -> int:
        """Get count of new shared documents."""
        return self.user_doc_repo.get_new_shares_count(user.id)

    def mark_as_viewed(
        self,
        user: User,
        document_id: UUID
    ) -> Optional[UserDocument]:
        """Mark a shared document as viewed."""
        return self.user_doc_repo.mark_as_viewed(user.id, document_id)

    def mark_all_as_viewed(self, user: User) -> int:
        """Mark all shared documents as viewed."""
        return self.user_doc_repo.mark_all_as_viewed(user.id)

    def record_access(
        self,
        user: User,
        document_id: UUID
    ) -> Optional[UserDocument]:
        """Record document access for tracking."""
        return self.user_doc_repo.record_access(user.id, document_id)

    def transfer_ownership(
        self,
        document_id: UUID,
        current_owner: User,
        new_owner_id: UUID
    ) -> bool:
        """
        Transfer document ownership to another user.

        Args:
            document_id: Document ID
            current_owner: Current owner
            new_owner_id: New owner's user ID

        Returns:
            True if successful

        Raises:
            PermissionError: If user cannot transfer ownership
        """
        if not (self.is_owner(current_owner, document_id) or
                current_owner.role == UserRole.ADMIN):
            raise PermissionError(
                f"User {current_owner.id} cannot transfer ownership of document {document_id}"
            )

        return self.user_doc_repo.transfer_ownership(
            document_id=document_id,
            current_owner_id=current_owner.id,
            new_owner_id=new_owner_id
        )

    def admin_grant_access(
        self,
        document_id: UUID,
        admin: User,
        target_user_id: UUID,
        permissions: List[str] = None
    ) -> UserDocument:
        """
        Admin grants access to a document.

        Args:
            document_id: Document ID
            admin: Admin user
            target_user_id: User to grant access to
            permissions: Permissions to grant

        Returns:
            UserDocument permission record

        Raises:
            PermissionError: If user is not admin
        """
        if admin.role != UserRole.ADMIN:
            raise PermissionError("Only admins can use admin_grant_access")

        return self.user_doc_repo.grant_admin_permission(
            document_id=document_id,
            target_user_id=target_user_id,
            admin_user_id=admin.id,
            permissions=permissions
        )

    def cleanup_expired_permissions(self) -> int:
        """Clean up expired permissions. Should be run periodically."""
        return self.user_doc_repo.cleanup_expired_permissions()

    def delete_document_permissions(self, document_id: UUID) -> int:
        """Delete all permissions for a document (called when deleting document)."""
        return self.user_doc_repo.delete_all_permissions_for_document(document_id)
