"""
User document repository for permission and access control operations.
"""
import logging
from datetime import datetime
from typing import Optional, List, Set
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, literal

from db.models import (
    UserDocument, Document, User, UserRole,
    DocumentPermission, PermissionOrigin
)

logger = logging.getLogger(__name__)


class UserDocumentRepository:
    """Repository for user document permission operations."""

    def __init__(self, db: Session):
        """Initialize repository with database session."""
        self.db = db

    def create_owner_permission(
        self,
        user_id: UUID,
        document_id: UUID
    ) -> UserDocument:
        """
        Create owner permission when user uploads a document.
        Owner gets full permissions.
        """
        user_doc = UserDocument(
            user_id=user_id,
            document_id=document_id,
            permissions=[DocumentPermission.FULL.value],
            origin=PermissionOrigin.OWNER.value,
            is_owner=True,
            is_new=False  # Owner doesn't need notification
        )

        self.db.add(user_doc)
        self.db.commit()
        self.db.refresh(user_doc)

        logger.info(f"Created owner permission for user {user_id} on document {document_id}")
        return user_doc

    def share_document(
        self,
        document_id: UUID,
        target_user_id: UUID,
        shared_by_user_id: UUID,
        permissions: List[str] = None,
        message: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> UserDocument:
        """
        Share a document with another user.

        Args:
            document_id: Document to share
            target_user_id: User to share with
            shared_by_user_id: User who is sharing
            permissions: List of permissions to grant (default: ['read'])
            message: Optional message to recipient
            expires_at: Optional expiration time
        """
        if permissions is None:
            permissions = [DocumentPermission.READ.value]

        # Check if permission record already exists
        existing = self.get_user_document(target_user_id, document_id)

        if existing:
            # Update existing permission
            existing.permissions = permissions
            existing.shared_by = shared_by_user_id
            existing.shared_at = datetime.utcnow()
            existing.expires_at = expires_at
            existing.share_message = message
            existing.is_new = True
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(existing)
            logger.info(f"Updated share permission for user {target_user_id} on document {document_id}")
            return existing

        # Create new permission
        user_doc = UserDocument(
            user_id=target_user_id,
            document_id=document_id,
            permissions=permissions,
            origin=PermissionOrigin.SHARED.value,
            is_owner=False,
            shared_by=shared_by_user_id,
            shared_at=datetime.utcnow(),
            expires_at=expires_at,
            share_message=message,
            is_new=True
        )

        self.db.add(user_doc)
        self.db.commit()
        self.db.refresh(user_doc)

        logger.info(f"Shared document {document_id} with user {target_user_id}")
        return user_doc

    def grant_admin_permission(
        self,
        document_id: UUID,
        target_user_id: UUID,
        admin_user_id: UUID,
        permissions: List[str] = None
    ) -> UserDocument:
        """
        Admin grants permission to a user on a document.

        Args:
            document_id: Document ID
            target_user_id: User to grant permission to
            admin_user_id: Admin performing the action
            permissions: Permissions to grant (default: ['read'])
        """
        if permissions is None:
            permissions = [DocumentPermission.READ.value]

        existing = self.get_user_document(target_user_id, document_id)

        if existing:
            existing.permissions = permissions
            existing.origin = PermissionOrigin.ADMIN_GRANTED.value
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(existing)
            return existing

        user_doc = UserDocument(
            user_id=target_user_id,
            document_id=document_id,
            permissions=permissions,
            origin=PermissionOrigin.ADMIN_GRANTED.value,
            is_owner=False,
            shared_by=admin_user_id,
            shared_at=datetime.utcnow(),
            is_new=True
        )

        self.db.add(user_doc)
        self.db.commit()
        self.db.refresh(user_doc)

        logger.info(f"Admin {admin_user_id} granted access to document {document_id} for user {target_user_id}")
        return user_doc

    def get_user_document(
        self,
        user_id: UUID,
        document_id: UUID
    ) -> Optional[UserDocument]:
        """Get permission record for a user and document."""
        return self.db.query(UserDocument).filter(
            and_(
                UserDocument.user_id == user_id,
                UserDocument.document_id == document_id
            )
        ).first()

    def get_document_permissions(
        self,
        document_id: UUID,
        include_expired: bool = False
    ) -> List[UserDocument]:
        """Get all permission records for a document."""
        query = self.db.query(UserDocument).filter(
            UserDocument.document_id == document_id
        )

        if not include_expired:
            query = query.filter(
                or_(
                    UserDocument.expires_at.is_(None),
                    UserDocument.expires_at > datetime.utcnow()
                )
            )

        return query.all()

    def get_user_accessible_documents(
        self,
        user_id: UUID,
        include_expired: bool = False
    ) -> List[UserDocument]:
        """Get all documents a user has access to."""
        query = self.db.query(UserDocument).filter(
            UserDocument.user_id == user_id
        )

        if not include_expired:
            query = query.filter(
                or_(
                    UserDocument.expires_at.is_(None),
                    UserDocument.expires_at > datetime.utcnow()
                )
            )

        return query.all()

    def get_user_accessible_document_ids(
        self,
        user_id: UUID,
        permission: Optional[str] = None
    ) -> Set[UUID]:
        """
        Get set of document IDs a user can access.
        Optimized for filtering in searches.

        Args:
            user_id: User ID
            permission: Optional specific permission to check (e.g., 'read', 'update')
        """
        query = self.db.query(UserDocument.document_id).filter(
            UserDocument.user_id == user_id,
            or_(
                UserDocument.expires_at.is_(None),
                UserDocument.expires_at > datetime.utcnow()
            )
        )

        if permission:
            # Filter by specific permission using PostgreSQL's ANY operator
            # This checks if the permission value is in the permissions array
            query = query.filter(
                or_(
                    UserDocument.is_owner == True,
                    literal(permission).op('=')(func.any(UserDocument.permissions)),
                    literal('full').op('=')(func.any(UserDocument.permissions))
                )
            )

        result = query.all()
        return {row[0] for row in result}

    def check_permission(
        self,
        user_id: UUID,
        document_id: UUID,
        required_permission: str,
        user_role: Optional[UserRole] = None
    ) -> bool:
        """
        Check if a user has a specific permission on a document.

        Args:
            user_id: User ID
            document_id: Document ID
            required_permission: Permission to check (read, update, delete, share, full)
            user_role: User's role (admin bypasses permission check)

        Returns:
            True if user has permission, False otherwise
        """
        # Admin bypass
        if user_role == UserRole.ADMIN:
            return True

        user_doc = self.get_user_document(user_id, document_id)

        if not user_doc:
            return False

        # Check expiration
        if user_doc.expires_at and user_doc.expires_at < datetime.utcnow():
            return False

        # Owner or full permission has all access
        if user_doc.is_owner or 'full' in user_doc.permissions:
            return True

        return required_permission in user_doc.permissions

    def has_any_access(
        self,
        user_id: UUID,
        document_id: UUID,
        user_role: Optional[UserRole] = None
    ) -> bool:
        """Check if user has any access to a document."""
        # Admin has access to all documents
        if user_role == UserRole.ADMIN:
            return True

        user_doc = self.get_user_document(user_id, document_id)

        if not user_doc:
            return False

        # Check expiration
        if user_doc.expires_at and user_doc.expires_at < datetime.utcnow():
            return False

        return True

    def revoke_permission(
        self,
        user_id: UUID,
        document_id: UUID,
        revoked_by: Optional[UUID] = None
    ) -> bool:
        """
        Revoke a user's permission on a document.
        Cannot revoke owner permission.
        """
        user_doc = self.get_user_document(user_id, document_id)

        if not user_doc:
            return False

        if user_doc.is_owner:
            logger.warning(f"Cannot revoke owner permission for user {user_id} on document {document_id}")
            return False

        self.db.delete(user_doc)
        self.db.commit()

        logger.info(f"Revoked permission for user {user_id} on document {document_id}")
        return True

    def update_permissions(
        self,
        user_id: UUID,
        document_id: UUID,
        permissions: List[str]
    ) -> Optional[UserDocument]:
        """Update permissions for an existing user document record."""
        user_doc = self.get_user_document(user_id, document_id)

        if not user_doc or user_doc.is_owner:
            return None

        user_doc.permissions = permissions
        user_doc.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(user_doc)

        return user_doc

    def get_shared_documents(
        self,
        user_id: UUID,
        limit: int = 100,
        offset: int = 0
    ) -> List[UserDocument]:
        """Get documents shared with a user (not owned)."""
        return self.db.query(UserDocument).filter(
            and_(
                UserDocument.user_id == user_id,
                UserDocument.is_owner == False,
                or_(
                    UserDocument.expires_at.is_(None),
                    UserDocument.expires_at > datetime.utcnow()
                )
            )
        ).order_by(UserDocument.shared_at.desc()).offset(offset).limit(limit).all()

    def get_owned_documents(
        self,
        user_id: UUID,
        limit: int = 100,
        offset: int = 0
    ) -> List[UserDocument]:
        """Get documents owned by a user."""
        return self.db.query(UserDocument).filter(
            and_(
                UserDocument.user_id == user_id,
                UserDocument.is_owner == True
            )
        ).order_by(UserDocument.created_at.desc()).offset(offset).limit(limit).all()

    def get_new_shares(self, user_id: UUID) -> List[UserDocument]:
        """Get new (unread) shared documents for a user."""
        return self.db.query(UserDocument).filter(
            and_(
                UserDocument.user_id == user_id,
                UserDocument.is_new == True,
                UserDocument.is_owner == False,
                or_(
                    UserDocument.expires_at.is_(None),
                    UserDocument.expires_at > datetime.utcnow()
                )
            )
        ).order_by(UserDocument.shared_at.desc()).all()

    def get_new_shares_count(self, user_id: UUID) -> int:
        """Get count of new shared documents."""
        return self.db.query(func.count(UserDocument.id)).filter(
            and_(
                UserDocument.user_id == user_id,
                UserDocument.is_new == True,
                UserDocument.is_owner == False,
                or_(
                    UserDocument.expires_at.is_(None),
                    UserDocument.expires_at > datetime.utcnow()
                )
            )
        ).scalar() or 0

    def mark_as_viewed(
        self,
        user_id: UUID,
        document_id: UUID
    ) -> Optional[UserDocument]:
        """Mark a shared document as viewed."""
        user_doc = self.get_user_document(user_id, document_id)

        if not user_doc:
            return None

        user_doc.is_new = False
        user_doc.viewed_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(user_doc)

        return user_doc

    def mark_all_as_viewed(self, user_id: UUID) -> int:
        """Mark all shared documents as viewed."""
        count = self.db.query(UserDocument).filter(
            and_(
                UserDocument.user_id == user_id,
                UserDocument.is_new == True
            )
        ).update({
            "is_new": False,
            "viewed_at": datetime.utcnow()
        })

        self.db.commit()
        return count

    def record_access(
        self,
        user_id: UUID,
        document_id: UUID
    ) -> Optional[UserDocument]:
        """Record document access for tracking."""
        user_doc = self.get_user_document(user_id, document_id)

        if not user_doc:
            return None

        user_doc.last_accessed_at = datetime.utcnow()
        user_doc.access_count += 1
        self.db.commit()
        self.db.refresh(user_doc)

        return user_doc

    def transfer_ownership(
        self,
        document_id: UUID,
        current_owner_id: UUID,
        new_owner_id: UUID
    ) -> bool:
        """
        Transfer document ownership to another user.

        Args:
            document_id: Document ID
            current_owner_id: Current owner's user ID
            new_owner_id: New owner's user ID

        Returns:
            True if transfer successful, False otherwise
        """
        # Get current owner's permission
        current_owner_doc = self.get_user_document(current_owner_id, document_id)

        if not current_owner_doc or not current_owner_doc.is_owner:
            return False

        # Get or create new owner's permission
        new_owner_doc = self.get_user_document(new_owner_id, document_id)

        if new_owner_doc:
            # Update existing to owner
            new_owner_doc.is_owner = True
            new_owner_doc.permissions = [DocumentPermission.FULL.value]
            new_owner_doc.origin = PermissionOrigin.OWNER.value
        else:
            # Create owner permission for new owner
            new_owner_doc = UserDocument(
                user_id=new_owner_id,
                document_id=document_id,
                permissions=[DocumentPermission.FULL.value],
                origin=PermissionOrigin.OWNER.value,
                is_owner=True
            )
            self.db.add(new_owner_doc)

        # Demote current owner to shared with full access
        current_owner_doc.is_owner = False
        current_owner_doc.permissions = [DocumentPermission.FULL.value]
        current_owner_doc.origin = PermissionOrigin.SHARED.value

        self.db.commit()

        logger.info(f"Transferred ownership of document {document_id} from {current_owner_id} to {new_owner_id}")
        return True

    def get_document_owner(self, document_id: UUID) -> Optional[UserDocument]:
        """Get the owner's permission record for a document."""
        return self.db.query(UserDocument).filter(
            and_(
                UserDocument.document_id == document_id,
                UserDocument.is_owner == True
            )
        ).first()

    def delete_all_permissions_for_document(self, document_id: UUID) -> int:
        """Delete all permission records for a document (used when deleting document)."""
        count = self.db.query(UserDocument).filter(
            UserDocument.document_id == document_id
        ).delete()

        self.db.commit()

        logger.info(f"Deleted {count} permission records for document {document_id}")
        return count

    def cleanup_expired_permissions(self) -> int:
        """Remove expired permission records. Run periodically."""
        count = self.db.query(UserDocument).filter(
            and_(
                UserDocument.expires_at.is_not(None),
                UserDocument.expires_at < datetime.utcnow(),
                UserDocument.is_owner == False  # Never clean up owner permissions
            )
        ).delete()

        self.db.commit()

        if count > 0:
            logger.info(f"Cleaned up {count} expired permission records")

        return count
