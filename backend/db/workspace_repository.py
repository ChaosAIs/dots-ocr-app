"""
Workspace repository for database operations.
"""
import logging
import re
from datetime import datetime
from typing import Optional, List
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from db.models import Workspace, Document, User

logger = logging.getLogger(__name__)


class WorkspaceRepository:
    """Repository for workspace database operations."""

    def __init__(self, db: Session):
        """Initialize repository with database session."""
        self.db = db

    @staticmethod
    def sanitize_folder_name(name: str) -> str:
        """
        Convert a workspace name to a safe folder name.
        - Lowercase
        - Replace spaces with underscores
        - Remove special characters
        - Limit length
        """
        # Convert to lowercase and replace spaces
        folder_name = name.lower().replace(' ', '_')
        # Remove special characters (keep alphanumeric and underscores)
        folder_name = re.sub(r'[^a-z0-9_]', '', folder_name)
        # Limit length
        folder_name = folder_name[:50]
        # Ensure not empty
        if not folder_name:
            folder_name = 'workspace'
        return folder_name

    def create_workspace(
        self,
        user_id: UUID,
        name: str,
        normalized_username: str,
        description: Optional[str] = None,
        color: str = '#6366f1',
        icon: str = 'folder',
        is_default: bool = False,
        is_system: bool = False
    ) -> Workspace:
        """
        Create a new workspace.

        Args:
            user_id: Owner's user ID
            name: Display name for the workspace
            normalized_username: Owner's normalized username (for folder path - filesystem safe)
            description: Optional description
            color: Hex color code for UI
            icon: Icon identifier
            is_default: Whether this is the user's default workspace
            is_system: Whether this is a system workspace (no physical folder)
        """
        folder_name = self.sanitize_folder_name(name)

        # Ensure unique folder name for user
        folder_name = self._ensure_unique_folder_name(user_id, folder_name)

        # Build folder path: normalized_username/folder_name (using normalized username for filesystem safety)
        folder_path = f"{normalized_username}/{folder_name}"

        workspace = Workspace(
            user_id=user_id,
            name=name,
            folder_name=folder_name,
            folder_path=folder_path,
            description=description,
            color=color,
            icon=icon,
            is_default=is_default,
            is_system=is_system,
            display_order=self._get_next_display_order(user_id)
        )

        self.db.add(workspace)
        self.db.commit()
        self.db.refresh(workspace)

        logger.info(f"Created workspace: {name} (ID: {workspace.id}) for user {user_id}")
        return workspace

    def _ensure_unique_folder_name(self, user_id: UUID, folder_name: str) -> str:
        """Ensure folder name is unique for the user."""
        original_name = folder_name
        counter = 1

        while self.get_workspace_by_folder_name(user_id, folder_name):
            folder_name = f"{original_name}_{counter}"
            counter += 1

        return folder_name

    def _get_next_display_order(self, user_id: UUID) -> int:
        """Get the next display order for a user's workspaces."""
        max_order = self.db.query(func.max(Workspace.display_order)).filter(
            Workspace.user_id == user_id
        ).scalar()
        return (max_order or 0) + 1

    def get_workspace_by_id(self, workspace_id: UUID) -> Optional[Workspace]:
        """Get workspace by ID."""
        return self.db.query(Workspace).filter(
            Workspace.id == workspace_id
        ).first()

    def get_workspace_by_folder_name(self, user_id: UUID, folder_name: str) -> Optional[Workspace]:
        """Get workspace by folder name for a user."""
        return self.db.query(Workspace).filter(
            and_(
                Workspace.user_id == user_id,
                Workspace.folder_name == folder_name
            )
        ).first()

    def get_workspace_by_name(self, user_id: UUID, name: str) -> Optional[Workspace]:
        """Get workspace by display name for a user."""
        return self.db.query(Workspace).filter(
            and_(
                Workspace.user_id == user_id,
                Workspace.name == name
            )
        ).first()

    def get_user_workspaces(
        self,
        user_id: UUID,
        include_system: bool = True
    ) -> List[Workspace]:
        """
        Get all workspaces for a user.

        Args:
            user_id: User ID
            include_system: Whether to include system workspaces
        """
        query = self.db.query(Workspace).filter(Workspace.user_id == user_id)

        if not include_system:
            query = query.filter(Workspace.is_system == False)

        return query.order_by(
            Workspace.is_default.desc(),
            Workspace.display_order
        ).all()

    def get_default_workspace(self, user_id: UUID) -> Optional[Workspace]:
        """Get user's default workspace."""
        return self.db.query(Workspace).filter(
            and_(
                Workspace.user_id == user_id,
                Workspace.is_default == True,
                Workspace.is_system == False
            )
        ).first()

    def get_or_create_default_workspace(
        self,
        user_id: UUID,
        normalized_username: str
    ) -> Workspace:
        """Get or create the user's default workspace."""
        workspace = self.get_default_workspace(user_id)

        if not workspace:
            workspace = self.create_workspace(
                user_id=user_id,
                name="My Documents",
                normalized_username=normalized_username,
                description="Default workspace",
                is_default=True
            )
            logger.info(f"Created default workspace for user {user_id}")

        return workspace

    def get_shared_with_me_workspace(self, user_id: UUID) -> Optional[Workspace]:
        """Get user's 'Shared With Me' system workspace."""
        return self.db.query(Workspace).filter(
            and_(
                Workspace.user_id == user_id,
                Workspace.is_system == True,
                Workspace.name == "Shared With Me"
            )
        ).first()

    def get_or_create_shared_with_me_workspace(
        self,
        user_id: UUID,
        normalized_username: str
    ) -> Workspace:
        """Get or create the 'Shared With Me' system workspace."""
        workspace = self.get_shared_with_me_workspace(user_id)

        if not workspace:
            workspace = Workspace(
                user_id=user_id,
                name="Shared With Me",
                folder_name="_shared_with_me",
                folder_path=f"{normalized_username}/_shared_with_me",
                description="Documents shared with you by others",
                icon="share",
                is_system=True,
                display_order=999  # Always at the end
            )
            self.db.add(workspace)
            self.db.commit()
            self.db.refresh(workspace)
            logger.info(f"Created 'Shared With Me' workspace for user {user_id}")

        return workspace

    def update_workspace(
        self,
        workspace_id: UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
        color: Optional[str] = None,
        icon: Optional[str] = None
    ) -> Optional[Workspace]:
        """
        Update workspace metadata.
        Note: Cannot update folder_name or folder_path directly - use rename_workspace.
        """
        workspace = self.get_workspace_by_id(workspace_id)

        if not workspace:
            return None

        if name is not None:
            workspace.name = name
        if description is not None:
            workspace.description = description
        if color is not None:
            workspace.color = color
        if icon is not None:
            workspace.icon = icon

        workspace.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(workspace)

        logger.info(f"Updated workspace: {workspace.name} (ID: {workspace_id})")
        return workspace

    def rename_workspace(
        self,
        workspace_id: UUID,
        new_name: str,
        normalized_username: str
    ) -> Optional[Workspace]:
        """
        Rename a workspace and update folder paths.
        Returns the updated workspace or None if not found.
        Note: Physical folder rename should be handled by the service layer.
        """
        workspace = self.get_workspace_by_id(workspace_id)

        if not workspace or workspace.is_system:
            return None

        old_folder_name = workspace.folder_name
        new_folder_name = self.sanitize_folder_name(new_name)

        # Ensure unique folder name
        if new_folder_name != old_folder_name:
            new_folder_name = self._ensure_unique_folder_name(
                workspace.user_id,
                new_folder_name
            )

        workspace.name = new_name
        workspace.folder_name = new_folder_name
        workspace.folder_path = f"{normalized_username}/{new_folder_name}"
        workspace.updated_at = datetime.utcnow()

        self.db.commit()
        self.db.refresh(workspace)

        logger.info(f"Renamed workspace from {old_folder_name} to {new_folder_name}")
        return workspace

    def delete_workspace(self, workspace_id: UUID) -> bool:
        """
        Delete a workspace from database.
        Note: Physical folder deletion and document handling should be done by service layer.
        """
        workspace = self.get_workspace_by_id(workspace_id)

        if not workspace:
            return False

        if workspace.is_default:
            logger.warning(f"Cannot delete default workspace: {workspace_id}")
            return False

        if workspace.is_system:
            logger.warning(f"Cannot delete system workspace: {workspace_id}")
            return False

        self.db.delete(workspace)
        self.db.commit()

        logger.info(f"Deleted workspace: {workspace.name} (ID: {workspace_id})")
        return True

    def set_default_workspace(self, user_id: UUID, workspace_id: UUID) -> bool:
        """Set a workspace as the user's default."""
        # Get the workspace
        workspace = self.get_workspace_by_id(workspace_id)

        if not workspace or workspace.user_id != user_id or workspace.is_system:
            return False

        # Clear existing default
        self.db.query(Workspace).filter(
            and_(
                Workspace.user_id == user_id,
                Workspace.is_default == True
            )
        ).update({"is_default": False})

        # Set new default
        workspace.is_default = True
        self.db.commit()

        logger.info(f"Set workspace {workspace_id} as default for user {user_id}")
        return True

    def update_display_order(
        self,
        user_id: UUID,
        workspace_orders: List[dict]
    ) -> bool:
        """
        Update display order for multiple workspaces.

        Args:
            user_id: User ID
            workspace_orders: List of {"workspace_id": UUID, "order": int}
        """
        for item in workspace_orders:
            workspace = self.get_workspace_by_id(item["workspace_id"])
            if workspace and workspace.user_id == user_id:
                workspace.display_order = item["order"]

        self.db.commit()
        return True

    def update_document_count(self, workspace_id: UUID, delta: int = 0) -> None:
        """
        Update workspace document count.

        Args:
            workspace_id: Workspace ID
            delta: Amount to add (positive) or subtract (negative)
        """
        workspace = self.get_workspace_by_id(workspace_id)
        if workspace:
            workspace.document_count = max(0, workspace.document_count + delta)
            self.db.commit()

    def recalculate_document_count(self, workspace_id: UUID) -> int:
        """Recalculate and update document count from actual documents."""
        count = self.db.query(func.count(Document.id)).filter(
            Document.workspace_id == workspace_id,
            Document.deleted_at.is_(None)
        ).scalar() or 0

        workspace = self.get_workspace_by_id(workspace_id)
        if workspace:
            workspace.document_count = count
            self.db.commit()

        return count

    def get_workspace_documents(
        self,
        workspace_id: UUID,
        limit: int = 100,
        offset: int = 0
    ) -> List[Document]:
        """Get documents in a workspace."""
        return self.db.query(Document).filter(
            Document.workspace_id == workspace_id,
            Document.deleted_at.is_(None)
        ).order_by(Document.created_at.desc()).offset(offset).limit(limit).all()
