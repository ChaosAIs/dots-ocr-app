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
from unidecode import unidecode

from db.models import Workspace, Document, User

logger = logging.getLogger(__name__)

# Constants for folder name normalization
FOLDER_NAME_MAX_LENGTH = 100
TIMESTAMP_LENGTH = 15  # "_YYYYMMDDHHMMSS"
MAX_NAME_PORTION_LENGTH = FOLDER_NAME_MAX_LENGTH - TIMESTAMP_LENGTH  # 85 chars


class WorkspaceRepository:
    """Repository for workspace database operations."""

    def __init__(self, db: Session):
        """Initialize repository with database session."""
        self.db = db

    @staticmethod
    def normalize_workspace_folder_name(name: str) -> str:
        """
        Convert a workspace name to a normalized, ASCII-only folder name with timestamp.

        Steps:
        1. Transliterate non-ASCII characters to ASCII equivalents (e.g., Chinese to Pinyin)
        2. Lowercase
        3. Replace non-alphanumeric characters with underscores
        4. Collapse multiple underscores and strip leading/trailing underscores
        5. Append timestamp suffix for uniqueness
        6. Truncate name portion if needed (timestamp is never truncated)

        Args:
            name: Original workspace name (any language)

        Returns:
            Normalized folder name (ASCII-only, timestamped, max 100 chars)

        Examples:
            "我的项目" -> "wo_de_xiang_mu_20260102143025"
            "My Project!" -> "my_project_20260102143025"
            "プロジェクト" -> "purojiekuto_20260102143025"
        """
        # Step 1: Transliterate non-ASCII characters to ASCII equivalents
        transliterated = unidecode(name)

        # Step 2-3: Lowercase and replace non-alphanumeric with underscores
        sanitized = transliterated.lower()
        sanitized = re.sub(r'[^a-z0-9]', '_', sanitized)

        # Step 4: Collapse multiple underscores and strip leading/trailing
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')

        # Step 5: Generate timestamp suffix
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        suffix = f"_{timestamp}"  # 15 chars

        # Step 6: Truncate name portion, preserve timestamp
        truncated_name = sanitized[:MAX_NAME_PORTION_LENGTH].rstrip('_')

        # Handle empty name edge case
        if not truncated_name:
            truncated_name = 'workspace'

        return f"{truncated_name}{suffix}"

    @staticmethod
    def sanitize_folder_name(name: str) -> str:
        """
        Legacy method for backward compatibility.
        Convert a workspace name to a safe folder name without timestamp.
        Used only for system workspaces that have fixed folder names.

        Args:
            name: Workspace name

        Returns:
            Sanitized folder name (ASCII-only, no timestamp)
        """
        # Convert to lowercase and replace spaces
        folder_name = name.lower().replace(' ', '_')
        # Remove special characters (keep alphanumeric and underscores)
        folder_name = re.sub(r'[^a-z0-9_]', '', folder_name)
        # Limit length
        folder_name = folder_name[:100]
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
        is_system: bool = False,
        fixed_folder_name: Optional[str] = None
    ) -> Workspace:
        """
        Create a new workspace.

        Args:
            user_id: Owner's user ID
            name: Display name for the workspace (any language)
            normalized_username: Owner's normalized username (for folder path - filesystem safe)
            description: Optional description
            color: Hex color code for UI
            icon: Icon identifier
            is_default: Whether this is the user's default workspace
            is_system: Whether this is a system workspace (uses fixed folder name)
            fixed_folder_name: For system workspaces, use this exact folder name (no normalization)

        Returns:
            Created Workspace object

        Note:
            - Regular workspaces: folder_name is generated using normalize_workspace_folder_name()
              which transliterates non-ASCII chars and appends a timestamp for uniqueness.
            - System workspaces: folder_name uses the fixed_folder_name parameter (no timestamp).
        """
        if is_system and fixed_folder_name:
            # System workspaces use fixed folder names without timestamp
            folder_name = fixed_folder_name
        else:
            # Regular workspaces use normalized folder name with timestamp
            # Timestamp ensures uniqueness, no need for _ensure_unique_folder_name
            folder_name = self.normalize_workspace_folder_name(name)

        # Build folder path: normalized_username/folder_name
        folder_path = f"{normalized_username}/{folder_name}"

        workspace = Workspace(
            user_id=user_id,
            name=name,
            display_name=None,  # Initially NULL, user can set it later
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

        logger.info(f"Created workspace: {name} (folder: {folder_name}) for user {user_id}")
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
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        color: Optional[str] = None,
        icon: Optional[str] = None,
        clear_display_name: bool = False
    ) -> Optional[Workspace]:
        """
        Update workspace metadata.

        Args:
            workspace_id: Workspace ID
            display_name: New display name (any language). System workspaces cannot be renamed.
            description: Optional description
            color: Hex color code for UI
            icon: Icon identifier
            clear_display_name: If True, set display_name to NULL (falls back to name)

        Returns:
            Updated workspace or None if not found

        Note:
            - folder_name and folder_path are immutable after creation
            - System workspaces cannot have their display_name changed
        """
        workspace = self.get_workspace_by_id(workspace_id)

        if not workspace:
            return None

        # Handle display_name update (blocked for system workspaces)
        if clear_display_name:
            workspace.display_name = None
        elif display_name is not None:
            if workspace.is_system:
                logger.warning(f"Cannot rename system workspace: {workspace_id}")
                # Don't return None here, still allow other metadata updates
            else:
                workspace.display_name = display_name

        if description is not None:
            workspace.description = description
        if color is not None:
            workspace.color = color
        if icon is not None:
            workspace.icon = icon

        workspace.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(workspace)

        logger.info(f"Updated workspace: {workspace.get_effective_display_name()} (ID: {workspace_id})")
        return workspace

    def update_display_name(
        self,
        workspace_id: UUID,
        new_display_name: Optional[str]
    ) -> Optional[Workspace]:
        """
        Update workspace display name.

        This is the new rename operation that only updates display_name without
        affecting folder_name or folder_path. The physical folder remains unchanged.

        Args:
            workspace_id: Workspace ID
            new_display_name: New display name (any language) or None to clear

        Returns:
            Updated workspace or None if blocked (system workspace) or not found

        Note:
            - System workspaces cannot be renamed (returns None)
            - Setting new_display_name to None clears it, falling back to original name
        """
        workspace = self.get_workspace_by_id(workspace_id)

        if not workspace:
            return None

        # Block rename for system workspaces
        if workspace.is_system:
            logger.warning(f"Cannot rename system workspace: {workspace_id}")
            return None

        old_display_name = workspace.get_effective_display_name()
        workspace.display_name = new_display_name
        workspace.updated_at = datetime.utcnow()

        self.db.commit()
        self.db.refresh(workspace)

        new_effective_name = workspace.get_effective_display_name()
        logger.info(f"Updated workspace display name: '{old_display_name}' -> '{new_effective_name}' (folder unchanged: {workspace.folder_name})")
        return workspace

    def rename_workspace(
        self,
        workspace_id: UUID,
        new_name: str,
        normalized_username: str
    ) -> Optional[Workspace]:
        """
        DEPRECATED: Use update_display_name() instead.

        This method is kept for backward compatibility but now only updates display_name.
        It no longer renames the physical folder.

        Args:
            workspace_id: Workspace ID
            new_name: New display name
            normalized_username: Ignored (kept for API compatibility)

        Returns:
            Updated workspace or None if blocked/not found
        """
        logger.warning("rename_workspace() is deprecated. Use update_display_name() instead.")
        return self.update_display_name(workspace_id, new_name)

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
