"""
Workspace service for managing workspaces with file system synchronization.
"""
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from db.models import Workspace, Document, User
from db.workspace_repository import WorkspaceRepository
from db.user_document_repository import UserDocumentRepository

logger = logging.getLogger(__name__)


class WorkspaceService:
    """
    Service for workspace operations with file system synchronization.

    Workspace Structure:
    - storage/
      - {username}/
        - {workspace_folder}/   (input files)
        - _output/              (processed files)
      - _shared/                (for public/shared files if needed)
    """

    def __init__(
        self,
        db: Session,
        storage_base_path: str = "input",
        output_base_path: str = "output"
    ):
        """
        Initialize workspace service.

        Args:
            db: Database session
            storage_base_path: Base path for file storage (default: "input")
            output_base_path: Base path for output files (default: "output")
        """
        self.db = db
        self.workspace_repo = WorkspaceRepository(db)
        self.user_doc_repo = UserDocumentRepository(db)
        self.storage_base_path = Path(storage_base_path)
        self.output_base_path = Path(output_base_path)

    def get_user_storage_path(self, normalized_username: str) -> Path:
        """Get the storage path for a user (uses normalized username for filesystem safety)."""
        return self.storage_base_path / normalized_username

    def get_workspace_storage_path(self, workspace: Workspace) -> Path:
        """Get the storage path for a workspace."""
        return self.storage_base_path / workspace.folder_path

    def get_user_output_path(self, normalized_username: str) -> Path:
        """Get the output path for a user (uses normalized username for filesystem safety)."""
        return self.output_base_path / normalized_username

    def ensure_user_directories(self, normalized_username: str) -> Tuple[Path, Path]:
        """
        Ensure user's storage and output directories exist.

        Args:
            normalized_username: Normalized username for filesystem-safe paths

        Returns:
            Tuple of (storage_path, output_path)
        """
        storage_path = self.get_user_storage_path(normalized_username)
        output_path = self.get_user_output_path(normalized_username)

        storage_path.mkdir(parents=True, exist_ok=True)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Ensured directories for user {normalized_username}")
        return storage_path, output_path

    def create_workspace(
        self,
        user: User,
        name: str,
        description: Optional[str] = None,
        color: str = '#6366f1',
        icon: str = 'folder',
        is_default: bool = False
    ) -> Workspace:
        """
        Create a new workspace with physical folder.

        Args:
            user: User object
            name: Workspace display name
            description: Optional description
            color: Hex color for UI
            icon: Icon identifier
            is_default: Whether this is user's default workspace

        Returns:
            Created Workspace object
        """
        # Ensure user directories exist (using normalized username for filesystem safety)
        self.ensure_user_directories(user.normalized_username)

        # Create workspace in database
        workspace = self.workspace_repo.create_workspace(
            user_id=user.id,
            name=name,
            normalized_username=user.normalized_username,
            description=description,
            color=color,
            icon=icon,
            is_default=is_default,
            is_system=False
        )

        # Create physical folder
        folder_path = self.get_workspace_storage_path(workspace)
        folder_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created workspace '{name}' at {folder_path}")
        return workspace

    def get_or_create_default_workspace(self, user: User) -> Workspace:
        """Get or create user's default workspace."""
        workspace = self.workspace_repo.get_default_workspace(user.id)

        if not workspace:
            workspace = self.create_workspace(
                user=user,
                name="My Documents",
                description="Default workspace for your documents",
                is_default=True
            )

        return workspace

    def get_user_workspaces(
        self,
        user_id: UUID,
        include_system: bool = True
    ) -> List[Workspace]:
        """Get all workspaces for a user."""
        return self.workspace_repo.get_user_workspaces(
            user_id=user_id,
            include_system=include_system
        )

    def get_workspace(self, workspace_id: UUID) -> Optional[Workspace]:
        """Get workspace by ID."""
        return self.workspace_repo.get_workspace_by_id(workspace_id)

    def update_workspace(
        self,
        workspace_id: UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
        color: Optional[str] = None,
        icon: Optional[str] = None
    ) -> Optional[Workspace]:
        """Update workspace metadata (does not change folder name)."""
        return self.workspace_repo.update_workspace(
            workspace_id=workspace_id,
            name=name,
            description=description,
            color=color,
            icon=icon
        )

    def rename_workspace(
        self,
        workspace_id: UUID,
        new_name: str,
        user: User
    ) -> Optional[Workspace]:
        """
        Rename a workspace and its physical folder.

        Args:
            workspace_id: Workspace ID
            new_name: New display name
            user: User object (for username)

        Returns:
            Updated workspace or None if failed
        """
        workspace = self.workspace_repo.get_workspace_by_id(workspace_id)

        if not workspace or workspace.is_system:
            return None

        old_folder_path = self.get_workspace_storage_path(workspace)

        # Update database (this also updates folder_name and folder_path)
        updated_workspace = self.workspace_repo.rename_workspace(
            workspace_id=workspace_id,
            new_name=new_name,
            normalized_username=user.normalized_username
        )

        if not updated_workspace:
            return None

        # Rename physical folder if it changed
        if workspace.folder_name != updated_workspace.folder_name:
            new_folder_path = self.get_workspace_storage_path(updated_workspace)

            try:
                if old_folder_path.exists():
                    old_folder_path.rename(new_folder_path)
                    logger.info(f"Renamed folder from {old_folder_path} to {new_folder_path}")
                else:
                    # Create new folder if old one doesn't exist
                    new_folder_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to rename folder: {e}")
                # Rollback database change
                self.workspace_repo.rename_workspace(
                    workspace_id=workspace_id,
                    new_name=workspace.name,
                    normalized_username=user.normalized_username
                )
                return None

            # Update document file paths
            self._update_document_paths(
                workspace_id=workspace_id,
                old_folder_path=str(old_folder_path),
                new_folder_path=str(new_folder_path)
            )

        return updated_workspace

    def _update_document_paths(
        self,
        workspace_id: UUID,
        old_folder_path: str,
        new_folder_path: str
    ) -> None:
        """Update document file paths after folder rename."""
        documents = self.workspace_repo.get_workspace_documents(
            workspace_id=workspace_id,
            limit=10000  # Get all documents
        )

        for doc in documents:
            if doc.file_path and old_folder_path in doc.file_path:
                doc.file_path = doc.file_path.replace(old_folder_path, new_folder_path)
                self.db.commit()

    def delete_workspace(
        self,
        workspace_id: UUID,
        delete_documents: bool = False,
        move_to_default: bool = True,
        user: Optional[User] = None
    ) -> bool:
        """
        Delete a workspace.

        Args:
            workspace_id: Workspace ID to delete
            delete_documents: If True, delete all documents in workspace
            move_to_default: If True and delete_documents is False, move docs to default workspace
            user: User object (needed if move_to_default is True)

        Returns:
            True if successful
        """
        workspace = self.workspace_repo.get_workspace_by_id(workspace_id)

        if not workspace:
            return False

        if workspace.is_default or workspace.is_system:
            logger.warning(f"Cannot delete default or system workspace: {workspace_id}")
            return False

        documents = self.workspace_repo.get_workspace_documents(
            workspace_id=workspace_id,
            limit=10000
        )

        if documents:
            if delete_documents:
                # Delete documents and their files
                for doc in documents:
                    self._delete_document_files(doc)
                    # Delete permission records
                    self.user_doc_repo.delete_all_permissions_for_document(doc.id)
                    self.db.delete(doc)
                self.db.commit()
            elif move_to_default and user:
                # Move documents to default workspace
                default_workspace = self.get_or_create_default_workspace(user)
                self._move_documents_to_workspace(
                    documents=documents,
                    target_workspace=default_workspace,
                    user=user
                )

        # Delete physical folder
        folder_path = self.get_workspace_storage_path(workspace)
        if folder_path.exists():
            try:
                shutil.rmtree(folder_path)
                logger.info(f"Deleted folder: {folder_path}")
            except OSError as e:
                logger.error(f"Failed to delete folder {folder_path}: {e}")

        # Delete from database
        return self.workspace_repo.delete_workspace(workspace_id)

    def _generate_unique_filename(self, target_path: Path, original_filename: str) -> str:
        """
        Generate a unique filename if the file already exists in target directory.

        Args:
            target_path: Target directory path
            original_filename: Original filename to check

        Returns:
            Unique filename (may be same as original if no conflict)
        """
        file_path = target_path / original_filename
        if not file_path.exists():
            return original_filename

        # Split filename into name and extension
        stem = Path(original_filename).stem
        suffix = Path(original_filename).suffix

        # Check if stem already ends with a number pattern like "_1", "_2", etc.
        match = re.match(r'^(.+)_(\d+)$', stem)
        if match:
            base_name = match.group(1)
            counter = int(match.group(2))
        else:
            base_name = stem
            counter = 0

        # Find next available number
        while True:
            counter += 1
            new_filename = f"{base_name}_{counter}{suffix}"
            new_path = target_path / new_filename
            if not new_path.exists():
                logger.info(f"Generated unique filename: {original_filename} -> {new_filename}")
                return new_filename

    def _delete_document_files(self, document: Document) -> None:
        """Delete document's physical files (input and output)."""
        # Delete input file
        if document.file_path:
            # file_path is relative to storage_base_path (e.g., "username/workspace/filename")
            file_path = self.storage_base_path / document.file_path
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Deleted input file: {file_path}")
                except OSError as e:
                    logger.error(f"Failed to delete file {file_path}: {e}")

        # Delete output folder
        if document.output_path:
            # output_path is relative to output_base_path
            output_path = self.output_base_path / document.output_path
            if output_path.exists():
                try:
                    if output_path.is_dir():
                        shutil.rmtree(output_path)
                    else:
                        output_path.unlink()
                    logger.info(f"Deleted output path: {output_path}")
                except OSError as e:
                    logger.error(f"Failed to delete output {output_path}: {e}")
        elif document.file_path:
            # Fallback: construct output path from file_path
            # e.g., "username/workspace/filename.pdf" -> "username/workspace/filename"
            file_name_without_ext = Path(document.filename).stem
            rel_dir = str(Path(document.file_path).parent)
            if rel_dir and rel_dir != '.':
                output_path = self.output_base_path / rel_dir / file_name_without_ext
            else:
                output_path = self.output_base_path / file_name_without_ext

            if output_path.exists() and output_path.is_dir():
                try:
                    shutil.rmtree(output_path)
                    logger.info(f"Deleted output folder (fallback): {output_path}")
                except OSError as e:
                    logger.error(f"Failed to delete output folder {output_path}: {e}")

            # Also try to delete JSONL file
            jsonl_path = output_path.parent / f"{file_name_without_ext}.jsonl"
            if jsonl_path.exists():
                try:
                    jsonl_path.unlink()
                    logger.info(f"Deleted JSONL file: {jsonl_path}")
                except OSError as e:
                    logger.error(f"Failed to delete JSONL file {jsonl_path}: {e}")

    def _move_documents_to_workspace(
        self,
        documents: List[Document],
        target_workspace: Workspace,
        user: User
    ) -> None:
        """Move documents to another workspace with conflict resolution."""
        target_path = self.get_workspace_storage_path(target_workspace)
        target_path.mkdir(parents=True, exist_ok=True)

        # Also prepare output target path
        target_output_path = self.output_base_path / target_workspace.folder_path
        target_output_path.mkdir(parents=True, exist_ok=True)

        for doc in documents:
            original_filename = doc.filename
            original_file_path = doc.file_path  # Store before updating
            original_output_path = doc.output_path  # Store before updating
            new_filename = original_filename

            if original_file_path:
                # file_path is relative to storage_base_path
                old_file_path = self.storage_base_path / original_file_path
                if old_file_path.exists():
                    # Check for filename conflicts and generate unique name if needed
                    new_filename = self._generate_unique_filename(target_path, original_filename)
                    new_file_path = target_path / new_filename
                    try:
                        shutil.move(str(old_file_path), str(new_file_path))
                        # Store as relative path
                        doc.file_path = str(Path(target_workspace.folder_path) / new_filename)
                        # Update document filename if it was renamed
                        if new_filename != original_filename:
                            doc.filename = new_filename
                            logger.info(f"Renamed document due to conflict: {original_filename} -> {new_filename}")
                        logger.info(f"Moved input file from {old_file_path} to {new_file_path}")
                    except OSError as e:
                        logger.error(f"Failed to move file {old_file_path}: {e}")
                        continue  # Skip output folder move if input file move failed

            # Determine output folder name (use new filename stem if renamed)
            file_name_without_ext = Path(new_filename).stem
            original_stem = Path(original_filename).stem

            # Also move output folder if exists
            if original_output_path:
                old_output_path = self.output_base_path / original_output_path
                if old_output_path.exists() and old_output_path.is_dir():
                    # Check for output folder conflicts
                    new_output_folder_name = file_name_without_ext
                    new_output_path = target_output_path / new_output_folder_name
                    if new_output_path.exists():
                        # Generate unique output folder name
                        counter = 1
                        while new_output_path.exists():
                            new_output_folder_name = f"{file_name_without_ext}_{counter}"
                            new_output_path = target_output_path / new_output_folder_name
                            counter += 1
                    try:
                        shutil.move(str(old_output_path), str(new_output_path))
                        # Store as relative path
                        doc.output_path = str(Path(target_workspace.folder_path) / new_output_folder_name)
                        logger.info(f"Moved output folder from {old_output_path} to {new_output_path}")
                    except OSError as e:
                        logger.error(f"Failed to move output folder {old_output_path}: {e}")
            elif original_file_path:
                # Try fallback output path using original file path
                # Construct output path from the original source directory
                source_rel_dir = str(Path(original_file_path).parent)
                if source_rel_dir and source_rel_dir != '.':
                    old_output_path = self.output_base_path / source_rel_dir / original_stem
                else:
                    old_output_path = self.output_base_path / original_stem

                if old_output_path.exists() and old_output_path.is_dir():
                    new_output_folder_name = file_name_without_ext
                    new_output_path = target_output_path / new_output_folder_name
                    if new_output_path.exists():
                        counter = 1
                        while new_output_path.exists():
                            new_output_folder_name = f"{file_name_without_ext}_{counter}"
                            new_output_path = target_output_path / new_output_folder_name
                            counter += 1
                    try:
                        shutil.move(str(old_output_path), str(new_output_path))
                        doc.output_path = str(Path(target_workspace.folder_path) / new_output_folder_name)
                        logger.info(f"Moved output folder (fallback) from {old_output_path} to {new_output_path}")
                    except OSError as e:
                        logger.error(f"Failed to move output folder {old_output_path}: {e}")

            doc.workspace_id = target_workspace.id

        self.db.commit()

        # Update document counts
        self.workspace_repo.recalculate_document_count(target_workspace.id)

    def move_document_to_workspace(
        self,
        document: Document,
        target_workspace_id: UUID,
        user: User
    ) -> Optional[Document]:
        """
        Move a single document to another workspace.

        Args:
            document: Document to move
            target_workspace_id: Target workspace ID
            user: User performing the action

        Returns:
            Updated document or None if failed
        """
        target_workspace = self.workspace_repo.get_workspace_by_id(target_workspace_id)

        if not target_workspace:
            return None

        # Check target workspace belongs to user (or user is admin)
        if target_workspace.user_id != user.id:
            logger.warning(f"User {user.id} cannot move document to workspace {target_workspace_id}")
            return None

        old_workspace_id = document.workspace_id

        # Move physical input file
        if document.file_path:
            # file_path is relative to storage_base_path
            old_file_path = self.storage_base_path / document.file_path
            if old_file_path.exists():
                target_folder = self.get_workspace_storage_path(target_workspace)
                target_folder.mkdir(parents=True, exist_ok=True)
                new_file_path = target_folder / old_file_path.name

                try:
                    shutil.move(str(old_file_path), str(new_file_path))
                    # Store as relative path
                    document.file_path = str(Path(target_workspace.folder_path) / old_file_path.name)
                    logger.info(f"Moved input file from {old_file_path} to {new_file_path}")
                except OSError as e:
                    logger.error(f"Failed to move file: {e}")
                    return None

        # Move physical output folder
        file_name_without_ext = Path(document.filename).stem
        target_output_folder = self.output_base_path / target_workspace.folder_path
        target_output_folder.mkdir(parents=True, exist_ok=True)

        if document.output_path:
            old_output_path = self.output_base_path / document.output_path
            if old_output_path.exists() and old_output_path.is_dir():
                new_output_path = target_output_folder / file_name_without_ext
                try:
                    shutil.move(str(old_output_path), str(new_output_path))
                    document.output_path = str(Path(target_workspace.folder_path) / file_name_without_ext)
                    logger.info(f"Moved output folder from {old_output_path} to {new_output_path}")
                except OSError as e:
                    logger.error(f"Failed to move output folder: {e}")
        elif document.file_path:
            # Try fallback output path
            rel_dir = str(Path(document.file_path).parent)
            if rel_dir and rel_dir != '.':
                old_output_path = self.output_base_path / rel_dir / file_name_without_ext
            else:
                old_output_path = self.output_base_path / file_name_without_ext

            if old_output_path.exists() and old_output_path.is_dir():
                new_output_path = target_output_folder / file_name_without_ext
                try:
                    shutil.move(str(old_output_path), str(new_output_path))
                    document.output_path = str(Path(target_workspace.folder_path) / file_name_without_ext)
                    logger.info(f"Moved output folder (fallback) from {old_output_path} to {new_output_path}")
                except OSError as e:
                    logger.error(f"Failed to move output folder: {e}")

        # Update database
        document.workspace_id = target_workspace_id
        self.db.commit()
        self.db.refresh(document)

        # Update document counts
        if old_workspace_id:
            self.workspace_repo.update_document_count(old_workspace_id, delta=-1)
        self.workspace_repo.update_document_count(target_workspace_id, delta=1)

        logger.info(f"Moved document {document.id} to workspace {target_workspace_id}")
        return document

    def get_workspace_documents(
        self,
        workspace_id: UUID,
        limit: int = 100,
        offset: int = 0
    ) -> List[Document]:
        """Get documents in a workspace."""
        return self.workspace_repo.get_workspace_documents(
            workspace_id=workspace_id,
            limit=limit,
            offset=offset
        )

    def set_default_workspace(
        self,
        user_id: UUID,
        workspace_id: UUID
    ) -> bool:
        """Set a workspace as user's default."""
        return self.workspace_repo.set_default_workspace(
            user_id=user_id,
            workspace_id=workspace_id
        )

    def update_display_order(
        self,
        user_id: UUID,
        workspace_orders: List[dict]
    ) -> bool:
        """Update display order for workspaces."""
        return self.workspace_repo.update_display_order(
            user_id=user_id,
            workspace_orders=workspace_orders
        )

    def get_upload_path(
        self,
        user: User,
        workspace_id: Optional[UUID] = None,
        filename: str = ""
    ) -> Tuple[Path, Workspace]:
        """
        Get the upload path for a file.

        Args:
            user: User uploading the file
            workspace_id: Optional target workspace (uses default if not specified)
            filename: Filename for the upload

        Returns:
            Tuple of (upload_path, workspace)
        """
        logger.info(f"get_upload_path called: user={user.username} (normalized: {user.normalized_username}), workspace_id={workspace_id}, filename={filename}")

        if workspace_id:
            workspace = self.workspace_repo.get_workspace_by_id(workspace_id)
            logger.info(f"Looked up workspace by ID {workspace_id}: found={workspace is not None}")
            if workspace:
                logger.info(f"Workspace details: name={workspace.name}, user_id={workspace.user_id}, user.id={user.id}")
            if not workspace or workspace.user_id != user.id:
                logger.info(f"Workspace not found or user mismatch, using default workspace")
                workspace = self.get_or_create_default_workspace(user)
        else:
            logger.info(f"No workspace_id provided, using default workspace")
            workspace = self.get_or_create_default_workspace(user)

        logger.info(f"Final workspace: name={workspace.name}, folder_path={workspace.folder_path}")
        upload_path = self.get_workspace_storage_path(workspace)
        logger.info(f"Upload path computed: {upload_path} (storage_base={self.storage_base_path})")
        upload_path.mkdir(parents=True, exist_ok=True)

        if filename:
            upload_path = upload_path / filename

        return upload_path, workspace

    def sync_workspace_folders(self, user: User) -> dict:
        """
        Synchronize workspace folders with database.
        Creates missing folders and reports orphaned folders.

        Returns:
            Dict with sync results
        """
        results = {
            "created_folders": [],
            "orphaned_folders": [],
            "errors": []
        }

        user_storage = self.get_user_storage_path(user.normalized_username)

        # Get all workspaces from database
        workspaces = self.workspace_repo.get_user_workspaces(
            user_id=user.id,
            include_system=False
        )
        workspace_folders = {ws.folder_name for ws in workspaces}

        # Create missing folders
        for workspace in workspaces:
            folder_path = self.get_workspace_storage_path(workspace)
            if not folder_path.exists():
                try:
                    folder_path.mkdir(parents=True, exist_ok=True)
                    results["created_folders"].append(str(folder_path))
                except OSError as e:
                    results["errors"].append(f"Failed to create {folder_path}: {e}")

        # Find orphaned folders (folders without workspace record)
        if user_storage.exists():
            for item in user_storage.iterdir():
                if item.is_dir() and not item.name.startswith('_'):
                    if item.name not in workspace_folders:
                        results["orphaned_folders"].append(str(item))

        return results
