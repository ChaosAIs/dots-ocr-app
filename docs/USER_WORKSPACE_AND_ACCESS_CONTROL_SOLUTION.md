# User Workspace & Document Access Control Solution

## Executive Summary

This document outlines a comprehensive solution for implementing:
1. **Workspace Management** - One-level folder structure where each workspace maps to a physical folder
2. **User-Isolated File Storage** - Physical file storage organized by username with workspace subfolders
3. **Role-Based Access Control (RBAC)** - Document-level permissions with sharing capabilities

**Key Design Principle**: Each workspace directly corresponds to a physical folder on the file system. When a workspace is created, a folder is created. When deleted, the folder and its contents are deleted. Moving documents between workspaces moves the physical files.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [File System Design](#2-file-system-design)
3. [Database Design](#3-database-design)
4. [Permission Model](#4-permission-model)
5. [Backend Implementation](#5-backend-implementation)
6. [Vector & Graph Search Filtering](#6-vector--graph-search-filtering)
7. [API Endpoints](#7-api-endpoints)
8. [Frontend UI Design](#8-frontend-ui-design)
9. [Migration Strategy](#9-migration-strategy)
10. [Security Considerations](#10-security-considerations)

---

## 1. Architecture Overview

### 1.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Workspace   │  │  Document    │  │    Share     │  │    Chat      │        │
│  │  Manager     │  │  Browser     │  │   Manager    │  │  Interface   │        │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               API LAYER                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Workspace   │  │  Document    │  │  Permission  │  │    RAG       │        │
│  │  Routes      │  │  Routes      │  │   Routes     │  │   Routes     │        │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             SERVICE LAYER                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Workspace   │  │  Document    │  │  Permission  │  │   Search     │        │
│  │  Service     │  │  Service     │  │   Service    │  │  Filter Svc  │        │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
┌─────────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│      FILE SYSTEM        │  │     POSTGRESQL      │  │   VECTOR/GRAPH DB   │
│  ┌───────────────────┐  │  │  ┌───────────────┐  │  │  ┌───────────────┐  │
│  │ /storage/         │  │  │  │ users         │  │  │  │ Qdrant        │  │
│  │   └── john_doe/   │  │  │  │ workspaces    │  │  │  │ (vectors)     │  │
│  │       ├── default/│  │  │  │ documents     │  │  │  └───────────────┘  │
│  │       ├── project/│  │  │  │ user_documents│  │  │  ┌───────────────┐  │
│  │       └── output/ │  │  │  └───────────────┘  │  │  │ Neo4j         │  │
│  └───────────────────┘  │  └─────────────────────┘  │  │ (graph)       │  │
└─────────────────────────┘  └─────────────────────────┘  └───────────────────┘
```

### 1.2 Key Concepts

| Concept | Description |
|---------|-------------|
| **Workspace** | Maps 1:1 to a physical folder under user's directory |
| **Workspace Folder** | Physical subfolder named after workspace (normalized) |
| **User Directory** | Root folder for user, named by username |
| **Output Folder** | Shared output folder for all processed/converted files |
| **Document Owner** | User who uploaded the document (has full permissions) |
| **Shared Document** | Document accessible via explicit permission grant (shortcut only, no file copy) |
| **Admin Access** | Full access to all workspaces and documents |

### 1.3 Workspace-Folder Synchronization Rules

| Action | File System Effect |
|--------|-------------------|
| Create Workspace | Create folder: `/{username}/{workspace_folder}/` |
| Delete Workspace | Delete folder and all files inside |
| Rename Workspace | Rename folder, update all document paths |
| Upload to Workspace | Save file to workspace folder |
| Move Document | Move file from source folder to target folder |
| Delete Document | Delete file from workspace folder |

---

## 2. File System Design

### 2.1 Directory Structure

```
/storage/                                    # Root storage directory (configurable)
├── john_doe/                                # User folder (based on username)
│   ├── my_documents/                        # Default workspace folder
│   │   ├── report_q4.pdf                    # Original uploaded files
│   │   ├── budget_2024.xlsx
│   │   └── proposal.docx
│   │
│   ├── project_alpha/                       # Custom workspace folder
│   │   ├── specs_v1.pdf
│   │   ├── timeline.xlsx
│   │   └── meeting_notes.docx
│   │
│   ├── research/                            # Another workspace folder
│   │   ├── paper_draft.pdf
│   │   └── data_analysis.xlsx
│   │
│   └── _output/                             # Shared output folder (processed files)
│       ├── report_q4/                       # Per-document output
│       │   ├── page_001.md
│       │   ├── page_002.md
│       │   └── metadata.json
│       ├── budget_2024/
│       │   └── ...
│       └── specs_v1/
│           └── ...
│
├── jane_smith/                              # Another user's folder
│   ├── my_documents/
│   ├── client_projects/
│   └── _output/
│
└── _system/                                 # System folder (optional)
    └── templates/
```

### 2.2 Path Structure

| Component | Example | Description |
|-----------|---------|-------------|
| `storage_root` | `/storage` | Configurable base path (env var) |
| `username` | `john_doe` | Sanitized from user's username |
| `workspace_folder` | `project_alpha` | Normalized workspace name |
| `output_folder` | `_output` | Fixed name, prefixed with `_` |
| `input_file_path` | `john_doe/project_alpha/specs.pdf` | Relative path stored in DB |
| `output_file_path` | `john_doe/_output/specs/` | Relative path stored in DB |

### 2.3 Folder Naming Rules

```python
def normalize_folder_name(name: str) -> str:
    """
    Convert workspace/user name to safe folder name.
    - Lowercase
    - Replace spaces with underscore
    - Remove special characters
    - Limit length to 50 chars
    - Ensure no conflicts with reserved names
    """
    import re

    # Lowercase and replace spaces
    safe = name.lower().strip()
    safe = re.sub(r'\s+', '_', safe)

    # Remove special characters (keep alphanumeric and underscore)
    safe = re.sub(r'[^a-z0-9_]', '', safe)

    # Collapse multiple underscores
    safe = re.sub(r'_+', '_', safe).strip('_')

    # Limit length
    safe = safe[:50]

    # Reserved names check
    reserved = {'_output', '_system', '_shared', 'con', 'prn', 'aux', 'nul'}
    if safe in reserved:
        safe = f"{safe}_ws"

    return safe or 'workspace'

# Examples:
# "My Documents" -> "my_documents"
# "Project Alpha!" -> "project_alpha"
# "Research & Development" -> "research_development"
# "_output" -> "_output_ws" (reserved name protection)
```

### 2.4 Path Resolution

```python
class PathResolver:
    """Resolve file system paths for workspaces and documents."""

    def __init__(self, storage_root: str = '/storage'):
        self.storage_root = Path(storage_root)

    def get_user_root(self, username: str) -> Path:
        """Get user's root directory."""
        return self.storage_root / normalize_folder_name(username)

    def get_workspace_path(self, username: str, workspace_folder: str) -> Path:
        """Get workspace directory path."""
        return self.get_user_root(username) / workspace_folder

    def get_output_path(self, username: str) -> Path:
        """Get user's output directory."""
        return self.get_user_root(username) / '_output'

    def get_document_input_path(
        self,
        username: str,
        workspace_folder: str,
        filename: str
    ) -> Path:
        """Get full path for document input file."""
        return self.get_workspace_path(username, workspace_folder) / filename

    def get_document_output_path(
        self,
        username: str,
        doc_name: str
    ) -> Path:
        """Get full path for document output directory."""
        return self.get_output_path(username) / doc_name

    def get_relative_input_path(
        self,
        username: str,
        workspace_folder: str,
        filename: str
    ) -> str:
        """Get relative path for storage in database."""
        return f"{normalize_folder_name(username)}/{workspace_folder}/{filename}"

    def get_relative_output_path(
        self,
        username: str,
        doc_name: str
    ) -> str:
        """Get relative output path for storage in database."""
        return f"{normalize_folder_name(username)}/_output/{doc_name}"
```

---

## 3. Database Design

### 3.1 Complete Entity Relationship Diagram

```
┌─────────────────┐
│     users       │
├─────────────────┤
│ id (PK)         │──────────────────────────────────────────────────────┐
│ username        │  ◄── Used to derive folder name                      │
│ email           │                                                       │
│ role            │                                                       │
│ ...             │                                                       │
└─────────────────┘                                                       │
        │                                                                 │
        │ 1:N                                                             │
        ▼                                                                 │
┌───────────────────────────┐                                             │
│       workspaces          │                                             │
├───────────────────────────┤                                             │
│ id (PK)                   │                                             │
│ user_id (FK)              │─────────────────────────────────────────────┤
│ name                      │  ◄── Display name: "Project Alpha"          │
│ folder_name               │  ◄── Physical folder: "project_alpha"       │
│ folder_path               │  ◄── Relative: "john_doe/project_alpha"     │
│ description               │                                             │
│ color                     │                                             │
│ icon                      │                                             │
│ is_default                │                                             │
│ is_system                 │  ◄── TRUE for "Shared With Me" (virtual)   │
│ display_order             │                                             │
│ document_count            │  ◄── Cached count for performance          │
│ created_at, updated_at    │                                             │
└───────────────────────────┘                                             │
        │                                                                 │
        │ 1:N                                                             │
        ▼                                                                 │
┌───────────────────────────────────────────────────────────────┐        │
│                        documents                               │        │
├───────────────────────────────────────────────────────────────┤        │
│ id (PK)                                                        │        │
│ workspace_id (FK)        ◄── Which workspace contains this doc │        │
│ owner_id (FK)            ◄── Who owns this document            │────────┘
│ filename                  ◄── Unique filename: "report_q4.pdf"  │
│ original_filename         ◄── User's original: "Report Q4.pdf" │
│ file_path                 ◄── "john_doe/project_alpha/report.pdf"│
│ output_path               ◄── "john_doe/_output/report_q4/"    │
│ visibility                ◄── private/shared/public            │
│ file_size, mime_type                                           │
│ upload_status, convert_status, index_status                    │
│ document_metadata (JSONB)                                      │
│ created_at, updated_at, deleted_at                             │
└───────────────────────────────────────────────────────────────┘
        │
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                     user_documents                             │
├───────────────────────────────────────────────────────────────┤
│ id (PK)                                                        │
│ user_id (FK)              ◄── Who has access                   │
│ document_id (FK)          ◄── Which document                   │
│ permissions[]             ◄── [read, update, delete, share, full]│
│ origin                    ◄── owner/shared/admin_granted       │
│ is_owner                  ◄── TRUE if document owner           │
│ shared_by (FK)            ◄── Who granted access               │
│ shared_at, expires_at                                          │
│ is_new                    ◄── For new share notification       │
│ last_accessed_at, access_count                                 │
│ created_at, updated_at                                         │
└───────────────────────────────────────────────────────────────┘
```

### 3.2 Tables Definition

#### `workspaces` Table

```sql
-- Migration: 012_create_workspaces_table.sql

CREATE TABLE workspaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Owner reference
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Workspace identity
    name VARCHAR(100) NOT NULL,              -- Display name: "Project Alpha"
    folder_name VARCHAR(50) NOT NULL,        -- Folder name: "project_alpha"
    folder_path VARCHAR(200) NOT NULL,       -- Full relative path: "john_doe/project_alpha"

    -- Metadata
    description TEXT,
    color VARCHAR(7) DEFAULT '#6366f1',      -- Hex color for UI
    icon VARCHAR(50) DEFAULT 'folder',       -- Icon identifier

    -- Flags
    is_default BOOLEAN DEFAULT FALSE,        -- User's default workspace
    is_system BOOLEAN DEFAULT FALSE,         -- System workspace (e.g., "Shared With Me")

    -- Cached stats
    document_count INTEGER DEFAULT 0,        -- Cached for performance

    -- Display
    display_order INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT unique_workspace_name_per_user UNIQUE (user_id, name),
    CONSTRAINT unique_folder_path UNIQUE (folder_path),
    CONSTRAINT unique_folder_per_user UNIQUE (user_id, folder_name),
    CONSTRAINT valid_color CHECK (color ~ '^#[0-9A-Fa-f]{6}$'),
    CONSTRAINT valid_folder_name CHECK (folder_name ~ '^[a-z0-9_]+$')
);

-- Indexes
CREATE INDEX idx_workspaces_user_id ON workspaces(user_id);
CREATE INDEX idx_workspaces_folder_path ON workspaces(folder_path);
CREATE INDEX idx_workspaces_user_default ON workspaces(user_id) WHERE is_default = TRUE;
CREATE INDEX idx_workspaces_display_order ON workspaces(user_id, display_order);

-- Trigger for updated_at
CREATE TRIGGER update_workspaces_updated_at
    BEFORE UPDATE ON workspaces
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Ensure only one default workspace per user
CREATE UNIQUE INDEX idx_one_default_workspace_per_user
    ON workspaces(user_id) WHERE is_default = TRUE AND is_system = FALSE;
```

#### Updated `documents` Table

```sql
-- Migration: 013_update_documents_table.sql

-- Add workspace reference and owner
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS workspace_id UUID REFERENCES workspaces(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS owner_id UUID REFERENCES users(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS visibility VARCHAR(20) DEFAULT 'private'
        CHECK (visibility IN ('private', 'shared', 'public'));

-- Update file_path and output_path to use new structure
-- file_path format: "{username}/{workspace_folder}/{filename}"
-- output_path format: "{username}/_output/{doc_name}/"

-- Indexes
CREATE INDEX IF NOT EXISTS idx_documents_workspace ON documents(workspace_id);
CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id);
CREATE INDEX IF NOT EXISTS idx_documents_visibility ON documents(visibility);

-- Trigger to update workspace document count
CREATE OR REPLACE FUNCTION update_workspace_document_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE workspaces SET document_count = document_count + 1
        WHERE id = NEW.workspace_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE workspaces SET document_count = document_count - 1
        WHERE id = OLD.workspace_id;
    ELSIF TG_OP = 'UPDATE' AND OLD.workspace_id != NEW.workspace_id THEN
        UPDATE workspaces SET document_count = document_count - 1
        WHERE id = OLD.workspace_id;
        UPDATE workspaces SET document_count = document_count + 1
        WHERE id = NEW.workspace_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_workspace_doc_count
    AFTER INSERT OR UPDATE OR DELETE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_workspace_document_count();
```

#### `user_documents` Table (Permissions Bridge)

```sql
-- Migration: 014_create_user_documents_table.sql

-- Permission type enum
CREATE TYPE document_permission AS ENUM ('read', 'update', 'delete', 'share', 'full');

-- Access origin enum
CREATE TYPE permission_origin AS ENUM ('owner', 'shared', 'admin_granted', 'public');

CREATE TABLE user_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Core references
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Permission details
    permissions document_permission[] NOT NULL DEFAULT '{read}',
    origin permission_origin NOT NULL DEFAULT 'shared',
    is_owner BOOLEAN NOT NULL DEFAULT FALSE,

    -- Sharing metadata
    shared_by UUID REFERENCES users(id) ON DELETE SET NULL,
    shared_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    share_message TEXT,                      -- Optional message from sharer

    -- Notification
    is_new BOOLEAN DEFAULT TRUE,             -- For "new share" indicator
    viewed_at TIMESTAMP WITH TIME ZONE,

    -- Access tracking
    last_accessed_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER DEFAULT 0,

    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT unique_user_document UNIQUE (user_id, document_id),
    CONSTRAINT valid_owner_permissions CHECK (
        (is_owner = TRUE AND permissions @> '{full}') OR is_owner = FALSE
    )
);

-- Indexes
CREATE INDEX idx_user_documents_user_id ON user_documents(user_id);
CREATE INDEX idx_user_documents_document_id ON user_documents(document_id);
CREATE INDEX idx_user_documents_user_doc ON user_documents(user_id, document_id);
CREATE INDEX idx_user_documents_owner ON user_documents(document_id) WHERE is_owner = TRUE;
CREATE INDEX idx_user_documents_expires ON user_documents(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX idx_user_documents_permissions ON user_documents USING GIN(permissions);
CREATE INDEX idx_user_documents_new_shares ON user_documents(user_id) WHERE is_new = TRUE;
CREATE INDEX idx_user_documents_shared ON user_documents(user_id, origin) WHERE origin = 'shared';

-- Trigger for updated_at
CREATE TRIGGER update_user_documents_updated_at
    BEFORE UPDATE ON user_documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

### 3.3 System Workspaces

When a user is created, automatically create:

| Workspace | folder_name | is_default | is_system | Has Physical Folder? |
|-----------|-------------|------------|-----------|---------------------|
| My Documents | my_documents | TRUE | FALSE | YES |
| Shared With Me | _shared_with_me | FALSE | TRUE | NO (virtual) |

**Note**: "Shared With Me" is a **virtual workspace** - it has no physical folder. It aggregates documents shared by other users via the `user_documents` table.

---

## 4. Permission Model

### 4.1 Permission Types

| Permission | Description | Allows |
|------------|-------------|--------|
| `read` | View document | View content, chat with document, see in search |
| `update` | Modify document | Edit markdown, re-index, update metadata |
| `delete` | Remove document | Delete document and all related data |
| `share` | Share with others | Grant read/update permissions to others |
| `full` | Full access | All permissions (owner default) |

### 4.2 Role-Based Access Matrix

| Action | Admin | Owner | Shared (read) | Shared (update) | Shared (share) | No Access |
|--------|-------|-------|---------------|-----------------|----------------|-----------|
| View document | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Chat with document | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| See in vector search | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| See in graph search | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Edit markdown | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Re-index document | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Delete document | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Move to workspace | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Share document | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| Revoke access | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |

### 4.3 Access Resolution Logic

```python
def check_document_access(user_id: UUID, document_id: UUID, required_permission: str) -> bool:
    """
    Access resolution order:
    1. Admin bypass - admins have full access to all documents
    2. Check user_documents table for explicit permission
    3. Default deny - no record = no access
    """
    user = get_user(user_id)

    # 1. Admin bypass
    if user.role == 'admin':
        return True

    # 2. Check user_documents table
    user_doc = get_user_document(user_id, document_id)

    if user_doc is None:
        return False  # No access record = denied

    # 3. Check expiration
    if user_doc.expires_at and user_doc.expires_at < now():
        return False

    # 4. Check permission
    if 'full' in user_doc.permissions:
        return True

    return required_permission in user_doc.permissions
```

---

## 5. Backend Implementation

### 5.1 Workspace Service

```python
# backend/services/workspace_service.py

from typing import List, Optional
from uuid import UUID
from pathlib import Path
import shutil
import os

class WorkspaceService:
    """Service for workspace management with file system synchronization."""

    def __init__(self, db: Session, storage_root: str = '/storage'):
        self.db = db
        self.storage_root = Path(storage_root)
        self.workspace_repo = WorkspaceRepository(db)
        self.document_repo = DocumentRepository(db)
        self.path_resolver = PathResolver(storage_root)

    # ==================== Workspace CRUD ====================

    async def create_workspace(
        self,
        user: User,
        name: str,
        description: Optional[str] = None,
        color: str = '#6366f1',
        icon: str = 'folder'
    ) -> Workspace:
        """
        Create a new workspace.
        1. Normalize folder name
        2. Create physical folder
        3. Create database record
        """
        # Normalize folder name
        folder_name = normalize_folder_name(name)
        username_folder = normalize_folder_name(user.username)

        # Check for name conflict
        existing = await self.workspace_repo.get_by_folder_name(user.id, folder_name)
        if existing:
            # Append number to make unique
            counter = 1
            while existing:
                folder_name = f"{normalize_folder_name(name)}_{counter}"
                existing = await self.workspace_repo.get_by_folder_name(user.id, folder_name)
                counter += 1

        # Build folder path
        folder_path = f"{username_folder}/{folder_name}"
        full_path = self.storage_root / folder_path

        # Create physical folder
        full_path.mkdir(parents=True, exist_ok=True)

        # Get next display order
        order = await self.workspace_repo.get_next_display_order(user.id)

        # Create database record
        workspace = await self.workspace_repo.create(
            user_id=user.id,
            name=name,
            folder_name=folder_name,
            folder_path=folder_path,
            description=description,
            color=color,
            icon=icon,
            display_order=order,
            is_default=False,
            is_system=False
        )

        return workspace

    async def rename_workspace(
        self,
        workspace_id: UUID,
        user: User,
        new_name: str
    ) -> Workspace:
        """
        Rename a workspace.
        1. Validate ownership
        2. Normalize new folder name
        3. Rename physical folder
        4. Update all document paths
        5. Update database record
        """
        workspace = await self.workspace_repo.get_by_id(workspace_id)

        if not workspace:
            raise WorkspaceNotFoundError()

        # Check ownership (or admin)
        if workspace.user_id != user.id and user.role != 'admin':
            raise PermissionDeniedError()

        # Prevent renaming system workspaces
        if workspace.is_system:
            raise InvalidOperationError("Cannot rename system workspaces")

        # Normalize new folder name
        new_folder_name = normalize_folder_name(new_name)
        username_folder = normalize_folder_name(user.username)

        # Check for conflict
        if new_folder_name != workspace.folder_name:
            existing = await self.workspace_repo.get_by_folder_name(user.id, new_folder_name)
            if existing:
                raise WorkspaceExistsError(f"Folder name '{new_folder_name}' already exists")

        # Get paths
        old_path = self.storage_root / workspace.folder_path
        new_folder_path = f"{username_folder}/{new_folder_name}"
        new_path = self.storage_root / new_folder_path

        # Rename physical folder
        if old_path.exists() and new_folder_name != workspace.folder_name:
            old_path.rename(new_path)

        # Update all document paths in this workspace
        documents = await self.document_repo.get_by_workspace(workspace_id)
        for doc in documents:
            new_file_path = doc.file_path.replace(
                workspace.folder_path,
                new_folder_path
            )
            await self.document_repo.update_path(doc.id, new_file_path)

        # Update database record
        workspace = await self.workspace_repo.update(
            workspace_id,
            name=new_name,
            folder_name=new_folder_name,
            folder_path=new_folder_path
        )

        return workspace

    async def delete_workspace(
        self,
        workspace_id: UUID,
        user: User,
        delete_documents: bool = True
    ) -> bool:
        """
        Delete a workspace and optionally its documents.

        Args:
            workspace_id: Workspace to delete
            user: Current user
            delete_documents: If True, delete all documents. If False, move to default.
        """
        workspace = await self.workspace_repo.get_by_id(workspace_id)

        if not workspace:
            raise WorkspaceNotFoundError()

        # Check ownership
        if workspace.user_id != user.id and user.role != 'admin':
            raise PermissionDeniedError()

        # Prevent deleting system/default workspaces
        if workspace.is_system:
            raise InvalidOperationError("Cannot delete system workspaces")

        if workspace.is_default:
            raise InvalidOperationError("Cannot delete default workspace")

        if delete_documents:
            # Delete all documents in workspace
            documents = await self.document_repo.get_by_workspace(workspace_id)
            for doc in documents:
                await self.delete_document_files(doc)
                # Also remove from vector/graph stores
                await self.vectorstore.delete_by_source(doc.filename)
                await self.graph_store.delete_by_source(doc.filename)

            # Delete physical folder and all contents
            folder_path = self.storage_root / workspace.folder_path
            if folder_path.exists():
                shutil.rmtree(folder_path)

        else:
            # Move documents to default workspace
            default_ws = await self.workspace_repo.get_default(user.id)
            documents = await self.document_repo.get_by_workspace(workspace_id)

            for doc in documents:
                await self.move_document(doc.id, workspace_id, default_ws.id, user)

            # Delete empty folder
            folder_path = self.storage_root / workspace.folder_path
            if folder_path.exists():
                folder_path.rmdir()  # Only works if empty

        # Delete database record (cascades to documents if delete_documents=True)
        return await self.workspace_repo.delete(workspace_id)

    # ==================== Document Operations ====================

    async def upload_document(
        self,
        user: User,
        workspace_id: UUID,
        file: UploadFile
    ) -> Document:
        """
        Upload a document to a workspace.
        1. Validate workspace ownership
        2. Generate unique filename
        3. Save file to workspace folder
        4. Create document record
        5. Create owner permission record
        """
        workspace = await self.workspace_repo.get_by_id(workspace_id)

        if not workspace:
            raise WorkspaceNotFoundError()

        if workspace.user_id != user.id:
            raise PermissionDeniedError("Cannot upload to another user's workspace")

        if workspace.is_system:
            raise InvalidOperationError("Cannot upload to system workspace")

        # Generate unique filename
        filename = self._generate_unique_filename(
            file.filename,
            workspace.folder_path
        )
        doc_name = Path(filename).stem

        # Build paths
        file_path = f"{workspace.folder_path}/{filename}"
        output_path = f"{normalize_folder_name(user.username)}/_output/{doc_name}"

        # Save physical file
        full_path = self.storage_root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)

        # Create output directory
        output_full_path = self.storage_root / output_path
        output_full_path.mkdir(parents=True, exist_ok=True)

        # Create document record
        document = await self.document_repo.create(
            workspace_id=workspace_id,
            owner_id=user.id,
            filename=filename,
            original_filename=file.filename,
            file_path=file_path,
            output_path=output_path,
            file_size=file.size,
            mime_type=file.content_type,
            visibility='private'
        )

        # Create owner permission record
        await self.user_doc_repo.grant_access(
            user_id=user.id,
            document_id=document.id,
            permissions=['full'],
            origin='owner',
            is_owner=True
        )

        return document

    async def move_document(
        self,
        document_id: UUID,
        from_workspace_id: UUID,
        to_workspace_id: UUID,
        user: User
    ) -> Document:
        """
        Move a document between workspaces.
        1. Validate ownership of both workspaces
        2. Move physical file
        3. Update document record
        """
        document = await self.document_repo.get_by_id(document_id)
        from_ws = await self.workspace_repo.get_by_id(from_workspace_id)
        to_ws = await self.workspace_repo.get_by_id(to_workspace_id)

        if not document or not from_ws or not to_ws:
            raise NotFoundError()

        # Check ownership
        if document.owner_id != user.id and user.role != 'admin':
            raise PermissionDeniedError("Only owner can move documents")

        if to_ws.user_id != user.id and user.role != 'admin':
            raise PermissionDeniedError("Cannot move to another user's workspace")

        if to_ws.is_system:
            raise InvalidOperationError("Cannot move to system workspace")

        # Calculate new path
        new_file_path = f"{to_ws.folder_path}/{document.filename}"

        # Move physical file
        old_full_path = self.storage_root / document.file_path
        new_full_path = self.storage_root / new_file_path

        if old_full_path.exists():
            new_full_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_full_path), str(new_full_path))

        # Update document record
        document = await self.document_repo.update(
            document_id,
            workspace_id=to_workspace_id,
            file_path=new_file_path
        )

        return document

    async def delete_document(
        self,
        document_id: UUID,
        user: User
    ) -> bool:
        """
        Delete a document.
        1. Check permission
        2. Delete physical files (input + output)
        3. Remove from vector/graph stores
        4. Delete database record
        """
        document = await self.document_repo.get_by_id(document_id)

        if not document:
            raise DocumentNotFoundError()

        # Check permission
        has_permission = await self.permission_service.check_access(
            user.id, document_id, 'delete'
        )
        if not has_permission:
            raise PermissionDeniedError()

        # Delete input file
        input_path = self.storage_root / document.file_path
        if input_path.exists():
            input_path.unlink()

        # Delete output directory
        output_path = self.storage_root / document.output_path
        if output_path.exists():
            shutil.rmtree(output_path)

        # Remove from vector store
        await self.vectorstore.delete_by_source(document.filename)

        # Remove from graph store
        await self.graph_store.delete_by_source(document.filename)

        # Delete database record (cascades to user_documents)
        return await self.document_repo.delete(document_id)

    # ==================== User Initialization ====================

    async def initialize_user_storage(self, user: User) -> None:
        """
        Initialize storage for a new user.
        1. Create user folder
        2. Create default workspace folder
        3. Create output folder
        4. Create database records
        """
        username_folder = normalize_folder_name(user.username)
        user_path = self.storage_root / username_folder

        # Create user directory structure
        (user_path / 'my_documents').mkdir(parents=True, exist_ok=True)
        (user_path / '_output').mkdir(parents=True, exist_ok=True)

        # Create default workspace record
        await self.workspace_repo.create(
            user_id=user.id,
            name="My Documents",
            folder_name="my_documents",
            folder_path=f"{username_folder}/my_documents",
            description="Your default workspace",
            icon="folder",
            is_default=True,
            is_system=False,
            display_order=0
        )

        # Create "Shared With Me" virtual workspace
        await self.workspace_repo.create(
            user_id=user.id,
            name="Shared With Me",
            folder_name="_shared_with_me",
            folder_path="",  # No physical folder
            description="Documents shared by others",
            icon="share",
            is_default=False,
            is_system=True,
            display_order=999  # Always at bottom
        )

    # ==================== Helper Methods ====================

    def _generate_unique_filename(
        self,
        original_filename: str,
        workspace_folder_path: str
    ) -> str:
        """Generate unique filename within workspace."""
        base = Path(original_filename).stem
        ext = Path(original_filename).suffix
        safe_base = normalize_folder_name(base)

        filename = f"{safe_base}{ext}"
        full_path = self.storage_root / workspace_folder_path / filename

        counter = 1
        while full_path.exists():
            filename = f"{safe_base}_{counter}{ext}"
            full_path = self.storage_root / workspace_folder_path / filename
            counter += 1

        return filename
```

### 5.2 Permission Service

```python
# backend/services/permission_service.py

class PermissionService:
    """Service for document permission operations."""

    def __init__(self, db: Session):
        self.db = db
        self.user_doc_repo = UserDocumentRepository(db)
        self.user_repo = UserRepository(db)

    async def check_access(
        self,
        user_id: UUID,
        document_id: UUID,
        required_permission: str = 'read'
    ) -> bool:
        """Main access check method."""
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            return False

        # Admin bypass
        if user.role == 'admin':
            return True

        # Check user_documents table
        user_doc = await self.user_doc_repo.get_user_document(user_id, document_id)

        if user_doc is None:
            return False

        # Check expiration
        if user_doc.expires_at and user_doc.expires_at < datetime.utcnow():
            return False

        # Check permission
        if 'full' in user_doc.permissions:
            return True

        return required_permission in user_doc.permissions

    async def share_document(
        self,
        sharer_id: UUID,
        document_id: UUID,
        target_user_id: UUID,
        permissions: List[str],
        expires_at: Optional[datetime] = None,
        message: Optional[str] = None
    ) -> UserDocument:
        """Share document with another user."""
        # Verify sharer has share permission
        if not await self.check_access(sharer_id, document_id, 'share'):
            raise PermissionDenied("You don't have permission to share this document")

        # Prevent sharing with self
        if sharer_id == target_user_id:
            raise InvalidOperationError("Cannot share with yourself")

        # Validate permissions (non-admin can only grant read/update)
        sharer = await self.user_repo.get_by_id(sharer_id)
        if sharer.role != 'admin':
            allowed = {'read', 'update'}
            invalid = set(permissions) - allowed
            if invalid:
                raise InvalidPermission(f"Cannot grant: {invalid}")

        # Create or update permission record
        return await self.user_doc_repo.grant_access(
            user_id=target_user_id,
            document_id=document_id,
            permissions=permissions,
            origin='shared',
            shared_by=sharer_id,
            shared_at=datetime.utcnow(),
            expires_at=expires_at,
            is_owner=False,
            is_new=True,
            share_message=message
        )

    async def get_shared_with_me(
        self,
        user_id: UUID
    ) -> List[DocumentWithShareInfo]:
        """Get documents shared with user."""
        return await self.user_doc_repo.get_shared_documents(user_id)

    async def get_accessible_document_ids(
        self,
        user_id: UUID,
        user_role: str
    ) -> Optional[List[UUID]]:
        """
        Get document IDs user can access.
        Returns None for admin (no filter needed).
        """
        if user_role == 'admin':
            return None

        return await self.user_doc_repo.get_accessible_ids(user_id)
```

---

## 6. Vector & Graph Search Filtering

### 6.1 Search Filter Service

```python
# backend/rag_service/search_filter.py

class SearchFilterService:
    """Service for building access-filtered search queries."""

    def __init__(self, permission_service: PermissionService, doc_repo: DocumentRepository):
        self.permission_service = permission_service
        self.doc_repo = doc_repo

    async def get_accessible_sources(
        self,
        user_id: UUID,
        user_role: str
    ) -> Optional[Set[str]]:
        """
        Get set of source filenames user can access.
        Returns None for admin (no filtering needed).
        """
        if user_role == 'admin':
            return None

        accessible_doc_ids = await self.permission_service.get_accessible_document_ids(
            user_id, user_role
        )

        if not accessible_doc_ids:
            return set()

        filenames = await self.doc_repo.get_filenames_by_ids(accessible_doc_ids)
        return set(filenames)

    def build_qdrant_filter(
        self,
        accessible_sources: Optional[Set[str]]
    ) -> Optional[Filter]:
        """Build Qdrant filter for vector search."""
        if accessible_sources is None:
            return None  # Admin - no filter

        if not accessible_sources:
            # No access - return impossible filter
            return Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchAny(any=["__NO_ACCESS__"])
                    )
                ]
            )

        return Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchAny(any=list(accessible_sources))
                )
            ]
        )

    def build_neo4j_filter(
        self,
        accessible_sources: Optional[Set[str]]
    ) -> str:
        """Build Neo4j WHERE clause."""
        if accessible_sources is None:
            return ""

        if not accessible_sources:
            return "WHERE false"

        sources_list = ", ".join([f"'{s}'" for s in accessible_sources])
        return f"WHERE n.source IN [{sources_list}]"
```

### 6.2 Workspace-Scoped Search

```python
async def search_within_workspace(
    self,
    query: str,
    user_id: UUID,
    workspace_id: UUID,
    top_k: int = 10
) -> List[SearchResult]:
    """Search only within documents in a specific workspace."""
    # Get documents in workspace
    documents = await self.doc_repo.get_by_workspace(workspace_id)
    filenames = [doc.filename for doc in documents]

    if not filenames:
        return []

    # Build filter
    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchAny(any=filenames)
            )
        ]
    )

    results = self.qdrant_client.search(
        collection_name=self.collection_name,
        query_vector=self.embed_query(query),
        query_filter=qdrant_filter,
        limit=top_k
    )

    return self._process_results(results)
```

---

## 7. API Endpoints

### 7.1 Workspace Endpoints

```python
# backend/routes/workspace_routes.py

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])

@router.get("/")
async def list_workspaces(
    current_user: User = Depends(get_current_user)
) -> List[WorkspaceResponse]:
    """List all workspaces for current user."""
    pass

@router.post("/")
async def create_workspace(
    request: CreateWorkspaceRequest,
    current_user: User = Depends(get_current_user)
) -> WorkspaceResponse:
    """
    Create a new workspace.
    - Creates physical folder
    - Creates database record
    """
    pass

@router.get("/{workspace_id}")
async def get_workspace(
    workspace_id: UUID,
    current_user: User = Depends(get_current_user)
) -> WorkspaceDetailResponse:
    """Get workspace details with documents."""
    pass

@router.put("/{workspace_id}")
async def update_workspace(
    workspace_id: UUID,
    request: UpdateWorkspaceRequest,
    current_user: User = Depends(get_current_user)
) -> WorkspaceResponse:
    """
    Update workspace.
    - If name changes, renames physical folder
    - Updates all document paths
    """
    pass

@router.delete("/{workspace_id}")
async def delete_workspace(
    workspace_id: UUID,
    delete_documents: bool = True,
    current_user: User = Depends(get_current_user)
):
    """
    Delete a workspace.
    - If delete_documents=True: Deletes folder and all files
    - If delete_documents=False: Moves files to default workspace
    """
    pass

@router.get("/{workspace_id}/documents")
async def list_workspace_documents(
    workspace_id: UUID,
    current_user: User = Depends(get_current_user)
) -> List[DocumentResponse]:
    """List documents in a workspace."""
    pass

@router.post("/{workspace_id}/upload")
async def upload_to_workspace(
    workspace_id: UUID,
    file: UploadFile,
    current_user: User = Depends(get_current_user)
) -> DocumentResponse:
    """
    Upload document to specific workspace.
    - Saves file to workspace folder
    - Creates document record
    - Creates owner permission
    """
    pass
```

### 7.2 Document Endpoints

```python
# backend/routes/document_routes.py

router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.post("/{document_id}/move")
async def move_document(
    document_id: UUID,
    request: MoveDocumentRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Move document to another workspace.
    - Moves physical file
    - Updates document path
    """
    pass

@router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user)
):
    """
    Delete document.
    - Deletes physical files (input + output)
    - Removes from vector/graph stores
    - Deletes database records
    """
    pass

@router.post("/{document_id}/share")
async def share_document(
    document_id: UUID,
    request: ShareDocumentRequest,
    current_user: User = Depends(get_current_user)
):
    """Share document with another user (no file copy)."""
    pass

@router.get("/shared-with-me")
async def get_shared_with_me(
    current_user: User = Depends(get_current_user)
) -> List[SharedDocumentResponse]:
    """Get documents shared with current user."""
    pass
```

### 7.3 Request/Response Models

```python
# backend/models/workspace_models.py

class CreateWorkspaceRequest(BaseModel):
    name: str                           # Display name
    description: Optional[str] = None
    color: str = '#6366f1'
    icon: str = 'folder'

class UpdateWorkspaceRequest(BaseModel):
    name: Optional[str] = None          # Changing name renames folder
    description: Optional[str] = None
    color: Optional[str] = None
    icon: Optional[str] = None

class WorkspaceResponse(BaseModel):
    id: UUID
    name: str
    folder_name: str                    # Physical folder name
    folder_path: str                    # Full relative path
    description: Optional[str]
    color: str
    icon: str
    is_default: bool
    is_system: bool
    document_count: int
    display_order: int
    created_at: datetime

class MoveDocumentRequest(BaseModel):
    target_workspace_id: UUID

class ShareDocumentRequest(BaseModel):
    user_id: UUID
    permissions: List[str] = ['read']   # read, update
    expires_at: Optional[datetime] = None
    message: Optional[str] = None

class SharedDocumentResponse(BaseModel):
    document_id: UUID
    filename: str
    original_filename: str
    owner_username: str
    permissions: List[str]
    shared_by_username: str
    shared_at: datetime
    expires_at: Optional[datetime]
    message: Optional[str]
    is_new: bool
```

---

## 8. Frontend UI Design

### 8.1 Component Architecture

```
src/components/
├── workspace/
│   ├── WorkspaceSidebar.jsx       # Left sidebar with workspace list
│   ├── WorkspaceItem.jsx          # Single workspace with context menu
│   ├── WorkspaceCreateDialog.jsx  # Create new workspace
│   ├── WorkspaceEditDialog.jsx    # Edit/rename workspace
│   ├── WorkspaceDeleteDialog.jsx  # Delete with options
│   └── WorkspaceContextMenu.jsx   # Right-click menu
│
├── documents/
│   ├── DocumentBrowser.jsx        # Main document grid/list view
│   ├── DocumentCard.jsx           # Document card with actions
│   ├── DocumentUpload.jsx         # Upload to current workspace
│   ├── DocumentMoveDialog.jsx     # Move to workspace dialog
│   └── DocumentContextMenu.jsx    # Right-click menu
│
├── sharing/
│   ├── ShareDialog.jsx            # Share document dialog
│   ├── SharedWithMeView.jsx       # Virtual workspace view
│   ├── UserSearchSelect.jsx       # Search users to share
│   └── PermissionSelector.jsx     # Permission checkboxes
│
└── common/
    ├── PermissionBadge.jsx        # Owner/Shared indicator
    └── ConfirmDialog.jsx          # Confirmation dialogs
```

### 8.2 Main Layout

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  🔷 Dots OCR                           🔍 Search...    👤 John Doe ▼   [Logout] │
├────────────────────┬────────────────────────────────────────────────────────────┤
│                    │                                                            │
│  WORKSPACES        │  📁 My Documents                           [+ Upload]      │
│  ─────────────     │  ────────────────────────────────────────────────────────  │
│                    │  Path: /john_doe/my_documents/                             │
│  ┌──────────────┐  │                                                            │
│  │ 📁 My Docs   │◄─┤  ┌─────────────────────────────────────────────────────┐   │
│  │    12 files  │  │  │  📄 Report_Q4.pdf        📄 Budget.xlsx             │   │
│  └──────────────┘  │  │  👑 Owner | 2.4 MB       👑 Owner | 1.2 MB          │   │
│                    │  │  ✅ Indexed              ⏳ Converting               │   │
│  ┌──────────────┐  │  │                                                     │   │
│  │ 🟣 Project A │  │  │  📄 Proposal.docx        📄 Notes.pdf               │   │
│  │    5 files   │  │  │  👑 Owner | 856 KB       👑 Owner | 1.8 MB          │   │
│  └──────────────┘  │  │  ✅ Indexed              ✅ Indexed                 │   │
│                    │  └─────────────────────────────────────────────────────┘   │
│  ┌──────────────┐  │                                                            │
│  │ 🔬 Research  │  │  Showing 4 of 12 files                        [Load More]  │
│  │    8 files   │  │                                                            │
│  └──────────────┘  │                                                            │
│                    │                                                            │
│  ─────────────     │                                                            │
│                    │                                                            │
│  ┌──────────────┐  │                                                            │
│  │ 🔗 Shared    │  │                                                            │
│  │    3 🔴2     │◄─┼── Virtual workspace (no folder)                           │
│  └──────────────┘  │                                                            │
│                    │                                                            │
│  ─────────────     │                                                            │
│  [+ New Workspace] │                                                            │
│                    │                                                            │
└────────────────────┴────────────────────────────────────────────────────────────┘
```

### 8.3 Create Workspace Dialog

```
┌─────────────────────────────────────────────────────────────┐
│  Create New Workspace                                  [✕]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Workspace Name *                                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Project Alpha                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  📁 Folder will be created: project_alpha                   │
│                                                             │
│  Description (optional)                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Documents for the Alpha project                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Color                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 🔵 🟣 🟢 🟡 🟠 🔴 ⚫                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Icon                                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 📁 📂 📦 🔬 💼 📊 📈                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│                              [Cancel]  [Create Workspace]   │
└─────────────────────────────────────────────────────────────┘
```

### 8.4 Delete Workspace Dialog

```
┌─────────────────────────────────────────────────────────────┐
│  Delete Workspace                                      [✕]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ⚠️ You are about to delete workspace "Project Alpha"      │
│                                                             │
│  This workspace contains 5 documents.                       │
│  📁 Folder: /john_doe/project_alpha/                        │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  What would you like to do with the documents?              │
│                                                             │
│  ○ Delete all documents                                     │
│    ⚠️ This will permanently delete 5 files and their       │
│    processed data. This cannot be undone.                   │
│                                                             │
│  ● Move documents to "My Documents"                         │
│    Files will be moved to your default workspace.           │
│                                                             │
│                                                             │
│                              [Cancel]  [Delete Workspace]   │
└─────────────────────────────────────────────────────────────┘
```

### 8.5 Move Document Dialog

```
┌─────────────────────────────────────────────────────────────┐
│  Move Document                                         [✕]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📄 Report_Q4.pdf                                           │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  Current location:                                          │
│  📁 My Documents (/john_doe/my_documents/)                  │
│                                                             │
│  Move to:                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │   ○  📁 My Documents        (current)               │   │
│  │   ●  🟣 Project Alpha                               │   │
│  │   ○  🔬 Research                                    │   │
│  │   ○  📦 Archive                                     │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  📁 New location: /john_doe/project_alpha/Report_Q4.pdf    │
│                                                             │
│                                  [Cancel]  [Move Document]  │
└─────────────────────────────────────────────────────────────┘
```

### 8.6 Shared With Me View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🔗 Shared With Me                                         [Mark All Read]  │
│  ────────────────────────────────────────────────────────────────────────── │
│  ℹ️ These documents are shared by other users. No files are stored here.   │
│                                                                              │
│  📌 NEW (2)                                                                  │
│  ────────────                                                               │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 📄 Project_Roadmap.pdf                                    🆕 NEW      │  │
│  │ ──────────────────────────────────────────────────────────────────── │  │
│  │ 👤 Owner: john.doe         Shared: 2 hours ago                        │  │
│  │ 💬 "Please review the roadmap"                                        │  │
│  │ 🏷️ [Read] [Edit]           ⏰ Never expires                          │  │
│  │                                                                       │  │
│  │ [View] [Edit] [Chat]                                                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  PREVIOUSLY SHARED                                                          │
│  ─────────────────                                                          │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 📄 Marketing_Strategy.docx                                            │  │
│  │ 👤 jane.smith  │  📅 Dec 10  │  🏷️ [Read Only]  │  [View] [Chat]      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.7 Document Context Menu

```
Right-click on document:

For OWNED documents:
┌─────────────────────┐
│ 👁️ View             │
│ ✏️ Edit             │
│ 💬 Chat             │
│ ────────────────── │
│ 📤 Share            │
│ 📁 Move to...       │
│ ────────────────── │
│ 🔄 Re-index         │
│ ────────────────── │
│ 🗑️ Delete           │
└─────────────────────┘

For SHARED documents:
┌─────────────────────┐
│ 👁️ View             │
│ ✏️ Edit             │  ◄── Only if has 'update' permission
│ 💬 Chat             │
│ ────────────────── │
│ ℹ️ Sharing Info     │
└─────────────────────┘
```

### 8.8 Workspace Context Menu

```
Right-click on workspace:

┌─────────────────────┐
│ 📂 Open             │
│ ────────────────── │
│ ✏️ Rename           │  ◄── Renames folder too
│ 🎨 Change Color     │
│ ────────────────── │
│ 📤 Move Up          │
│ 📥 Move Down        │
│ ────────────────── │
│ 🗑️ Delete           │  ◄── Not for default/system
└─────────────────────┘
```

---

## 9. Migration Strategy

### 9.1 Migration Steps

```sql
-- Execute in order:

-- 1. Create workspaces table
-- (012_create_workspaces_table.sql)

-- 2. Update documents table
-- (013_update_documents_table.sql)

-- 3. Create user_documents table
-- (014_create_user_documents_table.sql)
```

### 9.2 Data Migration Script

```python
async def migrate_existing_data():
    """Migrate existing documents to workspace structure."""

    # 1. For each existing user, create workspace structure
    users = await user_repo.get_all()

    for user in users:
        username_folder = normalize_folder_name(user.username)
        user_path = storage_root / username_folder

        # Create directories
        (user_path / 'my_documents').mkdir(parents=True, exist_ok=True)
        (user_path / '_output').mkdir(parents=True, exist_ok=True)

        # Create default workspace
        default_ws = await workspace_repo.create(
            user_id=user.id,
            name="My Documents",
            folder_name="my_documents",
            folder_path=f"{username_folder}/my_documents",
            is_default=True
        )

        # Create shared workspace
        await workspace_repo.create(
            user_id=user.id,
            name="Shared With Me",
            folder_name="_shared_with_me",
            folder_path="",
            is_system=True
        )

    # 2. Move existing documents
    old_input = Path('/backend/input')
    old_output = Path('/backend/output')

    documents = await doc_repo.get_all()

    for doc in documents:
        owner = await user_repo.get_by_id(doc.owner_id or doc.created_by)
        if not owner:
            continue

        username_folder = normalize_folder_name(owner.username)
        default_ws = await workspace_repo.get_default(owner.id)

        # Move input file
        old_file = old_input / doc.filename
        new_file = storage_root / username_folder / 'my_documents' / doc.filename

        if old_file.exists():
            shutil.move(str(old_file), str(new_file))

        # Move output folder
        doc_stem = Path(doc.filename).stem
        old_out = old_output / doc_stem
        new_out = storage_root / username_folder / '_output' / doc_stem

        if old_out.exists():
            shutil.move(str(old_out), str(new_out))

        # Update document record
        await doc_repo.update(
            doc.id,
            workspace_id=default_ws.id,
            file_path=f"{username_folder}/my_documents/{doc.filename}",
            output_path=f"{username_folder}/_output/{doc_stem}"
        )

        # Create owner permission
        await user_doc_repo.grant_access(
            user_id=owner.id,
            document_id=doc.id,
            permissions=['full'],
            origin='owner',
            is_owner=True
        )
```

### 9.3 Implementation Phases

#### Phase 1: Database & File Structure
- [ ] Create migration files
- [ ] Implement path resolver
- [ ] Implement workspace service
- [ ] Add file system operations

#### Phase 2: Workspace Backend
- [ ] Implement workspace CRUD with folder sync
- [ ] Implement document upload to workspace
- [ ] Implement document move between workspaces
- [ ] Update delete to remove files

#### Phase 3: Permission System
- [ ] Implement user_documents table
- [ ] Implement permission service
- [ ] Add permission checks to endpoints
- [ ] Implement sharing endpoints

#### Phase 4: Search Filtering
- [ ] Update vector search with filtering
- [ ] Update graph search with filtering
- [ ] Add workspace-scoped search

#### Phase 5: Frontend
- [ ] Implement workspace sidebar
- [ ] Implement document browser
- [ ] Implement create/rename/delete dialogs
- [ ] Implement move document dialog
- [ ] Implement sharing UI

#### Phase 6: Data Migration
- [ ] Migrate existing users and documents
- [ ] Verify file integrity
- [ ] Update all paths in database

---

## 10. Security Considerations

### 10.1 File System Security

1. **Path Traversal Prevention**
   ```python
   def validate_path(path: str, allowed_root: Path) -> bool:
       """Ensure path doesn't escape allowed root."""
       resolved = (allowed_root / path).resolve()
       return resolved.is_relative_to(allowed_root.resolve())
   ```

2. **Folder Name Sanitization**
   - Only allow alphanumeric and underscore
   - No `.` or `..` components
   - Reserved name protection

3. **User Isolation**
   - Each user's files in separate directory
   - No cross-user file access

### 10.2 Permission Validation

1. Always validate on backend before file operations
2. Check workspace ownership before folder operations
3. Check document ownership/permission before file access
4. Audit all permission changes

### 10.3 Sharing Security

1. Shared documents are references only (no file copy)
2. Revoking access immediately removes permission
3. Expiration checked on every access
4. Cannot share with higher permissions than you have

---

## Summary

This solution provides:

1. **Workspace-Folder Mapping**
   - Each workspace = one physical folder
   - Create workspace = create folder
   - Delete workspace = delete folder + files
   - Rename workspace = rename folder + update paths

2. **Simplified File Structure**
   - `/storage/{username}/{workspace_folder}/` for input files
   - `/storage/{username}/_output/` for processed files
   - No `users.storage_path` column needed

3. **Document Access Control**
   - Owner has full access
   - Sharing via `user_documents` table (no file copy)
   - Admin bypasses all permissions

4. **Complete UI**
   - Workspace sidebar with folders
   - Document browser with move/share actions
   - Dialogs for all operations
   - "Shared With Me" virtual workspace
