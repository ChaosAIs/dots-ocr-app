# Workspace Display Name Feature - Design Document

## Overview

This document describes the design for adding a `display_name` column to the workspace management system, allowing users to rename workspaces without affecting the underlying physical folder structure.

## Problem Statement

Currently, when a user "renames" a workspace:
- The physical folder is renamed on the file system
- All document paths within the workspace are updated
- This is expensive and error-prone (file system operations can fail)
- Users cannot have a friendly display name while maintaining a stable folder structure
- Non-ASCII characters in workspace names can cause file system compatibility issues

## Goals

1. Allow users to rename workspaces freely without file system changes
2. Generate stable, unique, ASCII-only folder names at creation time
3. Support multi-language display names (Chinese, Japanese, Korean, etc.)
4. Protect system workspaces from modification
5. Maintain backward compatibility with existing workspaces

---

## Schema Design

### Current Schema

| Column | Type | Purpose |
|--------|------|---------|
| `name` | String(100) | Display name shown in UI |
| `folder_name` | String(50) | Physical folder name (sanitized) |
| `folder_path` | String(200) | Full relative path: `{username}/{folder_name}` |

### New Schema

| Column | Type | Purpose | Transliterated | Mutable |
|--------|------|---------|----------------|---------|
| `name` | String(100) | Original user input at creation (any language) | NO | NO |
| `display_name` | String(100) | User-editable friendly name (**any language**) | **NO** | YES |
| `folder_name` | String(**100**) | ASCII-only, normalized + timestamped | **YES** | NO |
| `folder_path` | String(200) | `{username}/{folder_name}` | YES | NO |

### Column Changes Summary

| Column | Current | New |
|--------|---------|-----|
| `folder_name` | String(50) | **String(100)** |
| `display_name` | N/A | **String(100)** (new column, supports Unicode) |

---

## Normalization Formula

When creating a workspace, generate `folder_name` from user input with **transliteration** and **timestamp**:

```
Input: "我的项目 Reports 2024"
                ↓
Step 1: Transliterate non-ASCII to English (using unidecode)
        "wo de xiang mu Reports 2024"
                ↓
Step 2: Lowercase
        "wo de xiang mu reports 2024"
                ↓
Step 3: Remove special characters (keep alphanumeric only)
        "wo de xiang mu reports 2024"
                ↓
Step 4: Replace spaces with underscores, collapse multiples
        "wo_de_xiang_mu_reports_2024"
                ↓
Step 5: Generate timestamp suffix
        "_20260102143025" (15 chars including underscore)
                ↓
Step 6: Truncate NAME portion to fit limit (100 - 15 = 85 chars max)
        "wo_de_xiang_mu_reports_2024" + "_20260102143025"
                ↓
Result: "wo_de_xiang_mu_reports_2024_20260102143025" (42 chars)
```

### Truncation Logic

| Component | Max Length | Notes |
|-----------|------------|-------|
| `folder_name` total | **100 chars** | Database column limit |
| Timestamp suffix | 15 chars | `_YYYYMMDDHHMMSS` (fixed, never truncated) |
| Name portion | **85 chars** | `100 - 15 = 85` (truncated if needed) |

**Key Principle**: Timestamp is **never** truncated - it guarantees uniqueness.

### Algorithm Implementation

```python
from datetime import datetime
import re
from unidecode import unidecode

def normalize_workspace_name(user_input: str) -> str:
    """
    Convert user input to a normalized, ASCII-only folder name with timestamp.

    Args:
        user_input: Original workspace name (any language)

    Returns:
        Normalized folder name (ASCII-only, timestamped, max 100 chars)
    """
    FOLDER_NAME_MAX_LENGTH = 100
    TIMESTAMP_LENGTH = 15  # "_YYYYMMDDHHMMSS"

    # Step 1: Transliterate non-ASCII characters to ASCII equivalents
    transliterated = unidecode(user_input)

    # Step 2-4: Lowercase and sanitize
    sanitized = transliterated.lower()
    sanitized = re.sub(r'[^a-z0-9]', '_', sanitized)
    sanitized = re.sub(r'_+', '_', sanitized)  # Collapse multiple underscores
    sanitized = sanitized.strip('_')           # Remove leading/trailing underscores

    # Step 5: Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    suffix = f"_{timestamp}"  # 15 chars

    # Step 6: Truncate name portion, preserve timestamp
    max_name_length = FOLDER_NAME_MAX_LENGTH - TIMESTAMP_LENGTH  # 85 chars
    truncated_name = sanitized[:max_name_length].rstrip('_')

    # Handle empty name edge case
    if not truncated_name:
        truncated_name = "workspace"

    return f"{truncated_name}{suffix}"
```

### Transliteration Examples

| Language | User Input | folder_name | display_name |
|----------|------------|-------------|--------------|
| Chinese | "我的项目" | `wo_de_xiang_mu_20260102143025` | "我的项目" (preserved) |
| Japanese | "プロジェクト" | `purojiekuto_20260102143025` | "プロジェクト" (preserved) |
| Korean | "내 프로젝트" | `nae_peulojekteu_20260102143025` | "내 프로젝트" (preserved) |
| Russian | "Мой проект" | `moj_proekt_20260102143025` | "Мой проект" (preserved) |
| Arabic | "مشروعي" | `mshrwy_20260102143025` | "مشروعي" (preserved) |
| Thai | "โครงการ" | `khrongkar_20260102143025` | "โครงการ" (preserved) |
| Mixed | "项目 Alpha テスト" | `xiang_mu_alpha_tesuto_20260102143025` | "项目 Alpha テスト" (preserved) |
| English | "My Project" | `my_project_20260102143025` | "My Project" (preserved) |
| Empty/Whitespace | "   " | `workspace_20260102143025` | "   " → NULL (fallback) |

---

## Display Name - Multi-Language Support

The `display_name` column preserves user's original language input **without any transliteration or modification**:

| Feature | folder_name | display_name |
|---------|-------------|--------------|
| Transliteration | YES (ASCII only) | **NO** (original preserved) |
| Special characters | Removed | **Allowed** |
| Unicode/Multi-language | Converted to English | **Preserved as-is** |
| Max length | 100 chars | 100 chars |
| Used for | File system path | UI display only |

### UI Display Logic

```python
def get_effective_display_name(workspace: Workspace) -> str:
    """Return display_name if set, otherwise fall back to name."""
    return workspace.display_name or workspace.name
```

### Display Name Examples

| Action | name | display_name | UI Shows |
|--------|------|--------------|----------|
| Create "财务报表" | "财务报表" | NULL | "财务报表" (from name) |
| Rename to "Q4 財務レポート" | "财务报表" | "Q4 財務レポート" | "Q4 財務レポート" |
| Rename to "Отчёт за квартал" | "财务报表" | "Отчёт за квартал" | "Отчёт за квартал" |
| Clear display_name | "财务报表" | NULL | "财务报表" (fallback to name) |

---

## System Workspace Handling

**System workspaces** (e.g., "Shared With Me") are protected and excluded from all rename/normalization logic:

| Attribute | System Workspace Behavior |
|-----------|--------------------------|
| `is_system` | `True` |
| `folder_name` | Fixed (e.g., `_shared_with_me`) - no timestamp, no transliteration |
| `display_name` | Not applicable - cannot be renamed |
| Rename operation | **Blocked** - return error/None |

### Protection Rules

| Operation | Regular Workspace | System Workspace |
|-----------|-------------------|------------------|
| Create | Transliterate + Normalize + Timestamp | Use fixed folder name |
| Rename (display_name) | Allowed (any language) | **Blocked** |
| Delete | Allowed (if not default) | **Blocked** |
| Update metadata (color, icon) | Allowed | Allowed |

### Validation in Rename

```python
def update_display_name(
    self,
    workspace_id: UUID,
    new_display_name: Optional[str]
) -> Optional[Workspace]:
    """
    Update workspace display name.

    Args:
        workspace_id: Workspace ID
        new_display_name: New display name (any language) or None to clear

    Returns:
        Updated workspace or None if blocked/not found
    """
    workspace = self.get_workspace_by_id(workspace_id)

    # Block rename for system workspaces
    if not workspace or workspace.is_system:
        return None

    workspace.display_name = new_display_name
    workspace.updated_at = datetime.utcnow()
    self.db.commit()
    self.db.refresh(workspace)

    return workspace
```

---

## Behavioral Changes

| Operation | Current Behavior | New Behavior |
|-----------|-----------------|--------------|
| **Create** | `name` → sanitize → `folder_name` | `name` → transliterate + normalize + timestamp → `folder_name` (immutable) |
| **Rename** | Updates `name`, `folder_name`, renames physical folder | Only updates `display_name`, **no file system changes** |
| **Display** | Shows `name` | Shows `display_name` if set, else `name` |
| **System WS** | Cannot delete | Cannot delete **AND cannot rename** |

---

## Complete Flow Example

### Chinese User Creating and Renaming Workspace

```
Step 1: User creates workspace "财务报表 2024"
                ↓
Database after creation:
  - name: "财务报表 2024"              ← Original (Chinese preserved)
  - display_name: NULL
  - folder_name: "cai_wu_bao_biao_2024_20260102143025"  ← Transliterated + timestamped
  - folder_path: "john_doe/cai_wu_bao_biao_2024_20260102143025"
                ↓
UI displays: "财务报表 2024" (from name, since display_name is NULL)
File system: input/john_doe/cai_wu_bao_biao_2024_20260102143025/

Step 2: User renames to "Q4 財務レポート 报告"
                ↓
Database after rename:
  - name: "财务报表 2024"              ← Unchanged
  - display_name: "Q4 財務レポート 报告"  ← New value (mixed languages preserved)
  - folder_name: "cai_wu_bao_biao_2024_20260102143025"  ← Unchanged
  - folder_path: "john_doe/cai_wu_bao_biao_2024_20260102143025"  ← Unchanged
                ↓
UI displays: "Q4 財務レポート 报告" (from display_name)
File system: UNCHANGED

Step 3: User clears display_name
                ↓
Database:
  - display_name: NULL
                ↓
UI displays: "财务报表 2024" (fallback to name)
File system: UNCHANGED
```

---

## API Contract

### Create Workspace

```http
POST /workspaces
Content-Type: application/json

{
    "name": "我的项目 Reports"
}
```

**Response:**
```json
{
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "我的项目 Reports",
    "display_name": null,
    "folder_name": "wo_de_xiang_mu_reports_20260102143025",
    "folder_path": "john_doe/wo_de_xiang_mu_reports_20260102143025",
    "effective_name": "我的项目 Reports",
    "is_system": false,
    "created_at": "2026-01-02T14:30:25Z"
}
```

### Rename Workspace (Update Display Name)

```http
PATCH /workspaces/{id}
Content-Type: application/json

{
    "display_name": "Q4 財務レポート"
}
```

**Response:**
```json
{
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "我的项目 Reports",
    "display_name": "Q4 財務レポート",
    "folder_name": "wo_de_xiang_mu_reports_20260102143025",
    "folder_path": "john_doe/wo_de_xiang_mu_reports_20260102143025",
    "effective_name": "Q4 財務レポート",
    "is_system": false
}
```

### Rename System Workspace (Blocked)

```http
PATCH /workspaces/{system_workspace_id}
Content-Type: application/json

{
    "display_name": "My Shared Files"
}
```

**Response:**
```json
{
    "error": "Cannot rename system workspace",
    "code": "SYSTEM_WORKSPACE_IMMUTABLE"
}
```

---

## Key Benefits

| Benefit | Description |
|---------|-------------|
| **Stability** | Folder paths never change after creation |
| **Performance** | No file system operations on rename |
| **Reliability** | No risk of failed folder rename corrupting state |
| **Uniqueness** | Timestamp guarantees no folder name collisions |
| **Flexibility** | Users can rename freely without consequences |
| **I18n Support** | Display names support any language/script |
| **Compatibility** | ASCII-only folder names work on all file systems |

---

## Migration Strategy

### Database Migration

1. Add `display_name` column (nullable, String(100))
2. Alter `folder_name` column from String(50) to String(100)
3. No data migration needed - existing workspaces use `name` as fallback for display

### Migration Script

```python
"""Add display_name column and extend folder_name length.

Revision ID: xxx
"""
from alembic import op
import sqlalchemy as sa

def upgrade():
    # Add display_name column (nullable, supports Unicode)
    op.add_column(
        'workspaces',
        sa.Column('display_name', sa.String(100), nullable=True)
    )

    # Extend folder_name column length
    op.alter_column(
        'workspaces',
        'folder_name',
        existing_type=sa.String(50),
        type_=sa.String(100),
        existing_nullable=False
    )

def downgrade():
    op.drop_column('workspaces', 'display_name')
    op.alter_column(
        'workspaces',
        'folder_name',
        existing_type=sa.String(100),
        type_=sa.String(50),
        existing_nullable=False
    )
```

---

## Dependencies

### New Dependency

Add `unidecode` library for transliteration:

```bash
pip install unidecode
```

**Package Info:**
- **Name**: Unidecode
- **Purpose**: Converts Unicode text to ASCII transliteration
- **Supports**: Chinese (Pinyin), Japanese (Romaji), Korean, Cyrillic, Arabic, Thai, Greek, etc.
- **Downloads**: 30M+ monthly
- **License**: GPL-2.0

---

## Files Requiring Changes

| File | Changes |
|------|---------|
| `backend/db/models.py` | Add `display_name` column, change `folder_name` to String(100), update `to_dict()` |
| `backend/db/workspace_repository.py` | Replace `sanitize_folder_name()` with `normalize_workspace_name()`, update create/rename logic |
| `backend/services/workspace_service.py` | Remove physical folder rename logic, add system workspace guard |
| `backend/services/workspace_api.py` | Update API endpoints, return error for system workspace rename |
| `requirements.txt` | Add `unidecode` dependency |
| `backend/db/migrations/xxx_add_display_name.py` | New migration file |

---

## Edge Cases

| Case | Handling |
|------|----------|
| Empty display_name | Falls back to original `name` |
| NULL display_name | Falls back to original `name` |
| Clear display_name | Set to NULL, shows original `name` |
| Duplicate display_name | Allowed (only folder names need uniqueness) |
| Unicode in name | Preserved in `name` and `display_name`, transliterated for `folder_name` |
| Very long name (>85 chars) | Name portion truncated, timestamp preserved |
| Only special characters | Fallback to "workspace" + timestamp |
| System workspace rename | Return error, no changes made |
| Emoji in name | Transliterated or removed for folder, preserved in display_name |

---

## Testing Plan

### Unit Tests

1. `normalize_workspace_name()` function
   - ASCII input
   - Chinese input
   - Japanese input
   - Mixed language input
   - Special characters only
   - Empty/whitespace input
   - Very long input (>85 chars)
   - Emoji input

2. `update_display_name()` function
   - Regular workspace rename
   - System workspace rename (should fail)
   - Clear display_name (set to NULL)
   - Unicode display_name

3. `to_dict()` effective name logic
   - display_name set → return display_name
   - display_name NULL → return name

### Integration Tests

1. Create workspace with Chinese name → verify folder created with transliterated name
2. Rename workspace → verify folder unchanged
3. Attempt to rename system workspace → verify error returned
4. Create multiple workspaces with same name → verify unique folder names (timestamps differ)

---

## Rollback Plan

If issues arise:

1. **Database**: Run downgrade migration to remove `display_name` column and revert `folder_name` length
2. **Code**: Revert to previous version of affected files
3. **No data loss**: Existing workspaces continue to work (folder_name unchanged)

---

## Future Considerations

1. **Bulk rename**: Allow renaming multiple workspaces at once
2. **Rename history**: Track display_name changes for audit
3. **Search**: Index display_name for search functionality
4. **Localization**: Consider locale-specific transliteration rules

---

## Implementation Status

**Completed:** January 2, 2026

### Files Changed

| File | Status | Changes |
|------|--------|---------|
| `backend/requirements.txt` | ✅ Done | Added `unidecode` dependency |
| `backend/db/migrations/023_add_workspace_display_name.sql` | ✅ Done | Migration to add `display_name` column and extend `folder_name` to 100 chars |
| `backend/db/models.py` | ✅ Done | Added `display_name` column, `get_effective_display_name()` method, updated `to_dict()` |
| `backend/db/workspace_repository.py` | ✅ Done | Added `normalize_workspace_folder_name()`, updated `create_workspace()`, `update_workspace()`, added `update_display_name()` |
| `backend/services/workspace_service.py` | ✅ Done | Updated `update_workspace()`, added `update_display_name()`, deprecated old `rename_workspace()` |
| `backend/services/workspace_api.py` | ✅ Done | Updated request/response models, updated endpoints for display_name |

### Migration Command

Run the migration to apply database changes:

```bash
psql -U your_user -d your_database -f backend/db/migrations/023_add_workspace_display_name.sql
```

### Backward Compatibility

- Existing workspaces will continue to work
- `display_name` defaults to NULL, falling back to `name` for display
- Old `rename_workspace()` methods are deprecated but still functional (they now only update `display_name`)
