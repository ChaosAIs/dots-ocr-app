-- Migration: 023_add_workspace_display_name
-- Description: Add display_name column to workspaces table and extend folder_name length
-- This allows users to rename workspaces without affecting the physical folder structure.
-- The display_name supports any language/Unicode while folder_name remains ASCII-only.

-- Step 1: Add display_name column (nullable, supports Unicode)
-- This column stores the user-editable friendly name in any language
ALTER TABLE workspaces
ADD COLUMN IF NOT EXISTS display_name VARCHAR(100) NULL;

-- Step 2: Extend folder_name column from VARCHAR(50) to VARCHAR(100)
-- This accommodates the normalized name (85 chars) + timestamp suffix (15 chars)
ALTER TABLE workspaces
ALTER COLUMN folder_name TYPE VARCHAR(100);

-- Step 3: Add comment for documentation
COMMENT ON COLUMN workspaces.display_name IS 'User-editable display name (any language). Falls back to name if NULL.';
COMMENT ON COLUMN workspaces.folder_name IS 'Normalized ASCII folder name with timestamp suffix. Immutable after creation.';
COMMENT ON COLUMN workspaces.name IS 'Original workspace name at creation time (any language). Immutable.';

-- Verification query (optional - run manually to verify)
-- SELECT column_name, data_type, character_maximum_length, is_nullable
-- FROM information_schema.columns
-- WHERE table_name = 'workspaces'
-- AND column_name IN ('name', 'display_name', 'folder_name');
