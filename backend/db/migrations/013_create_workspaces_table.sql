-- Migration: 013_create_workspaces_table.sql
-- Description: Create workspaces table for user document organization
-- Each workspace maps to a physical folder on the file system

CREATE TABLE IF NOT EXISTS workspaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Owner reference
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Workspace identity
    name VARCHAR(100) NOT NULL,              -- Display name: "Project Alpha"
    folder_name VARCHAR(50) NOT NULL,        -- Physical folder name: "project_alpha"
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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_workspace_name_per_user UNIQUE (user_id, name),
    CONSTRAINT unique_folder_per_user UNIQUE (user_id, folder_name),
    CONSTRAINT valid_color CHECK (color ~ '^#[0-9A-Fa-f]{6}$')
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_workspaces_user_id ON workspaces(user_id);
CREATE INDEX IF NOT EXISTS idx_workspaces_folder_path ON workspaces(folder_path);
CREATE INDEX IF NOT EXISTS idx_workspaces_user_default ON workspaces(user_id) WHERE is_default = TRUE;
CREATE INDEX IF NOT EXISTS idx_workspaces_display_order ON workspaces(user_id, display_order);

-- Trigger for updated_at
DROP TRIGGER IF EXISTS update_workspaces_updated_at ON workspaces;
CREATE TRIGGER update_workspaces_updated_at
    BEFORE UPDATE ON workspaces
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Ensure only one default workspace per user (excluding system workspaces)
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_default_workspace_per_user
    ON workspaces(user_id) WHERE is_default = TRUE AND is_system = FALSE;
