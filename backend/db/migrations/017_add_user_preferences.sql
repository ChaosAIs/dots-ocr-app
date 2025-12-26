-- Migration: 017_add_user_preferences.sql
-- Description: Add preferences JSONB column to users table for storing user preferences
-- This includes chat workspace selections, UI settings, and other user preferences

-- Add preferences column to users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS preferences JSONB DEFAULT '{}'::jsonb;

-- Add GIN index for efficient JSON queries on preferences
CREATE INDEX IF NOT EXISTS idx_users_preferences ON users USING GIN(preferences);

-- Comment on column
COMMENT ON COLUMN users.preferences IS 'User preferences including chat workspace selections, UI settings, etc. Structure: { "chat": { "selectedWorkspaceIds": ["uuid1", "uuid2"], "lastUpdated": "timestamp" }, ... }';
