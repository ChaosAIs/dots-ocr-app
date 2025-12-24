-- Migration: Add normalized_username to users table
-- This column stores the sanitized username for use in folder paths
-- The normalization process:
--   - Converts to lowercase
--   - Replaces spaces with underscores
--   - Removes special characters (keeps only alphanumeric and underscores)
--   - Ensures the name is filesystem-safe

-- Add normalized_username column to users table
ALTER TABLE users
ADD COLUMN IF NOT EXISTS normalized_username VARCHAR(100);

-- Create a function to normalize usernames
CREATE OR REPLACE FUNCTION normalize_name(input_name VARCHAR)
RETURNS VARCHAR AS $$
DECLARE
    result VARCHAR;
BEGIN
    -- Convert to lowercase
    result := LOWER(input_name);
    -- Replace spaces with underscores
    result := REPLACE(result, ' ', '_');
    -- Remove all characters except alphanumeric and underscores
    result := REGEXP_REPLACE(result, '[^a-z0-9_]', '', 'g');
    -- Ensure not empty - use 'user' as fallback
    IF result = '' OR result IS NULL THEN
        result := 'user';
    END IF;
    -- Limit to 100 characters
    result := LEFT(result, 100);
    RETURN result;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Populate normalized_username for existing users
UPDATE users
SET normalized_username = normalize_name(username)
WHERE normalized_username IS NULL;

-- Handle potential duplicates by appending a counter
DO $$
DECLARE
    rec RECORD;
    counter INTEGER;
    new_normalized VARCHAR;
BEGIN
    -- Find duplicates
    FOR rec IN (
        SELECT normalized_username, array_agg(id ORDER BY created_at) as ids
        FROM users
        GROUP BY normalized_username
        HAVING COUNT(*) > 1
    ) LOOP
        counter := 1;
        -- Skip the first one (oldest), update the rest
        FOR i IN 2..array_length(rec.ids, 1) LOOP
            new_normalized := rec.normalized_username || '_' || counter;
            -- Make sure the new name doesn't conflict
            WHILE EXISTS (SELECT 1 FROM users WHERE normalized_username = new_normalized) LOOP
                counter := counter + 1;
                new_normalized := rec.normalized_username || '_' || counter;
            END LOOP;
            -- Update the duplicate
            UPDATE users SET normalized_username = new_normalized WHERE id = rec.ids[i];
            counter := counter + 1;
        END LOOP;
    END LOOP;
END $$;

-- Now make the column NOT NULL and UNIQUE
ALTER TABLE users
ALTER COLUMN normalized_username SET NOT NULL;

-- Add unique constraint (only if it doesn't exist)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'users_normalized_username_key'
    ) THEN
        ALTER TABLE users ADD CONSTRAINT users_normalized_username_key UNIQUE (normalized_username);
    END IF;
END $$;

-- Create index for faster lookups (only if it doesn't exist)
CREATE INDEX IF NOT EXISTS idx_users_normalized_username ON users(normalized_username);

-- Create a trigger to auto-populate normalized_username on insert/update
CREATE OR REPLACE FUNCTION update_normalized_username()
RETURNS TRIGGER AS $$
BEGIN
    -- Only update if normalized_username is not provided or username changed
    IF NEW.normalized_username IS NULL OR
       (TG_OP = 'UPDATE' AND OLD.username != NEW.username AND NEW.normalized_username = OLD.normalized_username) THEN
        NEW.normalized_username := normalize_name(NEW.username);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop trigger if exists and recreate
DROP TRIGGER IF EXISTS trigger_update_normalized_username ON users;

CREATE TRIGGER trigger_update_normalized_username
    BEFORE INSERT OR UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_normalized_username();

-- Update workspace folder_path to use normalized_username
-- First, let's update existing workspaces to use the normalized username in folder_path
UPDATE workspaces w
SET folder_path = u.normalized_username || '/' || w.folder_name
FROM users u
WHERE w.user_id = u.id
AND w.folder_path != u.normalized_username || '/' || w.folder_name;

-- Add comment to document the column purpose
COMMENT ON COLUMN users.normalized_username IS 'Sanitized username for use in folder paths. Lowercase, spaces replaced with underscores, special chars removed.';
COMMENT ON FUNCTION normalize_name(VARCHAR) IS 'Normalizes a name for filesystem-safe usage: lowercase, spaces to underscores, remove special chars.';
