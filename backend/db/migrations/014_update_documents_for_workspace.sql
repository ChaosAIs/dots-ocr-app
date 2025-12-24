-- Migration: 014_update_documents_for_workspace.sql
-- Description: Add workspace reference and owner to documents table

-- Add workspace_id column
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS workspace_id UUID REFERENCES workspaces(id) ON DELETE SET NULL;

-- Add owner_id column
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS owner_id UUID REFERENCES users(id) ON DELETE SET NULL;

-- Add visibility column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'documents' AND column_name = 'visibility'
    ) THEN
        ALTER TABLE documents ADD COLUMN visibility VARCHAR(20) DEFAULT 'private';
        ALTER TABLE documents ADD CONSTRAINT check_visibility
            CHECK (visibility IN ('private', 'shared', 'public'));
    END IF;
END $$;

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_documents_workspace ON documents(workspace_id);
CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id);
CREATE INDEX IF NOT EXISTS idx_documents_visibility ON documents(visibility);

-- Trigger function to update workspace document count
CREATE OR REPLACE FUNCTION update_workspace_document_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.workspace_id IS NOT NULL THEN
            UPDATE workspaces SET document_count = document_count + 1
            WHERE id = NEW.workspace_id;
        END IF;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        IF OLD.workspace_id IS NOT NULL THEN
            UPDATE workspaces SET document_count = document_count - 1
            WHERE id = OLD.workspace_id;
        END IF;
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        IF OLD.workspace_id IS DISTINCT FROM NEW.workspace_id THEN
            IF OLD.workspace_id IS NOT NULL THEN
                UPDATE workspaces SET document_count = document_count - 1
                WHERE id = OLD.workspace_id;
            END IF;
            IF NEW.workspace_id IS NOT NULL THEN
                UPDATE workspaces SET document_count = document_count + 1
                WHERE id = NEW.workspace_id;
            END IF;
        END IF;
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
DROP TRIGGER IF EXISTS trigger_update_workspace_doc_count ON documents;
CREATE TRIGGER trigger_update_workspace_doc_count
    AFTER INSERT OR UPDATE OR DELETE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_workspace_document_count();
