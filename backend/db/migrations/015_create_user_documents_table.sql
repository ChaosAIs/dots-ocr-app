-- Migration: 015_create_user_documents_table.sql
-- Description: Create user_documents table for document access permissions

-- Permission type enum
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'document_permission') THEN
        CREATE TYPE document_permission AS ENUM ('read', 'update', 'delete', 'share', 'full');
    END IF;
END $$;

-- Access origin enum
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'permission_origin') THEN
        CREATE TYPE permission_origin AS ENUM ('owner', 'shared', 'admin_granted', 'public');
    END IF;
END $$;

-- User documents permissions table
CREATE TABLE IF NOT EXISTS user_documents (
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
    share_message TEXT,

    -- Notification
    is_new BOOLEAN DEFAULT TRUE,
    viewed_at TIMESTAMP WITH TIME ZONE,

    -- Access tracking
    last_accessed_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER DEFAULT 0,

    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_user_document UNIQUE (user_id, document_id),
    CONSTRAINT valid_owner_permissions CHECK (
        (is_owner = TRUE AND 'full' = ANY(permissions)) OR is_owner = FALSE
    )
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_user_documents_user_id ON user_documents(user_id);
CREATE INDEX IF NOT EXISTS idx_user_documents_document_id ON user_documents(document_id);
CREATE INDEX IF NOT EXISTS idx_user_documents_user_doc ON user_documents(user_id, document_id);
CREATE INDEX IF NOT EXISTS idx_user_documents_owner ON user_documents(document_id) WHERE is_owner = TRUE;
CREATE INDEX IF NOT EXISTS idx_user_documents_expires ON user_documents(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_user_documents_permissions ON user_documents USING GIN(permissions);
CREATE INDEX IF NOT EXISTS idx_user_documents_new_shares ON user_documents(user_id) WHERE is_new = TRUE;
CREATE INDEX IF NOT EXISTS idx_user_documents_shared ON user_documents(user_id, origin) WHERE origin = 'shared';

-- Trigger for updated_at
DROP TRIGGER IF EXISTS update_user_documents_updated_at ON user_documents;
CREATE TRIGGER update_user_documents_updated_at
    BEFORE UPDATE ON user_documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
