-- Migration: 018_unified_document_intelligence.sql
-- Description: Add tables for unified document intelligence solution
--              - Structured data extraction
--              - Conversational analytics sessions
--              - Schema definitions and semantic mappings
-- Date: 2024-12-25

-- ============================================================================
-- PART 1: EXTEND DOCUMENTS TABLE FOR EXTRACTION
-- ============================================================================

-- Add extraction-related columns to documents table
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS extraction_eligible BOOLEAN DEFAULT NULL,
ADD COLUMN IF NOT EXISTS extraction_status VARCHAR(20) DEFAULT 'pending',
ADD COLUMN IF NOT EXISTS extraction_schema_type VARCHAR(64),
ADD COLUMN IF NOT EXISTS extraction_started_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS extraction_completed_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS extraction_error TEXT;

-- Create enum for extraction status if needed
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'extraction_status_enum') THEN
        CREATE TYPE extraction_status_enum AS ENUM ('pending', 'processing', 'completed', 'failed', 'skipped');
    END IF;
END$$;

-- ============================================================================
-- PART 2: DATA SCHEMAS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS data_schemas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Schema identification
    schema_type VARCHAR(64) NOT NULL UNIQUE,
    schema_version VARCHAR(16) DEFAULT '1.0',
    domain VARCHAR(32) NOT NULL,
    display_name VARCHAR(128),
    description TEXT,

    -- Schema definitions (JSON Schema format)
    header_schema JSONB NOT NULL DEFAULT '{}',
    line_items_schema JSONB,
    summary_schema JSONB,

    -- Extraction configuration
    extraction_prompt TEXT,
    field_mappings JSONB DEFAULT '{}',

    -- Validation rules
    validation_rules JSONB DEFAULT '{}',

    -- Metadata
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on schema_type
CREATE INDEX IF NOT EXISTS idx_data_schemas_type ON data_schemas(schema_type);
CREATE INDEX IF NOT EXISTS idx_data_schemas_domain ON data_schemas(domain);

-- ============================================================================
-- PART 3: MAIN EXTRACTED DATA TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS documents_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Schema Reference
    schema_type VARCHAR(64) NOT NULL,
    schema_version VARCHAR(16) DEFAULT '1.0',

    -- Extracted Data (structured JSON)
    header_data JSONB NOT NULL DEFAULT '{}',
    line_items JSONB DEFAULT '[]',
    summary_data JSONB DEFAULT '{}',

    -- For large tables (> 500 rows)
    line_items_storage VARCHAR(16) DEFAULT 'inline',
    line_items_count INTEGER DEFAULT 0,

    -- Validation & Quality
    validation_status VARCHAR(20) DEFAULT 'pending',
    overall_confidence DECIMAL(5,4),
    field_confidences JSONB DEFAULT '{}',
    validation_results JSONB,

    -- Extraction Metadata
    extraction_method VARCHAR(32),
    extraction_model VARCHAR(64),
    extraction_duration_ms INTEGER,
    extraction_metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- One extraction per document
    CONSTRAINT uq_documents_data_document UNIQUE(document_id)
);

-- Create indexes for querying
CREATE INDEX IF NOT EXISTS idx_documents_data_document ON documents_data(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_data_schema ON documents_data(schema_type);
CREATE INDEX IF NOT EXISTS idx_documents_data_validation ON documents_data(validation_status);
CREATE INDEX IF NOT EXISTS idx_documents_data_header ON documents_data USING GIN (header_data);
CREATE INDEX IF NOT EXISTS idx_documents_data_line_items ON documents_data USING GIN (line_items);

-- ============================================================================
-- PART 4: LINE ITEMS OVERFLOW TABLE (for large tables)
-- ============================================================================

CREATE TABLE IF NOT EXISTS documents_data_line_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    documents_data_id UUID NOT NULL REFERENCES documents_data(id) ON DELETE CASCADE,
    line_number INTEGER NOT NULL,
    data JSONB NOT NULL,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT uq_line_items_data_line UNIQUE(documents_data_id, line_number)
);

CREATE INDEX IF NOT EXISTS idx_line_items_data_id ON documents_data_line_items(documents_data_id);
CREATE INDEX IF NOT EXISTS idx_line_items_line_number ON documents_data_line_items(documents_data_id, line_number);

-- ============================================================================
-- PART 5: SEMANTIC MAPPINGS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS semantic_mappings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Concept identification
    concept_name VARCHAR(128) NOT NULL,
    concept_aliases TEXT[] DEFAULT '{}',
    concept_description TEXT,

    -- Mapping to schema fields
    applicable_schema_types VARCHAR(64)[] DEFAULT '{}',
    json_path VARCHAR(256) NOT NULL,
    data_type VARCHAR(32) DEFAULT 'string',

    -- Aggregation configuration
    default_aggregation VARCHAR(32),
    is_calculated BOOLEAN DEFAULT FALSE,
    calculation_formula TEXT,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_semantic_mappings_concept ON semantic_mappings(concept_name);
CREATE INDEX IF NOT EXISTS idx_semantic_mappings_schemas ON semantic_mappings USING GIN (applicable_schema_types);

-- ============================================================================
-- PART 6: ENTITY REGISTRY TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS entity_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,

    -- Entity identification
    entity_type VARCHAR(64) NOT NULL,
    canonical_name VARCHAR(256) NOT NULL,
    aliases TEXT[] DEFAULT '{}',

    -- Statistics
    document_count INTEGER DEFAULT 0,
    last_seen_at TIMESTAMP WITH TIME ZONE,

    -- Additional metadata
    entity_metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT uq_entity_registry UNIQUE(workspace_id, entity_type, canonical_name)
);

CREATE INDEX IF NOT EXISTS idx_entity_registry_workspace ON entity_registry(workspace_id);
CREATE INDEX IF NOT EXISTS idx_entity_registry_type ON entity_registry(workspace_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_entity_registry_name ON entity_registry(canonical_name);
CREATE INDEX IF NOT EXISTS idx_entity_registry_aliases ON entity_registry USING GIN (aliases);

-- ============================================================================
-- PART 7: ANALYTICS SESSIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS analytics_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Links to existing infrastructure
    chat_session_id UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,

    -- State Machine
    state VARCHAR(32) NOT NULL DEFAULT 'INITIAL',
    state_entered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Query Context
    original_query TEXT NOT NULL,
    intent_classification JSONB,

    -- Gathered Information (from gap analysis)
    gathered_info JSONB DEFAULT '{}',

    -- Plan Management
    current_plan JSONB,
    plan_version INTEGER DEFAULT 0,
    plan_history JSONB DEFAULT '[]',

    -- Execution State
    execution_progress JSONB,

    -- Results Cache
    cached_results JSONB,
    result_generated_at TIMESTAMP WITH TIME ZONE,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() + INTERVAL '24 hours',

    -- State constraint
    CONSTRAINT valid_analytics_state CHECK (state IN (
        'INITIAL', 'CLASSIFYING', 'QUESTIONING', 'PLANNING',
        'REVIEWING', 'REFINING', 'EXECUTING', 'COMPLETE',
        'ERROR', 'FOLLOW_UP', 'EXPIRED'
    ))
);

CREATE INDEX IF NOT EXISTS idx_analytics_sessions_user ON analytics_sessions(user_id, state);
CREATE INDEX IF NOT EXISTS idx_analytics_sessions_chat ON analytics_sessions(chat_session_id);
CREATE INDEX IF NOT EXISTS idx_analytics_sessions_workspace ON analytics_sessions(workspace_id);
CREATE INDEX IF NOT EXISTS idx_analytics_sessions_expires ON analytics_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_analytics_sessions_active ON analytics_sessions(user_id)
    WHERE state NOT IN ('COMPLETE', 'EXPIRED', 'ERROR');

-- ============================================================================
-- PART 8: INSERT DEFAULT SCHEMAS
-- ============================================================================

-- Insert default data schemas
INSERT INTO data_schemas (schema_type, domain, display_name, description, header_schema, line_items_schema, summary_schema)
VALUES
-- Invoice Schema
('invoice', 'financial', 'Invoice', 'Standard invoice with line items',
 '{
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string", "description": "Unique invoice identifier"},
        "invoice_date": {"type": "string", "format": "date"},
        "due_date": {"type": "string", "format": "date"},
        "vendor_name": {"type": "string"},
        "vendor_address": {"type": "string"},
        "vendor_tax_id": {"type": "string"},
        "customer_name": {"type": "string"},
        "customer_address": {"type": "string"},
        "customer_tax_id": {"type": "string"},
        "payment_terms": {"type": "string"},
        "currency": {"type": "string", "default": "USD"}
    },
    "required": ["invoice_number"]
 }'::jsonb,
 '{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "item_number": {"type": "integer"},
            "description": {"type": "string"},
            "quantity": {"type": "number"},
            "unit": {"type": "string"},
            "unit_price": {"type": "number"},
            "discount": {"type": "number"},
            "tax_rate": {"type": "number"},
            "amount": {"type": "number"}
        }
    }
 }'::jsonb,
 '{
    "type": "object",
    "properties": {
        "subtotal": {"type": "number"},
        "discount_total": {"type": "number"},
        "tax_amount": {"type": "number"},
        "shipping": {"type": "number"},
        "total_amount": {"type": "number"}
    }
 }'::jsonb),

-- Receipt Schema
('receipt', 'financial', 'Receipt', 'Point-of-sale receipt',
 '{
    "type": "object",
    "properties": {
        "receipt_number": {"type": "string"},
        "transaction_date": {"type": "string", "format": "date"},
        "transaction_time": {"type": "string", "format": "time"},
        "store_name": {"type": "string"},
        "store_address": {"type": "string"},
        "store_phone": {"type": "string"},
        "cashier": {"type": "string"},
        "payment_method": {"type": "string"},
        "card_last_four": {"type": "string"},
        "currency": {"type": "string", "default": "USD"}
    }
 }'::jsonb,
 '{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "quantity": {"type": "number"},
            "unit_price": {"type": "number"},
            "amount": {"type": "number"}
        }
    }
 }'::jsonb,
 '{
    "type": "object",
    "properties": {
        "subtotal": {"type": "number"},
        "tax_amount": {"type": "number"},
        "tip": {"type": "number"},
        "total_amount": {"type": "number"},
        "amount_paid": {"type": "number"},
        "change": {"type": "number"}
    }
 }'::jsonb),

-- Bank Statement Schema
('bank_statement', 'financial', 'Bank Statement', 'Bank account statement with transactions',
 '{
    "type": "object",
    "properties": {
        "account_number": {"type": "string"},
        "account_holder": {"type": "string"},
        "account_type": {"type": "string"},
        "bank_name": {"type": "string"},
        "bank_address": {"type": "string"},
        "statement_period_start": {"type": "string", "format": "date"},
        "statement_period_end": {"type": "string", "format": "date"},
        "currency": {"type": "string", "default": "USD"}
    },
    "required": ["account_number", "statement_period_start", "statement_period_end"]
 }'::jsonb,
 '{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "format": "date"},
            "description": {"type": "string"},
            "reference": {"type": "string"},
            "type": {"type": "string"},
            "debit": {"type": "number"},
            "credit": {"type": "number"},
            "balance": {"type": "number"}
        }
    }
 }'::jsonb,
 '{
    "type": "object",
    "properties": {
        "opening_balance": {"type": "number"},
        "total_deposits": {"type": "number"},
        "total_withdrawals": {"type": "number"},
        "closing_balance": {"type": "number"},
        "total_fees": {"type": "number"},
        "interest_earned": {"type": "number"}
    }
 }'::jsonb),

-- Purchase Order Schema
('purchase_order', 'financial', 'Purchase Order', 'Purchase order document',
 '{
    "type": "object",
    "properties": {
        "po_number": {"type": "string"},
        "order_date": {"type": "string", "format": "date"},
        "delivery_date": {"type": "string", "format": "date"},
        "vendor_name": {"type": "string"},
        "vendor_address": {"type": "string"},
        "ship_to_name": {"type": "string"},
        "ship_to_address": {"type": "string"},
        "shipping_method": {"type": "string"},
        "payment_terms": {"type": "string"},
        "currency": {"type": "string", "default": "USD"}
    },
    "required": ["po_number"]
 }'::jsonb,
 '{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "line_number": {"type": "integer"},
            "item_code": {"type": "string"},
            "description": {"type": "string"},
            "quantity": {"type": "number"},
            "unit": {"type": "string"},
            "unit_price": {"type": "number"},
            "amount": {"type": "number"}
        }
    }
 }'::jsonb,
 '{
    "type": "object",
    "properties": {
        "subtotal": {"type": "number"},
        "tax_amount": {"type": "number"},
        "shipping_cost": {"type": "number"},
        "total_amount": {"type": "number"}
    }
 }'::jsonb),

-- Expense Report Schema
('expense_report', 'financial', 'Expense Report', 'Employee expense report',
 '{
    "type": "object",
    "properties": {
        "report_number": {"type": "string"},
        "employee_name": {"type": "string"},
        "employee_id": {"type": "string"},
        "department": {"type": "string"},
        "report_period_start": {"type": "string", "format": "date"},
        "report_period_end": {"type": "string", "format": "date"},
        "submission_date": {"type": "string", "format": "date"},
        "purpose": {"type": "string"},
        "currency": {"type": "string", "default": "USD"}
    }
 }'::jsonb,
 '{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "format": "date"},
            "category": {"type": "string"},
            "description": {"type": "string"},
            "vendor": {"type": "string"},
            "amount": {"type": "number"},
            "receipt_attached": {"type": "boolean"}
        }
    }
 }'::jsonb,
 '{
    "type": "object",
    "properties": {
        "total_expenses": {"type": "number"},
        "advance_received": {"type": "number"},
        "amount_due": {"type": "number"}
    }
 }'::jsonb),

-- Shipping Manifest Schema
('shipping_manifest', 'logistics', 'Shipping Manifest', 'Shipping/delivery manifest',
 '{
    "type": "object",
    "properties": {
        "manifest_number": {"type": "string"},
        "shipment_date": {"type": "string", "format": "date"},
        "carrier": {"type": "string"},
        "origin": {"type": "string"},
        "destination": {"type": "string"},
        "shipper_name": {"type": "string"},
        "consignee_name": {"type": "string"},
        "total_packages": {"type": "integer"},
        "total_weight": {"type": "number"},
        "weight_unit": {"type": "string"}
    }
 }'::jsonb,
 '{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "package_number": {"type": "string"},
            "description": {"type": "string"},
            "quantity": {"type": "integer"},
            "weight": {"type": "number"},
            "dimensions": {"type": "string"},
            "tracking_number": {"type": "string"}
        }
    }
 }'::jsonb,
 '{
    "type": "object",
    "properties": {
        "total_packages": {"type": "integer"},
        "total_weight": {"type": "number"},
        "declared_value": {"type": "number"}
    }
 }'::jsonb),

-- Inventory Report Schema
('inventory_report', 'inventory', 'Inventory Report', 'Inventory/stock report',
 '{
    "type": "object",
    "properties": {
        "report_number": {"type": "string"},
        "report_date": {"type": "string", "format": "date"},
        "warehouse": {"type": "string"},
        "location": {"type": "string"},
        "report_type": {"type": "string"},
        "prepared_by": {"type": "string"}
    }
 }'::jsonb,
 '{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "sku": {"type": "string"},
            "item_name": {"type": "string"},
            "description": {"type": "string"},
            "category": {"type": "string"},
            "quantity_on_hand": {"type": "number"},
            "unit": {"type": "string"},
            "unit_cost": {"type": "number"},
            "total_value": {"type": "number"},
            "reorder_point": {"type": "number"},
            "location": {"type": "string"}
        }
    }
 }'::jsonb,
 '{
    "type": "object",
    "properties": {
        "total_items": {"type": "integer"},
        "total_quantity": {"type": "number"},
        "total_value": {"type": "number"}
    }
 }'::jsonb),

-- Spreadsheet Schema (generic)
('spreadsheet', 'generic', 'Spreadsheet', 'Generic spreadsheet data',
 '{
    "type": "object",
    "properties": {
        "sheet_name": {"type": "string"},
        "source_file": {"type": "string"},
        "column_headers": {"type": "array", "items": {"type": "string"}},
        "total_rows": {"type": "integer"},
        "total_columns": {"type": "integer"}
    }
 }'::jsonb,
 '{
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": true
    }
 }'::jsonb,
 '{
    "type": "object",
    "properties": {
        "row_count": {"type": "integer"},
        "column_count": {"type": "integer"}
    }
 }'::jsonb)

ON CONFLICT (schema_type) DO NOTHING;

-- ============================================================================
-- PART 9: INSERT DEFAULT SEMANTIC MAPPINGS
-- ============================================================================

INSERT INTO semantic_mappings (concept_name, concept_aliases, applicable_schema_types, json_path, data_type, default_aggregation)
VALUES
-- Amount-related concepts
('total_amount', ARRAY['total', 'grand total', 'amount due', 'sum', 'final amount'],
 ARRAY['invoice', 'receipt', 'purchase_order', 'expense_report'],
 'summary_data.total_amount', 'number', 'sum'),

('subtotal', ARRAY['sub total', 'net amount', 'before tax'],
 ARRAY['invoice', 'receipt', 'purchase_order'],
 'summary_data.subtotal', 'number', 'sum'),

('tax_amount', ARRAY['tax', 'vat', 'gst', 'sales tax'],
 ARRAY['invoice', 'receipt'],
 'summary_data.tax_amount', 'number', 'sum'),

-- Date-related concepts
('transaction_date', ARRAY['date', 'invoice date', 'receipt date', 'order date'],
 ARRAY['invoice', 'receipt', 'purchase_order', 'expense_report'],
 'header_data.invoice_date', 'date', NULL),

-- Entity-related concepts
('vendor_name', ARRAY['vendor', 'supplier', 'seller', 'merchant', 'store'],
 ARRAY['invoice', 'receipt', 'purchase_order'],
 'header_data.vendor_name', 'string', NULL),

('customer_name', ARRAY['customer', 'buyer', 'client', 'purchaser'],
 ARRAY['invoice', 'purchase_order'],
 'header_data.customer_name', 'string', NULL),

-- Balance-related concepts
('opening_balance', ARRAY['beginning balance', 'start balance', 'prior balance'],
 ARRAY['bank_statement'],
 'summary_data.opening_balance', 'number', NULL),

('closing_balance', ARRAY['ending balance', 'final balance', 'current balance'],
 ARRAY['bank_statement'],
 'summary_data.closing_balance', 'number', NULL),

-- Line item concepts
('item_quantity', ARRAY['quantity', 'qty', 'units', 'count'],
 ARRAY['invoice', 'receipt', 'purchase_order', 'inventory_report'],
 'line_items[*].quantity', 'number', 'sum'),

('unit_price', ARRAY['price', 'rate', 'cost per unit'],
 ARRAY['invoice', 'receipt', 'purchase_order'],
 'line_items[*].unit_price', 'number', 'avg')

ON CONFLICT DO NOTHING;

-- ============================================================================
-- PART 10: GRANT PERMISSIONS (if needed)
-- ============================================================================

-- Grant permissions on new tables to application role (adjust role name as needed)
-- GRANT ALL ON data_schemas TO your_app_role;
-- GRANT ALL ON documents_data TO your_app_role;
-- GRANT ALL ON documents_data_line_items TO your_app_role;
-- GRANT ALL ON semantic_mappings TO your_app_role;
-- GRANT ALL ON entity_registry TO your_app_role;
-- GRANT ALL ON analytics_sessions TO your_app_role;

-- ============================================================================
-- Migration complete
-- ============================================================================
