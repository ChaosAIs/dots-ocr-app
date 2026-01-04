-- Migration: 025_fix_date_time_field_mappings.sql
-- Description: Fix date/time field mapping conflicts in data_schemas
--              Adds exclude_patterns to prevent field overwriting issues:
--              - transaction_date should exclude "time" patterns
--              - transaction_time should exclude "date" patterns
--              - statement_date should exclude "period", "start", "end" patterns
--              - period_start should exclude "end" patterns
--              - period_end should exclude "start" patterns
--              - spreadsheet date/time fields separated properly
-- Date: 2025-01-04
-- Issue: transaction_date was being overwritten by transaction_time values
--        because "time" was included in transaction_date patterns

-- ============================================================================
-- PART 1: FIX RECEIPT SCHEMA - Add transaction_time, fix transaction_date
-- ============================================================================

UPDATE data_schemas
SET
    schema_version = '2.1',
    field_mappings = jsonb_set(
        field_mappings,
        '{header_mappings}',
        '[
            {"canonical": "store_name", "data_type": "string", "patterns": ["store", "merchant", "restaurant", "shop", "retailer"], "exclude_patterns": ["/"]},
            {"canonical": "store_address", "data_type": "string", "patterns": ["address", "location", "store address"], "exclude_patterns": ["/"]},
            {"canonical": "transaction_date", "data_type": "datetime", "patterns": ["date", "transaction date", "purchase date", "transaction_date"], "exclude_patterns": ["time"]},
            {"canonical": "transaction_time", "data_type": "string", "patterns": ["time", "transaction time", "transaction_time"], "exclude_patterns": ["date"]},
            {"canonical": "receipt_number", "data_type": "string", "patterns": ["receipt #", "transaction #", "order #", "ticket #", "check #"]},
            {"canonical": "payment_method", "data_type": "string", "patterns": ["payment", "paid by", "card", "cash", "credit", "debit"]},
            {"canonical": "cashier", "data_type": "string", "patterns": ["cashier", "server", "employee", "served by"]}
        ]'::jsonb
    ),
    updated_at = NOW()
WHERE schema_type = 'receipt' AND is_active = true;

-- ============================================================================
-- PART 2: FIX SPREADSHEET SCHEMA - Separate date and time fields
-- ============================================================================

UPDATE data_schemas
SET
    schema_version = '2.1',
    field_mappings = jsonb_set(
        jsonb_set(
            field_mappings,
            '{header_mappings}',
            '[
                {"canonical": "title", "data_type": "string", "patterns": ["title", "name", "sheet name"]},
                {"canonical": "date", "data_type": "datetime", "patterns": ["date", "created", "modified"], "exclude_patterns": ["time"]}
            ]'::jsonb
        ),
        '{line_item_mappings}',
        '[
            {"canonical": "description", "data_type": "string", "patterns": ["description", "name", "item", "product", "title"], "exclude_patterns": ["/"]},
            {"canonical": "quantity", "data_type": "number", "patterns": ["qty", "quantity", "count", "units", "stock"], "exclude_patterns": ["/"]},
            {"canonical": "amount", "data_type": "number", "patterns": ["amount", "total", "value", "price", "cost", "sales"], "exclude_patterns": ["/", "subtotal"]},
            {"canonical": "date", "data_type": "datetime", "patterns": ["date", "created", "updated"], "exclude_patterns": ["time"]},
            {"canonical": "time", "data_type": "string", "patterns": ["time", "timestamp"], "exclude_patterns": ["date"]},
            {"canonical": "category", "data_type": "string", "patterns": ["category", "type", "group", "class"]},
            {"canonical": "status", "data_type": "string", "patterns": ["status", "state", "condition"]}
        ]'::jsonb
    ),
    updated_at = NOW()
WHERE schema_type = 'spreadsheet' AND is_active = true;

-- ============================================================================
-- PART 3: FIX BANK STATEMENT SCHEMA - More specific date patterns
-- ============================================================================

UPDATE data_schemas
SET
    schema_version = '2.1',
    field_mappings = jsonb_set(
        jsonb_set(
            field_mappings,
            '{header_mappings}',
            '[
                {"canonical": "bank_name", "data_type": "string", "patterns": ["bank", "institution", "financial institution"]},
                {"canonical": "account_number", "data_type": "string", "patterns": ["account", "account #", "account number"]},
                {"canonical": "statement_date", "data_type": "datetime", "patterns": ["statement date", "statement_date"], "exclude_patterns": ["period", "start", "end"]},
                {"canonical": "period_start", "data_type": "datetime", "patterns": ["period start", "start date", "from date"], "exclude_patterns": ["end"]},
                {"canonical": "period_end", "data_type": "datetime", "patterns": ["period end", "end date", "to date", "through"], "exclude_patterns": ["start"]}
            ]'::jsonb
        ),
        '{line_item_mappings}',
        '[
            {"canonical": "transaction_date", "data_type": "datetime", "patterns": ["date", "transaction date", "post date", "trans date"]},
            {"canonical": "description", "data_type": "string", "patterns": ["description", "transaction", "details", "memo"], "exclude_patterns": ["/"]},
            {"canonical": "debit", "data_type": "number", "patterns": ["debit", "withdrawal", "payment", "out"], "exclude_patterns": ["/"]},
            {"canonical": "credit", "data_type": "number", "patterns": ["credit", "deposit", "in"], "exclude_patterns": ["/"]},
            {"canonical": "balance", "data_type": "number", "patterns": ["balance", "running balance"], "exclude_patterns": ["/"]},
            {"canonical": "reference", "data_type": "string", "patterns": ["reference", "ref #", "check #", "confirmation"]}
        ]'::jsonb
    ),
    updated_at = NOW()
WHERE schema_type = 'bank_statement' AND is_active = true;

-- ============================================================================
-- PART 4: Clear extraction_prompt from receipt schema to use updated config
-- ============================================================================
-- The extraction_prompt was storing old instructions that didn't properly
-- separate date and time extraction. Setting to NULL allows the system to
-- use the updated extraction_config.py prompts instead.

UPDATE data_schemas
SET extraction_prompt = NULL
WHERE schema_type = 'receipt' AND is_active = true;

-- ============================================================================
-- Log migration completion
-- ============================================================================
DO $$
BEGIN
    RAISE NOTICE 'Migration 025: Fixed date/time field mapping conflicts';
    RAISE NOTICE '  - receipt: Added transaction_time, fixed transaction_date exclude_patterns';
    RAISE NOTICE '  - spreadsheet: Separated date and time fields with exclude_patterns';
    RAISE NOTICE '  - bank_statement: More specific date patterns with exclude_patterns';
    RAISE NOTICE '  - receipt: Cleared extraction_prompt to use updated config';
END $$;
