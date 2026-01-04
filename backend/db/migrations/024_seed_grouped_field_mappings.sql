-- Migration: 024_seed_grouped_field_mappings.sql
-- Description: Seed data_schemas table with grouped field_mappings
--              New format separates header_mappings, line_item_mappings, summary_mappings
--              Includes bilingual/label detection patterns
-- Date: 2025-01-03

-- ============================================================================
-- PART 1: UPDATE INVOICE SCHEMA WITH GROUPED FIELD MAPPINGS
-- ============================================================================

INSERT INTO data_schemas (
    schema_type,
    schema_version,
    domain,
    display_name,
    description,
    header_schema,
    line_items_schema,
    summary_schema,
    field_mappings,
    is_active
) VALUES (
    'invoice',
    '2.0',
    'financial',
    'Invoice',
    'Standard invoice schema with grouped field mappings for header, line items, and summary',
    '{
        "type": "object",
        "properties": {
            "vendor_name": {"type": "string"},
            "customer_name": {"type": "string"},
            "invoice_number": {"type": "string"},
            "invoice_date": {"type": "string", "format": "date"},
            "due_date": {"type": "string", "format": "date"},
            "currency": {"type": "string"},
            "payment_terms": {"type": "string"},
            "po_number": {"type": "string"}
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
                "amount": {"type": "number"},
                "sku": {"type": "string"}
            }
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "subtotal": {"type": "number"},
            "tax_amount": {"type": "number"},
            "discount_amount": {"type": "number"},
            "total_amount": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_mappings": [
            {"canonical": "vendor_name", "data_type": "string", "patterns": ["vendor", "supplier", "seller", "from", "fournisseur", "lieferant"], "exclude_patterns": ["/"]},
            {"canonical": "customer_name", "data_type": "string", "patterns": ["customer", "client", "buyer", "bill to", "sold to"], "exclude_patterns": ["/"]},
            {"canonical": "vendor_address", "data_type": "string", "patterns": ["vendor address", "seller address", "from address"], "exclude_patterns": ["/"]},
            {"canonical": "customer_address", "data_type": "string", "patterns": ["customer address", "bill to address", "billing address"], "exclude_patterns": ["/"]},
            {"canonical": "invoice_number", "data_type": "string", "patterns": ["invoice #", "invoice no", "invoice number", "inv #", "facture no"]},
            {"canonical": "invoice_date", "data_type": "datetime", "patterns": ["invoice date", "date", "dated", "issue date"]},
            {"canonical": "due_date", "data_type": "datetime", "patterns": ["due date", "payment due", "due by"]},
            {"canonical": "currency", "data_type": "string", "patterns": ["currency", "devise"]},
            {"canonical": "payment_terms", "data_type": "string", "patterns": ["terms", "payment terms", "net 30", "net 60"]},
            {"canonical": "po_number", "data_type": "string", "patterns": ["po #", "po number", "purchase order", "order #"]}
        ],
        "line_item_mappings": [
            {"canonical": "description", "data_type": "string", "patterns": ["description", "item", "product", "service", "article", "name"], "exclude_patterns": ["/"]},
            {"canonical": "quantity", "data_type": "number", "patterns": ["qty", "quantity", "units", "count"], "exclude_patterns": ["/"]},
            {"canonical": "unit_price", "data_type": "number", "patterns": ["unit price", "price", "rate", "each"], "exclude_patterns": ["/"]},
            {"canonical": "amount", "data_type": "number", "patterns": ["amount", "total", "extended", "line total"], "exclude_patterns": ["/", "subtotal", "tax", "grand total", "total amount"]},
            {"canonical": "sku", "data_type": "string", "patterns": ["sku", "item code", "product code", "part #", "code"]},
            {"canonical": "category", "data_type": "string", "patterns": ["category", "type"]}
        ],
        "summary_mappings": [
            {"canonical": "subtotal", "data_type": "number", "patterns": ["subtotal", "sub-total", "sous-total", "net amount"], "exclude_patterns": ["/"]},
            {"canonical": "tax_amount", "data_type": "number", "patterns": ["tax", "vat", "gst", "hst", "pst", "sales tax", "tva"], "exclude_patterns": ["/"]},
            {"canonical": "discount_amount", "data_type": "number", "patterns": ["discount", "reduction", "rabais"], "exclude_patterns": ["/"]},
            {"canonical": "shipping_amount", "data_type": "number", "patterns": ["shipping", "freight", "delivery"], "exclude_patterns": ["/"]},
            {"canonical": "total_amount", "data_type": "number", "patterns": ["total", "grand total", "amount due", "balance due"], "exclude_patterns": ["/"]}
        ]
    }'::jsonb,
    TRUE
)
ON CONFLICT (schema_type) DO UPDATE SET
    schema_version = EXCLUDED.schema_version,
    field_mappings = EXCLUDED.field_mappings,
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    updated_at = NOW();

-- ============================================================================
-- PART 2: UPDATE RECEIPT SCHEMA WITH GROUPED FIELD MAPPINGS
-- ============================================================================

INSERT INTO data_schemas (
    schema_type,
    schema_version,
    domain,
    display_name,
    description,
    header_schema,
    line_items_schema,
    summary_schema,
    field_mappings,
    is_active
) VALUES (
    'receipt',
    '2.0',
    'financial',
    'Receipt',
    'Standard receipt schema with grouped field mappings for header, line items, and summary',
    '{
        "type": "object",
        "properties": {
            "store_name": {"type": "string"},
            "store_address": {"type": "string"},
            "transaction_date": {"type": "string", "format": "date"},
            "receipt_number": {"type": "string"},
            "payment_method": {"type": "string"},
            "cashier": {"type": "string"}
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
            "tip_amount": {"type": "number"},
            "total_amount": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_mappings": [
            {"canonical": "store_name", "data_type": "string", "patterns": ["store", "merchant", "restaurant", "shop", "retailer"], "exclude_patterns": ["/"]},
            {"canonical": "store_address", "data_type": "string", "patterns": ["address", "location", "store address"], "exclude_patterns": ["/"]},
            {"canonical": "transaction_date", "data_type": "datetime", "patterns": ["date", "transaction date", "purchase date", "time"]},
            {"canonical": "receipt_number", "data_type": "string", "patterns": ["receipt #", "transaction #", "order #", "ticket #", "check #"]},
            {"canonical": "payment_method", "data_type": "string", "patterns": ["payment", "paid by", "card", "cash", "credit", "debit"]},
            {"canonical": "cashier", "data_type": "string", "patterns": ["cashier", "server", "employee", "served by"]}
        ],
        "line_item_mappings": [
            {"canonical": "description", "data_type": "string", "patterns": ["description", "item", "product", "article", "name"], "exclude_patterns": ["/"]},
            {"canonical": "quantity", "data_type": "number", "patterns": ["qty", "quantity", "x", "units"], "exclude_patterns": ["/"]},
            {"canonical": "unit_price", "data_type": "number", "patterns": ["unit price", "each", "@", "price"], "exclude_patterns": ["/"]},
            {"canonical": "amount", "data_type": "number", "patterns": ["amount", "price", "total", "cost"], "exclude_patterns": ["/", "subtotal", "tax", "grand total", "tip"]}
        ],
        "summary_mappings": [
            {"canonical": "subtotal", "data_type": "number", "patterns": ["subtotal", "sub-total", "food total", "items total"], "exclude_patterns": ["/"]},
            {"canonical": "tax_amount", "data_type": "number", "patterns": ["tax", "hst", "gst", "pst", "sales tax", "vat"], "exclude_patterns": ["/"]},
            {"canonical": "tip_amount", "data_type": "number", "patterns": ["tip", "gratuity", "service charge"], "exclude_patterns": ["/"]},
            {"canonical": "total_amount", "data_type": "number", "patterns": ["total", "grand total", "amount", "balance"], "exclude_patterns": ["/"]}
        ]
    }'::jsonb,
    TRUE
)
ON CONFLICT (schema_type) DO UPDATE SET
    schema_version = EXCLUDED.schema_version,
    field_mappings = EXCLUDED.field_mappings,
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    updated_at = NOW();

-- ============================================================================
-- PART 3: UPDATE SPREADSHEET SCHEMA WITH GROUPED FIELD MAPPINGS
-- ============================================================================

INSERT INTO data_schemas (
    schema_type,
    schema_version,
    domain,
    display_name,
    description,
    header_schema,
    line_items_schema,
    summary_schema,
    field_mappings,
    is_active
) VALUES (
    'spreadsheet',
    '2.0',
    'general',
    'Spreadsheet',
    'Generic spreadsheet schema with grouped field mappings for dynamic columns',
    '{
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "date": {"type": "string", "format": "date"}
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
            "total_amount": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_mappings": [
            {"canonical": "title", "data_type": "string", "patterns": ["title", "name", "sheet name"]},
            {"canonical": "date", "data_type": "datetime", "patterns": ["date", "created", "modified"]}
        ],
        "line_item_mappings": [
            {"canonical": "description", "data_type": "string", "patterns": ["description", "name", "item", "product", "title"], "exclude_patterns": ["/"]},
            {"canonical": "quantity", "data_type": "number", "patterns": ["qty", "quantity", "count", "units", "stock"], "exclude_patterns": ["/"]},
            {"canonical": "amount", "data_type": "number", "patterns": ["amount", "total", "value", "price", "cost", "sales"], "exclude_patterns": ["/", "subtotal"]},
            {"canonical": "date", "data_type": "datetime", "patterns": ["date", "time", "created", "updated"]},
            {"canonical": "category", "data_type": "string", "patterns": ["category", "type", "group", "class"]},
            {"canonical": "status", "data_type": "string", "patterns": ["status", "state", "condition"]}
        ],
        "summary_mappings": [
            {"canonical": "total_amount", "data_type": "number", "patterns": ["total", "sum", "grand total"], "exclude_patterns": ["/"]}
        ]
    }'::jsonb,
    TRUE
)
ON CONFLICT (schema_type) DO UPDATE SET
    schema_version = EXCLUDED.schema_version,
    field_mappings = EXCLUDED.field_mappings,
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    updated_at = NOW();

-- ============================================================================
-- PART 4: UPDATE BANK STATEMENT SCHEMA WITH GROUPED FIELD MAPPINGS
-- ============================================================================

INSERT INTO data_schemas (
    schema_type,
    schema_version,
    domain,
    display_name,
    description,
    header_schema,
    line_items_schema,
    summary_schema,
    field_mappings,
    is_active
) VALUES (
    'bank_statement',
    '2.0',
    'financial',
    'Bank Statement',
    'Bank statement schema with grouped field mappings for transactions',
    '{
        "type": "object",
        "properties": {
            "bank_name": {"type": "string"},
            "account_number": {"type": "string"},
            "statement_date": {"type": "string", "format": "date"},
            "period_start": {"type": "string", "format": "date"},
            "period_end": {"type": "string", "format": "date"}
        }
    }'::jsonb,
    '{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "transaction_date": {"type": "string", "format": "date"},
                "description": {"type": "string"},
                "debit": {"type": "number"},
                "credit": {"type": "number"},
                "balance": {"type": "number"},
                "reference": {"type": "string"}
            }
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "opening_balance": {"type": "number"},
            "total_debits": {"type": "number"},
            "total_credits": {"type": "number"},
            "closing_balance": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_mappings": [
            {"canonical": "bank_name", "data_type": "string", "patterns": ["bank", "institution", "financial institution"]},
            {"canonical": "account_number", "data_type": "string", "patterns": ["account", "account #", "account number"]},
            {"canonical": "statement_date", "data_type": "datetime", "patterns": ["statement date", "date", "period end"]},
            {"canonical": "period_start", "data_type": "datetime", "patterns": ["period start", "from", "start date"]},
            {"canonical": "period_end", "data_type": "datetime", "patterns": ["period end", "to", "end date"]}
        ],
        "line_item_mappings": [
            {"canonical": "transaction_date", "data_type": "datetime", "patterns": ["date", "transaction date", "post date"]},
            {"canonical": "description", "data_type": "string", "patterns": ["description", "transaction", "details", "memo"], "exclude_patterns": ["/"]},
            {"canonical": "debit", "data_type": "number", "patterns": ["debit", "withdrawal", "payment", "out"], "exclude_patterns": ["/"]},
            {"canonical": "credit", "data_type": "number", "patterns": ["credit", "deposit", "in"], "exclude_patterns": ["/"]},
            {"canonical": "balance", "data_type": "number", "patterns": ["balance", "running balance"], "exclude_patterns": ["/"]},
            {"canonical": "reference", "data_type": "string", "patterns": ["reference", "ref #", "check #", "confirmation"]}
        ],
        "summary_mappings": [
            {"canonical": "opening_balance", "data_type": "number", "patterns": ["opening balance", "beginning balance", "previous balance"], "exclude_patterns": ["/"]},
            {"canonical": "total_debits", "data_type": "number", "patterns": ["total debits", "total withdrawals", "total payments"], "exclude_patterns": ["/"]},
            {"canonical": "total_credits", "data_type": "number", "patterns": ["total credits", "total deposits"], "exclude_patterns": ["/"]},
            {"canonical": "closing_balance", "data_type": "number", "patterns": ["closing balance", "ending balance", "current balance"], "exclude_patterns": ["/"]}
        ]
    }'::jsonb,
    TRUE
)
ON CONFLICT (schema_type) DO UPDATE SET
    schema_version = EXCLUDED.schema_version,
    field_mappings = EXCLUDED.field_mappings,
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    updated_at = NOW();

-- ============================================================================
-- PART 5: ADD PURCHASE ORDER SCHEMA WITH GROUPED FIELD MAPPINGS
-- ============================================================================

INSERT INTO data_schemas (
    schema_type,
    schema_version,
    domain,
    display_name,
    description,
    header_schema,
    line_items_schema,
    summary_schema,
    field_mappings,
    is_active
) VALUES (
    'purchase_order',
    '2.0',
    'financial',
    'Purchase Order',
    'Purchase order schema with grouped field mappings',
    '{
        "type": "object",
        "properties": {
            "vendor_name": {"type": "string"},
            "customer_name": {"type": "string"},
            "po_number": {"type": "string"},
            "order_date": {"type": "string", "format": "date"},
            "delivery_date": {"type": "string", "format": "date"},
            "shipping_address": {"type": "string"}
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
                "amount": {"type": "number"},
                "sku": {"type": "string"}
            }
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "subtotal": {"type": "number"},
            "tax_amount": {"type": "number"},
            "shipping_amount": {"type": "number"},
            "total_amount": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_mappings": [
            {"canonical": "vendor_name", "data_type": "string", "patterns": ["vendor", "supplier", "seller"], "exclude_patterns": ["/"]},
            {"canonical": "customer_name", "data_type": "string", "patterns": ["customer", "buyer", "ship to", "bill to"], "exclude_patterns": ["/"]},
            {"canonical": "po_number", "data_type": "string", "patterns": ["po #", "po number", "purchase order", "order #"]},
            {"canonical": "order_date", "data_type": "datetime", "patterns": ["order date", "date", "created"]},
            {"canonical": "delivery_date", "data_type": "datetime", "patterns": ["delivery date", "ship date", "expected"]},
            {"canonical": "shipping_address", "data_type": "string", "patterns": ["ship to", "shipping address", "delivery address"]}
        ],
        "line_item_mappings": [
            {"canonical": "description", "data_type": "string", "patterns": ["description", "item", "product", "part"], "exclude_patterns": ["/"]},
            {"canonical": "quantity", "data_type": "number", "patterns": ["qty", "quantity", "units", "ordered"], "exclude_patterns": ["/"]},
            {"canonical": "unit_price", "data_type": "number", "patterns": ["unit price", "price", "cost", "each"], "exclude_patterns": ["/"]},
            {"canonical": "amount", "data_type": "number", "patterns": ["amount", "total", "extended", "line total"], "exclude_patterns": ["/", "subtotal", "tax", "grand total"]},
            {"canonical": "sku", "data_type": "string", "patterns": ["sku", "part #", "item code", "product code"]}
        ],
        "summary_mappings": [
            {"canonical": "subtotal", "data_type": "number", "patterns": ["subtotal", "sub-total", "merchandise total"], "exclude_patterns": ["/"]},
            {"canonical": "tax_amount", "data_type": "number", "patterns": ["tax", "vat", "gst", "hst"], "exclude_patterns": ["/"]},
            {"canonical": "shipping_amount", "data_type": "number", "patterns": ["shipping", "freight", "delivery"], "exclude_patterns": ["/"]},
            {"canonical": "total_amount", "data_type": "number", "patterns": ["total", "grand total", "order total"], "exclude_patterns": ["/"]}
        ]
    }'::jsonb,
    TRUE
)
ON CONFLICT (schema_type) DO UPDATE SET
    schema_version = EXCLUDED.schema_version,
    field_mappings = EXCLUDED.field_mappings,
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    updated_at = NOW();

-- ============================================================================
-- Log migration completion
-- ============================================================================
DO $$
BEGIN
    RAISE NOTICE 'Migration 024: Seeded grouped field_mappings for 5 schema types (invoice, receipt, spreadsheet, bank_statement, purchase_order)';
END $$;
