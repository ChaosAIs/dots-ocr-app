-- =============================================================================
-- DATA SCHEMAS SEED SCRIPT
-- Populates the data_schemas table with all 12 extractable document types
--
-- Run this script to initialize the schema registry with formal schemas
-- for all supported document types.
--
-- Usage:
--   psql -d your_database -f seed_data_schemas.sql
-- =============================================================================

-- First, ensure the table exists and has the required columns
-- (This is a safety check - the actual table should be created by the ORM)

-- 1. INVOICE SCHEMA
INSERT INTO data_schemas (
    id,
    schema_type,
    domain,
    display_name,
    description,
    header_schema,
    line_items_schema,
    summary_schema,
    field_mappings,
    extraction_prompt,
    validation_rules,
    schema_version,
    is_active,
    created_at,
    updated_at
) VALUES (
    gen_random_uuid(),
    'invoice',
    'financial',
    'Invoice',
    'Schema for invoice documents including vendor invoices, billing documents, and purchase invoices',
    '{
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string", "description": "Unique invoice identifier"},
            "invoice_date": {"type": "string", "format": "date", "description": "Date of invoice in YYYY-MM-DD format"},
            "due_date": {"type": "string", "format": "date", "description": "Payment due date"},
            "vendor_name": {"type": "string", "description": "Name of the vendor/supplier"},
            "vendor_address": {"type": "string", "description": "Full address of the vendor"},
            "customer_name": {"type": "string", "description": "Name of the customer/buyer"},
            "customer_address": {"type": "string", "description": "Full address of the customer"},
            "payment_terms": {"type": "string", "description": "Payment terms (e.g., Net 30)"},
            "currency": {"type": "string", "description": "Currency code (e.g., CAD, USD, EUR)"},
            "po_number": {"type": "string", "description": "Purchase order reference number"}
        },
        "required": ["invoice_number"]
    }'::jsonb,
    '{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Item description"},
                "quantity": {"type": "number", "description": "Quantity ordered"},
                "unit_price": {"type": "number", "description": "Price per unit"},
                "amount": {"type": "number", "description": "Line total (quantity × unit_price)"},
                "sku": {"type": "string", "description": "Product SKU or code"},
                "tax_rate": {"type": "number", "description": "Tax rate percentage"}
            },
            "required": ["description", "amount"]
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "subtotal": {"type": "number", "description": "Sum before taxes"},
            "tax_amount": {"type": "number", "description": "Total tax amount"},
            "discount_amount": {"type": "number", "description": "Total discounts applied"},
            "shipping_amount": {"type": "number", "description": "Shipping/freight charges"},
            "total_amount": {"type": "number", "description": "Final total amount due"}
        },
        "required": ["total_amount"]
    }'::jsonb,
    '{
        "header_fields": {
            "invoice_number": {"semantic_type": "identifier", "data_type": "string", "source": "header", "aggregation": null},
            "invoice_date": {"semantic_type": "date", "data_type": "datetime", "source": "header", "aggregation": null},
            "due_date": {"semantic_type": "date", "data_type": "datetime", "source": "header", "aggregation": null},
            "vendor_name": {"semantic_type": "entity", "data_type": "string", "source": "header", "aggregation": "group_by"},
            "customer_name": {"semantic_type": "entity", "data_type": "string", "source": "header", "aggregation": "group_by"},
            "currency": {"semantic_type": "currency", "data_type": "string", "source": "header", "aggregation": null}
        },
        "line_item_fields": {
            "description": {"semantic_type": "product", "data_type": "string", "source": "line_item", "aggregation": "group_by"},
            "quantity": {"semantic_type": "quantity", "data_type": "number", "source": "line_item", "aggregation": "sum"},
            "unit_price": {"semantic_type": "amount", "data_type": "number", "source": "line_item", "aggregation": null},
            "amount": {"semantic_type": "amount", "data_type": "number", "source": "line_item", "aggregation": "sum"},
            "sku": {"semantic_type": "identifier", "data_type": "string", "source": "line_item", "aggregation": "group_by"}
        }
    }'::jsonb,
    'Extract structured data from this invoice document.

Return a JSON object with the following structure:
{
    "header_data": {
        "invoice_number": "string or null",
        "invoice_date": "YYYY-MM-DD or null",
        "due_date": "YYYY-MM-DD or null",
        "vendor_name": "string or null",
        "vendor_address": "string or null",
        "customer_name": "string or null",
        "customer_address": "string or null",
        "payment_terms": "string or null",
        "currency": "string or null (e.g., CAD, USD, EUR - ONLY if explicitly shown)",
        "po_number": "string or null"
    },
    "line_items": [
        {
            "description": "string",
            "quantity": number,
            "unit_price": number,
            "amount": number,
            "sku": "string or null",
            "tax_rate": number or null
        }
    ],
    "summary_data": {
        "subtotal": number or null,
        "tax_amount": number or null,
        "discount_amount": number or null,
        "shipping_amount": number or null,
        "total_amount": number
    }
}

IMPORTANT:
- Use null for any field where the value is NOT explicitly visible in the document.
- Do NOT assume or guess values. Only extract what is actually shown.
- For currency, only include if explicitly printed. Do not assume based on location.
- Ensure numbers are actual numbers, not strings.
- Dates should be in YYYY-MM-DD format.',
    '{
        "required_fields": ["invoice_number", "total_amount"],
        "date_format": "YYYY-MM-DD",
        "number_fields": ["quantity", "unit_price", "amount", "subtotal", "tax_amount", "total_amount"]
    }'::jsonb,
    '1.0',
    true,
    NOW(),
    NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    description = EXCLUDED.description,
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    validation_rules = EXCLUDED.validation_rules,
    schema_version = EXCLUDED.schema_version,
    updated_at = NOW();

-- 2. RECEIPT SCHEMA
INSERT INTO data_schemas (
    id, schema_type, domain, display_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, schema_version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'receipt',
    'financial',
    'Receipt',
    'Schema for receipt documents including retail purchases, restaurant bills, and transaction receipts',
    '{
        "type": "object",
        "properties": {
            "receipt_number": {"type": "string", "description": "Receipt or transaction number"},
            "transaction_date": {"type": "string", "format": "date", "description": "Date of transaction"},
            "transaction_time": {"type": "string", "description": "Time of transaction (HH:MM)"},
            "store_name": {"type": "string", "description": "Name of the store or merchant"},
            "store_address": {"type": "string", "description": "Store location address"},
            "payment_method": {"type": "string", "description": "Payment method (cash, card, etc.)"},
            "currency": {"type": "string", "description": "Currency code"}
        }
    }'::jsonb,
    '{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Item description"},
                "quantity": {"type": "number", "description": "Quantity purchased"},
                "unit_price": {"type": "number", "description": "Price per unit"},
                "amount": {"type": "number", "description": "Line total"}
            },
            "required": ["description", "amount"]
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "subtotal": {"type": "number"},
            "tax_amount": {"type": "number"},
            "tip_amount": {"type": "number"},
            "total_amount": {"type": "number"}
        },
        "required": ["total_amount"]
    }'::jsonb,
    '{
        "header_fields": {
            "receipt_number": {"semantic_type": "identifier", "data_type": "string", "source": "header", "aggregation": null},
            "transaction_date": {"semantic_type": "date", "data_type": "datetime", "source": "header", "aggregation": null},
            "store_name": {"semantic_type": "entity", "data_type": "string", "source": "header", "aggregation": "group_by"},
            "payment_method": {"semantic_type": "method", "data_type": "string", "source": "header", "aggregation": "group_by"},
            "currency": {"semantic_type": "currency", "data_type": "string", "source": "header", "aggregation": null}
        },
        "line_item_fields": {
            "description": {"semantic_type": "product", "data_type": "string", "source": "line_item", "aggregation": "group_by"},
            "quantity": {"semantic_type": "quantity", "data_type": "number", "source": "line_item", "aggregation": "sum"},
            "unit_price": {"semantic_type": "amount", "data_type": "number", "source": "line_item", "aggregation": null},
            "amount": {"semantic_type": "amount", "data_type": "number", "source": "line_item", "aggregation": "sum"}
        }
    }'::jsonb,
    'Extract structured data from this receipt.

Return a JSON object with:
{
    "header_data": {
        "receipt_number": "string or null",
        "transaction_date": "YYYY-MM-DD or null",
        "transaction_time": "HH:MM or null",
        "store_name": "string or null",
        "store_address": "string or null",
        "payment_method": "string or null",
        "currency": "string or null (ONLY if explicitly shown)"
    },
    "line_items": [
        {
            "description": "string",
            "quantity": number,
            "unit_price": number,
            "amount": number
        }
    ],
    "summary_data": {
        "subtotal": number or null,
        "tax_amount": number or null,
        "tip_amount": number or null,
        "total_amount": number
    }
}

IMPORTANT:
- Use null for any field not explicitly visible.
- For currency: ONLY set if currency code is EXPLICITLY printed.
- The "$" symbol alone does NOT indicate USD.',
    '{"required_fields": ["total_amount"], "date_format": "YYYY-MM-DD", "time_format": "HH:MM"}'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    validation_rules = EXCLUDED.validation_rules,
    updated_at = NOW();

-- 3. BANK STATEMENT SCHEMA
INSERT INTO data_schemas (
    id, schema_type, domain, display_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, schema_version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'bank_statement',
    'financial',
    'Bank Statement',
    'Schema for bank account statements and transaction history',
    '{
        "type": "object",
        "properties": {
            "account_number": {"type": "string", "description": "Account number (masked)"},
            "account_holder": {"type": "string", "description": "Account holder name"},
            "bank_name": {"type": "string", "description": "Name of the bank"},
            "statement_period_start": {"type": "string", "format": "date"},
            "statement_period_end": {"type": "string", "format": "date"},
            "currency": {"type": "string", "description": "Account currency"}
        }
    }'::jsonb,
    '{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "format": "date"},
                "description": {"type": "string"},
                "reference": {"type": "string"},
                "debit": {"type": "number"},
                "credit": {"type": "number"},
                "balance": {"type": "number"}
            },
            "required": ["date", "description"]
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "opening_balance": {"type": "number"},
            "total_deposits": {"type": "number"},
            "total_withdrawals": {"type": "number"},
            "closing_balance": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_fields": {
            "account_number": {"semantic_type": "identifier", "data_type": "string", "source": "header", "aggregation": null},
            "account_holder": {"semantic_type": "entity", "data_type": "string", "source": "header", "aggregation": "group_by"},
            "bank_name": {"semantic_type": "entity", "data_type": "string", "source": "header", "aggregation": "group_by"},
            "statement_period_start": {"semantic_type": "date", "data_type": "datetime", "source": "header", "aggregation": null},
            "statement_period_end": {"semantic_type": "date", "data_type": "datetime", "source": "header", "aggregation": null}
        },
        "line_item_fields": {
            "date": {"semantic_type": "date", "data_type": "datetime", "source": "line_item", "aggregation": null},
            "description": {"semantic_type": "product", "data_type": "string", "source": "line_item", "aggregation": "group_by"},
            "debit": {"semantic_type": "amount", "data_type": "number", "source": "line_item", "aggregation": "sum"},
            "credit": {"semantic_type": "amount", "data_type": "number", "source": "line_item", "aggregation": "sum"},
            "balance": {"semantic_type": "amount", "data_type": "number", "source": "line_item", "aggregation": null}
        }
    }'::jsonb,
    'Extract structured data from this bank statement.

Return a JSON object with:
{
    "header_data": {
        "account_number": "string (last 4 digits only for security)",
        "account_holder": "string or null",
        "bank_name": "string or null",
        "statement_period_start": "YYYY-MM-DD or null",
        "statement_period_end": "YYYY-MM-DD or null",
        "currency": "string or null"
    },
    "line_items": [
        {
            "date": "YYYY-MM-DD",
            "description": "string",
            "reference": "string or null",
            "debit": number or null,
            "credit": number or null,
            "balance": number or null
        }
    ],
    "summary_data": {
        "opening_balance": number or null,
        "total_deposits": number or null,
        "total_withdrawals": number or null,
        "closing_balance": number or null
    }
}

List all transactions in chronological order.',
    '{"required_fields": ["account_number"], "security_fields": ["account_number"], "mask_rules": {"account_number": "last_4_only"}}'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    updated_at = NOW();

-- 4. EXPENSE REPORT SCHEMA
INSERT INTO data_schemas (
    id, schema_type, domain, display_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, schema_version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'expense_report',
    'financial',
    'Expense Report',
    'Schema for expense claims and reimbursement requests',
    '{
        "type": "object",
        "properties": {
            "report_id": {"type": "string"},
            "employee_name": {"type": "string"},
            "department": {"type": "string"},
            "report_period_start": {"type": "string", "format": "date"},
            "report_period_end": {"type": "string", "format": "date"},
            "submission_date": {"type": "string", "format": "date"},
            "currency": {"type": "string"}
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
                "project_code": {"type": "string"}
            }
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "total_claimed": {"type": "number"},
            "total_approved": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_fields": {
            "report_id": {"semantic_type": "identifier", "data_type": "string", "source": "header", "aggregation": null},
            "employee_name": {"semantic_type": "person", "data_type": "string", "source": "header", "aggregation": "group_by"},
            "department": {"semantic_type": "category", "data_type": "string", "source": "header", "aggregation": "group_by"},
            "submission_date": {"semantic_type": "date", "data_type": "datetime", "source": "header", "aggregation": null}
        },
        "line_item_fields": {
            "date": {"semantic_type": "date", "data_type": "datetime", "source": "line_item", "aggregation": null},
            "category": {"semantic_type": "category", "data_type": "string", "source": "line_item", "aggregation": "group_by"},
            "description": {"semantic_type": "product", "data_type": "string", "source": "line_item", "aggregation": null},
            "vendor": {"semantic_type": "entity", "data_type": "string", "source": "line_item", "aggregation": "group_by"},
            "amount": {"semantic_type": "amount", "data_type": "number", "source": "line_item", "aggregation": "sum"}
        }
    }'::jsonb,
    'Extract structured data from this expense report.

Return a JSON object with:
{
    "header_data": {
        "report_id": "string or null",
        "employee_name": "string or null",
        "department": "string or null",
        "report_period_start": "YYYY-MM-DD or null",
        "report_period_end": "YYYY-MM-DD or null",
        "submission_date": "YYYY-MM-DD or null",
        "currency": "string or null"
    },
    "line_items": [
        {
            "date": "YYYY-MM-DD",
            "category": "string (Travel, Meals, Lodging, etc.)",
            "description": "string",
            "vendor": "string or null",
            "amount": number,
            "project_code": "string or null"
        }
    ],
    "summary_data": {
        "total_claimed": number,
        "total_approved": number or null
    }
}',
    '{"required_fields": ["employee_name", "total_claimed"]}'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    updated_at = NOW();

-- 5. PURCHASE ORDER SCHEMA
INSERT INTO data_schemas (
    id, schema_type, domain, display_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, schema_version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'purchase_order',
    'financial',
    'Purchase Order',
    'Schema for purchase orders and procurement documents',
    '{
        "type": "object",
        "properties": {
            "po_number": {"type": "string"},
            "po_date": {"type": "string", "format": "date"},
            "delivery_date": {"type": "string", "format": "date"},
            "vendor_name": {"type": "string"},
            "vendor_address": {"type": "string"},
            "ship_to_address": {"type": "string"},
            "payment_terms": {"type": "string"},
            "currency": {"type": "string"}
        }
    }'::jsonb,
    '{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "line_number": {"type": "number"},
                "sku": {"type": "string"},
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
            "total_amount": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_fields": {
            "po_number": {"semantic_type": "identifier", "data_type": "string", "source": "header", "aggregation": null},
            "po_date": {"semantic_type": "date", "data_type": "datetime", "source": "header", "aggregation": null},
            "vendor_name": {"semantic_type": "entity", "data_type": "string", "source": "header", "aggregation": "group_by"}
        },
        "line_item_fields": {
            "sku": {"semantic_type": "identifier", "data_type": "string", "source": "line_item", "aggregation": "group_by"},
            "description": {"semantic_type": "product", "data_type": "string", "source": "line_item", "aggregation": null},
            "quantity": {"semantic_type": "quantity", "data_type": "number", "source": "line_item", "aggregation": "sum"},
            "unit_price": {"semantic_type": "amount", "data_type": "number", "source": "line_item", "aggregation": null},
            "amount": {"semantic_type": "amount", "data_type": "number", "source": "line_item", "aggregation": "sum"}
        }
    }'::jsonb,
    'Extract structured data from this purchase order.

Return a JSON object with:
{
    "header_data": {
        "po_number": "string",
        "po_date": "YYYY-MM-DD or null",
        "delivery_date": "YYYY-MM-DD or null",
        "vendor_name": "string or null",
        "vendor_address": "string or null",
        "ship_to_address": "string or null",
        "payment_terms": "string or null",
        "currency": "string or null"
    },
    "line_items": [
        {
            "line_number": number,
            "sku": "string or null",
            "description": "string",
            "quantity": number,
            "unit": "string or null",
            "unit_price": number,
            "amount": number
        }
    ],
    "summary_data": {
        "subtotal": number or null,
        "tax_amount": number or null,
        "total_amount": number
    }
}',
    '{"required_fields": ["po_number", "total_amount"]}'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    updated_at = NOW();

-- 6. SHIPPING MANIFEST SCHEMA
INSERT INTO data_schemas (
    id, schema_type, domain, display_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, schema_version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'shipping_manifest',
    'logistics',
    'Shipping Manifest',
    'Schema for shipping documents, delivery notes, and packing lists',
    '{
        "type": "object",
        "properties": {
            "manifest_number": {"type": "string"},
            "ship_date": {"type": "string", "format": "date"},
            "delivery_date": {"type": "string", "format": "date"},
            "shipper_name": {"type": "string"},
            "consignee_name": {"type": "string"},
            "carrier_name": {"type": "string"},
            "tracking_number": {"type": "string"}
        }
    }'::jsonb,
    '{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "item_number": {"type": "string"},
                "description": {"type": "string"},
                "quantity": {"type": "number"},
                "unit": {"type": "string"},
                "weight": {"type": "number"},
                "weight_unit": {"type": "string"}
            }
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "total_packages": {"type": "number"},
            "total_weight": {"type": "number"},
            "freight_charges": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_fields": {
            "manifest_number": {"semantic_type": "identifier", "data_type": "string", "source": "header", "aggregation": null},
            "ship_date": {"semantic_type": "date", "data_type": "datetime", "source": "header", "aggregation": null},
            "shipper_name": {"semantic_type": "entity", "data_type": "string", "source": "header", "aggregation": "group_by"},
            "consignee_name": {"semantic_type": "entity", "data_type": "string", "source": "header", "aggregation": "group_by"},
            "carrier_name": {"semantic_type": "entity", "data_type": "string", "source": "header", "aggregation": "group_by"}
        },
        "line_item_fields": {
            "description": {"semantic_type": "product", "data_type": "string", "source": "line_item", "aggregation": "group_by"},
            "quantity": {"semantic_type": "quantity", "data_type": "number", "source": "line_item", "aggregation": "sum"},
            "weight": {"semantic_type": "quantity", "data_type": "number", "source": "line_item", "aggregation": "sum"}
        }
    }'::jsonb,
    'Extract structured data from this shipping manifest/delivery note.

Return a JSON object with:
{
    "header_data": {
        "manifest_number": "string or null",
        "ship_date": "YYYY-MM-DD or null",
        "delivery_date": "YYYY-MM-DD or null",
        "shipper_name": "string or null",
        "consignee_name": "string or null",
        "carrier_name": "string or null",
        "tracking_number": "string or null"
    },
    "line_items": [
        {
            "item_number": "string or null",
            "description": "string",
            "quantity": number,
            "unit": "string or null",
            "weight": number or null,
            "weight_unit": "string or null"
        }
    ],
    "summary_data": {
        "total_packages": number or null,
        "total_weight": number or null,
        "freight_charges": number or null
    }
}',
    '{"required_fields": ["manifest_number"]}'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    updated_at = NOW();

-- 7. INVENTORY REPORT SCHEMA
INSERT INTO data_schemas (
    id, schema_type, domain, display_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, schema_version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'inventory_report',
    'inventory',
    'Inventory Report',
    'Schema for inventory reports and stock level documents',
    '{
        "type": "object",
        "properties": {
            "report_id": {"type": "string"},
            "report_date": {"type": "string", "format": "date"},
            "warehouse_name": {"type": "string"},
            "warehouse_location": {"type": "string"},
            "report_type": {"type": "string"}
        }
    }'::jsonb,
    '{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "product_name": {"type": "string"},
                "category": {"type": "string"},
                "location": {"type": "string"},
                "quantity_on_hand": {"type": "number"},
                "quantity_available": {"type": "number"},
                "unit_cost": {"type": "number"},
                "total_value": {"type": "number"}
            }
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "total_skus": {"type": "number"},
            "total_quantity": {"type": "number"},
            "total_value": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_fields": {
            "report_id": {"semantic_type": "identifier", "data_type": "string", "source": "header", "aggregation": null},
            "report_date": {"semantic_type": "date", "data_type": "datetime", "source": "header", "aggregation": null},
            "warehouse_name": {"semantic_type": "entity", "data_type": "string", "source": "header", "aggregation": "group_by"}
        },
        "line_item_fields": {
            "sku": {"semantic_type": "identifier", "data_type": "string", "source": "line_item", "aggregation": "group_by"},
            "product_name": {"semantic_type": "product", "data_type": "string", "source": "line_item", "aggregation": "group_by"},
            "category": {"semantic_type": "category", "data_type": "string", "source": "line_item", "aggregation": "group_by"},
            "quantity_on_hand": {"semantic_type": "quantity", "data_type": "number", "source": "line_item", "aggregation": "sum"},
            "unit_cost": {"semantic_type": "amount", "data_type": "number", "source": "line_item", "aggregation": null},
            "total_value": {"semantic_type": "amount", "data_type": "number", "source": "line_item", "aggregation": "sum"}
        }
    }'::jsonb,
    'Extract structured data from this inventory report.

Return a JSON object with:
{
    "header_data": {
        "report_id": "string or null",
        "report_date": "YYYY-MM-DD or null",
        "warehouse_name": "string or null",
        "warehouse_location": "string or null",
        "report_type": "string or null"
    },
    "line_items": [
        {
            "sku": "string",
            "product_name": "string",
            "category": "string or null",
            "location": "string or null",
            "quantity_on_hand": number,
            "quantity_available": number or null,
            "unit_cost": number or null,
            "total_value": number or null
        }
    ],
    "summary_data": {
        "total_skus": number or null,
        "total_quantity": number or null,
        "total_value": number or null
    }
}',
    '{"required_fields": ["report_date"]}'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    updated_at = NOW();

-- 8. SPREADSHEET SCHEMA (Dynamic/Generic)
INSERT INTO data_schemas (
    id, schema_type, domain, display_name, description,
    header_schema, line_items_schema, summary_schema, field_mappings,
    extraction_prompt, validation_rules, schema_version, is_active, created_at, updated_at
) VALUES (
    gen_random_uuid(),
    'spreadsheet',
    'generic',
    'Spreadsheet',
    'Dynamic schema for spreadsheets, CSVs, and tabular data with unknown structure',
    '{
        "type": "object",
        "properties": {
            "sheet_name": {"type": "string"},
            "column_headers": {"type": "array", "items": {"type": "string"}},
            "total_rows": {"type": "number"},
            "total_columns": {"type": "number"},
            "has_header_row": {"type": "boolean"}
        }
    }'::jsonb,
    '{
        "type": "array",
        "description": "Dynamic structure - fields determined at extraction time",
        "items": {
            "type": "object",
            "additionalProperties": true
        }
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "row_count": {"type": "number"},
            "column_count": {"type": "number"}
        }
    }'::jsonb,
    '{
        "header_fields": {
            "sheet_name": {"semantic_type": "identifier", "data_type": "string", "source": "header", "aggregation": null},
            "total_rows": {"semantic_type": "quantity", "data_type": "number", "source": "header", "aggregation": null},
            "total_columns": {"semantic_type": "quantity", "data_type": "number", "source": "header", "aggregation": null}
        },
        "line_item_fields": {},
        "dynamic": true,
        "note": "Line item fields are inferred at extraction time based on column headers"
    }'::jsonb,
    'Extract structured data from this spreadsheet.

Identify the column headers and extract all data rows.

Return a JSON object with:
{
    "header_data": {
        "sheet_name": "string",
        "column_headers": ["col1", "col2", ...],
        "total_rows": number,
        "total_columns": number
    },
    "line_items": [
        {
            "row_number": number,
            ... // key-value pairs for each column
        }
    ],
    "summary_data": {
        "row_count": number,
        "column_count": number
    }
}

IMPORTANT:
- Preserve the original column names as keys in line_items.
- Convert numeric values to numbers, dates to YYYY-MM-DD format.
- Handle empty cells as null values.
- Detect and use the first row as headers if it contains column names.',
    '{"dynamic_schema": true, "infer_types": true, "header_detection": "auto"}'::jsonb,
    '1.0', true, NOW(), NOW()
)
ON CONFLICT (schema_type) DO UPDATE SET
    header_schema = EXCLUDED.header_schema,
    line_items_schema = EXCLUDED.line_items_schema,
    summary_schema = EXCLUDED.summary_schema,
    field_mappings = EXCLUDED.field_mappings,
    extraction_prompt = EXCLUDED.extraction_prompt,
    updated_at = NOW();

-- =============================================================================
-- VERIFICATION QUERY
-- =============================================================================
SELECT schema_type, display_name, domain, is_active, created_at
FROM data_schemas
ORDER BY domain, schema_type;
