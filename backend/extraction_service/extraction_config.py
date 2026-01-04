"""
Extraction Configuration

Defines which document types are extractable and maps them to schemas.

NOTE: As of migration 022, the following have been REMOVED:
- ExtractionStrategy enum (LLM_DIRECT, LLM_CHUNKED, HYBRID, PARSED)
- EXTRACTION_*_MAX_ROWS thresholds
- determine_extraction_strategy() function
- EXTRACTION_STRATEGIES configuration

All tabular data extraction now uses direct parsing only.
LLM is only used for document classification and field mapping inference.
All extracted row data is stored in documents_data_line_items table (external storage only).
"""

from typing import Dict, Optional, List


# Document types that can be extracted to structured data
# Maps document_type (from document_metadata) to schema_type
EXTRACTABLE_DOCUMENT_TYPES: Dict[str, str] = {
    # Financial Domain
    "invoice": "invoice",
    "receipt": "receipt",
    "bank_statement": "bank_statement",
    "bank statement": "bank_statement",
    "financial_report": "financial_report",
    "expense_report": "expense_report",
    "expense report": "expense_report",
    "purchase_order": "purchase_order",
    "purchase order": "purchase_order",
    "tax_document": "tax_document",
    "tax document": "tax_document",

    # Logistics Domain
    "shipping_document": "shipping_manifest",
    "shipping document": "shipping_manifest",
    "shipping_manifest": "shipping_manifest",
    "bill_of_lading": "bill_of_lading",
    "bill of lading": "bill_of_lading",
    "customs_declaration": "customs_declaration",
    "delivery_note": "shipping_manifest",
    "delivery note": "shipping_manifest",

    # Inventory Domain
    "inventory_report": "inventory_report",
    "inventory report": "inventory_report",
    "stock_report": "inventory_report",
    "stock report": "inventory_report",

    # Spreadsheet Types (always extractable)
    "spreadsheet": "spreadsheet",
    "excel": "spreadsheet",
    "csv": "spreadsheet",
    "worksheet": "spreadsheet",
}


# Document types that should NOT be extracted
NON_EXTRACTABLE_TYPES: List[str] = [
    "contract",
    "legal_filing",
    "legal filing",
    "research_paper",
    "research paper",
    "textbook",
    "manual",
    "guide",
    "resume",
    "cv",
    "news_article",
    "news article",
    "email",
    "memo",
    "letter",
    "presentation",
    "whitepaper",
    "white paper",
    "thesis",
    "dissertation",
    "report",  # Generic reports - too varied
    "documentation",
    "specification",
]




def is_extractable_document_type(document_type: str) -> bool:
    """
    Check if a document type is extractable.

    Args:
        document_type: Document type string

    Returns:
        True if extractable, False otherwise
    """
    if not document_type:
        return False

    doc_type_lower = document_type.lower().strip()

    # Check if in non-extractable list
    if doc_type_lower in NON_EXTRACTABLE_TYPES:
        return False

    # Check if in extractable list
    return doc_type_lower in EXTRACTABLE_DOCUMENT_TYPES


def get_schema_for_document_type(document_type: str) -> Optional[str]:
    """
    Get the schema type for a document type.

    Args:
        document_type: Document type string

    Returns:
        Schema type string or None if not extractable
    """
    if not document_type:
        return None

    doc_type_lower = document_type.lower().strip()
    return EXTRACTABLE_DOCUMENT_TYPES.get(doc_type_lower)


# Default extraction prompts for each schema type
# NOTE: These prompts are used for LLM-based header/metadata extraction only.
# Row data is extracted via direct parsing, not LLM.
EXTRACTION_PROMPTS: Dict[str, str] = {
    "invoice": """Extract structured data from this invoice document.

Return a JSON object with the following structure:
{
    "header_data": {
        "invoice_number": "string or null",
        "invoice_date": "YYYY-MM-DD or null",
        "due_date": "YYYY-MM-DD or null",
        "vendor_name": "string or null - The company/business NAME only, NOT the address (e.g., 'Augment Code', 'Acme Corp')",
        "vendor_address": "string or null - The vendor's physical address ONLY, NOT the company name",
        "customer_name": "string or null - The customer/buyer NAME only, NOT the address",
        "customer_address": "string or null - The customer's address ONLY, NOT the customer name",
        "payment_terms": "string or null",
        "currency": "null (MUST be null unless a 3-letter currency code like CAD, USD, EUR is explicitly printed)"
    },
    "line_items": [
        {
            "description": "string - the item/service name",
            "quantity": number or null,
            "unit_price": number or null,
            "amount": number or null,
            "price_type": "individual | group_total | included | unknown",
            "is_currency": boolean - true if amount has $ € £ ¥ symbol, false if plain number without currency symbol
        }
    ],
    "summary_data": {
        "subtotal": number or null,
        "tax_amount": number or null,
        "total_amount": number
    }
}

CRITICAL FIELD RULES:

1. NUMERIC FIELDS - These MUST contain numbers:
   - quantity: a number like 1, 5, 465, 600 (or null)
   - unit_price: a number like 50.00, 10.00 (or null)
   - amount: a number like 50.00, 465, 600.00 (or null)

2. is_currency FIELD - This is a SEPARATE boolean field:
   - Set is_currency: true if the amount in the source has $ € £ ¥ symbol (e.g., "$50.00")
   - Set is_currency: false if the amount is a plain number without currency symbol (e.g., "465 User Messages")

3. EXAMPLES - Pay attention to both amount (number) AND is_currency (boolean):

   Example 1: Line shows "Seats 1 x $50.00 $50.00"
   {
       "description": "Seats",
       "quantity": 1,
       "unit_price": 50.00,
       "amount": 50.00,
       "is_currency": true
   }

   Example 2: Line shows "User Messages 600 x 1 User Messages 600 User Messages"
   {
       "description": "User Messages",
       "quantity": 600,
       "unit_price": null,
       "amount": 600,
       "is_currency": false
   }

4. DO NOT ASSUME OR DUPLICATE PRICES:
   - If an item does NOT have a price/amount directly associated with it, set amount to NULL.
   - NEVER copy or duplicate a price from one line item to another.

5. GROUPED/BUNDLED PRICING:
   - For items WITHOUT their own price: set amount: null, price_type: "included"
   - For the item WITH the bundle/group total: set the amount and price_type: "group_total"

6. GENERAL RULES:
   - Use null for any field where the value is NOT explicitly visible.
   - CURRENCY header field MUST BE NULL unless a 3-letter code (CAD, USD, EUR) is explicitly printed.
   - Dates should be in YYYY-MM-DD format.""",

    "receipt": """Extract structured data from this receipt.

Return a JSON object with:
{
    "header_data": {
        "receipt_number": "string or null",
        "transaction_date": "YYYY-MM-DD or null - THE DATE ONLY, NOT the time. Look for dates in various formats like MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD, etc. The date may appear on a separate line from the time.",
        "transaction_time": "HH:MM or null - THE TIME ONLY, extracted from 'Time:' labels or timestamp fields. Do NOT put date information here.",
        "store_name": "string or null",
        "store_address": "string or null",
        "payment_method": "string or null",
        "currency": "null (MUST be null unless a 3-letter currency code like CAD, USD, EUR is explicitly printed)"
    },
    "line_items": [
        {
            "description": "string",
            "quantity": number or null,
            "unit_price": number or null,
            "amount": number or null,
            "is_complimentary": boolean,
            "price_type": "individual | group_total | included | unknown",
            "is_currency": boolean - true if amount has $ € £ ¥ symbol, false if plain number
        }
    ],
    "summary_data": {
        "subtotal": number or null,
        "tax_amount": number or null,
        "total_amount": number
    }
}

CRITICAL FIELD RULES:

1. NUMERIC FIELDS - These MUST contain numbers:
   - quantity: a number like 1, 5, 465, 600 (or null)
   - unit_price: a number like 50.00, 10.00 (or null)
   - amount: a number like 50.00, 465, 600.00 (or null)

2. is_currency FIELD - This is a SEPARATE boolean field:
   - Set is_currency: true if the amount in the source has $ € £ ¥ symbol (e.g., "$5.00")
   - Set is_currency: false if the amount is a plain number without currency symbol (e.g., "100 points")

3. EXAMPLES - Pay attention to both amount (number) AND is_currency (boolean):

   Example 1: Line shows "Coffee 1 x $5.00 $5.00"
   {
       "description": "Coffee",
       "quantity": 1,
       "unit_price": 5.00,
       "amount": 5.00,
       "is_currency": true
   }

   Example 2: Line shows "Loyalty Points 100 points"
   {
       "description": "Loyalty Points",
       "quantity": 100,
       "unit_price": null,
       "amount": 100,
       "is_currency": false
   }

4. DO NOT ASSUME OR DUPLICATE PRICES:
   - If an item does NOT have a price directly next to it on the SAME LINE, set amount to NULL.
   - NEVER copy or duplicate a price from an adjacent item to another item.
   - NEVER assume items without prices cost the same as nearby items.

5. GROUPED PRICING (common in restaurant receipts):
   - Multiple items may share ONE price (e.g., set meals, combo dishes).
   - For items in a group WITHOUT their own price: set amount: null, price_type: "included"
   - For the item WITH the group total: set the amount and price_type: "group_total"
   - Example: "Fish", "Rice", "Soup" with only "Soup" showing "$138":
     * Fish: amount: null, price_type: "included"
     * Rice: amount: null, price_type: "included"
     * Soup: amount: 138, price_type: "group_total", is_currency: true

6. COMPLIMENTARY/FREE ITEMS:
   - Items marked with "(送)", "free", "complimentary", "gift", or showing $0.00:
     set is_complimentary: true, amount: 0, price_type: "individual", is_currency: true
   - Chinese receipts: "(送)" means complimentary/gifted

7. PRICE TYPE VALUES:
   - "individual": Item has its own specific price
   - "group_total": This price covers multiple items above it
   - "included": Item is part of a group, price shown on another item
   - "unknown": Cannot determine pricing structure

8. DATE AND TIME EXTRACTION (CRITICAL - common mistakes):
   - transaction_date MUST be a DATE (YYYY-MM-DD format), NOT a time
   - transaction_time MUST be a TIME (HH:MM format), NOT a date
   - ALWAYS extract date and time SEPARATELY even if they appear together

   EXAMPLE 1 - Date and time on SEPARATE lines:
   Receipt shows:
     "11/8/2025"
     "Time: 13:23"
   Extract as:
     "transaction_date": "2025-11-08"   ← from "11/8/2025"
     "transaction_time": "13:23"        ← from "Time: 13:23"

   EXAMPLE 2 - Date and time COMBINED on one line:
   Receipt shows:
     "Time:2025-06-18 11:55:01 AM"
   Extract as:
     "transaction_date": "2025-06-18"   ← extract ONLY the date portion
     "transaction_time": "11:55"        ← extract ONLY the time portion

   EXAMPLE 3 - Various date formats to recognize:
     "11/8/2025" → "2025-11-08"
     "2025-06-18" → "2025-06-18"
     "Jun 18, 2025" → "2025-06-18"
     "18/06/2025" → "2025-06-18"

9. GENERAL RULES:
   - Use null for any field where the value is NOT explicitly visible.
   - Do NOT assume or guess values. Only extract what is actually shown.
   - CURRENCY MUST BE NULL unless a 3-letter currency code (CAD, USD, EUR, GBP, etc.) is EXPLICITLY printed.""",

    "bank_statement": """Extract structured data from this bank statement.

Return a JSON object with:
{
    "header_data": {
        "account_number": "string or null (last 4 digits only for security)",
        "account_holder": "string or null",
        "bank_name": "string or null",
        "statement_period_start": "YYYY-MM-DD or null",
        "statement_period_end": "YYYY-MM-DD or null",
        "currency": "null (MUST be null unless a 3-letter currency code like CAD, USD, EUR is explicitly printed)"
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

CRITICAL RULES:
- Use null for any field where the value is NOT explicitly visible in the document.
- Do NOT assume or guess values. Only extract what is actually shown.
- CURRENCY MUST BE NULL unless a 3-letter currency code (CAD, USD, EUR, GBP, etc.) is EXPLICITLY printed.
  * The "$" symbol DOES NOT indicate USD - it is used by CAD, AUD, NZD, and many other currencies.
- List all transactions in chronological order.""",

    "spreadsheet": """Extract structured data from this spreadsheet.

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

Preserve the original column names as keys in line_items.
Convert numeric values to numbers, dates to YYYY-MM-DD format.""",
}


def get_extraction_prompt(schema_type: str) -> str:
    """
    Get the extraction prompt for a schema type.

    Args:
        schema_type: Schema type string

    Returns:
        Extraction prompt string
    """
    return EXTRACTION_PROMPTS.get(
        schema_type,
        EXTRACTION_PROMPTS.get("spreadsheet")  # Default to spreadsheet prompt
    )
