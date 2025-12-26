"""
Extraction Configuration

Defines which document types are extractable and maps them to schemas.
"""

import os
from typing import Dict, Optional, List
from enum import Enum


class ExtractionStrategy(str, Enum):
    """Extraction strategy types."""
    LLM_DIRECT = "llm_direct"       # Single LLM call (< 50 rows)
    LLM_CHUNKED = "llm_chunked"     # Parallel chunked LLM (50-500 rows)
    HYBRID = "hybrid"                # LLM for headers, rules for data (500-5000 rows)
    PARSED = "parsed"                # Pure pattern matching (> 5000 rows)


# Thresholds from environment
DIRECT_LLM_MAX_ROWS = int(os.getenv("EXTRACTION_DIRECT_LLM_MAX_ROWS", "50"))
CHUNKED_LLM_MAX_ROWS = int(os.getenv("EXTRACTION_CHUNKED_LLM_MAX_ROWS", "500"))
HYBRID_MAX_ROWS = int(os.getenv("EXTRACTION_HYBRID_MAX_ROWS", "5000"))


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


# Strategies for each source type
EXTRACTION_STRATEGIES: Dict[str, Dict[str, str]] = {
    # Spreadsheet sources can use parsing more easily
    "xlsx": {
        "default": ExtractionStrategy.PARSED.value,
        "small": ExtractionStrategy.LLM_DIRECT.value,
    },
    "xls": {
        "default": ExtractionStrategy.PARSED.value,
        "small": ExtractionStrategy.LLM_DIRECT.value,
    },
    "csv": {
        "default": ExtractionStrategy.PARSED.value,
        "small": ExtractionStrategy.LLM_DIRECT.value,
    },

    # PDF needs more LLM involvement
    "pdf": {
        "default": ExtractionStrategy.HYBRID.value,
        "small": ExtractionStrategy.LLM_DIRECT.value,
        "medium": ExtractionStrategy.LLM_CHUNKED.value,
    },

    # Images need OCR + LLM
    "image": {
        "default": ExtractionStrategy.LLM_DIRECT.value,
        "small": ExtractionStrategy.LLM_DIRECT.value,
    },
    "png": {
        "default": ExtractionStrategy.LLM_DIRECT.value,
    },
    "jpg": {
        "default": ExtractionStrategy.LLM_DIRECT.value,
    },
    "jpeg": {
        "default": ExtractionStrategy.LLM_DIRECT.value,
    },
}


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


def determine_extraction_strategy(
    source_type: str,
    estimated_rows: int
) -> ExtractionStrategy:
    """
    Determine the best extraction strategy based on document characteristics.

    Args:
        source_type: Source file type (pdf, xlsx, csv, etc.)
        estimated_rows: Estimated number of rows in the document

    Returns:
        ExtractionStrategy enum value
    """
    source_lower = source_type.lower() if source_type else "unknown"

    # Get strategy config for source type
    strategy_config = EXTRACTION_STRATEGIES.get(source_lower, {
        "default": ExtractionStrategy.LLM_DIRECT.value,
        "small": ExtractionStrategy.LLM_DIRECT.value,
    })

    # Determine based on row count
    if estimated_rows < DIRECT_LLM_MAX_ROWS:
        return ExtractionStrategy(strategy_config.get("small", strategy_config["default"]))
    elif estimated_rows < CHUNKED_LLM_MAX_ROWS:
        return ExtractionStrategy(strategy_config.get("medium", strategy_config["default"]))
    elif estimated_rows < HYBRID_MAX_ROWS:
        return ExtractionStrategy(strategy_config.get("large", ExtractionStrategy.HYBRID.value))
    else:
        return ExtractionStrategy.PARSED


# Default extraction prompts for each schema type
EXTRACTION_PROMPTS: Dict[str, str] = {
    "invoice": """Extract structured data from this invoice document.

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
        "currency": "string or null (e.g., CAD, USD, EUR - ONLY if explicitly shown on invoice, otherwise null)"
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
        "total_amount": number
    }
}

IMPORTANT:
- Use null for any field where the value is NOT explicitly visible in the document.
- Do NOT assume or guess values. Only extract what is actually shown.
- For currency, only include if explicitly printed. Do not assume based on location.
- Ensure numbers are actual numbers, not strings.
- Dates should be in YYYY-MM-DD format.""",

    "receipt": """Extract structured data from this receipt.

Return a JSON object with:
{
    "header_data": {
        "receipt_number": "string or null",
        "transaction_date": "YYYY-MM-DD or null",
        "transaction_time": "HH:MM or null",
        "store_name": "string or null",
        "store_address": "string or null",
        "payment_method": "string or null",
        "currency": "string or null (e.g., CAD, USD, EUR - ONLY if explicitly shown on receipt, otherwise null)"
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
        "total_amount": number
    }
}

IMPORTANT:
- Use null for any field where the value is NOT explicitly visible in the receipt.
- Do NOT assume or guess values. Only extract what is actually shown.
- For currency: ONLY set a value if the currency code (CAD, USD, EUR, etc.) is EXPLICITLY printed on the receipt.
  The "$" symbol alone does NOT indicate USD - it could be CAD, AUD, or other dollar currencies.
  If only "$" symbols are shown without a currency code, set currency to null.""",

    "bank_statement": """Extract structured data from this bank statement.

Return a JSON object with:
{
    "header_data": {
        "account_number": "string or null (last 4 digits only for security)",
        "account_holder": "string or null",
        "bank_name": "string or null",
        "statement_period_start": "YYYY-MM-DD or null",
        "statement_period_end": "YYYY-MM-DD or null",
        "currency": "string or null (e.g., CAD, USD, EUR - ONLY if explicitly shown, otherwise null)"
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

IMPORTANT:
- Use null for any field where the value is NOT explicitly visible in the document.
- Do NOT assume or guess values. Only extract what is actually shown.
- For currency, only include if explicitly printed. Do not assume based on location.
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
