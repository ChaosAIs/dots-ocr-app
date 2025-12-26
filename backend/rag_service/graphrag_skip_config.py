"""
GraphRAG Skip Configuration

This module defines which document types and file extensions should skip GraphRAG indexing.
GraphRAG entity extraction is valuable for narrative content but wasteful for tabular/structured data.

The skip logic is aligned with existing document type definitions in:
- rag_service/chunking/document_types.py (27 core types)
- rag_service/chunking/domain_patterns.py (54 detailed types)
- extraction_service/extraction_config.py (extractable types)
"""

import re
from pathlib import Path
from typing import Tuple, Optional


class GraphRAGSkipReason:
    """Reasons for skipping GraphRAG indexing."""
    FILE_TYPE = "file_type"
    DOCUMENT_TYPE = "document_type"
    CONTENT_PATTERN = "content_pattern"
    USER_DISABLED = "user_disabled"


# =============================================================================
# File Extensions - Always Skip GraphRAG (inherently tabular/structured)
# =============================================================================

GRAPHRAG_SKIP_FILE_EXTENSIONS = {
    # Spreadsheet formats
    '.xlsx', '.xls', '.xlsm', '.xlsb',
    # Delimited data
    '.csv', '.tsv',
    # Structured data formats
    '.json', '.xml', '.yaml', '.yml',
    # Technical/logs
    '.log', '.sql',
}


# =============================================================================
# Document Types - Skip GraphRAG (aligned with existing document_types.py)
# =============================================================================

# These document types produce low-value entities from GraphRAG
# because they contain primarily tabular data, line items, or structured fields

GRAPHRAG_SKIP_DOCUMENT_TYPES = {
    # ===== Financial/Transactional (tabular line items) =====
    'invoice',              # Line items: product, qty, price
    'receipt',              # Transaction items, amounts
    'bank_statement',       # Transaction rows: date, desc, amount
    'expense_report',       # Expense line items
    'purchase_order',       # Order line items
    'quotation',            # Price quotations
    'tax_document',         # Tax forms, numeric data
    'tax_form',             # Tax forms

    # ===== Spreadsheet/Data Types =====
    'spreadsheet',          # Generic spreadsheet
    'excel',                # Excel workbook
    'csv',                  # CSV data
    'worksheet',            # Spreadsheet worksheet
    'data_export',          # Exported data tables
    'database_export',      # Database dumps
    'data_table',           # Generic data tables

    # ===== Logistics (manifests, lists) =====
    'shipping_document',    # Shipping manifests
    'shipping_manifest',    # Item lists
    'bill_of_lading',       # Cargo lists
    'customs_declaration',  # Item declarations
    'delivery_note',        # Delivery item lists
    'packing_list',         # Package contents

    # ===== Inventory/Stock =====
    'inventory_report',     # Stock lists
    'inventory_list',       # Inventory counts
    'stock_report',         # Stock reports
    'price_list',           # Product/price tables
    'catalog',              # Product catalogs
    'product_list',         # Product listings

    # ===== Financial Statements (numeric tables) =====
    'financial_report',     # Financial tables
    'financial_statement',  # Financial statements
    'balance_sheet',        # Accounting tables
    'income_statement',     # P&L tables
    'cash_flow',            # Cash flow tables
    'cash_flow_statement',  # Cash flow statements
    'ledger',               # Accounting ledger
    'journal_entry',        # Accounting entries
    'general_ledger',       # GL entries

    # ===== Forms (structured fields, not narrative) =====
    'government_form',      # Form fields, not narrative
    'application_form',     # Application fields
    'registration_form',    # Registration data
    'form',                 # Generic forms

    # ===== Technical Data =====
    'datasheet',            # Specification tables
    'schematic',            # Diagrams, not text
    'log_file',             # Log entries
    'audit_log',            # Audit entries

    # ===== Transaction Reports =====
    'transaction_report',   # Transaction listings
    'transaction_history',  # Transaction history
    'payment_history',      # Payment records
    'order_history',        # Order records
}


# Document types that SHOULD be processed by GraphRAG (narrative content)
# These types benefit from entity/relationship extraction

GRAPHRAG_PROCESS_DOCUMENT_TYPES = {
    # ===== Legal (rich in entities/relationships) =====
    'contract',             # Parties, terms, obligations
    'agreement',            # Parties, conditions
    'legal_brief',          # Cases, arguments, citations
    'court_filing',         # Parties, claims
    'patent',               # Inventors, claims, references
    'policy',               # Rules, conditions
    'terms_of_service',     # Terms, conditions
    'privacy_policy',       # Data handling policies
    'compliance_doc',       # Regulations, requirements

    # ===== Technical Narrative =====
    'technical_doc',        # Concepts, relationships
    'user_manual',          # Procedures, components
    'api_documentation',    # Endpoints, parameters
    'architecture_doc',     # Components, interactions
    'installation_guide',   # Steps, requirements
    'technical_spec',       # Technical specifications

    # ===== Academic/Research =====
    'research_paper',       # Authors, citations, findings
    'thesis',               # Arguments, references
    'case_study',           # Subjects, outcomes
    'academic_article',     # Authors, findings
    'literature_review',    # References, analysis

    # ===== Medical =====
    'medical_record',       # Patients, conditions, treatments
    'clinical_report',      # Findings, diagnoses
    'patient_summary',      # Patient information

    # ===== Professional =====
    'resume',               # Person, skills, experience
    'cv',                   # Person, skills, experience
    'job_description',      # Role, requirements
    'cover_letter',         # Person, qualifications

    # ===== Creative/Narrative =====
    'article',              # Topics, people, events
    'blog_post',            # Topics, opinions
    'news_article',         # Events, people, places
    'report',               # Findings, recommendations
    'memo',                 # Topics, actions
    'letter',               # Parties, topics
    'meeting_notes',        # Attendees, decisions
    'presentation',         # Topics, key points
    'book_chapter',         # Narrative content
    'manuscript',           # Narrative content

    # ===== Educational =====
    'course_material',      # Concepts, relationships
    'textbook',             # Educational content
    'tutorial',             # Instructions, concepts
}


# Aliases for flexible matching
DOCUMENT_TYPE_ALIASES = {
    # Invoice aliases
    'invoice': ['bill', 'billing', 'billing_document', 'vendor_invoice'],
    # Receipt aliases
    'receipt': ['meal_receipt', 'expense_receipt', 'purchase_receipt', 'sales_receipt'],
    # Bank statement aliases
    'bank_statement': ['account_statement', 'statement', 'bank_account_statement'],
    # Spreadsheet aliases
    'spreadsheet': ['workbook', 'worksheet', 'table_data', 'excel_file'],
    # Inventory aliases
    'inventory_report': ['inventory', 'stock_list', 'inventory_list'],
    # Financial report aliases
    'financial_report': ['financial', 'financials', 'financial_statement'],
}


def should_skip_graphrag_for_document_type(document_type: str) -> Tuple[bool, Optional[str]]:
    """
    Determine if a document type should skip GraphRAG indexing.

    Args:
        document_type: The document type string (from metadata extraction)

    Returns:
        Tuple of (should_skip: bool, reason: str or None)
    """
    if not document_type:
        return False, None

    doc_type_lower = document_type.lower().strip()

    # Direct match against skip list
    if doc_type_lower in GRAPHRAG_SKIP_DOCUMENT_TYPES:
        return True, f"{GraphRAGSkipReason.DOCUMENT_TYPE}:{doc_type_lower}"

    # Check aliases
    for skip_type, aliases in DOCUMENT_TYPE_ALIASES.items():
        if doc_type_lower in aliases:
            return True, f"{GraphRAGSkipReason.DOCUMENT_TYPE}:{skip_type}"

    # Check if it's explicitly in the process list (definitely don't skip)
    if doc_type_lower in GRAPHRAG_PROCESS_DOCUMENT_TYPES:
        return False, None

    # Default: don't skip unknown types (let GraphRAG try)
    return False, None


def should_skip_graphrag_for_file(filename: str) -> Tuple[bool, Optional[str]]:
    """
    Determine if a file should skip GraphRAG based on its extension.

    Args:
        filename: The filename with extension

    Returns:
        Tuple of (should_skip: bool, reason: str or None)
    """
    if not filename:
        return False, None

    ext = Path(filename).suffix.lower()

    if ext in GRAPHRAG_SKIP_FILE_EXTENSIONS:
        return True, f"{GraphRAGSkipReason.FILE_TYPE}:{ext}"

    return False, None


def is_tabular_chunk_content(content: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if chunk content is primarily tabular/list data.

    Used for mixed documents (e.g., PDF with embedded tables) to skip
    specific chunks while processing narrative chunks.

    Args:
        content: The chunk text content

    Returns:
        Tuple of (is_tabular: bool, reason: str or None)
    """
    if not content or len(content) < 50:
        return False, None

    lines = content.strip().split('\n')
    if len(lines) < 3:
        return False, None

    # Check 1: Markdown table pattern (pipe-separated with header separator)
    pipe_lines = [line for line in lines if line.count('|') >= 2]
    separator_lines = [line for line in lines if re.match(r'^[\s|:\-]+$', line)]
    if len(pipe_lines) >= 3 and len(separator_lines) >= 1:
        return True, f"{GraphRAGSkipReason.CONTENT_PATTERN}:markdown_table"

    # Check 2: High numeric density (>40% numeric characters)
    content_no_space = re.sub(r'\s', '', content)
    if len(content_no_space) > 0:
        numeric_chars = len(re.findall(r'[\d.,]', content))
        if numeric_chars / len(content_no_space) > 0.4:
            return True, f"{GraphRAGSkipReason.CONTENT_PATTERN}:high_numeric_density"

    # Check 3: Line item pattern (description + quantity + price)
    # Pattern: some text, then a number, then a currency amount
    line_item_pattern = r'.{5,50}\s+\d+\s*[xX×]?\s*[\$€£¥]?\d+[.,]\d{2}'
    matches = re.findall(line_item_pattern, content)
    if len(matches) >= 3:
        return True, f"{GraphRAGSkipReason.CONTENT_PATTERN}:line_items"

    # Check 4: Transaction pattern (date + description + amount)
    transaction_pattern = r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}.{10,50}[\$€£¥]?\d+[.,]\d{2}'
    matches = re.findall(transaction_pattern, content)
    if len(matches) >= 3:
        return True, f"{GraphRAGSkipReason.CONTENT_PATTERN}:transaction_list"

    # Check 5: Short repetitive lines (typical of tabular data)
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    if len(non_empty_lines) >= 8:
        avg_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
        if avg_length < 40:
            return True, f"{GraphRAGSkipReason.CONTENT_PATTERN}:short_repetitive_lines"

    # Check 6: CSV-like pattern (consistent comma count per line)
    if len(lines) >= 5:
        comma_counts = [line.count(',') for line in lines[:10] if line.strip()]
        if comma_counts and min(comma_counts) >= 2:
            # Check if comma counts are consistent (variance is low)
            avg_commas = sum(comma_counts) / len(comma_counts)
            variance = sum((c - avg_commas) ** 2 for c in comma_counts) / len(comma_counts)
            if variance < 2:  # Very consistent comma counts
                return True, f"{GraphRAGSkipReason.CONTENT_PATTERN}:csv_like"

    return False, None


def should_skip_graphrag(
    filename: str = None,
    document_type: str = None,
    chunk_content: str = None,
    skip_graphrag_flag: bool = False,
    skip_graphrag_reason: str = None
) -> Tuple[bool, Optional[str]]:
    """
    Comprehensive check for whether to skip GraphRAG indexing.

    Checks in priority order:
    1. Document-level skip flag (highest priority)
    2. File extension
    3. Document type from metadata
    4. Content pattern analysis (optional, for mixed documents)

    Args:
        filename: The document filename
        document_type: The extracted document type from metadata
        chunk_content: The chunk text content (optional, for content-based detection)
        skip_graphrag_flag: Document-level skip flag from database
        skip_graphrag_reason: Existing skip reason from database

    Returns:
        Tuple of (should_skip: bool, reason: str or None)
    """
    # 1. Check document-level skip flag
    if skip_graphrag_flag:
        return True, skip_graphrag_reason or f"{GraphRAGSkipReason.USER_DISABLED}:document_flag"

    # 2. Check file extension
    if filename:
        should_skip, reason = should_skip_graphrag_for_file(filename)
        if should_skip:
            return True, reason

    # 3. Check document type
    if document_type:
        should_skip, reason = should_skip_graphrag_for_document_type(document_type)
        if should_skip:
            return True, reason

    # 4. Check content patterns (optional, more expensive)
    if chunk_content:
        should_skip, reason = is_tabular_chunk_content(chunk_content)
        if should_skip:
            return True, reason

    return False, None
