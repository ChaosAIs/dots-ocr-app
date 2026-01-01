"""
Centralized Document Type Classifier

Provides unified LLM-driven document type classification for:
- Extraction eligibility (schema type for data extraction)
- Chunking strategy (document structure for RAG)

IMPORTANT: This module imports document types from rag_service.chunking.document_types
which is the single source of truth for all document type definitions.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Import from the single source of truth
# Note: Using absolute import because this is in 'common' module, not 'rag_service'
try:
    from rag_service.chunking.document_types import (
        DOCUMENT_TYPES,
        DocumentTypeInfo,
        DocumentCategory,
        TYPE_ALIASES,
        get_document_type_info,
        is_extractable_type,
        get_schema_type,
        get_chunking_config,
        get_extractable_types,
        get_all_document_types,
        generate_metadata_extraction_types,
    )
except ImportError:
    # Fallback for standalone testing
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rag_service.chunking.document_types import (
        DOCUMENT_TYPES,
        DocumentTypeInfo,
        DocumentCategory,
        TYPE_ALIASES,
        get_document_type_info,
        is_extractable_type,
        get_schema_type,
        get_chunking_config,
        get_extractable_types,
        get_all_document_types,
        generate_metadata_extraction_types,
    )

logger = logging.getLogger(__name__)


# Re-export DocumentCategory as DocumentDomain for backward compatibility
class DocumentDomain(str, Enum):
    """Document domain categories (alias for DocumentCategory)."""
    FINANCIAL = "business_finance"
    LEGAL = "legal"
    MEDICAL = "medical"
    ACADEMIC = "academic"
    EDUCATION = "education"
    ENGINEERING = "technical"
    GOVERNMENT = "government"
    PROFESSIONAL = "professional"
    LOGISTICS = "logistics"
    GENERIC = "general"


# Backward compatibility: simple alias mapping for resolve_type
SIMPLE_TYPE_ALIASES: Dict[str, str] = {
    # Receipt aliases
    "sales_receipt": "receipt",
    "transaction": "receipt",
    "purchase_receipt": "receipt",
    "meal_receipt": "receipt",
    "restaurant_bill": "receipt",
    "pos_receipt": "receipt",

    # Invoice aliases
    "bill": "invoice",
    "billing": "invoice",
    "vendor_invoice": "invoice",

    # Bank statement aliases
    "account_statement": "bank_statement",
    "financial_statement": "bank_statement",
    "statement": "bank_statement",

    # Expense aliases
    "expense_claim": "expense_report",
    "reimbursement": "expense_report",

    # Purchase order aliases
    "po": "purchase_order",
    "procurement": "purchase_order",

    # Shipping aliases
    "packing_list": "shipping_manifest",
    "delivery_note": "shipping_manifest",
    "bill_of_lading": "shipping_manifest",

    # Inventory aliases
    "stock_report": "inventory_report",
    "inventory_list": "inventory_report",

    # Spreadsheet aliases
    "excel": "spreadsheet",
    "csv": "spreadsheet",
    "table": "spreadsheet",
    "tabular_data": "spreadsheet",

    # Legal/Policy aliases
    "agreement": "contract",
    "terms_and_conditions": "terms_of_service",
    "return_policy": "policy",
    "refund_policy": "policy",

    # Academic aliases
    "dissertation": "thesis",
    "paper": "research_paper",

    # Resume aliases
    "cv": "resume",
    "curriculum_vitae": "resume",

    # Technical aliases
    "technical_spec": "technical_doc",
    "manual": "user_manual",
    "guide": "user_manual",
}


# LLM Classification Prompt - uses dynamically generated type list
DOCUMENT_CLASSIFICATION_PROMPT = """You are a document classification expert. Analyze the provided document information and classify it into one of the supported document types.

## Available Document Types:
{available_types}

## Document Information:
- Filename: {filename}
- Current Classification: {current_type}
- Subject/Title: {subject_name}
- Summary: {summary}
- Domain: {domain}
- Entities Found: {entities}
- Has Tables/Structured Data: {has_tables}
- Has Monetary Amounts: {has_amounts}
- File Extension: {file_extension}

## Classification Guidelines:
1. Choose the MOST SPECIFIC type that matches the document
2. For financial transactions with itemized lists (meals, purchases), use "receipt"
3. For billing documents with charges to pay, use "invoice"
4. For Excel/CSV files with tabular data, use "spreadsheet"
5. For bank account statements, use "bank_statement"
6. For shipping/delivery documents, use "shipping_manifest"
7. For company policies, return policies, refund policies, use "policy"
8. For terms of service documents, use "terms_of_service"
9. For privacy policies, use "privacy_policy"
10. If the document doesn't clearly match any type, use "other"

## Response Format (JSON only):
{{
    "document_type": "the_type_name",
    "confidence": 0.0-1.0,
    "domain": "the_domain",
    "reasoning": "Brief explanation"
}}

Respond with JSON only:"""


# LLM Multi-Type Classification Prompt - for documents that can match multiple types
MULTI_TYPE_CLASSIFICATION_PROMPT = """You are a document classification expert. A document can belong to MULTIPLE types based on its content and format.

For example:
- A CSV file with product inventory data is both a "spreadsheet" AND an "inventory_report"
- A PDF with financial transactions could be both a "report" AND a "financial_report"
- An Excel file with sales data could be "spreadsheet", "report", and potentially "invoice" if it contains invoice data

## Available Document Types:
{available_types}

## Document Information:
- Filename: {filename}
- File Extension: {file_extension}
- Column Headers (if tabular): {columns}
- Row Count: {row_count}
- Sample Data Preview: {data_preview}
- Is Tabular Data: {is_tabular}

## Classification Guidelines:
1. Identify ALL document types that apply to this document
2. The PRIMARY type should be the most specific match
3. Include SECONDARY types based on the data content:
   - If columns contain product/SKU/stock/inventory → include "inventory_report"
   - If columns contain amount/price/payment/total → include "financial_report"
   - If columns contain customer/order/sale → include "report"
   - If columns contain shipping/delivery/tracking → include "shipping_manifest"
   - If columns contain invoice/bill → include "invoice"
   - If columns contain receipt/tip/subtotal → include "receipt"
   - If columns contain expense/reimbursement → include "expense_report"
4. Always include "spreadsheet" for CSV/Excel files
5. Return types ordered by relevance (most specific first)

## Response Format (JSON only):
{{
    "document_types": ["primary_type", "secondary_type1", "secondary_type2"],
    "primary_type": "the_most_specific_type",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of why each type applies"
}}

Respond with JSON only:"""


@dataclass
class ClassificationResult:
    """Result of document classification."""
    document_type: str
    type_info: DocumentTypeInfo
    confidence: float
    reasoning: str
    is_extractable: bool
    schema_type: Optional[str]


@dataclass
class MultiTypeClassificationResult:
    """Result of multi-type document classification."""
    document_types: List[str]  # List of all applicable types
    primary_type: str  # The most specific/relevant type
    confidence: float
    reasoning: str
    is_extractable: bool
    schema_type: Optional[str]


class DocumentTypeClassifier:
    """
    Centralized document type classifier using LLM-driven analysis.

    Used by:
    - ExtractionEligibilityChecker: To determine schema type for data extraction
    - DocumentClassifier: To determine chunking strategy for RAG
    """

    def __init__(self, db=None, llm_client=None):
        """
        Initialize the classifier.

        Args:
            db: Optional database session for schema lookup
            llm_client: Optional LLM client for classification
        """
        self.db = db
        self.llm_client = llm_client
        self._db_schemas_cache = None

    def classify(
        self,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
        content_preview: Optional[str] = None
    ) -> ClassificationResult:
        """
        Classify a document using LLM-driven analysis.

        Args:
            filename: Document filename
            metadata: Document metadata dict
            content_preview: Optional content preview (first N chars)

        Returns:
            ClassificationResult with type info
        """
        metadata = metadata or {}

        # Quick check for file extensions
        file_ext = self._get_file_extension(filename)
        if file_ext in ('.xlsx', '.xls', '.csv'):
            return self._create_result("spreadsheet", 0.95, "File extension indicates spreadsheet")

        # Build classification context
        context = self._build_classification_context(filename, metadata, content_preview)

        # Try LLM classification first
        if self.llm_client:
            result = self._classify_with_llm(context)
            if result:
                return result

        # Fallback: Use metadata document_type
        metadata_type = metadata.get('document_type', '').lower().strip()
        if metadata_type:
            resolved_type = self._resolve_type(metadata_type)
            if resolved_type:
                return self._create_result(
                    resolved_type,
                    0.7,
                    f"Resolved from metadata document_type: {metadata_type}"
                )

        # Pattern-based fallback: Check filename and content for keywords
        pattern_result = self._classify_by_patterns(filename, context)
        if pattern_result:
            return pattern_result

        # Last resort: Domain-based default
        domain = metadata.get('domain', '').lower()
        default_type = self._get_domain_default(domain, context)
        return self._create_result(default_type, 0.5, f"Default for domain: {domain or 'unknown'}")

    def classify_multi_type(
        self,
        filename: str,
        columns: List[str] = None,
        row_count: int = 0,
        data_preview: str = None,
        is_tabular: bool = False
    ) -> MultiTypeClassificationResult:
        """
        Classify a document into multiple applicable types using LLM.

        This is especially useful for tabular documents (CSV/Excel) where the
        data content determines additional semantic types beyond just "spreadsheet".

        Args:
            filename: Document filename
            columns: Column headers for tabular data
            row_count: Number of data rows
            data_preview: Sample of data content
            is_tabular: Whether this is tabular data

        Returns:
            MultiTypeClassificationResult with list of applicable types
        """
        columns = columns or []
        data_preview = data_preview or ""

        # Get file extension
        file_ext = self._get_file_extension(filename)

        # Try LLM classification if available
        if self.llm_client:
            try:
                prompt = MULTI_TYPE_CLASSIFICATION_PROMPT.format(
                    available_types=self._get_available_types_description(),
                    filename=filename,
                    file_extension=file_ext,
                    columns=", ".join(columns[:20]) if columns else "N/A",
                    row_count=row_count,
                    data_preview=data_preview[:500] if data_preview else "N/A",
                    is_tabular=is_tabular
                )

                # Use query model for classification (lighter model, faster)
                model = self.llm_client.get_query_model(temperature=0.1, num_predict=512)
                from langchain_core.messages import HumanMessage
                response_msg = model.invoke([HumanMessage(content=prompt)])
                response = response_msg.content if hasattr(response_msg, 'content') else str(response_msg)
                result = self._parse_llm_response(response)

                if result and result.get('document_types'):
                    doc_types = result['document_types']
                    # Validate and resolve each type
                    valid_types = []
                    for t in doc_types:
                        resolved = self._resolve_type(t)
                        if resolved:
                            valid_types.append(resolved)

                    if valid_types:
                        primary = result.get('primary_type', valid_types[0])
                        resolved_primary = self._resolve_type(primary) or valid_types[0]
                        confidence = float(result.get('confidence', 0.8))
                        reasoning = result.get('reasoning', 'LLM multi-type classification')

                        # Check if any type is extractable
                        is_extractable = any(
                            DOCUMENT_TYPES.get(t, DocumentTypeInfo("", "", DocumentDomain.GENERIC, [])).is_extractable
                            for t in valid_types
                        )
                        schema_type = self.get_schema_type(resolved_primary)

                        logger.info(
                            f"[DocTypeClassifier] LLM multi-type: {valid_types} "
                            f"(primary: {resolved_primary}, confidence: {confidence:.2f})"
                        )

                        return MultiTypeClassificationResult(
                            document_types=valid_types,
                            primary_type=resolved_primary,
                            confidence=confidence,
                            reasoning=reasoning,
                            is_extractable=is_extractable,
                            schema_type=schema_type
                        )

            except Exception as e:
                logger.warning(f"[DocTypeClassifier] LLM multi-type classification failed: {e}")

        # Fallback: Determine types from filename and columns
        types = self._infer_types_from_content(filename, columns, is_tabular)
        primary = types[0] if types else "other"

        is_extractable = any(
            DOCUMENT_TYPES.get(t, DocumentTypeInfo("", "", DocumentDomain.GENERIC, [])).is_extractable
            for t in types
        )
        schema_type = self.get_schema_type(primary)

        logger.info(f"[DocTypeClassifier] Fallback multi-type: {types} (primary: {primary})")

        return MultiTypeClassificationResult(
            document_types=types,
            primary_type=primary,
            confidence=0.7,
            reasoning="Inferred from filename and column headers",
            is_extractable=is_extractable,
            schema_type=schema_type
        )

    def _infer_types_from_content(
        self,
        filename: str,
        columns: List[str],
        is_tabular: bool
    ) -> List[str]:
        """Fallback type inference from filename and columns."""
        types = set()
        file_ext = self._get_file_extension(filename)

        # Base type from extension
        if file_ext in ('.csv', '.xlsx', '.xls'):
            types.add("spreadsheet")

        if is_tabular and columns:
            columns_text = " ".join([c.lower() for c in columns])

            # Inventory/Product
            if any(kw in columns_text for kw in ["product", "inventory", "stock", "sku", "quantity", "item", "warehouse"]):
                types.add("inventory_report")

            # Financial
            if any(kw in columns_text for kw in ["amount", "total", "payment", "transaction", "balance", "credit", "debit"]):
                types.add("financial_report")
                if "invoice" in columns_text:
                    types.add("invoice")

            # Sales/Orders
            if any(kw in columns_text for kw in ["sale", "order", "revenue", "customer", "purchase"]):
                types.add("report")
                if "order" in columns_text or "purchase" in columns_text:
                    types.add("purchase_order")

            # Bank
            if any(kw in columns_text for kw in ["account", "bank", "deposit", "withdrawal"]):
                types.add("bank_statement")

            # Receipt
            if any(kw in columns_text for kw in ["receipt", "tip", "subtotal", "tax", "meal"]):
                types.add("receipt")

            # Shipping
            if any(kw in columns_text for kw in ["shipping", "delivery", "tracking", "freight"]):
                types.add("shipping_manifest")

            # Expense
            if any(kw in columns_text for kw in ["expense", "reimbursement", "claim"]):
                types.add("expense_report")

        return list(types) if types else ["spreadsheet" if is_tabular else "other"]

    def get_extractable_types(self) -> List[str]:
        """Get list of document types that support data extraction."""
        return [name for name, info in DOCUMENT_TYPES.items() if info.is_extractable]

    def get_schema_type(self, document_type: str) -> Optional[str]:
        """Get the schema_type for a document type."""
        resolved = self._resolve_type(document_type)
        if resolved and resolved in DOCUMENT_TYPES:
            return DOCUMENT_TYPES[resolved].schema_type
        return None

    def is_extractable(self, document_type: str) -> bool:
        """Check if a document type supports data extraction."""
        resolved = self._resolve_type(document_type)
        if resolved and resolved in DOCUMENT_TYPES:
            return DOCUMENT_TYPES[resolved].is_extractable
        return False

    def get_chunking_config(self, document_type: str) -> Dict[str, Any]:
        """Get chunking configuration for a document type."""
        resolved = self._resolve_type(document_type)
        if resolved and resolved in DOCUMENT_TYPES:
            info = DOCUMENT_TYPES[resolved]
            return {
                "strategy": info.chunking_strategy,
                "chunk_size": info.chunk_size,
                "chunk_overlap": info.chunk_overlap,
                "domain": info.domain.value
            }
        # Default config
        return {
            "strategy": "semantic",
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "domain": "generic"
        }

    def get_type_info(self, document_type: str) -> Optional[DocumentTypeInfo]:
        """Get full type info for a document type."""
        resolved = self._resolve_type(document_type)
        if resolved and resolved in DOCUMENT_TYPES:
            return DOCUMENT_TYPES[resolved]
        return None

    def _resolve_type(self, type_name: str) -> Optional[str]:
        """Resolve a type name through aliases."""
        if not type_name:
            return None

        type_lower = type_name.lower().strip().replace(' ', '_')

        # Direct match
        if type_lower in DOCUMENT_TYPES:
            return type_lower

        # Check simple aliases first (direct mapping)
        if type_lower in SIMPLE_TYPE_ALIASES:
            return SIMPLE_TYPE_ALIASES[type_lower]

        # Check TYPE_ALIASES (returns list, take first match)
        if type_lower in TYPE_ALIASES:
            alias_list = TYPE_ALIASES[type_lower]
            if alias_list and isinstance(alias_list, list) and len(alias_list) > 0:
                # Return first alias that exists in DOCUMENT_TYPES
                for alias in alias_list:
                    if alias in DOCUMENT_TYPES:
                        return alias

        return None

    def _get_file_extension(self, filename: str) -> str:
        """Get lowercase file extension."""
        if not filename:
            return ""
        import os
        return os.path.splitext(filename.lower())[1]

    def _build_classification_context(
        self,
        filename: str,
        metadata: Dict[str, Any],
        content_preview: Optional[str]
    ) -> Dict[str, Any]:
        """Build context dict for classification."""
        return {
            'filename': filename,
            'file_extension': self._get_file_extension(filename),
            'current_type': metadata.get('document_type', 'unknown'),
            'subject_name': metadata.get('subject_name', ''),
            'summary': (metadata.get('summary', '') or '')[:500],
            'domain': metadata.get('domain', 'unknown'),
            'entities': self._extract_entities(metadata),
            'has_tables': metadata.get('has_tables', False),
            'has_amounts': self._check_has_amounts(metadata),
            'content_preview': (content_preview or '')[:1000],
        }

    def _extract_entities(self, metadata: Dict[str, Any]) -> str:
        """Extract entity summary from metadata."""
        entities = []

        for key in ['entities', 'named_entities', 'organizations', 'people', 'locations']:
            if key in metadata and metadata[key]:
                if isinstance(metadata[key], list):
                    entities.extend(str(e) for e in metadata[key][:5])
                elif isinstance(metadata[key], str):
                    entities.append(metadata[key])

        for key in ['vendor_name', 'customer_name', 'store_name', 'company_name', 'merchant']:
            if key in metadata and metadata[key]:
                entities.append(f"{key}: {metadata[key]}")

        return ", ".join(entities[:10]) if entities else "none"

    def _check_has_amounts(self, metadata: Dict[str, Any]) -> bool:
        """Check if metadata indicates monetary amounts."""
        amount_fields = ['total_amount', 'subtotal', 'tax', 'amount', 'price', 'cost', 'total']
        for field in amount_fields:
            if field in metadata and metadata[field]:
                return True

        summary = (metadata.get('summary', '') or '').lower()
        indicators = ['$', '€', '£', 'usd', 'cad', 'eur', 'total:', 'amount:']
        return any(ind in summary for ind in indicators)

    def _get_available_types_description(self) -> str:
        """Get formatted list of available types for LLM."""
        # Use the centralized function to generate type list
        return generate_metadata_extraction_types()

    def _classify_with_llm(self, context: Dict[str, Any]) -> Optional[ClassificationResult]:
        """Use LLM to classify document."""
        try:
            prompt = DOCUMENT_CLASSIFICATION_PROMPT.format(
                available_types=self._get_available_types_description(),
                filename=context['filename'],
                current_type=context['current_type'],
                subject_name=context['subject_name'],
                summary=context['summary'],
                domain=context['domain'],
                entities=context['entities'],
                has_tables=context['has_tables'],
                has_amounts=context['has_amounts'],
                file_extension=context['file_extension']
            )

            response = self.llm_client.generate(prompt)
            result = self._parse_llm_response(response)

            if result and result.get('document_type'):
                doc_type = result['document_type'].lower().strip()
                resolved = self._resolve_type(doc_type)

                if resolved:
                    confidence = float(result.get('confidence', 0.8))
                    reasoning = result.get('reasoning', 'LLM classification')
                    logger.info(
                        f"[DocTypeClassifier] LLM classified as '{resolved}' "
                        f"(confidence: {confidence:.2f}): {reasoning}"
                    )
                    return self._create_result(resolved, confidence, reasoning)
                else:
                    logger.warning(f"[DocTypeClassifier] LLM returned unknown type: {doc_type}")

        except Exception as e:
            logger.warning(f"[DocTypeClassifier] LLM classification failed: {e}")

        return None

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        import re
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def _classify_by_patterns(
        self,
        filename: str,
        context: Dict[str, Any]
    ) -> Optional[ClassificationResult]:
        """
        Classify document by pattern matching on filename and content.

        This is the fallback when LLM is not available.

        IMPORTANT: Filename patterns are checked FIRST with higher confidence,
        since filename is usually the most reliable indicator of document type.

        Args:
            filename: Document filename
            context: Classification context

        Returns:
            ClassificationResult or None if no pattern matches
        """
        filename_lower = filename.lower()
        content_preview = (context.get('content_preview', '') or '').lower()
        summary = (context.get('summary', '') or '').lower()

        # STEP 1: Check FILENAME patterns FIRST with higher confidence
        # Filename is the most reliable indicator since it's explicitly named by the user
        filename_pattern_rules = [
            # Policy documents - check these early since "policy" in filename is definitive
            ("policy", ["policy", "policies", "_policy", "-policy"], 0.90),
            ("privacy_policy", ["privacy_policy", "privacy-policy", "privacypolicy"], 0.90),
            ("terms_of_service", ["terms_of_service", "terms-of-service", "tos", "terms_and_conditions"], 0.90),

            # Financial documents
            ("receipt", ["receipt", "rcpt", "_receipt", "-receipt"], 0.85),
            ("invoice", ["invoice", "inv_", "inv-", "_invoice", "-invoice"], 0.85),
            ("bank_statement", ["bank_statement", "bank-statement", "statement"], 0.85),
            ("expense_report", ["expense_report", "expense-report", "expenses"], 0.85),
            ("purchase_order", ["purchase_order", "purchase-order", "po_", "po-"], 0.85),

            # Inventory - require more specific patterns in filename
            ("inventory_report", ["inventory_report", "inventory-report", "stock_report", "stock-report"], 0.85),

            # Logistics
            ("shipping_manifest", ["shipping", "manifest", "packing_list", "delivery_note", "bill_of_lading"], 0.85),

            # Legal
            ("contract", ["contract", "_contract", "-contract"], 0.85),
            ("patent", ["patent"], 0.85),

            # Academic
            ("research_paper", ["research_paper", "research-paper"], 0.85),
            ("thesis", ["thesis", "dissertation"], 0.85),

            # Medical
            ("medical_record", ["medical_record", "medical-record", "patient_record"], 0.85),
            ("prescription", ["prescription", "rx_"], 0.85),
            ("lab_result", ["lab_result", "lab-result", "blood_test"], 0.85),

            # Technical
            ("technical_spec", ["spec_", "specification", "requirement"], 0.85),
            ("user_manual", ["manual", "user_guide", "user-guide"], 0.85),

            # Professional
            ("resume", ["resume", "cv_", "_cv"], 0.85),
        ]

        for doc_type, patterns, confidence in filename_pattern_rules:
            for pattern in patterns:
                if pattern in filename_lower:
                    return self._create_result(
                        doc_type,
                        confidence,
                        f"Filename pattern match: '{pattern}'"
                    )

        # STEP 2: Check CONTENT patterns with standard confidence
        # Content is a secondary indicator
        content_combined = f"{content_preview} {summary}"

        # Pattern definitions: (doc_type, patterns, confidence)
        # Ordered by specificity - more specific patterns first within each category
        content_pattern_rules = [
            # Legal/Policy - Check BEFORE financial/inventory to catch policy documents
            ("policy", ["return policy", "refund policy", "exchange policy", "company policy",
                       "our policy", "this policy", "policy applies", "policy statement"], 0.75),
            ("privacy_policy", ["privacy policy", "data protection", "personal information", "gdpr"], 0.75),
            ("terms_of_service", ["terms of service", "terms and conditions", "by using this"], 0.75),
            ("contract", ["contract", "agreement", "whereas", "hereby", "party agrees"], 0.70),

            # Financial - Extractable (specific patterns)
            ("receipt", ["receipt", "rcpt", "meal-", "transaction receipt", "purchase receipt",
                        "your receipt", "receipt number"], 0.75),
            ("invoice", ["invoice", "inv-", "bill to", "amount due", "invoice number",
                        "invoice date", "payment terms"], 0.75),
            ("bank_statement", ["bank statement", "account balance", "statement of account",
                               "opening balance", "closing balance", "account summary"], 0.75),
            ("expense_report", ["expense report", "expense claim", "reimbursement request",
                               "employee expenses"], 0.75),
            ("purchase_order", ["purchase order", "p.o. number", "po number", "order confirmation"], 0.75),

            # Logistics - Extractable
            ("shipping_manifest", ["shipping manifest", "packing list", "delivery note",
                                  "bill of lading", "shipment details", "tracking number"], 0.75),

            # Inventory - Use MORE SPECIFIC patterns to avoid false positives
            # "inventory" alone is too generic and matches policy docs that mention inventory
            ("inventory_report", ["inventory report", "inventory list", "stock report",
                                 "warehouse inventory", "inventory count", "stock level",
                                 "inventory management", "stock on hand", "quantity in stock"], 0.75),

            # Academic
            ("research_paper", ["research paper", "abstract", "methodology", "literature review",
                               "findings", "conclusion", "references"], 0.70),
            ("thesis", ["thesis", "dissertation", "submitted in partial fulfillment"], 0.70),
            ("academic_article", ["et al.", "et al,", "[1]", "[2]", "doi:"], 0.65),

            # Medical
            ("medical_record", ["chief complaint", "history of present illness", "assessment and plan",
                               "physical examination", "hpi", "ros"], 0.70),
            ("prescription", ["prescription", "rx", "take as directed", "refills"], 0.70),
            ("lab_result", ["lab result", "laboratory report", "blood test", "reference range"], 0.70),
            ("clinical_report", ["clinical report", "diagnosis", "treatment plan"], 0.65),

            # Education
            ("textbook", ["textbook", "chapter exercises", "learning objectives"], 0.70),
            ("course_material", ["chapter", "learning objective", "key concepts"], 0.65),
            ("syllabus", ["syllabus", "course schedule", "grading policy"], 0.70),
            ("exam", ["exam", "quiz", "test questions", "answer the following"], 0.65),

            # Technical/Engineering
            ("technical_spec", ["requirement", "req-", "spec-", "specification",
                               "functional requirement", "technical requirement"], 0.70),
            ("api_documentation", ["api", "endpoint", "request body", "response format"], 0.65),
            ("user_manual", ["user manual", "user guide", "instructions", "how to use"], 0.70),

            # Professional
            ("resume", ["resume", "curriculum vitae", "work experience", "education background"], 0.70),
            ("cover_letter", ["cover letter", "dear hiring manager", "position applied"], 0.70),
        ]

        for doc_type, patterns, confidence in content_pattern_rules:
            for pattern in patterns:
                if pattern in content_combined:
                    return self._create_result(
                        doc_type,
                        confidence,
                        f"Content pattern match: '{pattern}'"
                    )

        return None

    def _get_domain_default(self, domain: str, context: Dict[str, Any]) -> str:
        """Get default type for a domain."""
        domain_defaults = {
            'financial': 'receipt' if context.get('has_amounts') else 'report',
            'legal': 'contract',
            'medical': 'medical_record',
            'academic': 'research_paper',
            'education': 'course_material',
            'engineering': 'technical_spec',
            'logistics': 'shipping_manifest',
            'inventory': 'inventory_report',
            'professional': 'resume',
        }
        return domain_defaults.get(domain, 'other')

    def _create_result(
        self,
        document_type: str,
        confidence: float,
        reasoning: str
    ) -> ClassificationResult:
        """Create a classification result."""
        type_info = DOCUMENT_TYPES.get(document_type, DOCUMENT_TYPES['other'])

        return ClassificationResult(
            document_type=document_type,
            type_info=type_info,
            confidence=confidence,
            reasoning=reasoning,
            is_extractable=type_info.is_extractable,
            schema_type=type_info.schema_type
        )


# ============================================================================
# TABULAR DATA DETECTION
# ============================================================================

# File extensions that indicate tabular/dataset-style data
TABULAR_FILE_EXTENSIONS = {
    '.csv', '.tsv', '.xlsx', '.xls', '.xlsm', '.xlsb', '.ods'
}

# Document types that use the optimized tabular data pathway
# These types have structured rows and benefit from SQL-based analytics
TABULAR_DOCUMENT_TYPES = {
    'spreadsheet', 'invoice', 'receipt', 'bank_statement',
    'credit_card_statement', 'expense_report', 'inventory_report',
    'purchase_order', 'sales_order', 'payroll', 'financial_statement',
    'price_list', 'data_export', 'transaction_log', 'shipping_manifest'
}


class TabularDataDetector:
    """
    Detector for tabular/dataset-style documents.

    Used to determine if a document should use the optimized tabular workflow:
    - Skip row-level chunking
    - Generate summary/metadata embeddings only (1-3 chunks)
    - Route queries through SQL-based analytics
    """

    @staticmethod
    def is_tabular_data(
        filename: str = None,
        document_type: str = None,
        content: str = None
    ) -> Tuple[bool, str]:
        """
        Determine if document should use the tabular data pathway.

        Args:
            filename: Document filename
            document_type: Detected document type
            content: Optional document content for analysis

        Returns:
            Tuple of (is_tabular, reason)
            - is_tabular: True if document should use tabular pathway
            - reason: Explanation for the decision
        """
        # Check 1: File extension
        if filename:
            ext = os.path.splitext(filename.lower())[1]
            if ext in TABULAR_FILE_EXTENSIONS:
                return True, f"file_extension:{ext}"

        # Check 2: Document type
        if document_type:
            doc_type_lower = document_type.lower().strip()
            # Check direct match
            if doc_type_lower in TABULAR_DOCUMENT_TYPES:
                return True, f"document_type:{doc_type_lower}"
            # Check aliases (TYPE_ALIASES returns a list, so we need to handle it properly)
            resolved = TYPE_ALIASES.get(doc_type_lower)
            if resolved:
                # TYPE_ALIASES returns a list of aliases
                if isinstance(resolved, list):
                    for alias in resolved:
                        if alias in TABULAR_DOCUMENT_TYPES:
                            return True, f"document_type:{alias}"
                elif isinstance(resolved, str) and resolved in TABULAR_DOCUMENT_TYPES:
                    return True, f"document_type:{resolved}"

        # Check 3: Content analysis (if provided)
        if content:
            table_density = TabularDataDetector._calculate_table_density(content)
            if table_density > 0.6:
                return True, f"table_density:{table_density:.2f}"

            numeric_density = TabularDataDetector._calculate_numeric_density(content)
            if numeric_density > 0.4:
                return True, f"numeric_density:{numeric_density:.2f}"

        return False, "not_tabular"

    @staticmethod
    def _calculate_table_density(content: str) -> float:
        """Calculate ratio of content that is markdown/HTML tables."""
        if not content:
            return 0.0

        lines = content.split('\n')
        if not lines:
            return 0.0

        # Count lines that look like table rows
        table_lines = 0
        for line in lines:
            line = line.strip()
            # Markdown table: | col1 | col2 |
            if '|' in line and line.count('|') >= 2:
                table_lines += 1
            # CSV-like: value, value, value
            elif ',' in line and line.count(',') >= 2:
                parts = line.split(',')
                if len(parts) >= 3:
                    table_lines += 1

        return table_lines / max(len(lines), 1)

    @staticmethod
    def _calculate_numeric_density(content: str) -> float:
        """Calculate ratio of numeric characters in content."""
        if not content:
            return 0.0

        import re
        numeric_chars = len(re.findall(r'[\d.,]', content))
        return numeric_chars / max(len(content), 1)

    @staticmethod
    def get_processing_path(
        filename: str = None,
        document_type: str = None,
        content: str = None
    ) -> str:
        """
        Determine the processing path for a document.

        Returns:
            "tabular" - Use optimized tabular workflow (summary-only indexing)
            "standard" - Use full chunking workflow
            "hybrid" - Document has both narrative and tabular content
        """
        is_tabular, reason = TabularDataDetector.is_tabular_data(
            filename=filename,
            document_type=document_type,
            content=content
        )

        if is_tabular:
            return "tabular"

        # Future: Could detect hybrid documents here
        # For now, everything else is standard
        return "standard"


# Convenience functions for backward compatibility
def get_document_type_info(document_type: str) -> Optional[DocumentTypeInfo]:
    """Get info for a document type."""
    classifier = DocumentTypeClassifier()
    return classifier.get_type_info(document_type)


def is_extractable_type(document_type: str) -> bool:
    """Check if a document type is extractable."""
    classifier = DocumentTypeClassifier()
    return classifier.is_extractable(document_type)


def get_schema_for_type(document_type: str) -> Optional[str]:
    """Get schema_type for a document type."""
    classifier = DocumentTypeClassifier()
    return classifier.get_schema_type(document_type)


def get_chunking_config_for_type(document_type: str) -> Dict[str, Any]:
    """Get chunking config for a document type."""
    classifier = DocumentTypeClassifier()
    return classifier.get_chunking_config(document_type)
