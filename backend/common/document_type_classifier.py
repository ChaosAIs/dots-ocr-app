"""
Centralized Document Type Classifier

Provides unified LLM-driven document type classification for:
- Extraction eligibility (schema type for data extraction)
- Chunking strategy (document structure for RAG)

Document types are consistent with data_schemas table and can be extended.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentDomain(str, Enum):
    """Document domain categories."""
    FINANCIAL = "financial"
    LEGAL = "legal"
    MEDICAL = "medical"
    ACADEMIC = "academic"
    EDUCATION = "education"
    ENGINEERING = "engineering"
    GOVERNMENT = "government"
    PROFESSIONAL = "professional"
    LOGISTICS = "logistics"
    INVENTORY = "inventory"
    GENERIC = "generic"


@dataclass
class DocumentTypeInfo:
    """Information about a document type."""
    type_name: str
    display_name: str
    domain: DocumentDomain
    description: str
    is_extractable: bool  # Has structured data to extract
    schema_type: Optional[str]  # Maps to data_schemas.schema_type
    chunking_strategy: str  # semantic, hierarchical, fixed, hybrid
    chunk_size: int
    chunk_overlap: int


# Unified document type definitions
# Combines types from extraction (data_schemas) and chunking systems
DOCUMENT_TYPES: Dict[str, DocumentTypeInfo] = {
    # ==========================================================================
    # EXTRACTABLE TYPES (have structured data, map to data_schemas)
    # ==========================================================================

    # Financial - Extractable
    "invoice": DocumentTypeInfo(
        type_name="invoice",
        display_name="Invoice",
        domain=DocumentDomain.FINANCIAL,
        description="Vendor invoices, billing documents, purchase invoices with line items",
        is_extractable=True,
        schema_type="invoice",
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "receipt": DocumentTypeInfo(
        type_name="receipt",
        display_name="Receipt",
        domain=DocumentDomain.FINANCIAL,
        description="Retail receipts, restaurant bills, transaction receipts with itemized purchases",
        is_extractable=True,
        schema_type="receipt",
        chunking_strategy="semantic",
        chunk_size=800,
        chunk_overlap=80
    ),
    "bank_statement": DocumentTypeInfo(
        type_name="bank_statement",
        display_name="Bank Statement",
        domain=DocumentDomain.FINANCIAL,
        description="Bank account statements with transaction history",
        is_extractable=True,
        schema_type="bank_statement",
        chunking_strategy="semantic",
        chunk_size=1200,
        chunk_overlap=120
    ),
    "expense_report": DocumentTypeInfo(
        type_name="expense_report",
        display_name="Expense Report",
        domain=DocumentDomain.FINANCIAL,
        description="Employee expense claims and reimbursement requests",
        is_extractable=True,
        schema_type="expense_report",
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "purchase_order": DocumentTypeInfo(
        type_name="purchase_order",
        display_name="Purchase Order",
        domain=DocumentDomain.FINANCIAL,
        description="Purchase orders and procurement documents",
        is_extractable=True,
        schema_type="purchase_order",
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),

    # Logistics - Extractable
    "shipping_manifest": DocumentTypeInfo(
        type_name="shipping_manifest",
        display_name="Shipping Manifest",
        domain=DocumentDomain.LOGISTICS,
        description="Shipping documents, delivery notes, packing lists, bills of lading",
        is_extractable=True,
        schema_type="shipping_manifest",
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),

    # Inventory - Extractable
    "inventory_report": DocumentTypeInfo(
        type_name="inventory_report",
        display_name="Inventory Report",
        domain=DocumentDomain.INVENTORY,
        description="Inventory reports, stock level documents, warehouse reports",
        is_extractable=True,
        schema_type="inventory_report",
        chunking_strategy="semantic",
        chunk_size=1200,
        chunk_overlap=120
    ),

    # Generic - Extractable
    "spreadsheet": DocumentTypeInfo(
        type_name="spreadsheet",
        display_name="Spreadsheet",
        domain=DocumentDomain.GENERIC,
        description="Excel files, CSV files, tabular data with dynamic structure",
        is_extractable=True,
        schema_type="spreadsheet",
        chunking_strategy="fixed",
        chunk_size=1500,
        chunk_overlap=150
    ),

    # ==========================================================================
    # NON-EXTRACTABLE TYPES (no structured line items, for RAG only)
    # ==========================================================================

    # Legal
    "contract": DocumentTypeInfo(
        type_name="contract",
        display_name="Contract",
        domain=DocumentDomain.LEGAL,
        description="Legal contracts and agreements",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "terms_of_service": DocumentTypeInfo(
        type_name="terms_of_service",
        display_name="Terms of Service",
        domain=DocumentDomain.LEGAL,
        description="Terms of service and conditions documents",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "privacy_policy": DocumentTypeInfo(
        type_name="privacy_policy",
        display_name="Privacy Policy",
        domain=DocumentDomain.LEGAL,
        description="Privacy policy documents",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "patent": DocumentTypeInfo(
        type_name="patent",
        display_name="Patent",
        domain=DocumentDomain.LEGAL,
        description="Patent documents and filings",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="hierarchical",
        chunk_size=2000,
        chunk_overlap=300
    ),

    # Medical
    "medical_record": DocumentTypeInfo(
        type_name="medical_record",
        display_name="Medical Record",
        domain=DocumentDomain.MEDICAL,
        description="Patient medical records, clinical notes",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="semantic",
        chunk_size=1200,
        chunk_overlap=150
    ),
    "prescription": DocumentTypeInfo(
        type_name="prescription",
        display_name="Prescription",
        domain=DocumentDomain.MEDICAL,
        description="Medical prescriptions",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="semantic",
        chunk_size=500,
        chunk_overlap=50
    ),
    "lab_result": DocumentTypeInfo(
        type_name="lab_result",
        display_name="Lab Result",
        domain=DocumentDomain.MEDICAL,
        description="Laboratory test results",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="semantic",
        chunk_size=800,
        chunk_overlap=80
    ),
    "clinical_report": DocumentTypeInfo(
        type_name="clinical_report",
        display_name="Clinical Report",
        domain=DocumentDomain.MEDICAL,
        description="Clinical reports and diagnostic documents",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="semantic",
        chunk_size=1200,
        chunk_overlap=150
    ),

    # Academic
    "research_paper": DocumentTypeInfo(
        type_name="research_paper",
        display_name="Research Paper",
        domain=DocumentDomain.ACADEMIC,
        description="Academic research papers and journal articles",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "thesis": DocumentTypeInfo(
        type_name="thesis",
        display_name="Thesis/Dissertation",
        domain=DocumentDomain.ACADEMIC,
        description="Academic theses and dissertations",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="hierarchical",
        chunk_size=2000,
        chunk_overlap=300
    ),
    "academic_article": DocumentTypeInfo(
        type_name="academic_article",
        display_name="Academic Article",
        domain=DocumentDomain.ACADEMIC,
        description="Academic articles with citations",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),

    # Education
    "textbook": DocumentTypeInfo(
        type_name="textbook",
        display_name="Textbook",
        domain=DocumentDomain.EDUCATION,
        description="Educational textbooks",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "course_material": DocumentTypeInfo(
        type_name="course_material",
        display_name="Course Material",
        domain=DocumentDomain.EDUCATION,
        description="Course materials, lecture notes, learning content",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="hierarchical",
        chunk_size=1200,
        chunk_overlap=150
    ),
    "syllabus": DocumentTypeInfo(
        type_name="syllabus",
        display_name="Syllabus",
        domain=DocumentDomain.EDUCATION,
        description="Course syllabi",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "exam": DocumentTypeInfo(
        type_name="exam",
        display_name="Exam/Quiz",
        domain=DocumentDomain.EDUCATION,
        description="Exams, quizzes, and assessments",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="semantic",
        chunk_size=800,
        chunk_overlap=80
    ),

    # Engineering/Technical
    "technical_spec": DocumentTypeInfo(
        type_name="technical_spec",
        display_name="Technical Specification",
        domain=DocumentDomain.ENGINEERING,
        description="Technical specifications and requirements documents",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "api_documentation": DocumentTypeInfo(
        type_name="api_documentation",
        display_name="API Documentation",
        domain=DocumentDomain.ENGINEERING,
        description="API documentation and reference guides",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="hierarchical",
        chunk_size=1200,
        chunk_overlap=150
    ),
    "user_manual": DocumentTypeInfo(
        type_name="user_manual",
        display_name="User Manual",
        domain=DocumentDomain.ENGINEERING,
        description="User manuals and guides",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="hierarchical",
        chunk_size=1200,
        chunk_overlap=150
    ),

    # Professional
    "resume": DocumentTypeInfo(
        type_name="resume",
        display_name="Resume/CV",
        domain=DocumentDomain.PROFESSIONAL,
        description="Resumes and curriculum vitae",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="semantic",
        chunk_size=800,
        chunk_overlap=80
    ),
    "cover_letter": DocumentTypeInfo(
        type_name="cover_letter",
        display_name="Cover Letter",
        domain=DocumentDomain.PROFESSIONAL,
        description="Job application cover letters",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="semantic",
        chunk_size=600,
        chunk_overlap=60
    ),

    # Generic/Fallback
    "report": DocumentTypeInfo(
        type_name="report",
        display_name="Report",
        domain=DocumentDomain.GENERIC,
        description="General reports and documents",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="semantic",
        chunk_size=1200,
        chunk_overlap=150
    ),
    "other": DocumentTypeInfo(
        type_name="other",
        display_name="Other Document",
        domain=DocumentDomain.GENERIC,
        description="Unclassified documents",
        is_extractable=False,
        schema_type=None,
        chunking_strategy="fixed",
        chunk_size=1000,
        chunk_overlap=100
    ),
}

# Aliases for common variations
TYPE_ALIASES: Dict[str, str] = {
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

    # Legal aliases
    "agreement": "contract",
    "terms_and_conditions": "terms_of_service",

    # Academic aliases
    "dissertation": "thesis",
    "paper": "research_paper",

    # Resume aliases
    "cv": "resume",
    "curriculum_vitae": "resume",
}


# LLM Classification Prompt
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
7. If the document doesn't clearly match any type, use "other"

## Response Format (JSON only):
{{
    "document_type": "the_type_name",
    "confidence": 0.0-1.0,
    "domain": "the_domain",
    "reasoning": "Brief explanation"
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

        # Check aliases
        if type_lower in TYPE_ALIASES:
            return TYPE_ALIASES[type_lower]

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
        lines = []

        # Group by domain
        by_domain: Dict[str, List[DocumentTypeInfo]] = {}
        for info in DOCUMENT_TYPES.values():
            domain = info.domain.value
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(info)

        for domain, types in sorted(by_domain.items()):
            lines.append(f"\n### {domain.upper()} Domain:")
            for info in sorted(types, key=lambda x: x.type_name):
                extractable = " [EXTRACTABLE]" if info.is_extractable else ""
                lines.append(f"- {info.type_name}: {info.description}{extractable}")

        return "\n".join(lines)

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

        Args:
            filename: Document filename
            context: Classification context

        Returns:
            ClassificationResult or None if no pattern matches
        """
        filename_lower = filename.lower()
        content_preview = (context.get('content_preview', '') or '').lower()
        summary = (context.get('summary', '') or '').lower()

        # Combined text for pattern matching
        combined = f"{filename_lower} {content_preview} {summary}"

        # Pattern definitions: (doc_type, patterns, confidence)
        # Ordered by specificity (more specific first)
        pattern_rules = [
            # Financial - Extractable
            ("receipt", ["receipt", "rcpt", "meal-", "transaction", "purchase receipt"], 0.75),
            ("invoice", ["invoice", "inv-", "bill to", "amount due", "billing"], 0.75),
            ("bank_statement", ["bank statement", "bank_statement", "account balance", "statement of account"], 0.75),
            ("expense_report", ["expense report", "expense_report", "expense claim", "reimbursement"], 0.75),
            ("purchase_order", ["purchase order", "p.o.", "po-"], 0.75),

            # Logistics - Extractable
            ("shipping_manifest", ["shipping", "packing list", "delivery note", "bill of lading"], 0.75),

            # Inventory - Extractable
            ("inventory_report", ["inventory", "stock report", "warehouse"], 0.75),

            # Legal
            ("contract", ["contract", "agreement", "whereas", "hereby"], 0.7),
            ("terms_of_service", ["terms of service", "terms and conditions"], 0.7),
            ("privacy_policy", ["privacy policy"], 0.7),
            ("patent", ["patent"], 0.7),

            # Academic
            ("research_paper", ["research paper", "abstract", "methodology", "references"], 0.7),
            ("thesis", ["thesis", "dissertation"], 0.7),
            ("academic_article", ["et al.", "et al,", "[1]", "[2]"], 0.65),

            # Medical
            ("medical_record", ["chief complaint", "history of present", "assessment", "plan", "hpi"], 0.7),
            ("prescription", ["prescription", "rx"], 0.7),
            ("lab_result", ["lab result", "laboratory", "blood test"], 0.7),
            ("clinical_report", ["clinical", "diagnosis"], 0.65),

            # Education
            ("textbook", ["textbook"], 0.7),
            ("course_material", ["chapter", "learning objective"], 0.65),
            ("syllabus", ["syllabus"], 0.7),
            ("exam", ["exam", "quiz"], 0.65),

            # Technical/Engineering
            ("technical_spec", ["requirement", "req-", "spec-", "specification"], 0.7),
            ("api_documentation", ["api", "endpoint"], 0.65),
            ("user_manual", ["user manual", "user_manual", "user guide", "instructions"], 0.7),

            # Professional
            ("resume", ["resume", "curriculum vitae", "cv"], 0.7),
            ("cover_letter", ["cover letter", "dear hiring"], 0.7),
        ]

        for doc_type, patterns, confidence in pattern_rules:
            for pattern in patterns:
                if pattern in combined:
                    return self._create_result(
                        doc_type,
                        confidence,
                        f"Pattern match: '{pattern}' in filename/content"
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
            # Check aliases
            resolved = TYPE_ALIASES.get(doc_type_lower)
            if resolved and resolved in TABULAR_DOCUMENT_TYPES:
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
