"""
Domain-specific structure patterns for intelligent document chunking.

This module defines structural patterns and rules for different document domains
including legal, academic, medical, engineering, education, financial, and government.

DEPRECATION NOTICE (V3.0):
--------------------------
This module contains fragile regex patterns that can cause misclassifications.
For example, "rx" in "arXiv" matches prescription patterns, causing academic papers
to be classified as medical prescriptions.

The new LLM-driven chunking system (V3.0) replaces pattern-based classification
with a single LLM call per uploaded file for structure analysis.

To enable the new approach:
    export LLM_DRIVEN_CHUNKING_ENABLED=true

Or programmatically:
    chunker = AdaptiveChunker(use_llm_driven_chunking=True)

When V3.0 is enabled, this module's pattern-matching functions are bypassed.
The DocumentDomain and DocumentType enums are still used for compatibility.

This module will be removed in a future version once V3.0 is fully adopted.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Optional, Set
from enum import Enum


class DocumentDomain(Enum):
    """Document domain categories."""
    LEGAL = "legal"
    ACADEMIC = "academic"
    MEDICAL = "medical"
    ENGINEERING = "engineering"
    EDUCATION = "education"
    FINANCIAL = "financial"
    GOVERNMENT = "government"
    PROFESSIONAL = "professional"
    GENERAL = "general"


class DocumentType(Enum):
    """Comprehensive document type taxonomy."""
    # Business & Finance
    RECEIPT = "receipt"
    INVOICE = "invoice"
    FINANCIAL_REPORT = "financial_report"
    BANK_STATEMENT = "bank_statement"
    TAX_DOCUMENT = "tax_document"
    EXPENSE_REPORT = "expense_report"
    PURCHASE_ORDER = "purchase_order"
    QUOTATION = "quotation"

    # Legal & Compliance
    CONTRACT = "contract"
    AGREEMENT = "agreement"
    LEGAL_BRIEF = "legal_brief"
    COURT_FILING = "court_filing"
    PATENT = "patent"
    REGULATORY_FILING = "regulatory_filing"
    TERMS_OF_SERVICE = "terms_of_service"
    PRIVACY_POLICY = "privacy_policy"
    COMPLIANCE_DOC = "compliance_doc"

    # Technical & Engineering
    TECHNICAL_SPEC = "technical_spec"
    API_DOCUMENTATION = "api_documentation"
    USER_MANUAL = "user_manual"
    INSTALLATION_GUIDE = "installation_guide"
    ARCHITECTURE_DOC = "architecture_doc"
    DATASHEET = "datasheet"
    SCHEMATIC = "schematic"
    CODE_DOCUMENTATION = "code_documentation"

    # Academic & Research
    RESEARCH_PAPER = "research_paper"
    THESIS = "thesis"
    ACADEMIC_ARTICLE = "academic_article"
    LITERATURE_REVIEW = "literature_review"
    CASE_STUDY = "case_study"
    LAB_REPORT = "lab_report"
    CONFERENCE_PAPER = "conference_paper"
    GRANT_PROPOSAL = "grant_proposal"

    # Medical & Healthcare
    MEDICAL_RECORD = "medical_record"
    CLINICAL_REPORT = "clinical_report"
    PRESCRIPTION = "prescription"
    LAB_RESULT = "lab_result"
    INSURANCE_CLAIM = "insurance_claim"
    PATIENT_SUMMARY = "patient_summary"
    DISCHARGE_SUMMARY = "discharge_summary"

    # Education & Training
    TEXTBOOK = "textbook"
    COURSE_MATERIAL = "course_material"
    SYLLABUS = "syllabus"
    LECTURE_NOTES = "lecture_notes"
    EXAM = "exam"
    TUTORIAL = "tutorial"
    CERTIFICATION_MATERIAL = "certification_material"

    # Government & Public
    GOVERNMENT_FORM = "government_form"
    PERMIT = "permit"
    LICENSE = "license"
    OFFICIAL_NOTICE = "official_notice"
    POLICY_DOCUMENT = "policy_document"
    LEGISLATION = "legislation"

    # Professional & HR
    RESUME = "resume"
    COVER_LETTER = "cover_letter"
    JOB_DESCRIPTION = "job_description"
    PERFORMANCE_REVIEW = "performance_review"
    EMPLOYEE_HANDBOOK = "employee_handbook"
    TRAINING_MATERIAL = "training_material"

    # Creative & Media
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    NEWS_ARTICLE = "news_article"
    BOOK_CHAPTER = "book_chapter"
    MANUSCRIPT = "manuscript"
    SCRIPT = "script"

    # General
    REPORT = "report"
    MEMO = "memo"
    LETTER = "letter"
    EMAIL = "email"
    MEETING_NOTES = "meeting_notes"
    PRESENTATION = "presentation"
    OTHER = "other"


# Mapping from document types to domains
DOCUMENT_TYPE_TO_DOMAIN: Dict[DocumentType, DocumentDomain] = {
    # Financial
    DocumentType.RECEIPT: DocumentDomain.FINANCIAL,
    DocumentType.INVOICE: DocumentDomain.FINANCIAL,
    DocumentType.FINANCIAL_REPORT: DocumentDomain.FINANCIAL,
    DocumentType.BANK_STATEMENT: DocumentDomain.FINANCIAL,
    DocumentType.TAX_DOCUMENT: DocumentDomain.FINANCIAL,
    DocumentType.EXPENSE_REPORT: DocumentDomain.FINANCIAL,
    DocumentType.PURCHASE_ORDER: DocumentDomain.FINANCIAL,
    DocumentType.QUOTATION: DocumentDomain.FINANCIAL,

    # Legal
    DocumentType.CONTRACT: DocumentDomain.LEGAL,
    DocumentType.AGREEMENT: DocumentDomain.LEGAL,
    DocumentType.LEGAL_BRIEF: DocumentDomain.LEGAL,
    DocumentType.COURT_FILING: DocumentDomain.LEGAL,
    DocumentType.PATENT: DocumentDomain.LEGAL,
    DocumentType.REGULATORY_FILING: DocumentDomain.LEGAL,
    DocumentType.TERMS_OF_SERVICE: DocumentDomain.LEGAL,
    DocumentType.PRIVACY_POLICY: DocumentDomain.LEGAL,
    DocumentType.COMPLIANCE_DOC: DocumentDomain.LEGAL,

    # Engineering
    DocumentType.TECHNICAL_SPEC: DocumentDomain.ENGINEERING,
    DocumentType.API_DOCUMENTATION: DocumentDomain.ENGINEERING,
    DocumentType.USER_MANUAL: DocumentDomain.ENGINEERING,
    DocumentType.INSTALLATION_GUIDE: DocumentDomain.ENGINEERING,
    DocumentType.ARCHITECTURE_DOC: DocumentDomain.ENGINEERING,
    DocumentType.DATASHEET: DocumentDomain.ENGINEERING,
    DocumentType.SCHEMATIC: DocumentDomain.ENGINEERING,
    DocumentType.CODE_DOCUMENTATION: DocumentDomain.ENGINEERING,

    # Academic
    DocumentType.RESEARCH_PAPER: DocumentDomain.ACADEMIC,
    DocumentType.THESIS: DocumentDomain.ACADEMIC,
    DocumentType.ACADEMIC_ARTICLE: DocumentDomain.ACADEMIC,
    DocumentType.LITERATURE_REVIEW: DocumentDomain.ACADEMIC,
    DocumentType.CASE_STUDY: DocumentDomain.ACADEMIC,
    DocumentType.LAB_REPORT: DocumentDomain.ACADEMIC,
    DocumentType.CONFERENCE_PAPER: DocumentDomain.ACADEMIC,
    DocumentType.GRANT_PROPOSAL: DocumentDomain.ACADEMIC,

    # Medical
    DocumentType.MEDICAL_RECORD: DocumentDomain.MEDICAL,
    DocumentType.CLINICAL_REPORT: DocumentDomain.MEDICAL,
    DocumentType.PRESCRIPTION: DocumentDomain.MEDICAL,
    DocumentType.LAB_RESULT: DocumentDomain.MEDICAL,
    DocumentType.INSURANCE_CLAIM: DocumentDomain.MEDICAL,
    DocumentType.PATIENT_SUMMARY: DocumentDomain.MEDICAL,
    DocumentType.DISCHARGE_SUMMARY: DocumentDomain.MEDICAL,

    # Education
    DocumentType.TEXTBOOK: DocumentDomain.EDUCATION,
    DocumentType.COURSE_MATERIAL: DocumentDomain.EDUCATION,
    DocumentType.SYLLABUS: DocumentDomain.EDUCATION,
    DocumentType.LECTURE_NOTES: DocumentDomain.EDUCATION,
    DocumentType.EXAM: DocumentDomain.EDUCATION,
    DocumentType.TUTORIAL: DocumentDomain.EDUCATION,
    DocumentType.CERTIFICATION_MATERIAL: DocumentDomain.EDUCATION,

    # Government
    DocumentType.GOVERNMENT_FORM: DocumentDomain.GOVERNMENT,
    DocumentType.PERMIT: DocumentDomain.GOVERNMENT,
    DocumentType.LICENSE: DocumentDomain.GOVERNMENT,
    DocumentType.OFFICIAL_NOTICE: DocumentDomain.GOVERNMENT,
    DocumentType.POLICY_DOCUMENT: DocumentDomain.GOVERNMENT,
    DocumentType.LEGISLATION: DocumentDomain.GOVERNMENT,

    # Professional
    DocumentType.RESUME: DocumentDomain.PROFESSIONAL,
    DocumentType.COVER_LETTER: DocumentDomain.PROFESSIONAL,
    DocumentType.JOB_DESCRIPTION: DocumentDomain.PROFESSIONAL,
    DocumentType.PERFORMANCE_REVIEW: DocumentDomain.PROFESSIONAL,
    DocumentType.EMPLOYEE_HANDBOOK: DocumentDomain.PROFESSIONAL,
    DocumentType.TRAINING_MATERIAL: DocumentDomain.PROFESSIONAL,

    # General
    DocumentType.ARTICLE: DocumentDomain.GENERAL,
    DocumentType.BLOG_POST: DocumentDomain.GENERAL,
    DocumentType.NEWS_ARTICLE: DocumentDomain.GENERAL,
    DocumentType.BOOK_CHAPTER: DocumentDomain.GENERAL,
    DocumentType.MANUSCRIPT: DocumentDomain.GENERAL,
    DocumentType.SCRIPT: DocumentDomain.GENERAL,
    DocumentType.REPORT: DocumentDomain.GENERAL,
    DocumentType.MEMO: DocumentDomain.GENERAL,
    DocumentType.LETTER: DocumentDomain.GENERAL,
    DocumentType.EMAIL: DocumentDomain.GENERAL,
    DocumentType.MEETING_NOTES: DocumentDomain.GENERAL,
    DocumentType.PRESENTATION: DocumentDomain.GENERAL,
    DocumentType.OTHER: DocumentDomain.GENERAL,
}


@dataclass
class DomainPattern:
    """Pattern definition for domain-specific structure detection."""
    name: str
    pattern: Pattern
    description: str
    priority: int = 5  # 1-10, higher = more important


@dataclass
class DomainConfig:
    """Configuration for domain-specific chunking behavior."""
    domain: DocumentDomain

    # Patterns for structure detection
    patterns: Dict[str, DomainPattern] = field(default_factory=dict)

    # Elements that should never be split
    never_split: Set[str] = field(default_factory=set)

    # Elements that should be preserved as atomic units
    preserve_elements: Set[str] = field(default_factory=set)

    # Default chunking parameters
    default_chunk_size: int = 512
    default_overlap_percent: int = 10

    # Section headers specific to this domain
    section_headers: List[str] = field(default_factory=list)

    # Recommended strategy for this domain
    recommended_strategy: str = "semantic_header"


# ============================================================================
# LEGAL DOMAIN PATTERNS
# ============================================================================

LEGAL_PATTERNS = {
    "clause_number": DomainPattern(
        name="clause_number",
        pattern=re.compile(r"^\s*(\d+\.)+\d*\s+", re.MULTILINE),
        description="Numbered clauses like 1.1, 1.1.1",
        priority=9
    ),
    "article": DomainPattern(
        name="article",
        pattern=re.compile(r"^Article\s+[IVXLCDM\d]+[:\.\s]", re.MULTILINE | re.IGNORECASE),
        description="Article headers",
        priority=10
    ),
    "section": DomainPattern(
        name="section",
        pattern=re.compile(r"^Section\s+\d+[:\.\s]", re.MULTILINE | re.IGNORECASE),
        description="Section headers",
        priority=9
    ),
    "whereas": DomainPattern(
        name="whereas",
        pattern=re.compile(r"^WHEREAS[,:\s]", re.MULTILINE),
        description="Recital clauses",
        priority=8
    ),
    "definition": DomainPattern(
        name="definition",
        pattern=re.compile(r'"[^"]+"\s+(?:means|shall mean|refers to)', re.IGNORECASE),
        description="Defined terms",
        priority=7
    ),
    "signature_block": DomainPattern(
        name="signature_block",
        pattern=re.compile(r"^(?:Signed|Executed|By|Witness)[:\s]*$", re.MULTILINE | re.IGNORECASE),
        description="Signature blocks",
        priority=8
    ),
    "party_reference": DomainPattern(
        name="party_reference",
        pattern=re.compile(r"\b(?:Party|Parties|Licensor|Licensee|Vendor|Client|Contractor)\b", re.IGNORECASE),
        description="Contract party references",
        priority=6
    ),
    "legal_reference": DomainPattern(
        name="legal_reference",
        pattern=re.compile(r"(?:pursuant to|in accordance with|subject to|notwithstanding)\s+(?:Section|Article|Clause)", re.IGNORECASE),
        description="Cross-references to other clauses",
        priority=7
    ),
}

LEGAL_CONFIG = DomainConfig(
    domain=DocumentDomain.LEGAL,
    patterns=LEGAL_PATTERNS,
    never_split={"individual_clause", "definition_block", "recital", "signature_block"},
    preserve_elements={"clauses", "definitions", "signature_blocks", "recitals"},
    default_chunk_size=1024,
    default_overlap_percent=5,
    section_headers=["RECITALS", "DEFINITIONS", "TERMS AND CONDITIONS", "REPRESENTATIONS",
                     "WARRANTIES", "INDEMNIFICATION", "TERMINATION", "MISCELLANEOUS"],
    recommended_strategy="clause_preserving"
)


# ============================================================================
# ACADEMIC DOMAIN PATTERNS
# ============================================================================

ACADEMIC_PATTERNS = {
    "abstract": DomainPattern(
        name="abstract",
        pattern=re.compile(r"^(?:Abstract|ABSTRACT)[:\s]*$", re.MULTILINE),
        description="Abstract section",
        priority=10
    ),
    "citation_bracket": DomainPattern(
        name="citation_bracket",
        pattern=re.compile(r"\[[\d,\s\-]+\]"),
        description="Bracket citations like [1,2,3]",
        priority=7
    ),
    "citation_author_year": DomainPattern(
        name="citation_author_year",
        pattern=re.compile(r"\([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][a-z]+)?,?\s*\d{4}[a-z]?\)"),
        description="Author-year citations like (Smith, 2024)",
        priority=7
    ),
    "equation": DomainPattern(
        name="equation",
        pattern=re.compile(r"\$\$[\s\S]*?\$\$|\\begin\{equation\}[\s\S]*?\\end\{equation\}"),
        description="LaTeX equations",
        priority=8
    ),
    "figure_reference": DomainPattern(
        name="figure_reference",
        pattern=re.compile(r"(?:Figure|Fig\.?|Table|Exhibit)\s+\d+(?:\.\d+)?", re.IGNORECASE),
        description="Figure and table references",
        priority=6
    ),
    "bibliography": DomainPattern(
        name="bibliography",
        pattern=re.compile(r"^(?:References|Bibliography|Works Cited)[:\s]*$", re.MULTILINE | re.IGNORECASE),
        description="Bibliography section",
        priority=9
    ),
    "doi": DomainPattern(
        name="doi",
        pattern=re.compile(r"(?:doi|DOI)[:\s]*10\.\d+/[^\s]+"),
        description="DOI references",
        priority=5
    ),
    "keywords": DomainPattern(
        name="keywords",
        pattern=re.compile(r"^(?:Keywords|Key\s*words)[:\s]*", re.MULTILINE | re.IGNORECASE),
        description="Keywords section",
        priority=6
    ),
}

ACADEMIC_CONFIG = DomainConfig(
    domain=DocumentDomain.ACADEMIC,
    patterns=ACADEMIC_PATTERNS,
    never_split={"abstract", "equation_block", "citation_with_context"},
    preserve_elements={"abstract", "equations", "figure_references", "citations"},
    default_chunk_size=768,
    default_overlap_percent=15,
    section_headers=["Abstract", "Introduction", "Background", "Literature Review",
                     "Methods", "Methodology", "Results", "Discussion", "Conclusion",
                     "Acknowledgments", "References", "Appendix"],
    recommended_strategy="academic_structure"
)


# ============================================================================
# MEDICAL DOMAIN PATTERNS
# ============================================================================

MEDICAL_PATTERNS = {
    "soap_section": DomainPattern(
        name="soap_section",
        pattern=re.compile(
            r"^(?:#+\s*)?(?:Chief Complaint|History of Present Illness|HPI|"
            r"Past Medical History|PMH|Past Surgical History|PSH|"
            r"Medications|Current Medications|Allergies|"
            r"Social History|Family History|"
            r"Review of Systems|ROS|Physical Exam|PE|Physical Examination|"
            r"Assessment|Plan|Impression|Diagnosis|Diagnoses|"
            r"Vital Signs|Vitals|Labs|Laboratory|Imaging|Radiology|"
            r"Provider|Patient|Encounter)[:\s]*",
            re.MULTILINE | re.IGNORECASE
        ),
        description="SOAP and medical section headers",
        priority=10
    ),
    "icd_code": DomainPattern(
        name="icd_code",
        pattern=re.compile(r"\b[A-Z]\d{2}(?:\.\d{1,2})?\b"),
        description="ICD-10 codes",
        priority=7
    ),
    "cpt_code": DomainPattern(
        name="cpt_code",
        pattern=re.compile(r"\b\d{5}(?:-\d{2})?\b"),
        description="CPT codes",
        priority=6
    ),
    "vital_signs": DomainPattern(
        name="vital_signs",
        pattern=re.compile(
            r"(?:BP|Blood Pressure|HR|Heart Rate|RR|Respiratory Rate|"
            r"Temp|Temperature|SpO2|O2 Sat|Weight|Height|BMI)[:\s]*[\d/\.]+",
            re.IGNORECASE
        ),
        description="Vital sign measurements",
        priority=8
    ),
    "medication": DomainPattern(
        name="medication",
        pattern=re.compile(r"\b\d+\s*(?:mg|mcg|ml|mL|units?|IU|g|kg)\b", re.IGNORECASE),
        description="Medication dosages",
        priority=7
    ),
    "lab_value": DomainPattern(
        name="lab_value",
        pattern=re.compile(r"\b[\d\.]+\s*(?:mg/dL|mmol/L|mEq/L|%|g/dL|x10\^?\d*|U/L|IU/L)\b"),
        description="Lab values with units",
        priority=7
    ),
    "date_of_service": DomainPattern(
        name="date_of_service",
        pattern=re.compile(r"(?:Date of Service|DOS|Visit Date|Encounter Date)[:\s]*", re.IGNORECASE),
        description="Date of service indicators",
        priority=6
    ),
}

MEDICAL_CONFIG = DomainConfig(
    domain=DocumentDomain.MEDICAL,
    patterns=MEDICAL_PATTERNS,
    never_split={"medication_list", "vital_signs_block", "assessment_plan"},
    preserve_elements={"medical_sections", "vitals", "medications", "lab_values"},
    default_chunk_size=1024,
    default_overlap_percent=5,
    section_headers=["Chief Complaint", "HPI", "History of Present Illness",
                     "Past Medical History", "Medications", "Allergies",
                     "Physical Exam", "Assessment", "Plan", "Vital Signs"],
    recommended_strategy="medical_section"
)


# ============================================================================
# ENGINEERING DOMAIN PATTERNS
# ============================================================================

ENGINEERING_PATTERNS = {
    "requirement_id": DomainPattern(
        name="requirement_id",
        pattern=re.compile(r"\b(?:REQ|SPEC|SRS|SDD|ICD|HLD|LLD)[-_]?\d+(?:\.\d+)*\b"),
        description="Requirement IDs",
        priority=9
    ),
    "specification": DomainPattern(
        name="specification",
        pattern=re.compile(r"^\s*\d+(?:\.\d+)*\s+[A-Z]", re.MULTILINE),
        description="Numbered specifications",
        priority=8
    ),
    "tolerance": DomainPattern(
        name="tolerance",
        pattern=re.compile(r"[±+\-]\s*[\d\.]+\s*(?:mm|cm|m|in|ft|°|%|ppm|dB)"),
        description="Tolerance specifications",
        priority=7
    ),
    "version": DomainPattern(
        name="version",
        pattern=re.compile(r"[Vv](?:ersion)?\s*[\d\.]+(?:[a-z])?"),
        description="Version numbers",
        priority=6
    ),
    "diagram_reference": DomainPattern(
        name="diagram_reference",
        pattern=re.compile(r"(?:Figure|Diagram|Drawing|Schematic|Appendix)\s+[\d\w\.\-]+", re.IGNORECASE),
        description="Diagram and figure references",
        priority=7
    ),
    "code_block": DomainPattern(
        name="code_block",
        pattern=re.compile(r"```[\s\S]*?```|<code>[\s\S]*?</code>|<pre>[\s\S]*?</pre>"),
        description="Code blocks",
        priority=8
    ),
    "shall_requirement": DomainPattern(
        name="shall_requirement",
        pattern=re.compile(r"\b(?:shall|must|will|should|may)\s+(?:be|have|provide|support|include)", re.IGNORECASE),
        description="Requirement language (shall/must/should)",
        priority=7
    ),
    "interface": DomainPattern(
        name="interface",
        pattern=re.compile(r"\b(?:API|interface|endpoint|method|function|class)\s+\w+", re.IGNORECASE),
        description="Interface definitions",
        priority=6
    ),
}

ENGINEERING_CONFIG = DomainConfig(
    domain=DocumentDomain.ENGINEERING,
    patterns=ENGINEERING_PATTERNS,
    never_split={"spec_table", "code_block", "requirement_with_rationale"},
    preserve_elements={"specifications", "code_blocks", "diagrams", "requirements"},
    default_chunk_size=1024,
    default_overlap_percent=5,
    section_headers=["Introduction", "Scope", "Requirements", "Specifications",
                     "Interface", "Architecture", "Design", "Implementation",
                     "Testing", "Appendix"],
    recommended_strategy="requirement_based"
)


# ============================================================================
# EDUCATION DOMAIN PATTERNS
# ============================================================================

EDUCATION_PATTERNS = {
    "chapter": DomainPattern(
        name="chapter",
        pattern=re.compile(r"^(?:Chapter|Unit|Module|Lesson)\s+\d+", re.MULTILINE | re.IGNORECASE),
        description="Chapter or unit headers",
        priority=10
    ),
    "learning_objective": DomainPattern(
        name="learning_objective",
        pattern=re.compile(r"^(?:Learning\s+)?(?:Objectives?|Outcomes?|Goals?)[:\s]*$", re.MULTILINE | re.IGNORECASE),
        description="Learning objectives section",
        priority=9
    ),
    "exercise": DomainPattern(
        name="exercise",
        pattern=re.compile(r"^(?:Exercise|Problem|Question|Activity|Practice)\s+\d+", re.MULTILINE | re.IGNORECASE),
        description="Exercise or problem headers",
        priority=8
    ),
    "answer": DomainPattern(
        name="answer",
        pattern=re.compile(r"^(?:Answer|Solution|Key|Explanation)[:\s]*", re.MULTILINE | re.IGNORECASE),
        description="Answer or solution sections",
        priority=8
    ),
    "definition": DomainPattern(
        name="definition",
        pattern=re.compile(r"^(?:Definition|Term)[:\s]*", re.MULTILINE | re.IGNORECASE),
        description="Definition markers",
        priority=7
    ),
    "example": DomainPattern(
        name="example",
        pattern=re.compile(r"^Example\s*\d*[:\s]*", re.MULTILINE | re.IGNORECASE),
        description="Example markers",
        priority=7
    ),
    "summary": DomainPattern(
        name="summary",
        pattern=re.compile(r"^(?:Summary|Key Points|Takeaways|Review)[:\s]*$", re.MULTILINE | re.IGNORECASE),
        description="Summary sections",
        priority=8
    ),
    "step": DomainPattern(
        name="step",
        pattern=re.compile(r"^(?:Step\s+\d+|First|Second|Third|Next|Finally)[:\.\s]", re.MULTILINE | re.IGNORECASE),
        description="Procedural steps",
        priority=6
    ),
}

EDUCATION_CONFIG = DomainConfig(
    domain=DocumentDomain.EDUCATION,
    patterns=EDUCATION_PATTERNS,
    never_split={"question_answer_pair", "definition_with_example", "step_by_step"},
    preserve_elements={"learning_objectives", "exercises", "examples", "definitions"},
    default_chunk_size=768,
    default_overlap_percent=10,
    section_headers=["Introduction", "Objectives", "Content", "Summary",
                     "Exercises", "Review Questions", "Key Terms", "Further Reading"],
    recommended_strategy="educational_unit"
)


# ============================================================================
# FINANCIAL DOMAIN PATTERNS
# ============================================================================

FINANCIAL_PATTERNS = {
    "line_item": DomainPattern(
        name="line_item",
        pattern=re.compile(r"^\s*[\d\w\-]+\s+.*[\$€£¥][\d,\.]+", re.MULTILINE),
        description="Line items with amounts",
        priority=8
    ),
    "total": DomainPattern(
        name="total",
        pattern=re.compile(r"^(?:Total|Subtotal|Grand Total|Tax|Tip|Discount|Balance)[:\s]*", re.MULTILINE | re.IGNORECASE),
        description="Total/subtotal lines",
        priority=9
    ),
    "currency": DomainPattern(
        name="currency",
        pattern=re.compile(r"[\$€£¥]\s*[\d,]+\.?\d*"),
        description="Currency amounts",
        priority=7
    ),
    "date": DomainPattern(
        name="date",
        pattern=re.compile(r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b"),
        description="Date patterns",
        priority=6
    ),
    "account_number": DomainPattern(
        name="account_number",
        pattern=re.compile(r"(?:Account|Invoice|Receipt|Order|Transaction)\s*#?\s*:?\s*[\d\w\-]+", re.IGNORECASE),
        description="Account/invoice numbers",
        priority=7
    ),
    "payment_method": DomainPattern(
        name="payment_method",
        pattern=re.compile(r"\b(?:Cash|Credit|Debit|Check|Wire|ACH|PayPal|Visa|Mastercard|Amex)\b", re.IGNORECASE),
        description="Payment method indicators",
        priority=5
    ),
    "tax_id": DomainPattern(
        name="tax_id",
        pattern=re.compile(r"(?:Tax ID|EIN|VAT|GST)[:\s]*[\d\-]+", re.IGNORECASE),
        description="Tax identification numbers",
        priority=6
    ),
}

FINANCIAL_CONFIG = DomainConfig(
    domain=DocumentDomain.FINANCIAL,
    patterns=FINANCIAL_PATTERNS,
    never_split={"transaction_block", "line_item_table", "header_block"},
    preserve_elements={"line_items", "totals", "transaction_details"},
    default_chunk_size=1024,
    default_overlap_percent=15,
    section_headers=["Bill To", "Ship To", "Items", "Summary", "Payment"],
    recommended_strategy="table_preserving"
)


# ============================================================================
# GOVERNMENT DOMAIN PATTERNS
# ============================================================================

GOVERNMENT_PATTERNS = {
    "form_field": DomainPattern(
        name="form_field",
        pattern=re.compile(r"^[A-Z][A-Za-z\s]+:\s*(?:_+|\[[\s\]]*\]|\.{3,})", re.MULTILINE),
        description="Form fields with blanks",
        priority=8
    ),
    "instruction": DomainPattern(
        name="instruction",
        pattern=re.compile(r"^(?:Instructions?|Directions?|How to|Please)[:\s]*", re.MULTILINE | re.IGNORECASE),
        description="Instruction sections",
        priority=7
    ),
    "legal_reference": DomainPattern(
        name="legal_reference",
        pattern=re.compile(r"(?:\d+\s*)?(?:USC|U\.S\.C\.|CFR|C\.F\.R\.)\s*§?\s*[\d\.]+"),
        description="Legal code references (USC, CFR)",
        priority=8
    ),
    "effective_date": DomainPattern(
        name="effective_date",
        pattern=re.compile(r"(?:Effective|Valid|Expires?)[:\s]*(?:Date|From|Until|Through)?", re.IGNORECASE),
        description="Effective date indicators",
        priority=6
    ),
    "authority": DomainPattern(
        name="authority",
        pattern=re.compile(r"(?:Authority|Authorized by|Pursuant to|Under)[:\s]*", re.IGNORECASE),
        description="Authority references",
        priority=6
    ),
    "section_number": DomainPattern(
        name="section_number",
        pattern=re.compile(r"^(?:Part|Section|§)\s*\d+(?:\.\d+)*", re.MULTILINE | re.IGNORECASE),
        description="Government section numbers",
        priority=9
    ),
    "penalty": DomainPattern(
        name="penalty",
        pattern=re.compile(r"(?:penalty|fine|imprisonment|violation)", re.IGNORECASE),
        description="Penalty/violation language",
        priority=5
    ),
}

GOVERNMENT_CONFIG = DomainConfig(
    domain=DocumentDomain.GOVERNMENT,
    patterns=GOVERNMENT_PATTERNS,
    never_split={"field_with_instructions", "signature_block", "certification"},
    preserve_elements={"form_fields", "instructions", "legal_references"},
    default_chunk_size=1024,
    default_overlap_percent=5,
    section_headers=["Instructions", "Information", "Certification", "Signature",
                     "Privacy Act", "Paperwork Reduction Act"],
    recommended_strategy="form_structure"
)


# ============================================================================
# PROFESSIONAL DOMAIN PATTERNS
# ============================================================================

PROFESSIONAL_PATTERNS = {
    "section_header": DomainPattern(
        name="section_header",
        pattern=re.compile(
            r"^(?:Experience|Education|Skills|Summary|Objective|"
            r"Qualifications|Projects|Certifications|Awards|Publications)[:\s]*$",
            re.MULTILINE | re.IGNORECASE
        ),
        description="Resume section headers",
        priority=9
    ),
    "date_range": DomainPattern(
        name="date_range",
        pattern=re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}\s*[-–—]\s*(?:Present|Current|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4})", re.IGNORECASE),
        description="Employment date ranges",
        priority=7
    ),
    "job_title": DomainPattern(
        name="job_title",
        pattern=re.compile(r"^[A-Z][A-Za-z\s]+(?:Engineer|Manager|Director|Developer|Analyst|Designer|Specialist|Coordinator|Lead|Senior|Junior|Principal)", re.MULTILINE),
        description="Job title patterns",
        priority=6
    ),
    "contact_info": DomainPattern(
        name="contact_info",
        pattern=re.compile(r"(?:Email|Phone|Tel|Mobile|LinkedIn|GitHub)[:\s]*", re.IGNORECASE),
        description="Contact information markers",
        priority=5
    ),
    "bullet_point": DomainPattern(
        name="bullet_point",
        pattern=re.compile(r"^\s*[•\-\*\u2022]\s+", re.MULTILINE),
        description="Bullet points",
        priority=4
    ),
}

PROFESSIONAL_CONFIG = DomainConfig(
    domain=DocumentDomain.PROFESSIONAL,
    patterns=PROFESSIONAL_PATTERNS,
    never_split={"contact_block", "skills_list", "education_entry"},
    preserve_elements={"section_headers", "job_entries", "education_entries"},
    default_chunk_size=768,
    default_overlap_percent=10,
    section_headers=["Summary", "Experience", "Education", "Skills",
                     "Projects", "Certifications", "Awards"],
    recommended_strategy="semantic_header"
)


# ============================================================================
# GENERAL DOMAIN PATTERNS
# ============================================================================

GENERAL_PATTERNS = {
    "heading": DomainPattern(
        name="heading",
        pattern=re.compile(r"^#{1,6}\s+.+$", re.MULTILINE),
        description="Markdown headings",
        priority=8
    ),
    "paragraph": DomainPattern(
        name="paragraph",
        pattern=re.compile(r"\n\n+"),
        description="Paragraph breaks",
        priority=5
    ),
    "list_item": DomainPattern(
        name="list_item",
        pattern=re.compile(r"^\s*(?:\d+\.|[•\-\*])\s+", re.MULTILINE),
        description="List items",
        priority=6
    ),
    "quote": DomainPattern(
        name="quote",
        pattern=re.compile(r"^>\s+.+$", re.MULTILINE),
        description="Block quotes",
        priority=5
    ),
}

GENERAL_CONFIG = DomainConfig(
    domain=DocumentDomain.GENERAL,
    patterns=GENERAL_PATTERNS,
    never_split={"paragraph"},
    preserve_elements={"headings", "lists"},
    default_chunk_size=512,
    default_overlap_percent=10,
    section_headers=[],
    recommended_strategy="semantic_header"
)


# ============================================================================
# DOMAIN CONFIG REGISTRY
# ============================================================================

DOMAIN_CONFIGS: Dict[DocumentDomain, DomainConfig] = {
    DocumentDomain.LEGAL: LEGAL_CONFIG,
    DocumentDomain.ACADEMIC: ACADEMIC_CONFIG,
    DocumentDomain.MEDICAL: MEDICAL_CONFIG,
    DocumentDomain.ENGINEERING: ENGINEERING_CONFIG,
    DocumentDomain.EDUCATION: EDUCATION_CONFIG,
    DocumentDomain.FINANCIAL: FINANCIAL_CONFIG,
    DocumentDomain.GOVERNMENT: GOVERNMENT_CONFIG,
    DocumentDomain.PROFESSIONAL: PROFESSIONAL_CONFIG,
    DocumentDomain.GENERAL: GENERAL_CONFIG,
}


def get_domain_config(domain: DocumentDomain) -> DomainConfig:
    """Get the configuration for a specific domain."""
    return DOMAIN_CONFIGS.get(domain, GENERAL_CONFIG)


def get_domain_for_document_type(doc_type: str) -> DocumentDomain:
    """Get the domain for a given document type string."""
    try:
        doc_type_enum = DocumentType(doc_type.lower())
        return DOCUMENT_TYPE_TO_DOMAIN.get(doc_type_enum, DocumentDomain.GENERAL)
    except ValueError:
        return DocumentDomain.GENERAL


def detect_domain_from_content(content: str) -> Dict[DocumentDomain, float]:
    """
    Detect likely domains based on pattern matching in content.
    Returns a dictionary of domain -> confidence score.

    DEPRECATED (V3.0): This function uses fragile regex patterns that can cause
    misclassifications (e.g., "arXiv" matching prescription patterns).
    Use LLM-driven chunking instead:
        export LLM_DRIVEN_CHUNKING_ENABLED=true

    Or programmatically:
        from rag_service.chunking import analyze_content_structure
        config = analyze_content_structure(content)
    """
    import warnings
    warnings.warn(
        "detect_domain_from_content is deprecated. "
        "Use LLM-driven chunking (LLM_DRIVEN_CHUNKING_ENABLED=true) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    scores: Dict[DocumentDomain, float] = {domain: 0.0 for domain in DocumentDomain}

    for domain, config in DOMAIN_CONFIGS.items():
        for pattern_name, pattern_def in config.patterns.items():
            matches = pattern_def.pattern.findall(content)
            if matches:
                # Score based on number of matches and pattern priority
                score = len(matches) * (pattern_def.priority / 10.0)
                scores[domain] += score

    # Normalize scores
    max_score = max(scores.values()) if scores.values() else 1.0
    if max_score > 0:
        scores = {k: v / max_score for k, v in scores.items()}

    return scores
