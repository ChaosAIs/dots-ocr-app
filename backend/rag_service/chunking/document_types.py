"""
Centralized document type definitions for consistent classification and retrieval.

This module provides a single source of truth for document types used throughout
the system, ensuring consistency between:
1. Document classification during upload/metadata extraction
2. Query enhancement during search
3. Document routing and matching
4. Extraction eligibility checking

IMPORTANT: This is the CANONICAL source. All other modules should import from here.
When adding new document types, update:
- DOCUMENT_TYPES dict (for classification)
- TYPE_CATEGORIES dict (for category-based matching)
- TYPE_ALIASES dict (for flexible query matching)
"""

from enum import Enum
from typing import Dict, List, Set, Optional
from dataclasses import dataclass


class DocumentCategory(Enum):
    """High-level document categories for grouping related types."""
    BUSINESS_FINANCE = "business_finance"
    LEGAL = "legal"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    MEDICAL = "medical"
    EDUCATION = "education"
    GOVERNMENT = "government"
    PROFESSIONAL = "professional"
    CREATIVE = "creative"
    LOGISTICS = "logistics"
    GENERAL = "general"


@dataclass
class DocumentTypeInfo:
    """Complete information about a document type."""
    type_name: str
    description: str
    category: DocumentCategory
    keywords: List[str]
    # Extraction settings
    is_extractable: bool = False  # Has structured data to extract (line items, tables)
    schema_type: Optional[str] = None  # Maps to data_schemas.schema_type for extraction
    # Chunking settings
    chunking_strategy: str = "semantic"  # semantic, hierarchical, fixed, hybrid
    chunk_size: int = 1000
    chunk_overlap: int = 100


# =============================================================================
# DOCUMENT TYPES - Single Source of Truth
# =============================================================================
# This is the CANONICAL list used by:
# - Metadata extraction (prompts.py)
# - Query enhancement
# - Document routing
# - Extraction eligibility
# - Chunking strategy selection

DOCUMENT_TYPES: Dict[str, DocumentTypeInfo] = {
    # =========================================================================
    # BUSINESS & FINANCE (Extractable types have structured line items)
    # =========================================================================
    "receipt": DocumentTypeInfo(
        type_name="receipt",
        description="meal receipts, restaurant bills, grocery receipts, retail purchases, transaction receipts",
        category=DocumentCategory.BUSINESS_FINANCE,
        keywords=["meal", "restaurant", "purchase", "expense", "payment", "bill", "dining", "grocery", "retail"],
        is_extractable=True,
        schema_type="receipt",
        chunking_strategy="semantic",
        chunk_size=800,
        chunk_overlap=80
    ),
    "invoice": DocumentTypeInfo(
        type_name="invoice",
        description="vendor invoices, billing documents, purchase invoices with line items, order confirmations",
        category=DocumentCategory.BUSINESS_FINANCE,
        keywords=["invoice", "order", "billing", "vendor", "payment due", "amount due", "bill to"],
        is_extractable=True,
        schema_type="invoice",
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "bank_statement": DocumentTypeInfo(
        type_name="bank_statement",
        description="bank account statements, transaction history, account balance reports",
        category=DocumentCategory.BUSINESS_FINANCE,
        keywords=["bank", "statement", "account", "transaction", "balance", "deposit", "withdrawal"],
        is_extractable=True,
        schema_type="bank_statement",
        chunking_strategy="semantic",
        chunk_size=1200,
        chunk_overlap=120
    ),
    "expense_report": DocumentTypeInfo(
        type_name="expense_report",
        description="expense claims, reimbursement requests, travel expense reports",
        category=DocumentCategory.BUSINESS_FINANCE,
        keywords=["expense", "reimbursement", "claim", "travel", "per diem"],
        is_extractable=True,
        schema_type="expense_report",
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "purchase_order": DocumentTypeInfo(
        type_name="purchase_order",
        description="purchase orders, procurement documents, PO forms",
        category=DocumentCategory.BUSINESS_FINANCE,
        keywords=["purchase order", "PO", "procurement", "requisition", "order"],
        is_extractable=True,
        schema_type="purchase_order",
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "quotation": DocumentTypeInfo(
        type_name="quotation",
        description="price quotes, sales quotes, estimates, proposals with pricing",
        category=DocumentCategory.BUSINESS_FINANCE,
        keywords=["quote", "quotation", "estimate", "proposal", "pricing"],
        is_extractable=True,
        schema_type="quotation",
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "financial_report": DocumentTypeInfo(
        type_name="financial_report",
        description="financial statements, earnings reports, budget reports, annual reports",
        category=DocumentCategory.BUSINESS_FINANCE,
        keywords=["financial", "earnings", "budget", "quarterly", "annual report", "balance sheet", "P&L"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "tax_document": DocumentTypeInfo(
        type_name="tax_document",
        description="tax returns, tax forms, W-2, 1099, tax statements",
        category=DocumentCategory.BUSINESS_FINANCE,
        keywords=["tax", "W-2", "1099", "IRS", "return", "deduction"],
        is_extractable=True,
        schema_type="tax_document",
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),

    # =========================================================================
    # LOGISTICS & INVENTORY (Extractable)
    # =========================================================================
    "shipping_manifest": DocumentTypeInfo(
        type_name="shipping_manifest",
        description="shipping documents, delivery notes, packing lists, bills of lading",
        category=DocumentCategory.LOGISTICS,
        keywords=["shipping", "delivery", "packing list", "bill of lading", "freight", "shipment"],
        is_extractable=True,
        schema_type="shipping_manifest",
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "inventory_report": DocumentTypeInfo(
        type_name="inventory_report",
        description="inventory reports, stock level documents, warehouse reports",
        category=DocumentCategory.LOGISTICS,
        keywords=["inventory", "stock", "warehouse", "quantity", "SKU"],
        is_extractable=True,
        schema_type="inventory_report",
        chunking_strategy="semantic",
        chunk_size=1200,
        chunk_overlap=120
    ),

    # =========================================================================
    # LEGAL & COMPLIANCE (Non-extractable, narrative documents)
    # =========================================================================
    "contract": DocumentTypeInfo(
        type_name="contract",
        description="legal contracts, service agreements, binding documents, NDAs",
        category=DocumentCategory.LEGAL,
        keywords=["contract", "agreement", "terms", "parties", "obligations", "binding", "NDA"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "policy": DocumentTypeInfo(
        type_name="policy",
        description="company policies, return policies, refund policies, business rules and guidelines",
        category=DocumentCategory.LEGAL,
        keywords=["policy", "policies", "return", "refund", "exchange", "guidelines", "rules", "procedures"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "terms_of_service": DocumentTypeInfo(
        type_name="terms_of_service",
        description="terms of service, terms and conditions, user agreements",
        category=DocumentCategory.LEGAL,
        keywords=["terms of service", "terms and conditions", "TOS", "user agreement", "acceptable use"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "privacy_policy": DocumentTypeInfo(
        type_name="privacy_policy",
        description="privacy policies, data protection notices, GDPR compliance documents",
        category=DocumentCategory.LEGAL,
        keywords=["privacy", "data protection", "GDPR", "personal data", "cookies"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "legal_brief": DocumentTypeInfo(
        type_name="legal_brief",
        description="legal arguments, court briefs, legal opinions, case summaries",
        category=DocumentCategory.LEGAL,
        keywords=["legal", "court", "plaintiff", "defendant", "ruling", "brief", "motion"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "patent": DocumentTypeInfo(
        type_name="patent",
        description="patent documents, patent applications, intellectual property filings",
        category=DocumentCategory.LEGAL,
        keywords=["patent", "claims", "invention", "intellectual property", "prior art"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=2000,
        chunk_overlap=300
    ),
    "regulatory_filing": DocumentTypeInfo(
        type_name="regulatory_filing",
        description="regulatory submissions, compliance filings, SEC filings",
        category=DocumentCategory.LEGAL,
        keywords=["regulatory", "filing", "SEC", "compliance", "submission"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "insurance_document": DocumentTypeInfo(
        type_name="insurance_document",
        description="insurance policies, coverage documents, claims, certificates of insurance",
        category=DocumentCategory.LEGAL,
        keywords=["insurance", "policy", "coverage", "claim", "premium", "deductible"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "warranty": DocumentTypeInfo(
        type_name="warranty",
        description="warranty documents, guarantee certificates, product warranties",
        category=DocumentCategory.LEGAL,
        keywords=["warranty", "guarantee", "coverage", "limited warranty", "extended warranty"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1200,
        chunk_overlap=150
    ),

    # =========================================================================
    # TECHNICAL & ENGINEERING
    # =========================================================================
    "technical_doc": DocumentTypeInfo(
        type_name="technical_doc",
        description="technical documentation, API docs, specifications, architecture documents",
        category=DocumentCategory.TECHNICAL,
        keywords=["api", "specification", "technical", "architecture", "implementation", "system design"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "user_manual": DocumentTypeInfo(
        type_name="user_manual",
        description="user guides, how-to documents, instructions, product manuals",
        category=DocumentCategory.TECHNICAL,
        keywords=["manual", "guide", "instructions", "how to", "setup", "user guide"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1200,
        chunk_overlap=150
    ),
    "datasheet": DocumentTypeInfo(
        type_name="datasheet",
        description="product datasheets, component specifications, technical specs",
        category=DocumentCategory.TECHNICAL,
        keywords=["datasheet", "specifications", "parameters", "component", "product spec"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "installation_guide": DocumentTypeInfo(
        type_name="installation_guide",
        description="installation instructions, setup guides, deployment documentation",
        category=DocumentCategory.TECHNICAL,
        keywords=["installation", "setup", "deploy", "configure", "install"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1200,
        chunk_overlap=150
    ),
    "api_documentation": DocumentTypeInfo(
        type_name="api_documentation",
        description="API reference, endpoint documentation, SDK guides",
        category=DocumentCategory.TECHNICAL,
        keywords=["API", "endpoint", "REST", "SDK", "request", "response"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1200,
        chunk_overlap=150
    ),
    "release_notes": DocumentTypeInfo(
        type_name="release_notes",
        description="release notes, changelogs, version history, update notes",
        category=DocumentCategory.TECHNICAL,
        keywords=["release", "changelog", "version", "update", "fix", "feature"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "troubleshooting_guide": DocumentTypeInfo(
        type_name="troubleshooting_guide",
        description="troubleshooting guides, FAQ, problem-solution documents",
        category=DocumentCategory.TECHNICAL,
        keywords=["troubleshoot", "FAQ", "problem", "solution", "error", "fix"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),

    # =========================================================================
    # ACADEMIC & RESEARCH
    # =========================================================================
    "research_paper": DocumentTypeInfo(
        type_name="research_paper",
        description="academic papers, research articles, scientific publications, journal articles",
        category=DocumentCategory.ACADEMIC,
        keywords=["research", "study", "findings", "methodology", "abstract", "references", "citation"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "thesis": DocumentTypeInfo(
        type_name="thesis",
        description="dissertations, thesis documents, academic projects, PhD/Masters thesis",
        category=DocumentCategory.ACADEMIC,
        keywords=["thesis", "dissertation", "hypothesis", "conclusion", "PhD", "Masters"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=2000,
        chunk_overlap=300
    ),
    "case_study": DocumentTypeInfo(
        type_name="case_study",
        description="case studies, analysis of specific examples, business cases",
        category=DocumentCategory.ACADEMIC,
        keywords=["case study", "analysis", "example", "findings", "business case"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "literature_review": DocumentTypeInfo(
        type_name="literature_review",
        description="literature reviews, systematic reviews, research summaries",
        category=DocumentCategory.ACADEMIC,
        keywords=["literature review", "systematic review", "meta-analysis", "survey"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "conference_paper": DocumentTypeInfo(
        type_name="conference_paper",
        description="conference papers, proceedings, workshop papers",
        category=DocumentCategory.ACADEMIC,
        keywords=["conference", "proceedings", "workshop", "symposium"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "white_paper": DocumentTypeInfo(
        type_name="white_paper",
        description="white papers, position papers, thought leadership documents",
        category=DocumentCategory.ACADEMIC,
        keywords=["white paper", "position paper", "thought leadership", "industry report"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),

    # =========================================================================
    # MEDICAL & HEALTHCARE
    # =========================================================================
    "medical_record": DocumentTypeInfo(
        type_name="medical_record",
        description="patient records, medical history, clinical notes, health records",
        category=DocumentCategory.MEDICAL,
        keywords=["patient", "medical", "diagnosis", "treatment", "history", "clinical"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1200,
        chunk_overlap=150
    ),
    "clinical_report": DocumentTypeInfo(
        type_name="clinical_report",
        description="clinical reports, diagnostic reports, medical findings",
        category=DocumentCategory.MEDICAL,
        keywords=["clinical", "diagnostic", "findings", "assessment"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1200,
        chunk_overlap=150
    ),
    "lab_result": DocumentTypeInfo(
        type_name="lab_result",
        description="laboratory test results, blood work, diagnostic test results",
        category=DocumentCategory.MEDICAL,
        keywords=["lab", "results", "test", "blood", "diagnostic"],
        is_extractable=True,
        schema_type="lab_result",
        chunking_strategy="semantic",
        chunk_size=800,
        chunk_overlap=80
    ),
    "prescription": DocumentTypeInfo(
        type_name="prescription",
        description="medical prescriptions, medication orders",
        category=DocumentCategory.MEDICAL,
        keywords=["prescription", "medication", "Rx", "dosage", "pharmacy"],
        is_extractable=True,
        schema_type="prescription",
        chunking_strategy="semantic",
        chunk_size=500,
        chunk_overlap=50
    ),
    "discharge_summary": DocumentTypeInfo(
        type_name="discharge_summary",
        description="hospital discharge summaries, patient discharge instructions",
        category=DocumentCategory.MEDICAL,
        keywords=["discharge", "summary", "instructions", "follow-up"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1200,
        chunk_overlap=150
    ),

    # =========================================================================
    # EDUCATION & TRAINING
    # =========================================================================
    "course_material": DocumentTypeInfo(
        type_name="course_material",
        description="course content, educational materials, lecture notes, learning modules",
        category=DocumentCategory.EDUCATION,
        keywords=["course", "lesson", "chapter", "learning", "education", "module"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1200,
        chunk_overlap=150
    ),
    "textbook": DocumentTypeInfo(
        type_name="textbook",
        description="textbooks, educational books, reference books",
        category=DocumentCategory.EDUCATION,
        keywords=["textbook", "chapter", "education", "learning"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "tutorial": DocumentTypeInfo(
        type_name="tutorial",
        description="step-by-step tutorials, learning guides, how-to guides",
        category=DocumentCategory.EDUCATION,
        keywords=["tutorial", "step by step", "learn", "exercise", "walkthrough"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "syllabus": DocumentTypeInfo(
        type_name="syllabus",
        description="course syllabi, curriculum outlines, program requirements",
        category=DocumentCategory.EDUCATION,
        keywords=["syllabus", "curriculum", "schedule", "requirements", "grading"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "exam": DocumentTypeInfo(
        type_name="exam",
        description="exams, quizzes, tests, assessments, practice problems",
        category=DocumentCategory.EDUCATION,
        keywords=["exam", "quiz", "test", "assessment", "question", "answer"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=800,
        chunk_overlap=80
    ),
    "certification_material": DocumentTypeInfo(
        type_name="certification_material",
        description="certification study guides, exam prep materials, professional certifications",
        category=DocumentCategory.EDUCATION,
        keywords=["certification", "exam prep", "study guide", "professional"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1200,
        chunk_overlap=150
    ),

    # =========================================================================
    # GOVERNMENT & PUBLIC
    # =========================================================================
    "government_form": DocumentTypeInfo(
        type_name="government_form",
        description="official forms, permits, licenses, applications, government documents",
        category=DocumentCategory.GOVERNMENT,
        keywords=["form", "permit", "license", "application", "official", "government"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "legislation": DocumentTypeInfo(
        type_name="legislation",
        description="laws, regulations, legislative documents, statutes, ordinances",
        category=DocumentCategory.GOVERNMENT,
        keywords=["law", "regulation", "statute", "act", "section", "ordinance"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "public_notice": DocumentTypeInfo(
        type_name="public_notice",
        description="public notices, announcements, official communications",
        category=DocumentCategory.GOVERNMENT,
        keywords=["notice", "announcement", "public", "official"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),

    # =========================================================================
    # PROFESSIONAL & HR
    # =========================================================================
    "resume": DocumentTypeInfo(
        type_name="resume",
        description="resumes, CVs, curriculum vitae, professional profiles",
        category=DocumentCategory.PROFESSIONAL,
        keywords=["resume", "cv", "experience", "skills", "employment", "career", "education"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=800,
        chunk_overlap=80
    ),
    "cover_letter": DocumentTypeInfo(
        type_name="cover_letter",
        description="cover letters, job application letters, introduction letters",
        category=DocumentCategory.PROFESSIONAL,
        keywords=["cover letter", "application", "dear hiring", "position"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=600,
        chunk_overlap=60
    ),
    "job_description": DocumentTypeInfo(
        type_name="job_description",
        description="job postings, role descriptions, position requirements",
        category=DocumentCategory.PROFESSIONAL,
        keywords=["job", "position", "requirements", "responsibilities", "qualifications"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "employee_handbook": DocumentTypeInfo(
        type_name="employee_handbook",
        description="employee handbooks, HR policies, company guidelines",
        category=DocumentCategory.PROFESSIONAL,
        keywords=["handbook", "employee", "HR", "policy", "guidelines"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "performance_review": DocumentTypeInfo(
        type_name="performance_review",
        description="performance reviews, employee evaluations, appraisals",
        category=DocumentCategory.PROFESSIONAL,
        keywords=["performance", "review", "evaluation", "appraisal", "goals"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "training_material": DocumentTypeInfo(
        type_name="training_material",
        description="training documents, onboarding materials, employee training guides",
        category=DocumentCategory.PROFESSIONAL,
        keywords=["training", "onboarding", "orientation", "learning"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1200,
        chunk_overlap=150
    ),

    # =========================================================================
    # CREATIVE & MEDIA
    # =========================================================================
    "article": DocumentTypeInfo(
        type_name="article",
        description="news articles, blog posts, written content, opinion pieces, editorials",
        category=DocumentCategory.CREATIVE,
        keywords=["article", "blog", "news", "post", "opinion", "editorial"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "presentation": DocumentTypeInfo(
        type_name="presentation",
        description="slide decks, presentations, pitch documents, keynotes",
        category=DocumentCategory.CREATIVE,
        keywords=["presentation", "slides", "deck", "pitch", "keynote"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=800,
        chunk_overlap=80
    ),
    "newsletter": DocumentTypeInfo(
        type_name="newsletter",
        description="newsletters, bulletins, periodic communications",
        category=DocumentCategory.CREATIVE,
        keywords=["newsletter", "bulletin", "update", "digest"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "marketing_material": DocumentTypeInfo(
        type_name="marketing_material",
        description="marketing brochures, promotional materials, advertising content",
        category=DocumentCategory.CREATIVE,
        keywords=["marketing", "brochure", "promotional", "advertising", "campaign"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "book": DocumentTypeInfo(
        type_name="book",
        description="books, ebooks, long-form publications, manuscripts",
        category=DocumentCategory.CREATIVE,
        keywords=["book", "ebook", "chapter", "manuscript", "publication"],
        is_extractable=False,
        chunking_strategy="hierarchical",
        chunk_size=1500,
        chunk_overlap=200
    ),
    "script": DocumentTypeInfo(
        type_name="script",
        description="scripts, screenplays, dialogue, video scripts",
        category=DocumentCategory.CREATIVE,
        keywords=["script", "screenplay", "dialogue", "scene"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),

    # =========================================================================
    # GENERAL & MISCELLANEOUS
    # =========================================================================
    "report": DocumentTypeInfo(
        type_name="report",
        description="general reports, analysis, summaries, findings documents",
        category=DocumentCategory.GENERAL,
        keywords=["report", "analysis", "summary", "findings", "overview"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1200,
        chunk_overlap=150
    ),
    "memo": DocumentTypeInfo(
        type_name="memo",
        description="internal memos, notes, communications, office memos",
        category=DocumentCategory.GENERAL,
        keywords=["memo", "note", "internal", "communication"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=800,
        chunk_overlap=80
    ),
    "letter": DocumentTypeInfo(
        type_name="letter",
        description="formal letters, correspondence, business letters",
        category=DocumentCategory.GENERAL,
        keywords=["letter", "dear", "sincerely", "correspondence"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=800,
        chunk_overlap=80
    ),
    "email": DocumentTypeInfo(
        type_name="email",
        description="email messages, email threads, electronic correspondence",
        category=DocumentCategory.GENERAL,
        keywords=["email", "from", "to", "subject", "re:"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=800,
        chunk_overlap=80
    ),
    "meeting_notes": DocumentTypeInfo(
        type_name="meeting_notes",
        description="meeting minutes, discussion notes, action items, agendas",
        category=DocumentCategory.GENERAL,
        keywords=["meeting", "minutes", "attendees", "action items", "discussed", "agenda"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "spreadsheet": DocumentTypeInfo(
        type_name="spreadsheet",
        description="Excel files, CSV files, tabular data, data exports",
        category=DocumentCategory.GENERAL,
        keywords=["spreadsheet", "excel", "csv", "table", "data"],
        is_extractable=True,
        schema_type="spreadsheet",
        chunking_strategy="fixed",
        chunk_size=1500,
        chunk_overlap=150
    ),
    "form": DocumentTypeInfo(
        type_name="form",
        description="fillable forms, questionnaires, surveys, applications",
        category=DocumentCategory.GENERAL,
        keywords=["form", "questionnaire", "survey", "application"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "checklist": DocumentTypeInfo(
        type_name="checklist",
        description="checklists, to-do lists, verification lists, audit checklists",
        category=DocumentCategory.GENERAL,
        keywords=["checklist", "to-do", "verification", "audit"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=800,
        chunk_overlap=80
    ),
    "catalog": DocumentTypeInfo(
        type_name="catalog",
        description="product catalogs, service catalogs, directories",
        category=DocumentCategory.GENERAL,
        keywords=["catalog", "catalogue", "directory", "listing"],
        is_extractable=False,
        chunking_strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100
    ),
    "other": DocumentTypeInfo(
        type_name="other",
        description="documents that don't fit other categories, unclassified documents",
        category=DocumentCategory.GENERAL,
        keywords=[],
        is_extractable=False,
        chunking_strategy="fixed",
        chunk_size=1000,
        chunk_overlap=100
    ),
}


# =============================================================================
# TYPE ALIASES - For flexible matching between query and document types
# =============================================================================
# Maps query type hints to equivalent document types that should match
# This handles vocabulary differences between what users search for vs stored types

TYPE_ALIASES: Dict[str, List[str]] = {
    # Technical document aliases
    "technical_doc": ["technical_doc", "api_documentation", "datasheet",
                      "installation_guide", "release_notes", "troubleshooting_guide"],
    "manual": ["user_manual", "installation_guide", "tutorial", "troubleshooting_guide"],
    "guide": ["user_manual", "installation_guide", "tutorial", "troubleshooting_guide"],
    "api_doc": ["technical_doc", "api_documentation"],
    "documentation": ["technical_doc", "api_documentation", "user_manual"],
    "spec": ["technical_doc", "datasheet"],
    "specification": ["technical_doc", "datasheet"],

    # Academic/Research aliases
    "article": ["article", "research_paper", "newsletter", "white_paper"],
    "paper": ["research_paper", "thesis", "conference_paper", "white_paper"],
    "research": ["research_paper", "case_study", "thesis", "literature_review", "white_paper"],
    "academic": ["research_paper", "thesis", "conference_paper", "literature_review"],
    "study": ["research_paper", "case_study", "literature_review"],

    # Business/Finance aliases
    "receipt": ["receipt", "expense_report"],
    "invoice": ["invoice", "purchase_order", "quotation"],
    "financial": ["financial_report", "bank_statement", "expense_report",
                  "invoice", "receipt", "tax_document"],
    "bill": ["invoice", "receipt"],
    "order": ["invoice", "purchase_order", "quotation"],
    "quote": ["quotation"],
    "statement": ["bank_statement", "financial_report"],

    # Logistics aliases
    "shipping": ["shipping_manifest", "inventory_report"],
    "inventory": ["inventory_report"],
    "delivery": ["shipping_manifest"],

    # Legal/Policy aliases - IMPORTANT for routing
    "policy": ["policy", "terms_of_service", "privacy_policy", "employee_handbook"],
    "return_policy": ["policy"],
    "refund_policy": ["policy"],
    "contract": ["contract", "terms_of_service"],
    "agreement": ["contract", "terms_of_service"],
    "legal": ["contract", "legal_brief", "policy", "terms_of_service",
              "privacy_policy", "patent", "regulatory_filing", "legislation"],
    "terms": ["terms_of_service", "policy", "contract"],
    "privacy": ["privacy_policy"],
    "compliance": ["regulatory_filing", "policy"],
    "insurance": ["insurance_document"],
    "warranty": ["warranty"],

    # Medical aliases
    "medical": ["medical_record", "clinical_report", "lab_result",
                "prescription", "discharge_summary"],
    "clinical": ["clinical_report", "medical_record", "lab_result"],
    "lab": ["lab_result"],
    "prescription": ["prescription"],

    # Education aliases
    "course": ["course_material", "textbook", "syllabus"],
    "education": ["course_material", "textbook", "tutorial", "syllabus",
                  "exam", "certification_material"],
    "learning": ["course_material", "tutorial", "textbook"],
    "training": ["training_material", "tutorial", "certification_material"],
    "test": ["exam"],
    "quiz": ["exam"],

    # Professional/HR aliases
    "resume": ["resume", "cover_letter"],
    "cv": ["resume"],
    "job": ["job_description", "resume", "cover_letter"],
    "hr": ["employee_handbook", "performance_review", "job_description"],
    "handbook": ["employee_handbook"],

    # Government aliases
    "government": ["government_form", "legislation", "public_notice"],
    "form": ["government_form", "form"],
    "law": ["legislation"],
    "regulation": ["legislation", "regulatory_filing"],

    # Creative/Media aliases
    "blog": ["article", "newsletter"],
    "news": ["article", "newsletter", "public_notice"],
    "presentation": ["presentation"],
    "slides": ["presentation"],
    "marketing": ["marketing_material", "newsletter"],
    "brochure": ["marketing_material", "catalog"],

    # General aliases
    "report": ["report", "financial_report", "clinical_report", "case_study",
               "white_paper", "inventory_report"],
    "document": ["report", "article", "memo", "letter", "other"],
    "memo": ["memo", "meeting_notes"],
    "notes": ["meeting_notes", "memo"],
    "email": ["email", "letter"],
    "spreadsheet": ["spreadsheet"],
    "excel": ["spreadsheet"],
    "csv": ["spreadsheet"],
    "table": ["spreadsheet"],
    "data": ["spreadsheet", "inventory_report"],
    "list": ["checklist", "catalog", "inventory_report"],
    "catalog": ["catalog"],
    "directory": ["catalog"],
}


# =============================================================================
# CATEGORY MAPPINGS - Group types by category for broader matching
# =============================================================================

TYPE_CATEGORIES: Dict[str, List[str]] = {
    DocumentCategory.BUSINESS_FINANCE.value: [
        "receipt", "invoice", "financial_report", "bank_statement",
        "expense_report", "purchase_order", "quotation", "tax_document"
    ],
    DocumentCategory.LOGISTICS.value: [
        "shipping_manifest", "inventory_report"
    ],
    DocumentCategory.LEGAL.value: [
        "contract", "policy", "terms_of_service", "privacy_policy",
        "legal_brief", "patent", "regulatory_filing", "insurance_document", "warranty"
    ],
    DocumentCategory.TECHNICAL.value: [
        "technical_doc", "user_manual", "datasheet", "installation_guide",
        "api_documentation", "release_notes", "troubleshooting_guide"
    ],
    DocumentCategory.ACADEMIC.value: [
        "research_paper", "thesis", "case_study", "literature_review",
        "conference_paper", "white_paper"
    ],
    DocumentCategory.MEDICAL.value: [
        "medical_record", "clinical_report", "lab_result",
        "prescription", "discharge_summary"
    ],
    DocumentCategory.EDUCATION.value: [
        "course_material", "textbook", "tutorial", "syllabus",
        "exam", "certification_material"
    ],
    DocumentCategory.GOVERNMENT.value: [
        "government_form", "legislation", "public_notice"
    ],
    DocumentCategory.PROFESSIONAL.value: [
        "resume", "cover_letter", "job_description",
        "employee_handbook", "performance_review", "training_material"
    ],
    DocumentCategory.CREATIVE.value: [
        "article", "presentation", "newsletter", "marketing_material",
        "book", "script"
    ],
    DocumentCategory.GENERAL.value: [
        "report", "memo", "letter", "email", "meeting_notes",
        "spreadsheet", "form", "checklist", "catalog", "other"
    ],
}


def get_all_document_types() -> List[str]:
    """Get list of all valid document types."""
    return list(DOCUMENT_TYPES.keys())


def get_document_type_info(doc_type: str) -> Optional[DocumentTypeInfo]:
    """Get DocumentTypeInfo for a document type."""
    doc_type_lower = doc_type.lower().strip()
    return DOCUMENT_TYPES.get(doc_type_lower)


def get_document_type_descriptions() -> Dict[str, str]:
    """Get document types with their descriptions (for prompts)."""
    return {k: v.description for k, v in DOCUMENT_TYPES.items()}


def get_type_category(doc_type: str) -> str:
    """Get the category for a document type."""
    doc_type = doc_type.lower().strip()
    if doc_type in DOCUMENT_TYPES:
        return DOCUMENT_TYPES[doc_type].category.value
    return DocumentCategory.GENERAL.value


def is_extractable_type(doc_type: str) -> bool:
    """Check if a document type supports data extraction."""
    doc_type_lower = doc_type.lower().strip()
    if doc_type_lower in DOCUMENT_TYPES:
        return DOCUMENT_TYPES[doc_type_lower].is_extractable
    return False


def get_schema_type(doc_type: str) -> Optional[str]:
    """Get the schema_type for extraction, if applicable."""
    doc_type_lower = doc_type.lower().strip()
    if doc_type_lower in DOCUMENT_TYPES:
        return DOCUMENT_TYPES[doc_type_lower].schema_type
    return None


def get_chunking_config(doc_type: str) -> Dict[str, any]:
    """Get chunking configuration for a document type."""
    doc_type_lower = doc_type.lower().strip()
    if doc_type_lower in DOCUMENT_TYPES:
        info = DOCUMENT_TYPES[doc_type_lower]
        return {
            "strategy": info.chunking_strategy,
            "chunk_size": info.chunk_size,
            "chunk_overlap": info.chunk_overlap,
            "category": info.category.value
        }
    # Default config
    return {
        "strategy": "semantic",
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "category": "general"
    }


def get_extractable_types() -> List[str]:
    """Get list of document types that support data extraction."""
    return [name for name, info in DOCUMENT_TYPES.items() if info.is_extractable]


def expand_type_aliases(type_hints: List[str]) -> Set[str]:
    """
    Expand type hints to include all aliases.

    Args:
        type_hints: List of document type hints from query

    Returns:
        Set of all matching document types (including aliases)
    """
    expanded = set()
    for hint in type_hints:
        hint_lower = hint.lower().strip()
        # Add the hint itself
        expanded.add(hint_lower)
        # Add all aliases
        if hint_lower in TYPE_ALIASES:
            expanded.update(TYPE_ALIASES[hint_lower])
    return expanded


def types_match(query_type_hints: List[str], doc_type: str) -> bool:
    """
    Check if document type matches any of the query type hints.
    Uses alias expansion for flexible matching.

    Args:
        query_type_hints: List of type hints from query enhancement
        doc_type: Document type from stored document

    Returns:
        True if there's a match, False otherwise
    """
    if not query_type_hints or not doc_type:
        return True  # No filtering if no hints or no doc type

    doc_type_lower = doc_type.lower().strip()
    expanded_hints = expand_type_aliases(query_type_hints)

    # Direct match
    if doc_type_lower in expanded_hints:
        return True

    # Check if document type has aliases that match
    for hint in expanded_hints:
        if hint in TYPE_ALIASES:
            if doc_type_lower in TYPE_ALIASES[hint]:
                return True

    # Category-based match (if doc is in same category as any hint)
    doc_category = get_type_category(doc_type_lower)
    for hint in query_type_hints:
        hint_category = get_type_category(hint.lower())
        if doc_category == hint_category and doc_category != DocumentCategory.GENERAL.value:
            return True

    return False


def generate_type_prompt_section() -> str:
    """
    Generate the document type section for LLM prompts.
    Ensures consistent type definitions across classification and query enhancement.
    """
    lines = ["## Document Types (choose from these EXACT types):"]

    # Group by category for better organization
    for category in DocumentCategory:
        category_types = TYPE_CATEGORIES.get(category.value, [])
        if category_types:
            # Get types that are in our main DOCUMENT_TYPES dict
            valid_types = [t for t in category_types if t in DOCUMENT_TYPES]
            if valid_types:
                lines.append(f"\n### {category.value.replace('_', ' ').title()}:")
                for doc_type in valid_types:
                    info = DOCUMENT_TYPES[doc_type]
                    lines.append(f'  * "{doc_type}" - {info.description}')

    return "\n".join(lines)


def generate_classification_taxonomy() -> str:
    """
    Generate the document type taxonomy for classification prompts.
    Used in document_classifier.py.
    """
    lines = ["DOCUMENT TYPE TAXONOMY:"]

    category_names = {
        DocumentCategory.BUSINESS_FINANCE: "Business/Finance",
        DocumentCategory.LOGISTICS: "Logistics/Inventory",
        DocumentCategory.LEGAL: "Legal/Policy",
        DocumentCategory.TECHNICAL: "Technical/Engineering",
        DocumentCategory.ACADEMIC: "Academic/Research",
        DocumentCategory.MEDICAL: "Medical/Healthcare",
        DocumentCategory.EDUCATION: "Education/Training",
        DocumentCategory.GOVERNMENT: "Government/Public",
        DocumentCategory.PROFESSIONAL: "Professional/HR",
        DocumentCategory.CREATIVE: "Creative/Media",
        DocumentCategory.GENERAL: "General",
    }

    for category, display_name in category_names.items():
        category_types = TYPE_CATEGORIES.get(category.value, [])
        valid_types = [t for t in category_types if t in DOCUMENT_TYPES]
        if valid_types:
            types_str = ", ".join(valid_types)
            lines.append(f"- {display_name}: {types_str}")

    return "\n".join(lines)


def generate_metadata_extraction_types() -> str:
    """
    Generate the document type list for metadata extraction prompt.
    This is the comprehensive list used by METADATA_STRUCTURED_EXTRACTION_PROMPT.
    """
    lines = []

    # Group by category for clarity
    category_order = [
        (DocumentCategory.BUSINESS_FINANCE, "Business/Finance"),
        (DocumentCategory.LOGISTICS, "Logistics"),
        (DocumentCategory.LEGAL, "Legal/Policy"),
        (DocumentCategory.TECHNICAL, "Technical"),
        (DocumentCategory.ACADEMIC, "Academic"),
        (DocumentCategory.MEDICAL, "Medical"),
        (DocumentCategory.EDUCATION, "Education"),
        (DocumentCategory.GOVERNMENT, "Government"),
        (DocumentCategory.PROFESSIONAL, "Professional"),
        (DocumentCategory.CREATIVE, "Creative"),
        (DocumentCategory.GENERAL, "General"),
    ]

    for category, display_name in category_order:
        category_types = TYPE_CATEGORIES.get(category.value, [])
        valid_types = [t for t in category_types if t in DOCUMENT_TYPES]
        if valid_types:
            lines.append(f"   # {display_name}:")
            for doc_type in valid_types:
                info = DOCUMENT_TYPES[doc_type]
                extractable_marker = " [EXTRACTABLE]" if info.is_extractable else ""
                lines.append(f'   - "{doc_type}" - {info.description}{extractable_marker}')

    return "\n".join(lines)
