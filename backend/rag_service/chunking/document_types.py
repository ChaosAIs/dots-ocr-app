"""
Centralized document type definitions for consistent classification and retrieval.

This module provides a single source of truth for document types used throughout
the system, ensuring consistency between:
1. Document classification during upload
2. Query enhancement during search
3. Document routing and matching

IMPORTANT: When adding new document types, update both:
- DOCUMENT_TYPES dict (for classification)
- TYPE_CATEGORIES dict (for category-based matching)
- TYPE_ALIASES dict (for flexible query matching)
"""

from enum import Enum
from typing import Dict, List, Set


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
    GENERAL = "general"


# =============================================================================
# DOCUMENT TYPES - Single Source of Truth
# =============================================================================
# Each type has: description, category, and example use cases
# This is the CANONICAL list used by both classification and query enhancement

DOCUMENT_TYPES: Dict[str, Dict] = {
    # Business & Finance
    "receipt": {
        "description": "meals, restaurant bills, purchases, transactions, expenses",
        "category": DocumentCategory.BUSINESS_FINANCE,
        "keywords": ["meal", "restaurant", "purchase", "expense", "payment", "bill", "dining"]
    },
    "invoice": {
        "description": "product purchases, orders, billing documents, vendor invoices",
        "category": DocumentCategory.BUSINESS_FINANCE,
        "keywords": ["invoice", "order", "billing", "vendor", "payment due"]
    },
    "financial_report": {
        "description": "financial statements, earnings reports, budget reports",
        "category": DocumentCategory.BUSINESS_FINANCE,
        "keywords": ["financial", "earnings", "budget", "quarterly", "annual report"]
    },
    "bank_statement": {
        "description": "bank account statements, transaction history",
        "category": DocumentCategory.BUSINESS_FINANCE,
        "keywords": ["bank", "statement", "account", "transaction"]
    },
    "expense_report": {
        "description": "expense claims, reimbursement requests",
        "category": DocumentCategory.BUSINESS_FINANCE,
        "keywords": ["expense", "reimbursement", "claim"]
    },

    # Legal & Compliance
    "contract": {
        "description": "legal contracts, agreements, binding documents",
        "category": DocumentCategory.LEGAL,
        "keywords": ["contract", "agreement", "terms", "parties", "obligations"]
    },
    "legal_brief": {
        "description": "legal arguments, court briefs, legal opinions",
        "category": DocumentCategory.LEGAL,
        "keywords": ["legal", "court", "plaintiff", "defendant", "ruling"]
    },
    "policy": {
        "description": "terms of service, privacy policy, compliance documents",
        "category": DocumentCategory.LEGAL,
        "keywords": ["policy", "terms", "privacy", "compliance", "regulation"]
    },

    # Technical & Engineering
    "technical_doc": {
        "description": "API docs, specifications, technical references, architecture docs",
        "category": DocumentCategory.TECHNICAL,
        "keywords": ["api", "specification", "technical", "architecture", "implementation", "system design"]
    },
    "user_manual": {
        "description": "user guides, how-to documents, instructions, manuals",
        "category": DocumentCategory.TECHNICAL,
        "keywords": ["manual", "guide", "instructions", "how to", "tutorial", "setup"]
    },
    "datasheet": {
        "description": "product datasheets, component specifications",
        "category": DocumentCategory.TECHNICAL,
        "keywords": ["datasheet", "specifications", "parameters", "component"]
    },

    # Academic & Research
    "research_paper": {
        "description": "academic papers, research articles, scientific publications",
        "category": DocumentCategory.ACADEMIC,
        "keywords": ["research", "study", "findings", "methodology", "abstract", "references", "citation"]
    },
    "thesis": {
        "description": "dissertations, thesis documents, academic projects",
        "category": DocumentCategory.ACADEMIC,
        "keywords": ["thesis", "dissertation", "hypothesis", "conclusion"]
    },
    "case_study": {
        "description": "case studies, analysis of specific examples",
        "category": DocumentCategory.ACADEMIC,
        "keywords": ["case study", "analysis", "example", "findings"]
    },

    # Medical & Healthcare
    "medical_record": {
        "description": "patient records, medical history, clinical notes",
        "category": DocumentCategory.MEDICAL,
        "keywords": ["patient", "medical", "diagnosis", "treatment", "history"]
    },
    "clinical_report": {
        "description": "clinical reports, lab results, medical findings",
        "category": DocumentCategory.MEDICAL,
        "keywords": ["clinical", "lab", "results", "findings", "test"]
    },

    # Education & Training
    "course_material": {
        "description": "textbooks, course content, educational materials",
        "category": DocumentCategory.EDUCATION,
        "keywords": ["course", "lesson", "chapter", "learning", "education"]
    },
    "tutorial": {
        "description": "step-by-step tutorials, learning guides",
        "category": DocumentCategory.EDUCATION,
        "keywords": ["tutorial", "step by step", "learn", "exercise"]
    },

    # Government & Public
    "government_form": {
        "description": "official forms, permits, licenses, applications",
        "category": DocumentCategory.GOVERNMENT,
        "keywords": ["form", "permit", "license", "application", "official"]
    },
    "legislation": {
        "description": "laws, regulations, legislative documents",
        "category": DocumentCategory.GOVERNMENT,
        "keywords": ["law", "regulation", "statute", "act", "section"]
    },

    # Professional & HR
    "resume": {
        "description": "CV, job applications, career history, professional profiles",
        "category": DocumentCategory.PROFESSIONAL,
        "keywords": ["resume", "cv", "experience", "skills", "employment", "career"]
    },
    "job_description": {
        "description": "job postings, role descriptions, requirements",
        "category": DocumentCategory.PROFESSIONAL,
        "keywords": ["job", "position", "requirements", "responsibilities", "qualifications"]
    },

    # Creative & Media
    "article": {
        "description": "news articles, blog posts, written content, opinion pieces",
        "category": DocumentCategory.CREATIVE,
        "keywords": ["article", "blog", "news", "post", "opinion", "editorial"]
    },
    "presentation": {
        "description": "slide decks, presentations, pitch documents",
        "category": DocumentCategory.CREATIVE,
        "keywords": ["presentation", "slides", "deck", "pitch"]
    },

    # General
    "report": {
        "description": "general reports, analysis, summaries, findings, white papers",
        "category": DocumentCategory.GENERAL,
        "keywords": ["report", "analysis", "summary", "findings", "overview", "white paper"]
    },
    "memo": {
        "description": "internal memos, notes, communications",
        "category": DocumentCategory.GENERAL,
        "keywords": ["memo", "note", "internal", "communication"]
    },
    "letter": {
        "description": "formal letters, correspondence",
        "category": DocumentCategory.GENERAL,
        "keywords": ["letter", "dear", "sincerely", "correspondence"]
    },
    "email": {
        "description": "email messages, email threads",
        "category": DocumentCategory.GENERAL,
        "keywords": ["email", "from", "to", "subject", "re:"]
    },
    "meeting_notes": {
        "description": "meeting minutes, discussion notes",
        "category": DocumentCategory.GENERAL,
        "keywords": ["meeting", "minutes", "attendees", "action items", "discussed"]
    },
    "other": {
        "description": "documents that don't fit other categories",
        "category": DocumentCategory.GENERAL,
        "keywords": []
    },
}


# =============================================================================
# TYPE ALIASES - For flexible matching between query and document types
# =============================================================================
# Maps query type hints to equivalent document types that should match
# This handles vocabulary differences between what users search for vs stored types

TYPE_ALIASES: Dict[str, List[str]] = {
    # Technical document aliases
    "technical_doc": ["technical_doc", "technical_spec", "api_documentation",
                      "architecture_doc", "code_documentation", "datasheet",
                      "research_paper", "report", "white_paper"],
    "manual": ["user_manual", "manual", "installation_guide", "guide", "tutorial"],
    "api_doc": ["technical_doc", "api_documentation", "technical_spec"],

    # Academic/Research aliases
    "article": ["article", "blog_post", "news_article", "research_paper",
                "academic_article", "report"],
    "paper": ["research_paper", "academic_article", "thesis", "report",
              "conference_paper", "white_paper"],
    "research": ["research_paper", "case_study", "thesis", "report"],

    # Business aliases
    "receipt": ["receipt", "expense_report", "invoice"],
    "invoice": ["invoice", "purchase_order", "quotation", "receipt"],
    "financial": ["financial_report", "bank_statement", "expense_report",
                  "invoice", "receipt"],

    # Professional aliases
    "resume": ["resume", "cv", "cover_letter"],
    "cv": ["resume", "cv", "cover_letter"],

    # Legal aliases
    "contract": ["contract", "agreement", "legal_brief"],
    "legal": ["contract", "agreement", "legal_brief", "policy", "legislation"],

    # General aliases
    "report": ["report", "financial_report", "clinical_report", "case_study",
               "research_paper", "analysis"],
    "document": ["report", "article", "memo", "letter", "other"],
}


# =============================================================================
# CATEGORY MAPPINGS - Group types by category for broader matching
# =============================================================================

TYPE_CATEGORIES: Dict[str, List[str]] = {
    DocumentCategory.BUSINESS_FINANCE.value: [
        "receipt", "invoice", "financial_report", "bank_statement",
        "expense_report", "purchase_order", "quotation"
    ],
    DocumentCategory.LEGAL.value: [
        "contract", "agreement", "legal_brief", "policy",
        "terms_of_service", "privacy_policy", "compliance_doc"
    ],
    DocumentCategory.TECHNICAL.value: [
        "technical_doc", "technical_spec", "api_documentation",
        "user_manual", "datasheet", "architecture_doc", "code_documentation"
    ],
    DocumentCategory.ACADEMIC.value: [
        "research_paper", "thesis", "academic_article", "case_study",
        "literature_review", "conference_paper"
    ],
    DocumentCategory.MEDICAL.value: [
        "medical_record", "clinical_report", "prescription",
        "lab_result", "patient_summary"
    ],
    DocumentCategory.EDUCATION.value: [
        "course_material", "textbook", "tutorial", "syllabus",
        "lecture_notes", "exam"
    ],
    DocumentCategory.GOVERNMENT.value: [
        "government_form", "permit", "license", "legislation",
        "policy_document", "official_notice"
    ],
    DocumentCategory.PROFESSIONAL.value: [
        "resume", "cv", "cover_letter", "job_description",
        "performance_review", "employee_handbook"
    ],
    DocumentCategory.CREATIVE.value: [
        "article", "blog_post", "news_article", "presentation",
        "manuscript", "book_chapter"
    ],
    DocumentCategory.GENERAL.value: [
        "report", "memo", "letter", "email", "meeting_notes", "other"
    ],
}


def get_all_document_types() -> List[str]:
    """Get list of all valid document types."""
    return list(DOCUMENT_TYPES.keys())


def get_document_type_descriptions() -> Dict[str, str]:
    """Get document types with their descriptions (for prompts)."""
    return {k: v["description"] for k, v in DOCUMENT_TYPES.items()}


def get_type_category(doc_type: str) -> str:
    """Get the category for a document type."""
    doc_type = doc_type.lower()
    if doc_type in DOCUMENT_TYPES:
        return DOCUMENT_TYPES[doc_type]["category"].value
    return DocumentCategory.GENERAL.value


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
        hint_lower = hint.lower()
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

    doc_type_lower = doc_type.lower()
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
                    lines.append(f'  * "{doc_type}" - {info["description"]}')

    return "\n".join(lines)


def generate_classification_taxonomy() -> str:
    """
    Generate the document type taxonomy for classification prompts.
    Used in document_classifier.py.
    """
    lines = ["DOCUMENT TYPE TAXONOMY:"]

    category_names = {
        DocumentCategory.BUSINESS_FINANCE: "Business/Finance",
        DocumentCategory.LEGAL: "Legal",
        DocumentCategory.TECHNICAL: "Technical",
        DocumentCategory.ACADEMIC: "Academic",
        DocumentCategory.MEDICAL: "Medical",
        DocumentCategory.EDUCATION: "Education",
        DocumentCategory.GOVERNMENT: "Government",
        DocumentCategory.PROFESSIONAL: "Professional",
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
