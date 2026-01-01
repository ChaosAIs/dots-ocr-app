"""
Extraction Service Module

Provides automatic structured data extraction from documents:
- Eligibility checking
- Schema-based extraction
- Direct parsing for tabular data
- Validation pipeline

NOTE: As of migration 022, EXTRACTION_STRATEGIES has been removed.
All tabular data uses direct parsing only. LLM is only used for
document classification and field mapping inference.
"""

from .extraction_config import (
    EXTRACTABLE_DOCUMENT_TYPES,
    get_schema_for_document_type,
    is_extractable_document_type,
)
from .eligibility_checker import ExtractionEligibilityChecker
from .extraction_service import ExtractionService

__all__ = [
    'EXTRACTABLE_DOCUMENT_TYPES',
    'get_schema_for_document_type',
    'is_extractable_document_type',
    'ExtractionEligibilityChecker',
    'ExtractionService',
]
