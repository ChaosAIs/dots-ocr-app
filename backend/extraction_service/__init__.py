"""
Extraction Service Module

Provides automatic structured data extraction from documents:
- Eligibility checking
- Schema-based extraction
- Multiple extraction strategies
- Validation pipeline
"""

from .extraction_config import (
    EXTRACTABLE_DOCUMENT_TYPES,
    EXTRACTION_STRATEGIES,
    get_schema_for_document_type,
    is_extractable_document_type,
)
from .eligibility_checker import ExtractionEligibilityChecker
from .extraction_service import ExtractionService

__all__ = [
    'EXTRACTABLE_DOCUMENT_TYPES',
    'EXTRACTION_STRATEGIES',
    'get_schema_for_document_type',
    'is_extractable_document_type',
    'ExtractionEligibilityChecker',
    'ExtractionService',
]
