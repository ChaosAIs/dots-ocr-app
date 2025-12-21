"""
Chunking module for domain-aware document processing.

This module provides intelligent chunking strategies that adapt to different
document types and domains including legal, academic, medical, engineering,
education, financial, and government documents.
"""

from .domain_patterns import (
    DocumentDomain,
    DocumentType,
    DomainConfig,
    DomainPattern,
    DOMAIN_CONFIGS,
    get_domain_config,
    get_domain_for_document_type,
    detect_domain_from_content,
)

from .chunk_metadata import (
    ChunkingProfile,
    UniversalChunkMetadata,
    LegalChunkMetadata,
    AcademicChunkMetadata,
    MedicalChunkMetadata,
    EngineeringChunkMetadata,
    EducationChunkMetadata,
    FinancialChunkMetadata,
)

from .chunking_strategies import (
    ChunkingStrategy,
    WholeDocumentStrategy,
    SemanticHeaderStrategy,
    ClausePreservingStrategy,
    AcademicStructureStrategy,
    MedicalSectionStrategy,
    RequirementBasedStrategy,
    EducationalUnitStrategy,
    TablePreservingStrategy,
    ParagraphStrategy,
    get_strategy_for_profile,
)

from .document_classifier import (
    DocumentClassifier,
    classify_document,
)

from .adaptive_chunker import (
    AdaptiveChunker,
    AdaptiveChunkingResult,
    chunk_document_adaptive,
    chunk_file_adaptive,
)

__all__ = [
    # Domain patterns
    "DocumentDomain",
    "DocumentType",
    "DomainConfig",
    "DomainPattern",
    "DOMAIN_CONFIGS",
    "get_domain_config",
    "get_domain_for_document_type",
    "detect_domain_from_content",
    # Metadata
    "ChunkingProfile",
    "UniversalChunkMetadata",
    "LegalChunkMetadata",
    "AcademicChunkMetadata",
    "MedicalChunkMetadata",
    "EngineeringChunkMetadata",
    "EducationChunkMetadata",
    "FinancialChunkMetadata",
    # Strategies
    "ChunkingStrategy",
    "WholeDocumentStrategy",
    "SemanticHeaderStrategy",
    "ClausePreservingStrategy",
    "AcademicStructureStrategy",
    "MedicalSectionStrategy",
    "RequirementBasedStrategy",
    "EducationalUnitStrategy",
    "TablePreservingStrategy",
    "ParagraphStrategy",
    "get_strategy_for_profile",
    # Classifier
    "DocumentClassifier",
    "classify_document",
    # Adaptive chunker
    "AdaptiveChunker",
    "AdaptiveChunkingResult",
    "chunk_document_adaptive",
    "chunk_file_adaptive",
]
