"""
Chunking module for domain-aware document processing.

This module provides intelligent chunking strategies that adapt to different
document types and domains including legal, academic, medical, engineering,
education, financial, and government documents.

V3.0: LLM-Driven Structure Analysis
- Replaces pattern-based domain classification with LLM structure analysis
- Single LLM call per uploaded file to select optimal chunking strategy
- Predefined strategy library with 8 strategies
- Support for multi-page OCR (page-based sampling) and single-file conversion
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

from .parent_chunk_summarizer import (
    ParentChunkSummarizer,
    ParentChunkSummary,
    get_parent_chunk_summarizer,
    summarize_parent_chunk,
    needs_parent_chunk_summary,
    estimate_tokens,
)

# V3.0: LLM-driven chunking modules
from .content_sampler import (
    ContentSampler,
    SampledContent,
    sample_content,
)

from .structure_analyzer import (
    StructureAnalyzer,
    StrategyConfig,
    DEFAULT_STRATEGY,
    analyze_document_structure,
    analyze_content_structure,
)

from .strategy_executor import (
    StrategyExecutor,
    StrategyDefinition,
    Chunk,
    STRATEGIES,
    execute_strategy,
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
    # Parent chunk summarizer
    "ParentChunkSummarizer",
    "ParentChunkSummary",
    "get_parent_chunk_summarizer",
    "summarize_parent_chunk",
    "needs_parent_chunk_summary",
    "estimate_tokens",
    # V3.0: LLM-driven chunking
    "ContentSampler",
    "SampledContent",
    "sample_content",
    "StructureAnalyzer",
    "StrategyConfig",
    "DEFAULT_STRATEGY",
    "analyze_document_structure",
    "analyze_content_structure",
    "StrategyExecutor",
    "StrategyDefinition",
    "Chunk",
    "STRATEGIES",
    "execute_strategy",
]
