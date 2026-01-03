"""
Adaptive chunker that orchestrates domain-aware document processing.

This module provides the main entry point for intelligent document chunking,
combining document classification with strategy-based splitting.

V2.0 Enhancement: Tabular Skip Analysis
- After classification, if strategy is "table_preserving", analyze whether to skip row-level chunking
- For pure tabular documents (high table ratio, tabular document types), generate 1-3 summary chunks
- Hybrid documents (research papers with tables) continue with full chunking

V3.0 Enhancement: LLM-Driven Structure Analysis
- Replace pattern-based domain classification with LLM structure analysis
- Single LLM call per uploaded file to select optimal chunking strategy
- Predefined strategy library with 8 strategies
- Support for multi-page OCR (page-based sampling) and single-file conversion (char-based sampling)
"""

import logging
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from langchain_core.documents import Document

from .chunk_metadata import ChunkingProfile, UniversalChunkMetadata
from .document_classifier import DocumentClassifier, classify_document
from .chunking_strategies import (
    ChunkingStrategy,
    get_strategy_for_profile,
    get_strategy_by_name,
    SemanticHeaderStrategy,
)
from .domain_patterns import DocumentDomain, get_domain_for_document_type

# V3.0: LLM-driven chunking imports
from .structure_analyzer import (
    StructureAnalyzer,
    StrategyConfig,
    DEFAULT_STRATEGY,
    analyze_document_structure,
    analyze_content_structure,
)
from .strategy_executor import (
    StrategyExecutor,
    Chunk,
    execute_strategy,
    STRATEGIES,
)
from .content_sampler import ContentSampler, sample_content

# Import date normalization utilities
try:
    from ..utils.date_normalizer import find_and_normalize_dates, augment_text_with_dates
    DATE_NORMALIZER_AVAILABLE = True
except ImportError:
    DATE_NORMALIZER_AVAILABLE = False


logger = logging.getLogger(__name__)

# Pattern to detect base64 encoded content
BASE64_PATTERN = re.compile(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]{100,}')
BASE64_LIKE_PATTERN = re.compile(r'[A-Za-z0-9+/=]{200,}')

# ============================================================================
# TABULAR SKIP ANALYSIS CONFIGURATION (V2.0)
# ============================================================================

# Document types that strongly indicate tabular data - favor skipping row chunking
STRONG_TABULAR_TYPES = {
    'spreadsheet', 'invoice', 'receipt', 'bank_statement',
    'credit_card_statement', 'expense_report', 'inventory_report',
    'purchase_order', 'sales_order', 'payroll', 'financial_statement',
    'price_list', 'data_export', 'transaction_log', 'shipping_manifest'
}

# Document types that should NEVER skip chunking even with tables
NEVER_SKIP_TYPES = {
    'research_paper', 'academic_article', 'thesis', 'dissertation',
    'contract', 'agreement', 'legal_document', 'terms_of_service',
    'technical_spec', 'user_manual', 'documentation', 'tutorial',
    'news_article', 'blog_post', 'report'  # General reports have narrative
}

# Threshold for skip decision (score >= threshold -> skip)
TABULAR_SKIP_THRESHOLD = float(os.environ.get("TABULAR_SKIP_THRESHOLD", "0.6"))

# Enable/disable tabular skip feature
TABULAR_SKIP_ENABLED = os.environ.get("TABULAR_SKIP_ENABLED", "true").lower() == "true"

# V3.0: Enable/disable LLM-driven structure analysis
# When enabled, uses LLM to analyze document structure and select chunking strategy
# When disabled, uses the existing pattern-based domain classification
LLM_DRIVEN_CHUNKING_ENABLED = os.environ.get("LLM_DRIVEN_CHUNKING_ENABLED", "false").lower() == "true"


@dataclass
class AdaptiveChunkingResult:
    """Result of adaptive chunking operation."""
    chunks: List[Document]
    profile: ChunkingProfile
    file_summary: str = ""
    stats: Dict[str, Any] = None
    # V2.0: Skip info for tabular documents
    skip_info: Dict[str, Any] = field(default_factory=lambda: {
        "skip_row_chunking": False,
        "reason": "not_analyzed",
        "analysis": {}
    })

    def __post_init__(self):
        if self.stats is None:
            self.stats = {
                "total_chunks": len(self.chunks),
                "strategy_used": self.profile.recommended_strategy,
                "document_type": self.profile.document_type,
                "document_domain": self.profile.document_domain,
                "skip_row_chunking": self.skip_info.get("skip_row_chunking", False),
                "skip_reason": self.skip_info.get("reason", ""),
            }


class AdaptiveChunker:
    """
    Adaptive chunker that selects optimal strategy based on document analysis.

    This class orchestrates the entire chunking process:
    1. Pre-classification to determine document type and domain
    2. Strategy selection based on classification
    3. Content chunking with the selected strategy
    4. Post-processing (date normalization, metadata enhancement)
    """

    def __init__(
        self,
        use_llm_classification: bool = True,
        fallback_strategy: str = "semantic_header",
        default_chunk_size: int = 512,
        default_overlap: int = 51,
        use_llm_driven_chunking: Optional[bool] = None,
        llm_client=None,
    ):
        """
        Initialize the adaptive chunker.

        Args:
            use_llm_classification: Whether to use LLM for document classification (V2.0)
            fallback_strategy: Strategy to use if classification fails
            default_chunk_size: Default chunk size if not determined by profile
            default_overlap: Default overlap if not determined by profile
            use_llm_driven_chunking: Whether to use LLM-driven structure analysis (V3.0)
                                    If None, uses LLM_DRIVEN_CHUNKING_ENABLED env var
            llm_client: Optional LLM client for structure analysis
        """
        self.use_llm_classification = use_llm_classification
        self.fallback_strategy = fallback_strategy
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap

        # V3.0: LLM-driven chunking
        if use_llm_driven_chunking is None:
            self.use_llm_driven_chunking = LLM_DRIVEN_CHUNKING_ENABLED
        else:
            self.use_llm_driven_chunking = use_llm_driven_chunking

        self.llm_client = llm_client

        # Only instantiate components needed for the selected mode
        if self.use_llm_driven_chunking:
            # V3.0: LLM-driven chunking components
            self.structure_analyzer = StructureAnalyzer(llm_client=llm_client)
            self.strategy_executor = StrategyExecutor()
            self.content_sampler = ContentSampler()
            self.classifier = None  # Not used in V3.0
            logger.info("[Chunker] Initialized with V3.0 LLM-driven chunking mode")
        else:
            # V2.0: Pattern-based classification
            self.classifier = DocumentClassifier(use_llm=use_llm_classification)
            self.structure_analyzer = None
            self.strategy_executor = None
            self.content_sampler = None
            logger.info("[Chunker] Initialized with V2.0 pattern-based classification mode")

    def chunk_file(
        self,
        file_path: str,
        source_name: Optional[str] = None,
        force_strategy: Optional[str] = None,
        skip_classification: bool = False,
    ) -> AdaptiveChunkingResult:
        """
        Chunk a markdown file using adaptive strategy selection.

        Args:
            file_path: Path to the markdown file
            source_name: Name to use as source metadata (defaults to filename)
            force_strategy: Force a specific strategy (skip classification)
            skip_classification: Skip LLM classification, use rules only

        Returns:
            AdaptiveChunkingResult with chunks and metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return AdaptiveChunkingResult(
                chunks=[],
                profile=ChunkingProfile(
                    document_type="unknown",
                    document_domain="general",
                    recommended_strategy=self.fallback_strategy,
                ),
            )

        # Read file content
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return AdaptiveChunkingResult(
                chunks=[],
                profile=ChunkingProfile(
                    document_type="unknown",
                    document_domain="general",
                    recommended_strategy=self.fallback_strategy,
                ),
            )

        # Determine source name
        if source_name is None:
            source_name = file_path.stem

        return self.chunk_content(
            content=content,
            source_name=source_name,
            file_path=str(file_path),
            force_strategy=force_strategy,
            skip_classification=skip_classification,
        )

    def chunk_content(
        self,
        content: str,
        source_name: str,
        file_path: str = "",
        force_strategy: Optional[str] = None,
        skip_classification: bool = False,
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> AdaptiveChunkingResult:
        """
        Chunk content using adaptive strategy selection.

        Args:
            content: The text content to chunk
            source_name: Name to use as source metadata
            file_path: Optional file path for metadata
            force_strategy: Force a specific strategy
            skip_classification: Skip LLM classification, use rules only
            existing_metadata: Additional metadata to include in chunks

        Returns:
            AdaptiveChunkingResult with chunks and metadata
        """
        logger.info("=" * 80)
        logger.info("[Chunker] ========== ADAPTIVE CHUNKING START ==========")
        logger.info("=" * 80)
        logger.info(f"[Chunker] Source: {source_name}")
        logger.info(f"[Chunker] File Path: {file_path or 'N/A'}")
        logger.info(f"[Chunker] Content Length: {len(content) if content else 0} chars")
        logger.info(f"[Chunker] Force Strategy: {force_strategy or 'None'}")
        logger.info(f"[Chunker] Skip Classification: {skip_classification}")
        logger.info(f"[Chunker] LLM-Driven Chunking: {self.use_llm_driven_chunking}")
        logger.info("-" * 80)

        if not content or not content.strip():
            logger.warning(f"[Chunker] Empty content for {source_name} - returning empty result")
            logger.info("=" * 80)
            return AdaptiveChunkingResult(
                chunks=[],
                profile=ChunkingProfile(
                    document_type="unknown",
                    document_domain="general",
                    recommended_strategy=self.fallback_strategy,
                ),
            )

        # V3.0: Use LLM-driven chunking if enabled and not forcing a specific strategy
        if self.use_llm_driven_chunking and not force_strategy:
            logger.info("[Chunker] Using V3.0 LLM-driven structure analysis...")
            return self._chunk_content_llm_driven(
                content=content,
                source_name=source_name,
                file_path=file_path,
                existing_metadata=existing_metadata,
            )

        # Pre-process content
        logger.info("[Chunker] STEP 1: Pre-processing content...")
        original_length = len(content)
        content = self._preprocess_content(content)
        logger.info(f"[Chunker] Pre-processing: {original_length} -> {len(content)} chars")

        # Step 2: Classify document (unless forcing strategy)
        logger.info("-" * 80)
        logger.info("[Chunker] STEP 2: Document classification...")
        if force_strategy:
            logger.info(f"[Chunker] Using forced strategy: {force_strategy}")
            profile = ChunkingProfile(
                document_type="unknown",
                document_domain="general",
                recommended_strategy=force_strategy,
                recommended_chunk_size=self.default_chunk_size,
                recommended_overlap_percent=int(self.default_overlap / self.default_chunk_size * 100),
                confidence=1.0,
                reasoning=f"Forced strategy: {force_strategy}",
            )
            # Still detect markers
            profile.total_chars = len(content)
            profile.total_tokens = len(content) // 4
        else:
            use_llm = self.use_llm_classification and not skip_classification
            logger.info(f"[Chunker] Classification method: {'LLM' if use_llm else 'Rule-based'}")
            profile = self.classifier.classify(
                content=content,
                filename=source_name,
                use_llm=use_llm,
            )

        logger.info(f"[Chunker] Classification result:")
        logger.info(f"[Chunker]   - Document Type: {profile.document_type}")
        logger.info(f"[Chunker]   - Document Domain: {profile.document_domain}")
        logger.info(f"[Chunker]   - Recommended Strategy: {profile.recommended_strategy}")
        logger.info(f"[Chunker]   - Chunk Size: {profile.recommended_chunk_size}")
        logger.info(f"[Chunker]   - Overlap %: {profile.recommended_overlap_percent}")
        logger.info(f"[Chunker]   - Total Tokens: {profile.total_tokens}")
        logger.info(f"[Chunker]   - Confidence: {profile.confidence}")
        logger.info(f"[Chunker]   - Reasoning: {profile.reasoning}")

        # ========== V2.0: STEP 3 - TABULAR SKIP ANALYSIS ==========
        skip_info = {
            "skip_row_chunking": False,
            "reason": "not_analyzed",
            "analysis": {}
        }

        if TABULAR_SKIP_ENABLED and profile.recommended_strategy == "table_preserving":
            logger.info("-" * 80)
            logger.info("[Chunker] STEP 3: Tabular Skip Analysis...")
            logger.info(f"[Chunker] Strategy is 'table_preserving' - analyzing if row chunking should be skipped")

            should_skip, reason, analysis = self._analyze_tabular_skip(
                profile=profile,
                content=content,
                source_name=source_name
            )

            skip_info = {
                "skip_row_chunking": should_skip,
                "reason": reason,
                "analysis": analysis
            }

            logger.info(f"[Chunker] Skip Analysis Result:")
            logger.info(f"[Chunker]   - Should Skip Row Chunking: {should_skip}")
            logger.info(f"[Chunker]   - Reason: {reason}")
            logger.info(f"[Chunker]   - Skip Score: {analysis.get('skip_score', 0):.2f}")
            logger.info(f"[Chunker]   - Table Content Ratio: {analysis.get('table_content_ratio', 0):.2f}")
            logger.info(f"[Chunker]   - Row Count Estimate: {analysis.get('row_count_estimate', 0)}")

            if should_skip:
                # Generate summary chunks instead of row chunks
                logger.info("-" * 80)
                logger.info("[Chunker] STEP 3b: Generating Summary Chunks (skipping row chunking)...")

                chunks = self._generate_tabular_summary_chunks(
                    content=content,
                    source_name=source_name,
                    profile=profile,
                    analysis=analysis,
                    file_path=file_path,
                    existing_metadata=existing_metadata
                )

                logger.info(f"[Chunker] Generated {len(chunks)} summary chunks (row chunking skipped)")

                # Build stats and return early
                stats = {
                    "total_chunks": len(chunks),
                    "strategy_used": "tabular_summary",
                    "document_type": profile.document_type,
                    "document_domain": profile.document_domain,
                    "chunk_size_used": profile.recommended_chunk_size,
                    "classification_confidence": profile.confidence,
                    "is_atomic": True,  # Summary chunks are atomic
                    "skip_row_chunking": True,
                    "skip_reason": reason,
                }

                logger.info("=" * 80)
                logger.info("[Chunker] ========== ADAPTIVE CHUNKING COMPLETE (TABULAR SUMMARY) ==========")
                logger.info("=" * 80)
                logger.info(f"[Chunker] FINAL SUMMARY:")
                logger.info(f"[Chunker]   - Source: {source_name}")
                logger.info(f"[Chunker]   - Document Type: {profile.document_type}")
                logger.info(f"[Chunker]   - Processing: TABULAR SUMMARY (row chunking skipped)")
                logger.info(f"[Chunker]   - Total Chunks: {len(chunks)}")
                logger.info(f"[Chunker]   - Skip Reason: {reason}")
                logger.info("=" * 80)

                return AdaptiveChunkingResult(
                    chunks=chunks,
                    profile=profile,
                    stats=stats,
                    skip_info=skip_info,
                )
        else:
            if not TABULAR_SKIP_ENABLED:
                logger.info("-" * 80)
                logger.info("[Chunker] STEP 3: Tabular skip analysis disabled (TABULAR_SKIP_ENABLED=false)")
            elif profile.recommended_strategy != "table_preserving":
                logger.info("-" * 80)
                logger.info(f"[Chunker] STEP 3: Skipping tabular analysis (strategy is '{profile.recommended_strategy}', not 'table_preserving')")

        # Step 4: Get appropriate strategy (was Step 3)
        logger.info("-" * 80)
        logger.info("[Chunker] STEP 4: Selecting chunking strategy...")
        try:
            strategy = get_strategy_for_profile(profile)
            logger.info(f"[Chunker] Strategy selected: {strategy.strategy_name}")
        except Exception as e:
            logger.warning(f"[Chunker] Failed to get strategy for profile: {e}")
            logger.info(f"[Chunker] Using fallback strategy: {self.fallback_strategy}")
            strategy = get_strategy_by_name(
                self.fallback_strategy,
                chunk_size=self.default_chunk_size,
                chunk_overlap=self.default_overlap,
            )

        # Step 5: Apply date normalization if available (was Step 4)
        logger.info("-" * 80)
        logger.info("[Chunker] STEP 5: Date normalization...")
        if DATE_NORMALIZER_AVAILABLE:
            normalized_dates = find_and_normalize_dates(content)
            if normalized_dates:
                logger.info(f"[Chunker] Found {len(normalized_dates)} dates to normalize")
                content = augment_text_with_dates(content, normalized_dates)
            else:
                logger.info("[Chunker] No dates found to normalize")
        else:
            logger.info("[Chunker] Date normalizer not available")

        # Step 6: Chunk content (was Step 5)
        logger.info("-" * 80)
        logger.info("[Chunker] STEP 6: Executing chunking...")
        logger.info(f"[Chunker] Strategy: {strategy.strategy_name}")
        try:
            chunks = strategy.chunk(
                content=content,
                source_name=source_name,
                file_path=file_path,
                existing_metadata=existing_metadata,
            )
            logger.info(f"[Chunker] Chunking successful: {len(chunks)} chunks created")
        except Exception as e:
            logger.error(f"[Chunker] Chunking failed with {strategy.strategy_name}: {e}")
            logger.info("[Chunker] Falling back to SemanticHeaderStrategy...")
            fallback = SemanticHeaderStrategy(
                chunk_size=self.default_chunk_size,
                chunk_overlap=self.default_overlap,
            )
            chunks = fallback.chunk(content, source_name, file_path, existing_metadata)
            logger.info(f"[Chunker] Fallback chunking: {len(chunks)} chunks created")

        # Step 7: Post-process chunks (enhance metadata) (was Step 6)
        logger.info("-" * 80)
        logger.info("[Chunker] STEP 7: Post-processing chunks...")
        chunks = self._postprocess_chunks(chunks, profile, source_name)
        logger.info(f"[Chunker] Post-processing complete")

        # Build stats
        stats = {
            "total_chunks": len(chunks),
            "strategy_used": profile.recommended_strategy,
            "document_type": profile.document_type,
            "document_domain": profile.document_domain,
            "chunk_size_used": profile.recommended_chunk_size,
            "classification_confidence": profile.confidence,
            "is_atomic": profile.is_atomic_unit,
        }

        # Log chunk details
        if chunks:
            logger.info("-" * 80)
            logger.info("[Chunker] CHUNK DETAILS:")
            total_chars = sum(len(c.page_content) for c in chunks)
            avg_chars = total_chars // len(chunks) if chunks else 0
            logger.info(f"[Chunker]   - Total chunks: {len(chunks)}")
            logger.info(f"[Chunker]   - Total chars in chunks: {total_chars}")
            logger.info(f"[Chunker]   - Average chunk size: {avg_chars} chars")
            # Log first 3 chunk previews
            for i, chunk in enumerate(chunks[:3]):
                preview = chunk.page_content[:80].replace('\n', ' ')
                logger.info(f"[Chunker]   - Chunk {i+1}: '{preview}...'")
            if len(chunks) > 3:
                logger.info(f"[Chunker]   ... and {len(chunks) - 3} more chunks")

        logger.info("=" * 80)
        logger.info("[Chunker] ========== ADAPTIVE CHUNKING COMPLETE ==========")
        logger.info("=" * 80)
        logger.info(f"[Chunker] FINAL SUMMARY:")
        logger.info(f"[Chunker]   - Source: {source_name}")
        logger.info(f"[Chunker]   - Document Type: {profile.document_type}")
        logger.info(f"[Chunker]   - Strategy Used: {profile.recommended_strategy}")
        logger.info(f"[Chunker]   - Total Chunks: {len(chunks)}")
        logger.info(f"[Chunker]   - Chunk Size Setting: {profile.recommended_chunk_size}")
        logger.info(f"[Chunker]   - Is Atomic: {profile.is_atomic_unit}")
        logger.info(f"[Chunker]   - Row Chunking Skipped: {skip_info.get('skip_row_chunking', False)}")
        logger.info("=" * 80)

        return AdaptiveChunkingResult(
            chunks=chunks,
            profile=profile,
            stats=stats,
            skip_info=skip_info,
        )

    def _preprocess_content(self, content: str) -> str:
        """Pre-process content before chunking."""
        # Remove base64 images
        content = BASE64_PATTERN.sub('[image]', content)
        content = BASE64_LIKE_PATTERN.sub('', content)

        return content

    def _postprocess_chunks(
        self,
        chunks: List[Document],
        profile: ChunkingProfile,
        source_name: str,
    ) -> List[Document]:
        """Post-process chunks to enhance metadata."""
        for chunk in chunks:
            # Ensure profile info is in metadata
            chunk.metadata["document_type"] = profile.document_type
            chunk.metadata["document_domain"] = profile.document_domain
            chunk.metadata["content_density"] = profile.content_density
            chunk.metadata["structure_type"] = profile.structure_type

            # Add date metadata if dates were found
            if DATE_NORMALIZER_AVAILABLE:
                chunk_dates = find_and_normalize_dates(chunk.page_content)
                if chunk_dates:
                    chunk.metadata["dates_normalized"] = [d.normalized for d in chunk_dates]
                    chunk.metadata["dates_raw"] = [d.raw for d in chunk_dates]
                    # Add primary date
                    primary = chunk_dates[0]
                    chunk.metadata["primary_date"] = primary.normalized
                    chunk.metadata["date_year"] = primary.year
                    chunk.metadata["date_month"] = primary.month
                    chunk.metadata["date_day"] = primary.day

            # Calculate importance score if not set
            if "importance_score" not in chunk.metadata:
                chunk.metadata["importance_score"] = self._calculate_importance(
                    chunk, profile
                )

        return chunks

    def _calculate_importance(
        self,
        chunk: Document,
        profile: ChunkingProfile,
    ) -> float:
        """Calculate importance score for a chunk."""
        score = 0.5  # Base score

        # Key sections are more important
        if chunk.metadata.get("is_key_section"):
            score += 0.3

        # Abstract, summary, conclusion are important
        heading_path = chunk.metadata.get("heading_path", "").lower()
        important_sections = ["abstract", "summary", "conclusion", "key findings", "executive summary"]
        if any(section in heading_path for section in important_sections):
            score += 0.2

        # First chunk often contains important context
        if chunk.metadata.get("chunk_index", 0) == 0:
            score += 0.1

        # Domain-specific importance
        domain = profile.document_domain
        if domain == "legal" and chunk.metadata.get("is_definition"):
            score += 0.2
        if domain == "academic" and chunk.metadata.get("is_abstract"):
            score += 0.3
        if domain == "medical" and chunk.metadata.get("section_type") in ["assessment", "plan"]:
            score += 0.2

        return min(1.0, score)

    # ========================================================================
    # V3.0: LLM-DRIVEN CHUNKING METHODS
    # ========================================================================

    def _chunk_content_llm_driven(
        self,
        content: str,
        source_name: str,
        file_path: str = "",
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> AdaptiveChunkingResult:
        """
        Chunk content using LLM-driven structure analysis (V3.0).

        This method replaces the pattern-based domain classification with
        a single LLM call to analyze document structure and select the
        optimal chunking strategy.

        Args:
            content: The text content to chunk
            source_name: Name to use as source metadata
            file_path: Optional file path for metadata
            existing_metadata: Additional metadata to include in chunks

        Returns:
            AdaptiveChunkingResult with chunks and metadata
        """
        logger.info("-" * 80)
        logger.info("[Chunker V3.0] LLM-Driven Structure Analysis")
        logger.info("-" * 80)

        # Step 1: Pre-process content
        logger.info("[Chunker V3.0] STEP 1: Pre-processing content...")
        original_length = len(content)
        content = self._preprocess_content(content)
        logger.info(f"[Chunker V3.0] Pre-processing: {original_length} -> {len(content)} chars")

        # Step 2: Analyze document structure with LLM
        logger.info("-" * 80)
        logger.info("[Chunker V3.0] STEP 2: LLM Structure Analysis...")
        try:
            strategy_config = self.structure_analyzer.analyze_from_content(content)
            logger.info(f"[Chunker V3.0] LLM selected strategy: {strategy_config.selected_strategy}")
            logger.info(f"[Chunker V3.0]   - Chunk size: {strategy_config.chunk_size}")
            logger.info(f"[Chunker V3.0]   - Overlap: {strategy_config.overlap_percent}%")
            logger.info(f"[Chunker V3.0]   - Preserve: {strategy_config.preserve_elements}")
            logger.info(f"[Chunker V3.0]   - Reasoning: {strategy_config.reasoning}")
        except Exception as e:
            logger.error(f"[Chunker V3.0] LLM analysis failed: {e}")
            logger.info("[Chunker V3.0] Using default fallback strategy...")
            strategy_config = DEFAULT_STRATEGY

        # Step 3: Apply date normalization if available
        logger.info("-" * 80)
        logger.info("[Chunker V3.0] STEP 3: Date normalization...")
        if DATE_NORMALIZER_AVAILABLE:
            normalized_dates = find_and_normalize_dates(content)
            if normalized_dates:
                logger.info(f"[Chunker V3.0] Found {len(normalized_dates)} dates to normalize")
                content = augment_text_with_dates(content, normalized_dates)
            else:
                logger.info("[Chunker V3.0] No dates found to normalize")
        else:
            logger.info("[Chunker V3.0] Date normalizer not available")

        # Step 4: Execute chunking strategy
        logger.info("-" * 80)
        logger.info("[Chunker V3.0] STEP 4: Executing chunking strategy...")
        logger.info(f"[Chunker V3.0] Strategy: {strategy_config.selected_strategy}")

        try:
            raw_chunks = self.strategy_executor.execute(content, strategy_config)
            logger.info(f"[Chunker V3.0] Created {len(raw_chunks)} raw chunks")
        except Exception as e:
            logger.error(f"[Chunker V3.0] Strategy execution failed: {e}")
            # Fallback to paragraph-based chunking
            logger.info("[Chunker V3.0] Falling back to paragraph_based strategy...")
            fallback_config = StrategyConfig(
                selected_strategy="paragraph_based",
                chunk_size=self.default_chunk_size,
                overlap_percent=10,
                preserve_elements=["tables", "code_blocks", "lists"],
                reasoning="Fallback due to execution error"
            )
            raw_chunks = self.strategy_executor.execute(content, fallback_config)

        # Step 5: Convert raw chunks to Document objects
        logger.info("-" * 80)
        logger.info("[Chunker V3.0] STEP 5: Converting to Document objects...")
        documents = self._convert_chunks_to_documents(
            raw_chunks=raw_chunks,
            source_name=source_name,
            file_path=file_path,
            strategy_config=strategy_config,
            existing_metadata=existing_metadata,
        )

        # Step 6: Post-process chunks (add dates, importance scores)
        logger.info("-" * 80)
        logger.info("[Chunker V3.0] STEP 6: Post-processing chunks...")
        # Create a minimal profile for post-processing
        profile = ChunkingProfile(
            document_type="llm_analyzed",
            document_domain="general",
            recommended_strategy=strategy_config.selected_strategy,
            recommended_chunk_size=strategy_config.chunk_size,
            recommended_overlap_percent=strategy_config.overlap_percent,
            confidence=0.9,
            reasoning=strategy_config.reasoning,
            total_chars=len(content),
            total_tokens=len(content) // 4,
        )
        documents = self._postprocess_chunks(documents, profile, source_name)

        # Build stats
        stats = {
            "total_chunks": len(documents),
            "strategy_used": strategy_config.selected_strategy,
            "document_type": "llm_analyzed",
            "document_domain": "general",
            "chunk_size_used": strategy_config.chunk_size,
            "overlap_percent": strategy_config.overlap_percent,
            "preserve_elements": strategy_config.preserve_elements,
            "classification_method": "llm_structure_analysis",
            "reasoning": strategy_config.reasoning,
        }

        # Log final summary
        logger.info("=" * 80)
        logger.info("[Chunker V3.0] ========== LLM-DRIVEN CHUNKING COMPLETE ==========")
        logger.info("=" * 80)
        logger.info(f"[Chunker V3.0] FINAL SUMMARY:")
        logger.info(f"[Chunker V3.0]   - Source: {source_name}")
        logger.info(f"[Chunker V3.0]   - Strategy: {strategy_config.selected_strategy}")
        logger.info(f"[Chunker V3.0]   - Total Chunks: {len(documents)}")
        logger.info(f"[Chunker V3.0]   - Chunk Size: {strategy_config.chunk_size}")
        logger.info(f"[Chunker V3.0]   - Overlap: {strategy_config.overlap_percent}%")
        logger.info(f"[Chunker V3.0]   - Reasoning: {strategy_config.reasoning}")
        logger.info("=" * 80)

        return AdaptiveChunkingResult(
            chunks=documents,
            profile=profile,
            stats=stats,
        )

    def _convert_chunks_to_documents(
        self,
        raw_chunks: List[Chunk],
        source_name: str,
        file_path: str,
        strategy_config: StrategyConfig,
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Convert raw Chunk objects to LangChain Document objects.

        Args:
            raw_chunks: List of Chunk objects from strategy executor
            source_name: Name for the source
            file_path: Path to the source file
            strategy_config: Strategy configuration used
            existing_metadata: Additional metadata to include

        Returns:
            List of Document objects
        """
        documents = []
        total_chunks = len(raw_chunks)

        for chunk in raw_chunks:
            metadata = {
                # Source info
                "source": source_name,
                "file_path": file_path,
                # Chunk positioning
                "chunk_index": chunk.index,
                "total_chunks": total_chunks,
                "start_position": chunk.start_position,
                "end_position": chunk.end_position,
                "chunk_chars": len(chunk.content),
                # Strategy info
                "strategy_used": strategy_config.selected_strategy,
                "chunk_size_setting": strategy_config.chunk_size,
                "overlap_percent": strategy_config.overlap_percent,
                # V3.0 marker
                "chunking_version": "3.0",
                "classification_method": "llm_structure_analysis",
            }

            # Merge with chunk's own metadata
            if chunk.metadata:
                metadata.update(chunk.metadata)

            # Merge with existing metadata if provided
            if existing_metadata:
                metadata.update(existing_metadata)

            documents.append(Document(
                page_content=chunk.content,
                metadata=metadata,
            ))

        return documents

    def chunk_folder(
        self,
        output_folder: str,
        source_name: Optional[str] = None,
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> AdaptiveChunkingResult:
        """
        Chunk documents from an output folder using LLM-driven analysis.

        This method is designed for multi-page documents processed by OCR,
        where each page is a separate *_nohf.md file in the folder.

        Uses page-based sampling (first 3 + middle 2 + last 2 pages) for
        LLM structure analysis, then applies the selected strategy to
        all pages.

        NOTE: This is a V3.0-only method. If V3.0 components are not initialized,
        they will be lazily created.

        Args:
            output_folder: Path to folder containing *_nohf.md files
            source_name: Optional name for the source (defaults to folder name)
            existing_metadata: Additional metadata to include in chunks

        Returns:
            AdaptiveChunkingResult with chunks from all pages
        """
        # Ensure V3.0 components are available (lazy initialization)
        if self.content_sampler is None:
            self.content_sampler = ContentSampler()
        if self.structure_analyzer is None:
            self.structure_analyzer = StructureAnalyzer(llm_client=self.llm_client)
        if self.strategy_executor is None:
            self.strategy_executor = StrategyExecutor()

        logger.info("=" * 80)
        logger.info("[Chunker V3.0] ========== FOLDER CHUNKING START ==========")
        logger.info("=" * 80)
        logger.info(f"[Chunker V3.0] Folder: {output_folder}")

        # Determine source name
        if source_name is None:
            source_name = Path(output_folder).name

        # Detect scenario (multi-page or single-file)
        scenario, nohf_files = self.content_sampler.detect_scenario(output_folder)
        logger.info(f"[Chunker V3.0] Scenario: {scenario}")
        logger.info(f"[Chunker V3.0] Found {len(nohf_files)} *_nohf.md files")

        if not nohf_files:
            logger.warning(f"[Chunker V3.0] No files found in {output_folder}")
            return AdaptiveChunkingResult(
                chunks=[],
                profile=ChunkingProfile(
                    document_type="unknown",
                    document_domain="general",
                    recommended_strategy="paragraph_based",
                ),
            )

        # Step 1: Sample content for LLM analysis
        logger.info("-" * 80)
        logger.info("[Chunker V3.0] STEP 1: Sampling content for analysis...")
        sampled = self.content_sampler.sample_from_folder(output_folder)
        logger.info(f"[Chunker V3.0] Sampled {sampled.total_chars} chars from {sampled.total_pages} pages")

        # Step 2: Analyze structure with LLM
        logger.info("-" * 80)
        logger.info("[Chunker V3.0] STEP 2: LLM Structure Analysis...")
        try:
            strategy_config = self.structure_analyzer._analyze_with_llm(sampled)
            logger.info(f"[Chunker V3.0] LLM selected strategy: {strategy_config.selected_strategy}")
        except Exception as e:
            logger.error(f"[Chunker V3.0] LLM analysis failed: {e}")
            strategy_config = DEFAULT_STRATEGY

        # Step 3: Process each file with the selected strategy
        logger.info("-" * 80)
        logger.info("[Chunker V3.0] STEP 3: Processing files with selected strategy...")
        all_chunks = []

        for i, file_path in enumerate(nohf_files):
            logger.info(f"[Chunker V3.0] Processing file {i+1}/{len(nohf_files)}: {Path(file_path).name}")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                # Pre-process content
                file_content = self._preprocess_content(file_content)

                # Apply date normalization
                if DATE_NORMALIZER_AVAILABLE:
                    normalized_dates = find_and_normalize_dates(file_content)
                    if normalized_dates:
                        file_content = augment_text_with_dates(file_content, normalized_dates)

                # Execute strategy
                raw_chunks = self.strategy_executor.execute(file_content, strategy_config)

                # Convert to documents
                page_num = i + 1
                file_metadata = {
                    "page_number": page_num,
                    "file_name": Path(file_path).name,
                }
                if existing_metadata:
                    file_metadata.update(existing_metadata)

                documents = self._convert_chunks_to_documents(
                    raw_chunks=raw_chunks,
                    source_name=source_name,
                    file_path=file_path,
                    strategy_config=strategy_config,
                    existing_metadata=file_metadata,
                )

                # Update chunk indices to be global
                for doc in documents:
                    doc.metadata["page_chunk_index"] = doc.metadata["chunk_index"]
                    doc.metadata["chunk_index"] = len(all_chunks) + documents.index(doc)

                all_chunks.extend(documents)
                logger.info(f"[Chunker V3.0]   Created {len(documents)} chunks from page {page_num}")

            except Exception as e:
                logger.error(f"[Chunker V3.0] Error processing {file_path}: {e}")

        # Create profile
        profile = ChunkingProfile(
            document_type="llm_analyzed",
            document_domain="general",
            recommended_strategy=strategy_config.selected_strategy,
            recommended_chunk_size=strategy_config.chunk_size,
            recommended_overlap_percent=strategy_config.overlap_percent,
            confidence=0.9,
            reasoning=strategy_config.reasoning,
        )

        # Build stats
        stats = {
            "total_chunks": len(all_chunks),
            "total_pages": len(nohf_files),
            "scenario": scenario,
            "strategy_used": strategy_config.selected_strategy,
            "chunk_size_used": strategy_config.chunk_size,
            "classification_method": "llm_structure_analysis",
        }

        logger.info("=" * 80)
        logger.info("[Chunker V3.0] ========== FOLDER CHUNKING COMPLETE ==========")
        logger.info("=" * 80)
        logger.info(f"[Chunker V3.0] FINAL SUMMARY:")
        logger.info(f"[Chunker V3.0]   - Source: {source_name}")
        logger.info(f"[Chunker V3.0]   - Total Pages: {len(nohf_files)}")
        logger.info(f"[Chunker V3.0]   - Total Chunks: {len(all_chunks)}")
        logger.info(f"[Chunker V3.0]   - Strategy: {strategy_config.selected_strategy}")
        logger.info("=" * 80)

        return AdaptiveChunkingResult(
            chunks=all_chunks,
            profile=profile,
            stats=stats,
        )

    # ========================================================================
    # V2.0: TABULAR SKIP ANALYSIS METHODS
    # ========================================================================

    def _analyze_tabular_skip(
        self,
        profile: ChunkingProfile,
        content: str,
        source_name: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Analyze whether to skip row-level chunking for tabular documents.

        This function is called AFTER document classification, when we already
        know the recommended_strategy is "table_preserving".

        Returns:
            Tuple of (should_skip, reason, analysis_details)
        """
        document_type = profile.document_type or ""
        doc_type_lower = document_type.lower()

        analysis = {
            "document_type": document_type,
            "table_content_ratio": 0.0,
            "row_count_estimate": 0,
            "avg_row_length": 0.0,
            "narrative_ratio": 0.0,
            "skip_score": 0.0,
            "doc_type_score": 0.0,
            "table_score": 0.0,
            "row_score": 0.0,
        }

        # ========== CHECK 1: Never-skip document types ==========
        if doc_type_lower in NEVER_SKIP_TYPES:
            logger.info(f"[Tabular] Document type '{doc_type_lower}' is in NEVER_SKIP_TYPES - will not skip")
            return False, f"document_type_excluded:{doc_type_lower}", analysis

        # ========== FACTOR 1: Document Type Score (Weight: 40%) ==========
        if doc_type_lower in STRONG_TABULAR_TYPES:
            doc_type_score = 0.4
            logger.info(f"[Tabular] Document type '{doc_type_lower}' in STRONG_TABULAR_TYPES: score=0.4")
        else:
            doc_type_score = 0.1
            logger.info(f"[Tabular] Document type '{doc_type_lower}' not in STRONG_TABULAR_TYPES: score=0.1")

        analysis["doc_type_score"] = doc_type_score

        # ========== FACTOR 2: Table Content Ratio (Weight: 35%) ==========
        table_ratio = self._calculate_table_content_ratio(content)
        analysis["table_content_ratio"] = table_ratio
        analysis["narrative_ratio"] = 1.0 - table_ratio

        if table_ratio < 0.5:
            table_score = 0.0
            logger.info(f"[Tabular] Table ratio {table_ratio:.2f} < 0.5: score=0.0 (hybrid document)")
        elif table_ratio < 0.7:
            table_score = 0.15
            logger.info(f"[Tabular] Table ratio {table_ratio:.2f} in [0.5, 0.7): score=0.15")
        elif table_ratio < 0.85:
            table_score = 0.25
            logger.info(f"[Tabular] Table ratio {table_ratio:.2f} in [0.7, 0.85): score=0.25")
        else:
            table_score = 0.35
            logger.info(f"[Tabular] Table ratio {table_ratio:.2f} >= 0.85: score=0.35")

        analysis["table_score"] = table_score

        # ========== FACTOR 3: Row Density Analysis (Weight: 25%) ==========
        row_count, avg_row_length = self._analyze_table_rows(content)
        analysis["row_count_estimate"] = row_count
        analysis["avg_row_length"] = avg_row_length

        # Many short rows = typical dataset (high skip benefit)
        # Few long rows = formatted tables in document (low skip benefit)
        if row_count > 50 and avg_row_length < 200:
            row_score = 0.25
            logger.info(f"[Tabular] Row count {row_count} > 50, avg_length {avg_row_length:.0f} < 200: score=0.25 (dataset)")
        elif row_count > 20 and avg_row_length < 300:
            row_score = 0.15
            logger.info(f"[Tabular] Row count {row_count} > 20, avg_length {avg_row_length:.0f} < 300: score=0.15")
        elif row_count > 10:
            row_score = 0.1
            logger.info(f"[Tabular] Row count {row_count} > 10: score=0.1")
        else:
            row_score = 0.0
            logger.info(f"[Tabular] Row count {row_count} <= 10: score=0.0 (too few rows)")

        analysis["row_score"] = row_score

        # ========== CALCULATE FINAL SCORE ==========
        total_score = doc_type_score + table_score + row_score
        analysis["skip_score"] = total_score

        should_skip = total_score >= TABULAR_SKIP_THRESHOLD

        reason = f"score:{total_score:.2f}|doc_type:{doc_type_score:.2f}|table:{table_score:.2f}|rows:{row_score:.2f}"

        logger.info(f"[Tabular] Final score: {total_score:.2f} (threshold: {TABULAR_SKIP_THRESHOLD})")
        logger.info(f"[Tabular] Decision: {'SKIP row chunking' if should_skip else 'KEEP row chunking'}")

        return should_skip, reason, analysis

    def _calculate_table_content_ratio(self, content: str) -> float:
        """Calculate what percentage of content is markdown tables."""
        if not content:
            return 0.0

        lines = content.split('\n')
        total_chars = len(content)
        table_chars = 0

        for line in lines:
            # Markdown table line: | col1 | col2 | col3 |
            if '|' in line and line.strip().startswith('|'):
                table_chars += len(line)
            # Table separator: |---|---|---|
            elif '|' in line and '-' in line and line.count('-') > 5:
                table_chars += len(line)

        return table_chars / max(total_chars, 1)

    def _analyze_table_rows(self, content: str) -> Tuple[int, float]:
        """Count table rows and calculate average row length."""
        lines = content.split('\n')
        table_rows = []

        for line in lines:
            if '|' in line and line.strip().startswith('|'):
                # Skip separator lines (|---|---|---|)
                if not (line.count('-') > 5 and '|' in line):
                    table_rows.append(line)

        if not table_rows:
            return 0, 0.0

        avg_length = sum(len(row) for row in table_rows) / len(table_rows)
        return len(table_rows), avg_length

    def _generate_tabular_summary_chunks(
        self,
        content: str,
        source_name: str,
        profile: ChunkingProfile,
        analysis: Dict[str, Any],
        file_path: str = "",
        existing_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Generate a single comprehensive summary chunk for tabular document discovery.

        Instead of creating 100+ row chunks, create ONE strategic summary chunk
        that combines document overview, schema info, and sample data context
        for efficient vector search without duplicate results.

        Benefits:
        - Single chunk = no duplicate search results
        - All relevant info in one place for RAG
        - Simpler architecture
        """
        chunks = []

        # Extract table structure info
        headers = self._extract_table_headers(content)
        row_count = analysis.get("row_count_estimate", 0)
        table_ratio = analysis.get("table_content_ratio", 0)

        # Metadata for the comprehensive chunk
        base_metadata = {
            "source": source_name,
            "file_path": file_path,
            # Classification metadata (important for consistency)
            "document_type": profile.document_type,
            "document_domain": profile.document_domain,
            # Tabular-specific metadata
            "chunk_type": "tabular_summary",
            "is_tabular": True,
            "is_summary_chunk": True,
            "skip_row_chunking": True,
            "schema_type": "spreadsheet",
            "row_count": row_count,
            "column_count": len(headers),
            "columns": headers[:20] if headers else [],  # Limit to first 20 columns
            "table_content_ratio": table_ratio,
            # Processing info
            "recommended_strategy": profile.recommended_strategy,
            "skip_score": analysis.get("skip_score", 0),
            "skip_reason": f"score:{analysis.get('skip_score', 0):.2f}",
            "chunk_index": 0,
        }

        # Merge with existing metadata if provided
        if existing_metadata:
            base_metadata.update(existing_metadata)

        # ===== SINGLE COMPREHENSIVE CHUNK =====
        # Combines document summary, schema info, and sample data into one chunk
        comprehensive_text = self._generate_comprehensive_summary_text(
            source_name=source_name,
            headers=headers,
            row_count=row_count,
            content=content,
            profile=profile
        )

        chunks.append(Document(
            page_content=comprehensive_text,
            metadata=base_metadata
        ))
        logger.info(f"[Tabular] Generated comprehensive summary chunk: {len(comprehensive_text)} chars")

        return chunks

    def _extract_table_headers(self, content: str) -> List[str]:
        """Extract column headers from markdown table."""
        lines = content.split('\n')
        headers = []

        for line in lines:
            # Look for header row (first table row before separator)
            if '|' in line and line.strip().startswith('|'):
                # Skip separator lines
                if line.count('-') > 5:
                    continue
                # Parse headers
                parts = [p.strip() for p in line.split('|')]
                parts = [p for p in parts if p]  # Remove empty parts
                if parts and not headers:  # First non-separator row
                    headers = parts
                    break

        return headers

    def _generate_document_summary_text(
        self,
        source_name: str,
        headers: List[str],
        row_count: int,
        content: str,
        profile: ChunkingProfile
    ) -> str:
        """Generate a natural language summary of the tabular document."""
        doc_type = profile.document_type or "tabular data"

        # Build summary
        summary_parts = [
            f"# {source_name}",
            f"",
            f"This is a {doc_type} document containing {row_count} data rows.",
        ]

        if headers:
            summary_parts.append(f"")
            summary_parts.append(f"## Columns ({len(headers)} fields)")
            summary_parts.append(f"The data includes the following fields: {', '.join(headers[:15])}")
            if len(headers) > 15:
                summary_parts.append(f"...and {len(headers) - 15} more columns.")

        # Try to infer content type from headers/source name
        source_lower = source_name.lower()
        if any(kw in source_lower for kw in ["sales", "revenue", "transaction", "order"]):
            summary_parts.append(f"")
            summary_parts.append("This appears to be sales/transaction data suitable for revenue analysis.")
        elif any(kw in source_lower for kw in ["inventory", "stock", "product", "sku"]):
            summary_parts.append(f"")
            summary_parts.append("This appears to be inventory/product data suitable for stock analysis.")
        elif any(kw in source_lower for kw in ["employee", "hr", "payroll", "staff"]):
            summary_parts.append(f"")
            summary_parts.append("This appears to be HR/employee data.")
        elif any(kw in source_lower for kw in ["invoice", "receipt", "billing"]):
            summary_parts.append(f"")
            summary_parts.append("This appears to be financial/billing data.")

        return '\n'.join(summary_parts)

    def _generate_schema_description_text(
        self,
        headers: List[str],
        row_count: int,
        content: str
    ) -> str:
        """Generate a schema description for the tabular data."""
        schema_parts = [
            f"## Data Schema",
            f"",
            f"Tabular dataset with {len(headers)} columns and {row_count} rows.",
            f"",
            f"### Column Names",
        ]

        for i, header in enumerate(headers[:20], 1):
            schema_parts.append(f"{i}. {header}")

        if len(headers) > 20:
            schema_parts.append(f"...and {len(headers) - 20} more columns")

        # Try to identify field types from header names
        amount_fields = [h for h in headers if any(kw in h.lower() for kw in
            ["amount", "price", "cost", "total", "sum", "revenue", "value", "qty", "quantity"])]
        date_fields = [h for h in headers if any(kw in h.lower() for kw in
            ["date", "time", "created", "updated", "timestamp", "day", "month", "year"])]
        category_fields = [h for h in headers if any(kw in h.lower() for kw in
            ["category", "type", "status", "group", "class", "department", "region"])]

        if amount_fields or date_fields or category_fields:
            schema_parts.append(f"")
            schema_parts.append(f"### Field Types (inferred)")

        if amount_fields:
            schema_parts.append(f"- **Numeric/Amount fields** (for aggregation): {', '.join(amount_fields[:5])}")
        if date_fields:
            schema_parts.append(f"- **Date fields** (for time-based analysis): {', '.join(date_fields[:5])}")
        if category_fields:
            schema_parts.append(f"- **Category fields** (for grouping): {', '.join(category_fields[:5])}")

        return '\n'.join(schema_parts)

    def _generate_sample_data_text(self, content: str, headers: List[str]) -> str:
        """Generate sample data context from first few rows."""
        lines = content.split('\n')
        sample_rows = []
        found_header = False

        for line in lines:
            if '|' in line and line.strip().startswith('|'):
                # Skip separator lines
                if line.count('-') > 5:
                    found_header = True
                    continue
                if found_header and len(sample_rows) < 5:
                    sample_rows.append(line)

        if not sample_rows:
            return ""

        sample_parts = [
            f"## Sample Data",
            f"",
            f"First {len(sample_rows)} rows of data:",
            f"",
        ]

        # Add header row if we have headers
        if headers:
            sample_parts.append("| " + " | ".join(headers[:10]) + " |")
            sample_parts.append("|" + "|".join(["---"] * min(len(headers), 10)) + "|")

        for row in sample_rows[:5]:
            # Truncate very long rows
            if len(row) > 200:
                row = row[:200] + "..."
            sample_parts.append(row)

        return '\n'.join(sample_parts)

    def _generate_comprehensive_summary_text(
        self,
        source_name: str,
        headers: List[str],
        row_count: int,
        content: str,
        profile: ChunkingProfile
    ) -> str:
        """
        Generate a single comprehensive summary combining document overview,
        schema info, and sample data context.

        This replaces the previous multi-chunk approach with a single unified text
        that captures all relevant information for vector search.
        """
        doc_type = profile.document_type or "tabular data"
        parts = []

        # ===== SECTION 1: Document Overview =====
        parts.append(f"# {source_name}")
        parts.append("")
        parts.append(f"This is a {doc_type} document containing {row_count} data rows with {len(headers)} columns.")
        parts.append("")

        # Try to infer content type from headers/source name
        source_lower = source_name.lower()
        if any(kw in source_lower for kw in ["sales", "revenue", "transaction", "order"]):
            parts.append("This appears to be sales/transaction data suitable for revenue analysis.")
        elif any(kw in source_lower for kw in ["inventory", "stock", "product", "sku"]):
            parts.append("This appears to be inventory/product data suitable for stock analysis.")
        elif any(kw in source_lower for kw in ["employee", "hr", "payroll", "staff"]):
            parts.append("This appears to be HR/employee data.")
        elif any(kw in source_lower for kw in ["invoice", "receipt", "billing"]):
            parts.append("This appears to be financial/billing data.")
        parts.append("")

        # ===== SECTION 2: Schema/Column Information =====
        if headers:
            parts.append(f"## Data Schema ({len(headers)} columns)")
            parts.append("")
            parts.append(f"Column names: {', '.join(headers[:15])}")
            if len(headers) > 15:
                parts.append(f"...and {len(headers) - 15} more columns.")
            parts.append("")

            # Try to identify field types from header names
            amount_fields = [h for h in headers if any(kw in h.lower() for kw in
                ["amount", "price", "cost", "total", "sum", "revenue", "value", "qty", "quantity"])]
            date_fields = [h for h in headers if any(kw in h.lower() for kw in
                ["date", "time", "created", "updated", "timestamp", "day", "month", "year"])]
            category_fields = [h for h in headers if any(kw in h.lower() for kw in
                ["category", "type", "status", "group", "class", "department", "region"])]

            if amount_fields or date_fields or category_fields:
                parts.append("### Field Types (inferred)")
                if amount_fields:
                    parts.append(f"- Numeric/Amount fields (for aggregation): {', '.join(amount_fields[:5])}")
                if date_fields:
                    parts.append(f"- Date fields (for time-based analysis): {', '.join(date_fields[:5])}")
                if category_fields:
                    parts.append(f"- Category fields (for grouping): {', '.join(category_fields[:5])}")
                parts.append("")

        # ===== SECTION 3: Sample Data Context =====
        sample_text = self._generate_sample_data_text(content, headers)
        if sample_text:
            parts.append(sample_text)

        return '\n'.join(parts)


def chunk_document_adaptive(
    content: str,
    source_name: str,
    file_path: str = "",
    use_llm: bool = True,
    force_strategy: Optional[str] = None,
) -> List[Document]:
    """
    Convenience function to chunk a document with adaptive strategy.

    Args:
        content: The document content
        source_name: Name for the source
        file_path: Optional file path
        use_llm: Whether to use LLM for classification
        force_strategy: Optional strategy to force

    Returns:
        List of Document chunks
    """
    chunker = AdaptiveChunker(use_llm_classification=use_llm)
    result = chunker.chunk_content(
        content=content,
        source_name=source_name,
        file_path=file_path,
        force_strategy=force_strategy,
    )
    return result.chunks


def chunk_file_adaptive(
    file_path: str,
    source_name: Optional[str] = None,
    use_llm: bool = True,
    force_strategy: Optional[str] = None,
) -> AdaptiveChunkingResult:
    """
    Convenience function to chunk a file with adaptive strategy.

    Args:
        file_path: Path to the file
        source_name: Optional source name
        use_llm: Whether to use LLM for classification
        force_strategy: Optional strategy to force

    Returns:
        AdaptiveChunkingResult
    """
    chunker = AdaptiveChunker(use_llm_classification=use_llm)
    return chunker.chunk_file(
        file_path=file_path,
        source_name=source_name,
        force_strategy=force_strategy,
    )
