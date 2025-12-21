"""
Adaptive chunker that orchestrates domain-aware document processing.

This module provides the main entry point for intelligent document chunking,
combining document classification with strategy-based splitting.
"""

import logging
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

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


@dataclass
class AdaptiveChunkingResult:
    """Result of adaptive chunking operation."""
    chunks: List[Document]
    profile: ChunkingProfile
    file_summary: str = ""
    stats: Dict[str, Any] = None

    def __post_init__(self):
        if self.stats is None:
            self.stats = {
                "total_chunks": len(self.chunks),
                "strategy_used": self.profile.recommended_strategy,
                "document_type": self.profile.document_type,
                "document_domain": self.profile.document_domain,
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
    ):
        """
        Initialize the adaptive chunker.

        Args:
            use_llm_classification: Whether to use LLM for document classification
            fallback_strategy: Strategy to use if classification fails
            default_chunk_size: Default chunk size if not determined by profile
            default_overlap: Default overlap if not determined by profile
        """
        self.use_llm_classification = use_llm_classification
        self.fallback_strategy = fallback_strategy
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
        self.classifier = DocumentClassifier(use_llm=use_llm_classification)

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
        if not content or not content.strip():
            logger.warning(f"Empty content for {source_name}")
            return AdaptiveChunkingResult(
                chunks=[],
                profile=ChunkingProfile(
                    document_type="unknown",
                    document_domain="general",
                    recommended_strategy=self.fallback_strategy,
                ),
            )

        # Pre-process content
        content = self._preprocess_content(content)

        # Step 1: Classify document (unless forcing strategy)
        if force_strategy:
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
            profile = self.classifier.classify(
                content=content,
                filename=source_name,
                use_llm=use_llm,
            )

        logger.info(
            f"[AdaptiveChunker] Document: {source_name} | "
            f"Type: {profile.document_type} | Domain: {profile.document_domain} | "
            f"Strategy: {profile.recommended_strategy} | "
            f"Size: {profile.total_tokens} tokens"
        )

        # Step 2: Get appropriate strategy
        try:
            strategy = get_strategy_for_profile(profile)
        except Exception as e:
            logger.warning(f"Failed to get strategy for profile, using fallback: {e}")
            strategy = get_strategy_by_name(
                self.fallback_strategy,
                chunk_size=self.default_chunk_size,
                chunk_overlap=self.default_overlap,
            )

        # Step 3: Apply date normalization if available
        if DATE_NORMALIZER_AVAILABLE:
            normalized_dates = find_and_normalize_dates(content)
            if normalized_dates:
                logger.info(f"[AdaptiveChunker] Found {len(normalized_dates)} dates in {source_name}")
                content = augment_text_with_dates(content, normalized_dates)

        # Step 4: Chunk content
        try:
            chunks = strategy.chunk(
                content=content,
                source_name=source_name,
                file_path=file_path,
                existing_metadata=existing_metadata,
            )
        except Exception as e:
            logger.error(f"Chunking failed with {strategy.strategy_name}, using fallback: {e}")
            fallback = SemanticHeaderStrategy(
                chunk_size=self.default_chunk_size,
                chunk_overlap=self.default_overlap,
            )
            chunks = fallback.chunk(content, source_name, file_path, existing_metadata)

        # Step 5: Post-process chunks (enhance metadata)
        chunks = self._postprocess_chunks(chunks, profile, source_name)

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

        logger.info(
            f"[AdaptiveChunker] Completed: {source_name} -> {len(chunks)} chunks "
            f"using {profile.recommended_strategy}"
        )

        return AdaptiveChunkingResult(
            chunks=chunks,
            profile=profile,
            stats=stats,
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
