"""
Document classifier for pre-chunking analysis.

This module provides LLM-based and rule-based document classification
to determine the optimal chunking strategy before processing.

Uses the centralized DocumentTypeClassifier for consistent document
type classification across extraction and chunking services.

NOTE (V3.0): This module is part of the V2.0 pattern-based classification system.
When LLM_DRIVEN_CHUNKING_ENABLED=true, the AdaptiveChunker bypasses this module
and uses the new structure_analyzer.py for LLM-driven strategy selection.

The V3.0 approach:
- Uses a single LLM call per file to analyze document structure
- Selects from 8 predefined chunking strategies
- Avoids the fragile regex patterns that cause misclassifications

To enable V3.0:
    export LLM_DRIVEN_CHUNKING_ENABLED=true

Or programmatically:
    chunker = AdaptiveChunker(use_llm_driven_chunking=True)
"""

import logging
import re
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .chunk_metadata import ChunkingProfile
from .domain_patterns import (
    DocumentDomain,
    DocumentType,
    DomainConfig,
    DOMAIN_CONFIGS,
    get_domain_config,
    get_domain_for_document_type,
    detect_domain_from_content,
)
from .document_types import generate_classification_taxonomy

# Import centralized document type classifier
from common.document_type_classifier import (
    DocumentTypeClassifier as CentralizedClassifier,
    DocumentDomain as CentralDomain,
)

logger = logging.getLogger(__name__)

# Token estimation: ~4 chars per token
CHARS_PER_TOKEN = 4

# Size thresholds
SMALL_DOC_TOKENS = 1000   # < 1000 tokens = small document
LARGE_DOC_TOKENS = 10000  # > 10000 tokens = large document


# ============================================================================
# Pre-Classification Prompt
# ============================================================================

UNIVERSAL_PRE_CHUNKING_PROMPT_TEMPLATE = """You are a document analysis expert. Analyze this document to determine optimal chunking strategy.

DOCUMENT NAME: {{filename}}
DOCUMENT SIZE: {{total_chars}} characters (~{{estimated_tokens}} tokens)

DOCUMENT PREVIEW (first 2000 chars):
{{preview_start}}

DOCUMENT END (last 1000 chars):
{{preview_end}}

STRUCTURAL MARKERS DETECTED:
- Tables: {{has_tables}}
- Headers: {{has_headers}}
- Lists: {{has_lists}}
- Code blocks: {{has_code}}
- Equations: {{has_equations}}
- Numbered clauses: {{has_numbered_clauses}}
- Citations: {{has_citations}}
- Form fields: {{has_form_fields}}

Analyze and respond with JSON only (no markdown, no code blocks):
{{{{
  "document_type": "<type from taxonomy below>",
  "document_domain": "legal|academic|medical|engineering|education|financial|government|professional|general",
  "content_density": "sparse|normal|dense",
  "structure_type": "tabular|hierarchical|narrative|clause_based|form_based|mixed",
  "complexity": "simple|moderate|complex",
  "is_atomic_unit": true or false,
  "recommended_strategy": "whole_document|clause_preserving|academic_structure|medical_section|requirement_based|educational_unit|table_preserving|semantic_header|paragraph",
  "recommended_chunk_size": <256-2048>,
  "recommended_overlap_percent": <0-25>,
  "preserve_elements": ["list of elements that must not be split"],
  "special_handling": ["list of special processing needed"],
  "confidence": <0.0-1.0>,
  "reasoning": "Brief explanation"
}}}}

{taxonomy}

STRATEGY GUIDELINES:
- "whole_document": Small atomic documents (<1000 tokens) that should stay together
- "clause_preserving": Legal documents with numbered sections/clauses
- "academic_structure": Research papers with citations/equations/abstract
- "medical_section": Clinical notes with SOAP or similar structure
- "requirement_based": Technical specs with requirement IDs
- "educational_unit": Learning materials with objectives/exercises
- "table_preserving": Documents with significant tabular data
- "semantic_header": Well-structured documents with clear headers
- "paragraph": Narrative documents without strong structure"""


def get_pre_chunking_prompt() -> str:
    """
    Get the pre-chunking prompt with centralized document type taxonomy.
    This ensures consistency between classification and query enhancement.
    """
    taxonomy = generate_classification_taxonomy()
    return UNIVERSAL_PRE_CHUNKING_PROMPT_TEMPLATE.format(taxonomy=taxonomy)


# Generate the prompt once at module load time for efficiency
UNIVERSAL_PRE_CHUNKING_PROMPT = get_pre_chunking_prompt()


class DocumentClassifier:
    """
    Classifies documents to determine optimal chunking strategy.

    Supports both LLM-based and rule-based classification.
    Uses the centralized DocumentTypeClassifier for consistent
    document type classification.
    """

    def __init__(self, use_llm: bool = True, llm_service=None):
        """
        Initialize the classifier.

        Args:
            use_llm: Whether to use LLM for classification (falls back to rules if False)
            llm_service: Optional LLM service instance
        """
        self.use_llm = use_llm
        self.llm_service = llm_service
        self._llm = None
        self._centralized_classifier = None

    def _get_llm(self):
        """Get or initialize LLM instance."""
        if self._llm is None and self.use_llm:
            try:
                if self.llm_service is None:
                    from ..llm_service import get_llm_service
                    self.llm_service = get_llm_service()
                # Use a fast model for classification
                self._llm = self.llm_service.get_chat_model(temperature=0.1, num_ctx=4096)
            except Exception as e:
                logger.warning(f"Could not initialize LLM for classification: {e}")
                self.use_llm = False
        return self._llm

    def _get_centralized_classifier(self) -> CentralizedClassifier:
        """Get or initialize the centralized document type classifier."""
        if self._centralized_classifier is None:
            # Create LLM client wrapper for centralized classifier
            llm_client = None
            if self.use_llm:
                llm = self._get_llm()
                if llm:
                    class LLMClientWrapper:
                        def __init__(self, llm):
                            self.llm = llm
                        def generate(self, prompt: str) -> str:
                            from langchain_core.messages import HumanMessage
                            response = self.llm.invoke([HumanMessage(content=prompt)])
                            return response.content
                    llm_client = LLMClientWrapper(llm)

            self._centralized_classifier = CentralizedClassifier(llm_client=llm_client)
        return self._centralized_classifier

    def classify(
        self,
        content: str,
        filename: str = "",
        use_llm: Optional[bool] = None,
    ) -> ChunkingProfile:
        """
        Classify a document and return a ChunkingProfile.

        Args:
            content: The document content
            filename: The document filename
            use_llm: Override instance-level LLM setting

        Returns:
            ChunkingProfile with classification results
        """
        should_use_llm = use_llm if use_llm is not None else self.use_llm

        # First, detect structural markers (rule-based, always done)
        markers = self._detect_structural_markers(content)

        # Calculate size metrics
        total_chars = len(content)
        estimated_tokens = total_chars // CHARS_PER_TOKEN
        is_small = estimated_tokens < SMALL_DOC_TOKENS
        is_large = estimated_tokens > LARGE_DOC_TOKENS

        # Try LLM classification
        if should_use_llm:
            try:
                profile = self._classify_with_llm(
                    content=content,
                    filename=filename,
                    markers=markers,
                    total_chars=total_chars,
                    estimated_tokens=estimated_tokens,
                )
                if profile:
                    # Enhance with size info
                    profile.total_chars = total_chars
                    profile.total_tokens = estimated_tokens
                    profile.is_small_doc = is_small
                    profile.is_large_doc = is_large
                    return profile
            except Exception as e:
                logger.warning(f"LLM classification failed, falling back to rules: {e}")

        # Fall back to rule-based classification
        return self._classify_with_rules(
            content=content,
            filename=filename,
            markers=markers,
            total_chars=total_chars,
            estimated_tokens=estimated_tokens,
            is_small=is_small,
            is_large=is_large,
        )

    def _detect_structural_markers(self, content: str) -> Dict[str, bool]:
        """Detect structural markers in the content."""
        # Sample content for faster detection
        sample = content[:10000] if len(content) > 10000 else content

        return {
            "has_tables": bool(re.search(r"<table>|^\|.+\|.+\|$", sample, re.MULTILINE | re.IGNORECASE)),
            "has_headers": bool(re.search(r"^#{1,6}\s+", sample, re.MULTILINE)),
            "has_lists": bool(re.search(r"^\s*[-*•]\s+|^\s*\d+\.\s+", sample, re.MULTILINE)),
            "has_code": bool(re.search(r"```|<code>|<pre>", sample)),
            "has_equations": bool(re.search(r"\$\$|\\\[|\\begin\{equation\}", sample)),
            "has_numbered_clauses": bool(re.search(r"^\s*\d+\.\d+[\.\d]*\s+", sample, re.MULTILINE)),
            "has_citations": bool(re.search(r"\[\d+\]|\([A-Z][a-z]+,?\s*\d{4}\)", sample)),
            "has_form_fields": bool(re.search(r":\s*_{3,}|\[\s*\]", sample)),
            "has_dates": bool(re.search(r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}", sample)),
            "has_amounts": bool(re.search(r"[\$€£¥]\s*[\d,]+\.?\d*", sample)),
        }

    def _classify_with_llm(
        self,
        content: str,
        filename: str,
        markers: Dict[str, bool],
        total_chars: int,
        estimated_tokens: int,
    ) -> Optional[ChunkingProfile]:
        """Classify using LLM."""
        llm = self._get_llm()
        if llm is None:
            return None

        # Prepare preview
        preview_start = content[:2000]
        preview_end = content[-1000:] if len(content) > 1000 else ""

        prompt = UNIVERSAL_PRE_CHUNKING_PROMPT.format(
            filename=filename,
            total_chars=total_chars,
            estimated_tokens=estimated_tokens,
            preview_start=preview_start,
            preview_end=preview_end,
            has_tables=markers["has_tables"],
            has_headers=markers["has_headers"],
            has_lists=markers["has_lists"],
            has_code=markers["has_code"],
            has_equations=markers["has_equations"],
            has_numbered_clauses=markers["has_numbered_clauses"],
            has_citations=markers["has_citations"],
            has_form_fields=markers["has_form_fields"],
        )

        try:
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            # Remove markdown code blocks if present
            response_clean = response_text.strip()
            if response_clean.startswith("```"):
                lines = response_clean.split("\n")
                response_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else response_clean

            result = json.loads(response_clean)

            # Build profile from LLM response
            profile = ChunkingProfile(
                document_type=result.get("document_type", "other"),
                document_domain=result.get("document_domain", "general"),
                content_density=result.get("content_density", "normal"),
                structure_type=result.get("structure_type", "mixed"),
                complexity=result.get("complexity", "moderate"),
                total_tokens=estimated_tokens,
                total_chars=total_chars,
                is_small_doc=estimated_tokens < SMALL_DOC_TOKENS,
                is_large_doc=estimated_tokens > LARGE_DOC_TOKENS,
                has_tables=markers["has_tables"],
                has_headers=markers["has_headers"],
                has_lists=markers["has_lists"],
                has_code=markers["has_code"],
                has_equations=markers["has_equations"],
                has_numbered_clauses=markers["has_numbered_clauses"],
                has_citations=markers["has_citations"],
                has_form_fields=markers["has_form_fields"],
                has_dates=markers["has_dates"],
                has_amounts=markers["has_amounts"],
                recommended_strategy=result.get("recommended_strategy", "semantic_header"),
                recommended_chunk_size=result.get("recommended_chunk_size", 512),
                recommended_overlap_percent=result.get("recommended_overlap_percent", 10),
                is_atomic_unit=result.get("is_atomic_unit", False),
                preserve_elements=result.get("preserve_elements", []),
                special_handling=result.get("special_handling", []),
                confidence=result.get("confidence", 0.7),
                reasoning=result.get("reasoning", ""),
            )

            logger.info(
                f"[Classifier] LLM classification: {profile.document_type} "
                f"({profile.document_domain}) -> {profile.recommended_strategy} "
                f"(confidence: {profile.confidence:.2f})"
            )

            return profile

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM classification response: {e}")
            return None
        except Exception as e:
            logger.warning(f"LLM classification error: {e}")
            return None

    def _classify_with_rules(
        self,
        content: str,
        filename: str,
        markers: Dict[str, bool],
        total_chars: int,
        estimated_tokens: int,
        is_small: bool,
        is_large: bool,
    ) -> ChunkingProfile:
        """Classify using rule-based approach."""
        # Detect domain from content patterns
        domain_scores = detect_domain_from_content(content)
        primary_domain = max(domain_scores, key=domain_scores.get)
        domain_confidence = domain_scores[primary_domain]

        # Determine document type from filename and content first
        doc_type = self._infer_document_type(filename, content, markers, primary_domain)

        # Override domain based on document type for better accuracy
        doc_type_to_domain = {
            # Academic types -> Academic domain
            "research_paper": DocumentDomain.ACADEMIC,
            "thesis": DocumentDomain.ACADEMIC,
            "academic_article": DocumentDomain.ACADEMIC,
            # Education types -> Education domain
            "textbook": DocumentDomain.EDUCATION,
            "course_material": DocumentDomain.EDUCATION,
            "syllabus": DocumentDomain.EDUCATION,
            "exam": DocumentDomain.EDUCATION,
            "tutorial": DocumentDomain.EDUCATION,
            # Engineering types -> Engineering domain
            "technical_spec": DocumentDomain.ENGINEERING,
            "api_documentation": DocumentDomain.ENGINEERING,
            "user_manual": DocumentDomain.ENGINEERING,
            "architecture_doc": DocumentDomain.ENGINEERING,
            # Medical types -> Medical domain
            "medical_record": DocumentDomain.MEDICAL,
            "clinical_report": DocumentDomain.MEDICAL,
            "prescription": DocumentDomain.MEDICAL,
            "lab_result": DocumentDomain.MEDICAL,
            # Legal types -> Legal domain (includes policies)
            "contract": DocumentDomain.LEGAL,
            "agreement": DocumentDomain.LEGAL,
            "terms_of_service": DocumentDomain.LEGAL,
            "privacy_policy": DocumentDomain.LEGAL,
            "policy": DocumentDomain.LEGAL,  # Return policies, refund policies, etc.
            # Financial types -> Financial domain
            "receipt": DocumentDomain.FINANCIAL,
            "invoice": DocumentDomain.FINANCIAL,
            "bank_statement": DocumentDomain.FINANCIAL,
            "expense_report": DocumentDomain.FINANCIAL,
        }

        if doc_type in doc_type_to_domain:
            primary_domain = doc_type_to_domain[doc_type]
            domain_confidence = max(domain_confidence, 0.7)  # Boost confidence

        # Get domain config
        domain_config = get_domain_config(primary_domain)

        # Determine strategy based on domain and markers
        strategy, chunk_size, overlap = self._determine_strategy(
            domain=primary_domain,
            doc_type=doc_type,
            markers=markers,
            is_small=is_small,
            is_large=is_large,
            domain_config=domain_config,
        )

        # Determine structure type
        structure_type = self._determine_structure_type(markers)

        # Build profile
        profile = ChunkingProfile(
            document_type=doc_type,
            document_domain=primary_domain.value,
            content_density="dense" if is_large else ("sparse" if is_small else "normal"),
            structure_type=structure_type,
            complexity="simple" if is_small else ("complex" if is_large else "moderate"),
            total_tokens=estimated_tokens,
            total_chars=total_chars,
            is_small_doc=is_small,
            is_large_doc=is_large,
            has_tables=markers["has_tables"],
            has_headers=markers["has_headers"],
            has_lists=markers["has_lists"],
            has_code=markers["has_code"],
            has_equations=markers["has_equations"],
            has_numbered_clauses=markers["has_numbered_clauses"],
            has_citations=markers["has_citations"],
            has_form_fields=markers["has_form_fields"],
            has_dates=markers["has_dates"],
            has_amounts=markers["has_amounts"],
            recommended_strategy=strategy,
            recommended_chunk_size=chunk_size,
            recommended_overlap_percent=overlap,
            is_atomic_unit=is_small and not markers["has_headers"],
            preserve_elements=list(domain_config.preserve_elements),
            special_handling=[],
            confidence=max(0.4, domain_confidence),
            reasoning=f"Rule-based: detected {primary_domain.value} domain with {doc_type} type",
        )

        logger.info(
            f"[Classifier] Rule-based classification: {profile.document_type} "
            f"({profile.document_domain}) -> {profile.recommended_strategy}"
        )

        return profile

    def _infer_document_type(
        self,
        filename: str,
        content: str,
        markers: Dict[str, bool],
        domain: DocumentDomain,
    ) -> str:
        """
        Infer document type using the centralized DocumentTypeClassifier.

        Uses LLM-driven analysis when available, with fallback to rule-based.

        Args:
            filename: Document filename
            content: Document content
            markers: Structural markers detected in content
            domain: Detected domain from pattern analysis

        Returns:
            Document type string
        """
        try:
            # Build metadata from markers for centralized classifier
            metadata = {
                'document_type': '',  # Will be inferred
                'domain': domain.value,
                'has_tables': markers.get('has_tables', False),
                'has_amounts': markers.get('has_amounts', False),
            }

            # Use centralized classifier
            # Clean content from embedded base64 images before classification
            from ..markdown_chunker import clean_markdown_images
            clean_content = clean_markdown_images(content) if content else None

            classifier = self._get_centralized_classifier()
            result = classifier.classify(
                filename=filename,
                metadata=metadata,
                content_preview=clean_content[:2000] if clean_content else None
            )

            if result and result.document_type:
                logger.info(
                    f"[Classifier] Centralized classification: {result.document_type} "
                    f"(confidence: {result.confidence:.2f})"
                )
                return result.document_type

        except Exception as e:
            logger.warning(f"[Classifier] Centralized classification failed: {e}")

        # Fallback to domain-based defaults
        domain_defaults = {
            DocumentDomain.FINANCIAL: "receipt",
            DocumentDomain.LEGAL: "contract",
            DocumentDomain.ACADEMIC: "research_paper",
            DocumentDomain.MEDICAL: "medical_record",
            DocumentDomain.EDUCATION: "course_material",
            DocumentDomain.ENGINEERING: "technical_spec",
            DocumentDomain.GOVERNMENT: "government_form",
            DocumentDomain.PROFESSIONAL: "resume",
            DocumentDomain.GENERAL: "report",
        }

        return domain_defaults.get(domain, "other")

    def _determine_strategy(
        self,
        domain: DocumentDomain,
        doc_type: str,
        markers: Dict[str, bool],
        is_small: bool,
        is_large: bool,
        domain_config: DomainConfig,
    ) -> Tuple[str, int, int]:
        """Determine chunking strategy based on document characteristics."""
        # Small atomic documents -> whole document
        if is_small and not markers["has_headers"]:
            return "whole_document", 0, 0

        # Domain-specific strategies
        strategy_map = {
            DocumentDomain.LEGAL: ("clause_preserving", 1024, 5),
            DocumentDomain.ACADEMIC: ("academic_structure", 768, 15),
            DocumentDomain.MEDICAL: ("medical_section", 1024, 5),
            DocumentDomain.ENGINEERING: ("requirement_based", 1024, 5),
            DocumentDomain.EDUCATION: ("educational_unit", 768, 10),
            DocumentDomain.GOVERNMENT: ("semantic_header", 1024, 5),  # or form_structure
        }

        if domain in strategy_map:
            base_strategy, base_size, base_overlap = strategy_map[domain]

            # Override for specific patterns
            if markers["has_tables"] and domain != DocumentDomain.LEGAL:
                return "table_preserving", 1024, 10

            return base_strategy, base_size, base_overlap

        # Financial documents
        if domain == DocumentDomain.FINANCIAL:
            if markers["has_tables"]:
                return "table_preserving", 1024, 15
            if is_small:
                return "whole_document", 0, 0
            return "semantic_header", 512, 10

        # Default strategies based on structure
        if markers["has_numbered_clauses"]:
            return "clause_preserving", 1024, 5
        if markers["has_citations"] or markers["has_equations"]:
            return "academic_structure", 768, 15
        if markers["has_tables"]:
            return "table_preserving", 1024, 10
        if markers["has_headers"]:
            return "semantic_header", 512, 10

        # Fallback to paragraph
        return "paragraph", 768, 15

    def _determine_structure_type(self, markers: Dict[str, bool]) -> str:
        """Determine document structure type."""
        if markers["has_tables"]:
            if markers["has_headers"]:
                return "mixed"
            return "tabular"
        if markers["has_numbered_clauses"]:
            return "clause_based"
        if markers["has_form_fields"]:
            return "form_based"
        if markers["has_headers"]:
            return "hierarchical"
        return "narrative"


# Convenience function
def classify_document(
    content: str,
    filename: str = "",
    use_llm: bool = True,
) -> ChunkingProfile:
    """
    Classify a document and return a ChunkingProfile.

    Args:
        content: The document content
        filename: The document filename
        use_llm: Whether to use LLM for classification

    Returns:
        ChunkingProfile with classification results
    """
    classifier = DocumentClassifier(use_llm=use_llm)
    return classifier.classify(content, filename)
