"""
Chunking strategy implementations for domain-aware document processing.

This module provides various chunking strategies optimized for different
document types and domains. Each strategy implements specific splitting
logic to preserve semantic coherence.
"""

import logging
import os
import re
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Configuration for parent chunk summarization
GENERATE_PARENT_SUMMARIES = os.getenv("GENERATE_PARENT_SUMMARIES", "true").lower() == "true"
PARENT_SUMMARY_TOKEN_THRESHOLD = int(os.getenv("PARENT_SUMMARY_TOKEN_THRESHOLD", "1500"))

from .chunk_metadata import (
    ChunkingProfile,
    UniversalChunkMetadata,
    LegalChunkMetadata,
    AcademicChunkMetadata,
    MedicalChunkMetadata,
    EngineeringChunkMetadata,
    EducationChunkMetadata,
    FinancialChunkMetadata,
    GovernmentChunkMetadata,
)
from .domain_patterns import (
    DocumentDomain,
    DomainConfig,
    get_domain_config,
    LEGAL_PATTERNS,
    ACADEMIC_PATTERNS,
    MEDICAL_PATTERNS,
    ENGINEERING_PATTERNS,
    EDUCATION_PATTERNS,
    FINANCIAL_PATTERNS,
)

logger = logging.getLogger(__name__)

# Markdown headers for splitting
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
]


@dataclass
class ChunkResult:
    """Result of chunking operation."""
    content: str
    metadata: Dict[str, Any]
    chunk_type: str = "default"


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 51,
        profile: Optional[ChunkingProfile] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.profile = profile
        self.strategy_name = "base"

    @abstractmethod
    def chunk(
        self,
        content: str,
        source_name: str,
        file_path: str = "",
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Chunk the content according to the strategy.

        Args:
            content: The text content to chunk
            source_name: Name of the source document
            file_path: Path to the source file
            existing_metadata: Any existing metadata to include

        Returns:
            List of Document objects with chunked content and metadata
        """
        pass

    def _generate_chunk_id(self) -> str:
        """Generate a unique chunk ID."""
        return str(uuid.uuid4())

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: 1 token ≈ 4 chars)."""
        return len(text) // 4

    def _generate_parent_summary(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a summary for a parent chunk if it exceeds the token threshold.

        This method uses the ParentChunkSummarizer to create a concise summary
        that can be used during retrieval instead of the full content.

        Args:
            content: The full parent chunk content
            metadata: Optional metadata for context

        Returns:
            Dictionary with summary info, or None if not needed/available
        """
        if not GENERATE_PARENT_SUMMARIES:
            return None

        token_count = self._estimate_tokens(content)
        if token_count <= PARENT_SUMMARY_TOKEN_THRESHOLD:
            return None

        try:
            from .parent_chunk_summarizer import summarize_parent_chunk
            result = summarize_parent_chunk(content, metadata)

            if result.was_summarized:
                return {
                    "parent_summary": result.summary,
                    "parent_summary_tokens": result.summary_token_count,
                    "parent_original_tokens": result.original_token_count,
                    "parent_compression_ratio": result.compression_ratio,
                    "parent_content_type": result.content_type,
                    "parent_key_points": result.key_points,
                    "parent_entities": result.preserved_entities,
                }
        except ImportError:
            logger.debug("Parent chunk summarizer not available")
        except Exception as e:
            logger.warning(f"Failed to generate parent summary: {e}")

        return None

    def _build_heading_path(self, metadata: Dict[str, Any]) -> str:
        """Build heading path from header metadata."""
        parts = []
        for i in range(1, 6):
            header_key = f"Header {i}"
            if header_key in metadata and metadata[header_key]:
                parts.append(metadata[header_key])
        return " → ".join(parts) if parts else "General"

    def _detect_content_markers(self, content: str) -> Dict[str, bool]:
        """Detect content type markers in text."""
        return {
            "contains_table": bool(re.search(r"<table>|^\|.*\|$", content, re.MULTILINE)),
            "contains_list": bool(re.search(r"^\s*[-*•]\s+|\d+\.\s+", content, re.MULTILINE)),
            "contains_code": bool(re.search(r"```|<code>|<pre>", content)),
            "contains_image_ref": bool(re.search(r"!\[.*?\]\(.*?\)|<img", content)),
            "contains_equation": bool(re.search(r"\$\$.*?\$\$|\\begin\{equation\}", content)),
        }

    def _create_base_metadata(
        self,
        source_name: str,
        file_path: str,
        chunk_index: int,
        total_chunks: int,
        content: str,
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create base metadata for a chunk."""
        metadata = {
            "chunk_id": self._generate_chunk_id(),
            "source": source_name,
            "file_path": file_path,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "chunking_strategy": self.strategy_name,
            "chunk_size_used": self.chunk_size,
            "overlap_applied": self.chunk_overlap,
        }

        # Add content markers
        metadata.update(self._detect_content_markers(content))

        # Add profile info if available
        if self.profile:
            metadata["document_type"] = self.profile.document_type
            metadata["document_domain"] = self.profile.document_domain

        # Merge existing metadata
        if existing_metadata:
            for key, value in existing_metadata.items():
                if key not in metadata:
                    metadata[key] = value

        # Build heading path
        metadata["heading_path"] = self._build_heading_path(metadata)

        return metadata


class WholeDocumentStrategy(ChunkingStrategy):
    """
    Strategy for small, atomic documents that should not be split.
    Ideal for: Single-page receipts, short invoices, memos, emails.
    """

    def __init__(self, profile: Optional[ChunkingProfile] = None):
        super().__init__(chunk_size=0, chunk_overlap=0, profile=profile)
        self.strategy_name = "whole_document"

    def chunk(
        self,
        content: str,
        source_name: str,
        file_path: str = "",
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Keep document as single chunk."""
        metadata = self._create_base_metadata(
            source_name=source_name,
            file_path=file_path,
            chunk_index=0,
            total_chunks=1,
            content=content,
            existing_metadata=existing_metadata,
        )
        metadata["is_atomic"] = True
        metadata["chunk_type"] = "whole_document"

        return [Document(page_content=content, metadata=metadata)]


class SemanticHeaderStrategy(ChunkingStrategy):
    """
    Strategy for documents with clear header structure.
    Splits on markdown headers while preserving section context.
    Ideal for: Reports, manuals, resumes, structured documents.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 51,
        max_chunk_size: int = 1024,
        profile: Optional[ChunkingProfile] = None,
    ):
        super().__init__(chunk_size, chunk_overlap, profile)
        self.max_chunk_size = max_chunk_size
        self.strategy_name = "semantic_header"

    def chunk(
        self,
        content: str,
        source_name: str,
        file_path: str = "",
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Split by headers, then recursively split large sections.

        Implements parent/child chunk design:
        - When a section is split, a parent chunk (full section) is created
        - Child chunks reference the parent via parent_chunk_id
        - Parent chunks have is_parent_chunk=True and child_chunk_ids list
        """
        # Split by headers first
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )

        try:
            header_chunks = header_splitter.split_text(content)
        except Exception as e:
            logger.warning(f"Header splitting failed: {e}, using full content")
            header_chunks = [Document(page_content=content, metadata={})]

        # Secondary splitter for large chunks
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        final_chunks = []
        chunk_index = 0

        for i, chunk in enumerate(header_chunks):
            page_content = chunk.page_content
            chunk_metadata = dict(chunk.metadata) if chunk.metadata else {}

            # Check if chunk needs further splitting
            if len(page_content) > self.max_chunk_size:
                # Generate parent chunk ID
                parent_chunk_id = self._generate_chunk_id()

                # Split into child chunks first to get IDs upfront
                sub_chunks = recursive_splitter.split_text(page_content)
                child_ids = [self._generate_chunk_id() for _ in sub_chunks]

                # Create parent chunk with child IDs already populated
                parent_metadata = self._create_base_metadata(
                    source_name=source_name,
                    file_path=file_path,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will update later
                    content=page_content,
                    existing_metadata={**chunk_metadata, **(existing_metadata or {})},
                )
                parent_metadata["chunk_type"] = "parent"
                parent_metadata["is_parent_chunk"] = True
                parent_metadata["chunk_id"] = parent_chunk_id  # Override generated ID
                parent_metadata["child_chunk_ids"] = child_ids  # Set child IDs before creating Document
                parent_metadata["parent_chunk_index"] = i

                # Generate summary for large parent chunks
                summary_info = self._generate_parent_summary(page_content, parent_metadata)
                if summary_info:
                    parent_metadata.update(summary_info)

                parent_doc = Document(page_content=page_content, metadata=parent_metadata)
                final_chunks.append(parent_doc)
                chunk_index += 1

                # Create child chunks with pre-generated IDs
                for j, sub_content in enumerate(sub_chunks):
                    child_chunk_id = child_ids[j]

                    metadata = self._create_base_metadata(
                        source_name=source_name,
                        file_path=file_path,
                        chunk_index=chunk_index,
                        total_chunks=0,  # Will update later
                        content=sub_content,
                        existing_metadata={**chunk_metadata, **(existing_metadata or {})},
                    )
                    metadata["chunk_type"] = "child"
                    metadata["chunk_id"] = child_chunk_id  # Use pre-generated ID
                    metadata["parent_chunk_id"] = parent_chunk_id
                    metadata["parent_chunk_index"] = i
                    metadata["sub_chunk_index"] = j
                    metadata["is_parent_chunk"] = False

                    final_chunks.append(Document(page_content=sub_content, metadata=metadata))
                    chunk_index += 1
            else:
                # Keep chunk as-is (atomic chunk, no parent/child relationship)
                metadata = self._create_base_metadata(
                    source_name=source_name,
                    file_path=file_path,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will update later
                    content=page_content,
                    existing_metadata={**chunk_metadata, **(existing_metadata or {})},
                )
                metadata["chunk_type"] = "semantic_header"
                metadata["is_parent_chunk"] = False
                metadata["is_atomic"] = True  # No splitting needed

                final_chunks.append(Document(page_content=page_content, metadata=metadata))
                chunk_index += 1

        # Update total_chunks in all metadata
        total = len(final_chunks)
        for chunk in final_chunks:
            chunk.metadata["total_chunks"] = total

        return final_chunks


class ClausePreservingStrategy(ChunkingStrategy):
    """
    Strategy for legal documents with numbered clauses.
    Never splits within a clause; preserves clause hierarchy.
    Ideal for: Contracts, agreements, terms of service.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 50,
        profile: Optional[ChunkingProfile] = None,
    ):
        super().__init__(chunk_size, chunk_overlap, profile)
        self.strategy_name = "clause_preserving"

        # Patterns for clause detection
        self.clause_pattern = re.compile(r"^\s*((\d+\.)+\d*)\s+(.+?)(?=^\s*(\d+\.)+\d*\s+|\Z)", re.MULTILINE | re.DOTALL)
        self.article_pattern = re.compile(r"^(Article\s+[IVXLCDM\d]+)[:\.\s](.+?)(?=^Article\s+[IVXLCDM\d]+|\Z)", re.MULTILINE | re.DOTALL | re.IGNORECASE)
        self.section_pattern = re.compile(r"^(Section\s+\d+)[:\.\s](.+?)(?=^Section\s+\d+|\Z)", re.MULTILINE | re.DOTALL | re.IGNORECASE)
        self.whereas_pattern = re.compile(r"(WHEREAS.+?)(?=WHEREAS|NOW,?\s*THEREFORE|\Z)", re.DOTALL)
        self.definition_pattern = re.compile(r'"([^"]+)"\s+(means|shall mean|refers to)\s+([^;]+;?)', re.IGNORECASE)

    def _extract_clauses(self, content: str) -> List[Tuple[str, str, str]]:
        """Extract clauses with their numbers and content."""
        clauses = []

        # Try different clause patterns
        for match in self.clause_pattern.finditer(content):
            clause_num = match.group(1)
            clause_content = match.group(0).strip()
            clauses.append((clause_num, clause_content, "numbered_clause"))

        # If no numbered clauses, try articles
        if not clauses:
            for match in self.article_pattern.finditer(content):
                article_id = match.group(1)
                article_content = match.group(0).strip()
                clauses.append((article_id, article_content, "article"))

        # If still none, try sections
        if not clauses:
            for match in self.section_pattern.finditer(content):
                section_id = match.group(1)
                section_content = match.group(0).strip()
                clauses.append((section_id, section_content, "section"))

        return clauses

    def _extract_definitions(self, content: str) -> List[Tuple[str, str]]:
        """Extract defined terms."""
        definitions = []
        for match in self.definition_pattern.finditer(content):
            term = match.group(1)
            definition = match.group(0)
            definitions.append((term, definition))
        return definitions

    def chunk(
        self,
        content: str,
        source_name: str,
        file_path: str = "",
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Split by clauses while preserving integrity."""
        clauses = self._extract_clauses(content)
        definitions = self._extract_definitions(content)

        final_chunks = []
        chunk_index = 0

        # If we found structured clauses, use them
        if clauses:
            current_group = []
            current_size = 0

            for clause_num, clause_content, clause_type in clauses:
                clause_size = len(clause_content)

                # If adding this clause exceeds limit, save current group
                if current_size + clause_size > self.chunk_size and current_group:
                    combined_content = "\n\n".join([c[1] for c in current_group])
                    metadata = self._create_base_metadata(
                        source_name=source_name,
                        file_path=file_path,
                        chunk_index=chunk_index,
                        total_chunks=0,
                        content=combined_content,
                        existing_metadata=existing_metadata,
                    )
                    metadata["chunk_type"] = "clause_group"
                    metadata["clause_numbers"] = [c[0] for c in current_group]
                    metadata["clause_range"] = f"{current_group[0][0]} - {current_group[-1][0]}"

                    final_chunks.append(Document(page_content=combined_content, metadata=metadata))
                    chunk_index += 1
                    current_group = []
                    current_size = 0

                current_group.append((clause_num, clause_content, clause_type))
                current_size += clause_size

            # Save remaining clauses
            if current_group:
                combined_content = "\n\n".join([c[1] for c in current_group])
                metadata = self._create_base_metadata(
                    source_name=source_name,
                    file_path=file_path,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    content=combined_content,
                    existing_metadata=existing_metadata,
                )
                metadata["chunk_type"] = "clause_group"
                metadata["clause_numbers"] = [c[0] for c in current_group]

                final_chunks.append(Document(page_content=combined_content, metadata=metadata))
                chunk_index += 1
        else:
            # Fall back to semantic header strategy
            fallback = SemanticHeaderStrategy(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                profile=self.profile,
            )
            final_chunks = fallback.chunk(content, source_name, file_path, existing_metadata)
            # Update strategy name
            for chunk in final_chunks:
                chunk.metadata["chunking_strategy"] = "clause_preserving_fallback"

        # Update total counts
        total = len(final_chunks)
        for chunk in final_chunks:
            chunk.metadata["total_chunks"] = total

        return final_chunks


class AcademicStructureStrategy(ChunkingStrategy):
    """
    Strategy for academic/research documents.
    Preserves abstract, citations, equations, and figure references.
    Ideal for: Research papers, theses, academic articles.
    """

    def __init__(
        self,
        chunk_size: int = 768,
        chunk_overlap: int = 100,
        profile: Optional[ChunkingProfile] = None,
    ):
        super().__init__(chunk_size, chunk_overlap, profile)
        self.strategy_name = "academic_structure"

        # Academic section patterns
        self.abstract_pattern = re.compile(r"^(?:Abstract|ABSTRACT)[:\s]*\n(.+?)(?=^(?:Introduction|INTRODUCTION|1\.|Keywords|KEY\s*WORDS))", re.MULTILINE | re.DOTALL | re.IGNORECASE)
        self.section_pattern = re.compile(r"^(?:#+\s*)?(\d+\.?\s+)?(Introduction|Background|Literature Review|Methods?|Methodology|Results?|Discussion|Conclusion|Acknowledgments?|References|Bibliography|Appendix)s?[:\s]*$", re.MULTILINE | re.IGNORECASE)
        self.citation_pattern = re.compile(r"\[[\d,\s\-]+\]|\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}[a-z]?\)")
        self.figure_pattern = re.compile(r"(?:Figure|Fig\.?|Table)\s+(\d+(?:\.\d+)?)", re.IGNORECASE)

    def _extract_abstract(self, content: str) -> Optional[str]:
        """Extract abstract section."""
        match = self.abstract_pattern.search(content)
        if match:
            return match.group(0).strip()

        # Try simpler pattern
        lines = content.split('\n')
        in_abstract = False
        abstract_lines = []

        for line in lines:
            if re.match(r"^(?:Abstract|ABSTRACT)[:\s]*$", line.strip(), re.IGNORECASE):
                in_abstract = True
                continue
            if in_abstract:
                if re.match(r"^(?:#+\s*)?(Introduction|Keywords|1\.)", line, re.IGNORECASE):
                    break
                abstract_lines.append(line)

        if abstract_lines:
            return "Abstract\n" + "\n".join(abstract_lines).strip()
        return None

    def _extract_citations(self, content: str) -> List[str]:
        """Extract citations from content."""
        return self.citation_pattern.findall(content)

    def _extract_figure_refs(self, content: str) -> List[str]:
        """Extract figure/table references."""
        matches = self.figure_pattern.findall(content)
        return [f"Figure {m}" for m in matches]

    def chunk(
        self,
        content: str,
        source_name: str,
        file_path: str = "",
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Split academic document while preserving key sections."""
        final_chunks = []
        chunk_index = 0

        # Extract and handle abstract separately
        abstract = self._extract_abstract(content)
        if abstract:
            metadata = self._create_base_metadata(
                source_name=source_name,
                file_path=file_path,
                chunk_index=chunk_index,
                total_chunks=0,
                content=abstract,
                existing_metadata=existing_metadata,
            )
            metadata["chunk_type"] = "abstract"
            metadata["is_abstract"] = True
            metadata["is_key_section"] = True
            metadata["section_type"] = "abstract"
            metadata["importance_score"] = 1.0

            final_chunks.append(Document(page_content=abstract, metadata=metadata))
            chunk_index += 1

            # Remove abstract from content for further processing
            content = content.replace(abstract, "", 1)

        # Use semantic header for the rest
        header_strategy = SemanticHeaderStrategy(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            profile=self.profile,
        )

        remaining_chunks = header_strategy.chunk(content, source_name, file_path, existing_metadata)

        # Enhance with academic metadata
        for chunk in remaining_chunks:
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["chunking_strategy"] = self.strategy_name

            # Detect section type from heading
            heading_path = chunk.metadata.get("heading_path", "")
            section_type = self._detect_section_type(heading_path, chunk.page_content)
            chunk.metadata["section_type"] = section_type

            # Extract citations and figure refs
            chunk.metadata["citations_in_chunk"] = self._extract_citations(chunk.page_content)
            chunk.metadata["figures_referenced"] = self._extract_figure_refs(chunk.page_content)

            # Mark bibliography
            if section_type in ["references", "bibliography"]:
                chunk.metadata["is_bibliography"] = True

            final_chunks.append(chunk)
            chunk_index += 1

        # Update total counts
        total = len(final_chunks)
        for chunk in final_chunks:
            chunk.metadata["total_chunks"] = total

        return final_chunks

    def _detect_section_type(self, heading_path: str, content: str) -> str:
        """Detect academic section type."""
        heading_lower = heading_path.lower()
        content_start = content[:200].lower()

        section_keywords = {
            "abstract": ["abstract"],
            "introduction": ["introduction", "intro"],
            "background": ["background", "related work", "literature"],
            "methods": ["method", "methodology", "approach", "procedure"],
            "results": ["result", "finding", "experiment"],
            "discussion": ["discussion", "analysis"],
            "conclusion": ["conclusion", "summary", "future work"],
            "references": ["reference", "bibliography", "works cited"],
            "appendix": ["appendix"],
        }

        for section_type, keywords in section_keywords.items():
            for keyword in keywords:
                if keyword in heading_lower or keyword in content_start:
                    return section_type

        return "body"


class MedicalSectionStrategy(ChunkingStrategy):
    """
    Strategy for medical/clinical documents.
    Preserves SOAP structure and medical section integrity.
    Ideal for: Medical records, clinical notes, discharge summaries.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 50,
        profile: Optional[ChunkingProfile] = None,
    ):
        super().__init__(chunk_size, chunk_overlap, profile)
        self.strategy_name = "medical_section"

        # Medical section headers (SOAP and extended)
        self.section_headers = [
            "Chief Complaint", "CC",
            "History of Present Illness", "HPI",
            "Past Medical History", "PMH",
            "Past Surgical History", "PSH",
            "Medications", "Current Medications", "Meds",
            "Allergies",
            "Social History", "SH",
            "Family History", "FH",
            "Review of Systems", "ROS",
            "Physical Exam", "PE", "Physical Examination",
            "Vital Signs", "Vitals",
            "Laboratory", "Labs", "Lab Results",
            "Imaging", "Radiology",
            "Assessment", "Impression",
            "Plan", "Treatment Plan",
            "Diagnosis", "Diagnoses",
            "Recommendations",
            "Follow-up", "Follow Up",
            "Disposition",
        ]

        self.section_pattern = re.compile(
            r"^(" + "|".join(re.escape(h) for h in self.section_headers) + r")[:\s]*",
            re.MULTILINE | re.IGNORECASE
        )

        # Medical code patterns
        self.icd_pattern = re.compile(r"\b[A-Z]\d{2}(?:\.\d{1,2})?\b")
        self.medication_pattern = re.compile(r"\b(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|ml|mL|units?|g)\b", re.IGNORECASE)
        self.vitals_pattern = re.compile(r"(?:BP|HR|RR|Temp|SpO2|Weight|Height)[:\s]*([\d/\.]+)", re.IGNORECASE)

    def _split_by_sections(self, content: str) -> List[Tuple[str, str]]:
        """Split content by medical sections."""
        sections = []
        matches = list(self.section_pattern.finditer(content))

        if not matches:
            return [("General", content)]

        for i, match in enumerate(matches):
            section_name = match.group(1).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start:end].strip()
            sections.append((section_name, section_content))

        return sections

    def _extract_medical_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract medical codes and entities."""
        return {
            "icd_codes": self.icd_pattern.findall(content),
            "medications": [f"{m[0]} {m[1]}{m[2]}" for m in self.medication_pattern.findall(content)],
            "vitals": self.vitals_pattern.findall(content),
        }

    def chunk(
        self,
        content: str,
        source_name: str,
        file_path: str = "",
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Split by medical sections.

        Implements parent/child chunk design:
        - When a section is split, a parent chunk (full section) is created
        - Child chunks reference the parent via parent_chunk_id
        """
        sections = self._split_by_sections(content)
        final_chunks = []
        chunk_index = 0

        for section_name, section_content in sections:
            # Check if section fits in one chunk
            if len(section_content) <= self.chunk_size:
                metadata = self._create_base_metadata(
                    source_name=source_name,
                    file_path=file_path,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    content=section_content,
                    existing_metadata=existing_metadata,
                )
                metadata["chunk_type"] = "medical_section"
                metadata["section_type"] = section_name
                metadata["is_parent_chunk"] = False
                metadata["is_atomic"] = True

                # Extract medical entities
                entities = self._extract_medical_entities(section_content)
                metadata["icd_codes"] = entities["icd_codes"]
                metadata["medications_mentioned"] = entities["medications"]
                metadata["vitals_present"] = bool(entities["vitals"])

                # Mark key sections
                if section_name.lower() in ["assessment", "plan", "impression", "diagnosis"]:
                    metadata["is_key_section"] = True
                    metadata["importance_score"] = 0.9

                final_chunks.append(Document(page_content=section_content, metadata=metadata))
                chunk_index += 1
            else:
                # Generate parent chunk ID
                parent_chunk_id = self._generate_chunk_id()

                # Split into child chunks first to get IDs upfront
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                sub_chunks = splitter.split_text(section_content)
                child_ids = [self._generate_chunk_id() for _ in sub_chunks]

                # Create parent chunk with child IDs already populated
                parent_metadata = self._create_base_metadata(
                    source_name=source_name,
                    file_path=file_path,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    content=section_content,
                    existing_metadata=existing_metadata,
                )
                parent_metadata["chunk_type"] = "parent"
                parent_metadata["is_parent_chunk"] = True
                parent_metadata["chunk_id"] = parent_chunk_id
                parent_metadata["child_chunk_ids"] = child_ids  # Set child IDs before creating Document
                parent_metadata["section_type"] = section_name

                # Extract medical entities for parent
                entities = self._extract_medical_entities(section_content)
                parent_metadata["icd_codes"] = entities["icd_codes"]
                parent_metadata["medications_mentioned"] = entities["medications"]
                parent_metadata["vitals_present"] = bool(entities["vitals"])

                # Mark key sections
                if section_name.lower() in ["assessment", "plan", "impression", "diagnosis"]:
                    parent_metadata["is_key_section"] = True
                    parent_metadata["importance_score"] = 0.9

                # Generate summary for large parent chunks
                summary_info = self._generate_parent_summary(section_content, parent_metadata)
                if summary_info:
                    parent_metadata.update(summary_info)

                parent_doc = Document(page_content=section_content, metadata=parent_metadata)
                final_chunks.append(parent_doc)
                chunk_index += 1

                # Create child chunks with pre-generated IDs
                for j, sub_content in enumerate(sub_chunks):
                    child_chunk_id = child_ids[j]

                    metadata = self._create_base_metadata(
                        source_name=source_name,
                        file_path=file_path,
                        chunk_index=chunk_index,
                        total_chunks=0,
                        content=sub_content,
                        existing_metadata=existing_metadata,
                    )
                    metadata["chunk_type"] = "child"
                    metadata["chunk_id"] = child_chunk_id  # Use pre-generated ID
                    metadata["parent_chunk_id"] = parent_chunk_id
                    metadata["is_parent_chunk"] = False
                    metadata["section_type"] = section_name
                    metadata["section_part"] = j + 1
                    metadata["sub_chunk_index"] = j

                    entities = self._extract_medical_entities(sub_content)
                    metadata["icd_codes"] = entities["icd_codes"]
                    metadata["medications_mentioned"] = entities["medications"]
                    metadata["vitals_present"] = bool(entities["vitals"])

                    final_chunks.append(Document(page_content=sub_content, metadata=metadata))
                    chunk_index += 1

        # Update total counts
        total = len(final_chunks)
        for chunk in final_chunks:
            chunk.metadata["total_chunks"] = total

        return final_chunks


class RequirementBasedStrategy(ChunkingStrategy):
    """
    Strategy for engineering/technical specification documents.
    Preserves requirement IDs and traceability.
    Ideal for: Technical specs, requirements docs, system designs.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 50,
        profile: Optional[ChunkingProfile] = None,
    ):
        super().__init__(chunk_size, chunk_overlap, profile)
        self.strategy_name = "requirement_based"

        # Requirement ID patterns
        self.req_id_pattern = re.compile(r"\b(REQ|SPEC|SRS|SDD|ICD|HLD|LLD)[-_]?(\d+(?:\.\d+)*)\b")
        self.shall_pattern = re.compile(r"\b(shall|must|will|should|may)\s+", re.IGNORECASE)
        self.version_pattern = re.compile(r"[Vv](?:ersion)?\s*([\d\.]+)")
        self.diagram_pattern = re.compile(r"(?:Figure|Diagram|Drawing|Schematic)\s+([\d\w\.\-]+)", re.IGNORECASE)

    def _extract_requirement_ids(self, content: str) -> List[str]:
        """Extract requirement IDs from content."""
        matches = self.req_id_pattern.findall(content)
        return [f"{m[0]}-{m[1]}" for m in matches]

    def _detect_priority(self, content: str) -> str:
        """Detect requirement priority level."""
        content_lower = content.lower()
        if "shall" in content_lower or "must" in content_lower:
            return "shall"
        elif "should" in content_lower:
            return "should"
        elif "may" in content_lower:
            return "may"
        return ""

    def chunk(
        self,
        content: str,
        source_name: str,
        file_path: str = "",
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Split while preserving requirement context."""
        # Use semantic header as base, then enhance
        header_strategy = SemanticHeaderStrategy(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            profile=self.profile,
        )

        chunks = header_strategy.chunk(content, source_name, file_path, existing_metadata)

        # Enhance with engineering metadata
        for chunk in chunks:
            chunk.metadata["chunking_strategy"] = self.strategy_name

            # Extract requirement IDs
            req_ids = self._extract_requirement_ids(chunk.page_content)
            chunk.metadata["requirement_ids"] = req_ids

            # Detect priority
            chunk.metadata["priority_level"] = self._detect_priority(chunk.page_content)

            # Extract version
            version_match = self.version_pattern.search(chunk.page_content)
            if version_match:
                chunk.metadata["version"] = version_match.group(1)

            # Extract diagram references
            diagrams = self.diagram_pattern.findall(chunk.page_content)
            chunk.metadata["diagrams_referenced"] = diagrams

            # Mark chunks with requirements as important
            if req_ids:
                chunk.metadata["importance_score"] = 0.8

        return chunks


class EducationalUnitStrategy(ChunkingStrategy):
    """
    Strategy for educational/training materials.
    Preserves learning objectives, exercises, and Q&A pairs.
    Ideal for: Textbooks, course materials, exams, tutorials.
    """

    def __init__(
        self,
        chunk_size: int = 768,
        chunk_overlap: int = 75,
        profile: Optional[ChunkingProfile] = None,
    ):
        super().__init__(chunk_size, chunk_overlap, profile)
        self.strategy_name = "educational_unit"

        # Educational patterns
        self.chapter_pattern = re.compile(r"^(?:Chapter|Unit|Module|Lesson)\s+(\d+)", re.MULTILINE | re.IGNORECASE)
        self.objective_pattern = re.compile(r"^(?:Learning\s+)?(?:Objectives?|Outcomes?|Goals?)[:\s]*", re.MULTILINE | re.IGNORECASE)
        self.exercise_pattern = re.compile(r"^(?:Exercise|Problem|Question|Activity|Practice)\s+(\d+)", re.MULTILINE | re.IGNORECASE)
        self.answer_pattern = re.compile(r"^(?:Answer|Solution|Key)[:\s]*", re.MULTILINE | re.IGNORECASE)
        self.example_pattern = re.compile(r"^Example\s*(\d*)[:\s]*", re.MULTILINE | re.IGNORECASE)
        self.definition_pattern = re.compile(r"^(?:Definition|Term)[:\s]*(.+)", re.MULTILINE | re.IGNORECASE)

    def _detect_educational_markers(self, content: str) -> Dict[str, Any]:
        """Detect educational content markers."""
        return {
            "chapter": self.chapter_pattern.search(content),
            "has_objectives": bool(self.objective_pattern.search(content)),
            "has_exercises": bool(self.exercise_pattern.search(content)),
            "has_answers": bool(self.answer_pattern.search(content)),
            "has_examples": bool(self.example_pattern.search(content)),
            "has_definitions": bool(self.definition_pattern.search(content)),
        }

    def chunk(
        self,
        content: str,
        source_name: str,
        file_path: str = "",
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Split while preserving educational units."""
        # Use semantic header as base
        header_strategy = SemanticHeaderStrategy(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            profile=self.profile,
        )

        chunks = header_strategy.chunk(content, source_name, file_path, existing_metadata)

        # Enhance with educational metadata
        for chunk in chunks:
            chunk.metadata["chunking_strategy"] = self.strategy_name

            markers = self._detect_educational_markers(chunk.page_content)

            # Add chapter info
            if markers["chapter"]:
                chunk.metadata["chapter"] = f"Chapter {markers['chapter'].group(1)}"

            # Mark content types
            chunk.metadata["contains_exercise"] = markers["has_exercises"]
            chunk.metadata["contains_answer"] = markers["has_answers"]
            chunk.metadata["is_example"] = markers["has_examples"]

            # Learning objectives are key sections
            if markers["has_objectives"]:
                chunk.metadata["is_key_section"] = True
                chunk.metadata["importance_score"] = 0.9

            # Extract key terms from definitions
            definitions = self.definition_pattern.findall(chunk.page_content)
            if definitions:
                chunk.metadata["key_terms"] = [d.strip() for d in definitions[:10]]

        return chunks


class TablePreservingStrategy(ChunkingStrategy):
    """
    Strategy for documents with significant tabular data.
    Preserves table integrity and includes headers in each chunk.
    Ideal for: Spreadsheets, bank statements, invoices with line items.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
        profile: Optional[ChunkingProfile] = None,
    ):
        super().__init__(chunk_size, chunk_overlap, profile)
        self.strategy_name = "table_preserving"

        # Table patterns
        self.html_table_pattern = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)
        self.markdown_table_pattern = re.compile(r"(?:^\|.+\|$\n?)+", re.MULTILINE)

    def _extract_tables(self, content: str) -> List[Tuple[int, int, str, str]]:
        """Extract table boundaries and content."""
        tables = []

        # Find HTML tables
        for match in self.html_table_pattern.finditer(content):
            tables.append((match.start(), match.end(), "html", match.group(0)))

        # Find Markdown tables
        for match in self.markdown_table_pattern.finditer(content):
            tables.append((match.start(), match.end(), "markdown", match.group(0)))

        return sorted(tables, key=lambda x: x[0])

    def _split_html_table(self, table_html: str, max_rows: int = 20) -> List[str]:
        """Split HTML table by rows while preserving headers."""
        thead_match = re.search(r"<thead>.*?</thead>", table_html, re.DOTALL | re.IGNORECASE)
        tbody_match = re.search(r"<tbody>(.*?)</tbody>", table_html, re.DOTALL | re.IGNORECASE)

        if not tbody_match:
            return [table_html]

        thead = thead_match.group(0) if thead_match else ""
        tbody_content = tbody_match.group(1)

        rows = re.findall(r"<tr>.*?</tr>", tbody_content, re.DOTALL | re.IGNORECASE)

        if len(rows) <= max_rows:
            return [table_html]

        chunks = []
        for i in range(0, len(rows), max_rows):
            batch_rows = rows[i:i + max_rows]
            chunk = f"<table>{thead}<tbody>{''.join(batch_rows)}</tbody></table>"
            chunks.append(chunk)

        return chunks

    def chunk(
        self,
        content: str,
        source_name: str,
        file_path: str = "",
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Split while preserving table integrity."""
        tables = self._extract_tables(content)
        final_chunks = []
        chunk_index = 0
        last_end = 0

        for start, end, table_type, table_content in tables:
            # Add text before table
            if start > last_end:
                text_before = content[last_end:start].strip()
                if text_before:
                    # Use semantic header for non-table content
                    header_strategy = SemanticHeaderStrategy(
                        chunk_size=self.chunk_size,
                        profile=self.profile,
                    )
                    text_chunks = header_strategy.chunk(text_before, source_name, file_path, existing_metadata)
                    for tc in text_chunks:
                        tc.metadata["chunk_index"] = chunk_index
                        tc.metadata["chunking_strategy"] = self.strategy_name
                        final_chunks.append(tc)
                        chunk_index += 1

            # Handle table
            if table_type == "html" and len(table_content) > self.chunk_size:
                # Split large HTML table with parent/child design
                table_parts = self._split_html_table(table_content)

                if len(table_parts) > 1:
                    # Generate parent chunk ID and child IDs upfront
                    parent_chunk_id = self._generate_chunk_id()
                    child_ids = [self._generate_chunk_id() for _ in table_parts]

                    # Create parent chunk with child IDs already populated
                    parent_metadata = self._create_base_metadata(
                        source_name=source_name,
                        file_path=file_path,
                        chunk_index=chunk_index,
                        total_chunks=0,
                        content=table_content,
                        existing_metadata=existing_metadata,
                    )
                    parent_metadata["chunk_type"] = "parent"
                    parent_metadata["is_parent_chunk"] = True
                    parent_metadata["chunk_id"] = parent_chunk_id
                    parent_metadata["child_chunk_ids"] = child_ids  # Set child IDs before creating Document
                    parent_metadata["contains_table"] = True
                    parent_metadata["table_type"] = table_type

                    # Generate summary for large table parent chunks
                    summary_info = self._generate_parent_summary(table_content, parent_metadata)
                    if summary_info:
                        parent_metadata.update(summary_info)

                    parent_doc = Document(page_content=table_content, metadata=parent_metadata)
                    final_chunks.append(parent_doc)
                    chunk_index += 1

                    # Create child chunks with pre-generated IDs
                    for j, part in enumerate(table_parts):
                        child_chunk_id = child_ids[j]

                        metadata = self._create_base_metadata(
                            source_name=source_name,
                            file_path=file_path,
                            chunk_index=chunk_index,
                            total_chunks=0,
                            content=part,
                            existing_metadata=existing_metadata,
                        )
                        metadata["chunk_type"] = "child"
                        metadata["chunk_id"] = child_chunk_id  # Use pre-generated ID
                        metadata["parent_chunk_id"] = parent_chunk_id
                        metadata["is_parent_chunk"] = False
                        metadata["table_part"] = j + 1
                        metadata["table_total_parts"] = len(table_parts)
                        metadata["sub_chunk_index"] = j
                        metadata["contains_table"] = True

                        final_chunks.append(Document(page_content=part, metadata=metadata))
                        chunk_index += 1
                else:
                    # Only one part, treat as atomic
                    metadata = self._create_base_metadata(
                        source_name=source_name,
                        file_path=file_path,
                        chunk_index=chunk_index,
                        total_chunks=0,
                        content=table_parts[0],
                        existing_metadata=existing_metadata,
                    )
                    metadata["chunk_type"] = "table"
                    metadata["table_type"] = table_type
                    metadata["contains_table"] = True
                    metadata["is_parent_chunk"] = False
                    metadata["is_atomic"] = True

                    final_chunks.append(Document(page_content=table_parts[0], metadata=metadata))
                    chunk_index += 1
            else:
                # Keep table as single chunk (atomic)
                metadata = self._create_base_metadata(
                    source_name=source_name,
                    file_path=file_path,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    content=table_content,
                    existing_metadata=existing_metadata,
                )
                metadata["chunk_type"] = "table"
                metadata["table_type"] = table_type
                metadata["contains_table"] = True
                metadata["is_parent_chunk"] = False
                metadata["is_atomic"] = True

                final_chunks.append(Document(page_content=table_content, metadata=metadata))
                chunk_index += 1

            last_end = end

        # Add remaining text after last table
        if last_end < len(content):
            text_after = content[last_end:].strip()
            if text_after:
                header_strategy = SemanticHeaderStrategy(
                    chunk_size=self.chunk_size,
                    profile=self.profile,
                )
                text_chunks = header_strategy.chunk(text_after, source_name, file_path, existing_metadata)
                for tc in text_chunks:
                    tc.metadata["chunk_index"] = chunk_index
                    tc.metadata["chunking_strategy"] = self.strategy_name
                    final_chunks.append(tc)
                    chunk_index += 1

        # If no tables found, use semantic header
        if not final_chunks:
            header_strategy = SemanticHeaderStrategy(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                profile=self.profile,
            )
            final_chunks = header_strategy.chunk(content, source_name, file_path, existing_metadata)
            for chunk in final_chunks:
                chunk.metadata["chunking_strategy"] = self.strategy_name

        # Update total counts
        total = len(final_chunks)
        for chunk in final_chunks:
            chunk.metadata["total_chunks"] = total

        return final_chunks


class ParagraphStrategy(ChunkingStrategy):
    """
    Strategy for narrative documents without strong structure.
    Splits at paragraph boundaries with overlap.
    Ideal for: Articles, blog posts, narrative text.
    """

    def __init__(
        self,
        chunk_size: int = 768,
        chunk_overlap: int = 100,
        profile: Optional[ChunkingProfile] = None,
    ):
        super().__init__(chunk_size, chunk_overlap, profile)
        self.strategy_name = "paragraph"

    def chunk(
        self,
        content: str,
        source_name: str,
        file_path: str = "",
        existing_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Split at paragraph boundaries."""
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r"\n\n+", content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        final_chunks = []
        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            test_chunk = current_chunk + ("\n\n" if current_chunk else "") + para

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if it has content
                if current_chunk:
                    metadata = self._create_base_metadata(
                        source_name=source_name,
                        file_path=file_path,
                        chunk_index=chunk_index,
                        total_chunks=0,
                        content=current_chunk,
                        existing_metadata=existing_metadata,
                    )
                    metadata["chunk_type"] = "paragraph"

                    final_chunks.append(Document(page_content=current_chunk, metadata=metadata))
                    chunk_index += 1

                # Start new chunk (with overlap from previous)
                if self.chunk_overlap > 0 and current_chunk:
                    # Get last portion for overlap
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para

                # If single paragraph is too large, split it
                if len(current_chunk) > self.chunk_size:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )
                    sub_chunks = splitter.split_text(current_chunk)
                    for sub in sub_chunks[:-1]:  # All but last
                        metadata = self._create_base_metadata(
                            source_name=source_name,
                            file_path=file_path,
                            chunk_index=chunk_index,
                            total_chunks=0,
                            content=sub,
                            existing_metadata=existing_metadata,
                        )
                        metadata["chunk_type"] = "paragraph_split"

                        final_chunks.append(Document(page_content=sub, metadata=metadata))
                        chunk_index += 1
                    current_chunk = sub_chunks[-1] if sub_chunks else ""

        # Add remaining content
        if current_chunk:
            metadata = self._create_base_metadata(
                source_name=source_name,
                file_path=file_path,
                chunk_index=chunk_index,
                total_chunks=0,
                content=current_chunk,
                existing_metadata=existing_metadata,
            )
            metadata["chunk_type"] = "paragraph"

            final_chunks.append(Document(page_content=current_chunk, metadata=metadata))

        # Update total counts
        total = len(final_chunks)
        for chunk in final_chunks:
            chunk.metadata["total_chunks"] = total

        return final_chunks


# ============================================================================
# Strategy Factory
# ============================================================================

STRATEGY_MAP = {
    "whole_document": WholeDocumentStrategy,
    "semantic_header": SemanticHeaderStrategy,
    "clause_preserving": ClausePreservingStrategy,
    "academic_structure": AcademicStructureStrategy,
    "medical_section": MedicalSectionStrategy,
    "requirement_based": RequirementBasedStrategy,
    "educational_unit": EducationalUnitStrategy,
    "table_preserving": TablePreservingStrategy,
    "paragraph": ParagraphStrategy,
}


def get_strategy_for_profile(profile: ChunkingProfile) -> ChunkingStrategy:
    """
    Get the appropriate chunking strategy based on the document profile.

    Args:
        profile: The ChunkingProfile from document classification

    Returns:
        Configured ChunkingStrategy instance
    """
    strategy_name = profile.recommended_strategy
    chunk_size = profile.recommended_chunk_size
    overlap = int(chunk_size * profile.recommended_overlap_percent / 100)

    strategy_class = STRATEGY_MAP.get(strategy_name, SemanticHeaderStrategy)

    # WholeDocumentStrategy doesn't use size/overlap
    if strategy_name == "whole_document":
        return strategy_class(profile=profile)

    return strategy_class(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        profile=profile,
    )


def get_strategy_by_name(
    name: str,
    chunk_size: int = 512,
    chunk_overlap: int = 51,
    profile: Optional[ChunkingProfile] = None,
) -> ChunkingStrategy:
    """
    Get a chunking strategy by name.

    Args:
        name: Strategy name
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        profile: Optional document profile

    Returns:
        ChunkingStrategy instance
    """
    strategy_class = STRATEGY_MAP.get(name, SemanticHeaderStrategy)

    if name == "whole_document":
        return strategy_class(profile=profile)

    return strategy_class(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        profile=profile,
    )
