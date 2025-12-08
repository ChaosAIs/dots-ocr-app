"""
Markdown chunking utilities for semantic document splitting.
Uses LangChain's text splitters to chunk markdown files by headers.
Includes LLM-based summarization for enhanced RAG retrieval.
"""

import logging
import re
import uuid
from typing import List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# Minimum content size for chunk summarization
MIN_CONTENT_SIZE_FOR_SUMMARIZATION = 1000


@dataclass
class ChunkSummaryInfo:
    """Information about a chunk summary for embedding.

    A single summary may cover multiple chunks when individual chunks are small.
    """
    chunk_indices: List[int]  # List of chunk indices covered by this summary
    chunk_ids: List[str]  # List of chunk_ids covered by this summary
    summary: str
    heading_path: str = ""  # Combined heading path for the chunks


@dataclass
class ChunkingResult:
    """Result of chunking a markdown file with summaries."""
    chunks: List[Document]
    file_summary: str
    chunk_summaries: List[str]
    chunk_summary_infos: List[ChunkSummaryInfo] = field(default_factory=list)

# Headers to split on for markdown documents
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
]

# Chunk size parameters
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MAX_CHUNK_SIZE_FOR_SPLIT = 1400

# Pattern to detect base64 encoded content
BASE64_PATTERN = re.compile(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]{100,}')
# Pattern to detect long base64-like strings (without data: prefix)
BASE64_LIKE_PATTERN = re.compile(r'[A-Za-z0-9+/=]{200,}')


def is_mostly_base64(content: str, threshold: float = 0.5) -> bool:
    """
    Check if content is mostly base64 encoded data.

    Args:
        content: The text content to check.
        threshold: Ratio threshold above which content is considered base64.

    Returns:
        True if content appears to be mostly base64 data.
    """
    if not content or len(content) < 50:
        return False

    # Remove base64 image tags and check remaining content
    cleaned = BASE64_PATTERN.sub('', content)
    cleaned = BASE64_LIKE_PATTERN.sub('', cleaned)

    # If most of the content was removed, it's base64
    original_len = len(content)
    cleaned_len = len(cleaned.strip())

    if original_len > 0 and (cleaned_len / original_len) < (1 - threshold):
        return True

    return False


def clean_base64_from_content(content: str) -> str:
    """
    Remove base64 encoded images from content while preserving text.

    Args:
        content: The markdown content.

    Returns:
        Content with base64 images replaced with placeholder.
    """
    # Replace base64 images with placeholder
    cleaned = BASE64_PATTERN.sub('[image]', content)
    return cleaned


def chunk_markdown_file(md_path: str, source_name: str = None) -> List[Document]:
    """
    Chunk a markdown file into semantic chunks based on headers.

    Args:
        md_path: Path to the markdown file.
        source_name: Name to use as source metadata. If None, uses the filename.

    Returns:
        List of Document objects with chunked content and metadata.
    """
    md_path = Path(md_path)

    if not md_path.exists():
        logger.error(f"Markdown file not found: {md_path}")
        return []

    # Use filename as source name if not provided
    if source_name is None:
        source_name = md_path.stem

    try:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Error reading markdown file {md_path}: {e}")
        return []

    if not content.strip():
        logger.warning(f"Empty markdown file: {md_path}")
        return []

    # Clean base64 images from content before chunking
    content = clean_base64_from_content(content)

    # Split by headers first
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    try:
        header_chunks = header_splitter.split_text(content)
    except Exception as e:
        logger.error(f"Error splitting by headers: {e}")
        # Fall back to treating entire content as one chunk
        header_chunks = [
            Document(page_content=content, metadata={"source": source_name})
        ]

    # Secondary splitter for large chunks
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    final_chunks = []

    for i, chunk in enumerate(header_chunks):
        # Build heading path from metadata
        heading_parts = []
        for key in ["Header 1", "Header 2", "Header 3", "Header 4", "Header 5"]:
            if key in chunk.metadata and chunk.metadata[key]:
                heading_parts.append(chunk.metadata[key])

        heading_path = " → ".join(heading_parts) if heading_parts else "General"
        page_content = chunk.page_content

        # Skip chunks that are mostly base64 data
        if is_mostly_base64(page_content):
            logger.debug(f"Skipping base64 chunk {i} in {md_path}")
            continue

        # Check if chunk needs further splitting
        if len(page_content) > MAX_CHUNK_SIZE_FOR_SPLIT:
            # Split large chunks recursively
            sub_chunks = recursive_splitter.split_text(page_content)
            for j, sub_content in enumerate(sub_chunks):
                # Skip sub-chunks that are mostly base64
                if is_mostly_base64(sub_content):
                    continue
                # Generate unique chunk_id for each chunk
                chunk_id = str(uuid.uuid4())
                final_chunks.append(
                    Document(
                        page_content=sub_content,
                        metadata={
                            **chunk.metadata,
                            "source": source_name,
                            "file_path": str(md_path),
                            "heading_path": heading_path,
                            "chunk_type": "recursive",
                            "chunk_index": j,
                            "parent_chunk_index": i,
                            "chunk_id": chunk_id,
                        },
                    )
                )
        else:
            # Generate unique chunk_id for each chunk
            chunk_id = str(uuid.uuid4())
            # Keep chunk as-is
            final_chunks.append(
                Document(
                    page_content=page_content,
                    metadata={
                        **chunk.metadata,
                        "source": source_name,
                        "file_path": str(md_path),
                        "heading_path": heading_path,
                        "chunk_type": "semantic_header",
                        "chunk_index": i,
                        "chunk_id": chunk_id,
                    },
                )
            )

    logger.info(f"Chunked {md_path} into {len(final_chunks)} chunks")
    return final_chunks


def chunk_text_content(
    content: str,
    source_name: str,
    file_path: str = None,
) -> List[Document]:
    """
    Chunk text content directly (for non-file content).

    Args:
        content: The text content to chunk.
        source_name: Name to use as source metadata.
        file_path: Optional file path for metadata.

    Returns:
        List of Document objects with chunked content.
    """
    if not content.strip():
        return []

    chunks = chunk_markdown_file.__wrapped__ if hasattr(chunk_markdown_file, "__wrapped__") else chunk_markdown_file

    # Create a temporary approach by using the same logic
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    try:
        header_chunks = header_splitter.split_text(content)
    except Exception:
        header_chunks = [Document(page_content=content, metadata={})]

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    final_chunks = []
    for i, chunk in enumerate(header_chunks):
        page_content = chunk.page_content
        if len(page_content) > MAX_CHUNK_SIZE_FOR_SPLIT:
            sub_chunks = recursive_splitter.split_text(page_content)
            for j, sub_content in enumerate(sub_chunks):
                final_chunks.append(
                    Document(
                        page_content=sub_content,
                        metadata={
                            "source": source_name,
                            "file_path": file_path or "",
                            "chunk_type": "recursive",
                            "chunk_index": j,
                        },
                    )
                )
        else:
            final_chunks.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "source": source_name,
                        "file_path": file_path or "",
                        "chunk_type": "semantic_header",
                        "chunk_index": i,
                    },
                )
            )

    return final_chunks


def chunk_markdown_with_summaries(
    md_path: str,
    source_name: str = None,
    generate_summaries: bool = True,
) -> ChunkingResult:
    """
    Chunk a markdown file with LLM-based summarization.

    This function:
    1. Chunks the markdown file into semantic chunks based on headers
    2. Combines small chunks (< MIN_CONTENT_SIZE_FOR_SUMMARIZATION) for efficient summarization
    3. Skips chunk summarization if total file content < MIN_CONTENT_SIZE_FOR_SUMMARIZATION
    4. Generates a summary for each chunk group using LLM
    5. Combines chunk summaries to generate a comprehensive file summary
    6. Returns chunk summary infos for embedding in chunk summary collection

    Args:
        md_path: Path to the markdown file.
        source_name: Name to use as source metadata. If None, uses the filename.
        generate_summaries: Whether to generate LLM summaries. If False, uses truncated content.

    Returns:
        ChunkingResult with chunks, file_summary, chunk_summaries, and chunk_summary_infos.
    """
    from .summarizer import summarize_chunk, generate_file_summary, ChunkSummary

    # First, get the regular chunks
    chunks = chunk_markdown_file(md_path, source_name)

    if not chunks:
        return ChunkingResult(
            chunks=[],
            file_summary="",
            chunk_summaries=[],
            chunk_summary_infos=[],
        )

    # Determine source name for file summary
    if source_name is None:
        source_name = Path(md_path).stem

    # Calculate total content size
    total_content_size = sum(len(chunk.page_content) for chunk in chunks)

    # If total file content is too small, skip chunk summarization
    skip_chunk_summaries = total_content_size < MIN_CONTENT_SIZE_FOR_SUMMARIZATION
    if skip_chunk_summaries:
        logger.info(
            f"File {source_name} has {total_content_size} chars (< {MIN_CONTENT_SIZE_FOR_SUMMARIZATION}), "
            "skipping chunk summarization"
        )

    chunk_summary_objects: List[ChunkSummary] = []
    chunk_summaries: List[str] = []
    chunk_summary_infos: List[ChunkSummaryInfo] = []

    if skip_chunk_summaries:
        # Skip chunk summarization - just use truncated content for each chunk
        for i, chunk in enumerate(chunks):
            heading_path = chunk.metadata.get("heading_path", "")
            summary = chunk.page_content[:300].strip()
            chunk_summaries.append(summary)
            chunk_summary_objects.append(ChunkSummary(
                chunk_index=i,
                summary=summary,
                heading_path=heading_path,
            ))
        # No chunk_summary_infos when skipping (nothing to embed)
    else:
        # Group chunks for summarization - combine small chunks together
        chunk_groups = _group_chunks_for_summarization(chunks)

        for group_indices in chunk_groups:
            group_chunks = [chunks[i] for i in group_indices]
            group_chunk_ids = [chunk.metadata.get("chunk_id", "") for chunk in group_chunks]

            # Combine content from all chunks in the group
            combined_content = "\n\n".join(chunk.page_content for chunk in group_chunks)

            # Combine heading paths
            heading_paths = [chunk.metadata.get("heading_path", "") for chunk in group_chunks]
            unique_headings = []
            for hp in heading_paths:
                if hp and hp not in unique_headings:
                    unique_headings.append(hp)
            combined_heading_path = " | ".join(unique_headings) if unique_headings else ""

            if generate_summaries:
                summary = summarize_chunk(combined_content, max_words=200)
            else:
                summary = combined_content[:300].strip()

            # Store summary for each chunk in the group (for file summary generation)
            for idx in group_indices:
                chunk_summaries.append(summary)
                chunk_summary_objects.append(ChunkSummary(
                    chunk_index=idx,
                    summary=summary,
                    heading_path=chunks[idx].metadata.get("heading_path", ""),
                ))

            # Build chunk summary info for embedding (one per group)
            chunk_summary_infos.append(ChunkSummaryInfo(
                chunk_indices=group_indices,
                chunk_ids=group_chunk_ids,
                summary=summary,
                heading_path=combined_heading_path,
            ))

            logger.debug(
                f"Summarized chunk group {group_indices}: {summary[:50]}..."
            )

    # Generate comprehensive file summary from chunk summaries
    if generate_summaries:
        file_summary = generate_file_summary(source_name, chunk_summary_objects)
    else:
        file_summary = f"Document: {source_name}\n" + "\n".join(chunk_summaries[:5])

    logger.info(f"Generated file summary for {source_name}: {file_summary[:100]}...")

    return ChunkingResult(
        chunks=chunks,
        file_summary=file_summary,
        chunk_summaries=chunk_summaries,
        chunk_summary_infos=chunk_summary_infos,
    )


def _group_chunks_for_summarization(chunks: List[Document]) -> List[List[int]]:
    """
    Group chunks for summarization based on content size.

    Combines small chunks together until their combined content size
    is >= MIN_CONTENT_SIZE_FOR_SUMMARIZATION.

    Args:
        chunks: List of document chunks.

    Returns:
        List of chunk index groups, where each group should be summarized together.
    """
    groups = []
    current_group = []
    current_size = 0

    for i, chunk in enumerate(chunks):
        chunk_size = len(chunk.page_content)

        if current_size + chunk_size >= MIN_CONTENT_SIZE_FOR_SUMMARIZATION:
            if current_group:
                # Current group is big enough, finalize it
                current_group.append(i)
                groups.append(current_group)
                current_group = []
                current_size = 0
            else:
                # Single chunk is big enough
                groups.append([i])
        else:
            # Add to current group
            current_group.append(i)
            current_size += chunk_size

    # Handle remaining chunks
    if current_group:
        if groups:
            # Merge remaining small chunks with the last group
            groups[-1].extend(current_group)
        else:
            # All chunks are small, create one group
            groups.append(current_group)

    return groups
