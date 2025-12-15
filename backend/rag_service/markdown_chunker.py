"""
Markdown chunking utilities for semantic document splitting.
Uses LangChain's text splitters to chunk markdown files by headers.
"""

import logging
import re
import uuid
from typing import List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class ChunkingResult:
    """Result of chunking a markdown file."""
    chunks: List[Document]
    file_summary: str = ""

# Headers to split on for markdown documents
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
]

# Chunk size parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 51  # 10% of the chunk_size
MAX_CHUNK_SIZE_FOR_SPLIT = 1024  # Max size for recursive splitting

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
            # content is whole file content.
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
    Chunk a markdown file.

    Note: The 'generate_summaries' parameter is deprecated and ignored.
    File summary generation has been removed due to performance issues.

    Args:
        md_path: Path to the markdown file.
        source_name: Name to use as source metadata. If None, uses the filename.
        generate_summaries: Deprecated, ignored. Kept for backward compatibility.

    Returns:
        ChunkingResult with chunks and empty file_summary.
    """
    # Get the chunks
    chunks = chunk_markdown_file(md_path, source_name)

    if not chunks:
        return ChunkingResult(chunks=[])

    # Determine source name for logging
    if source_name is None:
        source_name = Path(md_path).stem

    logger.info(f"Chunked {source_name}: {len(chunks)} chunks")

    return ChunkingResult(chunks=chunks)
