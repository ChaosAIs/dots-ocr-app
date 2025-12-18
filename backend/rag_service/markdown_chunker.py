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

from .utils.date_normalizer import find_and_normalize_dates, augment_text_with_dates

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

# Pattern to detect HTML tables
HTML_TABLE_PATTERN = re.compile(r'<table>.*?</table>', re.DOTALL | re.IGNORECASE)
# Pattern to detect markdown tables (rows with | separators)
MARKDOWN_TABLE_PATTERN = re.compile(r'(?:^\|.+\|$\n?)+', re.MULTILINE)


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


def extract_table_boundaries(content: str) -> List[Tuple[int, int, str]]:
    """
    Extract table boundaries from content (both HTML and Markdown tables).

    Args:
        content: The text content to analyze.

    Returns:
        List of tuples (start_pos, end_pos, table_type) for each table found.
    """
    boundaries = []

    # Find HTML tables
    for match in HTML_TABLE_PATTERN.finditer(content):
        boundaries.append((match.start(), match.end(), 'html'))

    # Find Markdown tables
    for match in MARKDOWN_TABLE_PATTERN.finditer(content):
        boundaries.append((match.start(), match.end(), 'markdown'))

    # Sort by start position
    boundaries.sort(key=lambda x: x[0])
    return boundaries


def split_html_table_by_rows(table_html: str, max_size: int = 1024) -> List[str]:
    """
    Split an HTML table by rows while preserving headers.

    Args:
        table_html: The HTML table string.
        max_size: Maximum size for each chunk.

    Returns:
        List of table chunks, each with headers preserved.
    """
    # Extract table parts
    thead_match = re.search(r'<thead>.*?</thead>', table_html, re.DOTALL | re.IGNORECASE)
    tbody_match = re.search(r'<tbody>(.*?)</tbody>', table_html, re.DOTALL | re.IGNORECASE)

    if not tbody_match:
        # No tbody, return whole table
        return [table_html]

    thead = thead_match.group(0) if thead_match else ''
    tbody_content = tbody_match.group(1)

    # Extract individual rows
    row_pattern = re.compile(r'<tr>.*?</tr>', re.DOTALL | re.IGNORECASE)
    rows = row_pattern.findall(tbody_content)

    if not rows:
        return [table_html]

    # Calculate header size
    header_size = len(f'<table>{thead}<tbody></tbody></table>')

    # If whole table fits, return it
    if len(table_html) <= max_size:
        return [table_html]

    # Split into chunks by rows
    chunks = []
    current_rows = []
    current_size = header_size

    for row in rows:
        row_size = len(row)
        if current_size + row_size > max_size and current_rows:
            # Create chunk with current rows
            chunk = f'<table>{thead}<tbody>{"".join(current_rows)}</tbody></table>'
            chunks.append(chunk)
            current_rows = [row]
            current_size = header_size + row_size
        else:
            current_rows.append(row)
            current_size += row_size

    # Add remaining rows
    if current_rows:
        chunk = f'<table>{thead}<tbody>{"".join(current_rows)}</tbody></table>'
        chunks.append(chunk)

    return chunks if chunks else [table_html]


def split_content_with_table_awareness(content: str, max_size: int = 1024) -> List[str]:
    """
    Split content while preserving table boundaries.

    Args:
        content: The text content to split.
        max_size: Maximum size for each chunk.

    Returns:
        List of content chunks with tables preserved.
    """
    table_boundaries = extract_table_boundaries(content)

    if not table_boundaries:
        # No tables, use regular splitting
        return None  # Signal to use regular splitter

    chunks = []
    last_pos = 0

    for start, end, table_type in table_boundaries:
        # Add text before table
        if start > last_pos:
            text_before = content[last_pos:start].strip()
            if text_before:
                chunks.append(text_before)

        # Add table (split if too large)
        table_content = content[start:end]
        if len(table_content) > max_size and table_type == 'html':
            # Split large HTML table by rows
            table_chunks = split_html_table_by_rows(table_content, max_size)
            chunks.extend(table_chunks)
        else:
            # Keep table intact
            chunks.append(table_content)

        last_pos = end

    # Add remaining text after last table
    if last_pos < len(content):
        text_after = content[last_pos:].strip()
        if text_after:
            chunks.append(text_after)

    return chunks


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

    # Date normalization: Find and augment dates for better vector search
    normalized_dates = find_and_normalize_dates(content)
    if normalized_dates:
        logger.info(f"[Chunker] Found {len(normalized_dates)} dates in {source_name}")
        # Augment content with normalized dates for embedding
        content = augment_text_with_dates(content, normalized_dates)
        logger.debug(f"[Chunker] Augmented content with normalized dates for {source_name}")

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

        # Find dates in this chunk for metadata
        chunk_dates = find_and_normalize_dates(page_content)
        date_metadata = {}
        if chunk_dates:
            date_metadata["dates_normalized"] = [d.normalized for d in chunk_dates]
            date_metadata["dates_raw"] = [d.raw for d in chunk_dates]
            # Add primary date if available
            primary = chunk_dates[0]  # First date in chunk
            date_metadata["primary_date"] = primary.normalized
            date_metadata["date_year"] = primary.year
            date_metadata["date_month"] = primary.month
            date_metadata["date_day"] = primary.day

        # Check if chunk needs further splitting
        if len(page_content) > MAX_CHUNK_SIZE_FOR_SPLIT:
            # Try table-aware splitting first
            table_aware_chunks = split_content_with_table_awareness(
                page_content, MAX_CHUNK_SIZE_FOR_SPLIT
            )

            if table_aware_chunks:
                # Use table-aware chunks
                for j, sub_content in enumerate(table_aware_chunks):
                    chunk_id = str(uuid.uuid4())
                    final_chunks.append(
                        Document(
                            page_content=sub_content,
                            metadata={
                                **chunk.metadata,
                                **date_metadata,  # Add date metadata
                                "source": source_name,
                                "file_path": str(md_path),
                                "heading_path": heading_path,
                                "chunk_type": "table_aware",
                                "chunk_index": j,
                                "parent_chunk_index": i,
                                "chunk_id": chunk_id,
                            },
                        )
                    )
            else:
                # Fall back to recursive splitting (no tables found)
                sub_chunks = recursive_splitter.split_text(page_content)
                for j, sub_content in enumerate(sub_chunks):
                    chunk_id = str(uuid.uuid4())
                    final_chunks.append(
                        Document(
                            page_content=sub_content,
                            metadata={
                                **chunk.metadata,
                                **date_metadata,  # Add date metadata
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
                        **date_metadata,  # Add date metadata
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
            # Try table-aware splitting first
            table_aware_chunks = split_content_with_table_awareness(
                page_content, MAX_CHUNK_SIZE_FOR_SPLIT
            )

            if table_aware_chunks:
                # Use table-aware chunks
                for j, sub_content in enumerate(table_aware_chunks):
                    final_chunks.append(
                        Document(
                            page_content=sub_content,
                            metadata={
                                "source": source_name,
                                "file_path": file_path or "",
                                "chunk_type": "table_aware",
                                "chunk_index": j,
                            },
                        )
                    )
            else:
                # Fall back to recursive splitting
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
