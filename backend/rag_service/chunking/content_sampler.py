"""
Content Sampler Module for LLM-Driven Adaptive Chunking.

This module handles content sampling from documents for LLM structure analysis.
It supports two scenarios:
- Multi-page OCR documents: Page-based sampling (first 3 + middle 2 + last 2 pages)
- Single-file conversions: Character-based sampling (first 3000 + middle 1000 + last 1000 chars)
"""

import glob
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SampledContent:
    """Container for sampled content from a document."""
    first_content: str
    middle_content: str
    last_content: str
    scenario: str  # "multi_page" or "single_file"
    total_pages: int = 0
    total_chars: int = 0

    def get_combined_sample(self) -> str:
        """Get combined sample for LLM prompt."""
        return f"""=== DOCUMENT START ===
{self.first_content}

=== DOCUMENT MIDDLE ===
{self.middle_content}

=== DOCUMENT END ===
{self.last_content}"""


class ContentSampler:
    """
    Samples content from documents for LLM structure analysis.

    Handles both multi-page OCR documents and single-file conversions.
    """

    # Configuration
    FIRST_CHARS = 3000
    MIDDLE_CHARS = 1000
    LAST_CHARS = 1000
    SHORT_DOC_THRESHOLD = 5000  # Documents shorter than this use entire content

    # Page sampling configuration
    FIRST_PAGES = 3
    MIDDLE_PAGES = 2
    LAST_PAGES = 2

    def __init__(self):
        """Initialize the content sampler."""
        pass

    def detect_scenario(self, output_folder: str) -> Tuple[str, List[str]]:
        """
        Detect processing scenario based on output file count.

        Args:
            output_folder: Path like /output/{username}/{workspace}/{filename}/

        Returns:
            Tuple of (scenario: "multi_page" or "single_file", list of nohf files)
        """
        # Find all *_nohf.md files in the output folder
        pattern = os.path.join(output_folder, "*_nohf.md")
        nohf_files = sorted(glob.glob(pattern))

        if len(nohf_files) == 0:
            logger.warning(f"No *_nohf.md files found in {output_folder}")
            return "single_file", []
        elif len(nohf_files) == 1:
            return "single_file", nohf_files
        else:
            return "multi_page", nohf_files

    def sample_from_folder(self, output_folder: str) -> SampledContent:
        """
        Sample content from an output folder containing processed document(s).

        Args:
            output_folder: Path to folder containing *_nohf.md files

        Returns:
            SampledContent with first, middle, and last content samples
        """
        scenario, nohf_files = self.detect_scenario(output_folder)

        if not nohf_files:
            logger.warning(f"No files to sample from in {output_folder}")
            return SampledContent(
                first_content="",
                middle_content="",
                last_content="",
                scenario=scenario,
                total_pages=0,
                total_chars=0
            )

        if scenario == "multi_page":
            return self._sample_multi_page(nohf_files)
        else:
            return self._sample_single_file(nohf_files[0])

    def sample_from_content(self, content: str) -> SampledContent:
        """
        Sample content directly from a string (for single file scenario).

        Args:
            content: Full document content

        Returns:
            SampledContent with character-based samples
        """
        total_chars = len(content)

        # Short document: use entire content
        if total_chars < self.SHORT_DOC_THRESHOLD:
            return SampledContent(
                first_content=content,
                middle_content="",
                last_content="",
                scenario="single_file",
                total_pages=1,
                total_chars=total_chars
            )

        # Sample first, middle, and last regions
        first_content = content[:self.FIRST_CHARS]

        # Middle: center the sample around the midpoint
        mid_point = total_chars // 2
        mid_start = max(0, mid_point - self.MIDDLE_CHARS // 2)
        mid_end = min(total_chars, mid_start + self.MIDDLE_CHARS)
        middle_content = content[mid_start:mid_end]

        # Last content
        last_content = content[-self.LAST_CHARS:]

        return SampledContent(
            first_content=first_content,
            middle_content=middle_content,
            last_content=last_content,
            scenario="single_file",
            total_pages=1,
            total_chars=total_chars
        )

    def _sample_multi_page(self, nohf_files: List[str]) -> SampledContent:
        """
        Sample pages from multi-page document.

        Strategy: First 3 + Middle 2 + Last 2 pages
        For small documents, overlap is acceptable.

        Args:
            nohf_files: Sorted list of *_nohf.md file paths

        Returns:
            SampledContent with page-based samples
        """
        total_pages = len(nohf_files)

        # Determine which pages to read
        pages_to_read = self._calculate_page_indices(total_pages)

        # Read and combine content for each region
        first_indices = pages_to_read['first']
        middle_indices = pages_to_read['middle']
        last_indices = pages_to_read['last']

        first_content = self._read_pages(nohf_files, first_indices)
        middle_content = self._read_pages(nohf_files, middle_indices)
        last_content = self._read_pages(nohf_files, last_indices)

        total_chars = len(first_content) + len(middle_content) + len(last_content)

        logger.info(
            f"Multi-page sampling: {total_pages} pages, "
            f"reading first={first_indices}, middle={middle_indices}, last={last_indices}, "
            f"total chars sampled={total_chars}"
        )

        return SampledContent(
            first_content=first_content,
            middle_content=middle_content,
            last_content=last_content,
            scenario="multi_page",
            total_pages=total_pages,
            total_chars=total_chars
        )

    def _calculate_page_indices(self, total_pages: int) -> dict:
        """
        Calculate which page indices to read for each region.

        Args:
            total_pages: Total number of pages

        Returns:
            Dict with 'first', 'middle', 'last' keys containing lists of 0-based indices
        """
        if total_pages <= 0:
            return {'first': [], 'middle': [], 'last': []}

        if total_pages == 1:
            # Single page: read it for all regions
            return {'first': [0], 'middle': [], 'last': []}

        if total_pages == 2:
            # Two pages: read both
            return {'first': [0], 'middle': [], 'last': [1]}

        if total_pages == 3:
            # Three pages: read all
            return {'first': [0, 1], 'middle': [1], 'last': [2]}

        if total_pages == 4:
            # Four pages
            return {'first': [0, 1], 'middle': [1, 2], 'last': [2, 3]}

        # For larger documents:
        # First: pages 1, 2, 3 (indices 0, 1, 2)
        first = list(range(min(self.FIRST_PAGES, total_pages)))

        # Middle: 2 pages centered at midpoint
        mid_center = total_pages // 2
        mid_start = max(0, mid_center - self.MIDDLE_PAGES // 2)
        middle = list(range(mid_start, min(mid_start + self.MIDDLE_PAGES, total_pages)))

        # Last: last 2 pages
        last = list(range(max(0, total_pages - self.LAST_PAGES), total_pages))

        return {'first': first, 'middle': middle, 'last': last}

    def _read_pages(self, files: List[str], indices: List[int]) -> str:
        """
        Read and combine content from specified page files.

        Args:
            files: List of file paths
            indices: 0-based indices of files to read

        Returns:
            Combined content from all specified pages
        """
        # Import image cleanup utility
        from ..markdown_chunker import clean_markdown_images

        contents = []
        for idx in indices:
            if 0 <= idx < len(files):
                try:
                    with open(files[idx], 'r', encoding='utf-8') as f:
                        raw_content = f.read()
                        # Clean embedded base64 images before processing
                        page_content = clean_markdown_images(raw_content)
                        # Add page marker for context
                        page_num = self._extract_page_number(files[idx])
                        if page_num:
                            contents.append(f"--- Page {page_num} ---\n{page_content}")
                        else:
                            contents.append(page_content)
                except Exception as e:
                    logger.error(f"Error reading page file {files[idx]}: {e}")

        return "\n\n".join(contents)

    def _extract_page_number(self, filepath: str) -> Optional[int]:
        """
        Extract page number from filename like 'page_1_nohf.md'.

        Args:
            filepath: Path to the file

        Returns:
            Page number or None if not found
        """
        filename = os.path.basename(filepath)
        # Try to match patterns like: page_1_nohf.md, Page_01_nohf.md, 1_nohf.md
        match = re.search(r'(?:page[_-]?)?(\d+)[_-]nohf\.md$', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _sample_single_file(self, filepath: str) -> SampledContent:
        """
        Sample content from a single markdown file.

        Args:
            filepath: Path to the *_nohf.md file

        Returns:
            SampledContent with character-based samples
        """
        from ..markdown_chunker import clean_markdown_images

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            # Clean embedded base64 images before processing
            content = clean_markdown_images(raw_content)
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            return SampledContent(
                first_content="",
                middle_content="",
                last_content="",
                scenario="single_file",
                total_pages=1,
                total_chars=0
            )

        result = self.sample_from_content(content)

        logger.info(
            f"Single-file sampling: {filepath}, "
            f"total chars={result.total_chars}, "
            f"sampled={len(result.first_content) + len(result.middle_content) + len(result.last_content)}"
        )

        return result


# Convenience function
def sample_content(output_folder: str) -> SampledContent:
    """
    Sample content from a document output folder.

    Args:
        output_folder: Path to folder containing *_nohf.md files

    Returns:
        SampledContent with samples from the document
    """
    sampler = ContentSampler()
    return sampler.sample_from_folder(output_folder)
