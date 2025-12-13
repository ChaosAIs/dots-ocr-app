"""
LLM-based summarization for RAG chunks and documents.
Uses the configured LLM backend (Ollama or vLLM) for generating summaries.
"""

import os
import re
import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

from .llm_service import get_llm_service

logger = logging.getLogger(__name__)


@dataclass
class FileSummaryWithScopes:
    """Enhanced file summary with topic scopes for filtering."""
    summary: str
    scopes: List[str] = field(default_factory=list)
    content_type: str = "document"
    complexity: str = "intermediate"


# Constants for map-reduce summarization
MAX_DIRECT_SUMMARY_SIZE = 8000  # Characters - files larger than this use map-reduce
MAX_SECTION_SIZE = 4000  # Characters per section for map-reduce
MAX_SECTIONS_TO_PROCESS = 50  # Maximum sections to process in map-reduce
MAX_RECURSION_DEPTH = 5  # Maximum recursion depth for nested summarization


# Prompt for generating file summary with scopes (for direct summarization)
FILE_SUMMARY_WITH_SCOPES_PROMPT = """Analyze this document and provide a comprehensive analysis.

Document filename: {filename}

Document content:
{content}

Provide your analysis in the following JSON format:
{{
    "summary": "A comprehensive summary (400-500 words) covering: main purpose, key concepts, important details, procedures, and target audience",
    "scopes": ["topic1", "topic2", "topic3", ...],
    "content_type": "one of: technical_documentation, tutorial, reference, guide, specification, report, api_doc, architecture, policy, manual",
    "complexity": "one of: basic, intermediate, advanced"
}}

Requirements:
1. SUMMARY: Write 400-500 words covering the document's main purpose, key concepts, important details, and target audience
2. SCOPES: List 8-15 topic keywords/areas this document covers (e.g., "authentication", "JWT tokens", "user management")
3. CONTENT_TYPE: Classify the document type
4. COMPLEXITY: Assess the technical complexity level

Output ONLY valid JSON, nothing else."""


# Prompt for summarizing a section (for map-reduce)
SECTION_SUMMARY_PROMPT = """Summarize this section of a document in 100-250 words.
Focus on key concepts, procedures, and important details.

Section content:
{content}

Output ONLY the summary, nothing else."""


# Prompt for combining section summaries into final summary with scopes
COMBINE_SUMMARIES_PROMPT = """Based on these section summaries from a document, create a comprehensive file analysis.

Document filename: {filename}

Section summaries:
{section_summaries}

Provide your analysis in the following JSON format:
{{
    "summary": "A comprehensive summary (400-500 words) synthesizing all sections",
    "scopes": ["topic1", "topic2", "topic3", ...],
    "content_type": "one of: technical_documentation, tutorial, reference, guide, specification, report, api_doc, architecture, policy, manual",
    "complexity": "one of: basic, intermediate, advanced"
}}

Requirements:
1. SUMMARY: Write 400-500 words synthesizing all section summaries into a coherent overview
2. SCOPES: List 8-15 topic keywords/areas covered across all sections
3. CONTENT_TYPE: Classify the overall document type
4. COMPLEXITY: Assess the overall technical complexity

Output ONLY valid JSON, nothing else."""


def _create_summary_llm():
    """Create the LLM instance for section summarization using configured backend."""
    llm_service = get_llm_service()
    return llm_service.get_query_model(
        temperature=0.1,  # Low temperature for consistent summaries
        num_ctx=4096,
        num_predict=256,  # Limit output tokens
    )


def _create_summary_llm_extended():
    """Create the LLM instance for extended summarization (larger output) using configured backend."""
    llm_service = get_llm_service()
    return llm_service.get_query_model(
        temperature=0.1,
        num_ctx=8192,  # Larger context for file summaries
        num_predict=1024,  # Allow longer output for 500 word summaries
    )


def _split_by_headers(content: str, max_section_size: int = MAX_SECTION_SIZE) -> List[str]:
    """
    Split content by markdown headers for map-reduce summarization.
    Recursively splits sections that are still too large.

    Args:
        content: The markdown content to split.
        max_section_size: Maximum allowed size for each section.

    Returns:
        List of content sections, each within max_section_size.
    """
    # Pattern to match markdown headers (# to ######)
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    sections = []
    last_end = 0

    for match in header_pattern.finditer(content):
        # Get content before this header (if any)
        if match.start() > last_end:
            section_content = content[last_end:match.start()].strip()
            if section_content and len(section_content) > 100:
                sections.append(section_content)

        last_end = match.start()

    # Get remaining content after last header
    if last_end < len(content):
        section_content = content[last_end:].strip()
        if section_content and len(section_content) > 100:
            sections.append(section_content)

    # If no headers found or only one section, split by size
    if len(sections) <= 1:
        sections = _split_by_size(content, max_section_size)
    else:
        # Recursively split any sections that are still too large
        final_sections = []
        for section in sections:
            if len(section) > max_section_size:
                # Try to split by sub-headers first, then by size
                sub_sections = _split_large_section(section, max_section_size)
                final_sections.extend(sub_sections)
            else:
                final_sections.append(section)
        sections = final_sections

    return sections[:MAX_SECTIONS_TO_PROCESS]


def _split_large_section(content: str, max_section_size: int) -> List[str]:
    """
    Recursively split a large section into smaller chunks.
    First tries to split by sub-headers (##, ###, etc.), then by paragraphs, then by size.

    Args:
        content: The section content to split.
        max_section_size: Maximum allowed size for each section.

    Returns:
        List of smaller sections.
    """
    if len(content) <= max_section_size:
        return [content]

    # Try to find sub-headers to split on
    header_pattern = re.compile(r'^(#{2,6})\s+(.+)$', re.MULTILINE)
    header_matches = list(header_pattern.finditer(content))

    if header_matches:
        # Split by sub-headers
        sections = []
        last_end = 0
        for match in header_matches:
            if match.start() > last_end:
                section_content = content[last_end:match.start()].strip()
                if section_content and len(section_content) > 100:
                    sections.append(section_content)
            last_end = match.start()

        # Add remaining content
        if last_end < len(content):
            section_content = content[last_end:].strip()
            if section_content and len(section_content) > 100:
                sections.append(section_content)

        if len(sections) > 1:
            # Recursively split any sections that are still too large
            final_sections = []
            for section in sections:
                if len(section) > max_section_size:
                    sub_sections = _split_large_section(section, max_section_size)
                    final_sections.extend(sub_sections)
                else:
                    final_sections.append(section)
            return final_sections

    # Try to split by paragraph breaks (double newlines)
    paragraphs = content.split('\n\n')
    if len(paragraphs) > 1:
        # Group paragraphs into chunks that fit max_section_size
        sections = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para) + 2  # +2 for the \n\n separator

            if current_size + para_size > max_section_size and current_chunk:
                # Save current chunk and start new one
                sections.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        if current_chunk:
            sections.append('\n\n'.join(current_chunk))

        if len(sections) > 1:
            # Check if any section is still too large and split further
            final_sections = []
            for section in sections:
                if len(section) > max_section_size:
                    # Last resort: split by size
                    sub_sections = _split_by_size(section, max_section_size)
                    final_sections.extend(sub_sections)
                else:
                    final_sections.append(section)
            return final_sections

    # Fallback: split by size
    return _split_by_size(content, max_section_size)


def _split_by_size(content: str, max_size: int) -> List[str]:
    """
    Split content into chunks of approximately max_size characters.

    Args:
        content: The content to split.
        max_size: Maximum size per chunk.

    Returns:
        List of content chunks.
    """
    if len(content) <= max_size:
        return [content]

    sections = []
    current_pos = 0

    while current_pos < len(content):
        end_pos = min(current_pos + max_size, len(content))

        # Try to break at paragraph boundary
        if end_pos < len(content):
            # Look for paragraph break
            para_break = content.rfind('\n\n', current_pos, end_pos)
            if para_break > current_pos + max_size // 2:
                end_pos = para_break

        section = content[current_pos:end_pos].strip()
        if section:
            sections.append(section)

        current_pos = end_pos

    return sections


def _parse_json_response(response: str) -> Dict[str, Any]:
    """
    Parse JSON response from LLM, handling common issues.

    Args:
        response: The LLM response string.

    Returns:
        Parsed dictionary or default values.
    """
    # Clean up response
    cleaned = response.strip()

    # Remove thinking tags if present
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned)
    if json_match:
        cleaned = json_match.group(1).strip()

    # Try to find JSON object
    json_start = cleaned.find('{')
    json_end = cleaned.rfind('}')
    if json_start != -1 and json_end != -1:
        cleaned = cleaned[json_start:json_end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        return {}


def _summarize_section(content: str, depth: int = 0) -> str:
    """
    Summarize a single section for map-reduce.
    For very large sections, recursively splits and summarizes sub-sections first.

    Args:
        content: The section content.
        depth: Current recursion depth (to prevent infinite recursion).

    Returns:
        Section summary string.
    """
    content_len = len(content)
    logger.debug(f"[SECTION] Processing section: {content_len} chars (depth={depth})")

    if content_len < 200:
        logger.debug(f"[SECTION] Short section, returning as-is: '{content[:100]}...'")
        return content

    # If section is too large for direct summarization, use recursive map-reduce
    if content_len > MAX_SECTION_SIZE and depth < MAX_RECURSION_DEPTH:
        logger.info(f"[SECTION] Section too large ({content_len} chars), using recursive summarization (depth={depth})")

        # Split the large section into smaller sub-sections
        sub_sections = _split_large_section(content, MAX_SECTION_SIZE)
        logger.info(f"[SECTION] Split into {len(sub_sections)} sub-sections")

        if len(sub_sections) > 1:
            # Recursively summarize each sub-section
            sub_summaries = []
            for i, sub_section in enumerate(sub_sections):
                logger.debug(f"[SECTION] Processing sub-section {i+1}/{len(sub_sections)} ({len(sub_section)} chars)")
                sub_summary = _summarize_section(sub_section, depth + 1)
                sub_summaries.append(sub_summary)

            # Combine sub-summaries
            combined = " ".join(sub_summaries)

            # If combined summaries are still too large, summarize them again
            if len(combined) > MAX_SECTION_SIZE:
                logger.info(f"[SECTION] Combined sub-summaries still large ({len(combined)} chars), summarizing again")
                return _summarize_section(combined, depth + 1)

            # If combined summaries are reasonable, summarize them into a single summary
            logger.info(f"[SECTION] Combining {len(sub_summaries)} sub-summaries ({len(combined)} chars)")
            try:
                llm = _create_summary_llm()
                prompt = f"""Synthesize these sub-section summaries into a single coherent summary (100-250 words):

{combined}

Output ONLY the summary, nothing else."""
                response = llm.invoke([HumanMessage(content=prompt)])
                summary = response.content.strip()

                # Clean up thinking tags
                if "</think>" in summary:
                    summary = summary.split("</think>")[-1].strip()

                summary = summary[:500]  # Safety limit
                logger.info(f"[SECTION] Combined summary result ({len(summary)} chars): '{summary[:200]}...'")
                return summary

            except Exception as e:
                logger.warning(f"[SECTION] Combining summaries failed: {e}")
                # Fallback: return truncated combined summaries
                return combined[:500]

    # Direct summarization for sections within size limit
    logger.info(f"[SECTION] Summarizing section: {content_len} chars")
    logger.debug(f"[SECTION] Input text (first 500 chars): '{content[:500]}...'")

    try:
        llm = _create_summary_llm()
        prompt = SECTION_SUMMARY_PROMPT.format(content=content[:MAX_SECTION_SIZE])
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()

        # Clean up thinking tags
        if "</think>" in summary:
            summary = summary.split("</think>")[-1].strip()

        summary = summary[:500]  # Safety limit
        logger.info(f"[SECTION] Summary result ({len(summary)} chars): '{summary[:200]}...'")
        return summary

    except Exception as e:
        logger.warning(f"[SECTION] Summarization failed: {e}")
        result = content[:300]
        logger.debug(f"[SECTION] Fallback result: '{result[:100]}...'")
        return result


def generate_file_summary_with_scopes(
    filename: str,
    content: str,
) -> FileSummaryWithScopes:
    """
    Generate a comprehensive file summary with topic scopes.

    This function:
    1. For small files (< MAX_DIRECT_SUMMARY_SIZE): Direct LLM summarization
    2. For large files: Map-reduce approach
       - Split content by headers or size
       - Summarize each section
       - Combine section summaries into final summary with scopes

    Args:
        filename: The document filename.
        content: The full document content.

    Returns:
        FileSummaryWithScopes object with summary, scopes, content_type, complexity.
    """
    content_len = len(content) if content else 0
    logger.info(f"[FILE_SUMMARY] Starting summary generation for '{filename}': {content_len} chars")

    if not content or len(content.strip()) < 50:
        logger.warning(f"[FILE_SUMMARY] No content or too short for '{filename}', skipping")
        return FileSummaryWithScopes(
            summary=f"Document: {filename}\nNo content available for summarization.",
            scopes=[],
            content_type="document",
            complexity="basic",
        )

    logger.debug(f"[FILE_SUMMARY] Content preview (first 1000 chars): '{content[:1000]}...'")

    try:
        llm = _create_summary_llm_extended()

        # Decide between direct summarization and map-reduce
        if len(content) <= MAX_DIRECT_SUMMARY_SIZE:
            # Direct summarization for smaller files
            logger.info(f"[FILE_SUMMARY] Using DIRECT summarization for {filename} ({len(content)} chars)")
            prompt = FILE_SUMMARY_WITH_SCOPES_PROMPT.format(
                filename=filename,
                content=content
            )
            logger.debug(f"[FILE_SUMMARY] Sending prompt to LLM ({len(prompt)} chars)")
            response = llm.invoke([HumanMessage(content=prompt)])
            logger.debug(f"[FILE_SUMMARY] LLM raw response: '{response.content[:500]}...'")
            result = _parse_json_response(response.content)

        else:
            # Map-reduce for larger files
            logger.info(f"[FILE_SUMMARY] Using MAP-REDUCE summarization for {filename} ({len(content)} chars)")

            # Step 1: Split content into sections
            sections = _split_by_headers(content)
            logger.info(f"[FILE_SUMMARY] Split into {len(sections)} sections")
            for i, section in enumerate(sections):
                logger.debug(f"[FILE_SUMMARY] Section {i+1} size: {len(section)} chars, preview: '{section[:200]}...'")

            # Step 2: Summarize each section
            section_summaries = []
            for i, section in enumerate(sections):
                logger.info(f"[FILE_SUMMARY] Processing section {i + 1}/{len(sections)} ({len(section)} chars)")
                summary = _summarize_section(section)
                section_summaries.append(f"[Section {i + 1}]: {summary}")
                logger.info(f"[FILE_SUMMARY] Section {i + 1} summary: '{summary[:150]}...'")

            # Step 3: Combine section summaries
            combined_summaries = "\n\n".join(section_summaries)
            logger.debug(f"[FILE_SUMMARY] Combined summaries ({len(combined_summaries)} chars): '{combined_summaries[:500]}...'")
            prompt = COMBINE_SUMMARIES_PROMPT.format(
                filename=filename,
                section_summaries=combined_summaries
            )
            logger.debug(f"[FILE_SUMMARY] Sending combine prompt to LLM ({len(prompt)} chars)")
            response = llm.invoke([HumanMessage(content=prompt)])
            logger.debug(f"[FILE_SUMMARY] LLM raw response: '{response.content[:500]}...'")
            result = _parse_json_response(response.content)

        # Extract and validate fields
        summary = result.get("summary", "")
        scopes = result.get("scopes", [])
        content_type = result.get("content_type", "document")
        complexity = result.get("complexity", "intermediate")

        # Validate scopes is a list
        if not isinstance(scopes, list):
            scopes = []

        # Ensure scopes are strings
        scopes = [str(s).strip().lower() for s in scopes if s]

        # Validate content_type
        valid_content_types = [
            "technical_documentation", "tutorial", "reference", "guide",
            "specification", "report", "api_doc", "architecture", "policy", "manual"
        ]
        if content_type not in valid_content_types:
            content_type = "document"

        # Validate complexity
        valid_complexities = ["basic", "intermediate", "advanced"]
        if complexity not in valid_complexities:
            complexity = "intermediate"

        logger.info(
            f"[FILE_SUMMARY] Completed for '{filename}': "
            f"{len(summary)} chars, {len(scopes)} scopes, type={content_type}, complexity={complexity}"
        )
        logger.debug(f"[FILE_SUMMARY] Final summary: '{summary[:500]}...'")
        logger.debug(f"[FILE_SUMMARY] Scopes: {scopes}")

        return FileSummaryWithScopes(
            summary=summary,
            scopes=scopes,
            content_type=content_type,
            complexity=complexity,
        )

    except Exception as e:
        logger.error(f"[FILE_SUMMARY] Generation failed for '{filename}': {e}")
        # Fallback: basic summary without scopes
        return FileSummaryWithScopes(
            summary=f"Document: {filename}\nError generating summary: {str(e)}",
            scopes=[],
            content_type="document",
            complexity="intermediate",
        )
