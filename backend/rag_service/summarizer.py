"""
LLM-based summarization for RAG chunks and documents.
Uses Ollama with a fast model for generating summaries.
"""

import os
import re
import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# RAG Ollama configuration (Docker: ollama-llm, Port: 11434)
# Used for: Summarization (same server as RAG agent)
RAG_OLLAMA_HOST = os.getenv("RAG_OLLAMA_HOST", os.getenv("OLLAMA_HOST", "localhost"))
RAG_OLLAMA_PORT = os.getenv("RAG_OLLAMA_PORT", os.getenv("OLLAMA_PORT", "11434"))
# Use RAG_OLLAMA_QUERY_MODEL for summarization (same model used for query analysis)
RAG_OLLAMA_SUMMARY_MODEL = os.getenv("RAG_OLLAMA_QUERY_MODEL", os.getenv("OLLAMA_QUERY_MODEL",
                                     os.getenv("RAG_OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:latest"))))
RAG_OLLAMA_BASE_URL = f"http://{RAG_OLLAMA_HOST}:{RAG_OLLAMA_PORT}"

# Legacy aliases for backward compatibility
OLLAMA_HOST = RAG_OLLAMA_HOST
OLLAMA_PORT = RAG_OLLAMA_PORT
OLLAMA_SUMMARY_MODEL = RAG_OLLAMA_SUMMARY_MODEL
OLLAMA_BASE_URL = RAG_OLLAMA_BASE_URL


@dataclass
class ChunkSummary:
    """Summary of a single chunk."""
    chunk_index: int
    summary: str
    heading_path: str = ""


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
MAX_SECTIONS_TO_PROCESS = 20  # Maximum sections to process in map-reduce


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
SECTION_SUMMARY_PROMPT = """Summarize this section of a document in 100-150 words.
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


CHUNK_SUMMARY_PROMPT = """Summarize the following text in no more than {max_words} words.
Focus on the key information, main topics, and important concepts.
Output ONLY the summary, nothing else.

Text:
{content}"""

FILE_SUMMARY_PROMPT = """Based on the following chunk summaries from a document, generate a comprehensive file summary.

Document filename: {filename}

Chunk summaries:
{chunk_summaries}

Generate a summary that includes:
1. Main topic/title of the document
2. Category of the document (e.g., Technical Documentation, Research Paper, Legal Document, Financial Report, User Manual, Policy Document, Meeting Notes, Tutorial, Reference Guide, etc.)
3. Purpose of the document
4. Key themes and sections covered
5. Brief description (2-3 sentences)

Format your response as:
Title: [document title or main topic]
Category: [document category]
Purpose: [purpose of the document]
Topics: [comma-separated list of main topics]
Description: [2-3 sentence description]

Output ONLY the formatted summary, nothing else."""


def _create_summary_llm():
    """Create the Ollama LLM instance for summarization."""
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_SUMMARY_MODEL,
        temperature=0.1,  # Low temperature for consistent summaries
        num_ctx=4096,
        num_predict=256,  # Limit output tokens
    )


def summarize_chunk(content: str, max_words: int = 200) -> str:
    """
    Summarize a single chunk of text.

    Args:
        content: The text content to summarize.
        max_words: Maximum words for the summary.

    Returns:
        Summary string.
    """
    if not content or len(content.strip()) < 50:
        # Skip very short content
        return content.strip()[:200] if content else ""

    # Adjust max_words based on content length
    content_words = len(content.split())
    if content_words < max_words:
        # Content is already short, just return it trimmed
        return content.strip()[:500]

    try:
        llm = _create_summary_llm()
        prompt = CHUNK_SUMMARY_PROMPT.format(
            max_words=max_words,
            content=content[:4000]  # Limit input size
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()

        # Clean up any thinking tags if present
        if "</think>" in summary:
            summary = summary.split("</think>")[-1].strip()

        return summary[:500]  # Safety limit

    except Exception as e:
        logger.warning(f"Chunk summarization failed: {e}")
        # Fallback: return first 200 chars
        return content.strip()[:200]


def generate_file_summary(
    filename: str,
    chunk_summaries: List[ChunkSummary]
) -> str:
    """
    Generate a comprehensive file summary from chunk summaries.

    Args:
        filename: The document filename.
        chunk_summaries: List of ChunkSummary objects.

    Returns:
        Formatted file summary string.
    """
    if not chunk_summaries:
        return f"Document: {filename}\nNo content available for summarization."

    try:
        llm = _create_summary_llm()

        # Format chunk summaries for the prompt
        formatted_summaries = []
        for cs in chunk_summaries:
            if cs.heading_path:
                formatted_summaries.append(f"[{cs.heading_path}]: {cs.summary}")
            else:
                formatted_summaries.append(f"[Chunk {cs.chunk_index}]: {cs.summary}")

        prompt = FILE_SUMMARY_PROMPT.format(
            filename=filename,
            chunk_summaries="\n".join(formatted_summaries)
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()

        # Clean up any thinking tags if present
        if "</think>" in summary:
            summary = summary.split("</think>")[-1].strip()

        return summary

    except Exception as e:
        logger.error(f"File summary generation failed for {filename}: {e}")
        # Fallback: combine first few chunk summaries
        fallback_parts = [f"Document: {filename}"]
        for cs in chunk_summaries[:5]:
            fallback_parts.append(cs.summary)
        return "\n".join(fallback_parts)




def _create_summary_llm_extended():
    """Create the Ollama LLM instance for extended summarization (larger output)."""
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_SUMMARY_MODEL,
        temperature=0.1,
        num_ctx=8192,  # Larger context for file summaries
        num_predict=1024,  # Allow longer output for 500 word summaries
    )


def _split_by_headers(content: str) -> List[str]:
    """
    Split content by markdown headers for map-reduce summarization.

    Args:
        content: The markdown content to split.

    Returns:
        List of content sections.
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
        sections = _split_by_size(content, MAX_SECTION_SIZE)

    return sections[:MAX_SECTIONS_TO_PROCESS]


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


def _summarize_section(content: str) -> str:
    """
    Summarize a single section for map-reduce.

    Args:
        content: The section content.

    Returns:
        Section summary string.
    """
    if len(content) < 200:
        return content

    try:
        llm = _create_summary_llm()
        prompt = SECTION_SUMMARY_PROMPT.format(content=content[:MAX_SECTION_SIZE])
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()

        # Clean up thinking tags
        if "</think>" in summary:
            summary = summary.split("</think>")[-1].strip()

        return summary[:500]  # Safety limit

    except Exception as e:
        logger.warning(f"Section summarization failed: {e}")
        return content[:300]


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
    if not content or len(content.strip()) < 50:
        return FileSummaryWithScopes(
            summary=f"Document: {filename}\nNo content available for summarization.",
            scopes=[],
            content_type="document",
            complexity="basic",
        )

    try:
        llm = _create_summary_llm_extended()

        # Decide between direct summarization and map-reduce
        if len(content) <= MAX_DIRECT_SUMMARY_SIZE:
            # Direct summarization for smaller files
            logger.info(f"Using direct summarization for {filename} ({len(content)} chars)")
            prompt = FILE_SUMMARY_WITH_SCOPES_PROMPT.format(
                filename=filename,
                content=content
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            result = _parse_json_response(response.content)

        else:
            # Map-reduce for larger files
            logger.info(f"Using map-reduce summarization for {filename} ({len(content)} chars)")

            # Step 1: Split content into sections
            sections = _split_by_headers(content)
            logger.info(f"Split into {len(sections)} sections")

            # Step 2: Summarize each section
            section_summaries = []
            for i, section in enumerate(sections):
                summary = _summarize_section(section)
                section_summaries.append(f"[Section {i + 1}]: {summary}")
                logger.debug(f"Summarized section {i + 1}/{len(sections)}")

            # Step 3: Combine section summaries
            combined_summaries = "\n\n".join(section_summaries)
            prompt = COMBINE_SUMMARIES_PROMPT.format(
                filename=filename,
                section_summaries=combined_summaries
            )
            response = llm.invoke([HumanMessage(content=prompt)])
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
            f"Generated summary with scopes for {filename}: "
            f"{len(summary)} chars, {len(scopes)} scopes, type={content_type}, complexity={complexity}"
        )

        return FileSummaryWithScopes(
            summary=summary,
            scopes=scopes,
            content_type=content_type,
            complexity=complexity,
        )

    except Exception as e:
        logger.error(f"File summary with scopes generation failed for {filename}: {e}")
        # Fallback: basic summary without scopes
        return FileSummaryWithScopes(
            summary=f"Document: {filename}\nError generating summary: {str(e)}",
            scopes=[],
            content_type="document",
            complexity="intermediate",
        )
