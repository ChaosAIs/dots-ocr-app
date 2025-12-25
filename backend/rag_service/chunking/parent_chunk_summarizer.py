"""
Parent Chunk Summarizer Service.

This module provides LLM-based summarization for parent chunks to ensure
they don't exceed token limits while preserving semantic meaning.

The summarizer is used when:
1. A parent chunk exceeds the configured token threshold
2. Context needs to be compressed for LLM consumption
3. Large tables or sections need semantic summaries

Supports both sync and async operations, with caching for performance.
"""

import os
import logging
import hashlib
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)

# Configuration from environment
PARENT_CHUNK_SUMMARY_ENABLED = os.getenv("PARENT_CHUNK_SUMMARY_ENABLED", "true").lower() == "true"
PARENT_CHUNK_TOKEN_THRESHOLD = int(os.getenv("PARENT_CHUNK_TOKEN_THRESHOLD", "1500"))
PARENT_CHUNK_SUMMARY_MAX_TOKENS = int(os.getenv("PARENT_CHUNK_SUMMARY_MAX_TOKENS", "500"))
PARENT_CHUNK_CACHE_SIZE = int(os.getenv("PARENT_CHUNK_CACHE_SIZE", "1000"))


@dataclass
class ParentChunkSummary:
    """Result of parent chunk summarization."""
    original_content: str
    summary: str
    original_token_count: int
    summary_token_count: int
    compression_ratio: float
    content_type: str  # "text", "table", "code", "mixed"
    key_points: List[str] = field(default_factory=list)
    preserved_entities: List[str] = field(default_factory=list)
    was_summarized: bool = True
    error: Optional[str] = None


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    Uses a conservative estimate of 1 token per 4 characters.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return len(text) // 4


def _get_content_hash(content: str) -> str:
    """Generate a hash for content to use as cache key."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]


def _detect_content_type(content: str) -> str:
    """
    Detect the type of content for appropriate summarization strategy.

    Args:
        content: The content to analyze

    Returns:
        Content type: "table", "code", "text", or "mixed"
    """
    import re

    has_table = bool(re.search(r'<table>|^\|.*\|.*\|', content, re.MULTILINE))
    has_code = bool(re.search(r'```|<code>|<pre>', content))

    # Count structural elements
    table_lines = len(re.findall(r'<tr>|^\|.*\|', content, re.MULTILINE))
    code_blocks = len(re.findall(r'```', content)) // 2
    total_lines = content.count('\n') + 1

    if has_table and table_lines > total_lines * 0.5:
        return "table"
    elif has_code and code_blocks > 0:
        return "code"
    elif has_table or has_code:
        return "mixed"
    else:
        return "text"


def _build_summary_prompt(content: str, content_type: str, max_summary_tokens: int) -> str:
    """
    Build a prompt for summarizing content based on its type.

    Args:
        content: The content to summarize
        content_type: The detected content type
        max_summary_tokens: Maximum tokens for the summary

    Returns:
        The prompt string
    """
    base_instructions = f"""Summarize the following content concisely in approximately {max_summary_tokens} tokens or less.
Preserve all key information, entities, dates, and important details.
Output ONLY the summary, no explanations or meta-commentary."""

    if content_type == "table":
        return f"""{base_instructions}

For this TABLE content:
- Describe the table structure (columns, row count)
- Summarize the data patterns and key values
- List any notable entries or outliers
- Preserve column headers and key row identifiers

CONTENT:
{content}

SUMMARY:"""

    elif content_type == "code":
        return f"""{base_instructions}

For this CODE content:
- Describe the purpose and functionality
- List key functions/classes/methods
- Note important parameters or configurations
- Preserve critical logic explanations

CONTENT:
{content}

SUMMARY:"""

    elif content_type == "mixed":
        return f"""{base_instructions}

For this MIXED content (tables/code/text):
- Summarize each major component
- Preserve structural relationships
- Note transitions between different content types

CONTENT:
{content}

SUMMARY:"""

    else:  # text
        return f"""{base_instructions}

For this TEXT content:
- Capture main ideas and conclusions
- Preserve named entities (people, places, organizations)
- Keep dates, amounts, and specific details
- Maintain logical flow

CONTENT:
{content}

SUMMARY:"""


def _extract_key_points(content: str, content_type: str) -> List[str]:
    """
    Extract key points from content without LLM (fast heuristic).

    Args:
        content: The content to analyze
        content_type: The detected content type

    Returns:
        List of key points
    """
    import re

    key_points = []

    # Extract headers as key points
    headers = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
    key_points.extend([f"Section: {h}" for h in headers[:5]])

    # Extract dates
    dates = re.findall(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b', content)
    if dates:
        key_points.append(f"Dates mentioned: {', '.join(dates[:3])}")

    # Extract amounts/numbers
    amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})+(?:\.\d{2})?', content)
    if amounts:
        key_points.append(f"Amounts: {', '.join(amounts[:3])}")

    if content_type == "table":
        # Count rows
        rows = len(re.findall(r'<tr>|^\|.*\|', content, re.MULTILINE))
        key_points.append(f"Table with approximately {rows} rows")

    return key_points


def _extract_entities(content: str) -> List[str]:
    """
    Extract named entities from content (simple heuristic).

    Args:
        content: The content to analyze

    Returns:
        List of entity names
    """
    import re

    entities = []

    # Extract capitalized phrases (potential names/organizations)
    caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', content)
    entities.extend(caps[:10])

    # Extract quoted terms
    quoted = re.findall(r'"([^"]+)"', content)
    entities.extend(quoted[:5])

    # Deduplicate while preserving order
    seen = set()
    unique_entities = []
    for e in entities:
        if e.lower() not in seen:
            seen.add(e.lower())
            unique_entities.append(e)

    return unique_entities[:10]


class ParentChunkSummarizer:
    """
    Service for summarizing parent chunks using LLM.

    Uses the configured LLM backend (Ollama or vLLM) to generate
    concise summaries of large parent chunks while preserving
    key information for retrieval.
    """

    def __init__(
        self,
        token_threshold: int = None,
        max_summary_tokens: int = None,
        enabled: bool = None,
    ):
        """
        Initialize the summarizer.

        Args:
            token_threshold: Token count above which summarization is triggered
            max_summary_tokens: Maximum tokens for the generated summary
            enabled: Whether summarization is enabled
        """
        self.token_threshold = token_threshold or PARENT_CHUNK_TOKEN_THRESHOLD
        self.max_summary_tokens = max_summary_tokens or PARENT_CHUNK_SUMMARY_MAX_TOKENS
        self.enabled = enabled if enabled is not None else PARENT_CHUNK_SUMMARY_ENABLED
        self._llm = None
        self._cache: Dict[str, ParentChunkSummary] = {}

        logger.info(
            f"ParentChunkSummarizer initialized: enabled={self.enabled}, "
            f"threshold={self.token_threshold}, max_summary={self.max_summary_tokens}"
        )

    def _get_llm(self):
        """Get or create the LLM instance."""
        if self._llm is None:
            try:
                from ..llm_service import get_llm_service
                llm_service = get_llm_service()
                # Use a smaller context for summarization
                self._llm = llm_service.get_query_model(
                    temperature=0.1,
                    num_ctx=4096,
                    num_predict=self.max_summary_tokens + 100,
                )
                logger.info("ParentChunkSummarizer LLM initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM for summarization: {e}")
                self._llm = None
        return self._llm

    def needs_summarization(self, content: str) -> bool:
        """
        Check if content needs summarization based on token count.

        Args:
            content: The content to check

        Returns:
            True if content exceeds token threshold
        """
        if not self.enabled:
            return False
        return estimate_tokens(content) > self.token_threshold

    def summarize(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> ParentChunkSummary:
        """
        Summarize content if it exceeds the token threshold.

        Args:
            content: The content to summarize
            metadata: Optional metadata about the content (for context)
            force: Force summarization even if below threshold

        Returns:
            ParentChunkSummary with original and summarized content
        """
        original_tokens = estimate_tokens(content)

        # Check cache first
        content_hash = _get_content_hash(content)
        if content_hash in self._cache:
            logger.debug(f"Returning cached summary for {content_hash}")
            return self._cache[content_hash]

        # Check if summarization is needed
        if not force and not self.needs_summarization(content):
            result = ParentChunkSummary(
                original_content=content,
                summary=content,  # Use original as summary
                original_token_count=original_tokens,
                summary_token_count=original_tokens,
                compression_ratio=1.0,
                content_type=_detect_content_type(content),
                key_points=_extract_key_points(content, _detect_content_type(content)),
                preserved_entities=_extract_entities(content),
                was_summarized=False,
            )
            return result

        # Detect content type
        content_type = _detect_content_type(content)

        # Try LLM summarization
        summary = None
        error = None

        try:
            llm = self._get_llm()
            if llm:
                prompt = _build_summary_prompt(content, content_type, self.max_summary_tokens)
                response = llm.invoke(prompt)
                summary = response.content.strip()

                # Clean up any meta-commentary
                if summary.startswith("SUMMARY:"):
                    summary = summary[8:].strip()
                if summary.startswith("Here"):
                    lines = summary.split('\n', 1)
                    if len(lines) > 1:
                        summary = lines[1].strip()

                logger.debug(f"LLM summarization successful: {original_tokens} -> {estimate_tokens(summary)} tokens")
        except Exception as e:
            logger.warning(f"LLM summarization failed, using fallback: {e}")
            error = str(e)

        # Fallback: Create a structured summary without LLM
        if not summary:
            summary = self._create_fallback_summary(content, content_type, metadata)

        summary_tokens = estimate_tokens(summary)
        compression_ratio = summary_tokens / original_tokens if original_tokens > 0 else 1.0

        result = ParentChunkSummary(
            original_content=content,
            summary=summary,
            original_token_count=original_tokens,
            summary_token_count=summary_tokens,
            compression_ratio=compression_ratio,
            content_type=content_type,
            key_points=_extract_key_points(content, content_type),
            preserved_entities=_extract_entities(content),
            was_summarized=True,
            error=error,
        )

        # Cache the result
        if len(self._cache) < PARENT_CHUNK_CACHE_SIZE:
            self._cache[content_hash] = result

        logger.info(
            f"Summarized parent chunk: {original_tokens} -> {summary_tokens} tokens "
            f"(ratio: {compression_ratio:.2f}, type: {content_type})"
        )

        return result

    def _create_fallback_summary(
        self,
        content: str,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a fallback summary without LLM.

        This is used when LLM is unavailable or fails.
        Uses heuristics to extract the most important parts.

        Args:
            content: The content to summarize
            content_type: The detected content type
            metadata: Optional metadata

        Returns:
            A heuristically-generated summary
        """
        import re

        max_chars = self.max_summary_tokens * 4  # Rough token-to-char conversion

        # Build summary parts
        parts = []

        # Add source info from metadata
        if metadata:
            source = metadata.get("source", "")
            heading = metadata.get("heading_path", "")
            if source:
                parts.append(f"[Source: {source}]")
            if heading:
                parts.append(f"[Section: {heading}]")

        # Add content type indicator
        parts.append(f"[Content Type: {content_type}]")

        # Extract key points
        key_points = _extract_key_points(content, content_type)
        if key_points:
            parts.append("Key Points: " + "; ".join(key_points))

        # Extract entities
        entities = _extract_entities(content)
        if entities:
            parts.append("Entities: " + ", ".join(entities[:5]))

        # Add truncated content preview
        remaining_chars = max_chars - sum(len(p) for p in parts) - 50
        if remaining_chars > 100:
            # Get first and last parts of content
            preview_size = remaining_chars // 2

            # Clean content for preview
            clean_content = re.sub(r'\s+', ' ', content).strip()

            if len(clean_content) > remaining_chars:
                start = clean_content[:preview_size].rsplit(' ', 1)[0]
                end = clean_content[-preview_size:].split(' ', 1)[-1] if len(clean_content) > preview_size * 2 else ""

                preview = f"{start}... [content truncated] ...{end}" if end else f"{start}..."
            else:
                preview = clean_content

            parts.append(f"Content Preview: {preview}")

        return "\n".join(parts)

    def summarize_for_context(
        self,
        content: str,
        available_tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, int]:
        """
        Summarize content to fit within available token budget.

        This method is used during context building to ensure
        content fits within LLM context window.

        Args:
            content: The content to summarize
            available_tokens: Maximum tokens available for this content
            metadata: Optional metadata

        Returns:
            Tuple of (summarized content, actual token count)
        """
        original_tokens = estimate_tokens(content)

        # If already fits, return as-is
        if original_tokens <= available_tokens:
            return content, original_tokens

        # Try LLM summarization with specific target
        temp_max = self.max_summary_tokens
        self.max_summary_tokens = min(available_tokens, temp_max)

        try:
            result = self.summarize(content, metadata, force=True)

            # If still too large, truncate
            if result.summary_token_count > available_tokens:
                truncated = result.summary[:available_tokens * 4]
                if '...' not in truncated[-10:]:
                    truncated = truncated.rsplit(' ', 1)[0] + "..."
                return truncated, estimate_tokens(truncated)

            return result.summary, result.summary_token_count
        finally:
            self.max_summary_tokens = temp_max

    def clear_cache(self):
        """Clear the summary cache."""
        self._cache.clear()
        logger.info("ParentChunkSummarizer cache cleared")


# Singleton instance
_summarizer: Optional[ParentChunkSummarizer] = None


def get_parent_chunk_summarizer() -> ParentChunkSummarizer:
    """Get or create the singleton summarizer instance."""
    global _summarizer
    if _summarizer is None:
        _summarizer = ParentChunkSummarizer()
    return _summarizer


def summarize_parent_chunk(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> ParentChunkSummary:
    """
    Convenience function to summarize a parent chunk.

    Args:
        content: The content to summarize
        metadata: Optional metadata
        force: Force summarization even if below threshold

    Returns:
        ParentChunkSummary
    """
    summarizer = get_parent_chunk_summarizer()
    return summarizer.summarize(content, metadata, force)


def needs_parent_chunk_summary(content: str) -> bool:
    """
    Check if content needs summarization.

    Args:
        content: The content to check

    Returns:
        True if summarization is needed
    """
    summarizer = get_parent_chunk_summarizer()
    return summarizer.needs_summarization(content)
