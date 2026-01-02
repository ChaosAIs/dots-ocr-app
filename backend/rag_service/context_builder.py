"""
Token-Aware Context Builder for RAG.

This module provides intelligent context building that:
1. Respects LLM token limits
2. Prioritizes chunks by relevance and importance
3. Uses summarization for large parent chunks
4. Balances regular chunks, parent context, and graph context

The context builder ensures that the final context passed to the LLM
fits within the model's context window while maximizing information density.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Configuration from environment
CONTEXT_MAX_TOKENS = int(os.getenv("CONTEXT_MAX_TOKENS", "6000"))
CONTEXT_CHUNK_BUDGET_PERCENT = int(os.getenv("CONTEXT_CHUNK_BUDGET_PERCENT", "60"))
CONTEXT_PARENT_BUDGET_PERCENT = int(os.getenv("CONTEXT_PARENT_BUDGET_PERCENT", "25"))
CONTEXT_GRAPH_BUDGET_PERCENT = int(os.getenv("CONTEXT_GRAPH_BUDGET_PERCENT", "15"))
CONTEXT_SUMMARIZE_LARGE_CHUNKS = os.getenv("CONTEXT_SUMMARIZE_LARGE_CHUNKS", "true").lower() == "true"

# Query relevance scoring configuration
CONTEXT_ENABLE_QUERY_SCORING = os.getenv("CONTEXT_ENABLE_QUERY_SCORING", "true").lower() == "true"
CONTEXT_SCORE_THRESHOLD_CRITICAL = float(os.getenv("CONTEXT_SCORE_THRESHOLD_CRITICAL", "0.85"))
CONTEXT_SCORE_THRESHOLD_HIGH = float(os.getenv("CONTEXT_SCORE_THRESHOLD_HIGH", "0.70"))
CONTEXT_SCORE_THRESHOLD_MEDIUM = float(os.getenv("CONTEXT_SCORE_THRESHOLD_MEDIUM", "0.50"))


class ChunkPriority(Enum):
    """Priority levels for chunks in context building."""
    CRITICAL = 1  # Must include (direct matches, key sections)
    HIGH = 2      # Should include (parent context, high relevance)
    MEDIUM = 3    # Nice to have (supporting context)
    LOW = 4       # Include if space allows


@dataclass
class ContextChunk:
    """A chunk prepared for context inclusion."""
    content: str
    source: str
    heading: str
    chunk_id: str
    token_count: int
    priority: ChunkPriority
    is_parent: bool = False
    is_summarized: bool = False
    original_token_count: int = 0
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuiltContext:
    """Result of context building."""
    formatted_context: str
    total_tokens: int
    chunks_included: int
    chunks_excluded: int
    parent_chunks_included: int
    parent_chunks_summarized: int
    graph_context_tokens: int
    budget_usage: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)  # Unique source document names


def estimate_tokens(text: str) -> int:
    """Estimate token count (1 token ~ 4 chars)."""
    return len(text) // 4 if text else 0


class TokenAwareContextBuilder:
    """
    Builds context for LLM consumption with intelligent token management.

    This builder:
    1. Allocates token budget across different context types
    2. Prioritizes chunks by relevance and importance
    3. Summarizes large parent chunks when necessary
    4. Ensures total context fits within limits
    """

    def __init__(
        self,
        max_tokens: int = None,
        chunk_budget_percent: int = None,
        parent_budget_percent: int = None,
        graph_budget_percent: int = None,
        summarize_large_chunks: bool = None,
    ):
        """
        Initialize the context builder.

        Args:
            max_tokens: Maximum total tokens for context
            chunk_budget_percent: Percentage of budget for regular chunks
            parent_budget_percent: Percentage of budget for parent chunks
            graph_budget_percent: Percentage of budget for graph context
            summarize_large_chunks: Whether to summarize chunks exceeding budget
        """
        self.max_tokens = max_tokens or CONTEXT_MAX_TOKENS
        self.chunk_budget_percent = chunk_budget_percent or CONTEXT_CHUNK_BUDGET_PERCENT
        self.parent_budget_percent = parent_budget_percent or CONTEXT_PARENT_BUDGET_PERCENT
        self.graph_budget_percent = graph_budget_percent or CONTEXT_GRAPH_BUDGET_PERCENT
        self.summarize_large_chunks = (
            summarize_large_chunks if summarize_large_chunks is not None
            else CONTEXT_SUMMARIZE_LARGE_CHUNKS
        )

        # Calculate budgets
        self.chunk_budget = int(self.max_tokens * self.chunk_budget_percent / 100)
        self.parent_budget = int(self.max_tokens * self.parent_budget_percent / 100)
        self.graph_budget = int(self.max_tokens * self.graph_budget_percent / 100)

        logger.info(
            f"TokenAwareContextBuilder initialized: max={self.max_tokens}, "
            f"chunks={self.chunk_budget}, parents={self.parent_budget}, "
            f"graph={self.graph_budget}"
        )

    def _get_summarizer(self):
        """Get the parent chunk summarizer (lazy load)."""
        try:
            from .chunking.parent_chunk_summarizer import get_parent_chunk_summarizer
            return get_parent_chunk_summarizer()
        except ImportError:
            logger.warning("Parent chunk summarizer not available")
            return None

    def _calculate_chunk_priority(
        self,
        chunk: Dict[str, Any],
        is_parent: bool = False,
    ) -> ChunkPriority:
        """
        Calculate priority for a chunk based on its metadata and query relevance score.

        Priority is determined by:
        1. Query relevance score (_score) if available - dynamic, query-specific
        2. Key section markers (is_key_section, is_abstract) - always critical
        3. Static importance_score as fallback

        Args:
            chunk: The chunk dictionary
            is_parent: Whether this is a parent chunk

        Returns:
            ChunkPriority level
        """
        metadata = chunk.get("metadata", {})

        # Check for key section markers - these are always critical
        is_key_section = metadata.get("is_key_section", False)
        is_abstract = metadata.get("is_abstract", False)
        if is_key_section or is_abstract:
            return ChunkPriority.CRITICAL

        # Use query relevance score (_score) if available
        # This is the dynamic score computed against the current query
        query_score = chunk.get("_score") or metadata.get("_score")

        if query_score is not None:
            # Use query-specific relevance scoring
            if query_score >= CONTEXT_SCORE_THRESHOLD_CRITICAL:
                return ChunkPriority.CRITICAL
            elif query_score >= CONTEXT_SCORE_THRESHOLD_HIGH:
                return ChunkPriority.HIGH
            elif query_score >= CONTEXT_SCORE_THRESHOLD_MEDIUM:
                return ChunkPriority.MEDIUM
            else:
                return ChunkPriority.LOW

        # Fallback to static importance_score if no query score
        importance_score = metadata.get("importance_score", 0.5)

        # High importance score
        if importance_score >= 0.8:
            return ChunkPriority.HIGH if not is_parent else ChunkPriority.MEDIUM

        # Parent chunks are generally medium priority
        if is_parent:
            return ChunkPriority.MEDIUM

        # Regular chunks based on importance
        if importance_score >= 0.6:
            return ChunkPriority.HIGH
        elif importance_score >= 0.4:
            return ChunkPriority.MEDIUM
        else:
            return ChunkPriority.LOW

    def _prepare_chunk(
        self,
        chunk: Dict[str, Any],
        is_parent: bool = False,
        available_tokens: int = None,
    ) -> ContextChunk:
        """
        Prepare a chunk for context inclusion.

        May summarize if the chunk is too large. For parent chunks that already
        have a pre-computed summary from the indexing process, we use that
        summary instead of re-summarizing.

        Args:
            chunk: The chunk dictionary
            is_parent: Whether this is a parent chunk
            available_tokens: Max tokens available for this chunk

        Returns:
            ContextChunk ready for inclusion
        """
        content = chunk.get("page_content", "")
        metadata = chunk.get("metadata", {})

        source = metadata.get("source", "Unknown")
        heading = metadata.get("heading_path", "")
        chunk_id = chunk.get("chunk_id", metadata.get("chunk_id", ""))

        original_tokens = estimate_tokens(content)
        is_summarized = False
        final_content = content

        # Check if this parent chunk already has a pre-computed summary from indexing
        # This avoids duplicate summarization for large parent chunks
        existing_summary = metadata.get("parent_summary")
        existing_summary_tokens = metadata.get("parent_summary_tokens", 0)

        if existing_summary and is_parent:
            # Use the pre-computed summary if it fits within the available budget
            if available_tokens is None or existing_summary_tokens <= available_tokens:
                final_content = existing_summary
                is_summarized = True
                logger.debug(
                    f"Using pre-computed summary for parent chunk {chunk_id}: "
                    f"{original_tokens} -> {existing_summary_tokens} tokens"
                )
            elif available_tokens and existing_summary_tokens > available_tokens:
                # Pre-computed summary is still too large, need to further compress
                logger.debug(
                    f"Pre-computed summary ({existing_summary_tokens} tokens) exceeds budget "
                    f"({available_tokens}), will further compress"
                )
                # Fall through to summarization logic below with the summary as base
                content = existing_summary
                original_tokens = existing_summary_tokens

        # Check if summarization is needed (only if not already using pre-computed summary)
        # Skip summarization for:
        # 1. Spreadsheet/Excel documents (structured data shouldn't be summarized)
        # 2. Chunks that are close to budget (only summarize if exceeds by >20%)
        # 3. Small chunks below the global threshold (1500 tokens default)
        # Use document_types list (standard format)
        doc_types = [t.lower() for t in metadata.get("document_types", []) if t]

        spreadsheet_types = ["spreadsheet", "excel", "csv", "xlsx", "xls"]
        is_spreadsheet = any(t in spreadsheet_types for t in doc_types) or \
                         source.lower().endswith(('.xlsx', '.xls', '.csv'))

        # Only summarize if chunk significantly exceeds available budget AND is above global threshold
        # Also skip for spreadsheet documents which have structured data
        should_summarize = (
            not is_summarized and
            available_tokens and
            original_tokens > available_tokens * 1.2 and  # Must exceed by >20%
            original_tokens > 500 and  # Minimum size threshold
            self.summarize_large_chunks and
            not is_spreadsheet  # Skip spreadsheets
        )

        if should_summarize:
            summarizer = self._get_summarizer()
            if summarizer:
                try:
                    final_content, token_count = summarizer.summarize_for_context(
                        content, available_tokens, metadata
                    )
                    is_summarized = True
                    logger.debug(
                        f"Summarized chunk {chunk_id}: {original_tokens} -> {token_count} tokens"
                    )
                except Exception as e:
                    logger.warning(f"Failed to summarize chunk: {e}")
                    # Fallback: truncate
                    max_chars = available_tokens * 4
                    final_content = content[:max_chars]
                    if not final_content.endswith("..."):
                        final_content = final_content.rsplit(' ', 1)[0] + "..."
                    is_summarized = True
        elif not is_summarized and is_spreadsheet:
            logger.debug(f"Skipping summarization for spreadsheet chunk {chunk_id}")

        # Use query relevance score if available, otherwise fall back to static importance_score
        query_score = chunk.get("_score") or metadata.get("_score")
        relevance_score = query_score if query_score is not None else metadata.get("importance_score", 0.5)

        return ContextChunk(
            content=final_content,
            source=source,
            heading=heading,
            chunk_id=chunk_id,
            token_count=estimate_tokens(final_content),
            priority=self._calculate_chunk_priority(chunk, is_parent),
            is_parent=is_parent,
            is_summarized=is_summarized,
            original_token_count=original_tokens,
            relevance_score=relevance_score,
            metadata=metadata,
        )

    def _select_chunks_within_budget(
        self,
        chunks: List[ContextChunk],
        budget: int,
    ) -> Tuple[List[ContextChunk], int]:
        """
        Select chunks that fit within the token budget.

        Prioritizes by ChunkPriority, then by relevance score.

        Args:
            chunks: List of prepared chunks
            budget: Token budget

        Returns:
            Tuple of (selected chunks, tokens used)
        """
        # Sort by priority (lower value = higher priority), then by relevance
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (c.priority.value, -c.relevance_score)
        )

        selected = []
        used_tokens = 0

        for chunk in sorted_chunks:
            if used_tokens + chunk.token_count <= budget:
                selected.append(chunk)
                used_tokens += chunk.token_count
            elif chunk.priority == ChunkPriority.CRITICAL:
                # Try to fit critical chunks with summarization
                remaining = budget - used_tokens
                if remaining > 100 and self.summarize_large_chunks:
                    summarizer = self._get_summarizer()
                    if summarizer:
                        try:
                            summary, tokens = summarizer.summarize_for_context(
                                chunk.content, remaining, chunk.metadata
                            )
                            chunk.content = summary
                            chunk.token_count = tokens
                            chunk.is_summarized = True
                            selected.append(chunk)
                            used_tokens += tokens
                        except Exception:
                            pass  # Skip if summarization fails

        return selected, used_tokens

    def _format_chunk_for_context(
        self,
        chunk: ContextChunk,
        doc_num: int,
    ) -> str:
        """
        Format a chunk for inclusion in the context string.

        Args:
            chunk: The prepared chunk
            doc_num: Document number for labeling

        Returns:
            Formatted string
        """
        label = "Context" if chunk.is_parent else "Document"
        summary_marker = " (Summary)" if chunk.is_summarized else ""

        result = f"[{label} {doc_num}{summary_marker}: {chunk.source}]"
        if chunk.heading:
            result += f"\nSection: {chunk.heading}"
        result += f"\n{chunk.content}"

        return result

    def build_context(
        self,
        chunks: List[Dict[str, Any]],
        parent_chunks: List[Dict[str, Any]] = None,
        graph_context: str = None,
        query: str = None,
    ) -> BuiltContext:
        """
        Build a token-aware context from chunks and additional context.

        When a query is provided and CONTEXT_ENABLE_QUERY_SCORING is enabled,
        chunks are scored against the query for relevance-based prioritization.

        Args:
            chunks: Regular search result chunks
            parent_chunks: Parent chunks for context expansion
            graph_context: Optional graph RAG context string
            query: The original query (used for relevance scoring)

        Returns:
            BuiltContext with formatted context and statistics
        """
        parent_chunks = parent_chunks or []
        warnings = []

        # Score chunks by query relevance if query is provided
        if query and CONTEXT_ENABLE_QUERY_SCORING and (chunks or parent_chunks):
            try:
                from .vectorstore import score_chunks_by_query

                # Score regular chunks
                if chunks:
                    chunks = score_chunks_by_query(chunks, query)
                    logger.info(f"[ContextBuilder] Scored {len(chunks)} chunks against query")

                # Score parent chunks with the same query
                if parent_chunks:
                    parent_chunks = score_chunks_by_query(parent_chunks, query)
                    logger.info(f"[ContextBuilder] Scored {len(parent_chunks)} parent chunks against query")

            except Exception as e:
                logger.warning(f"[ContextBuilder] Query scoring failed, using static scores: {e}")
                warnings.append(f"Query scoring failed: {str(e)}")

        # Prepare all chunks
        prepared_chunks = []
        for chunk in chunks:
            prepared = self._prepare_chunk(
                chunk, is_parent=False,
                available_tokens=self.chunk_budget // max(1, len(chunks))
            )
            prepared_chunks.append(prepared)

        prepared_parents = []
        for chunk in parent_chunks:
            prepared = self._prepare_chunk(
                chunk, is_parent=True,
                available_tokens=self.parent_budget // max(1, len(parent_chunks))
            )
            prepared_parents.append(prepared)

        # Select chunks within budget
        selected_chunks, chunk_tokens = self._select_chunks_within_budget(
            prepared_chunks, self.chunk_budget
        )
        selected_parents, parent_tokens = self._select_chunks_within_budget(
            prepared_parents, self.parent_budget
        )

        # Handle graph context
        graph_tokens = 0
        final_graph_context = graph_context
        if graph_context:
            graph_tokens = estimate_tokens(graph_context)
            if graph_tokens > self.graph_budget:
                # Summarize graph context if too large
                summarizer = self._get_summarizer()
                if summarizer and self.summarize_large_chunks:
                    try:
                        final_graph_context, graph_tokens = summarizer.summarize_for_context(
                            graph_context, self.graph_budget
                        )
                        warnings.append(f"Graph context summarized: {estimate_tokens(graph_context)} -> {graph_tokens} tokens")
                    except Exception:
                        # Truncate as fallback
                        max_chars = self.graph_budget * 4
                        final_graph_context = graph_context[:max_chars] + "..."
                        graph_tokens = estimate_tokens(final_graph_context)
                        warnings.append("Graph context truncated due to size")

        # Collect unique source document names from selected chunks
        unique_sources = set()
        for chunk in selected_chunks:
            if chunk.source and chunk.source != "Unknown":
                unique_sources.add(chunk.source)
        for chunk in selected_parents:
            if chunk.source and chunk.source != "Unknown":
                unique_sources.add(chunk.source)
        sources_list = sorted(list(unique_sources))

        # Build formatted context
        context_parts = []
        doc_num = 1

        # Add graph context first (if available)
        if final_graph_context:
            context_parts.append(f"[Knowledge Graph Context]\n{final_graph_context}")

        # Add regular chunks
        for chunk in selected_chunks:
            context_parts.append(self._format_chunk_for_context(chunk, doc_num))
            doc_num += 1

        # Add parent chunks
        for chunk in selected_parents:
            context_parts.append(self._format_chunk_for_context(chunk, doc_num))
            doc_num += 1

        # Add data sources section at the end
        if sources_list:
            sources_section = "\n---\n**Data Sources:**\n" + "\n".join(f"- {source}" for source in sources_list)
            context_parts.append(sources_section)

        formatted_context = "\n\n---\n\n".join(context_parts)
        total_tokens = chunk_tokens + parent_tokens + graph_tokens

        # Calculate statistics
        chunks_excluded = len(chunks) - len(selected_chunks)
        parents_summarized = sum(1 for c in selected_parents if c.is_summarized)

        if chunks_excluded > 0:
            warnings.append(f"{chunks_excluded} chunks excluded due to token limit")

        budget_usage = {
            "chunks": chunk_tokens / self.chunk_budget if self.chunk_budget > 0 else 0,
            "parents": parent_tokens / self.parent_budget if self.parent_budget > 0 else 0,
            "graph": graph_tokens / self.graph_budget if self.graph_budget > 0 else 0,
            "total": total_tokens / self.max_tokens if self.max_tokens > 0 else 0,
        }

        logger.info(
            f"Context built: {total_tokens} tokens "
            f"(chunks={chunk_tokens}, parents={parent_tokens}, graph={graph_tokens}), "
            f"{len(selected_chunks)}/{len(chunks)} chunks, "
            f"{len(selected_parents)}/{len(parent_chunks)} parents, "
            f"{len(sources_list)} unique sources"
        )

        return BuiltContext(
            formatted_context=formatted_context,
            total_tokens=total_tokens,
            chunks_included=len(selected_chunks),
            chunks_excluded=chunks_excluded,
            parent_chunks_included=len(selected_parents),
            parent_chunks_summarized=parents_summarized,
            graph_context_tokens=graph_tokens,
            budget_usage=budget_usage,
            warnings=warnings,
            sources=sources_list,
        )


# Singleton instance
_context_builder: Optional[TokenAwareContextBuilder] = None


def get_context_builder() -> TokenAwareContextBuilder:
    """Get or create the singleton context builder instance."""
    global _context_builder
    if _context_builder is None:
        _context_builder = TokenAwareContextBuilder()
    return _context_builder


def build_token_aware_context(
    chunks: List[Dict[str, Any]],
    parent_chunks: List[Dict[str, Any]] = None,
    graph_context: str = None,
    query: str = None,
    max_tokens: int = None,
) -> BuiltContext:
    """
    Convenience function to build token-aware context.

    Args:
        chunks: Regular search result chunks
        parent_chunks: Parent chunks for context expansion
        graph_context: Optional graph RAG context
        query: The original query
        max_tokens: Override max tokens (uses default if None)

    Returns:
        BuiltContext with formatted context and statistics
    """
    builder = get_context_builder()

    # Override max tokens if specified
    if max_tokens and max_tokens != builder.max_tokens:
        temp_builder = TokenAwareContextBuilder(max_tokens=max_tokens)
        return temp_builder.build_context(chunks, parent_chunks, graph_context, query)

    return builder.build_context(chunks, parent_chunks, graph_context, query)
