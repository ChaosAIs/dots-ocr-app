"""
Strategy Executor Module for LLM-Driven Adaptive Chunking.

This module implements the predefined chunking strategy library and
executes strategies based on LLM-selected configuration.

Strategies:
- header_based: Split at markdown headers
- paragraph_based: Split at paragraph breaks
- sentence_based: Split at sentences
- list_item_based: Split at list items
- table_row_based: Split at table rows
- citation_based: Split between citations
- log_entry_based: Split at timestamps
- clause_based: Split at numbered clauses
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple

from rag_service.chunking.structure_analyzer import StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A document chunk with content and metadata."""
    content: str
    index: int
    start_position: int = 0
    end_position: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyDefinition:
    """Definition of a chunking strategy."""
    name: str
    description: str
    split_pattern: str
    default_chunk_size: int = 768
    default_overlap: int = 10
    preserve_header: bool = False  # For table_row_based: repeat header row


# Predefined strategy library
STRATEGIES: Dict[str, StrategyDefinition] = {
    "header_based": StrategyDefinition(
        name="header_based",
        description="Split at markdown headers (# ## ###)",
        split_pattern=r"^#{1,6}\s+",
        default_chunk_size=768,
        default_overlap=10
    ),
    "paragraph_based": StrategyDefinition(
        name="paragraph_based",
        description="Split at paragraph breaks (double newlines)",
        split_pattern=r"\n\n+",
        default_chunk_size=768,
        default_overlap=15
    ),
    "sentence_based": StrategyDefinition(
        name="sentence_based",
        description="Split at sentence boundaries (. ! ?)",
        split_pattern=r"(?<=[.!?])\s+",
        default_chunk_size=768,
        default_overlap=20
    ),
    "list_item_based": StrategyDefinition(
        name="list_item_based",
        description="Split at list item boundaries",
        split_pattern=r"^[\s]*[-*\u2022]\s+|^[\s]*\d+\.\s+",
        default_chunk_size=768,
        default_overlap=5
    ),
    "table_row_based": StrategyDefinition(
        name="table_row_based",
        description="Split at table row boundaries",
        split_pattern=r"^\|.*\|$",
        default_chunk_size=1024,
        default_overlap=0,
        preserve_header=True
    ),
    "citation_based": StrategyDefinition(
        name="citation_based",
        description="Split between citation entries",
        split_pattern=r"(?=^[A-Z][a-z]+,?\s+[A-Z])|(?=^\[\d+\])",
        default_chunk_size=512,
        default_overlap=5
    ),
    "log_entry_based": StrategyDefinition(
        name="log_entry_based",
        description="Split at timestamp boundaries",
        split_pattern=r"^\d{4}-\d{2}-\d{2}|^\[[\d:]+\]",
        default_chunk_size=1024,
        default_overlap=10
    ),
    "clause_based": StrategyDefinition(
        name="clause_based",
        description="Split at numbered clauses (1.1, 1.1.1)",
        split_pattern=r"^[\s]*\d+(\.\d+)*[\s]+",
        default_chunk_size=1024,
        default_overlap=5
    )
}


# Atomic element patterns (never split these)
ATOMIC_PATTERNS = {
    "table": re.compile(r"^\|.*\|$.*?(?=^\s*$|^[^|])", re.MULTILINE | re.DOTALL),
    "code_block": re.compile(r"```[\s\S]*?```", re.MULTILINE),
    "inline_code": re.compile(r"`[^`]+`"),
    "latex_block": re.compile(r"\$\$[\s\S]*?\$\$"),
    "latex_inline": re.compile(r"\$[^$]+\$"),
    "html_block": re.compile(r"<[a-zA-Z][^>]*>[\s\S]*?</[a-zA-Z]+>", re.MULTILINE)
}


class StrategyExecutor:
    """
    Executes chunking strategies based on LLM-selected configuration.

    Applies universal rules (sentence boundaries, atomic elements) and
    strategy-specific split patterns.
    """

    # Size constraints
    MIN_CHUNK_SIZE = 256  # tokens (approximate via chars/4)
    MAX_CHUNK_SIZE = 1536  # tokens
    CHARS_PER_TOKEN = 4  # Rough approximation

    def __init__(self):
        """Initialize the strategy executor."""
        pass

    def execute(self, content: str, config: StrategyConfig) -> List[Chunk]:
        """
        Execute chunking strategy on content.

        Args:
            content: Document content to chunk
            config: Strategy configuration from LLM

        Returns:
            List of Chunk objects
        """
        if not content or not content.strip():
            return []

        strategy_name = config.selected_strategy
        if strategy_name not in STRATEGIES:
            logger.warning(f"Unknown strategy '{strategy_name}', using paragraph_based")
            strategy_name = "paragraph_based"

        strategy = STRATEGIES[strategy_name]

        logger.info(
            f"Executing strategy: {strategy_name}, "
            f"chunk_size: {config.chunk_size}, overlap: {config.overlap_percent}%"
        )

        # Step 1: Find and protect atomic elements
        protected_ranges = self._find_atomic_elements(content, config.preserve_elements)

        # Step 2: Find split points using strategy pattern
        split_points = self._find_split_points(content, strategy.split_pattern)

        # Step 3: Filter splits inside protected ranges
        valid_splits = self._filter_protected_splits(split_points, protected_ranges)

        # Step 4: Create initial chunks at split points
        initial_chunks = self._create_chunks_at_splits(content, valid_splits)

        # Step 5: Apply size constraints
        sized_chunks = self._apply_size_constraints(
            initial_chunks,
            target_size=config.chunk_size * self.CHARS_PER_TOKEN,
            min_size=self.MIN_CHUNK_SIZE * self.CHARS_PER_TOKEN,
            max_size=self.MAX_CHUNK_SIZE * self.CHARS_PER_TOKEN
        )

        # Step 6: Add overlap
        final_chunks = self._add_overlap(sized_chunks, config.overlap_percent)

        # Step 7: For table_row_based, prepend header row
        if strategy.preserve_header and strategy_name == "table_row_based":
            final_chunks = self._prepend_table_header(content, final_chunks)

        logger.info(f"Created {len(final_chunks)} chunks using {strategy_name} strategy")

        return final_chunks

    def _find_atomic_elements(
        self,
        content: str,
        preserve_elements: List[str]
    ) -> List[Tuple[int, int]]:
        """
        Find ranges of atomic elements that should not be split.

        Args:
            content: Document content
            preserve_elements: List of element types to preserve

        Returns:
            List of (start, end) tuples for protected ranges
        """
        protected = []

        # Map preserve_elements to patterns
        element_pattern_map = {
            "tables": ["table"],
            "code_blocks": ["code_block", "inline_code"],
            "equations": ["latex_block", "latex_inline"],
            "lists": [],  # Lists are handled by split pattern, not protection
            "header_row": ["table"],
            "complete_citations": [],
            "stack_traces": [],
            "multi_line_entries": [],
            "definitions": [],
            "signature_blocks": []
        }

        patterns_to_use = set()
        for elem in preserve_elements:
            if elem in element_pattern_map:
                patterns_to_use.update(element_pattern_map[elem])

        # Always protect code blocks and tables by default
        patterns_to_use.add("code_block")
        patterns_to_use.add("table")

        for pattern_name in patterns_to_use:
            if pattern_name in ATOMIC_PATTERNS:
                pattern = ATOMIC_PATTERNS[pattern_name]
                for match in pattern.finditer(content):
                    protected.append((match.start(), match.end()))

        # Merge overlapping ranges
        protected.sort()
        merged = []
        for start, end in protected:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        return merged

    def _find_split_points(self, content: str, pattern: str) -> List[int]:
        """
        Find potential split points using strategy pattern.

        Args:
            content: Document content
            pattern: Regex pattern for finding split points

        Returns:
            Sorted list of character positions where splits can occur
        """
        try:
            regex = re.compile(pattern, re.MULTILINE)
            points = []

            for match in regex.finditer(content):
                # Split before the match
                points.append(match.start())

            return sorted(set(points))

        except re.error as e:
            logger.error(f"Invalid split pattern '{pattern}': {e}")
            # Fallback to paragraph splitting
            points = []
            for match in re.finditer(r"\n\n+", content):
                points.append(match.start())
            return sorted(set(points))

    def _filter_protected_splits(
        self,
        split_points: List[int],
        protected_ranges: List[Tuple[int, int]]
    ) -> List[int]:
        """
        Remove split points that fall inside protected ranges.

        Args:
            split_points: List of potential split positions
            protected_ranges: List of (start, end) protected ranges

        Returns:
            Filtered list of valid split points
        """
        def is_in_protected(pos: int) -> bool:
            for start, end in protected_ranges:
                if start <= pos < end:
                    return True
            return False

        return [p for p in split_points if not is_in_protected(p)]

    def _create_chunks_at_splits(
        self,
        content: str,
        split_points: List[int]
    ) -> List[Chunk]:
        """
        Create initial chunks at split points.

        Args:
            content: Document content
            split_points: List of positions to split at

        Returns:
            List of Chunk objects
        """
        if not split_points:
            # No splits found, return whole content as single chunk
            return [Chunk(
                content=content.strip(),
                index=0,
                start_position=0,
                end_position=len(content)
            )]

        chunks = []

        # Add 0 and end positions
        all_points = [0] + split_points + [len(content)]
        all_points = sorted(set(all_points))

        for i in range(len(all_points) - 1):
            start = all_points[i]
            end = all_points[i + 1]
            chunk_content = content[start:end].strip()

            if chunk_content:
                chunks.append(Chunk(
                    content=chunk_content,
                    index=len(chunks),
                    start_position=start,
                    end_position=end
                ))

        return chunks

    def _apply_size_constraints(
        self,
        chunks: List[Chunk],
        target_size: int,
        min_size: int,
        max_size: int
    ) -> List[Chunk]:
        """
        Apply size constraints to chunks (merge small, split large).

        Args:
            chunks: Initial chunk list
            target_size: Target chunk size in characters
            min_size: Minimum chunk size
            max_size: Maximum chunk size

        Returns:
            Size-adjusted chunk list
        """
        if not chunks:
            return []

        result = []

        for chunk in chunks:
            chunk_len = len(chunk.content)

            if chunk_len < min_size and result:
                # Merge with previous chunk
                prev = result[-1]
                merged_content = prev.content + "\n\n" + chunk.content
                result[-1] = Chunk(
                    content=merged_content,
                    index=prev.index,
                    start_position=prev.start_position,
                    end_position=chunk.end_position,
                    metadata=prev.metadata
                )
            elif chunk_len > max_size:
                # Force split at secondary boundaries
                sub_chunks = self._force_split(chunk, target_size, max_size)
                for sub in sub_chunks:
                    sub.index = len(result)
                    result.append(sub)
            else:
                chunk.index = len(result)
                result.append(chunk)

        return result

    def _force_split(
        self,
        chunk: Chunk,
        target_size: int,
        max_size: int
    ) -> List[Chunk]:
        """
        Force split an oversized chunk at secondary boundaries.

        Secondary boundaries (in order of preference):
        1. Paragraph breaks (\n\n)
        2. Sentence endings (. ! ?)
        3. Word boundaries (spaces)

        Args:
            chunk: Oversized chunk to split
            target_size: Target size for sub-chunks
            max_size: Maximum allowed size

        Returns:
            List of sub-chunks
        """
        content = chunk.content
        if len(content) <= max_size:
            return [chunk]

        sub_chunks = []
        current_pos = 0

        while current_pos < len(content):
            # Determine end position for this sub-chunk
            end_pos = min(current_pos + target_size, len(content))

            if end_pos >= len(content):
                # Last chunk
                sub_content = content[current_pos:].strip()
                if sub_content:
                    sub_chunks.append(Chunk(
                        content=sub_content,
                        index=len(sub_chunks),
                        start_position=chunk.start_position + current_pos,
                        end_position=chunk.start_position + len(content)
                    ))
                break

            # Look for split point near end_pos
            best_split = self._find_best_split_point(content, current_pos, end_pos, max_size)

            sub_content = content[current_pos:best_split].strip()
            if sub_content:
                sub_chunks.append(Chunk(
                    content=sub_content,
                    index=len(sub_chunks),
                    start_position=chunk.start_position + current_pos,
                    end_position=chunk.start_position + best_split
                ))

            current_pos = best_split

        return sub_chunks if sub_chunks else [chunk]

    def _find_best_split_point(
        self,
        content: str,
        start: int,
        target_end: int,
        max_end: int
    ) -> int:
        """
        Find best split point near target_end.

        Args:
            content: Full content
            start: Current chunk start
            target_end: Ideal end position
            max_end: Maximum end position

        Returns:
            Best split position
        """
        search_start = max(start + 100, target_end - 200)
        search_end = min(len(content), max_end)
        search_region = content[search_start:search_end]

        # Try paragraph break first
        para_match = re.search(r"\n\n+", search_region)
        if para_match:
            return search_start + para_match.end()

        # Try sentence ending
        sent_match = re.search(r"[.!?]\s+", search_region)
        if sent_match:
            return search_start + sent_match.end()

        # Try word boundary
        word_match = re.search(r"\s+", search_region[::-1])
        if word_match:
            return search_end - word_match.start()

        # Fallback to target_end
        return target_end

    def _add_overlap(self, chunks: List[Chunk], overlap_percent: int) -> List[Chunk]:
        """
        Add overlap between consecutive chunks.

        Args:
            chunks: List of chunks
            overlap_percent: Percentage overlap (5-20)

        Returns:
            Chunks with overlap added
        """
        if not chunks or len(chunks) < 2 or overlap_percent <= 0:
            return chunks

        result = [chunks[0]]  # First chunk stays as-is

        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]

            # Calculate overlap size
            overlap_chars = int(len(prev.content) * overlap_percent / 100)

            if overlap_chars > 0:
                # Get overlap text from end of previous chunk
                overlap_text = prev.content[-overlap_chars:]

                # Find a good break point (sentence or paragraph)
                break_match = re.search(r"[.!?]\s+", overlap_text)
                if break_match:
                    overlap_text = overlap_text[break_match.end():]

                # Prepend to current chunk
                if overlap_text.strip():
                    curr = Chunk(
                        content=overlap_text.strip() + "\n\n" + curr.content,
                        index=curr.index,
                        start_position=curr.start_position - len(overlap_text),
                        end_position=curr.end_position,
                        metadata=curr.metadata
                    )

            result.append(curr)

        return result

    def _prepend_table_header(self, content: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        Prepend table header row to each chunk (for table_row_based strategy).

        Args:
            content: Original content
            chunks: List of chunks

        Returns:
            Chunks with header prepended
        """
        # Find the first table header row
        header_match = re.search(r"^\|.*\|\s*\n\|[-:| ]+\|", content, re.MULTILINE)
        if not header_match:
            return chunks

        header_text = header_match.group(0)

        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk already has header
                result.append(chunk)
            else:
                # Prepend header to subsequent chunks
                if not chunk.content.startswith("|"):
                    result.append(chunk)
                else:
                    result.append(Chunk(
                        content=header_text + "\n" + chunk.content,
                        index=chunk.index,
                        start_position=chunk.start_position,
                        end_position=chunk.end_position,
                        metadata={**chunk.metadata, "header_prepended": True}
                    ))

        return result


# Convenience function
def execute_strategy(content: str, config: StrategyConfig) -> List[Chunk]:
    """
    Execute chunking strategy on content.

    Args:
        content: Document content
        config: Strategy configuration from LLM

    Returns:
        List of Chunk objects
    """
    executor = StrategyExecutor()
    return executor.execute(content, config)
