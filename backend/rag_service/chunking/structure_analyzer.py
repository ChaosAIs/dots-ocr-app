"""
Structure Analyzer Module for LLM-Driven Adaptive Chunking.

This module provides LLM-based document structure analysis to select
the optimal chunking strategy from a predefined library.

Single LLM call per uploaded file for structure analysis.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rag_service.llm_service import get_llm_service
from rag_service.chunking.content_sampler import ContentSampler, SampledContent

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for a chunking strategy selected by LLM."""
    selected_strategy: str
    chunk_size: int = 768
    overlap_percent: int = 10
    preserve_elements: List[str] = field(default_factory=lambda: ["tables", "code_blocks", "lists"])
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "selected_strategy": self.selected_strategy,
            "chunk_size": self.chunk_size,
            "overlap_percent": self.overlap_percent,
            "preserve_elements": self.preserve_elements,
            "reasoning": self.reasoning
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyConfig":
        """Create from dictionary."""
        return cls(
            selected_strategy=data.get("selected_strategy", "paragraph_based"),
            chunk_size=data.get("chunk_size", 768),
            overlap_percent=data.get("overlap_percent", 10),
            preserve_elements=data.get("preserve_elements", ["tables", "code_blocks", "lists"]),
            reasoning=data.get("reasoning", "")
        )


# Default fallback strategy when LLM is unavailable
DEFAULT_STRATEGY = StrategyConfig(
    selected_strategy="paragraph_based",
    chunk_size=768,
    overlap_percent=10,
    preserve_elements=["tables", "code_blocks", "lists"],
    reasoning="Default fallback - paragraph-based splitting works for most documents"
)


# LLM Prompt template for strategy selection
STRATEGY_SELECTION_PROMPT = """Analyze this document and select the best chunking strategy.

{sampled_content}

Available strategies:
1. header_based - Split at markdown headers (# ## ###), best for structured docs with sections
2. paragraph_based - Split at paragraph breaks, best for narrative text and articles
3. sentence_based - Split at sentences, best for dense text without structure
4. list_item_based - Split at list items, best for bullet points, numbered lists, TOCs
5. table_row_based - Split at table rows, best for spreadsheets and tabular data
6. citation_based - Split between citations, best for bibliographies and references
7. log_entry_based - Split at timestamps, best for log files and event logs
8. clause_based - Split at numbered clauses (1.1, 1.1.1), best for legal docs and specs

Respond in JSON only:
{{
  "selected_strategy": "<strategy_name>",
  "chunk_size": <512-1024>,
  "overlap_percent": <5-20>,
  "preserve_elements": ["tables", "code_blocks", "equations", "lists"],
  "reasoning": "Brief explanation of why this strategy fits"
}}"""


class StructureAnalyzer:
    """
    Analyzes document structure using LLM to select optimal chunking strategy.

    Uses a single LLM call per uploaded file to analyze sampled content
    and select from a predefined strategy library.
    """

    # Valid strategy names
    VALID_STRATEGIES = {
        "header_based",
        "paragraph_based",
        "sentence_based",
        "list_item_based",
        "table_row_based",
        "citation_based",
        "log_entry_based",
        "clause_based"
    }

    # Valid preserve elements
    VALID_PRESERVE_ELEMENTS = {
        "tables",
        "code_blocks",
        "equations",
        "lists",
        "citations",
        "header_row",
        "complete_citations",
        "stack_traces",
        "multi_line_entries",
        "definitions",
        "signature_blocks"
    }

    def __init__(self, llm_client=None):
        """
        Initialize the structure analyzer.

        Args:
            llm_client: Optional LLM client. If not provided, uses get_llm_service().
        """
        self.llm_client = llm_client
        self.content_sampler = ContentSampler()

    def _get_llm_model(self):
        """Get LLM model for structure analysis."""
        if self.llm_client:
            return self.llm_client

        try:
            service = get_llm_service()
            if service.is_available():
                # Use query model for faster response (structure analysis doesn't need large context)
                return service.get_query_model(
                    temperature=0.1,
                    num_ctx=4096,
                    num_predict=512
                )
        except Exception as e:
            logger.warning(f"Failed to get LLM service: {e}")

        return None

    def analyze_from_folder(self, output_folder: str) -> StrategyConfig:
        """
        Analyze document structure from output folder.

        Args:
            output_folder: Path to folder containing *_nohf.md files

        Returns:
            StrategyConfig with selected strategy and parameters
        """
        # Sample content from folder
        sampled = self.content_sampler.sample_from_folder(output_folder)

        if not sampled.first_content and not sampled.middle_content and not sampled.last_content:
            logger.warning(f"No content to analyze in {output_folder}, using default strategy")
            return DEFAULT_STRATEGY

        return self._analyze_with_llm(sampled)

    def analyze_from_content(self, content: str) -> StrategyConfig:
        """
        Analyze document structure from content string.

        Args:
            content: Full document content

        Returns:
            StrategyConfig with selected strategy and parameters
        """
        # Sample content
        sampled = self.content_sampler.sample_from_content(content)

        if not sampled.first_content:
            logger.warning("Empty content provided, using default strategy")
            return DEFAULT_STRATEGY

        return self._analyze_with_llm(sampled)

    def _analyze_with_llm(self, sampled: SampledContent) -> StrategyConfig:
        """
        Call LLM to analyze content and select strategy.

        Args:
            sampled: SampledContent with first, middle, last samples

        Returns:
            StrategyConfig selected by LLM or fallback
        """
        llm = self._get_llm_model()

        if not llm:
            logger.warning("LLM not available, using default strategy")
            return DEFAULT_STRATEGY

        # Build prompt
        prompt = STRATEGY_SELECTION_PROMPT.format(
            sampled_content=sampled.get_combined_sample()
        )

        try:
            # Call LLM
            logger.info(f"Calling LLM for structure analysis (scenario: {sampled.scenario})")
            response = llm.invoke(prompt)

            # Extract content from response
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            config = self._parse_llm_response(response_text)

            logger.info(
                f"LLM selected strategy: {config.selected_strategy}, "
                f"chunk_size: {config.chunk_size}, overlap: {config.overlap_percent}%, "
                f"reasoning: {config.reasoning}"
            )

            return config

        except Exception as e:
            logger.error(f"LLM structure analysis failed: {e}, using default strategy")
            return DEFAULT_STRATEGY

    def _parse_llm_response(self, response_text: str) -> StrategyConfig:
        """
        Parse LLM response JSON into StrategyConfig.

        Args:
            response_text: Raw text response from LLM

        Returns:
            Validated StrategyConfig
        """
        # Try to extract JSON from response (handle markdown code blocks)
        json_text = response_text

        # Remove markdown code blocks if present
        if "```json" in json_text:
            match = re.search(r'```json\s*(.*?)\s*```', json_text, re.DOTALL)
            if match:
                json_text = match.group(1)
        elif "```" in json_text:
            match = re.search(r'```\s*(.*?)\s*```', json_text, re.DOTALL)
            if match:
                json_text = match.group(1)

        # Try to find JSON object in text
        json_match = re.search(r'\{[^{}]*\}', json_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return DEFAULT_STRATEGY

        # Validate and normalize the response
        return self._validate_config(data)

    def _validate_config(self, data: Dict[str, Any]) -> StrategyConfig:
        """
        Validate and normalize LLM response data.

        Args:
            data: Parsed JSON data from LLM

        Returns:
            Validated StrategyConfig
        """
        # Validate strategy name
        strategy = data.get("selected_strategy", "paragraph_based")
        if strategy not in self.VALID_STRATEGIES:
            logger.warning(f"Invalid strategy '{strategy}', falling back to paragraph_based")
            strategy = "paragraph_based"

        # Validate chunk_size (512-1024)
        chunk_size = data.get("chunk_size", 768)
        try:
            chunk_size = int(chunk_size)
            chunk_size = max(512, min(1024, chunk_size))
        except (ValueError, TypeError):
            chunk_size = 768

        # Validate overlap_percent (5-20)
        overlap = data.get("overlap_percent", 10)
        try:
            overlap = int(overlap)
            overlap = max(5, min(20, overlap))
        except (ValueError, TypeError):
            overlap = 10

        # Validate preserve_elements
        preserve = data.get("preserve_elements", ["tables", "code_blocks", "lists"])
        if not isinstance(preserve, list):
            preserve = ["tables", "code_blocks", "lists"]
        else:
            # Filter to valid elements
            preserve = [e for e in preserve if e in self.VALID_PRESERVE_ELEMENTS]
            if not preserve:
                preserve = ["tables", "code_blocks", "lists"]

        # Get reasoning
        reasoning = data.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = ""

        return StrategyConfig(
            selected_strategy=strategy,
            chunk_size=chunk_size,
            overlap_percent=overlap,
            preserve_elements=preserve,
            reasoning=reasoning
        )


# Convenience functions

def analyze_document_structure(output_folder: str, llm_client=None) -> StrategyConfig:
    """
    Analyze document structure and get chunking strategy.

    Args:
        output_folder: Path to folder containing *_nohf.md files
        llm_client: Optional LLM client

    Returns:
        StrategyConfig with selected strategy
    """
    analyzer = StructureAnalyzer(llm_client=llm_client)
    return analyzer.analyze_from_folder(output_folder)


def analyze_content_structure(content: str, llm_client=None) -> StrategyConfig:
    """
    Analyze content structure and get chunking strategy.

    Args:
        content: Document content
        llm_client: Optional LLM client

    Returns:
        StrategyConfig with selected strategy
    """
    analyzer = StructureAnalyzer(llm_client=llm_client)
    return analyzer.analyze_from_content(content)
