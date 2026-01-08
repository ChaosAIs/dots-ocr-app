"""
Bridge to existing result formatter service.

DEPRECATED: This bridge is no longer used. The Summary Agent now directly uses
LLMSQLGenerator.generate_summary_report() from analytics_service to ensure
consistent response quality between agent and non-agent flows.

This file is kept for backwards compatibility but may be removed in future versions.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class FormatterBridge:
    """Bridge to existing LLMResultFormatter service."""

    def __init__(self):
        self._result_formatter = None
        self._response_formatter = None

    @property
    def result_formatter(self):
        """Lazy load result formatter."""
        if self._result_formatter is None:
            try:
                from backend.analytics_service.llm_result_formatter import LLMResultFormatter
                self._result_formatter = LLMResultFormatter()
            except ImportError:
                logger.warning("LLMResultFormatter not available")
        return self._result_formatter

    @property
    def response_formatter(self):
        """Lazy load response formatter."""
        if self._response_formatter is None:
            try:
                from backend.analytics_service.response_formatter import ResponseFormatter
                self._response_formatter = ResponseFormatter()
            except ImportError:
                logger.warning("ResponseFormatter not available")
        return self._response_formatter

    def format_results(
        self,
        data: Any,
        query: str,
        data_type: Optional[str] = None
    ) -> str:
        """Format query results using LLM.

        Args:
            data: Data to format
            query: Original query for context
            data_type: Type of data (aggregation, list, etc.)

        Returns:
            Formatted markdown string
        """
        if self.result_formatter:
            try:
                return self.result_formatter.format(
                    data=data,
                    query=query,
                    data_type=data_type
                )
            except Exception as e:
                logger.error(f"LLM formatting failed: {e}")

        # Fallback: Simple formatting
        return self._fallback_format(data, query)

    def format_response(
        self,
        results: List[Dict[str, Any]],
        data_sources: List[str],
        confidence: float = 1.0
    ) -> str:
        """Format complete response.

        Args:
            results: All results to include
            data_sources: Source documents
            confidence: Overall confidence score

        Returns:
            Formatted response
        """
        if self.response_formatter:
            try:
                return self.response_formatter.format_response(
                    results=results,
                    data_sources=data_sources,
                    confidence=confidence
                )
            except Exception as e:
                logger.error(f"Response formatting failed: {e}")

        # Fallback
        return self._fallback_response(results, data_sources)

    def _fallback_format(self, data: Any, query: str) -> str:
        """Fallback formatting without LLM."""
        if data is None:
            return "No data available."

        if isinstance(data, dict):
            # Check for aggregation
            if "aggregation" in data:
                agg_type = data.get("aggregation", "")
                value = data.get("value", 0)
                field = data.get("field", "value")

                if isinstance(value, float):
                    formatted = f"{value:,.2f}"
                else:
                    formatted = f"{value:,}" if isinstance(value, int) else str(value)

                return f"**{agg_type.title()} of {field}:** {formatted}"

            # Generic dict
            lines = []
            for key, value in data.items():
                if value is not None:
                    label = key.replace("_", " ").title()
                    lines.append(f"- **{label}:** {value}")
            return "\n".join(lines)

        if isinstance(data, list):
            if not data:
                return "No results found."

            lines = []
            for item in data[:10]:  # Limit to 10
                if isinstance(item, dict):
                    item_parts = [
                        f"{k}: {v}" for k, v in item.items()
                        if v is not None and k != "document_id"
                    ]
                    lines.append(f"- {', '.join(item_parts)}")
                else:
                    lines.append(f"- {item}")

            if len(data) > 10:
                lines.append(f"... and {len(data) - 10} more results")

            return "\n".join(lines)

        return str(data)

    def _fallback_response(
        self,
        results: List[Dict[str, Any]],
        data_sources: List[str]
    ) -> str:
        """Fallback response formatting."""
        parts = ["## Summary:\n"]

        for result in results:
            data = result.get("data")
            if data:
                formatted = self._fallback_format(data, "")
                parts.append(formatted)
                parts.append("")

        if data_sources:
            parts.append("\n**Data Sources:**")
            for source in list(set(data_sources)):
                parts.append(f"- {source}")

        return "\n".join(parts)

    def detect_format_type(self, data: Any) -> str:
        """Detect the best format type for data.

        Args:
            data: Data to analyze

        Returns:
            Format type string
        """
        if isinstance(data, dict):
            if "aggregation" in data or "total" in str(data).lower():
                return "aggregation"
            return "key_value"

        if isinstance(data, list):
            if not data:
                return "empty"
            if len(data) > 5:
                return "table"
            return "list"

        return "scalar"
