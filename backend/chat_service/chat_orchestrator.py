"""
Chat Orchestrator - Routes queries to appropriate services based on intent.

This is the central orchestrator that:
1. Classifies user intent (document_search, data_analytics, hybrid, general)
2. Routes to the appropriate service (RAG, SQL Analytics, or both)
3. Merges results for hybrid queries
4. Formats final response for streaming
"""

import logging
import os
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable
from uuid import UUID
from datetime import datetime

from sqlalchemy.orm import Session

from analytics_service.intent_classifier import IntentClassifier, QueryIntent, IntentClassification
from analytics_service.sql_query_executor import SQLQueryExecutor
from db.models import DataSchema

logger = logging.getLogger(__name__)

# Configuration
ANALYTICS_ENABLED = os.getenv("ANALYTICS_ENABLED", "true").lower() == "true"
ANALYTICS_MIN_CONFIDENCE = float(os.getenv("ANALYTICS_MIN_CONFIDENCE", "0.6"))


class ChatOrchestrator:
    """
    Orchestrates chat queries by routing to appropriate services.

    Flow:
    1. Receive user message
    2. Classify intent using IntentClassifier
    3. Route to service:
       - DOCUMENT_SEARCH → RAG Service (existing)
       - DATA_ANALYTICS → SQL Query Executor
       - HYBRID → Both services, merge results
       - GENERAL → Direct LLM response
    4. Stream response back to user
    """

    def __init__(self, db: Session):
        self.db = db
        self.intent_classifier = IntentClassifier()
        self.sql_executor = SQLQueryExecutor(db)

    def classify_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> IntentClassification:
        """
        Classify the user query to determine routing.

        Args:
            query: User's message
            conversation_history: Previous messages for context

        Returns:
            IntentClassification with intent and metadata
        """
        # Get available schemas for classification context
        available_schemas = self._get_available_schemas()
        logger.info(f"[Orchestrator] Available schemas from DB: {available_schemas}")

        # Classify the query
        classification = self.intent_classifier.classify(
            query=query,
            available_schemas=available_schemas
        )

        logger.info(f"[Orchestrator] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[Orchestrator] INTENT CLASSIFICATION RESULT:")
        logger.info(f"[Orchestrator]   • Query: '{query[:80]}...'")
        logger.info(f"[Orchestrator]   • Intent: {classification.intent.value}")
        logger.info(f"[Orchestrator]   • Confidence: {classification.confidence:.2f}")
        logger.info(f"[Orchestrator]   • Reasoning: {classification.reasoning}")
        logger.info(f"[Orchestrator]   • Requires extracted data: {classification.requires_extracted_data}")
        logger.info(f"[Orchestrator]   • Suggested schemas: {classification.suggested_schemas}")
        logger.info(f"[Orchestrator]   • Detected entities: {classification.detected_entities}")
        logger.info(f"[Orchestrator]   • Detected metrics: {classification.detected_metrics}")
        logger.info(f"[Orchestrator]   • Time range: {classification.detected_time_range}")
        logger.info(f"[Orchestrator] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return classification

    def _get_available_schemas(self) -> List[str]:
        """Get list of available schema types from database."""
        try:
            schemas = self.db.query(DataSchema.schema_type).filter(
                DataSchema.is_active == True
            ).all()
            return [s[0] for s in schemas]
        except Exception as e:
            logger.warning(f"Failed to get schemas: {e}")
            # Return default schemas
            return ['invoice', 'receipt', 'bank_statement', 'expense_report', 'purchase_order']

    def should_use_analytics(self, classification: IntentClassification) -> bool:
        """
        Determine if analytics service should be used.

        Args:
            classification: Intent classification result

        Returns:
            True if analytics/SQL path should be used
        """
        if not ANALYTICS_ENABLED:
            return False

        # Use analytics for DATA_ANALYTICS or HYBRID with sufficient confidence
        if classification.intent == QueryIntent.DATA_ANALYTICS:
            return classification.confidence >= ANALYTICS_MIN_CONFIDENCE

        if classification.intent == QueryIntent.HYBRID:
            return classification.requires_extracted_data

        return False

    def should_use_rag(self, classification: IntentClassification) -> bool:
        """
        Determine if RAG service should be used.

        Args:
            classification: Intent classification result

        Returns:
            True if RAG/vector search path should be used
        """
        # Use RAG for DOCUMENT_SEARCH, HYBRID, or low-confidence analytics
        if classification.intent == QueryIntent.DOCUMENT_SEARCH:
            return True

        if classification.intent == QueryIntent.HYBRID:
            return True

        # Fallback to RAG if analytics confidence is low
        if classification.intent == QueryIntent.DATA_ANALYTICS:
            return classification.confidence < ANALYTICS_MIN_CONFIDENCE

        # GENERAL queries don't need RAG
        return False

    def execute_analytics_query(
        self,
        query: str,
        classification: IntentClassification,
        accessible_doc_ids: List[UUID]
    ) -> Dict[str, Any]:
        """
        Execute analytics query using SQL executor.

        Args:
            query: User's original query
            classification: Intent classification result
            accessible_doc_ids: Document IDs user can access

        Returns:
            Analytics query results
        """
        logger.info(f"[Orchestrator] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[Orchestrator] EXECUTING ANALYTICS QUERY")
        logger.info(f"[Orchestrator]   • Query: '{query[:80]}...'")
        logger.info(f"[Orchestrator]   • Accessible documents: {len(accessible_doc_ids)}")

        # Convert classification to dict for executor
        intent_dict = {
            "suggested_schemas": classification.suggested_schemas,
            "detected_time_range": classification.detected_time_range,
            "detected_entities": classification.detected_entities,
            "detected_metrics": classification.detected_metrics
        }
        logger.info(f"[Orchestrator]   • Intent dict: {intent_dict}")
        logger.info(f"[Orchestrator] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        result = self.sql_executor.execute_natural_language_query(
            accessible_doc_ids=accessible_doc_ids,
            intent_classification=intent_dict,
            query=query
        )

        logger.info(f"[Orchestrator] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[Orchestrator] ANALYTICS QUERY RESULT:")
        logger.info(f"[Orchestrator]   • Data rows: {len(result.get('data', []))}")
        logger.info(f"[Orchestrator]   • Summary: {result.get('summary', {})}")
        logger.info(f"[Orchestrator]   • Metadata: {result.get('metadata', {})}")
        logger.info(f"[Orchestrator] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return result

    def get_data_availability(self, accessible_doc_ids: List[UUID]) -> Dict[str, Any]:
        """
        Check what extracted data is available for the user.

        Args:
            accessible_doc_ids: Document IDs user can access

        Returns:
            Summary of available data by schema type
        """
        return self.sql_executor.get_available_data_summary(accessible_doc_ids)

    def format_analytics_response(
        self,
        query: str,
        analytics_result: Dict[str, Any],
        classification: IntentClassification
    ) -> str:
        """
        Format analytics results into a readable response.

        Args:
            query: Original user query
            analytics_result: Results from SQL executor
            classification: Intent classification

        Returns:
            Formatted response string
        """
        data = analytics_result.get('data', [])
        summary = analytics_result.get('summary', {})
        metadata = analytics_result.get('metadata', {})

        if 'error' in summary:
            return f"I encountered an error while analyzing your data: {summary['error']}"

        if not data and summary.get('total_records', 0) == 0:
            return self._format_no_data_response(classification)

        # Build response based on query type
        response_parts = []

        # Header
        schema_types = metadata.get('schema_types', [])
        time_range = metadata.get('time_range', {})

        if schema_types:
            doc_types = ', '.join(schema_types)
            response_parts.append(f"Based on your {doc_types} documents")

        if time_range:
            start = time_range.get('start', '')
            end = time_range.get('end', '')
            if start and end:
                response_parts.append(f"from {start} to {end}")

        if response_parts:
            response_parts[0] = response_parts[0] + ":"
            header = " ".join(response_parts)
        else:
            header = "Here are the results:"

        # Summary stats
        summary_lines = []
        if 'total_amount' in summary:
            summary_lines.append(f"- **Total Amount**: ${summary['total_amount']:,.2f}")
        if 'count' in summary:
            summary_lines.append(f"- **Document Count**: {summary['count']}")
        if 'avg_amount' in summary:
            summary_lines.append(f"- **Average Amount**: ${summary['avg_amount']:,.2f}")
        if 'min_amount' in summary:
            summary_lines.append(f"- **Minimum**: ${summary['min_amount']:,.2f}")
        if 'max_amount' in summary:
            summary_lines.append(f"- **Maximum**: ${summary['max_amount']:,.2f}")

        # Data table (if grouped)
        table_lines = []
        if data and isinstance(data[0], dict) and 'group' in data[0]:
            # Grouped data - create table
            table_lines.append("\n| Group | Count | Total Amount |")
            table_lines.append("|-------|-------|--------------|")
            for row in data[:20]:  # Limit to 20 rows
                group = row.get('group', 'Unknown')
                count = row.get('count', 0)
                total = row.get('total_amount', 0)
                table_lines.append(f"| {group} | {count} | ${total:,.2f} |")

            if len(data) > 20:
                table_lines.append(f"\n*...and {len(data) - 20} more groups*")

        # Individual records (if not grouped)
        elif data and not ('group' in data[0] if data else False):
            if len(data) <= 10:
                table_lines.append("\n| Date | Entity | Amount | Type |")
                table_lines.append("|------|--------|--------|------|")
                for row in data:
                    date = row.get('date', 'N/A')
                    entity = row.get('entity_name', 'N/A')
                    amount = row.get('amount', 0)
                    schema = row.get('schema_type', 'N/A')
                    amount_str = f"${amount:,.2f}" if amount else "N/A"
                    table_lines.append(f"| {date} | {entity} | {amount_str} | {schema} |")
            else:
                table_lines.append(f"\n*Found {len(data)} matching documents.*")

        # Combine response
        response = header + "\n\n"
        if summary_lines:
            response += "**Summary:**\n" + "\n".join(summary_lines) + "\n"
        if table_lines:
            response += "\n".join(table_lines)

        return response

    def _format_no_data_response(self, classification: IntentClassification) -> str:
        """Format response when no data is found."""
        schemas = classification.suggested_schemas
        time_range = classification.detected_time_range

        response = "I couldn't find any matching data"

        if schemas:
            response += f" in your {', '.join(schemas)} documents"

        if time_range:
            start = time_range.get('start', '')
            end = time_range.get('end', '')
            if start and end:
                response += f" between {start} and {end}"

        response += ".\n\nThis could mean:\n"
        response += "- No documents of this type have been uploaded\n"
        response += "- The documents haven't been processed for data extraction yet\n"
        response += "- The filter criteria don't match any records\n"

        return response

    def merge_hybrid_results(
        self,
        analytics_result: Dict[str, Any],
        rag_context: str,
        query: str
    ) -> str:
        """
        Merge results from analytics and RAG for hybrid queries.

        Args:
            analytics_result: Results from SQL executor
            rag_context: Context retrieved from RAG
            query: Original user query

        Returns:
            Combined context for LLM response generation
        """
        merged_parts = []

        # Add analytics data summary
        if analytics_result.get('data'):
            analytics_summary = self._summarize_analytics_for_context(analytics_result)
            merged_parts.append(f"**Structured Data Analysis:**\n{analytics_summary}")

        # Add RAG context
        if rag_context:
            merged_parts.append(f"**Document Content:**\n{rag_context}")

        return "\n\n---\n\n".join(merged_parts)

    def _summarize_analytics_for_context(self, analytics_result: Dict[str, Any]) -> str:
        """Summarize analytics results for inclusion in LLM context."""
        summary = analytics_result.get('summary', {})
        data = analytics_result.get('data', [])

        parts = []

        # Add summary stats
        if 'total_amount' in summary:
            parts.append(f"Total amount: ${summary['total_amount']:,.2f}")
        if 'count' in summary:
            parts.append(f"Document count: {summary['count']}")
        if 'avg_amount' in summary:
            parts.append(f"Average: ${summary['avg_amount']:,.2f}")

        # Add grouped data if available
        if data and 'group' in data[0]:
            parts.append("\nBreakdown:")
            for row in data[:10]:
                group = row.get('group', 'Unknown')
                total = row.get('total_amount', 0)
                count = row.get('count', 0)
                parts.append(f"  - {group}: ${total:,.2f} ({count} documents)")

        return "\n".join(parts)


class OrchestratorResult:
    """Result from orchestrator processing."""

    def __init__(
        self,
        intent: QueryIntent,
        classification: IntentClassification,
        use_analytics: bool,
        use_rag: bool,
        analytics_result: Optional[Dict[str, Any]] = None,
        formatted_analytics: Optional[str] = None,
        rag_context: Optional[str] = None,
        merged_context: Optional[str] = None
    ):
        self.intent = intent
        self.classification = classification
        self.use_analytics = use_analytics
        self.use_rag = use_rag
        self.analytics_result = analytics_result
        self.formatted_analytics = formatted_analytics
        self.rag_context = rag_context
        self.merged_context = merged_context

    def get_context_for_llm(self) -> Optional[str]:
        """Get the appropriate context for LLM generation."""
        if self.merged_context:
            return self.merged_context
        if self.use_analytics and self.formatted_analytics:
            return self.formatted_analytics
        return self.rag_context

    def should_stream_analytics_directly(self) -> bool:
        """
        Determine if analytics results should be streamed directly
        without additional LLM processing.
        """
        # For pure analytics queries with good results, stream directly
        return (
            self.intent == QueryIntent.DATA_ANALYTICS and
            self.use_analytics and
            not self.use_rag and
            self.analytics_result and
            self.analytics_result.get('data')
        )
