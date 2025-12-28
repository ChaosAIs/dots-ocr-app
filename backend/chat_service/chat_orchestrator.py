"""
Chat Orchestrator - Routes queries to appropriate services based on intent.

This is the central orchestrator that:
1. Classifies user intent (document_search, data_analytics, hybrid, general)
2. Routes to the appropriate service (RAG, SQL Analytics, or both)
3. Merges results for hybrid queries
4. Formats final response for streaming

Enhanced with LLM-based dynamic SQL generation for more accurate analytics queries.
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
# Enable LLM-based SQL generation (multi-round analysis)
ANALYTICS_USE_LLM = os.getenv("ANALYTICS_USE_LLM", "true").lower() == "true"


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
        self.llm_client = self._create_llm_client() if ANALYTICS_USE_LLM else None

    def _create_llm_client(self):
        """Create LLM client for dynamic SQL generation."""
        try:
            from rag_service.llm_service import get_llm_service
            from langchain_core.messages import HumanMessage

            llm_service = get_llm_service()
            if not llm_service.is_available():
                logger.warning("[Orchestrator] LLM service not available for SQL generation")
                return None

            # Get a chat model optimized for SQL generation
            chat_model = llm_service.get_query_model(
                temperature=0.1,  # Low temperature for consistent SQL
                num_ctx=4096,
                num_predict=2048
            )

            # Wrapper to provide simple generate() interface
            class LLMClientWrapper:
                def __init__(self, model):
                    self.model = model

                def generate(self, prompt: str) -> str:
                    response = self.model.invoke([HumanMessage(content=prompt)])
                    return response.content

            logger.info("[Orchestrator] LLM client created for dynamic SQL generation")
            return LLMClientWrapper(chat_model)

        except Exception as e:
            logger.warning(f"[Orchestrator] Failed to create LLM client: {e}")
            return None

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
        Execute analytics query using LLM-driven dynamic SQL generation.

        This method uses LLM to:
        1. Analyze the query and understand user intent
        2. Generate appropriate SQL based on field mappings
        3. Execute SQL and format results
        4. Generate natural language summary

        Args:
            query: User's original query
            classification: Intent classification result
            accessible_doc_ids: Document IDs user can access

        Returns:
            Analytics query results with LLM-formatted report
        """
        logger.info(f"[Orchestrator] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[Orchestrator] EXECUTING ANALYTICS QUERY (LLM-DRIVEN)")
        logger.info(f"[Orchestrator]   • Query: '{query[:80]}...'")
        logger.info(f"[Orchestrator]   • Accessible documents: {len(accessible_doc_ids)}")
        logger.info(f"[Orchestrator]   • LLM client: {'available' if self.llm_client else 'not available'}")
        logger.info(f"[Orchestrator] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # Execute LLM-driven dynamic SQL query
        # Field mappings are inferred dynamically from the data structure
        result = self.sql_executor.execute_dynamic_sql_query(
            accessible_doc_ids=accessible_doc_ids,
            query=query,
            llm_client=self.llm_client
        )

        logger.info(f"[Orchestrator] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[Orchestrator] ANALYTICS QUERY RESULT:")
        logger.info(f"[Orchestrator]   • Data rows: {len(result.get('data', []))}")
        logger.info(f"[Orchestrator]   • Summary keys: {list(result.get('summary', {}).keys())}")
        logger.info(f"[Orchestrator]   • Has formatted_report: {'formatted_report' in result.get('summary', {})}")
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

        Prioritizes LLM-generated formatted_report when available.

        Args:
            query: Original user query
            analytics_result: Results from SQL executor
            classification: Intent classification

        Returns:
            Formatted response string
        """
        summary = analytics_result.get('summary', {})

        # Check for errors
        if 'error' in summary:
            return f"I encountered an error while analyzing your data: {summary['error']}"

        # Prioritize LLM-generated formatted_report
        formatted_report = summary.get('formatted_report', '')
        if formatted_report and formatted_report.strip():
            logger.info(f"[Orchestrator] Using LLM-generated formatted_report ({len(formatted_report)} chars)")
            return formatted_report

        # No data case
        data = analytics_result.get('data', [])
        if not data and summary.get('total_records', 0) == 0:
            return self._format_no_data_response(classification)

        # Fallback: basic summary (should rarely happen with LLM-only path)
        total_amount = summary.get('total_amount', summary.get('grand_total', 0))
        total_records = summary.get('total_records', len(data))
        return f"Found {total_records} records with a total of ${total_amount:,.2f}."

    def _format_dynamic_sql_response(
        self,
        data: list,
        summary: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Format response from dynamic SQL query results.

        This method dynamically formats all summary fields without hardcoding
        specific field names like 'year' or 'category'.

        Priority:
        1. If formatted_report exists (from LLM summary generation), use it directly
        2. Otherwise, build response from hierarchical_data and summary_by_* fields
        """
        # If LLM-generated formatted_report exists, use it directly
        # This handles cases like MIN/MAX queries where hierarchical_data may be empty
        # but the LLM has generated a proper formatted response
        formatted_report = summary.get('formatted_report', '')
        if formatted_report and formatted_report.strip():
            logger.info(f"[Orchestrator] Using LLM-generated formatted_report ({len(formatted_report)} chars)")
            return formatted_report

        response_parts = []

        # Title
        title = summary.get('report_title', 'Analytics Report')
        response_parts.append(f"## {title}\n")

        # Check for hierarchical data (primary -> secondary grouping)
        hierarchical_data = summary.get('hierarchical_data', [])
        if hierarchical_data:
            # Detect the secondary dimension label from sub_groups
            secondary_dimension = "Item"  # Default
            if hierarchical_data and hierarchical_data[0].get('sub_groups'):
                # Try to detect from data columns
                secondary_dimension = self._detect_secondary_dimension(data, summary)

            for group in hierarchical_data:
                group_name = group.get('group_name', 'Unknown')
                group_total = group.get('group_total', 0) or 0

                response_parts.append(f"\n### {group_name}")
                response_parts.append(f"**Total**: ${group_total:,.2f}\n")

                sub_groups = group.get('sub_groups', [])
                if sub_groups:
                    # Dynamic table header based on detected dimension
                    response_parts.append(f"| {secondary_dimension} | Total | Count |")
                    response_parts.append("|" + "-" * (len(secondary_dimension) + 2) + "|-------|-------|")
                    for sub in sub_groups:
                        name = sub.get('name', 'Unknown')
                        total = sub.get('total', 0) or 0
                        count = sub.get('count', 0)
                        response_parts.append(f"| {name} | ${total:,.2f} | {count} |")

        # Dynamically detect and format all summary_by_* fields
        summary_keys = [k for k in summary.keys() if k.startswith('summary_by_')]
        for summary_key in sorted(summary_keys):
            summary_data = summary.get(summary_key, {})
            if not summary_data:
                continue

            # Extract dimension name and format it
            dimension_name = summary_key.replace('summary_by_', '')
            dimension_label = self._format_dimension_label(dimension_name)

            # Check if this dimension is already shown as primary in hierarchical view
            if hierarchical_data:
                first_group = hierarchical_data[0].get('group_name', '')
                if self._is_same_dimension(first_group, dimension_name):
                    continue  # Skip, already shown in hierarchical view

            response_parts.append(f"\n### Summary by {dimension_label}")
            for key, total in sorted(summary_data.items(), key=lambda x: str(x[0])):
                if total is not None and key is not None:
                    response_parts.append(f"- **{key}**: ${float(total):,.2f}")

        # Grand total
        grand_total = summary.get('grand_total', 0) or 0
        total_records = summary.get('total_records', 0)
        response_parts.append(f"\n---\n**Grand Total**: ${grand_total:,.2f} ({total_records} records)")

        # Add generated SQL info (for debugging/transparency)
        explanation = metadata.get('explanation', '')
        if explanation:
            response_parts.append(f"\n*Query: {explanation}*")

        return "\n".join(response_parts)

    def _detect_secondary_dimension(self, data: list, summary: Dict[str, Any]) -> str:
        """
        Detect the secondary dimension label for hierarchical data tables.

        The secondary dimension is the one used in sub_groups, NOT the primary grouping.
        We need to find which dimension is NOT the primary (hierarchical group names).

        Args:
            data: The raw data rows
            summary: The summary dictionary containing summary_by_* keys

        Returns:
            A human-readable label for the secondary dimension
        """
        # First, identify what the PRIMARY dimension is (from hierarchical_data group names)
        hierarchical_data = summary.get('hierarchical_data', [])
        primary_dimension = None
        if hierarchical_data:
            first_group_name = hierarchical_data[0].get('group_name', '')
            # Detect primary dimension type
            if self._is_same_dimension(first_group_name, 'year'):
                primary_dimension = 'year'
            elif self._is_same_dimension(first_group_name, 'month'):
                primary_dimension = 'month'
            elif self._is_same_dimension(first_group_name, 'quarter'):
                primary_dimension = 'quarter'
            else:
                # Try to match against summary_by_* keys
                for key in summary.keys():
                    if key.startswith('summary_by_'):
                        dim = key.replace('summary_by_', '')
                        dim_data = summary.get(key, {})
                        if first_group_name in dim_data:
                            primary_dimension = dim
                            break

        # Now find the SECONDARY dimension (not the primary)
        # Look through summary_by_* keys and find one that's not the primary
        for key in sorted(summary.keys()):
            if key.startswith('summary_by_'):
                dimension = key.replace('summary_by_', '')
                # Skip if this is the primary dimension
                if dimension == primary_dimension:
                    continue
                # Skip time-based dimensions if primary is also time-based
                if dimension in ['year', 'month', 'quarter'] and primary_dimension in ['year', 'month', 'quarter']:
                    continue
                return self._format_dimension_label(dimension)

        # Fallback: Try to infer from data columns
        if data and len(data) > 0:
            first_row = data[0]
            # List of potential secondary columns (in order of preference)
            secondary_candidates = ['category', 'product', 'status', 'region', 'vendor', 'customer']
            for col in secondary_candidates:
                if col in first_row and col != primary_dimension:
                    return self._format_dimension_label(col)

        return "Item"

    def _format_standard_analytics_response(
        self,
        data: list,
        summary: Dict[str, Any],
        metadata: Dict[str, Any],
        classification: IntentClassification,
        query: str = ""
    ) -> str:
        """Format response from standard analytics query results."""
        response_parts = []

        # Detect if user wants detailed listing vs summary
        detail_keywords = ['details', 'detail', 'list', 'show all', 'all items', 'individual',
                           'each', 'breakdown', 'itemized', 'every', 'specific']
        query_lower = query.lower() if query else ""
        wants_details = any(keyword in query_lower for keyword in detail_keywords)

        # Header
        schema_types = metadata.get('schema_types', [])
        time_range = metadata.get('time_range', {})

        header_parts = []
        if schema_types:
            doc_types = ', '.join(schema_types)
            header_parts.append(f"Based on your {doc_types} documents")

        if time_range:
            start = time_range.get('start', '')
            end = time_range.get('end', '')
            if start and end:
                header_parts.append(f"from {start} to {end}")

        if header_parts:
            header_parts[0] = header_parts[0] + ":"
            header = " ".join(header_parts)
        else:
            header = "Here are the results:"

        response_parts.append(header)

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

        if summary_lines:
            response_parts.append("\n**Summary:**")
            response_parts.extend(summary_lines)

        # Data table (if grouped)
        if data and isinstance(data[0], dict) and 'group' in data[0]:
            # Grouped data - create table
            response_parts.append("\n| Group | Count | Total Amount |")
            response_parts.append("|-------|-------|--------------|")
            for row in data[:20]:  # Limit to 20 rows
                group = row.get('group', 'Unknown')
                count = row.get('count', 0)
                total = row.get('total_amount', 0) or 0
                response_parts.append(f"| {group} | {count} | ${total:,.2f} |")

            if len(data) > 20:
                response_parts.append(f"\n*...and {len(data) - 20} more groups*")

        # Individual records (if not grouped)
        elif data and not ('group' in data[0] if data else False):
            # Show all records if user asked for details, otherwise limit to 10
            max_records = len(data) if wants_details else 10
            show_all = wants_details or len(data) <= 10

            if show_all or len(data) <= 10:
                # Build table with all available columns from the data
                response_parts.append("\n**Details:**")
                response_parts.append("\n| Date | Description | Amount |")
                response_parts.append("|------|-------------|--------|")

                for row in data[:max_records]:
                    date = row.get('date', 'N/A')
                    # Try to find a description from various fields
                    description = (
                        row.get('product') or
                        row.get('entity_name') or
                        row.get('category') or
                        row.get('raw_item', {}).get('description') or
                        row.get('raw_item', {}).get('Description') or
                        row.get('raw_item', {}).get('item') or
                        row.get('raw_item', {}).get('Item') or
                        row.get('raw_item', {}).get('item_name') or
                        row.get('raw_item', {}).get('product_name') or
                        'N/A'
                    )
                    # Truncate long descriptions
                    if len(str(description)) > 40:
                        description = str(description)[:37] + "..."
                    amount = row.get('amount', 0) or 0
                    amount_str = f"${amount:,.2f}" if amount else "N/A"
                    response_parts.append(f"| {date} | {description} | {amount_str} |")

                if not wants_details and len(data) > max_records:
                    response_parts.append(f"\n*...showing first {max_records} of {len(data)} records. Ask for 'details' to see all.*")
            else:
                response_parts.append(f"\n*Found {len(data)} matching documents. Ask for 'details' or 'list' to see all items.*")

        return "\n".join(response_parts)

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

    def _generate_llm_formatted_report(
        self,
        query: str,
        data: List[Dict[str, Any]],
        summary: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate a formatted report using LLM from standard query results.

        This method converts the raw data to a format the LLM can understand
        and generates a natural language response that includes details when
        the user asks for them.

        Args:
            query: Original user query
            data: List of data rows from query results
            summary: Summary statistics

        Returns:
            Formatted report string, or None if generation fails
        """
        if not self.llm_client or not data:
            return None

        try:
            # Detect if user wants detailed listing vs summary
            detail_keywords = ['details', 'detail', 'list', 'show all', 'all items', 'individual',
                               'each', 'breakdown', 'itemized', 'every', 'specific']
            query_lower = query.lower()
            wants_details = any(keyword in query_lower for keyword in detail_keywords)

            # Build markdown table from data
            # Determine columns to show based on available data
            columns = []
            if data and len(data) > 0:
                sample = data[0]
                # Priority order for columns
                priority_cols = ['date', 'description', 'product', 'entity_name', 'category', 'amount', 'quantity']
                for col in priority_cols:
                    if col in sample or (sample.get('raw_item') and col in sample.get('raw_item', {})):
                        columns.append(col)

                # Also check raw_item for description
                if 'raw_item' in sample and isinstance(sample['raw_item'], dict):
                    raw_cols = sample['raw_item'].keys()
                    for rc in ['description', 'Description', 'item', 'Item']:
                        if rc in raw_cols and 'description' not in columns:
                            columns.append('description')
                            break

            if not columns:
                columns = ['date', 'description', 'amount']

            # Build data rows for markdown
            table_rows = []
            for row in data:
                row_data = {}
                for col in columns:
                    if col in row:
                        row_data[col] = row[col]
                    elif row.get('raw_item') and col in row.get('raw_item', {}):
                        row_data[col] = row['raw_item'][col]
                    elif row.get('raw_item'):
                        # Try case variations
                        raw_item = row['raw_item']
                        for key in raw_item:
                            if key.lower() == col.lower():
                                row_data[col] = raw_item[key]
                                break
                    if col not in row_data:
                        row_data[col] = 'N/A'
                table_rows.append(row_data)

            # Create markdown table
            header = "| " + " | ".join(col.replace('_', ' ').title() for col in columns) + " |"
            separator = "| " + " | ".join(["---"] * len(columns)) + " |"
            data_lines = [header, separator]

            for row in table_rows:
                values = []
                for col in columns:
                    val = row.get(col, 'N/A')
                    if col == 'amount' and isinstance(val, (int, float)):
                        values.append(f"${val:,.2f}")
                    elif val is None:
                        values.append('N/A')
                    else:
                        val_str = str(val)
                        if len(val_str) > 40:
                            val_str = val_str[:37] + "..."
                        values.append(val_str)
                data_lines.append("| " + " | ".join(values) + " |")

            results_markdown = "\n".join(data_lines)

            # Determine instruction based on user intent
            if wants_details:
                detail_instruction = "4. IMPORTANT: The user asked for details/list - you MUST include ALL individual items from the data table in your response. Show each item with its details (date, description, amount, etc.). Do NOT just show a summary total."
                report_type = "detailed"
            else:
                detail_instruction = "4. Be concise but complete - summarize the data appropriately"
                report_type = "summary"

            # Build prompt
            prompt = f"""Based on the user's question and the query results, write a clear {report_type} report.

## User's Question:
"{query}"

## Query Results ({len(data)} rows):

{results_markdown}

## Summary Statistics:
- Total Records: {summary.get('total_records', len(data))}
- Total Amount: ${summary.get('total_amount', 0):,.2f}

## Instructions:
1. Directly answer the user's question based on the data above
2. If they asked for minimum/maximum values, clearly identify them
3. Format amounts with $ and proper number formatting (e.g., $1,234.56)
{detail_instruction}
5. Use markdown formatting for readability (use tables for listing multiple items)
6. Always include a grand total at the end if showing amounts

Write the {report_type} report in markdown format:"""

            response = self.llm_client.generate(prompt)

            # Clean up the response - remove any code block markers
            formatted_report = response.strip()
            if formatted_report.startswith('```'):
                lines = formatted_report.split('\n')
                formatted_report = '\n'.join(
                    line for line in lines
                    if not line.startswith('```')
                )

            logger.info(f"[Orchestrator] Generated LLM formatted report ({len(formatted_report)} chars)")
            return formatted_report

        except Exception as e:
            logger.error(f"[Orchestrator] Failed to generate LLM formatted report: {e}")
            return None

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
        """
        Summarize analytics results for inclusion in LLM context.

        This creates a structured summary that the LLM should use for data statistics
        instead of trying to calculate from document chunks.

        Supports both standard and dynamic SQL result formats.
        """
        summary = analytics_result.get('summary', {})
        data = analytics_result.get('data', [])

        parts = []

        # Check for dynamic SQL results (has hierarchical_data or summary_by_year)
        if 'hierarchical_data' in summary or 'summary_by_year' in summary:
            return self._summarize_dynamic_sql_for_context(summary, data)

        # Standard format handling
        # Add summary stats
        if 'total_amount' in summary:
            parts.append(f"Total amount: ${summary['total_amount']:,.2f}")
        if 'grand_total' in summary:
            parts.append(f"Grand total: ${summary['grand_total']:,.2f}")
        if 'count' in summary:
            parts.append(f"Document count: {summary['count']}")
        if 'total_records' in summary:
            parts.append(f"Total records: {summary['total_records']}")
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

    def _summarize_dynamic_sql_for_context(self, summary: Dict[str, Any], data: list) -> str:
        """
        Summarize dynamic SQL results for LLM context.

        This method dynamically detects and formats all summary fields without
        hardcoding specific field names like 'year' or 'category'.

        It handles:
        - Grand totals and record counts
        - Hierarchical data (primary group -> sub-group breakdown)
        - Any summary_by_* fields (dynamically detected)

        Dimension hierarchy:
        - Primary dimension: First grouping level (e.g., Year, Category)
        - Secondary dimension: Sub-grouping level (e.g., Category under Year)
        - Additional dimensions: Other summary breakdowns

        IMPORTANT: Format is optimized for LLM to copy values directly.
        """
        parts = []

        # Grand total and record count - prominent header
        grand_total = summary.get('grand_total', 0) or 0
        total_records = summary.get('total_records', 0)
        parts.append(f"═══ GRAND TOTAL: ${grand_total:,.2f} ({total_records} records) ═══")

        # Detect dimension hierarchy from data structure
        hierarchical_data = summary.get('hierarchical_data', [])
        primary_dimension = None
        secondary_dimension = None

        if hierarchical_data:
            # Detect primary dimension from first group's name pattern
            first_group_name = hierarchical_data[0].get('group_name', '') if hierarchical_data else ''
            primary_dimension = self._detect_dimension_label(first_group_name, data)

            # Detect secondary dimension from sub_groups
            first_sub_groups = hierarchical_data[0].get('sub_groups', []) if hierarchical_data else []
            if first_sub_groups:
                secondary_dimension = self._detect_secondary_dimension(data, summary)

            # Format hierarchical breakdown with detected dimensions
            parts.append(f"\n══ BREAKDOWN BY {primary_dimension.upper()} ══")

            for group in hierarchical_data:
                group_name = group.get('group_name', 'Unknown')
                group_total = group.get('group_total', 0) or 0
                parts.append(f"\n▶ {group_name}: ${group_total:,.2f}")

                sub_groups = group.get('sub_groups', [])
                if sub_groups:
                    for sub in sub_groups:
                        name = sub.get('name', 'Unknown')
                        total = sub.get('total', 0) or 0
                        count = sub.get('count', 0)
                        # Format each sub-group clearly so LLM can copy
                        parts.append(f"    • {name}: ${total:,.2f} ({count} items)")

        # Dynamically detect and format all summary_by_* fields
        summary_keys = [k for k in summary.keys() if k.startswith('summary_by_')]

        # Track which dimensions we've already shown
        shown_dimensions = set()
        if primary_dimension:
            shown_dimensions.add(primary_dimension.lower())

        for summary_key in sorted(summary_keys):
            summary_data = summary.get(summary_key, {})
            if not summary_data:
                continue

            # Extract dimension name and format it
            dimension_name = summary_key.replace('summary_by_', '')
            dimension_label = self._format_dimension_label(dimension_name)

            # Check if this dimension is already shown in hierarchical view
            if hierarchical_data:
                first_group = hierarchical_data[0].get('group_name', '')
                if self._is_same_dimension(first_group, dimension_name):
                    continue  # Skip primary dimension, already shown hierarchically

            # Build context description dynamically
            # If we have a primary dimension, describe this as aggregated across it
            if primary_dimension and dimension_label.lower() != primary_dimension.lower():
                context_desc = f"(AGGREGATED ACROSS ALL {primary_dimension.upper()}S)"
            else:
                context_desc = "(TOTAL)"

            parts.append(f"\n══ TOTAL BY {dimension_label.upper()} {context_desc} ══")
            for key, total in sorted(summary_data.items(), key=lambda x: str(x[0])):
                if total is not None and key is not None:
                    # Make totals very explicit for LLM to copy
                    parts.append(f"    ★ {key}: ${float(total):,.2f}")

        return "\n".join(parts)

    def _detect_dimension_label(self, group_name: str, data: list) -> str:
        """
        Detect the dimension label from the group name or data structure.

        Args:
            group_name: The name of a group (e.g., "2023", "Electronics")
            data: The raw data rows

        Returns:
            A human-readable dimension label (e.g., "Year", "Category")
        """
        # Check if it looks like a year (4-digit number between 1900-2100)
        try:
            year_val = int(str(group_name))
            if 1900 <= year_val <= 2100:
                return "Year"
        except (ValueError, TypeError):
            pass

        # Check if it looks like a month (YYYY-MM format)
        if isinstance(group_name, str) and len(group_name) == 7 and '-' in group_name:
            return "Month"

        # Check if it looks like a quarter (e.g., "2023-Q1")
        if isinstance(group_name, str) and '-Q' in group_name:
            return "Quarter"

        # Check data columns to infer dimension
        if data and len(data) > 0:
            first_row = data[0]
            if 'year' in first_row:
                return "Year"
            if 'month' in first_row:
                return "Month"
            if 'quarter' in first_row:
                return "Quarter"
            if 'category' in first_row:
                return "Category"
            if 'region' in first_row:
                return "Region"
            if 'product' in first_row:
                return "Product"

        # Default to "Group" if we can't determine
        return "Group"

    def _format_dimension_label(self, dimension_name: str) -> str:
        """
        Format a dimension name into a human-readable label.

        Args:
            dimension_name: Raw dimension name (e.g., "year", "category", "product_type")

        Returns:
            Formatted label (e.g., "Year", "Category", "Product Type")
        """
        # Replace underscores with spaces and title case
        return dimension_name.replace('_', ' ').title()

    def _is_same_dimension(self, group_name: str, dimension_name: str) -> bool:
        """
        Check if a group name belongs to a specific dimension.

        Args:
            group_name: The group name (e.g., "2023", "Electronics")
            dimension_name: The dimension to check (e.g., "year", "category")

        Returns:
            True if the group_name appears to be from this dimension
        """
        dimension_lower = dimension_name.lower()

        # Check for year dimension
        if dimension_lower == 'year':
            try:
                year_val = int(str(group_name))
                return 1900 <= year_val <= 2100
            except (ValueError, TypeError):
                return False

        # Check for month dimension
        if dimension_lower == 'month':
            return isinstance(group_name, str) and len(group_name) == 7 and '-' in group_name

        # Check for quarter dimension
        if dimension_lower == 'quarter':
            return isinstance(group_name, str) and '-Q' in group_name

        # For other dimensions, we can't reliably determine from name alone
        return False


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
