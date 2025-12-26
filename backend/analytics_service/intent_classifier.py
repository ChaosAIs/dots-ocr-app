"""
Intent Classifier for Analytics Queries

Classifies user queries into different intent categories
to route them appropriately.
"""

import os
import json
import logging
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Types of query intents."""
    DOCUMENT_SEARCH = "document_search"    # Find documents, summarize content
    DATA_ANALYTICS = "data_analytics"      # Aggregate data, generate reports
    HYBRID = "hybrid"                       # Both search and analytics
    GENERAL = "general"                     # General questions, help


class IntentClassification(BaseModel):
    """Result of intent classification."""
    intent: QueryIntent
    confidence: float
    reasoning: str
    requires_extracted_data: bool
    suggested_schemas: List[str] = []
    detected_entities: List[str] = []
    detected_metrics: List[str] = []
    detected_time_range: Optional[Dict[str, str]] = None


# Keywords and patterns for rule-based classification
ANALYTICS_KEYWORDS = [
    "total", "sum", "average", "avg", "count", "how many", "how much",
    "compare", "comparison", "versus", "vs", "trend", "growth",
    "monthly", "weekly", "yearly", "quarterly", "by month", "by year",
    "aggregate", "statistics", "stats", "report", "analysis",
    "spending", "revenue", "sales", "expenses", "profit", "loss",
    "top", "bottom", "highest", "lowest", "maximum", "minimum",
    "percentage", "percent", "%", "ratio", "rate",
]

SEARCH_KEYWORDS = [
    "find", "search", "look for", "show me", "where is", "locate",
    "what is", "what does", "explain", "describe", "summarize",
    "about", "regarding", "related to", "contains", "mentions",
    "document", "file", "contract", "agreement", "report",
]

HYBRID_INDICATORS = [
    "find and calculate", "search and sum", "list with totals",
    "show documents and", "find all and aggregate",
]

GENERAL_KEYWORDS = [
    "help", "how do i", "what can", "tutorial", "guide",
    "hello", "hi", "thanks", "thank you", "bye",
]

# Schema-related keywords
SCHEMA_KEYWORDS = {
    "invoice": ["invoice", "bill", "billing", "invoiced"],
    "receipt": ["receipt", "purchase", "bought", "paid", "spending", "spent", "meal", "meals", "food", "restaurant", "dining", "grocery", "groceries", "store", "shop"],
    "bank_statement": ["bank", "statement", "transaction", "account", "balance"],
    "purchase_order": ["purchase order", "po", "ordering"],
    "expense_report": ["expense", "reimbursement", "travel expense", "expenses"],
    "shipping_manifest": ["shipping", "shipment", "delivery", "manifest"],
    "inventory_report": ["inventory", "stock", "warehouse", "item count"],
}

# Metric keywords
METRIC_KEYWORDS = {
    "total_amount": ["total", "amount", "sum", "grand total"],
    "quantity": ["quantity", "qty", "count", "number of items"],
    "tax": ["tax", "vat", "gst", "sales tax"],
    "discount": ["discount", "reduction", "savings"],
    "subtotal": ["subtotal", "sub-total", "before tax"],
}


class IntentClassifier:
    """
    Classifies user queries into intent categories.

    Supports both rule-based classification (fast, no LLM)
    and LLM-based classification (more accurate, slower).
    """

    def __init__(self, llm_client=None):
        """
        Initialize the intent classifier.

        Args:
            llm_client: Optional LLM client for advanced classification
        """
        self.llm_client = llm_client
        self.use_llm = os.getenv("ANALYTICS_AUTO_CLASSIFY_INTENT", "true").lower() == "true"

    def classify(
        self,
        query: str,
        available_schemas: Optional[List[str]] = None,
        use_llm: Optional[bool] = None
    ) -> IntentClassification:
        """
        Classify a user query.

        Args:
            query: User's natural language query
            available_schemas: List of schema types available for analytics
            use_llm: Override default LLM usage setting

        Returns:
            IntentClassification with intent and metadata
        """
        query_lower = query.lower().strip()

        # Always do rule-based first for speed
        rule_result = self._rule_based_classify(query_lower, available_schemas)

        # If high confidence or LLM disabled, return rule-based result
        should_use_llm = use_llm if use_llm is not None else self.use_llm
        if rule_result.confidence >= 0.85 or not should_use_llm or not self.llm_client:
            return rule_result

        # Use LLM for ambiguous cases
        try:
            llm_result = self._llm_classify(query, available_schemas)
            return llm_result
        except Exception as e:
            logger.warning(f"LLM classification failed, using rule-based: {e}")
            return rule_result

    def _rule_based_classify(
        self,
        query_lower: str,
        available_schemas: Optional[List[str]] = None
    ) -> IntentClassification:
        """
        Rule-based intent classification.

        Args:
            query_lower: Lowercase query string
            available_schemas: Available schema types

        Returns:
            IntentClassification result
        """
        # Count keyword matches
        analytics_score = sum(1 for kw in ANALYTICS_KEYWORDS if kw in query_lower)
        search_score = sum(1 for kw in SEARCH_KEYWORDS if kw in query_lower)
        general_score = sum(1 for kw in GENERAL_KEYWORDS if kw in query_lower)

        # Check for hybrid indicators
        has_hybrid_indicator = any(ind in query_lower for ind in HYBRID_INDICATORS)

        # Detect mentioned schemas
        suggested_schemas = []
        for schema, keywords in SCHEMA_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                if available_schemas is None or schema in available_schemas:
                    suggested_schemas.append(schema)

        # Detect metrics
        detected_metrics = []
        for metric, keywords in METRIC_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                detected_metrics.append(metric)

        # Detect entities (simple pattern matching for company names)
        detected_entities = self._extract_entities(query_lower)

        # Detect time range
        detected_time_range = self._extract_time_range(query_lower)

        # Determine intent
        if general_score > 0 and analytics_score == 0 and search_score == 0:
            intent = QueryIntent.GENERAL
            confidence = min(0.9, 0.5 + general_score * 0.1)
            reasoning = "Query appears to be a general question or greeting"

        elif has_hybrid_indicator or (analytics_score > 0 and search_score > 0):
            intent = QueryIntent.HYBRID
            confidence = min(0.9, 0.6 + (analytics_score + search_score) * 0.05)
            reasoning = "Query involves both document search and data aggregation"

        elif analytics_score > search_score:
            intent = QueryIntent.DATA_ANALYTICS
            confidence = min(0.95, 0.5 + analytics_score * 0.1)
            reasoning = f"Query contains analytics keywords: {analytics_score} matches"

        elif search_score > 0:
            intent = QueryIntent.DOCUMENT_SEARCH
            confidence = min(0.95, 0.5 + search_score * 0.1)
            reasoning = f"Query contains search keywords: {search_score} matches"

        else:
            # Default to search if no clear signals
            intent = QueryIntent.DOCUMENT_SEARCH
            confidence = 0.5
            reasoning = "No clear intent signals, defaulting to document search"

        return IntentClassification(
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            requires_extracted_data=intent in [QueryIntent.DATA_ANALYTICS, QueryIntent.HYBRID],
            suggested_schemas=suggested_schemas,
            detected_entities=detected_entities,
            detected_metrics=detected_metrics,
            detected_time_range=detected_time_range
        )

    def _extract_entities(self, query_lower: str) -> List[str]:
        """
        Extract potential entity names from query.

        Args:
            query_lower: Lowercase query string

        Returns:
            List of detected entity names
        """
        entities = []

        # Look for patterns like "Company A", "for ABC Corp"
        import re

        # Pattern: "for/from/to [Entity Name]"
        patterns = [
            r'(?:for|from|to|with|by)\s+([A-Z][a-zA-Z\s&]+?)(?:\s+(?:and|vs|versus|in|during|between)|$|,)',
            r'company\s+([A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:invoice|receipt|order)',
        ]

        # Use original case query for entity extraction
        for pattern in patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            entities.extend([m.strip() for m in matches if len(m.strip()) > 2])

        return list(set(entities))[:5]  # Limit to 5 entities

    def _extract_time_range(self, query_lower: str) -> Optional[Dict[str, str]]:
        """
        Extract time range from query.

        Args:
            query_lower: Lowercase query string

        Returns:
            Dict with 'start' and 'end' dates, or None
        """
        import re
        from datetime import datetime, timedelta

        now = datetime.now()
        current_year = now.year

        # Check for specific year mentions
        year_match = re.search(r'\b(20\d{2})\b', query_lower)
        mentioned_year = int(year_match.group(1)) if year_match else current_year

        # Check for month patterns
        months = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2,
            'march': 3, 'mar': 3, 'april': 4, 'apr': 4,
            'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }

        # Check for quarter patterns
        if 'q1' in query_lower or 'first quarter' in query_lower:
            return {"start": f"{mentioned_year}-01-01", "end": f"{mentioned_year}-03-31"}
        elif 'q2' in query_lower or 'second quarter' in query_lower:
            return {"start": f"{mentioned_year}-04-01", "end": f"{mentioned_year}-06-30"}
        elif 'q3' in query_lower or 'third quarter' in query_lower:
            return {"start": f"{mentioned_year}-07-01", "end": f"{mentioned_year}-09-30"}
        elif 'q4' in query_lower or 'fourth quarter' in query_lower:
            return {"start": f"{mentioned_year}-10-01", "end": f"{mentioned_year}-12-31"}

        # Check for "last X months/days"
        last_match = re.search(r'last\s+(\d+)\s+(month|day|week|year)s?', query_lower)
        if last_match:
            count = int(last_match.group(1))
            unit = last_match.group(2)
            if unit == 'day':
                start = now - timedelta(days=count)
            elif unit == 'week':
                start = now - timedelta(weeks=count)
            elif unit == 'month':
                start = now - timedelta(days=count * 30)
            elif unit == 'year':
                start = now - timedelta(days=count * 365)
            return {"start": start.strftime("%Y-%m-%d"), "end": now.strftime("%Y-%m-%d")}

        # Check for "this month/year"
        if 'this month' in query_lower:
            return {
                "start": f"{now.year}-{now.month:02d}-01",
                "end": now.strftime("%Y-%m-%d")
            }
        elif 'this year' in query_lower:
            return {"start": f"{now.year}-01-01", "end": now.strftime("%Y-%m-%d")}

        # Check for specific month mentions
        for month_name, month_num in months.items():
            if month_name in query_lower:
                import calendar
                last_day = calendar.monthrange(mentioned_year, month_num)[1]
                return {
                    "start": f"{mentioned_year}-{month_num:02d}-01",
                    "end": f"{mentioned_year}-{month_num:02d}-{last_day:02d}"
                }

        # If only year is mentioned
        if year_match and mentioned_year != current_year:
            return {"start": f"{mentioned_year}-01-01", "end": f"{mentioned_year}-12-31"}

        return None

    def _llm_classify(
        self,
        query: str,
        available_schemas: Optional[List[str]] = None
    ) -> IntentClassification:
        """
        LLM-based intent classification for ambiguous queries.

        Args:
            query: User's query
            available_schemas: Available schema types

        Returns:
            IntentClassification result
        """
        if not self.llm_client:
            raise ValueError("LLM client not configured")

        prompt = f"""Analyze the user's query and classify their intent.

Query: {query}

Available document types with extracted data: {available_schemas or ['invoice', 'receipt', 'bank_statement', 'purchase_order', 'expense_report']}

Classification options:
1. DOCUMENT_SEARCH: User wants to find, read, or summarize document content
   Examples: "Find contracts mentioning liability", "Summarize the Q3 report"

2. DATA_ANALYTICS: User wants numerical analysis, aggregations, comparisons
   Examples: "Total sales by month", "Average invoice amount for Company A"
   Requires: extracted structured data

3. HYBRID: User needs both document search AND data aggregation
   Examples: "Find all invoices over $10k and calculate total"

4. GENERAL: General questions, help requests, or unclear intent
   Examples: "What can you help me with?", "How do I upload files?"

Respond ONLY with valid JSON (no markdown, no explanation):
{{
    "intent": "DOCUMENT_SEARCH|DATA_ANALYTICS|HYBRID|GENERAL",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "requires_extracted_data": true/false,
    "suggested_schemas": ["schema1", "schema2"],
    "detected_entities": ["Entity1", "Entity2"],
    "detected_metrics": ["metric1", "metric2"]
}}"""

        try:
            response = self.llm_client.generate(prompt)
            result = json.loads(response)

            return IntentClassification(
                intent=QueryIntent(result["intent"].lower()),
                confidence=float(result["confidence"]),
                reasoning=result["reasoning"],
                requires_extracted_data=result.get("requires_extracted_data", False),
                suggested_schemas=result.get("suggested_schemas", []),
                detected_entities=result.get("detected_entities", []),
                detected_metrics=result.get("detected_metrics", []),
                detected_time_range=self._extract_time_range(query.lower())
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            raise
