"""
Intent Classifier for Analytics Queries

LLM-driven intent classification that routes queries to the appropriate service:
- Document search (RAG/vector search)
- Data analytics (SQL queries on extracted structured data)
- Hybrid (both)
- General (conversational)
"""

import os
import json
import logging
import re
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timedelta
import calendar

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


# Minimal schema hints for fallback only
SCHEMA_HINTS = {
    "invoice": ["invoice", "invoices", "bill", "billing"],
    "receipt": ["receipt", "receipts", "meal", "meals", "food", "restaurant", "dining", "grocery", "purchase"],
    "bank_statement": ["bank", "statement", "transaction", "account", "balance"],
    "purchase_order": ["purchase order", "po", "ordering"],
    "expense_report": ["expense", "expenses", "reimbursement", "travel"],
    "shipping_manifest": ["shipping", "shipment", "delivery", "manifest"],
    "inventory_report": ["inventory", "stock", "warehouse"],
    "spreadsheet": ["spreadsheet", "excel", "csv", "table"],
}


class IntentClassifier:
    """
    LLM-driven intent classifier.

    Uses LLM to understand query semantics rather than keyword matching.
    Falls back to simple heuristics only when LLM is unavailable.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the intent classifier.

        Args:
            llm_client: Optional LLM client for classification
        """
        self.llm_client = llm_client
        self._cached_llm_client = None

    def _get_llm_client(self):
        """Lazily initialize LLM client if not provided."""
        if self.llm_client:
            return self.llm_client

        if self._cached_llm_client:
            return self._cached_llm_client

        try:
            from rag_service.llm_service import get_llm_service
            from langchain_core.messages import HumanMessage

            llm_service = get_llm_service()
            if not llm_service.is_available():
                logger.warning("[IntentClassifier] LLM service not available")
                return None

            # Use a fast model for classification
            chat_model = llm_service.get_query_model(
                temperature=0.1,
                num_ctx=2048,
                num_predict=512
            )

            class LLMClientWrapper:
                def __init__(self, model):
                    self.model = model

                def generate(self, prompt: str) -> str:
                    response = self.model.invoke([HumanMessage(content=prompt)])
                    return response.content

            self._cached_llm_client = LLMClientWrapper(chat_model)
            logger.info("[IntentClassifier] LLM client initialized")
            return self._cached_llm_client

        except Exception as e:
            logger.warning(f"[IntentClassifier] Failed to create LLM client: {e}")
            return None

    def classify(
        self,
        query: str,
        available_schemas: Optional[List[str]] = None,
        use_llm: Optional[bool] = None
    ) -> IntentClassification:
        """
        Classify a user query using LLM.

        Args:
            query: User's natural language query
            available_schemas: List of schema types available for analytics
            use_llm: Override default LLM usage (defaults to True)

        Returns:
            IntentClassification with intent and metadata
        """
        should_use_llm = use_llm if use_llm is not None else True
        llm_client = self._get_llm_client() if should_use_llm else None

        if llm_client:
            try:
                return self._llm_classify(query, available_schemas, llm_client)
            except Exception as e:
                logger.warning(f"[IntentClassifier] LLM classification failed: {e}, using fallback")

        # Fallback to simple heuristics
        return self._fallback_classify(query, available_schemas)

    def _llm_classify(
        self,
        query: str,
        available_schemas: Optional[List[str]],
        llm_client
    ) -> IntentClassification:
        """
        LLM-based intent classification.
        """
        schemas_str = ", ".join(available_schemas) if available_schemas else "invoice, receipt, bank_statement, expense_report, purchase_order, shipping_manifest, inventory_report, spreadsheet"

        prompt = f"""You are an intent classifier for a document management system with OCR-extracted structured data.

TASK: Analyze the user's query and determine how to route it.

USER QUERY: "{query}"

AVAILABLE DOCUMENT TYPES WITH EXTRACTED DATA: {schemas_str}

ROUTING OPTIONS:

1. DATA_ANALYTICS - Use when the query requires:
   - Numerical aggregations: sum, total, average, count, max, min, etc.
   - Comparisons between values or time periods
   - Statistical analysis or trends
   - Filtering and grouping data by fields
   - Questions like "how many", "how much", "what is the total", "compare", "top N", "highest/lowest"
   - ANY query asking about amounts, quantities, or numerical summaries

2. DOCUMENT_SEARCH - Use when the query requires:
   - Finding specific documents by content or keywords
   - Reading or summarizing document text
   - Searching for mentions of specific terms or topics
   - Understanding what documents say or contain
   - Questions like "find documents about", "what does X say", "show me documents mentioning"

3. HYBRID - Use when the query requires BOTH:
   - Finding documents AND performing calculations on them
   - Example: "Find all invoices over $1000 and calculate the total"

4. GENERAL - Use when:
   - The query is a greeting, help request, or general question
   - Not related to documents or data analysis

IMPORTANT CONSIDERATIONS:
- Abbreviations like "max", "min", "avg" mean maximum, minimum, average
- Questions about "amounts", "totals", "prices", "costs" usually need DATA_ANALYTICS
- "List the X" with filtering/aggregation criteria = DATA_ANALYTICS
- "List documents about X" or "show me documents" = DOCUMENT_SEARCH
- When in doubt between DOCUMENT_SEARCH and DATA_ANALYTICS, prefer DATA_ANALYTICS if the query mentions any numerical fields or aggregation concepts

Respond with ONLY valid JSON (no markdown, no code blocks, no explanation):
{{
    "intent": "DATA_ANALYTICS|DOCUMENT_SEARCH|HYBRID|GENERAL",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of why this intent was chosen",
    "requires_extracted_data": true/false,
    "suggested_schemas": ["schema1"],
    "detected_entities": ["entity names mentioned"],
    "detected_metrics": ["metric types like total_amount, quantity, tax, etc."]
}}"""

        response = llm_client.generate(prompt)

        # Parse JSON from response (handle potential markdown wrapping)
        json_str = self._extract_json(response)
        result = json.loads(json_str)

        # Extract time range using helper
        time_range = self._extract_time_range(query.lower())

        intent = QueryIntent(result["intent"].lower())

        return IntentClassification(
            intent=intent,
            confidence=float(result.get("confidence", 0.8)),
            reasoning=result.get("reasoning", "LLM classification"),
            requires_extracted_data=result.get("requires_extracted_data", intent in [QueryIntent.DATA_ANALYTICS, QueryIntent.HYBRID]),
            suggested_schemas=result.get("suggested_schemas", []),
            detected_entities=result.get("detected_entities", []),
            detected_metrics=result.get("detected_metrics", []),
            detected_time_range=time_range
        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response, handling markdown code blocks."""
        # Remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```"):
            # Remove opening ```json or ```
            text = re.sub(r'^```(?:json)?\s*\n?', '', text)
            # Remove closing ```
            text = re.sub(r'\n?```\s*$', '', text)

        # Find JSON object
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            return match.group(0)

        return text

    def _fallback_classify(
        self,
        query: str,
        available_schemas: Optional[List[str]] = None
    ) -> IntentClassification:
        """
        Simple fallback classification when LLM is unavailable.
        Uses minimal heuristics for basic routing.
        """
        query_lower = query.lower().strip()

        # Detect schemas
        suggested_schemas = []
        for schema, hints in SCHEMA_HINTS.items():
            if any(hint in query_lower for hint in hints):
                if available_schemas is None or schema in available_schemas:
                    suggested_schemas.append(schema)

        # Simple pattern matching for analytics signals
        analytics_patterns = [
            r'\b(total|sum|average|avg|count|max|min|maximum|minimum)\b',
            r'\b(how many|how much)\b',
            r'\b(compare|comparison|versus|vs)\b',
            r'\b(highest|lowest|top|bottom)\b',
            r'\b(by month|by year|monthly|yearly|quarterly)\b',
            r'\b(statistics|stats|aggregate)\b',
        ]

        search_patterns = [
            r'\b(find|search|look for|locate)\b',
            r'\b(summarize|explain|describe)\b',
            r'\b(what does|what is|tell me about)\b',
            r'\b(show me documents|find documents)\b',
        ]

        general_patterns = [
            r'\b(hello|hi|hey|help|thanks|thank you)\b',
            r'\b(how do i|what can you)\b',
        ]

        analytics_score = sum(1 for p in analytics_patterns if re.search(p, query_lower))
        search_score = sum(1 for p in search_patterns if re.search(p, query_lower))
        general_score = sum(1 for p in general_patterns if re.search(p, query_lower))

        # Determine intent
        if general_score > 0 and analytics_score == 0 and search_score == 0:
            intent = QueryIntent.GENERAL
            confidence = 0.7
            reasoning = "Query appears to be a general greeting or help request"
        elif analytics_score > 0 and search_score > 0:
            intent = QueryIntent.HYBRID
            confidence = 0.7
            reasoning = "Query contains both analytics and search patterns"
        elif analytics_score > search_score:
            intent = QueryIntent.DATA_ANALYTICS
            confidence = min(0.85, 0.6 + analytics_score * 0.1)
            reasoning = f"Query contains analytics patterns (score: {analytics_score})"
        elif search_score > 0:
            intent = QueryIntent.DOCUMENT_SEARCH
            confidence = min(0.85, 0.6 + search_score * 0.1)
            reasoning = f"Query contains search patterns (score: {search_score})"
        else:
            # Default to document search for ambiguous queries
            intent = QueryIntent.DOCUMENT_SEARCH
            confidence = 0.5
            reasoning = "No clear intent signals, defaulting to document search"

        # Extract time range
        time_range = self._extract_time_range(query_lower)

        return IntentClassification(
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            requires_extracted_data=intent in [QueryIntent.DATA_ANALYTICS, QueryIntent.HYBRID],
            suggested_schemas=suggested_schemas,
            detected_entities=[],
            detected_metrics=[],
            detected_time_range=time_range
        )

    def _extract_time_range(self, query_lower: str) -> Optional[Dict[str, str]]:
        """
        Extract time range from query.
        """
        now = datetime.now()
        current_year = now.year

        # Check for specific year mentions
        year_match = re.search(r'\b(20\d{2})\b', query_lower)
        mentioned_year = int(year_match.group(1)) if year_match else current_year

        # Month mappings
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
            month_pattern = rf'\b{month_name}\b'
            if re.search(month_pattern, query_lower):
                # Check for time context
                time_context_patterns = [
                    rf'\b(in|for|during|from|since|until|by|of)\s+{month_name}\b',
                    rf'\b{month_name}\s+(20\d{{2}})\b',
                    rf'\b{month_name}\s+(sales|report|data|summary|total|amount)\b',
                    rf'\b{month_name}\s+\d{{1,2}}',
                    rf'\d{{1,2}}\s+{month_name}',
                ]

                is_time_reference = any(re.search(pat, query_lower) for pat in time_context_patterns)

                if is_time_reference or len(query_lower.split()) <= 5:
                    last_day = calendar.monthrange(mentioned_year, month_num)[1]
                    return {
                        "start": f"{mentioned_year}-{month_num:02d}-01",
                        "end": f"{mentioned_year}-{month_num:02d}-{last_day:02d}"
                    }

        # If only year is mentioned
        if year_match and mentioned_year != current_year:
            return {"start": f"{mentioned_year}-01-01", "end": f"{mentioned_year}-12-31"}

        return None
