"""
LLM-based Schema Analyzer for Dynamic Field Mapping

This module provides intelligent analysis of data schemas using LLM to:
1. Analyze column headers and sample data to infer semantic types
2. Generate field mappings for data extraction
3. Analyze user queries to determine grouping and aggregation fields
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class FieldMapping:
    """Represents a field's semantic mapping."""
    original_name: str
    semantic_type: str  # date, amount, quantity, category, entity, product, region, status, identifier, etc.
    data_type: str  # datetime, number, string
    aggregation: Optional[str]  # sum, avg, count, group_by, None
    description: str
    aliases: List[str]  # Alternative names this field might be called


@dataclass
class QueryAnalysis:
    """Represents analyzed query intent for data operations."""
    primary_metric_field: Optional[str]  # Field to aggregate (e.g., "Total Sales")
    aggregation_type: str  # sum, avg, count, min, max
    group_by_fields: List[str]  # Fields to group by
    filter_conditions: Dict[str, Any]  # Filters to apply
    time_grouping: Optional[str]  # yearly, quarterly, monthly, daily
    sort_order: Optional[str]  # asc, desc
    explanation: str  # Human-readable explanation of the query


class LLMSchemaAnalyzer:
    """
    Uses LLM to analyze data schemas and generate intelligent field mappings.
    """

    # Prompt templates
    SCHEMA_ANALYSIS_PROMPT = """Analyze the following spreadsheet/table data and classify each column.

Column Headers: {headers}

Sample Data (first 5 rows):
{sample_data}

For each column, determine:
1. semantic_type: One of [date, amount, quantity, category, status, entity, product, region, identifier, method, person, unknown]
   - date: Date/time fields (purchase date, created at, timestamp)
   - amount: Monetary values that should be summed (total sales, revenue, cost, price, amount)
   - quantity: Numeric counts that can be summed (qty, units, count)
   - category: Classification fields for grouping (category, type, class)
   - status: State/status fields (status, state, condition)
   - entity: Customer/vendor/company names
   - product: Product/item names or descriptions
   - region: Geographic fields (region, area, country, city)
   - identifier: Unique IDs (order id, invoice number)
   - method: Payment/shipping methods
   - person: Sales rep, manager, assignee names
   - unknown: Cannot determine

2. data_type: One of [datetime, number, string]

3. aggregation: One of [sum, avg, count, group_by, null]
   - sum: For amounts and quantities that should be totaled
   - avg: For values where average makes sense
   - count: For counting occurrences
   - group_by: For categorical fields used in grouping
   - null: For identifiers or fields not typically aggregated

4. description: Brief description of what the field contains

5. aliases: List of alternative names this field might be called in queries

Return ONLY valid JSON in this exact format:
{{
  "field_mappings": {{
    "Column Name": {{
      "semantic_type": "type",
      "data_type": "type",
      "aggregation": "type or null",
      "description": "description",
      "aliases": ["alias1", "alias2"]
    }}
  }},
  "primary_amount_field": "field name for main monetary aggregation",
  "primary_date_field": "field name for time-based operations",
  "recommended_groupings": ["field1", "field2"]
}}"""

    QUERY_ANALYSIS_PROMPT = """Analyze this user query about tabular data and determine how to process it.

User Query: "{query}"

Available Fields and Their Types:
{field_info}

Determine:
1. primary_metric_field: Which field should be aggregated (usually an amount/quantity field)
2. aggregation_type: How to aggregate (sum, avg, count, min, max)
3. group_by_fields: Which fields to group by (can be multiple)
4. filter_conditions: Any filters to apply (field: value or field: {{min: x, max: y}})
5. time_grouping: If grouping by time, what granularity (yearly, quarterly, monthly, daily, or null)
6. sort_order: How to sort results (asc, desc, or null)

Return ONLY valid JSON in this exact format:
{{
  "primary_metric_field": "field_name or null",
  "aggregation_type": "sum",
  "group_by_fields": ["field1", "field2"],
  "filter_conditions": {{}},
  "time_grouping": "yearly or null",
  "sort_order": "desc or null",
  "explanation": "Human readable explanation of what this query does"
}}"""

    def __init__(self, llm_client=None):
        """
        Initialize the schema analyzer.

        Args:
            llm_client: LLM client for generating analysis. If None, falls back to heuristics.
        """
        self.llm_client = llm_client

    def analyze_schema(
        self,
        headers: List[str],
        sample_data: List[Dict[str, Any]],
        use_llm: bool = True
    ) -> Dict[str, FieldMapping]:
        """
        Analyze column headers and sample data to generate field mappings.

        Args:
            headers: List of column header names
            sample_data: List of sample row dictionaries
            use_llm: Whether to use LLM for analysis (falls back to heuristics if False or LLM unavailable)

        Returns:
            Dictionary mapping column names to FieldMapping objects
        """
        if use_llm and self.llm_client:
            try:
                return self._analyze_schema_with_llm(headers, sample_data)
            except Exception as e:
                logger.warning(f"LLM schema analysis failed, falling back to heuristics: {e}")

        return self._analyze_schema_with_heuristics(headers, sample_data)

    def _analyze_schema_with_llm(
        self,
        headers: List[str],
        sample_data: List[Dict[str, Any]]
    ) -> Dict[str, FieldMapping]:
        """Use LLM to analyze schema."""
        # Format sample data for prompt
        sample_rows = []
        for i, row in enumerate(sample_data[:5]):  # Limit to 5 rows
            row_str = " | ".join(f"{h}: {row.get(h, 'N/A')}" for h in headers)
            sample_rows.append(f"Row {i+1}: {row_str}")

        prompt = self.SCHEMA_ANALYSIS_PROMPT.format(
            headers=json.dumps(headers),
            sample_data="\n".join(sample_rows)
        )

        logger.info(f"[LLM Schema Analyzer] Analyzing {len(headers)} columns with LLM")

        response = self.llm_client.generate(prompt)

        # Parse LLM response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            result = json.loads(json_str.strip())
            field_mappings = result.get("field_mappings", {})

            # Convert to FieldMapping objects
            mappings = {}
            for col_name, mapping_data in field_mappings.items():
                mappings[col_name] = FieldMapping(
                    original_name=col_name,
                    semantic_type=mapping_data.get("semantic_type", "unknown"),
                    data_type=mapping_data.get("data_type", "string"),
                    aggregation=mapping_data.get("aggregation"),
                    description=mapping_data.get("description", ""),
                    aliases=mapping_data.get("aliases", [])
                )

            logger.info(f"[LLM Schema Analyzer] Generated mappings for {len(mappings)} columns")
            return mappings

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise

    def _analyze_schema_with_heuristics(
        self,
        headers: List[str],
        sample_data: List[Dict[str, Any]]
    ) -> Dict[str, FieldMapping]:
        """Fallback heuristic-based schema analysis, if llm sevice is unavailable."""
        import re

        # Semantic patterns (ordered by priority)
        semantic_patterns = [
            ("date", ["date", "time", "created", "updated", "timestamp", "day", "when", "period"], "datetime", None),
            ("person", ["sales rep", "sales_rep", "representative", "agent", "salesperson", "manager", "assignee", "owner"], "string", "group_by"),
            ("method", ["payment method", "payment_method", "pay method", "shipping method", "channel"], "string", "group_by"),
            ("amount", ["total sales", "total_sales", "total amount", "total_amount", "grand total", "amount", "revenue", "cost", "fee", "tax", "subtotal"], "number", "sum"),
            ("unit_price", ["unit price", "unit_price", "unit cost", "price each", "rate"], "number", None),
            ("quantity", ["quantity", "qty", "count", "units", "items", "number of"], "number", "sum"),
            ("category", ["category", "type", "class", "group", "classification", "segment"], "string", "group_by"),
            ("status", ["status", "state", "condition", "stage", "phase"], "string", "group_by"),
            ("entity", ["customer name", "customer_name", "vendor name", "vendor_name", "supplier", "client", "company", "merchant"], "string", "group_by"),
            ("product", ["product", "item", "sku", "goods", "service"], "string", "group_by"),
            ("region", ["region", "area", "location", "zone", "territory", "country", "city", "state"], "string", "group_by"),
            ("identifier", ["order id", "order_id", "id", "number", "code", "reference", "invoice", "receipt"], "string", None),
        ]

        mappings = {}

        for header in headers:
            header_lower = header.lower().replace('_', ' ').replace('-', ' ')
            samples = [row.get(header) for row in sample_data if row.get(header) is not None][:10]

            # Try to match semantic patterns
            semantic_type = "unknown"
            data_type = "string"
            aggregation = None
            description = f"Column: {header}"

            for sem_type, keywords, dtype, agg in semantic_patterns:
                if any(kw in header_lower for kw in keywords):
                    semantic_type = sem_type
                    data_type = dtype
                    aggregation = agg
                    description = f"{sem_type.title()} field"
                    break

            # Infer from samples if not matched
            if semantic_type == "unknown" and samples:
                if all(isinstance(s, (int, float)) for s in samples):
                    semantic_type = "quantity"
                    data_type = "number"
                    aggregation = "sum"
                    description = "Numeric field"
                elif all(isinstance(s, str) and re.match(r'^\d{4}-\d{2}-\d{2}', s) for s in samples):
                    semantic_type = "date"
                    data_type = "datetime"
                    description = "Date field"
                else:
                    # Check cardinality for grouping potential
                    unique_ratio = len(set(str(s) for s in samples)) / len(samples) if samples else 1
                    if unique_ratio < 0.5:
                        semantic_type = "category"
                        aggregation = "group_by"
                        description = "Low cardinality field (good for grouping)"

            mappings[header] = FieldMapping(
                original_name=header,
                semantic_type=semantic_type,
                data_type=data_type,
                aggregation=aggregation,
                description=description,
                aliases=[]
            )

        return mappings

    def analyze_query(
        self,
        query: str,
        field_mappings: Dict[str, FieldMapping],
        use_llm: bool = True
    ) -> QueryAnalysis:
        """
        Analyze a user query to determine how to process it.

        Args:
            query: User's natural language query
            field_mappings: Available field mappings
            use_llm: Whether to use LLM for analysis

        Returns:
            QueryAnalysis object with processing instructions
        """
        if use_llm and self.llm_client:
            try:
                return self._analyze_query_with_llm(query, field_mappings)
            except Exception as e:
                logger.warning(f"LLM query analysis failed, falling back to heuristics: {e}")

        return self._analyze_query_with_heuristics(query, field_mappings)

    def _analyze_query_with_llm(
        self,
        query: str,
        field_mappings: Dict[str, FieldMapping]
    ) -> QueryAnalysis:
        """Use LLM to analyze query intent."""
        # Format field info for prompt
        field_info_lines = []
        for name, mapping in field_mappings.items():
            aliases_str = f" (aliases: {', '.join(mapping.aliases)})" if mapping.aliases else ""
            field_info_lines.append(
                f"- {name}: {mapping.semantic_type} ({mapping.data_type}), "
                f"aggregation: {mapping.aggregation}{aliases_str}"
            )

        prompt = self.QUERY_ANALYSIS_PROMPT.format(
            query=query,
            field_info="\n".join(field_info_lines)
        )

        logger.info(f"[LLM Query Analyzer] Analyzing query: {query[:100]}...")

        response = self.llm_client.generate(prompt)

        # Parse LLM response
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            result = json.loads(json_str.strip())

            return QueryAnalysis(
                primary_metric_field=result.get("primary_metric_field"),
                aggregation_type=result.get("aggregation_type", "sum"),
                group_by_fields=result.get("group_by_fields", []),
                filter_conditions=result.get("filter_conditions", {}),
                time_grouping=result.get("time_grouping"),
                sort_order=result.get("sort_order"),
                explanation=result.get("explanation", "")
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM query analysis response: {e}")
            raise

    def _analyze_query_with_heuristics(
        self,
        query: str,
        field_mappings: Dict[str, FieldMapping]
    ) -> QueryAnalysis:
        """Fallback heuristic-based query analysis."""
        query_lower = query.lower()

        # Find primary metric field (amount to aggregate)
        primary_metric = None
        for name, mapping in field_mappings.items():
            if mapping.semantic_type == "amount" and mapping.aggregation == "sum":
                primary_metric = name
                break

        # Determine aggregation type
        aggregation_type = "sum"
        if "average" in query_lower or "avg" in query_lower or "mean" in query_lower:
            aggregation_type = "avg"
        elif "count" in query_lower or "how many" in query_lower:
            aggregation_type = "count"
        elif "maximum" in query_lower or "max" in query_lower or "highest" in query_lower:
            aggregation_type = "max"
        elif "minimum" in query_lower or "min" in query_lower or "lowest" in query_lower:
            aggregation_type = "min"

        # Determine grouping fields
        group_by_fields = []

        # Check for time grouping
        time_grouping = None
        if any(x in query_lower for x in ['by year', 'yearly', 'per year', 'each year', 'and year', 'grouped by year']):
            time_grouping = "yearly"
        elif any(x in query_lower for x in ['by quarter', 'quarterly', 'per quarter', 'and quarter', 'grouped by quarter']):
            time_grouping = "quarterly"
        elif any(x in query_lower for x in ['by month', 'monthly', 'per month', 'and month', 'grouped by month']):
            time_grouping = "monthly"

        # Check for field-based grouping
        # Note: keywords include variations like "and category" to handle multi-grouping queries
        grouping_keywords = {
            "category": ["by category", "by categories", "per category", "each category", "and category", "grouped by category"],
            "region": ["by region", "by regions", "per region", "each region", "and region", "grouped by region"],
            "status": ["by status", "per status", "each status", "and status", "grouped by status"],
            "product": ["by product", "by products", "per product", "each product", "and product", "grouped by product"],
            "customer": ["by customer", "by customers", "per customer", "and customer", "grouped by customer"],
            "vendor": ["by vendor", "by vendors", "per vendor", "and vendor", "grouped by vendor"],
        }

        for field_type, keywords in grouping_keywords.items():
            if any(kw in query_lower for kw in keywords):
                # Find the actual field name for this type
                for name, mapping in field_mappings.items():
                    if mapping.semantic_type == field_type or field_type in name.lower():
                        group_by_fields.append(name)
                        break

        # Build explanation
        explanation_parts = []
        if aggregation_type:
            explanation_parts.append(f"{aggregation_type.upper()} of {primary_metric or 'values'}")
        if time_grouping:
            explanation_parts.append(f"grouped by {time_grouping}")
        if group_by_fields:
            explanation_parts.append(f"and by {', '.join(group_by_fields)}")

        return QueryAnalysis(
            primary_metric_field=primary_metric,
            aggregation_type=aggregation_type,
            group_by_fields=group_by_fields,
            filter_conditions={},
            time_grouping=time_grouping,
            sort_order="desc",
            explanation=" ".join(explanation_parts) if explanation_parts else "Aggregate data"
        )

    def to_field_mappings_dict(self, mappings: Dict[str, FieldMapping]) -> Dict[str, Dict[str, Any]]:
        """Convert FieldMapping objects to dictionary format for storage."""
        return {
            name: {
                "semantic_type": m.semantic_type,
                "data_type": m.data_type,
                "aggregation": m.aggregation,
                "description": m.description,
                "aliases": m.aliases,
                "original_name": m.original_name
            }
            for name, m in mappings.items()
        }


# Convenience function for use in extraction service
def analyze_spreadsheet_schema(
    headers: List[str],
    sample_data: List[Dict[str, Any]],
    llm_client=None
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze spreadsheet schema and return field mappings.

    Args:
        headers: Column headers
        sample_data: Sample row data
        llm_client: Optional LLM client for enhanced analysis

    Returns:
        Dictionary of field mappings suitable for storage
    """
    analyzer = LLMSchemaAnalyzer(llm_client)
    mappings = analyzer.analyze_schema(headers, sample_data, use_llm=bool(llm_client))
    return analyzer.to_field_mappings_dict(mappings)


# Convenience function for use in query executor
def analyze_user_query(
    query: str,
    field_mappings: Dict[str, Dict[str, Any]],
    llm_client=None
) -> Dict[str, Any]:
    """
    Analyze user query to determine processing approach.

    Args:
        query: User's natural language query
        field_mappings: Available field mappings (from stored schema)
        llm_client: Optional LLM client for enhanced analysis

    Returns:
        Dictionary with query analysis results
    """
    # Convert dict mappings to FieldMapping objects
    fm_objects = {
        name: FieldMapping(
            original_name=m.get("original_name", name),
            semantic_type=m.get("semantic_type", "unknown"),
            data_type=m.get("data_type", "string"),
            aggregation=m.get("aggregation"),
            description=m.get("description", ""),
            aliases=m.get("aliases", [])
        )
        for name, m in field_mappings.items()
    }

    analyzer = LLMSchemaAnalyzer(llm_client)
    analysis = analyzer.analyze_query(query, fm_objects, use_llm=bool(llm_client))

    return asdict(analysis)
