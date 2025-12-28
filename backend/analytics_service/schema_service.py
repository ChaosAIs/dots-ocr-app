"""
Schema Service for Data Schema Management

This module provides LLM-driven schema management including:
1. DataSchema CRUD operations
2. LLM-driven schema analysis and inference
3. Dynamic field mapping generation
4. Schema lookup with priority-based fallback
5. Multi-schema query support

Key Design Principles:
- LLM-driven analysis at each stage (no hardcoded logic)
- Smart prompts to handle generic business scenarios
- Dynamic schema inference for unknown document types
- Priority-based field mapping lookup
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import text

from db.models import DataSchema, DocumentData

logger = logging.getLogger(__name__)


# =============================================================================
# LLM PROMPT TEMPLATES - Smart prompts for generic business scenarios
# =============================================================================

SCHEMA_INFERENCE_PROMPT = """Analyze this document content and infer its complete data schema.

## Document Content (Sample):
{document_content}

## Task:
1. Identify ALL fields present in this document
2. For each field, determine:
   - semantic_type: What kind of data it represents (see types below)
   - data_type: The technical data type (string, number, datetime, boolean)
   - source: Where it comes from ('header' for document-level, 'line_item' for item-level)
   - aggregation: How it should be aggregated ('sum', 'avg', 'count', 'group_by', or null)
   - description: Brief description of the field

## Semantic Types (choose the most appropriate):
- date: Date/time fields (purchase date, invoice date, created at)
- amount: Monetary values (total, subtotal, price, cost, revenue)
- quantity: Numeric counts (qty, units, items)
- category: Classification fields (category, type, class, group)
- status: State/status fields (status, state, condition)
- entity: Organization names (customer, vendor, supplier, company)
- person: People names (sales rep, manager, cashier)
- product: Product/item names (product, item, description)
- region: Geographic fields (region, city, country, location)
- method: Method fields (payment method, shipping method)
- identifier: Unique IDs (order ID, invoice number, receipt number)
- currency: Currency codes (USD, EUR, CAD)
- percentage: Percentage values (tax rate, discount rate)
- unknown: Cannot determine

## Response Format (JSON only):
{{
    "inferred_schema_type": "best_guess_schema_type",
    "confidence": 0.0-1.0,
    "header_fields": {{
        "field_name": {{
            "semantic_type": "type",
            "data_type": "type",
            "source": "header",
            "aggregation": null,
            "description": "description"
        }}
    }},
    "line_item_fields": {{
        "field_name": {{
            "semantic_type": "type",
            "data_type": "type",
            "source": "line_item",
            "aggregation": "sum or group_by or null",
            "description": "description"
        }}
    }},
    "summary_fields": {{
        "field_name": {{
            "semantic_type": "type",
            "data_type": "type",
            "source": "summary",
            "aggregation": null,
            "description": "description"
        }}
    }},
    "recommended_groupings": ["field1", "field2"],
    "recommended_aggregations": ["field1", "field2"]
}}

Respond with JSON only:"""


EXTRACTION_PROMPT_GENERATION = """Generate an extraction prompt for this document type.

## Document Type: {schema_type}
## Description: {description}

## Sample Fields:
Header Fields: {header_fields}
Line Item Fields: {line_item_fields}
Summary Fields: {summary_fields}

## Task:
Generate a comprehensive extraction prompt that will instruct an LLM to:
1. Extract ALL the fields listed above from a document
2. Return data in a consistent JSON structure
3. Handle missing values appropriately (use null)
4. Format dates as YYYY-MM-DD
5. Ensure numbers are actual numbers, not strings
6. Only extract what is explicitly visible

## Response Format (JSON only):
{{
    "extraction_prompt": "The complete prompt text for extraction...",
    "json_template": {{
        "header_data": {{}},
        "line_items": [{{}}],
        "summary_data": {{}}
    }},
    "validation_rules": {{
        "required_fields": [],
        "date_format": "YYYY-MM-DD",
        "number_fields": []
    }}
}}

Respond with JSON only:"""


FIELD_MAPPING_ANALYSIS_PROMPT = """Analyze these field mappings from multiple documents and generate a unified schema.

## Document Schema Type: {schema_type}

## Field Mappings from Documents:
{field_mappings_list}

## Task:
1. Identify all unique fields across all documents
2. Resolve any naming inconsistencies (same field with different names)
3. Determine the canonical field name for each concept
4. Generate a unified field mapping that covers all variations

## Response Format (JSON only):
{{
    "unified_header_fields": {{
        "canonical_name": {{
            "semantic_type": "type",
            "data_type": "type",
            "source": "header",
            "aggregation": null,
            "aliases": ["variation1", "variation2"]
        }}
    }},
    "unified_line_item_fields": {{
        "canonical_name": {{
            "semantic_type": "type",
            "data_type": "type",
            "source": "line_item",
            "aggregation": "sum or group_by",
            "aliases": ["variation1", "variation2"]
        }}
    }},
    "field_name_mappings": {{
        "original_name": "canonical_name"
    }}
}}

Respond with JSON only:"""


REPORT_LAYOUT_PROMPT = """Design the optimal report layout for presenting this query result.

## User Query: {user_query}

## Data Schema:
{field_schema}

## Query Result Sample (first 5 rows):
{result_sample}

## Total Records: {total_records}

## Task:
Design a report layout that best presents this data to answer the user's query.
Consider:
1. What columns to show and their order
2. How to group/organize the data
3. What summary statistics to include
4. How to format the output (table, list, hierarchical)
5. What title and description to use

## Response Format (JSON only):
{{
    "report_title": "Clear title for the report",
    "report_description": "Brief description of what the report shows",
    "layout_type": "table" | "hierarchical" | "summary" | "detailed",
    "columns": [
        {{
            "field": "field_name",
            "header": "Display Header",
            "format": "currency" | "number" | "date" | "text" | "percentage",
            "width": "auto" | "narrow" | "wide",
            "alignment": "left" | "right" | "center"
        }}
    ],
    "grouping": {{
        "enabled": true | false,
        "group_by": ["field1", "field2"],
        "show_subtotals": true | false
    }},
    "summary": {{
        "show_grand_total": true | false,
        "total_fields": ["field1"],
        "count_label": "Total Records"
    }},
    "formatting": {{
        "currency_symbol": "$",
        "decimal_places": 2,
        "date_format": "YYYY-MM-DD"
    }}
}}

Respond with JSON only:"""


class SchemaService:
    """
    LLM-driven Schema Service for managing data schemas.

    Provides:
    - CRUD operations for DataSchema
    - LLM-driven schema inference
    - Priority-based field mapping lookup
    - Multi-schema query support
    """

    def __init__(self, db: Session, llm_client=None):
        """
        Initialize the schema service.

        Args:
            db: SQLAlchemy database session
            llm_client: LLM client for schema analysis
        """
        self.db = db
        self.llm_client = llm_client

    # =========================================================================
    # DataSchema CRUD Operations
    # =========================================================================

    def get_schema(self, schema_type: str) -> Optional[DataSchema]:
        """
        Get a DataSchema by schema_type.

        Args:
            schema_type: The schema type (e.g., 'invoice', 'receipt')

        Returns:
            DataSchema or None if not found
        """
        return self.db.query(DataSchema).filter(
            DataSchema.schema_type == schema_type,
            DataSchema.is_active == True
        ).first()

    def get_all_schemas(self, domain: Optional[str] = None) -> List[DataSchema]:
        """
        Get all active schemas, optionally filtered by domain.

        Args:
            domain: Optional domain filter (e.g., 'finance', 'logistics')

        Returns:
            List of DataSchema objects
        """
        query = self.db.query(DataSchema).filter(DataSchema.is_active == True)
        if domain:
            query = query.filter(DataSchema.domain == domain)
        return query.all()

    def create_schema(
        self,
        schema_type: str,
        domain: str,
        display_name: str,
        description: str,
        header_schema: Dict[str, Any],
        line_items_schema: Optional[Dict[str, Any]] = None,
        summary_schema: Optional[Dict[str, Any]] = None,
        field_mappings: Optional[Dict[str, Any]] = None,
        extraction_prompt: Optional[str] = None,
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> DataSchema:
        """
        Create a new DataSchema.

        Args:
            schema_type: Unique schema type identifier
            domain: Domain category (e.g., 'finance', 'logistics')
            display_name: Human-readable name
            description: Schema description
            header_schema: JSON schema for header fields
            line_items_schema: JSON schema for line items
            summary_schema: JSON schema for summary fields
            field_mappings: Field name to semantic type mappings
            extraction_prompt: LLM prompt for extraction
            validation_rules: Validation rules

        Returns:
            Created DataSchema
        """
        schema = DataSchema(
            schema_type=schema_type,
            domain=domain,
            display_name=display_name,
            description=description,
            header_schema=header_schema,
            line_items_schema=line_items_schema,
            summary_schema=summary_schema,
            field_mappings=field_mappings or {},
            extraction_prompt=extraction_prompt,
            validation_rules=validation_rules or {}
        )

        self.db.add(schema)
        self.db.commit()
        self.db.refresh(schema)

        logger.info(f"[Schema Service] Created schema: {schema_type}")
        return schema

    def update_schema(
        self,
        schema_type: str,
        updates: Dict[str, Any]
    ) -> Optional[DataSchema]:
        """
        Update an existing DataSchema.

        Args:
            schema_type: Schema type to update
            updates: Dictionary of fields to update

        Returns:
            Updated DataSchema or None if not found
        """
        schema = self.get_schema(schema_type)
        if not schema:
            return None

        for key, value in updates.items():
            if hasattr(schema, key):
                setattr(schema, key, value)

        self.db.commit()
        self.db.refresh(schema)

        logger.info(f"[Schema Service] Updated schema: {schema_type}")
        return schema

    # =========================================================================
    # LLM-Driven Schema Inference
    # =========================================================================

    def infer_schema_from_content(
        self,
        content: str,
        existing_schema_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to infer schema from document content.

        This is the core LLM-driven analysis that handles generic business scenarios.

        Args:
            content: Document content (markdown, text, etc.)
            existing_schema_type: Optional known schema type for context

        Returns:
            Inferred schema with field mappings
        """
        if not self.llm_client:
            logger.warning("[Schema Service] No LLM client, using heuristic inference")
            return self._infer_schema_heuristically(content)

        try:
            # Truncate content if too long
            truncated_content = content[:15000] if len(content) > 15000 else content

            prompt = SCHEMA_INFERENCE_PROMPT.format(
                document_content=truncated_content
            )

            logger.info("[Schema Service] Running LLM schema inference...")
            response = self.llm_client.generate(prompt)
            result = self._parse_json_response(response)

            if result:
                logger.info(f"[Schema Service] Inferred schema type: {result.get('inferred_schema_type')}, "
                           f"confidence: {result.get('confidence')}")
                return result

        except Exception as e:
            logger.error(f"[Schema Service] LLM schema inference failed: {e}")

        return self._infer_schema_heuristically(content)

    def generate_extraction_prompt(
        self,
        schema_type: str,
        header_fields: List[str],
        line_item_fields: List[str],
        summary_fields: List[str],
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Use LLM to generate an extraction prompt for a schema type.

        Args:
            schema_type: The schema type
            header_fields: List of header field names
            line_item_fields: List of line item field names
            summary_fields: List of summary field names
            description: Schema description

        Returns:
            Generated extraction prompt and related configs
        """
        if not self.llm_client:
            logger.warning("[Schema Service] No LLM client, using template prompt")
            return self._generate_template_prompt(
                schema_type, header_fields, line_item_fields, summary_fields
            )

        try:
            prompt = EXTRACTION_PROMPT_GENERATION.format(
                schema_type=schema_type,
                description=description or f"Documents of type {schema_type}",
                header_fields=json.dumps(header_fields),
                line_item_fields=json.dumps(line_item_fields),
                summary_fields=json.dumps(summary_fields)
            )

            response = self.llm_client.generate(prompt)
            result = self._parse_json_response(response)

            if result:
                logger.info(f"[Schema Service] Generated extraction prompt for {schema_type}")
                return result

        except Exception as e:
            logger.error(f"[Schema Service] LLM prompt generation failed: {e}")

        return self._generate_template_prompt(
            schema_type, header_fields, line_item_fields, summary_fields
        )

    def design_report_layout(
        self,
        user_query: str,
        field_schema: Dict[str, Any],
        result_sample: List[Dict[str, Any]],
        total_records: int
    ) -> Dict[str, Any]:
        """
        Use LLM to design optimal report layout for query results.

        This enables dynamic, intelligent report formatting based on:
        - What the user asked for
        - The actual data structure
        - The result contents

        Args:
            user_query: Original user query
            field_schema: Field mappings/schema
            result_sample: Sample of query results (first 5 rows)
            total_records: Total number of records

        Returns:
            Report layout configuration
        """
        if not self.llm_client:
            return self._default_report_layout(result_sample)

        try:
            prompt = REPORT_LAYOUT_PROMPT.format(
                user_query=user_query,
                field_schema=json.dumps(field_schema, indent=2),
                result_sample=json.dumps(result_sample[:5], indent=2),
                total_records=total_records
            )

            response = self.llm_client.generate(prompt)
            result = self._parse_json_response(response)

            if result:
                logger.info(f"[Schema Service] Designed report layout: {result.get('layout_type')}")
                return result

        except Exception as e:
            logger.error(f"[Schema Service] LLM report layout design failed: {e}")

        return self._default_report_layout(result_sample)

    # =========================================================================
    # Priority-Based Field Mapping Lookup
    # =========================================================================

    def get_field_mappings(
        self,
        document_ids: List[UUID],
        schema_type: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get field mappings using priority-based lookup:
        1. extraction_metadata from DocumentData (per-document cache)
        2. DataSchema table (formal schema)
        3. LLM-driven dynamic inference

        Args:
            document_ids: List of document IDs to get mappings for
            schema_type: Optional known schema type

        Returns:
            Unified field mappings with source information
        """
        field_mappings = {}

        # Priority 1: Check extraction_metadata in DocumentData
        logger.info("[Schema Service] Priority 1: Checking extraction_metadata...")
        cached_mappings = self._get_cached_field_mappings(document_ids)
        if cached_mappings:
            logger.info(f"[Schema Service] Found cached field mappings with {len(cached_mappings)} fields")
            field_mappings.update(cached_mappings)
            return field_mappings

        # Priority 2: Check DataSchema table
        logger.info("[Schema Service] Priority 2: Checking DataSchema table...")
        if schema_type:
            schema = self.get_schema(schema_type)
            if schema and schema.field_mappings:
                logger.info(f"[Schema Service] Found schema field mappings for {schema_type}")
                field_mappings.update(schema.field_mappings)

                # Cache to extraction_metadata for future queries
                self._cache_field_mappings(document_ids, schema.field_mappings)
                return field_mappings

        # Priority 3: LLM-driven dynamic inference
        logger.info("[Schema Service] Priority 3: Dynamic inference...")
        inferred_mappings = self._infer_field_mappings_from_data(document_ids)
        if inferred_mappings:
            logger.info(f"[Schema Service] Inferred {len(inferred_mappings)} field mappings")
            field_mappings.update(inferred_mappings)

            # Cache to extraction_metadata
            self._cache_field_mappings(document_ids, inferred_mappings)

        return field_mappings

    def _get_cached_field_mappings(self, document_ids: List[UUID]) -> Optional[Dict[str, Any]]:
        """Get field mappings from extraction_metadata cache."""
        try:
            doc_ids_str = ", ".join(f"'{str(doc_id)}'" for doc_id in document_ids)

            result = self.db.execute(text(f"""
                SELECT extraction_metadata->'field_mappings' as field_mappings
                FROM documents_data
                WHERE document_id IN ({doc_ids_str})
                AND extraction_metadata->'field_mappings' IS NOT NULL
                AND jsonb_typeof(extraction_metadata->'field_mappings') = 'object'
                LIMIT 1
            """))

            row = result.fetchone()
            if row and row[0] and isinstance(row[0], dict):
                return dict(row[0])
            return None

        except Exception as e:
            logger.error(f"[Schema Service] Failed to get cached mappings: {e}")
            return None

    def _cache_field_mappings(
        self,
        document_ids: List[UUID],
        field_mappings: Dict[str, Any]
    ):
        """Cache field mappings to extraction_metadata."""
        try:
            for doc_id in document_ids:
                doc_data = self.db.query(DocumentData).filter(
                    DocumentData.document_id == doc_id
                ).first()

                if doc_data:
                    extraction_metadata = doc_data.extraction_metadata or {}
                    extraction_metadata['field_mappings'] = field_mappings
                    extraction_metadata['cached_at'] = datetime.utcnow().isoformat()
                    doc_data.extraction_metadata = extraction_metadata

            self.db.commit()
            logger.info(f"[Schema Service] Cached field mappings for {len(document_ids)} documents")

        except Exception as e:
            logger.error(f"[Schema Service] Failed to cache field mappings: {e}")
            self.db.rollback()

    def _infer_field_mappings_from_data(
        self,
        document_ids: List[UUID]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Infer field mappings from actual document data using LLM.

        This is the LLM-driven fallback when no cached or formal schema exists.
        """
        try:
            doc_ids_str = ", ".join(f"'{str(doc_id)}'" for doc_id in document_ids)

            # Get sample data
            result = self.db.execute(text(f"""
                SELECT
                    schema_type,
                    header_data,
                    line_items->0 as sample_line_item,
                    summary_data
                FROM documents_data
                WHERE document_id IN ({doc_ids_str})
                AND line_items IS NOT NULL
                LIMIT 1
            """))

            row = result.fetchone()
            if not row:
                return {}

            schema_type = row[0]
            header_data = row[1] or {}
            sample_item = row[2] or {}
            summary_data = row[3] or {}

            # Use LLM if available
            if self.llm_client:
                content = json.dumps({
                    "header_data": header_data,
                    "sample_line_item": sample_item,
                    "summary_data": summary_data
                }, indent=2)

                inferred = self.infer_schema_from_content(content, schema_type)

                if inferred:
                    # Combine header and line_item fields
                    field_mappings = {}
                    field_mappings.update(inferred.get('header_fields', {}))
                    field_mappings.update(inferred.get('line_item_fields', {}))
                    field_mappings.update(inferred.get('summary_fields', {}))
                    return field_mappings

            # Fallback to heuristic inference
            return self._infer_fields_heuristically(header_data, sample_item, summary_data)

        except Exception as e:
            logger.error(f"[Schema Service] Field mapping inference failed: {e}")
            return {}

    def _infer_fields_heuristically(
        self,
        header_data: Dict[str, Any],
        sample_item: Dict[str, Any],
        summary_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Heuristic-based field inference when LLM is unavailable."""
        field_mappings = {}

        # Semantic patterns for field detection
        patterns = {
            'date': {
                'keywords': ['date', 'time', 'created', 'updated', 'timestamp'],
                'data_type': 'datetime',
                'aggregation': None
            },
            'amount': {
                'keywords': ['total', 'amount', 'price', 'cost', 'revenue', 'sales', 'subtotal', 'sum'],
                'data_type': 'number',
                'aggregation': 'sum'
            },
            'quantity': {
                'keywords': ['quantity', 'qty', 'count', 'units', 'items'],
                'data_type': 'number',
                'aggregation': 'sum'
            },
            'entity': {
                'keywords': ['customer', 'vendor', 'supplier', 'client', 'company', 'store', 'merchant'],
                'data_type': 'string',
                'aggregation': 'group_by'
            },
            'category': {
                'keywords': ['category', 'type', 'class', 'group', 'segment'],
                'data_type': 'string',
                'aggregation': 'group_by'
            },
            'product': {
                'keywords': ['product', 'item', 'description', 'name', 'sku'],
                'data_type': 'string',
                'aggregation': 'group_by'
            },
            'identifier': {
                'keywords': ['id', 'number', 'code', 'reference', 'invoice', 'receipt', 'order'],
                'data_type': 'string',
                'aggregation': None
            },
            'method': {
                'keywords': ['method', 'payment', 'shipping', 'channel'],
                'data_type': 'string',
                'aggregation': 'group_by'
            },
            'region': {
                'keywords': ['region', 'city', 'country', 'location', 'address', 'area'],
                'data_type': 'string',
                'aggregation': 'group_by'
            }
        }

        def classify_field(field_name: str, source: str) -> Dict[str, Any]:
            field_lower = field_name.lower().replace('_', ' ').replace('-', ' ')

            for sem_type, config in patterns.items():
                if any(kw in field_lower for kw in config['keywords']):
                    return {
                        'semantic_type': sem_type,
                        'data_type': config['data_type'],
                        'source': source,
                        'aggregation': config['aggregation'],
                        'original_name': field_name
                    }

            return {
                'semantic_type': 'unknown',
                'data_type': 'string',
                'source': source,
                'aggregation': None,
                'original_name': field_name
            }

        # Process header fields
        for field in header_data.keys():
            if field not in ['field_mappings']:  # Skip meta fields
                field_mappings[field] = classify_field(field, 'header')

        # Process line item fields
        for field in sample_item.keys():
            if field not in ['row_number']:
                field_mappings[field] = classify_field(field, 'line_item')

        # Process summary fields
        for field in summary_data.keys():
            if field not in field_mappings:
                field_mappings[field] = classify_field(field, 'summary')

        return field_mappings

    # =========================================================================
    # Multi-Schema Query Support
    # =========================================================================

    def group_documents_by_schema(
        self,
        document_ids: List[UUID]
    ) -> Dict[str, List[UUID]]:
        """
        Group documents by their schema type for independent processing.

        Args:
            document_ids: List of document IDs

        Returns:
            Dictionary mapping schema_type to list of document IDs
        """
        try:
            doc_ids_str = ", ".join(f"'{str(doc_id)}'" for doc_id in document_ids)

            result = self.db.execute(text(f"""
                SELECT schema_type, document_id
                FROM documents_data
                WHERE document_id IN ({doc_ids_str})
            """))

            groups = {}
            for row in result.fetchall():
                schema_type = row[0]
                doc_id = row[1]

                if schema_type not in groups:
                    groups[schema_type] = []
                groups[schema_type].append(doc_id)

            logger.info(f"[Schema Service] Grouped {len(document_ids)} documents into {len(groups)} schema types: {list(groups.keys())}")
            return groups

        except Exception as e:
            logger.error(f"[Schema Service] Failed to group documents: {e}")
            return {'unknown': document_ids}

    def get_schema_info_for_documents(
        self,
        document_ids: List[UUID]
    ) -> Dict[str, Any]:
        """
        Get schema information for a set of documents.

        Returns schema type, field mappings, and document counts.

        Args:
            document_ids: List of document IDs

        Returns:
            Schema information dictionary
        """
        groups = self.group_documents_by_schema(document_ids)

        schema_info = {
            'schema_types': list(groups.keys()),
            'document_counts': {k: len(v) for k, v in groups.items()},
            'schemas': {}
        }

        for schema_type, doc_ids in groups.items():
            # Get field mappings for this schema type
            field_mappings = self.get_field_mappings(doc_ids, schema_type)

            # Get formal schema if exists
            formal_schema = self.get_schema(schema_type)

            schema_info['schemas'][schema_type] = {
                'document_count': len(doc_ids),
                'document_ids': [str(d) for d in doc_ids],
                'field_mappings': field_mappings,
                'has_formal_schema': formal_schema is not None,
                'extraction_prompt': formal_schema.extraction_prompt if formal_schema else None
            }

        return schema_info

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response."""
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            json_str = json_str.strip()

            if not json_str.startswith('{'):
                start = json_str.find('{')
                end = json_str.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = json_str[start:end]

            return json.loads(json_str)

        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"[Schema Service] Failed to parse JSON: {e}")
            return None

    def _infer_schema_heuristically(self, content: str) -> Dict[str, Any]:
        """Heuristic schema inference when LLM unavailable."""
        # Basic pattern detection
        content_lower = content.lower()

        if any(kw in content_lower for kw in ['invoice', 'billing', 'due date']):
            schema_type = 'invoice'
        elif any(kw in content_lower for kw in ['receipt', 'transaction', 'store']):
            schema_type = 'receipt'
        elif any(kw in content_lower for kw in ['statement', 'account', 'balance']):
            schema_type = 'bank_statement'
        else:
            schema_type = 'spreadsheet'

        return {
            'inferred_schema_type': schema_type,
            'confidence': 0.5,
            'header_fields': {},
            'line_item_fields': {},
            'summary_fields': {},
            'recommended_groupings': [],
            'recommended_aggregations': []
        }

    def _generate_template_prompt(
        self,
        schema_type: str,
        header_fields: List[str],
        line_item_fields: List[str],
        summary_fields: List[str]
    ) -> Dict[str, Any]:
        """Generate a template extraction prompt."""
        header_template = {f: "string or null" for f in header_fields}
        line_item_template = {f: "string or number" for f in line_item_fields}
        summary_template = {f: "number or null" for f in summary_fields}

        prompt = f"""Extract structured data from this {schema_type} document.

Return a JSON object with:
{{
    "header_data": {json.dumps(header_template, indent=8)},
    "line_items": [
        {json.dumps(line_item_template, indent=12)}
    ],
    "summary_data": {json.dumps(summary_template, indent=8)}
}}

IMPORTANT:
- Use null for any field not explicitly visible.
- Ensure numbers are actual numbers, not strings.
- Dates should be in YYYY-MM-DD format."""

        return {
            'extraction_prompt': prompt,
            'json_template': {
                'header_data': header_template,
                'line_items': [line_item_template],
                'summary_data': summary_template
            },
            'validation_rules': {
                'required_fields': [],
                'date_format': 'YYYY-MM-DD',
                'number_fields': summary_fields
            }
        }

    def _default_report_layout(self, result_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate default report layout when LLM unavailable."""
        if not result_sample:
            return {
                'report_title': 'Query Results',
                'layout_type': 'table',
                'columns': []
            }

        columns = []
        for col in result_sample[0].keys():
            col_lower = col.lower()

            # Determine format
            if any(kw in col_lower for kw in ['amount', 'total', 'price', 'cost', 'sales']):
                format_type = 'currency'
                alignment = 'right'
            elif any(kw in col_lower for kw in ['date', 'time']):
                format_type = 'date'
                alignment = 'center'
            elif any(kw in col_lower for kw in ['count', 'quantity', 'qty']):
                format_type = 'number'
                alignment = 'right'
            else:
                format_type = 'text'
                alignment = 'left'

            columns.append({
                'field': col,
                'header': col.replace('_', ' ').title(),
                'format': format_type,
                'alignment': alignment
            })

        return {
            'report_title': 'Query Results',
            'report_description': f'{len(result_sample)} records found',
            'layout_type': 'table',
            'columns': columns,
            'grouping': {'enabled': False},
            'summary': {'show_grand_total': True},
            'formatting': {
                'currency_symbol': '$',
                'decimal_places': 2,
                'date_format': 'YYYY-MM-DD'
            }
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def get_schema_service(db: Session, llm_client=None) -> SchemaService:
    """Create a SchemaService instance."""
    return SchemaService(db, llm_client)


def get_field_mappings_for_query(
    db: Session,
    document_ids: List[UUID],
    llm_client=None
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to get field mappings for SQL query generation.

    Args:
        db: Database session
        document_ids: Document IDs to get mappings for
        llm_client: Optional LLM client

    Returns:
        Field mappings dictionary
    """
    service = SchemaService(db, llm_client)
    return service.get_field_mappings(document_ids)
