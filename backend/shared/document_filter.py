"""
Centralized document filtering logic for both agent and non-agent flows.

This module provides:
- Topic-based pre-filtering using ILIKE SQL queries
- Relevance scoring with entity matching
- Adaptive threshold filtering

Both agent flow (routing_tools.py) and non-agent flow (document_router.py)
should use these shared functions to ensure consistent document filtering.
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

# Scoring configuration constants
ENTITY_MATCH_BOOST = 0.3       # Boost for entity match
ENTITY_FUZZY_BOOST = 0.2       # Boost for fuzzy entity match
ENTITY_NO_MATCH_PENALTY = 0.4  # Multiply by this when no entity match
DOC_TYPE_BOOST = 0.25          # Boost for document type match
DOC_TYPE_MISMATCH_PENALTY = 0.5  # Penalty when document type doesn't match query hints
TABULAR_BOOST = 0.35           # Boost for tabular data on analytics queries
ADAPTIVE_THRESHOLD_RATIO = 0.5  # Documents must score >= 50% of top score

# Analytics keywords for query analysis
ANALYTICS_KEYWORDS = {
    'sum', 'total', 'count', 'average', 'avg', 'min', 'max',
    'how many', 'how much', 'calculate', 'aggregate', 'group by',
    'report', 'analysis', 'statistics', 'breakdown', 'list', 'show'
}

# Data-related keywords (include both singular and plural forms)
DATA_KEYWORDS = {
    'inventory', 'product', 'products', 'sales', 'quantity', 'amount', 'price',
    'cost', 'revenue', 'stock', 'order', 'orders', 'invoice', 'invoices',
    'receipt', 'receipts', 'meal', 'meals', 'food', 'expense', 'expenses'
}


def extract_query_hints(query: str) -> Dict[str, Any]:
    """
    Extract topics and document type hints from the query.

    Args:
        query: The user's natural language query

    Returns:
        Dict with 'topics' and 'document_type_hints' lists
    """
    query_lower = query.lower()

    # Document type patterns
    doc_type_patterns = {
        'receipt': ['receipt', 'receipts'],
        'invoice': ['invoice', 'invoices'],
        'spreadsheet': ['spreadsheet', 'excel', 'csv', 'sheet'],
        'document': ['document', 'documents', 'pdf'],
    }

    # Topic patterns (categories/subjects)
    topic_patterns = [
        'meal', 'food', 'grocery', 'office', 'supplies', 'equipment',
        'travel', 'hotel', 'flight', 'transport', 'subscription',
        'software', 'hardware', 'service', 'utilities'
    ]

    topics = []
    document_type_hints = []

    # Extract document types
    for doc_type, patterns in doc_type_patterns.items():
        if any(p in query_lower for p in patterns):
            document_type_hints.append(doc_type)

    # Extract topics
    for topic in topic_patterns:
        if topic in query_lower:
            topics.append(topic)

    return {
        'topics': topics,
        'document_type_hints': document_type_hints
    }


def topic_based_prefilter(
    topics: List[str],
    document_type_hints: List[str],
    accessible_document_ids: List[str],
    db_session_factory
) -> Tuple[List[str], List[str]]:
    """
    Pre-filter documents by topic/category matching using ILIKE.

    This handles queries like "meal receipts" where:
    - "meal" is a topic/category term to match against document metadata
    - "receipt" is a document type filter

    Args:
        topics: List of topic/category terms to match (e.g., ["meal", "food"])
        document_type_hints: List of document types to filter (e.g., ["receipt"])
        accessible_document_ids: List of accessible document IDs for access control
        db_session_factory: Function to get database session (e.g., get_db_session)

    Returns:
        Tuple of (matched_document_ids, matched_document_names)
    """
    if not topics and not document_type_hints:
        return [], []

    if not accessible_document_ids:
        return [], []

    logger.info(f"[DocumentFilter] Topic pre-filter: topics={topics}, doc_types={document_type_hints}")

    matched_ids = []
    matched_names = []

    try:
        with db_session_factory() as db:
            conditions = []
            params = {}

            # Convert list to PostgreSQL array format for document ID filter
            doc_ids_array = '{' + ','.join(accessible_document_ids) + '}' if accessible_document_ids else '{}'
            params["doc_ids"] = doc_ids_array

            # Topic matching conditions (OR between topics, AND with doc type)
            if topics:
                topic_conditions = []
                for i, topic in enumerate(topics):
                    param_name = f"topic_{i}"
                    params[param_name] = f"%{topic.lower()}%"
                    topic_conditions.append(f"""
                        (
                            LOWER(d.document_metadata::text) ILIKE :{param_name}
                            OR LOWER(COALESCE(dd.header_data->>'vendor_name', '')) ILIKE :{param_name}
                            OR LOWER(COALESCE(dd.header_data->>'store_name', '')) ILIKE :{param_name}
                            OR LOWER(COALESCE(dd.header_data->>'description', '')) ILIKE :{param_name}
                            OR LOWER(COALESCE(dd.header_data->>'category', '')) ILIKE :{param_name}
                            OR LOWER(COALESCE(dd.summary_data::text, '')) ILIKE :{param_name}
                            OR LOWER(COALESCE(d.original_filename, '')) ILIKE :{param_name}
                            OR EXISTS (
                                SELECT 1 FROM documents_data_line_items li
                                WHERE li.documents_data_id = dd.id
                                AND LOWER(COALESCE(li.data->>'description', '')) ILIKE :{param_name}
                            )
                        )
                    """)
                conditions.append(f"({' OR '.join(topic_conditions)})")

            # Document type filter (if specified) - exact match on schema_type
            if document_type_hints:
                type_conditions = []
                for j, doc_type in enumerate(document_type_hints):
                    param_name = f"doctype_{j}"
                    params[param_name] = doc_type.lower()
                    type_conditions.append(f"LOWER(dd.schema_type) = :{param_name}")
                conditions.append(f"({' OR '.join(type_conditions)})")

            # Build final SQL
            where_clause = " AND ".join(conditions) if conditions else "TRUE"

            sql = f"""
                SELECT DISTINCT
                    d.id::text as document_id,
                    COALESCE(dd.header_data->>'vendor_name', dd.header_data->>'store_name', d.original_filename) as source_display
                FROM documents d
                LEFT JOIN documents_data dd ON dd.document_id = d.id
                WHERE d.id = ANY(CAST(:doc_ids AS uuid[]))
                  AND d.deleted_at IS NULL
                  AND {where_clause}
            """

            result = db.execute(text(sql), params)

            for row in result.fetchall():
                doc_id = row[0]
                source_display = row[1] or "Unknown"
                matched_ids.append(doc_id)
                matched_names.append(source_display)

            logger.info(
                f"[DocumentFilter] Topic pre-filter found {len(matched_ids)} document(s) "
                f"matching topics={topics}, doc_types={document_type_hints}"
            )
            if matched_names:
                logger.info(f"[DocumentFilter] Filtered documents: {matched_names[:5]}{'...' if len(matched_names) > 5 else ''}")

    except Exception as e:
        logger.error(f"[DocumentFilter] Topic pre-filter error: {e}", exc_info=True)
        return [], []

    return matched_ids, matched_names


def is_analytics_query(query: str) -> bool:
    """Check if query is an analytics/aggregation query."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in ANALYTICS_KEYWORDS)


def extract_query_entities(query: str) -> List[str]:
    """
    Extract potential entity names from the query.

    Args:
        query: The user's query

    Returns:
        List of potential entity names
    """
    entities = []

    entity_indicators = [
        'from', 'by', 'for', 'vendor', 'supplier', 'company', 'customer',
        'client', 'merchant', 'store', 'seller'
    ]

    query_lower = query.lower()
    words = query.split()

    for i, word in enumerate(words):
        if word.lower() in ['the', 'a', 'an', 'and', 'or', 'for', 'from', 'by', 'to', 'in', 'on', 'at']:
            continue

        if i > 0 and words[i-1].lower() in entity_indicators:
            entity_parts = [word]
            for j in range(i + 1, min(i + 4, len(words))):
                if words[j][0].isupper() if words[j] else False:
                    entity_parts.append(words[j])
                else:
                    break
            if entity_parts:
                entities.append(" ".join(entity_parts).strip('.,!?'))

    return entities


def match_entity_to_document(
    query_entities: List[str],
    header_data: Dict[str, Any],
    doc_metadata: Dict[str, Any] = None
) -> Tuple[bool, float, str]:
    """
    Match query entities against document metadata.

    Args:
        query_entities: Entities extracted from query
        header_data: Document header data
        doc_metadata: Additional document metadata

    Returns:
        Tuple of (matched, score, matched_entity)
    """
    if not query_entities:
        return False, 0.0, ""

    # Build list of document entities to match against
    doc_entities = set()

    # From header_data
    if header_data:
        for field in ['vendor_name', 'customer_name', 'store_name', 'company_name', 'merchant_name']:
            value = header_data.get(field)
            if value and isinstance(value, str):
                doc_entities.add(value.lower().strip())

    # From doc_metadata
    if doc_metadata:
        for field in ['vendor_normalized', 'customer_normalized', 'entities']:
            value = doc_metadata.get(field)
            if isinstance(value, str):
                doc_entities.add(value.lower().strip())
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, str):
                        doc_entities.add(v.lower().strip())

    # Match query entities against document entities
    for query_entity in query_entities:
        query_normalized = query_entity.lower().strip()

        # Exact match
        if query_normalized in doc_entities:
            return True, 1.0, query_entity

        # Partial match
        for doc_entity in doc_entities:
            if query_normalized in doc_entity or doc_entity in query_normalized:
                return True, 0.8, query_entity

    return False, 0.0, ""


def calculate_relevance_score(
    query: str,
    schema_type: str,
    header_data: Dict[str, Any],
    fields: List[str],
    doc_metadata: Dict[str, Any] = None,
    document_type_hints: List[str] = None
) -> float:
    """
    Calculate relevance score based on header_data content.

    Uses entity matching and boost/penalty scoring for accurate filtering.

    Args:
        query: The user's query
        schema_type: Document schema type (tabular, invoice, etc.)
        header_data: The full header_data JSONB
        fields: Extracted field names
        doc_metadata: Additional document metadata
        document_type_hints: Expected document types from query

    Returns:
        Relevance score between 0 and 1
    """
    score = 0.0
    query_lower = query.lower()
    is_analytics = is_analytics_query(query)

    # Extract meaningful words from query
    stop_words = {"the", "a", "an", "for", "all", "how", "many", "we", "have", "is", "are",
                  "and", "or", "to", "in", "of", "with", "finally", "please", "can", "you"}
    query_words = {w for w in query_lower.split() if len(w) > 2 and w not in stop_words}

    aggregation_words = {"sum", "total", "count", "average", "avg", "min", "max", "group"}
    query_aggregations = query_words & aggregation_words
    query_data_terms = query_words & DATA_KEYWORDS

    # Entity matching
    query_entities = extract_query_entities(query)
    entity_matched = False
    entity_score = 0.0

    if query_entities and header_data:
        entity_matched, entity_score, matched_entity = match_entity_to_document(
            query_entities, header_data, doc_metadata
        )
        if entity_matched:
            score += ENTITY_MATCH_BOOST * entity_score
            logger.debug(f"[DocumentFilter] Entity match '{matched_entity}': +{ENTITY_MATCH_BOOST * entity_score:.2f}")

    # Schema type relevance and document type matching
    doc_type_matched = False
    if schema_type:
        schema_lower = schema_type.lower()

        # Check if document type matches query hints
        if document_type_hints:
            for hint in document_type_hints:
                if hint.lower() in schema_lower or schema_lower in hint.lower():
                    score += DOC_TYPE_BOOST
                    doc_type_matched = True
                    logger.debug(f"[DocumentFilter] Doc type match '{hint}' -> '{schema_type}': +{DOC_TYPE_BOOST:.2f}")
                    break

        # Additional boost for tabular data on analytics queries
        if schema_lower in ["tabular", "spreadsheet", "csv", "excel", "table"]:
            if is_analytics or query_aggregations:
                score += TABULAR_BOOST
            elif not doc_type_matched:
                score += 0.2
        elif schema_lower in ["invoice", "receipt", "bill"]:
            # If not already matched via document_type_hints, check data terms
            if not doc_type_matched and query_data_terms & {"invoice", "receipt", "amount", "total", "price"}:
                score += DOC_TYPE_BOOST
                doc_type_matched = True
        elif schema_lower in ["document", "text", "pdf"]:
            if not doc_type_matched:
                score += 0.1

        # Document type mismatch penalty (only if hints provided AND no match)
        if document_type_hints and not doc_type_matched:
            score *= DOC_TYPE_MISMATCH_PENALTY
            logger.debug(f"[DocumentFilter] Doc type mismatch penalty: ×{DOC_TYPE_MISMATCH_PENALTY}")

    # Field name matching
    if fields:
        field_match_count = 0
        strong_match_count = 0

        for field in fields:
            field_lower = field.lower()
            field_words = set(field_lower.replace('_', ' ').replace('-', ' ').split())

            if query_words & field_words:
                field_match_count += 1

            if query_data_terms & field_words:
                strong_match_count += 1

        if strong_match_count > 0:
            score += 0.3 * min(strong_match_count, 2)

        if field_match_count > 0:
            score += 0.15 * min(field_match_count, 3)

    # Check actual data values in header_data
    if header_data and isinstance(header_data, dict):
        if "column_headers" in header_data:
            columns = header_data.get("column_headers", [])
            if isinstance(columns, list):
                columns_text = " ".join(str(c).lower() for c in columns if c)
                matching_terms = sum(1 for term in query_data_terms if term in columns_text)
                if matching_terms > 0:
                    score += 0.2 * min(matching_terms, 2)

        if query_aggregations:
            has_numbers = any(
                isinstance(v, (int, float)) or
                (isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit())
                for v in header_data.values()
                if not isinstance(v, (dict, list))
            )
            if has_numbers:
                score += 0.1

    # Penalize documents with no useful fields
    if not fields:
        score = max(score - 0.2, 0.1)

    # Entity no-match penalty
    if query_entities and not entity_matched:
        score = score * ENTITY_NO_MATCH_PENALTY
        logger.debug(f"[DocumentFilter] No entity match penalty: ×{ENTITY_NO_MATCH_PENALTY}")

    return min(score, 1.0)


def apply_adaptive_threshold(
    scored_docs: List[Dict[str, Any]],
    score_key: str = "relevance_score"
) -> List[Dict[str, Any]]:
    """
    Apply adaptive threshold based on top score.

    Documents must score >= ADAPTIVE_THRESHOLD_RATIO * top_score.

    Args:
        scored_docs: List of document dicts with scores, sorted by score descending
        score_key: Key name for the score field

    Returns:
        Filtered list of documents meeting threshold
    """
    if not scored_docs:
        return []

    top_score = scored_docs[0].get(score_key, 0)
    threshold = top_score * ADAPTIVE_THRESHOLD_RATIO

    filtered = [doc for doc in scored_docs if doc.get(score_key, 0) >= threshold]

    if len(filtered) < len(scored_docs):
        logger.info(
            f"[DocumentFilter] Adaptive threshold {threshold:.2f} (top={top_score:.2f}): "
            f"{len(scored_docs)} -> {len(filtered)} documents"
        )

    return filtered


def extract_fields_from_header_data(header_data: Dict[str, Any]) -> List[str]:
    """
    Extract field names from header_data based on document type.

    Args:
        header_data: The header_data JSONB from documents_data

    Returns:
        List of field names that can be queried
    """
    if not header_data or not isinstance(header_data, dict):
        return []

    fields = []

    # For tabular data, column_headers is the primary source
    if "column_headers" in header_data:
        columns = header_data["column_headers"]
        if isinstance(columns, list):
            fields.extend([str(c) for c in columns if c])

    # Also extract top-level keys that look like data fields
    skip_keys = {"column_headers", "sheet_name", "row_count", "metadata", "extraction_info"}
    for key in header_data.keys():
        if key not in skip_keys and key not in fields:
            value = header_data[key]
            if value is not None and not isinstance(value, (dict, list)):
                fields.append(key)

    return fields


class DocumentFilter:
    """
    High-level document filtering interface.

    Combines topic pre-filtering, relevance scoring, and adaptive thresholding.
    """

    def __init__(self, db_session_factory=None):
        """
        Initialize the document filter.

        Args:
            db_session_factory: Function to get database session
        """
        self.db_session_factory = db_session_factory

    def filter_documents(
        self,
        query: str,
        available_document_ids: List[str],
        enriched_docs: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter documents using topic pre-filter and relevance scoring.

        Args:
            query: User's query
            available_document_ids: List of accessible document IDs
            enriched_docs: Optional pre-enriched documents with schema info

        Returns:
            Filtered list of documents sorted by relevance
        """
        if not available_document_ids:
            return []

        # Step 1: Extract query hints
        hints = extract_query_hints(query)
        topics = hints.get('topics', [])
        doc_type_hints = hints.get('document_type_hints', [])

        # Step 2: Apply topic pre-filter if we have topics or type hints
        filtered_ids = available_document_ids
        if (topics or doc_type_hints) and self.db_session_factory:
            prefiltered_ids, _ = topic_based_prefilter(
                topics=topics,
                document_type_hints=doc_type_hints,
                accessible_document_ids=available_document_ids,
                db_session_factory=self.db_session_factory
            )
            if prefiltered_ids:
                filtered_ids = prefiltered_ids
                logger.info(f"[DocumentFilter] Pre-filter reduced {len(available_document_ids)} -> {len(filtered_ids)} documents")

        # Step 3: If enriched docs provided, filter and score them
        if enriched_docs:
            # Filter to only pre-filtered IDs
            filtered_docs = [
                doc for doc in enriched_docs
                if doc.get('document_id') in filtered_ids
            ]

            # Calculate relevance scores
            for doc in filtered_docs:
                header_data = doc.get('header_data', {})
                fields = extract_fields_from_header_data(header_data)

                doc['relevance_score'] = calculate_relevance_score(
                    query=query,
                    schema_type=doc.get('schema_type', ''),
                    header_data=header_data,
                    fields=fields,
                    doc_metadata=doc.get('extraction_metadata'),
                    document_type_hints=doc_type_hints
                )

            # Sort by relevance
            filtered_docs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

            # Apply adaptive threshold
            filtered_docs = apply_adaptive_threshold(filtered_docs)

            return filtered_docs

        return [{'document_id': doc_id} for doc_id in filtered_ids]
