"""
Document routing tools for the Planner Agent.

These tools handle:
- Finding relevant documents for a query
- Analyzing document schemas
- Grouping documents by schema compatibility

Quick Win Integration:
- Reuses EntityExtractor from rag_service for entity extraction
- Reuses entity matching and fuzzy matching logic
- Applies boost/penalty scoring from non-agent flow
- Uses adaptive thresholding instead of fixed threshold
"""

import json
import logging
import os
import sys
from typing import Annotated, List, Dict, Any, Tuple, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from agents.state.models import DocumentSource, SchemaGroup

logger = logging.getLogger(__name__)

# Ensure backend path is in sys.path for imports
_BACKEND_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_PATH not in sys.path:
    sys.path.insert(0, _BACKEND_PATH)

# Import shared services from rag_service (Quick Win)
try:
    from rag_service.entity_extractor import (
        normalize_entity_name,
        are_entities_similar,
        entity_matches_query,
        extract_entities_from_header_data,
        FUZZY_MATCH_THRESHOLD
    )
    ENTITY_EXTRACTOR_AVAILABLE = True
    logger.info("[routing_tools] Loaded EntityExtractor from rag_service")
except ImportError as e:
    logger.warning(f"[routing_tools] EntityExtractor not available: {e}")
    ENTITY_EXTRACTOR_AVAILABLE = False
    FUZZY_MATCH_THRESHOLD = 80

# Scoring configuration (from non-agent flow)
ENTITY_MATCH_BOOST = 0.3       # Boost for entity match
ENTITY_FUZZY_BOOST = 0.2       # Boost for fuzzy entity match
ENTITY_NO_MATCH_PENALTY = 0.4  # Multiply by this when no entity match
DOC_TYPE_BOOST = 0.25          # Boost for document type match
TABULAR_BOOST = 0.35           # Boost for tabular data on analytics queries
ADAPTIVE_THRESHOLD_RATIO = 0.5  # Documents must score >= 50% of top score


# =============================================================================
# HELPER FUNCTIONS (defined before tools that use them)
# =============================================================================

def _extract_query_entities(query: str) -> List[str]:
    """
    Extract potential entity names from the query.

    QUICK WIN: Uses simple heuristics to identify entity names.
    In production, this could use the full EntityExtractor with SpaCy.

    Args:
        query: The user's query

    Returns:
        List of potential entity names
    """
    entities = []

    # Common entity indicators
    entity_indicators = [
        'from', 'by', 'for', 'vendor', 'supplier', 'company', 'customer',
        'client', 'merchant', 'store', 'seller'
    ]

    query_lower = query.lower()
    words = query.split()

    # Look for capitalized words that might be entity names
    for i, word in enumerate(words):
        # Skip common words
        if word.lower() in ['the', 'a', 'an', 'and', 'or', 'for', 'from', 'by', 'to', 'in', 'on', 'at']:
            continue

        # Check if previous word is an entity indicator
        if i > 0 and words[i-1].lower() in entity_indicators:
            # This word and following capitalized words might be an entity
            entity_parts = [word]
            for j in range(i + 1, min(i + 4, len(words))):
                if words[j][0].isupper() if words[j] else False:
                    entity_parts.append(words[j])
                else:
                    break
            if entity_parts:
                entities.append(' '.join(entity_parts))

        # Check if word is capitalized and not at sentence start
        elif i > 0 and word and word[0].isupper():
            # Might be an entity name
            entity_parts = [word]
            for j in range(i + 1, min(i + 4, len(words))):
                if words[j][0].isupper() if words[j] else False:
                    entity_parts.append(words[j])
                else:
                    break
            if len(entity_parts) > 0 and entity_parts[0] not in ['I', 'Show', 'List', 'Get', 'Find', 'What', 'How', 'Calculate']:
                entities.append(' '.join(entity_parts))

    # Deduplicate
    seen = set()
    unique_entities = []
    for e in entities:
        e_lower = e.lower()
        if e_lower not in seen:
            seen.add(e_lower)
            unique_entities.append(e)

    return unique_entities


def _match_entity_to_document(
    query_entities: List[str],
    doc_header_data: Dict[str, Any],
    doc_metadata: Dict[str, Any] = None
) -> Tuple[bool, float, str]:
    """
    Check if any query entity matches the document's entities.

    QUICK WIN: Uses shared entity matching from rag_service.

    Args:
        query_entities: List of entity names from query
        doc_header_data: Document's header_data
        doc_metadata: Additional document metadata (may contain normalized entities)

    Returns:
        Tuple of (matches: bool, score: float, matched_entity: str)
    """
    if not query_entities:
        return False, 0.0, ""

    # Extract entities from header_data if EntityExtractor is available
    if ENTITY_EXTRACTOR_AVAILABLE and doc_header_data:
        doc_entities = extract_entities_from_header_data(doc_header_data)
        doc_entity_names = doc_entities.get_all_normalized_entities() if hasattr(doc_entities, 'get_all_normalized_entities') else []

        # Check for matches
        for query_entity in query_entities:
            matches, score = entity_matches_query(query_entity, doc_entity_names)
            if matches:
                return True, score, query_entity

    # Fallback: Check vendor/customer fields directly
    vendor_fields = ['vendor_name', 'seller_name', 'merchant_name', 'store_name', 'company_name']
    customer_fields = ['customer_name', 'buyer_name', 'client_name']

    for query_entity in query_entities:
        query_normalized = normalize_entity_name(query_entity) if ENTITY_EXTRACTOR_AVAILABLE else query_entity.lower()

        # Check vendor fields
        for field in vendor_fields:
            value = doc_header_data.get(field, '')
            if value:
                doc_normalized = normalize_entity_name(value) if ENTITY_EXTRACTOR_AVAILABLE else value.lower()
                if query_normalized == doc_normalized:
                    return True, 1.0, query_entity
                if ENTITY_EXTRACTOR_AVAILABLE and are_entities_similar(query_entity, value):
                    return True, 0.85, query_entity
                # Simple substring match as fallback
                if query_normalized in doc_normalized or doc_normalized in query_normalized:
                    return True, 0.7, query_entity

        # Check customer fields
        for field in customer_fields:
            value = doc_header_data.get(field, '')
            if value:
                doc_normalized = normalize_entity_name(value) if ENTITY_EXTRACTOR_AVAILABLE else value.lower()
                if query_normalized == doc_normalized:
                    return True, 1.0, query_entity
                if ENTITY_EXTRACTOR_AVAILABLE and are_entities_similar(query_entity, value):
                    return True, 0.85, query_entity

    # Check normalized entity fields in metadata
    if doc_metadata:
        vendor_normalized = doc_metadata.get('vendor_normalized', '')
        customer_normalized = doc_metadata.get('customer_normalized', '')
        all_entities = doc_metadata.get('all_entities_normalized', [])

        for query_entity in query_entities:
            query_normalized = normalize_entity_name(query_entity) if ENTITY_EXTRACTOR_AVAILABLE else query_entity.lower()

            if vendor_normalized and query_normalized == vendor_normalized:
                return True, 1.0, query_entity
            if customer_normalized and query_normalized == customer_normalized:
                return True, 1.0, query_entity
            if query_normalized in all_entities:
                return True, 0.9, query_entity

    return False, 0.0, ""


def _is_analytics_query(query: str) -> bool:
    """Check if query is an analytics/aggregation query."""
    analytics_keywords = {
        'sum', 'total', 'count', 'average', 'avg', 'min', 'max',
        'how many', 'how much', 'calculate', 'aggregate', 'group by',
        'report', 'analysis', 'statistics', 'breakdown'
    }
    query_lower = query.lower()
    return any(kw in query_lower for kw in analytics_keywords)


def _apply_adaptive_threshold(scored_docs: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
    """
    Apply adaptive threshold based on top score.

    QUICK WIN: Uses same logic as non-agent flow.
    Documents must score >= ADAPTIVE_THRESHOLD_RATIO * top_score.

    Args:
        scored_docs: List of (doc_dict, score) tuples, sorted by score descending

    Returns:
        Filtered list of documents meeting threshold
    """
    if not scored_docs:
        return []

    top_score = scored_docs[0][1]
    threshold = top_score * ADAPTIVE_THRESHOLD_RATIO

    filtered = [(doc, score) for doc, score in scored_docs if score >= threshold]

    if len(filtered) < len(scored_docs):
        logger.info(f"[routing_tools] Adaptive threshold {threshold:.2f} (top={top_score:.2f}): "
                   f"{len(scored_docs)} -> {len(filtered)} documents")

    return filtered


def _extract_fields_from_header_data(header_data: Dict[str, Any]) -> List[str]:
    """Extract field names from header_data based on document type.

    For tabular data: extracts column_headers
    For invoices/receipts: extracts top-level keys
    For other types: extracts all meaningful keys

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
    # Skip metadata keys
    skip_keys = {"column_headers", "sheet_name", "row_count", "metadata", "extraction_info"}
    for key in header_data.keys():
        if key not in skip_keys and key not in fields:
            # Check if this looks like a data field (has actual value)
            value = header_data[key]
            if value is not None and not isinstance(value, (dict, list)):
                fields.append(key)

    return fields


def _calculate_relevance_from_header_data(
    query: str,
    schema_type: str,
    header_data: Dict[str, Any],
    fields: List[str],
    doc_metadata: Dict[str, Any] = None
) -> float:
    """Calculate relevance score based on header_data content.

    QUICK WIN: Enhanced with entity matching and boost/penalty scoring from non-agent flow.

    This is more reliable than filename-based matching because it uses
    the actual extracted data structure.

    Args:
        query: The user's query
        schema_type: Document schema type (tabular, invoice, etc.)
        header_data: The full header_data JSONB
        fields: Extracted field names
        doc_metadata: Additional document metadata (may contain normalized entities)

    Returns:
        Relevance score between 0 and 1
    """
    score = 0.0
    query_lower = query.lower()
    is_analytics = _is_analytics_query(query)

    # Extract meaningful words from query
    stop_words = {"the", "a", "an", "for", "all", "how", "many", "we", "have", "is", "are",
                  "and", "or", "to", "in", "of", "with", "finally", "please", "can", "you"}
    query_words = {w for w in query_lower.split() if len(w) > 2 and w not in stop_words}

    # Aggregation keywords that indicate need for structured data
    aggregation_words = {"sum", "total", "count", "average", "avg", "min", "max", "group"}
    data_words = {"inventory", "product", "products", "sales", "quantity", "amount", "price",
                  "cost", "revenue", "stock", "order", "orders", "invoice", "receipt"}

    query_aggregations = query_words & aggregation_words
    query_data_terms = query_words & data_words

    # QUICK WIN: Extract and match entities
    query_entities = _extract_query_entities(query)
    entity_matched = False
    entity_score = 0.0

    if query_entities and header_data:
        entity_matched, entity_score, matched_entity = _match_entity_to_document(
            query_entities, header_data, doc_metadata
        )
        if entity_matched:
            # BOOST: Entity match
            score += ENTITY_MATCH_BOOST * entity_score
            logger.debug(f"[routing_tools] Entity match '{matched_entity}': +{ENTITY_MATCH_BOOST * entity_score:.2f}")

    # 1. Schema type relevance (highest weight for tabular + aggregation queries)
    if schema_type:
        schema_lower = schema_type.lower()
        if schema_lower in ["tabular", "spreadsheet", "csv", "excel", "table"]:
            if is_analytics or query_aggregations:
                # BOOST: Tabular data for analytics queries
                score += TABULAR_BOOST
                logger.debug(f"[routing_tools] Tabular boost for analytics: +{TABULAR_BOOST:.2f}")
            else:
                score += 0.2  # Still good for data queries
        elif schema_lower in ["invoice", "receipt", "bill"]:
            # BOOST: Document type match
            if query_data_terms & {"invoice", "receipt", "amount", "total", "price"}:
                score += DOC_TYPE_BOOST
        elif schema_lower in ["document", "text", "pdf"]:
            score += 0.1  # Lower priority for unstructured

    # 2. Field name matching (most important signal)
    if fields:
        field_match_count = 0
        strong_match_count = 0

        for field in fields:
            field_lower = field.lower()
            field_words = set(field_lower.replace('_', ' ').replace('-', ' ').split())

            # Direct query word match
            if query_words & field_words:
                field_match_count += 1

            # Data term match (e.g., query mentions "inventory", field is "inventory_count")
            if query_data_terms & field_words:
                strong_match_count += 1

        # Strong matches are worth more
        if strong_match_count > 0:
            score += 0.3 * min(strong_match_count, 2)  # Up to 0.6

        if field_match_count > 0:
            score += 0.15 * min(field_match_count, 3)  # Up to 0.45

    # 3. Check actual data values in header_data for semantic relevance
    if header_data and isinstance(header_data, dict):
        # For tabular data, check column headers content
        if "column_headers" in header_data:
            columns = header_data.get("column_headers", [])
            if isinstance(columns, list):
                columns_text = " ".join(str(c).lower() for c in columns if c)
                # Check if query terms appear in column headers
                matching_terms = sum(1 for term in query_data_terms if term in columns_text)
                if matching_terms > 0:
                    score += 0.2 * min(matching_terms, 2)

        # Check if document has numerical data (good for aggregations)
        if query_aggregations:
            has_numbers = any(
                isinstance(v, (int, float)) or
                (isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit())
                for v in header_data.values()
                if not isinstance(v, (dict, list))
            )
            if has_numbers:
                score += 0.1

    # 4. Penalize documents with no useful fields
    if not fields:
        score = max(score - 0.2, 0.1)

    # QUICK WIN: Apply entity no-match penalty if query has entities but document doesn't match
    if query_entities and not entity_matched:
        # Only apply penalty if we had entities to match
        score = score * ENTITY_NO_MATCH_PENALTY
        logger.debug(f"[routing_tools] No entity match penalty: ×{ENTITY_NO_MATCH_PENALTY:.2f}")

    return min(score, 1.0)


def _calculate_relevance_simple(query: str, filename: str, schema_type: str, fields: List[str]) -> float:
    """Calculate relevance score for a document based on simple matching (fallback).

    Args:
        query: The user's query
        filename: Document filename
        schema_type: Document schema type
        fields: List of field names in the document

    Returns:
        Relevance score between 0 and 1
    """
    score = 0.0
    query_lower = query.lower()

    # Extract meaningful words from query (filter out stop words and short words)
    stop_words = {"the", "a", "an", "for", "all", "how", "many", "we", "have", "is", "are", "and", "or", "to", "in", "of", "with", "finally"}
    query_words = {w for w in query_lower.split() if len(w) > 2 and w not in stop_words}

    # Also look for aggregation-related words that map to field types
    aggregation_words = {"sum", "total", "count", "average", "avg", "min", "max", "inventory", "product", "products", "sales", "quantity", "amount", "price", "cost"}
    query_aggregations = query_words & aggregation_words

    # Check filename match (weak signal - not very reliable)
    if filename:
        filename_lower = filename.lower()
        filename_words = set(filename_lower.replace('_', ' ').replace('-', ' ').replace('.', ' ').split())
        matching_words = query_words & filename_words
        if matching_words:
            score += 0.2 * min(len(matching_words), 2)  # Reduced weight

    # Check schema type match
    if schema_type:
        schema_lower = schema_type.lower()
        # Tabular data is often relevant for aggregation queries
        if schema_lower in ["tabular", "spreadsheet", "csv", "excel"]:
            if query_aggregations:
                score += 0.3  # Tabular data good for aggregations
        if any(word in schema_lower for word in query_words):
            score += 0.2

    # Check field name matches (strong signal for relevance)
    field_match_count = 0
    for field in fields:
        field_lower = field.lower()
        # Check for direct word matches
        field_words = set(field_lower.replace('_', ' ').replace('-', ' ').split())
        if query_words & field_words:
            field_match_count += 1
        # Check for aggregation relevance
        if query_aggregations & field_words:
            field_match_count += 1

    if field_match_count > 0:
        score += 0.15 * min(field_match_count, 3)  # Up to 0.45 for field matches

    # Base relevance for having fields (structured data is generally useful)
    if fields and len(fields) > 3:
        score += 0.1

    return min(score, 1.0)


# =============================================================================
# TOOL FUNCTIONS
# =============================================================================

@tool
def get_relevant_documents(
    query: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Get relevant documents for the query.

    Uses documents already provided in the state (from user's accessible documents)
    and enriches them with schema information from the database.

    IMPORTANT: The system relies on document IDs passed in state.available_documents,
    NOT workspace_id. Documents are pre-filtered by access control before reaching
    the agentic workflow.

    Args:
        query: The user's natural language query

    Returns:
        JSON array of relevant documents with schema information including:
        - document_id: Unique document identifier
        - filename: Original filename
        - schema_type: Type of document (tabular, invoice, etc.)
        - schema_definition: Field definitions
        - relevance_score: How relevant to the query (0-1)
    """
    try:
        # Debug: Log state info
        logger.info(f"[get_relevant_documents] Called with query='{query[:50]}...'")
        logger.info(f"[get_relevant_documents] State type: {type(state)}")

        if state is None:
            logger.error("[get_relevant_documents] State is None!")
            return json.dumps({"error": "State is None - cannot access documents", "documents": []})

        state_keys = list(state.keys()) if isinstance(state, dict) else []
        logger.info(f"[get_relevant_documents] State keys: {state_keys}")

        # Get documents from state - these are pre-filtered by access control
        available_docs = state.get("available_documents", [])

        logger.info(f"[get_relevant_documents] available_docs type: {type(available_docs)}, length: {len(available_docs) if available_docs else 0}")

        if available_docs:
            logger.info(f"[get_relevant_documents] Found {len(available_docs)} documents in state")

            # Convert DocumentSource objects to dicts if needed
            doc_dicts = []
            for doc in available_docs:
                if hasattr(doc, 'dict'):
                    doc_dict = doc.dict()
                elif hasattr(doc, 'model_dump'):
                    doc_dict = doc.model_dump()
                elif isinstance(doc, dict):
                    doc_dict = doc
                else:
                    doc_dict = {"document_id": str(doc)}
                doc_dicts.append(doc_dict)

            # Extract document IDs for enrichment
            doc_ids = [d.get("document_id") for d in doc_dicts if d.get("document_id")]
            logger.info(f"[get_relevant_documents] Extracted {len(doc_ids)} document IDs for enrichment")

            if doc_ids:
                # Try to enrich with schema information from database
                try:
                    from sqlalchemy import text
                    # Import database session - try multiple paths for compatibility
                    try:
                        from db.database import get_db_session
                    except ImportError:
                        # Fallback for different import contexts
                        import sys
                        import os
                        backend_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        if backend_path not in sys.path:
                            sys.path.insert(0, backend_path)
                        from db.database import get_db_session

                    enriched_docs = []

                    with get_db_session() as db:
                        # QUICK WIN: Enhanced query to include extraction_metadata for entity matching
                        # Join documents_data (has schema_type, header_data) with documents (has filename)
                        placeholders = ", ".join([f"'{doc_id}'" for doc_id in doc_ids])
                        result = db.execute(text(f"""
                            SELECT
                                dd.document_id::text,
                                d.original_filename as filename,
                                dd.schema_type,
                                COALESCE(dd.header_data, '{{}}') as header_data,
                                COALESCE(dd.extraction_metadata, '{{}}') as extraction_metadata
                            FROM documents_data dd
                            JOIN documents d ON dd.document_id = d.id
                            WHERE dd.document_id::text IN ({placeholders})
                        """))

                        rows = result.fetchall()
                        db_docs = {str(row[0]): row for row in rows}

                        for doc_dict in doc_dicts:
                            doc_id = doc_dict.get("document_id")
                            row = db_docs.get(doc_id)

                            if row:
                                # Get schema fields from header_data
                                header_data = row[3] if row[3] else {}
                                if isinstance(header_data, str):
                                    try:
                                        header_data = json.loads(header_data)
                                    except:
                                        header_data = {}

                                # QUICK WIN: Get extraction_metadata for entity matching
                                extraction_metadata = row[4] if row[4] else {}
                                if isinstance(extraction_metadata, str):
                                    try:
                                        extraction_metadata = json.loads(extraction_metadata)
                                    except:
                                        extraction_metadata = {}

                                # Extract fields from header_data
                                # For tabular data: use column_headers if available
                                # For other types: use top-level keys
                                fields = _extract_fields_from_header_data(header_data)

                                # QUICK WIN: Calculate relevance with entity matching
                                relevance = _calculate_relevance_from_header_data(
                                    query, row[2] or "", header_data, fields, extraction_metadata
                                )

                                # IMPORTANT: Only send essential metadata to LLM, not the full header_data
                                # Full header_data can contain thousands of tokens (persons_normalized, etc.)
                                essential_metadata = {
                                    "column_headers": header_data.get("column_headers", []),
                                    "total_rows": header_data.get("total_rows"),
                                    "total_columns": header_data.get("total_columns"),
                                }

                                enriched_docs.append({
                                    "document_id": str(row[0]),
                                    "filename": row[1] or f"doc_{doc_id[:8]}",
                                    "schema_type": row[2] or "unknown",
                                    "schema_definition": {"fields": fields, "metadata": essential_metadata},
                                    "relevance_score": relevance,
                                    "data_location": "header"
                                })
                            else:
                                # Document not found in DB, use basic info
                                enriched_docs.append({
                                    "document_id": doc_id,
                                    "filename": doc_dict.get("filename", f"doc_{doc_id[:8]}"),
                                    "schema_type": doc_dict.get("schema_type", "unknown"),
                                    "schema_definition": {"fields": []},
                                    "relevance_score": 0.2,  # Lower score for documents without data
                                    "data_location": "header"
                                })

                    # Sort by relevance (descending)
                    enriched_docs.sort(key=lambda x: x["relevance_score"], reverse=True)

                    # QUICK WIN: Apply adaptive threshold
                    if enriched_docs:
                        top_score = enriched_docs[0]["relevance_score"]
                        threshold = top_score * ADAPTIVE_THRESHOLD_RATIO
                        original_count = len(enriched_docs)

                        # Filter documents below threshold
                        enriched_docs = [
                            doc for doc in enriched_docs
                            if doc["relevance_score"] >= threshold
                        ]

                        if len(enriched_docs) < original_count:
                            logger.info(
                                f"[get_relevant_documents] Adaptive threshold {threshold:.2f} "
                                f"(top={top_score:.2f}): {original_count} -> {len(enriched_docs)} documents"
                            )

                    logger.info(f"[get_relevant_documents] Returning {len(enriched_docs)} relevant documents")
                    for doc in enriched_docs[:5]:
                        logger.info(f"[get_relevant_documents]   - {doc['filename']}: score={doc['relevance_score']:.2f}")

                    return json.dumps(enriched_docs, indent=2)

                except Exception as db_error:
                    logger.warning(f"[get_relevant_documents] Database enrichment failed: {db_error}")
                    # Calculate relevance from available document metadata
                    basic_docs = []
                    for d in doc_dicts:
                        filename = d.get("filename", f"doc_{d.get('document_id', 'unknown')[:8]}")
                        schema_type = d.get("schema_type", "unknown")
                        schema_def = d.get("schema_definition", {})
                        fields = schema_def.get("fields", []) if isinstance(schema_def, dict) else []

                        # Calculate relevance even without DB enrichment
                        relevance = _calculate_relevance_simple(query, filename, schema_type, fields)

                        basic_docs.append({
                            "document_id": d.get("document_id"),
                            "filename": filename,
                            "schema_type": schema_type,
                            "schema_definition": {"fields": fields},
                            "relevance_score": relevance,
                            "data_location": "header"
                        })

                    # Sort by relevance
                    basic_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
                    logger.info(f"[get_relevant_documents] Returning {len(basic_docs)} documents with calculated relevance (no DB)")
                    return json.dumps(basic_docs, indent=2)

        # No documents in state - this is an error condition
        # Documents should have been passed from chat_api.py via accessible_doc_ids
        logger.error("[get_relevant_documents] No documents in state! Check if accessible_doc_ids was passed to workflow")
        return json.dumps({
            "error": "No accessible documents found in state. Documents must be pre-filtered by access control.",
            "documents": [],
            "hint": "Ensure accessible_doc_ids is passed to execute_analytics_query_async"
        })

    except Exception as e:
        logger.error(f"[get_relevant_documents] Error: {e}")
        return json.dumps({"error": str(e), "documents": []})


def _calculate_relevance(query: str, doc: Any, schema_def: Dict) -> float:
    """Calculate relevance score for a document object (legacy)."""
    score = 0.0
    query_lower = query.lower()

    # Check filename match
    if doc.filename:
        filename_lower = doc.filename.lower()
        if any(word in filename_lower for word in query_lower.split()):
            score += 0.3

    # Check schema type match
    if doc.schema_type:
        schema_lower = doc.schema_type.lower()
        if schema_lower in query_lower or query_lower in schema_lower:
            score += 0.2

    # Check field name matches
    fields = schema_def.get("fields", [])
    for field in fields:
        if field.lower() in query_lower:
            score += 0.2
            break

    # Base relevance for having data
    if doc.header_data or doc.summary_data:
        score += 0.2

    return min(score, 1.0)


@tool
def analyze_document_schemas(
    document_ids: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Analyze schemas of selected documents to identify commonalities and differences.

    This tool examines the schemas of multiple documents to determine:
    - Common fields across documents
    - Whether documents can be processed together
    - Recommended grouping strategy

    Args:
        document_ids: JSON array of document IDs to analyze

    Returns:
        Schema analysis with:
        - schemas: Individual schema information per document
        - common_fields: Fields shared across all documents
        - groupable: Whether documents can be combined
        - recommended_groups: Suggested grouping strategy
    """
    try:
        doc_ids = json.loads(document_ids)

        # Get available documents from state
        available_docs = state.get("available_documents", [])
        if isinstance(available_docs, str):
            available_docs = json.loads(available_docs)

        # Filter to requested documents
        docs_to_analyze = []
        for doc in available_docs:
            doc_dict = doc if isinstance(doc, dict) else doc.dict()
            if doc_dict.get("document_id") in doc_ids:
                docs_to_analyze.append(doc_dict)

        if not docs_to_analyze:
            return json.dumps({
                "error": "No matching documents found",
                "schemas": [],
                "common_fields": [],
                "groupable": False
            })

        # Analyze schemas
        all_fields_by_doc = {}
        schema_types = set()

        for doc in docs_to_analyze:
            doc_id = doc["document_id"]
            schema_def = doc.get("schema_definition", {})
            fields = set(schema_def.get("fields", []))
            all_fields_by_doc[doc_id] = fields
            schema_types.add(doc.get("schema_type", "unknown"))

        # Find common fields
        if all_fields_by_doc:
            common_fields = set.intersection(*all_fields_by_doc.values())
        else:
            common_fields = set()

        # Determine if groupable
        groupable = (
            len(schema_types) == 1 and
            len(common_fields) >= 2 and
            "unknown" not in schema_types
        )

        # Generate recommendations
        recommended_groups = []
        if groupable:
            recommended_groups.append({
                "group_id": f"group_{list(schema_types)[0]}",
                "schema_type": list(schema_types)[0],
                "document_ids": doc_ids,
                "can_combine": True,
                "reason": f"All documents share schema type and {len(common_fields)} common fields"
            })
        else:
            # Group by schema type
            by_type = {}
            for doc in docs_to_analyze:
                st = doc.get("schema_type", "unknown")
                if st not in by_type:
                    by_type[st] = []
                by_type[st].append(doc["document_id"])

            for st, ids in by_type.items():
                recommended_groups.append({
                    "group_id": f"group_{st}",
                    "schema_type": st,
                    "document_ids": ids,
                    "can_combine": len(ids) > 1,
                    "reason": f"Grouped by schema type: {st}"
                })

        result = {
            "schemas": [
                {
                    "document_id": doc["document_id"],
                    "schema_type": doc.get("schema_type"),
                    "fields": list(all_fields_by_doc.get(doc["document_id"], []))
                }
                for doc in docs_to_analyze
            ],
            "common_fields": list(common_fields),
            "groupable": groupable,
            "schema_types": list(schema_types),
            "recommended_groups": recommended_groups
        }

        return json.dumps(result, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})
    except Exception as e:
        logger.error(f"Error analyzing schemas: {e}")
        return json.dumps({"error": str(e)})


@tool
def group_documents_by_schema(
    documents: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Group documents by compatible schemas for efficient processing.

    Documents with similar schemas can be combined into single SQL queries.
    Documents with different schemas must be processed separately.

    Args:
        documents: JSON array of document sources with schema info

    Returns:
        JSON array of schema groups with:
        - group_id: Unique group identifier
        - schema_type: Schema type for the group
        - common_fields: Fields shared in the group
        - documents: Documents in the group
        - can_combine: Whether to process together
        - document_count: Number of documents
    """
    try:
        docs = json.loads(documents)

        # Group by schema_type
        groups: Dict[str, Dict[str, Any]] = {}

        for doc in docs:
            schema_type = doc.get("schema_type", "unknown")

            if schema_type not in groups:
                groups[schema_type] = {
                    "group_id": f"group_{schema_type}_{len(groups)}",
                    "schema_type": schema_type,
                    "documents": [],
                    "all_fields": []
                }

            groups[schema_type]["documents"].append(doc)

            # Track fields for this document
            schema_def = doc.get("schema_definition", {})
            fields = set(schema_def.get("fields", []))
            groups[schema_type]["all_fields"].append(fields)

        # Process each group to find common fields
        result = []
        for schema_type, group in groups.items():
            all_field_sets = group["all_fields"]

            # Find intersection of all field sets
            if all_field_sets:
                common_fields = set.intersection(*all_field_sets) if len(all_field_sets) > 1 else all_field_sets[0]
            else:
                common_fields = set()

            # Determine if documents can be combined
            # Can combine if: multiple docs, same schema, and common fields exist
            can_combine = (
                len(group["documents"]) > 1 and
                len(common_fields) > 0 and
                schema_type != "unknown"
            )

            result.append({
                "group_id": group["group_id"],
                "schema_type": schema_type,
                "common_fields": list(common_fields),
                "documents": group["documents"],
                "can_combine": can_combine,
                "document_count": len(group["documents"])
            })

        logger.info(f"Created {len(result)} schema groups from {len(docs)} documents")
        return json.dumps(result, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})
    except Exception as e:
        logger.error(f"Error grouping documents: {e}")
        return json.dumps({"error": str(e)})
