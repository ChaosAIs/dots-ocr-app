"""
Intelligent document routing based on query metadata matching.
Routes queries to the most relevant documents before vector search.

This module solves the "drowning out" problem where large documents dominate
search results by intelligently selecting which documents to search based on
metadata matching between the query and document metadata.

Routing Strategy (in order of preference):
1. Vector Search (Primary): Fast semantic search on document metadata embeddings
2. LLM Scoring (Fallback): When vector search returns insufficient results
3. Rule-based Scoring (Final Fallback): When LLM is unavailable
"""
import os
import logging
import json
from typing import List, Dict, Any, Tuple, Optional, Set

from sqlalchemy import text

from db.document_repository import DocumentRepository
from db.database import get_db_session
from rag_service.chunking.document_types import types_match, expand_type_aliases
from rag_service.entity_extractor import normalize_entity_name, entity_matches_query

logger = logging.getLogger(__name__)

# Configuration
ENABLE_DOCUMENT_ROUTING = True  # Feature flag
USE_VECTOR_ROUTING = os.getenv("METADATA_VECTOR_ROUTING_ENABLED", "true").lower() == "true"
USE_LLM_SCORING = True  # Use LLM-based scoring as fallback
DOCUMENT_ROUTING_MIN_SCORE = 0.3  # Minimum absolute score threshold (for rule-based scoring)

# Advanced filtering to prevent low-quality documents from polluting results
DOCUMENT_ROUTING_SCORE_RATIO = 0.25   # Must be at least 25% of top score
DOCUMENT_ROUTING_MAX_SCORE_GAP = float(os.getenv("DOCUMENT_ROUTING_MAX_SCORE_GAP", "12.0"))  # Max score gap from top document

# LLM scoring thresholds (when USE_LLM_SCORING=True)
LLM_SCORING_MIN_SCORE = 3.0  # Minimum LLM score (0-10 scale)
LLM_SCORING_SCORE_RATIO = 0.4  # Must be at least 40% of top score

# Vector routing thresholds - ADAPTIVE
VECTOR_ROUTING_MIN_RESULTS = int(os.getenv("METADATA_FALLBACK_MIN_RESULTS", "3"))
# Relative score threshold: results must be >= X% of top result score
# Higher = more strict (fewer documents), Lower = more permissive (more documents)
# Default 0.5 allows documents scoring 50%+ of top result (more inclusive for similar docs)
VECTOR_ROUTING_RELATIVE_THRESHOLD = float(os.getenv("METADATA_RELATIVE_THRESHOLD", "0.5"))

# Verbose logging for metadata embedding comparison
METADATA_VERBOSE_LOGGING = os.getenv("METADATA_VERBOSE_LOGGING", "false").lower() == "true"

# Hard entity filtering - when query has specific entities, filter documents by exact match first
ENABLE_HARD_ENTITY_FILTER = os.getenv("ENABLE_HARD_ENTITY_FILTER", "true").lower() == "true"
# Minimum fuzzy match threshold for entity matching (0-100)
ENTITY_MATCH_THRESHOLD = int(os.getenv("ENTITY_MATCH_THRESHOLD", "80"))


class DocumentRouter:
    """Routes queries to relevant documents based on metadata matching."""

    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self.use_vector_routing = USE_VECTOR_ROUTING
        self.use_llm_scoring = USE_LLM_SCORING and llm_service is not None

        # Rule-based scoring thresholds
        self.min_score = DOCUMENT_ROUTING_MIN_SCORE
        self.score_ratio = DOCUMENT_ROUTING_SCORE_RATIO
        self.max_score_gap = DOCUMENT_ROUTING_MAX_SCORE_GAP

        # LLM scoring thresholds
        self.llm_min_score = LLM_SCORING_MIN_SCORE
        self.llm_score_ratio = LLM_SCORING_SCORE_RATIO

        # Vector routing thresholds
        self.vector_min_results = VECTOR_ROUTING_MIN_RESULTS

        if self.use_vector_routing:
            logger.info("[Router] Using vector-based routing (primary) with LLM fallback")
        elif self.use_llm_scoring:
            logger.info("[Router] Using LLM-based scoring for document routing")
        else:
            logger.info("[Router] Using rule-based scoring for document routing")
    
    def route_query(
        self,
        query_metadata: Dict[str, Any],
        original_query: Optional[str] = None,
        accessible_source_names: Optional[set] = None,  # DEPRECATED - use accessible_document_ids
        accessible_document_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Route query to most relevant documents based on metadata matching.

        Routing strategy:
        1. Vector search on metadata embeddings (fast, primary)
        2. LLM scoring fallback (if vector results < min threshold)
        3. Rule-based scoring (if LLM unavailable)

        Args:
            query_metadata: Metadata extracted from query
                - entities: List[str]
                - topics: List[str]
                - document_type_hints: List[str]
                - intent: str
            original_query: Original query text (required for vector/LLM scoring)
            accessible_source_names: DEPRECATED - use accessible_document_ids instead.
            accessible_document_ids: List of document IDs the user can access (for access control).
                                     If provided, only documents with these IDs will be considered.
                                     If empty list, no documents are accessible.

        Returns:
            List of document IDs to search (UUIDs as strings).
            Empty list means no accessible documents.
        """
        # Access control: Use document_ids if provided, otherwise fall back to source_names
        if accessible_document_ids is not None:
            if len(accessible_document_ids) == 0:
                logger.warning("[Router] Access control: user has NO accessible documents - returning empty")
                return []
            else:
                logger.info(f"[Router] Access control: limiting to {len(accessible_document_ids)} accessible document IDs")
        elif accessible_source_names is not None:
            # Deprecated path - log warning
            logger.warning("[Router] Using deprecated accessible_source_names - please switch to accessible_document_ids")
            if len(accessible_source_names) == 0:
                logger.warning("[Router] Access control: user has NO accessible documents - returning empty")
                return []
            else:
                logger.info(f"[Router] Access control: limiting to {len(accessible_source_names)} accessible sources")

        if not ENABLE_DOCUMENT_ROUTING:
            logger.info("[Router] Document routing disabled, returning all accessible documents")
            # If access control is active, return accessible document IDs
            if accessible_document_ids is not None:
                return list(accessible_document_ids)
            return []

        if not original_query:
            logger.warning("[Router] No original query provided, returning all accessible documents")
            if accessible_document_ids is not None:
                return list(accessible_document_ids)
            return []

        try:
            # Extract query metadata
            query_entities = query_metadata.get("entities", [])
            query_topics = query_metadata.get("topics", [])
            query_doc_types = query_metadata.get("document_type_hints", [])

            logger.info(f"[Router] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"[Router] QUERY METADATA:")
            logger.info(f"[Router]   • Entities (named): {query_entities}")
            logger.info(f"[Router]   • Topics (categories): {query_topics}")
            logger.info(f"[Router]   • Document types: {query_doc_types}")
            logger.info(f"[Router] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            # Track current candidate set - starts with all accessible documents
            current_candidates = list(accessible_document_ids) if accessible_document_ids else []

            # =================================================================
            # Strategy 0: Hard entity filter (ONLY for proper named entities)
            # =================================================================
            # Only apply hard filter when we have actual named entities (not categories)
            if ENABLE_HARD_ENTITY_FILTER and query_entities:
                hard_filtered_ids, hard_filtered_names = self._hard_filter_by_entity(
                    query_entities=query_entities,
                    original_query=original_query,
                    accessible_document_ids=current_candidates
                )

                if hard_filtered_ids:
                    logger.info(
                        f"[Router] 🎯 HARD ENTITY FILTER: Found {len(hard_filtered_ids)} document(s) "
                        f"matching entities {query_entities}"
                    )
                    logger.info(f"[Router] 🎯 Hard filtered documents: {hard_filtered_names}")
                    # Narrow down candidates (don't return yet - apply topic filter too)
                    current_candidates = hard_filtered_ids
                else:
                    logger.info(
                        f"[Router] Hard entity filter found no matches for {query_entities}, "
                        f"continuing with topic filter"
                    )

            # =================================================================
            # Strategy 1: Topic-based pre-filter (for category terms)
            # =================================================================
            # Apply topic filter when we have topics OR document type hints
            # This uses ILIKE %keyword% for fuzzy matching
            if query_topics or query_doc_types:
                topic_filtered_ids, topic_filtered_names = self._topic_based_prefilter(
                    topics=query_topics,
                    document_type_hints=query_doc_types,
                    accessible_document_ids=current_candidates
                )

                if topic_filtered_ids:
                    logger.info(
                        f"[Router] 🏷️ TOPIC FILTER: Narrowed to {len(topic_filtered_ids)} document(s) "
                        f"matching topics={query_topics}, types={query_doc_types}"
                    )
                    current_candidates = topic_filtered_ids
                else:
                    # Topic filter found nothing - this might mean no documents match
                    # the category. Log but continue with vector search on original set.
                    logger.info(
                        f"[Router] Topic filter found no matches for topics={query_topics}, "
                        f"types={query_doc_types}. Continuing with vector search on {len(current_candidates)} candidates."
                    )

            # =================================================================
            # Strategy 2: Vector-based routing (semantic search)
            # =================================================================
            # If topic/entity filters narrowed candidates, use filtered set
            # Otherwise fall back to all accessible documents
            vector_search_candidates = current_candidates if current_candidates else accessible_document_ids

            logger.info(f"[Router] Vector search will use {len(vector_search_candidates) if vector_search_candidates else 0} candidate documents")

            if self.use_vector_routing and vector_search_candidates:
                vector_results, doc_id_to_name = self._route_with_vector_search(
                    query_metadata, original_query,
                    accessible_document_ids=vector_search_candidates
                )

                if len(vector_results) >= self.vector_min_results:
                    doc_names = [doc_id_to_name.get(doc_id, "unknown") for doc_id in vector_results]
                    logger.info(
                        f"[Router] 🎯 Vector routing returned {len(vector_results)} document(s): "
                        f"{vector_results}"
                    )
                    logger.info(f"[Router] 🎯 Document names: {doc_names}")
                    return vector_results
                elif vector_results:
                    # Got some results but below threshold - still use them but log
                    doc_names = [doc_id_to_name.get(doc_id, "unknown") for doc_id in vector_results]
                    logger.info(
                        f"[Router] Vector routing returned {len(vector_results)} document(s) "
                        f"(below threshold {self.vector_min_results}, using anyway): {vector_results}"
                    )
                    logger.info(f"[Router] Document names: {doc_names}")
                    return vector_results
                else:
                    logger.info("[Router] Vector routing returned no results, falling back to LLM/rule-based")

            # =================================================================
            # If topic filter found matches but vector search didn't help,
            # return the topic-filtered candidates directly
            # =================================================================
            if current_candidates and len(current_candidates) < len(accessible_document_ids or []):
                logger.info(
                    f"[Router] 🏷️ Returning {len(current_candidates)} topic-filtered candidates "
                    f"(vector search didn't narrow further)"
                )
                return current_candidates

            # Strategy 3: Fallback to LLM or rule-based scoring
            documents = self._get_documents_with_metadata()

            if not documents:
                logger.warning("[Router] No documents with metadata found, returning filtered candidates")
                # Return topic-filtered candidates if available, otherwise all accessible
                if current_candidates:
                    return current_candidates
                if accessible_document_ids is not None:
                    return list(accessible_document_ids)
                return []

            # Filter documents by current candidates (which may be topic-filtered)
            # Use current_candidates if we have topic-filtered results, otherwise use accessible_document_ids
            filter_ids = current_candidates if current_candidates else accessible_document_ids
            if filter_ids is not None:
                filter_ids_set = set(filter_ids)
                original_count = len(documents)
                documents = [doc for doc in documents if str(doc.get("id", "")) in filter_ids_set]
                if len(documents) < original_count:
                    logger.info(f"[Router] Candidate filter applied: {original_count} -> {len(documents)}")

            if not documents:
                logger.warning("[Router] No accessible documents with metadata found after access control filter")
                return []

            # Score and rank documents
            if self.use_llm_scoring and original_query:
                scored_docs = self._score_documents_llm(query_metadata, documents, original_query)
            else:
                scored_docs = self._score_documents_rule_based(query_metadata, documents)

            # Apply hybrid filtering strategy (no top_k limit)
            relevant_docs = self._apply_hybrid_filtering(scored_docs)

            if relevant_docs:
                sources = [source for source, score in relevant_docs]
                logger.info(
                    f"[Router] 🎯 Routed to {len(sources)} document(s) (fallback scoring): "
                    f"{[(s, f'{sc:.2f}') for s, sc in relevant_docs]}"
                )
                return sources
            else:
                logger.info("[Router] No documents met filtering criteria, returning filtered candidates")
                # Return topic-filtered candidates if available
                if current_candidates:
                    return current_candidates
                if accessible_document_ids is not None:
                    return list(accessible_document_ids)
                return []

        except Exception as e:
            logger.error(f"[Router] Error routing query: {e}", exc_info=True)
            if accessible_document_ids is not None:
                return list(accessible_document_ids)
            return []  # Fallback to searching all documents

    def _should_apply_hard_filter(
        self,
        query_entities: List[str],
        original_query: str
    ) -> bool:
        """
        Determine if hard entity filter should be applied.

        Hard filtering is appropriate when:
        - Query contains specific entity names (not generic terms)
        - Query uses targeting language ("list X invoices", "show X documents")

        Args:
            query_entities: List of extracted entity names
            original_query: Original query text

        Returns:
            True if hard filtering should be applied
        """
        if not query_entities:
            return False

        query_lower = original_query.lower()

        # Keywords that indicate specific entity targeting
        targeting_keywords = [
            'list', 'show', 'get', 'find', 'all', 'from', 'by',
            'vendor', 'supplier', 'company', 'customer', 'client',
            'invoice', 'receipt', 'bill', 'statement', 'order'
        ]

        # Check if query specifically targets an entity
        for entity in query_entities:
            entity_lower = entity.lower()
            # Entity must appear in query for hard filtering
            if entity_lower in query_lower:
                # Check if targeting language is used
                if any(kw in query_lower for kw in targeting_keywords):
                    logger.debug(
                        f"[Router] Hard filter enabled: entity '{entity}' found with targeting keywords"
                    )
                    return True

        return False

    def _hard_filter_by_entity(
        self,
        query_entities: List[str],
        original_query: str,
        accessible_document_ids: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Filter documents by entity match using normalized entity fields with fuzzy matching.

        This method provides precise filtering when users query for specific entities
        (e.g., "list Augment Code invoices"). It uses a hybrid approach:
        1. Fetch all accessible documents with their entity metadata from DB
        2. Apply Python-side fuzzy matching using rapidfuzz for accurate matching
        3. Use the same normalize_entity_name() function for consistency

        Matching checks (in priority order):
        1. vendor_normalized - exact match with normalized query
        2. customer_normalized - exact match with normalized query
        3. all_entities_normalized array - exact/fuzzy match
        4. vendor_name, customer_name, store_name - fuzzy match with rapidfuzz

        Args:
            query_entities: List of entity names from query
            original_query: Original query text
            accessible_document_ids: List of accessible document IDs for access control

        Returns:
            Tuple of (matched_document_ids, matched_document_names)
        """
        if not query_entities:
            return [], []

        # Check if hard filtering is appropriate for this query
        if not self._should_apply_hard_filter(query_entities, original_query):
            logger.debug("[Router] Hard filter not applicable for this query")
            return [], []

        # Normalize query entities using the same function used during extraction
        normalized_query_entities = [
            normalize_entity_name(entity) for entity in query_entities
        ]
        normalized_query_entities = [e for e in normalized_query_entities if e]  # Remove empty

        if not normalized_query_entities:
            return [], []

        logger.info(f"[Router] Hard filter: searching for entities {normalized_query_entities}")

        matched_ids = []
        matched_names = []

        try:
            # Import fuzzy matching function
            try:
                from rapidfuzz import fuzz
                fuzzy_ratio = fuzz.ratio
                logger.debug("[Router] Using rapidfuzz for entity matching")
            except ImportError:
                fuzzy_ratio = None
                logger.debug("[Router] rapidfuzz not available, using exact/substring matching only")

            with get_db_session() as db:
                # Fetch all accessible documents with their entity metadata
                # Cast text array to UUID array for proper comparison with d.id
                sql = """
                    SELECT
                        d.id::text as document_id,
                        COALESCE(dd.header_data->>'vendor_name', dd.header_data->>'store_name', d.filename) as source_display,
                        dd.header_data->>'vendor_normalized' as vendor_normalized,
                        dd.header_data->>'customer_normalized' as customer_normalized,
                        dd.header_data->>'vendor_name' as vendor_name,
                        dd.header_data->>'customer_name' as customer_name,
                        dd.header_data->>'store_name' as store_name,
                        dd.header_data->'all_entities_normalized' as all_entities_normalized
                    FROM documents d
                    LEFT JOIN documents_data dd ON dd.document_id = d.id
                    WHERE d.id = ANY(CAST(:doc_ids AS uuid[]))
                """

                doc_ids = accessible_document_ids if accessible_document_ids else []
                # Convert list to PostgreSQL array format
                doc_ids_array = '{' + ','.join(doc_ids) + '}' if doc_ids else '{}'
                result = db.execute(text(sql), {"doc_ids": doc_ids_array})

                for row in result.fetchall():
                    doc_id = row[0]
                    source_display = row[1] or "Unknown"
                    vendor_normalized = row[2] or ""
                    customer_normalized = row[3] or ""
                    vendor_name = row[4] or ""
                    customer_name = row[5] or ""
                    store_name = row[6] or ""
                    all_entities_raw = row[7]

                    # Parse all_entities_normalized JSON array
                    all_entities = []
                    if all_entities_raw:
                        try:
                            if isinstance(all_entities_raw, str):
                                import json
                                all_entities = json.loads(all_entities_raw)
                            elif isinstance(all_entities_raw, list):
                                all_entities = all_entities_raw
                        except (json.JSONDecodeError, TypeError):
                            all_entities = []

                    # Check if document matches any query entity
                    matched = False
                    match_reason = ""

                    for query_entity in normalized_query_entities:
                        # 1. Exact match on vendor_normalized (already normalized in DB)
                        if vendor_normalized and vendor_normalized.lower() == query_entity:
                            matched = True
                            match_reason = f"vendor_normalized exact: '{vendor_normalized}'"
                            break

                        # 2. Exact match on customer_normalized
                        if customer_normalized and customer_normalized.lower() == query_entity:
                            matched = True
                            match_reason = f"customer_normalized exact: '{customer_normalized}'"
                            break

                        # 3. Check all_entities_normalized array (exact match)
                        for entity in all_entities:
                            if isinstance(entity, str) and entity.lower() == query_entity:
                                matched = True
                                match_reason = f"all_entities exact: '{entity}'"
                                break
                        if matched:
                            break

                        # 4. Fuzzy match on raw name fields using rapidfuzz
                        name_fields = [
                            ("vendor_name", vendor_name),
                            ("customer_name", customer_name),
                            ("store_name", store_name),
                        ]

                        for field_name, field_value in name_fields:
                            if not field_value:
                                continue

                            # Normalize the field value the same way as query
                            field_normalized = normalize_entity_name(field_value)

                            # Exact match after normalization
                            if field_normalized == query_entity:
                                matched = True
                                match_reason = f"{field_name} normalized exact: '{field_value}' -> '{field_normalized}'"
                                break

                            # Substring match
                            if query_entity in field_normalized or field_normalized in query_entity:
                                matched = True
                                match_reason = f"{field_name} substring: '{query_entity}' in '{field_normalized}'"
                                break

                            # Fuzzy match with rapidfuzz
                            if fuzzy_ratio:
                                score = fuzzy_ratio(query_entity, field_normalized)
                                if score >= ENTITY_MATCH_THRESHOLD:
                                    matched = True
                                    match_reason = f"{field_name} fuzzy ({score:.0f}%): '{query_entity}' ~ '{field_normalized}'"
                                    break

                        if matched:
                            break

                        # 5. Fuzzy match on all_entities_normalized array
                        for entity in all_entities:
                            if not isinstance(entity, str):
                                continue
                            entity_lower = entity.lower()

                            # Substring match
                            if query_entity in entity_lower or entity_lower in query_entity:
                                matched = True
                                match_reason = f"all_entities substring: '{query_entity}' in '{entity}'"
                                break

                            # Fuzzy match
                            if fuzzy_ratio:
                                score = fuzzy_ratio(query_entity, entity_lower)
                                if score >= ENTITY_MATCH_THRESHOLD:
                                    matched = True
                                    match_reason = f"all_entities fuzzy ({score:.0f}%): '{query_entity}' ~ '{entity}'"
                                    break

                        if matched:
                            break

                    if matched:
                        matched_ids.append(doc_id)
                        matched_names.append(source_display)
                        logger.debug(f"[Router] ✓ Matched doc '{source_display}': {match_reason}")

                logger.info(
                    f"[Router] Hard filter returned {len(matched_ids)} matches "
                    f"for entities {normalized_query_entities}"
                )

        except Exception as e:
            logger.error(f"[Router] Hard entity filter error: {e}", exc_info=True)
            # Fall back to simpler in-memory filtering if main logic fails
            return self._hard_filter_by_entity_fallback(
                query_entities, accessible_document_ids
            )

        return matched_ids, matched_names

    def _hard_filter_by_entity_fallback(
        self,
        query_entities: List[str],
        accessible_document_ids: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Fallback method for hard entity filtering using in-memory matching.

        Used when main filtering method fails. Uses the same normalize_entity_name()
        function and rapidfuzz for consistent matching.
        """
        if not query_entities or not accessible_document_ids:
            return [], []

        # Normalize query entities using the same function used during extraction
        normalized_query_entities = [
            normalize_entity_name(entity) for entity in query_entities
        ]
        normalized_query_entities = [e for e in normalized_query_entities if e]

        if not normalized_query_entities:
            return [], []

        # Try to import rapidfuzz for fuzzy matching
        try:
            from rapidfuzz import fuzz
            fuzzy_ratio = fuzz.ratio
        except ImportError:
            fuzzy_ratio = None

        matched_ids = []
        matched_names = []

        try:
            with get_db_session() as db:
                repo = DocumentRepository(db)

                for doc_id in accessible_document_ids:
                    try:
                        # Get document with metadata from header_data
                        doc_data = repo.get_document_data(doc_id)
                        if not doc_data or not doc_data.header_data:
                            continue

                        header = doc_data.header_data

                        # Get display name from header_data fields
                        source_name = (
                            header.get("vendor_name") or
                            header.get("store_name") or
                            header.get("customer_name") or
                            "Unknown"
                        )

                        # Get normalized fields (already normalized in DB)
                        vendor_norm = header.get("vendor_normalized", "").lower()
                        customer_norm = header.get("customer_normalized", "").lower()
                        all_entities = header.get("all_entities_normalized", [])

                        # Get raw name fields for normalization
                        vendor_name = header.get("vendor_name", "")
                        customer_name = header.get("customer_name", "")
                        store_name = header.get("store_name", "")

                        matched = False

                        for query_entity in normalized_query_entities:
                            # 1. Exact match on pre-normalized fields
                            if query_entity == vendor_norm or query_entity == customer_norm:
                                matched = True
                                break

                            # 2. Check all_entities_normalized array
                            for entity in all_entities:
                                if isinstance(entity, str) and entity.lower() == query_entity:
                                    matched = True
                                    break
                            if matched:
                                break

                            # 3. Normalize raw fields and compare
                            for field_value in [vendor_name, customer_name, store_name]:
                                if not field_value:
                                    continue
                                field_normalized = normalize_entity_name(field_value)

                                # Exact match after normalization
                                if field_normalized == query_entity:
                                    matched = True
                                    break

                                # Substring match
                                if query_entity in field_normalized or field_normalized in query_entity:
                                    matched = True
                                    break

                                # Fuzzy match with rapidfuzz
                                if fuzzy_ratio:
                                    score = fuzzy_ratio(query_entity, field_normalized)
                                    if score >= ENTITY_MATCH_THRESHOLD:
                                        matched = True
                                        break

                            if matched:
                                break

                        if matched:
                            matched_ids.append(doc_id)
                            matched_names.append(source_name)

                    except Exception as doc_error:
                        logger.debug(f"[Router] Error checking document {doc_id}: {doc_error}")
                        continue

        except Exception as e:
            logger.error(f"[Router] Hard entity filter fallback error: {e}", exc_info=True)

        return matched_ids, matched_names

    def _topic_based_prefilter(
        self,
        topics: List[str],
        document_type_hints: List[str],
        accessible_document_ids: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Pre-filter documents by topic/category matching using ILIKE %keyword%.

        This handles queries like "meal receipts" where:
        - "meal" is a topic/category term to match against document metadata
        - "receipt" is a document type filter

        Uses fuzzy ILIKE matching to catch variations:
        - 'meal' matches 'meal', 'meals', 'meal expense', 'business meal'

        Searches in multiple fields:
        - document_metadata->>'topics' (array stored as JSON)
        - header_data->>'vendor_name', 'store_name', 'description'
        - summary_data content

        Args:
            topics: List of topic/category terms to match (e.g., ["meal", "food"])
            document_type_hints: List of document types to filter (e.g., ["receipt"])
            accessible_document_ids: List of accessible document IDs for access control

        Returns:
            Tuple of (matched_document_ids, matched_document_names)
        """
        if not topics and not document_type_hints:
            return [], []

        if not accessible_document_ids:
            return [], []

        logger.info(f"[Router] 🏷️ TOPIC PRE-FILTER: Searching for topics={topics}, doc_types={document_type_hints}")

        matched_ids = []
        matched_names = []

        try:
            with get_db_session() as db:
                # Build dynamic SQL with ILIKE for fuzzy topic matching
                # We need to check:
                # 1. document_metadata->'topics' array (JSONB)
                # 2. documents_data.header_data fields
                # 3. documents_data.schema_type for document type

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
                        # Check multiple fields for topic match
                        topic_conditions.append(f"""
                            (
                                -- Match in document_metadata topics array (cast to text for ILIKE)
                                LOWER(d.document_metadata::text) ILIKE :{param_name}
                                -- Match in header_data fields
                                OR LOWER(COALESCE(dd.header_data->>'vendor_name', '')) ILIKE :{param_name}
                                OR LOWER(COALESCE(dd.header_data->>'store_name', '')) ILIKE :{param_name}
                                OR LOWER(COALESCE(dd.header_data->>'description', '')) ILIKE :{param_name}
                                OR LOWER(COALESCE(dd.header_data->>'category', '')) ILIKE :{param_name}
                                -- Match in summary_data
                                OR LOWER(COALESCE(dd.summary_data::text, '')) ILIKE :{param_name}
                            )
                        """)
                    # Any topic match is OK (OR logic)
                    conditions.append(f"({' OR '.join(topic_conditions)})")

                # Document type filter (if specified) - exact match on schema_type
                if document_type_hints:
                    type_conditions = []
                    for j, doc_type in enumerate(document_type_hints):
                        param_name = f"doctype_{j}"
                        params[param_name] = doc_type.lower()
                        type_conditions.append(f"LOWER(dd.schema_type) = :{param_name}")
                    # Any type match is OK (OR logic)
                    conditions.append(f"({' OR '.join(type_conditions)})")

                # Build final SQL
                where_clause = " AND ".join(conditions) if conditions else "TRUE"

                sql = f"""
                    SELECT DISTINCT
                        d.id::text as document_id,
                        COALESCE(dd.header_data->>'vendor_name', dd.header_data->>'store_name', d.filename) as source_display
                    FROM documents d
                    LEFT JOIN documents_data dd ON dd.document_id = d.id
                    WHERE d.id = ANY(CAST(:doc_ids AS uuid[]))
                      AND d.deleted_at IS NULL
                      AND {where_clause}
                """

                logger.debug(f"[Router] Topic pre-filter SQL params: {list(params.keys())}")

                result = db.execute(text(sql), params)

                for row in result.fetchall():
                    doc_id = row[0]
                    source_display = row[1] or "Unknown"
                    matched_ids.append(doc_id)
                    matched_names.append(source_display)

                logger.info(
                    f"[Router] 🏷️ TOPIC PRE-FILTER: Found {len(matched_ids)} document(s) "
                    f"matching topics={topics}, doc_types={document_type_hints}"
                )
                if matched_names:
                    logger.info(f"[Router] 🏷️ Topic filtered documents: {matched_names}")

        except Exception as e:
            logger.error(f"[Router] Topic pre-filter error: {e}", exc_info=True)
            # On error, return empty - let vector search handle it
            return [], []

        return matched_ids, matched_names

    def _route_with_vector_search(
        self,
        query_metadata: Dict[str, Any],
        original_query: str,
        accessible_source_names: Optional[set] = None,  # DEPRECATED
        accessible_document_ids: Optional[List[str]] = None
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Route query using vector search on metadata embeddings.

        Uses adaptive thresholding:
        1. Get top results with base threshold
        2. Apply relative score filtering (must be >= X% of top score)
        3. Deduplicate results
        4. Filter by accessible document IDs (if provided)

        Args:
            query_metadata: Metadata extracted from query
            original_query: Original query text
            accessible_source_names: DEPRECATED - use accessible_document_ids
            accessible_document_ids: List of document IDs the user can access.
                                     If provided, only documents with these IDs will be returned.

        Returns:
            Tuple of:
            - List of document IDs (UUIDs as strings) from vector search results
            - Dict mapping document_id -> source_name for logging
        """
        try:
            from .vectorstore import (
                search_document_metadata,
                format_query_for_metadata_search
            )

            # Format query for metadata search
            query_text = format_query_for_metadata_search(
                query_text=original_query,
                entities=query_metadata.get("entities", []),
                topics=query_metadata.get("topics", []),
                document_type_hints=query_metadata.get("document_type_hints", [])
            )

            # Log the full formatted query for debugging
            logger.info(f"[Router Vector] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"[Router Vector] METADATA QUERY FORMATTING:")
            logger.info(f"[Router Vector]   • Original query: \"{original_query}\"")
            logger.info(f"[Router Vector]   • Entities: {query_metadata.get('entities', [])}")
            logger.info(f"[Router Vector]   • Topics: {query_metadata.get('topics', [])}")
            logger.info(f"[Router Vector]   • Doc type hints: {query_metadata.get('document_type_hints', [])}")
            logger.info(f"[Router Vector]   • Formatted query ({len(query_text)} chars):")
            logger.info(f"[Router Vector]     \"{query_text}\"")
            logger.info(f"[Router Vector] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            # Search metadata collection with access control filtering at Qdrant level
            results = search_document_metadata(
                query_text,
                accessible_document_ids=accessible_document_ids
            )

            if not results:
                logger.debug("[Router Vector] No results from metadata vector search")
                return [], {}

            # ========== ENTITY MATCHING BOOST ==========
            # Apply score boost when query entities match document entities/source names
            # This compensates for embedding model limitations with specific terms
            # IMPORTANT: Only boost for actual entity matches, NOT generic query terms
            import re

            def normalize_for_matching(text: str) -> str:
                """Normalize text for fuzzy matching: lowercase, remove special chars, collapse spaces."""
                text = text.lower().strip()
                text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special chars except spaces
                text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
                return text

            query_entities = [normalize_for_matching(e) for e in query_metadata.get("entities", [])]

            ENTITY_MATCH_BOOST = float(os.getenv("ENTITY_MATCH_BOOST", "0.3"))

            # Track the highest score after boosting (before penalties) for threshold calculation
            max_boosted_score = 0.0

            # Track documents that got TRUE entity matches (not just generic term matches)
            entity_matched_doc_ids = set()

            if query_entities:
                logger.info(f"[Router Vector] ENTITY MATCHING BOOST:")
                logger.info(f"[Router Vector]   • Query entities (normalized): {query_entities}")
                logger.info(f"[Router Vector]   • Boost factor: {ENTITY_MATCH_BOOST}")

                for result in results:
                    source_name_raw = result.get("source_name", "")
                    source_name = normalize_for_matching(source_name_raw)
                    # Also create a variant without spaces for matching "graph r1" with "graphr1"
                    source_name_compact = source_name.replace(" ", "")

                    # Get document entities from payload
                    payload = result.get("payload", {})
                    embedding_text = normalize_for_matching(payload.get("embedding_text", ""))
                    doc_id = result.get("payload", {}).get("document_id") or result.get("document_id", "")

                    original_score = result.get("score", 0)
                    boost = 0.0
                    match_reasons = []
                    is_entity_match = False  # True only if matched actual query entity

                    # Helper function to check if two strings share a significant prefix
                    def shares_prefix(s1: str, s2: str, min_len: int = 4) -> bool:
                        """Check if two strings share a common prefix of at least min_len chars."""
                        max_prefix = min(len(s1), len(s2))
                        for i in range(max_prefix, min_len - 1, -1):
                            if s1[:i] == s2[:i]:
                                return True
                        return False

                    # Check if any query entity appears in the document
                    for entity in query_entities:
                        entity_compact = entity.replace(" ", "")
                        # Also get significant words from entity (3+ chars)
                        entity_words = [w for w in entity.split() if len(w) >= 3]

                        # Check source name (with various matching strategies)
                        if entity in source_name or source_name in entity:
                            boost = max(boost, ENTITY_MATCH_BOOST)
                            match_reasons.append(f"entity '{entity}' in source")
                            is_entity_match = True
                        elif entity_compact in source_name_compact or source_name_compact in entity_compact:
                            boost = max(boost, ENTITY_MATCH_BOOST)
                            match_reasons.append(f"entity '{entity}' ~matches source")
                            is_entity_match = True
                        # Check if significant entity words appear in source (handles typos like "graphy" matching "graph")
                        elif any(w in source_name_compact or source_name_compact in w for w in entity_words if len(w) >= 4):
                            boost = max(boost, ENTITY_MATCH_BOOST * 0.9)
                            match_reasons.append(f"entity word from '{entity}' in source")
                            is_entity_match = True
                        # Check prefix matching for entity words (handles "graphy" matching "graphr1" via "graph" prefix)
                        elif any(shares_prefix(w, source_name_compact) for w in entity_words if len(w) >= 4):
                            boost = max(boost, ENTITY_MATCH_BOOST * 0.85)
                            match_reasons.append(f"entity '{entity}' prefix-matches source")
                            is_entity_match = True
                        # Check embedding text (contains entities, topics, summary)
                        elif entity in embedding_text or entity_compact in embedding_text.replace(" ", ""):
                            boost = max(boost, ENTITY_MATCH_BOOST * 0.8)
                            match_reasons.append(f"entity '{entity}' in content")
                            is_entity_match = True
                        else:
                            # Check metadata fields like vendor_name, customer_name, store_name
                            entity_fields = ['vendor_name', 'customer_name', 'store_name', 'company_name', 'subject_name']
                            field_matched = False
                            for field in entity_fields:
                                field_value = normalize_for_matching(str(payload.get(field, "")))
                                field_compact = field_value.replace(" ", "")
                                if field_value and (entity in field_value or entity_compact in field_compact):
                                    boost = max(boost, ENTITY_MATCH_BOOST)
                                    match_reasons.append(f"entity '{entity}' in {field}")
                                    is_entity_match = True
                                    field_matched = True
                                    break

                            # Also check key_entities list from metadata extraction
                            if not field_matched:
                                key_entities_list = payload.get("key_entities", [])
                                if key_entities_list:
                                    for ke in key_entities_list:
                                        ke_normalized = normalize_for_matching(str(ke))
                                        ke_compact = ke_normalized.replace(" ", "")
                                        if entity in ke_normalized or entity_compact in ke_compact:
                                            boost = max(boost, ENTITY_MATCH_BOOST * 0.9)
                                            match_reasons.append(f"entity '{entity}' in key_entities")
                                            is_entity_match = True
                                            break

                    # NOTE: We no longer boost for generic query terms like "invoice"
                    # When entities are specified, only entity matches matter

                    if boost > 0 and is_entity_match:
                        result["score"] = min(1.0, original_score + boost)
                        result["_boosted"] = True
                        result["_entity_matched"] = True
                        result["_boost_amount"] = boost
                        entity_matched_doc_ids.add(doc_id)
                        # Track max boosted score BEFORE penalties are applied
                        max_boosted_score = max(max_boosted_score, result["score"])
                        logger.info(
                            f"[Router Vector]   ↑ BOOSTED: {source_name_raw:40s} "
                            f"{original_score:.4f} → {result['score']:.4f} (+{boost:.4f}) "
                            f"[{', '.join(match_reasons)}]"
                        )

                # Re-sort by score after boosting
                results.sort(key=lambda x: x.get("score", 0), reverse=True)

                # Also track unboosted top score in case no boosting occurred
                if max_boosted_score == 0.0 and results:
                    max_boosted_score = results[0].get("score", 0)

                # ========== ENTITY NON-MATCH PENALTY ==========
                # When query explicitly mentions entity names (like vendor/customer/company names),
                # penalize documents that DON'T match these entities to improve filtering precision.
                # This is more discriminating than just boosting matching documents.
                # IMPORTANT: Documents that only match generic terms (not entities) are penalized!

                # Penalty factor for documents that don't match any query entity
                ENTITY_NONMATCH_PENALTY = float(os.getenv("ENTITY_NONMATCH_PENALTY", "0.4"))

                # Count how many documents got TRUE entity matches
                entity_matched_count = len(entity_matched_doc_ids)

                # Only apply penalty if SOME documents got entity matches (not a general query)
                if entity_matched_count > 0:
                    logger.info(f"[Router Vector] ENTITY NON-MATCH PENALTY:")
                    logger.info(f"[Router Vector]   • Entity-matched documents: {entity_matched_count}")
                    logger.info(f"[Router Vector]   • Penalty factor: {ENTITY_NONMATCH_PENALTY}")

                    for result in results:
                        # Penalize if document did NOT match any entity
                        if not result.get("_entity_matched", False):
                            original_score = result.get("score", 0)
                            # Apply penalty - multiply score by penalty factor
                            result["score"] = original_score * ENTITY_NONMATCH_PENALTY
                            result["_entity_penalized"] = True
                            source_name_raw = result.get("source_name", "")
                            logger.info(
                                f"[Router Vector]   ↓ PENALIZED: {source_name_raw:40s} "
                                f"{original_score:.4f} → {result['score']:.4f} (×{ENTITY_NONMATCH_PENALTY}) "
                                f"[no entity match]"
                            )

                    # Re-sort after penalty
                    results.sort(key=lambda x: x.get("score", 0), reverse=True)

            # ========== TABULAR DATA BOOST ==========
            # Boost tabular documents (CSV/Excel) when query appears to be data analytics
            # This ensures tabular data sources are found for aggregate queries
            TABULAR_ANALYTICS_KEYWORDS = {
                'max', 'min', 'maximum', 'minimum', 'average', 'avg', 'sum', 'total',
                'count', 'highest', 'lowest', 'top', 'bottom', 'most', 'least',
                'inventory', 'stock', 'product', 'products', 'items', 'records',
                'price', 'prices', 'quantity', 'quantities', 'sales', 'revenue',
                'list', 'find', 'show', 'display', 'get', 'filter', 'search',
                'data', 'dataset', 'spreadsheet', 'csv', 'excel', 'table'
            }
            query_lower = original_query.lower()
            query_words = set(query_lower.split())

            analytics_keyword_count = len(query_words.intersection(TABULAR_ANALYTICS_KEYWORDS))

            if analytics_keyword_count >= 2:  # At least 2 analytics-related keywords
                TABULAR_DATA_BOOST = float(os.getenv("TABULAR_DATA_BOOST", "0.35"))
                logger.info(f"[Router Vector] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                logger.info(f"[Router Vector] TABULAR DATA BOOST:")
                logger.info(f"[Router Vector]   • Analytics keywords found: {analytics_keyword_count}")
                logger.info(f"[Router Vector]   • Matching keywords: {query_words.intersection(TABULAR_ANALYTICS_KEYWORDS)}")
                logger.info(f"[Router Vector]   • Boost factor: {TABULAR_DATA_BOOST}")

                for result in results:
                    # Check if document is tabular (from payload or top-level field)
                    payload = result.get("payload", {})
                    is_tabular = result.get("is_tabular") or payload.get("is_tabular", False)

                    if is_tabular:
                        source_name = result.get("source_name", "")
                        original_score = result.get("score", 0)
                        boost = TABULAR_DATA_BOOST

                        # Apply boost
                        result["score"] = min(1.0, original_score + boost)
                        result["_tabular_boosted"] = True
                        result["_tabular_boost_amount"] = boost

                        # Update max boosted score
                        max_boosted_score = max(max_boosted_score, result["score"])

                        # Get column info for logging
                        columns = payload.get("columns", [])
                        columns_str = ", ".join(columns[:5]) + ("..." if len(columns) > 5 else "")

                        logger.info(
                            f"[Router Vector]   ↑ TABULAR BOOST: {source_name:40s} "
                            f"{original_score:.4f} → {result['score']:.4f} (+{boost:.4f}) "
                            f"[columns: {columns_str}]"
                        )

                # Re-sort after tabular boost
                results.sort(key=lambda x: x.get("score", 0), reverse=True)

            # ========== DOCUMENT TYPES MATCHING BOOST ==========
            # Boost documents whose document_types match query hints
            # This rewards documents that are semantically classified to match the query
            doc_type_hints = [t.lower() for t in query_metadata.get("document_type_hints", [])]
            if doc_type_hints:
                DOC_TYPE_MATCH_BOOST = float(os.getenv("DOC_TYPE_MATCH_BOOST", "0.25"))
                DOC_TYPE_MISMATCH_PENALTY = float(os.getenv("DOC_TYPE_MISMATCH_PENALTY", "0.5"))
                # Expand type hints to include aliases (e.g., "technical_doc" -> includes "report", "research_paper")
                expanded_hints = expand_type_aliases(doc_type_hints)

                # Generic types that should get reduced boost (too broad to be meaningful)
                GENERIC_TYPES = {"report", "spreadsheet", "document", "other", "data"}

                logger.info(f"[Router Vector] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                logger.info(f"[Router Vector] DOCUMENT TYPES MATCHING:")
                logger.info(f"[Router Vector]   • Query type hints: {doc_type_hints}")
                logger.info(f"[Router Vector]   • Expanded types (with aliases): {sorted(expanded_hints)}")
                logger.info(f"[Router Vector]   • Match boost: +{DOC_TYPE_MATCH_BOOST}")
                logger.info(f"[Router Vector]   • Mismatch penalty: ×{DOC_TYPE_MISMATCH_PENALTY}")

                for result in results:
                    # Get document_types list
                    payload = result.get("payload", {})
                    doc_types = result.get("document_types") or payload.get("document_types", [])
                    doc_types = [t.lower() for t in doc_types] if doc_types else []

                    source_name = result.get("source_name", "")
                    original_score = result.get("score", 0)

                    # Build document metadata for content-based type inference (fallback)
                    doc_metadata = {
                        "is_tabular": result.get("is_tabular") or payload.get("is_tabular", False),
                        "columns": result.get("columns") or payload.get("columns", []),
                        "schema_type": result.get("schema_type") or payload.get("schema_type", ""),
                        "document_types": doc_types,
                    }

                    # Calculate match score based on how many document types match query hints
                    matching_types = set(doc_types).intersection(expanded_hints)

                    if matching_types:
                        # Separate specific matches from generic matches
                        specific_matches = matching_types - GENERIC_TYPES
                        generic_matches = matching_types.intersection(GENERIC_TYPES)

                        # Check if document also has receipt type (conflicts with inventory queries)
                        has_receipt = "receipt" in doc_types
                        query_is_inventory = "inventory_report" in doc_type_hints or "inventory" in doc_type_hints

                        # If document is a receipt but query is about inventory, penalize instead of boost
                        if has_receipt and query_is_inventory and not specific_matches:
                            # Only generic match (like 'report') but document is actually a receipt
                            result["score"] = original_score * DOC_TYPE_MISMATCH_PENALTY
                            result["_type_penalized"] = True
                            logger.info(
                                f"[Router Vector]   ↓ PENALIZED: {source_name:40s} "
                                f"types={doc_types} is receipt, not inventory "
                                f"({original_score:.4f} → {result['score']:.4f})"
                            )
                            continue

                        # BOOST: Calculate boost based on match quality
                        if specific_matches:
                            # Specific type match (e.g., inventory_report) - full boost
                            match_count = len(specific_matches)
                            boost = DOC_TYPE_MATCH_BOOST * min(match_count, 2)  # Cap at 2x boost
                        else:
                            # Only generic type match (e.g., report) - reduced boost
                            boost = DOC_TYPE_MATCH_BOOST * 0.3  # 30% of normal boost for generic

                        result["score"] = min(1.0, original_score + boost)
                        result["_type_boosted"] = True
                        result["_type_boost_amount"] = boost
                        max_boosted_score = max(max_boosted_score, result["score"])

                        match_type_label = "specific" if specific_matches else "generic"
                        logger.info(
                            f"[Router Vector]   ↑ TYPE BOOST ({match_type_label}): {source_name:40s} "
                            f"types={doc_types} matches {list(matching_types)} "
                            f"({original_score:.4f} → {result['score']:.4f} +{boost:.4f})"
                        )
                    else:
                        # Use centralized type matching for fallback (content-based inference, aliases)
                        type_matches_result = types_match(doc_type_hints, doc_types, doc_metadata)

                        if type_matches_result:
                            # Matched via aliases or content-based inference - smaller boost
                            boost = DOC_TYPE_MATCH_BOOST * 0.5
                            result["score"] = min(1.0, original_score + boost)
                            max_boosted_score = max(max_boosted_score, result["score"])
                            logger.info(
                                f"[Router Vector]   ↑ TYPE INFER: {source_name:40s} "
                                f"types={doc_types} inferred match "
                                f"({original_score:.4f} → {result['score']:.4f} +{boost:.4f})"
                            )
                        elif doc_types:
                            # Apply penalty for type mismatch
                            result["score"] = original_score * DOC_TYPE_MISMATCH_PENALTY
                            result["_type_penalized"] = True
                            logger.info(
                                f"[Router Vector]   ↓ PENALIZED: {source_name:40s} "
                                f"types={doc_types} not in {list(expanded_hints)} "
                                f"({original_score:.4f} → {result['score']:.4f})"
                            )

                # Re-sort after type boost/penalty
                results.sort(key=lambda x: x.get("score", 0), reverse=True)

            # ========== ADAPTIVE THRESHOLD CALCULATION ==========
            # Log all raw results first for debugging
            logger.info(f"[Router Vector] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"[Router Vector] RESULTS AFTER BOOSTING & PENALTIES: {len(results)} results")
            logger.info(f"[Router Vector] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            for i, r in enumerate(results):
                # Get embedding text from payload for comparison
                payload = r.get("payload", {})
                embedding_text = payload.get("embedding_text", "N/A")

                logger.info(
                    f"[Router Vector]   [{i+1}] {r.get('source_name', 'N/A'):40s} "
                    f"score={r.get('score', 0):.4f}  type={r.get('document_type', 'N/A')}"
                )

                # Show stored embedding text for comparison (verbose mode or debug)
                if METADATA_VERBOSE_LOGGING:
                    logger.info(f"[Router Vector]       Stored embedding ({len(embedding_text)} chars):")
                    logger.info(f"[Router Vector]       \"{embedding_text}\"")
                else:
                    # Truncate for display in debug mode
                    embedding_preview = embedding_text[:100] + "..." if len(embedding_text) > 100 else embedding_text
                    logger.debug(f"[Router Vector]       Stored: \"{embedding_preview}\"")

            # Calculate adaptive threshold using the max boosted score (before penalties)
            # This ensures documents that got entity matching boost set a higher threshold
            # to filter out irrelevant documents that were never boosted
            top_score_after_penalty = results[0].get("score", 0) if results else 0

            # Use max_boosted_score if available (higher threshold), otherwise fall back to top_score
            # This is the key fix: threshold is based on PRE-PENALTY boosted score
            if max_boosted_score > 0:
                threshold_base_score = max_boosted_score
            else:
                threshold_base_score = top_score_after_penalty

            relative_threshold = threshold_base_score * VECTOR_ROUTING_RELATIVE_THRESHOLD

            logger.info(f"[Router Vector] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"[Router Vector] ADAPTIVE THRESHOLD CALCULATION:")
            logger.info(f"[Router Vector]   • Max boosted score (pre-penalty): {max_boosted_score:.4f}")
            logger.info(f"[Router Vector]   • Top score (post-penalty):        {top_score_after_penalty:.4f}")
            logger.info(f"[Router Vector]   • Threshold base score:            {threshold_base_score:.4f}")
            logger.info(f"[Router Vector]   • Relative threshold (%):          {VECTOR_ROUTING_RELATIVE_THRESHOLD:.0%}")
            logger.info(f"[Router Vector]   • Computed threshold:              {relative_threshold:.4f}")
            logger.info(f"[Router Vector]   • Formula: {threshold_base_score:.4f} × {VECTOR_ROUTING_RELATIVE_THRESHOLD} = {relative_threshold:.4f}")
            logger.info(f"[Router Vector] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            # Extract and deduplicate document IDs with adaptive filtering
            seen_doc_ids = set()
            document_ids = []
            doc_id_to_name = {}  # Track document_id -> source_name mapping for logging
            filtered_count = 0
            duplicate_count = 0
            entity_filtered_count = 0

            # Build set of accessible document IDs for fast lookup
            accessible_ids_set = set(accessible_document_ids) if accessible_document_ids else None

            # IMPORTANT: When entity matches exist, ONLY return entity-matched documents
            # This prevents non-matching documents from being included via relative threshold
            has_entity_matches = len(entity_matched_doc_ids) > 0
            if has_entity_matches:
                logger.info(f"[Router Vector] ENTITY-PRIORITY FILTERING MODE:")
                logger.info(f"[Router Vector]   • Entity-matched documents found: {len(entity_matched_doc_ids)}")
                logger.info(f"[Router Vector]   • Only entity-matched documents will be returned")

            logger.info(f"[Router Vector] FILTERING RESULTS:")
            for result in results:
                # Get document_id from payload (same as point ID in metadata collection)
                doc_id = result.get("payload", {}).get("document_id") or result.get("document_id")
                source_name = result.get("source_name", "unknown")
                score = result.get("score", 0)
                is_entity_matched = result.get("_entity_matched", False)

                if not doc_id:
                    logger.debug(f"[Router Vector]   ⊘ SKIP: {source_name} - no document_id")
                    continue

                # Apply access control FIRST - skip if not in accessible list
                if accessible_ids_set is not None and doc_id not in accessible_ids_set:
                    logger.debug(
                        f"[Router Vector]   ✗ NO ACCESS: {source_name} ({doc_id[:8]}...)"
                    )
                    continue

                # ENTITY PRIORITY: When entity matches exist, only accept entity-matched documents
                if has_entity_matches and not is_entity_matched:
                    entity_filtered_count += 1
                    logger.info(
                        f"[Router Vector]   ✗ NO ENTITY MATCH: {source_name:40s} "
                        f"score={score:.4f} [entity-matched docs exist, skipping non-matched]"
                    )
                    continue

                # Skip if below relative threshold (only applies when no entity matches exist)
                if score < relative_threshold:
                    filtered_count += 1
                    logger.info(
                        f"[Router Vector]   ✗ FILTERED: {source_name:40s} "
                        f"score={score:.4f} < threshold={relative_threshold:.4f}"
                    )
                    continue

                # Deduplicate
                if doc_id in seen_doc_ids:
                    duplicate_count += 1
                    logger.debug(
                        f"[Router Vector]   ⊘ DUPLICATE: {source_name:40s} (already added)"
                    )
                    continue

                seen_doc_ids.add(doc_id)
                document_ids.append(doc_id)
                doc_id_to_name[doc_id] = source_name  # Store mapping for final log
                entity_tag = " [ENTITY✓]" if is_entity_matched else ""
                logger.info(
                    f"[Router Vector]   ✓ ACCEPTED: {source_name:40s} ({doc_id[:8]}...) "
                    f"score={score:.4f}{entity_tag}"
                )

            # Summary
            logger.info(f"[Router Vector] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"[Router Vector] FILTERING SUMMARY:")
            logger.info(f"[Router Vector]   • Total raw results:    {len(results)}")
            logger.info(f"[Router Vector]   • Accepted (unique):    {len(document_ids)}")
            if entity_filtered_count > 0:
                logger.info(f"[Router Vector]   • Filtered (no entity): {entity_filtered_count}")
            logger.info(f"[Router Vector]   • Filtered (low score): {filtered_count}")
            logger.info(f"[Router Vector]   • Skipped (duplicates): {duplicate_count}")
            if accessible_ids_set:
                logger.info(f"[Router Vector]   • Access control:       {len(accessible_ids_set)} accessible IDs")
            if has_entity_matches:
                logger.info(f"[Router Vector]   • Entity-priority mode: YES (only entity-matched docs returned)")
            # Format final document list with names for easy tracking
            doc_names_list = [doc_id_to_name.get(doc_id, "unknown") for doc_id in document_ids]
            logger.info(f"[Router Vector]   • Final document_ids: {document_ids}")
            logger.info(f"[Router Vector]   • Final document_names: {doc_names_list}")
            logger.info(f"[Router Vector] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            return document_ids, doc_id_to_name

        except ImportError as e:
            logger.warning(f"[Router Vector] Vector search not available: {e}")
            return [], {}
        except Exception as e:
            logger.error(f"[Router Vector] Error in vector search: {e}", exc_info=True)
            return [], {}
    
    def _apply_hybrid_filtering(
        self,
        scored_docs: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Apply hybrid filtering strategy to remove low-quality documents.

        Filters applied in sequence:
        1. Absolute minimum score threshold
        2. Score ratio filter (must be >= X% of top score)
        3. Score gap filter (must be within Y points of top score) - only for rule-based

        Args:
            scored_docs: List of (source_name, score) tuples, sorted descending

        Returns:
            Filtered list of (source_name, score) tuples (all matching documents, no top_k limit)
        """
        if not scored_docs:
            return []

        # Use different thresholds for LLM vs rule-based scoring
        if self.use_llm_scoring:
            min_threshold = self.llm_min_score
            ratio = self.llm_score_ratio
            use_gap_filter = False  # LLM scores are already on 0-10 scale, gap filter not needed
        else:
            min_threshold = self.min_score
            ratio = self.score_ratio
            use_gap_filter = True

        # Step 1: Filter by absolute minimum score
        filtered = [
            (source, score) for source, score in scored_docs
            if score >= min_threshold
        ]

        if not filtered:
            logger.debug(f"[Router] No documents met minimum score threshold ({min_threshold})")
            return []

        # Get top score for ratio and gap calculations
        top_score = filtered[0][1]

        # Step 2: Filter by score ratio (must be at least X% of top score)
        ratio_threshold = top_score * ratio
        before_ratio = len(filtered)
        filtered = [
            (source, score) for source, score in filtered
            if score >= ratio_threshold
        ]

        if len(filtered) < before_ratio:
            removed = before_ratio - len(filtered)
            removed_docs = scored_docs[len(filtered):before_ratio]
            logger.info(
                f"[Router] Filtered out {removed} low-score document(s) by ratio filter "
                f"(threshold: {ratio_threshold:.2f}, {ratio*100:.0f}% of top): "
                f"{[(s, f'{sc:.2f}') for s, sc in removed_docs]}"
            )

        # Step 3: Filter by score gap (only for rule-based scoring)
        if use_gap_filter:
            gap_threshold = top_score - self.max_score_gap
            before_gap = len(filtered)
            filtered = [
                (source, score) for source, score in filtered
                if score >= gap_threshold
            ]

            if len(filtered) < before_gap:
                removed = before_gap - len(filtered)
                removed_docs = scored_docs[len(filtered):before_gap]
                logger.info(
                    f"[Router] Filtered out {removed} document(s) by score gap filter "
                    f"(threshold: {gap_threshold:.2f}, max gap: {self.max_score_gap:.1f}): "
                    f"{[(s, f'{sc:.2f}') for s, sc in removed_docs]}"
                )

        # Return all filtered documents (no top_k limit)
        return filtered

    def _get_documents_with_metadata(self) -> List[Dict[str, Any]]:
        """Get all documents that have metadata extracted."""
        try:
            with get_db_session() as db:
                repo = DocumentRepository(db)
                docs = repo.get_all_with_metadata()
                
                result = []
                for doc in docs:
                    if doc.document_metadata:
                        # Extract source name (filename without extension)
                        source_name = doc.filename.rsplit('.', 1)[0] if '.' in doc.filename else doc.filename
                        result.append({
                            "source_name": source_name,
                            "metadata": doc.document_metadata,
                            "filename": doc.filename,
                        })
                
                logger.debug(f"[Router] Found {len(result)} documents with metadata")
                return result
                
        except Exception as e:
            logger.error(f"[Router] Error fetching documents: {e}", exc_info=True)
            return []
    
    def _score_documents_rule_based(
        self,
        query_metadata: Dict[str, Any],
        documents: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """
        Score documents based on rule-based metadata matching.

        Returns:
            List of (source_name, score) tuples, sorted by score descending
        """
        scored = []

        for doc in documents:
            score = self._calculate_match_score_rule_based(query_metadata, doc["metadata"])
            scored.append((doc["source_name"], score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def _score_documents_llm(
        self,
        query_metadata: Dict[str, Any],
        documents: List[Dict[str, Any]],
        original_query: str
    ) -> List[Tuple[str, float]]:
        """
        Score documents using LLM-based relevance scoring (BATCH OPTIMIZED).

        Returns:
            List of (source_name, score) tuples, sorted by score descending
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from rag_service.graph_rag.prompts import DOCUMENT_RELEVANCE_BATCH_SCORING_PROMPT

        # Try batch scoring first (much faster - 1 LLM call instead of N calls)
        try:
            return self._score_documents_llm_batch(query_metadata, documents, original_query)
        except Exception as e:
            logger.warning(f"[Router LLM] Batch scoring failed: {e}, falling back to individual scoring")
            # Fallback to individual scoring
            return self._score_documents_llm_individual(query_metadata, documents, original_query)

    def _score_documents_llm_batch(
        self,
        query_metadata: Dict[str, Any],
        documents: List[Dict[str, Any]],
        original_query: str
    ) -> List[Tuple[str, float]]:
        """
        Score ALL documents in a single LLM call (OPTIMIZED).

        Returns:
            List of (source_name, score) tuples, sorted by score descending
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from rag_service.graph_rag.prompts import DOCUMENT_RELEVANCE_BATCH_SCORING_PROMPT

        # Get LangChain chat model from LLM service
        llm_model = self.llm_service.get_query_model(temperature=0.1, num_predict=2048)

        # Format all documents for batch scoring
        documents_list = []
        for i, doc in enumerate(documents, 1):
            doc_meta = doc["metadata"]
            doc_subject = doc_meta.get("subject_name", "Unknown")
            # Use document_types list (get primary type for display)
            doc_types = doc_meta.get("document_types", [])
            doc_type = doc_types[0] if doc_types else "other"
            doc_summary = doc_meta.get("summary", "No summary available")
            doc_topics = ", ".join(doc_meta.get("topics", []))

            # Get key entities (top 5)
            key_entities = doc_meta.get("key_entities", [])[:5]
            doc_entities = ", ".join([f"{e.get('name')} ({e.get('type')})" for e in key_entities])

            doc_info = f"""
Document {i}:
- Filename: {doc["filename"]}
- Subject: {doc_subject}
- Document Type: {doc_type}
- Summary: {doc_summary[:200]}...
- Topics: {doc_topics}
- Key Entities: {doc_entities}
"""
            documents_list.append(doc_info)

        # Query metadata
        query_entities = ", ".join(query_metadata.get("entities", []))
        query_topics = ", ".join(query_metadata.get("topics", []))
        query_doc_types = ", ".join(query_metadata.get("document_type_hints", []))

        # Create prompt and chain
        prompt = ChatPromptTemplate.from_template(DOCUMENT_RELEVANCE_BATCH_SCORING_PROMPT)
        chain = prompt | llm_model | StrOutputParser()

        # Single LLM call for all documents
        logger.info(f"[Router LLM] Batch scoring {len(documents)} documents in 1 LLM call...")
        response = chain.invoke({
            "query": original_query,
            "query_entities": query_entities,
            "query_topics": query_topics,
            "query_doc_types": query_doc_types,
            "documents_list": "\n".join(documents_list),
            "num_documents": len(documents),
        })

        # Parse JSON response
        response_clean = response.strip()
        if response_clean.startswith("```"):
            lines = response_clean.split("\n")
            response_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else response_clean

        results = json.loads(response_clean)

        # Build scored list
        scored = []
        for i, doc in enumerate(documents):
            if i < len(results):
                score = float(results[i].get("score", 0.0))
                reasoning = results[i].get("reasoning", "")
                scored.append((doc["source_name"], score))
                logger.debug(f"[Router LLM] {doc['source_name']}: {score:.2f} - {reasoning}")
            else:
                logger.warning(f"[Router LLM] Missing score for {doc['source_name']}, using 0.0")
                scored.append((doc["source_name"], 0.0))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"[Router LLM] ✅ Batch scoring complete: {len(scored)} documents scored")
        return scored

    def _score_documents_llm_individual(
        self,
        query_metadata: Dict[str, Any],
        documents: List[Dict[str, Any]],
        original_query: str
    ) -> List[Tuple[str, float]]:
        """
        Score documents individually (FALLBACK - slower but more reliable).

        Returns:
            List of (source_name, score) tuples, sorted by score descending
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from rag_service.graph_rag.prompts import DOCUMENT_RELEVANCE_SCORING_PROMPT

        scored = []

        # Get LangChain chat model from LLM service
        llm_model = self.llm_service.get_query_model(temperature=0.1, num_predict=256)

        prompt = ChatPromptTemplate.from_template(DOCUMENT_RELEVANCE_SCORING_PROMPT)
        chain = prompt | llm_model | StrOutputParser()

        logger.info(f"[Router LLM] Individual scoring {len(documents)} documents ({len(documents)} LLM calls)...")
        for doc in documents:
            try:
                score = self._calculate_match_score_llm(
                    query_metadata,
                    doc["metadata"],
                    doc["filename"],
                    original_query,
                    chain
                )
                scored.append((doc["source_name"], score))
                logger.debug(f"[Router LLM] {doc['source_name']}: {score:.2f}")
            except Exception as e:
                logger.error(f"[Router LLM] Failed to score {doc['source_name']}: {e}")
                # Fallback to rule-based scoring
                score = self._calculate_match_score_rule_based(query_metadata, doc["metadata"])
                scored.append((doc["source_name"], score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _calculate_match_score_rule_based(
        self,
        query_meta: Dict[str, Any],
        doc_meta: Dict[str, Any]
    ) -> float:
        """
        Calculate match score between query metadata and document metadata.

        Enhanced scoring using multiple metadata fields:
        - Subject name exact match: +10.0
        - Subject name partial match: +5.0
        - Key entity match: +3.0 per entity
        - Topic match: +2.0 per topic
        - Summary/meta_summary keyword match: +1.5 per keyword
        - Document type match: +1.0
        - Confidence multiplier: score * confidence
        """
        score = 0.0

        # Handle None metadata
        if doc_meta is None:
            doc_meta = {}

        query_entities = [e.lower() for e in query_meta.get("entities", [])]
        query_topics = set(t.lower() for t in query_meta.get("topics", []))

        # 1. Subject name matching (highest priority)
        doc_subject = (doc_meta.get("subject_name") or "").lower()

        if doc_subject:
            for entity in query_entities:
                if entity in doc_subject or doc_subject in entity:
                    # Exact or substring match
                    if entity == doc_subject:
                        score += 10.0  # Exact match
                        logger.debug(f"[Router] Subject exact match: '{entity}' = '{doc_subject}' (+10.0)")
                    else:
                        score += 5.0   # Partial match
                        logger.debug(f"[Router] Subject partial match: '{entity}' ~ '{doc_subject}' (+5.0)")

        # 2. Key entities overlap
        doc_entities = set(
            e["name"].lower()
            for e in doc_meta.get("key_entities", [])
        )

        for query_entity in query_entities:
            for doc_entity in doc_entities:
                if query_entity in doc_entity or doc_entity in query_entity:
                    score += 3.0
                    logger.debug(f"[Router] Entity match: '{query_entity}' ~ '{doc_entity}' (+3.0)")

        # 3. Topics overlap
        doc_topics = set(t.lower() for t in doc_meta.get("topics", []))

        for query_topic in query_topics:
            for doc_topic in doc_topics:
                # Allow partial matching (e.g., "cloud" matches "cloud computing")
                if query_topic in doc_topic or doc_topic in query_topic:
                    score += 2.0
                    logger.debug(f"[Router] Topic match: '{query_topic}' ~ '{doc_topic}' (+2.0)")

        # 4. Summary and meta_summary keyword matching (NEW)
        # Combine summary and meta_summary for comprehensive text matching
        summary_text = (doc_meta.get("summary") or "").lower()
        meta_summary = ""
        if "hierarchical_summary" in doc_meta and doc_meta["hierarchical_summary"]:
            meta_summary = (doc_meta["hierarchical_summary"].get("meta_summary") or "").lower()

        combined_summary = f"{summary_text} {meta_summary}"

        if combined_summary.strip():
            # Check for query entity matches in summary
            for entity in query_entities:
                # Split multi-word entities for better matching
                entity_words = entity.split()
                if len(entity_words) == 1:
                    # Single word - check for whole word match
                    if f" {entity} " in f" {combined_summary} " or combined_summary.startswith(entity) or combined_summary.endswith(entity):
                        score += 1.5
                        logger.debug(f"[Router] Summary entity match: '{entity}' (+1.5)")
                else:
                    # Multi-word entity - check for phrase match
                    if entity in combined_summary:
                        score += 1.5
                        logger.debug(f"[Router] Summary entity match: '{entity}' (+1.5)")

            # Check for query topic matches in summary
            for topic in query_topics:
                topic_words = topic.split()
                if len(topic_words) == 1:
                    if f" {topic} " in f" {combined_summary} " or combined_summary.startswith(topic) or combined_summary.endswith(topic):
                        score += 1.5
                        logger.debug(f"[Router] Summary topic match: '{topic}' (+1.5)")
                else:
                    if topic in combined_summary:
                        score += 1.5
                        logger.debug(f"[Router] Summary topic match: '{topic}' (+1.5)")

        # 5. Document type matching
        query_types = set(t.lower() for t in query_meta.get("document_type_hints", []))
        # Use document_types list - check if ANY type matches
        doc_types = [t.lower() for t in doc_meta.get("document_types", []) if t]

        matching_types = [t for t in doc_types if t in query_types]
        if matching_types:
            score += 1.0
            logger.debug(f"[Router] Document type match: '{matching_types}' (+1.0)")

        # 6. Apply confidence multiplier
        confidence = doc_meta.get("confidence", 0.5)
        final_score = score * confidence

        logger.debug(
            f"[Router] Score for '{doc_meta.get('subject_name', 'Unknown')}': "
            f"{score:.2f} * {confidence:.2f} = {final_score:.2f}"
        )

        return final_score

    def _calculate_match_score_llm(
        self,
        query_meta: Dict[str, Any],
        doc_meta: Dict[str, Any],
        doc_filename: str,
        original_query: str,
        chain: Any
    ) -> float:
        """
        Calculate match score using LLM-based relevance scoring.

        Returns:
            Score from 0.0 to 10.0
        """
        # Prepare document information for LLM
        doc_subject = doc_meta.get("subject_name", "Unknown")
        # Use document_types list (get primary type for display)
        doc_types = doc_meta.get("document_types", [])
        doc_type = doc_types[0] if doc_types else "other"
        doc_summary = doc_meta.get("summary", "No summary available")
        doc_topics = ", ".join(doc_meta.get("topics", []))

        # Get key entities (top 5)
        key_entities = doc_meta.get("key_entities", [])[:5]
        doc_entities = ", ".join([f"{e.get('name')} ({e.get('type')})" for e in key_entities])

        # Query metadata
        query_entities = ", ".join(query_meta.get("entities", []))
        query_topics = ", ".join(query_meta.get("topics", []))
        query_doc_types = ", ".join(query_meta.get("document_type_hints", []))

        try:
            # Invoke LLM
            response = chain.invoke({
                "query": original_query,
                "query_entities": query_entities,
                "query_topics": query_topics,
                "query_doc_types": query_doc_types,
                "doc_filename": doc_filename,
                "doc_subject": doc_subject,
                "doc_type": doc_type,
                "doc_summary": doc_summary,
                "doc_topics": doc_topics,
                "doc_entities": doc_entities,
            })

            # Parse JSON response
            response_clean = response.strip()
            if response_clean.startswith("```"):
                lines = response_clean.split("\n")
                response_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else response_clean

            result = json.loads(response_clean)
            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "")

            logger.debug(
                f"[Router LLM] '{doc_subject}': {score:.2f}/10.0 - {reasoning}"
            )

            return score

        except Exception as e:
            logger.error(f"[Router LLM] Failed to parse LLM response: {e}")
            return 0.0

