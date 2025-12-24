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
from typing import List, Dict, Any, Tuple, Optional

from db.document_repository import DocumentRepository
from db.database import get_db_session

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
            # Strategy 1: Try vector-based routing first (fastest)
            if self.use_vector_routing:
                vector_results = self._route_with_vector_search(
                    query_metadata, original_query,
                    accessible_document_ids=accessible_document_ids
                )

                if len(vector_results) >= self.vector_min_results:
                    logger.info(
                        f"[Router] 🎯 Vector routing returned {len(vector_results)} document(s): "
                        f"{vector_results}"
                    )
                    return vector_results
                elif vector_results:
                    # Got some results but below threshold - still use them but log
                    logger.info(
                        f"[Router] Vector routing returned {len(vector_results)} document(s) "
                        f"(below threshold {self.vector_min_results}, using anyway): {vector_results}"
                    )
                    return vector_results
                else:
                    logger.info("[Router] Vector routing returned no results, falling back to LLM/rule-based")

            # Strategy 2: Fallback to LLM or rule-based scoring
            documents = self._get_documents_with_metadata()

            if not documents:
                logger.warning("[Router] No documents with metadata found, returning accessible documents")
                if accessible_document_ids is not None:
                    return list(accessible_document_ids)
                return []

            # Filter documents by access control if provided
            if accessible_document_ids is not None:
                accessible_ids_set = set(accessible_document_ids)
                original_count = len(documents)
                documents = [doc for doc in documents if str(doc.get("id", "")) in accessible_ids_set]
                if len(documents) < original_count:
                    logger.info(f"[Router] Access control filtered documents: {original_count} -> {len(documents)}")

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
                logger.info("[Router] No documents met filtering criteria, searching all")
                if accessible_source_names is not None:
                    return list(accessible_source_names)
                return []

        except Exception as e:
            logger.error(f"[Router] Error routing query: {e}", exc_info=True)
            if accessible_document_ids is not None:
                return list(accessible_document_ids)
            return []  # Fallback to searching all documents

    def _route_with_vector_search(
        self,
        query_metadata: Dict[str, Any],
        original_query: str,
        accessible_source_names: Optional[set] = None,  # DEPRECATED
        accessible_document_ids: Optional[List[str]] = None
    ) -> List[str]:
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
            List of document IDs (UUIDs as strings) from vector search results
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
                return []

            # ========== ENTITY MATCHING BOOST ==========
            # Apply score boost when query entities match document entities/source names
            # This compensates for embedding model limitations with specific terms
            import re

            def normalize_for_matching(text: str) -> str:
                """Normalize text for fuzzy matching: lowercase, remove special chars, collapse spaces."""
                text = text.lower().strip()
                text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special chars except spaces
                text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
                return text

            query_entities = [normalize_for_matching(e) for e in query_metadata.get("entities", [])]
            query_terms = [t for t in original_query.lower().split() if len(t) > 2]

            ENTITY_MATCH_BOOST = float(os.getenv("ENTITY_MATCH_BOOST", "0.3"))

            if query_entities or query_terms:
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

                    original_score = result.get("score", 0)
                    boost = 0.0
                    match_reasons = []

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
                        elif entity_compact in source_name_compact or source_name_compact in entity_compact:
                            boost = max(boost, ENTITY_MATCH_BOOST)
                            match_reasons.append(f"entity '{entity}' ~matches source")
                        # Check if significant entity words appear in source (handles typos like "graphy" matching "graph")
                        elif any(w in source_name_compact or source_name_compact in w for w in entity_words if len(w) >= 4):
                            boost = max(boost, ENTITY_MATCH_BOOST * 0.9)
                            match_reasons.append(f"entity word from '{entity}' in source")
                        # Check prefix matching for entity words (handles "graphy" matching "graphr1" via "graph" prefix)
                        elif any(shares_prefix(w, source_name_compact) for w in entity_words if len(w) >= 4):
                            boost = max(boost, ENTITY_MATCH_BOOST * 0.85)
                            match_reasons.append(f"entity '{entity}' prefix-matches source")
                        # Check embedding text (contains entities, topics, summary)
                        elif entity in embedding_text or entity_compact in embedding_text.replace(" ", ""):
                            boost = max(boost, ENTITY_MATCH_BOOST * 0.8)
                            match_reasons.append(f"entity '{entity}' in content")

                    # Also check direct query terms against source name
                    for term in query_terms:
                        if len(term) > 3 and (term in source_name or term in source_name_compact):
                            if boost == 0:  # Only apply term boost if no entity boost
                                boost = max(boost, ENTITY_MATCH_BOOST * 0.5)
                                match_reasons.append(f"term '{term}' in source")

                    if boost > 0:
                        result["score"] = min(1.0, original_score + boost)
                        result["_boosted"] = True
                        result["_boost_amount"] = boost
                        logger.info(
                            f"[Router Vector]   ↑ BOOSTED: {source_name_raw:40s} "
                            f"{original_score:.4f} → {result['score']:.4f} (+{boost:.4f}) "
                            f"[{', '.join(match_reasons)}]"
                        )

                # Re-sort by score after boosting
                results.sort(key=lambda x: x.get("score", 0), reverse=True)

            # ========== DOCUMENT TYPE FILTERING ==========
            # If document_type_hints are provided, filter/penalize mismatched types
            doc_type_hints = [t.lower() for t in query_metadata.get("document_type_hints", [])]
            if doc_type_hints:
                DOC_TYPE_MISMATCH_PENALTY = float(os.getenv("DOC_TYPE_MISMATCH_PENALTY", "0.5"))
                logger.info(f"[Router Vector] DOCUMENT TYPE FILTERING:")
                logger.info(f"[Router Vector]   • Required types: {doc_type_hints}")
                logger.info(f"[Router Vector]   • Mismatch penalty: {DOC_TYPE_MISMATCH_PENALTY}")

                for result in results:
                    doc_type = (result.get("document_type") or "").lower()
                    source_name = result.get("source_name", "")
                    original_score = result.get("score", 0)

                    # Check if document type matches any of the hints
                    type_matches = any(
                        hint in doc_type or doc_type in hint
                        for hint in doc_type_hints
                    )

                    if not type_matches and doc_type:
                        # Apply penalty for type mismatch
                        result["score"] = original_score * DOC_TYPE_MISMATCH_PENALTY
                        result["_type_penalized"] = True
                        logger.info(
                            f"[Router Vector]   ↓ PENALIZED: {source_name:40s} "
                            f"type='{doc_type}' not in {doc_type_hints} "
                            f"({original_score:.4f} → {result['score']:.4f})"
                        )
                    else:
                        logger.debug(
                            f"[Router Vector]   ✓ TYPE OK: {source_name:40s} type='{doc_type}'"
                        )

                # Re-sort after type penalty
                results.sort(key=lambda x: x.get("score", 0), reverse=True)

            # ========== ADAPTIVE THRESHOLD CALCULATION ==========
            # Log all raw results first for debugging
            logger.info(f"[Router Vector] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"[Router Vector] RESULTS AFTER BOOSTING: {len(results)} results")
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

            # Calculate adaptive threshold
            top_score = results[0].get("score", 0) if results else 0
            relative_threshold = top_score * VECTOR_ROUTING_RELATIVE_THRESHOLD

            logger.info(f"[Router Vector] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"[Router Vector] ADAPTIVE THRESHOLD CALCULATION:")
            logger.info(f"[Router Vector]   • Top score:              {top_score:.4f}")
            logger.info(f"[Router Vector]   • Relative threshold (%): {VECTOR_ROUTING_RELATIVE_THRESHOLD:.0%}")
            logger.info(f"[Router Vector]   • Computed threshold:     {relative_threshold:.4f}")
            logger.info(f"[Router Vector]   • Formula: {top_score:.4f} × {VECTOR_ROUTING_RELATIVE_THRESHOLD} = {relative_threshold:.4f}")
            logger.info(f"[Router Vector] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            # Extract and deduplicate document IDs with adaptive filtering
            seen_doc_ids = set()
            document_ids = []
            filtered_count = 0
            duplicate_count = 0

            # Build set of accessible document IDs for fast lookup
            accessible_ids_set = set(accessible_document_ids) if accessible_document_ids else None

            logger.info(f"[Router Vector] FILTERING RESULTS:")
            for result in results:
                # Get document_id from payload (same as point ID in metadata collection)
                doc_id = result.get("payload", {}).get("document_id") or result.get("document_id")
                source_name = result.get("source_name", "unknown")
                score = result.get("score", 0)

                if not doc_id:
                    logger.debug(f"[Router Vector]   ⊘ SKIP: {source_name} - no document_id")
                    continue

                # Apply access control FIRST - skip if not in accessible list
                if accessible_ids_set is not None and doc_id not in accessible_ids_set:
                    logger.debug(
                        f"[Router Vector]   ✗ NO ACCESS: {source_name} ({doc_id[:8]}...)"
                    )
                    continue

                # Skip if below relative threshold
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
                logger.info(
                    f"[Router Vector]   ✓ ACCEPTED: {source_name:40s} ({doc_id[:8]}...) "
                    f"score={score:.4f} >= threshold={relative_threshold:.4f}"
                )

            # Summary
            logger.info(f"[Router Vector] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"[Router Vector] FILTERING SUMMARY:")
            logger.info(f"[Router Vector]   • Total raw results:    {len(results)}")
            logger.info(f"[Router Vector]   • Accepted (unique):    {len(document_ids)}")
            logger.info(f"[Router Vector]   • Filtered (low score): {filtered_count}")
            logger.info(f"[Router Vector]   • Skipped (duplicates): {duplicate_count}")
            if accessible_ids_set:
                logger.info(f"[Router Vector]   • Access control:       {len(accessible_ids_set)} accessible IDs")
            logger.info(f"[Router Vector]   • Final document_ids: {document_ids}")
            logger.info(f"[Router Vector] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            return document_ids

        except ImportError as e:
            logger.warning(f"[Router Vector] Vector search not available: {e}")
            return []
        except Exception as e:
            logger.error(f"[Router Vector] Error in vector search: {e}", exc_info=True)
            return []
    
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
            doc_type = doc_meta.get("document_type", "other")
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
        doc_type = doc_meta.get("document_type", "").lower()

        if doc_type in query_types:
            score += 1.0
            logger.debug(f"[Router] Document type match: '{doc_type}' (+1.0)")

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
        doc_type = doc_meta.get("document_type", "other")
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

