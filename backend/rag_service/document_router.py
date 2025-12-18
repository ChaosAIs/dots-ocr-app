"""
Intelligent document routing based on query metadata matching.
Routes queries to the most relevant documents before vector search.

This module solves the "drowning out" problem where large documents dominate
search results by intelligently selecting which documents to search based on
metadata matching between the query and document metadata.
"""
import logging
import json
from typing import List, Dict, Any, Tuple, Optional

from db.document_repository import DocumentRepository
from db.database import get_db_session

logger = logging.getLogger(__name__)

# Configuration
ENABLE_DOCUMENT_ROUTING = True  # Feature flag
USE_LLM_SCORING = True  # Use LLM-based scoring instead of rule-based scoring
DOCUMENT_ROUTING_MIN_SCORE = 0.3  # Minimum absolute score threshold (for rule-based scoring)

# Advanced filtering to prevent low-quality documents from polluting results
DOCUMENT_ROUTING_SCORE_RATIO = 0.25   # Must be at least 25% of top score
DOCUMENT_ROUTING_MAX_SCORE_GAP = 8.0  # Max score gap from top document

# LLM scoring thresholds (when USE_LLM_SCORING=True)
LLM_SCORING_MIN_SCORE = 3.0  # Minimum LLM score (0-10 scale)
LLM_SCORING_SCORE_RATIO = 0.4  # Must be at least 40% of top score


class DocumentRouter:
    """Routes queries to relevant documents based on metadata matching."""

    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self.use_llm_scoring = USE_LLM_SCORING and llm_service is not None

        # Rule-based scoring thresholds
        self.min_score = DOCUMENT_ROUTING_MIN_SCORE
        self.score_ratio = DOCUMENT_ROUTING_SCORE_RATIO
        self.max_score_gap = DOCUMENT_ROUTING_MAX_SCORE_GAP

        # LLM scoring thresholds
        self.llm_min_score = LLM_SCORING_MIN_SCORE
        self.llm_score_ratio = LLM_SCORING_SCORE_RATIO

        if self.use_llm_scoring:
            logger.info("[Router] Using LLM-based scoring for document routing")
        else:
            logger.info("[Router] Using rule-based scoring for document routing")
    
    def route_query(
        self,
        query_metadata: Dict[str, Any],
        original_query: Optional[str] = None
    ) -> List[str]:
        """
        Route query to most relevant documents based on metadata matching.

        Args:
            query_metadata: Metadata extracted from query
                - entities: List[str]
                - topics: List[str]
                - document_type_hints: List[str]
                - intent: str
            original_query: Original query text (required for LLM scoring)

        Returns:
            List of document source names to search (e.g., ["Felix Yang- Resume - 2025"])
            Empty list means search all documents
        """
        if not ENABLE_DOCUMENT_ROUTING:
            logger.info("[Router] Document routing disabled, searching all documents")
            return []

        try:
            # Get all documents with metadata
            documents = self._get_documents_with_metadata()

            if not documents:
                logger.warning("[Router] No documents with metadata found, searching all")
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
                    f"[Router] 🎯 Routed to {len(sources)} document(s): "
                    f"{[(s, f'{sc:.2f}') for s, sc in relevant_docs]}"
                )
                return sources
            else:
                logger.info("[Router] No documents met filtering criteria, searching all")
                return []

        except Exception as e:
            logger.error(f"[Router] Error routing query: {e}", exc_info=True)
            return []  # Fallback to searching all documents
    
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
        Score documents using LLM-based relevance scoring.

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

