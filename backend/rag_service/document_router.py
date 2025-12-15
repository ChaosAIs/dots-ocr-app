"""
Intelligent document routing based on query metadata matching.
Routes queries to the most relevant documents before vector search.

This module solves the "drowning out" problem where large documents dominate
search results by intelligently selecting which documents to search based on
metadata matching between the query and document metadata.
"""
import logging
from typing import List, Dict, Any, Tuple, Optional

from db.document_repository import DocumentRepository
from db.database import get_db_session

logger = logging.getLogger(__name__)

# Configuration
ENABLE_DOCUMENT_ROUTING = True  # Feature flag
DOCUMENT_ROUTING_TOP_K = 3      # Max documents to route to
DOCUMENT_ROUTING_MIN_SCORE = 0.3  # Minimum score threshold


class DocumentRouter:
    """Routes queries to relevant documents based on metadata matching."""
    
    def __init__(self):
        self.top_k = DOCUMENT_ROUTING_TOP_K
        self.min_score = DOCUMENT_ROUTING_MIN_SCORE
    
    def route_query(
        self, 
        query_metadata: Dict[str, Any],
        top_k: Optional[int] = None
    ) -> List[str]:
        """
        Route query to most relevant documents based on metadata matching.
        
        Args:
            query_metadata: Metadata extracted from query
                - entities: List[str]
                - topics: List[str]
                - document_type_hints: List[str]
                - intent: str
            top_k: Override default top_k
            
        Returns:
            List of document source names to search (e.g., ["Felix Yang- Resume - 2025"])
            Empty list means search all documents
        """
        if not ENABLE_DOCUMENT_ROUTING:
            logger.info("[Router] Document routing disabled, searching all documents")
            return []
        
        top_k = top_k or self.top_k
        
        try:
            # Get all documents with metadata
            documents = self._get_documents_with_metadata()
            
            if not documents:
                logger.warning("[Router] No documents with metadata found, searching all")
                return []
            
            # Score and rank documents
            scored_docs = self._score_documents(query_metadata, documents)
            
            # Filter by minimum score and take top-k
            relevant_docs = [
                (source, score) for source, score in scored_docs 
                if score >= self.min_score
            ][:top_k]
            
            if relevant_docs:
                sources = [source for source, score in relevant_docs]
                logger.info(
                    f"[Router] 🎯 Routed to {len(sources)} documents: "
                    f"{[(s, f'{sc:.2f}') for s, sc in relevant_docs]}"
                )
                return sources
            else:
                logger.info("[Router] No documents met minimum score, searching all")
                return []
                
        except Exception as e:
            logger.error(f"[Router] Error routing query: {e}", exc_info=True)
            return []  # Fallback to searching all documents
    
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
    
    def _score_documents(
        self, 
        query_metadata: Dict[str, Any], 
        documents: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """
        Score documents based on metadata match with query.
        
        Returns:
            List of (source_name, score) tuples, sorted by score descending
        """
        scored = []
        
        for doc in documents:
            score = self._calculate_match_score(query_metadata, doc["metadata"])
            scored.append((doc["source_name"], score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def _calculate_match_score(
        self,
        query_meta: Dict[str, Any],
        doc_meta: Dict[str, Any]
    ) -> float:
        """
        Calculate match score between query metadata and document metadata.

        Scoring:
        - Subject name exact match: +10.0
        - Subject name partial match: +5.0
        - Key entity match: +3.0 per entity
        - Topic match: +2.0 per topic
        - Document type match: +1.0
        - Confidence multiplier: score * confidence
        """
        score = 0.0

        # 1. Subject name matching (highest priority)
        query_entities = [e.lower() for e in query_meta.get("entities", [])]
        doc_subject = doc_meta.get("subject_name", "").lower()

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
        query_topics = set(t.lower() for t in query_meta.get("topics", []))
        doc_topics = set(t.lower() for t in doc_meta.get("topics", []))

        for query_topic in query_topics:
            for doc_topic in doc_topics:
                # Allow partial matching (e.g., "cloud" matches "cloud computing")
                if query_topic in doc_topic or doc_topic in query_topic:
                    score += 2.0
                    logger.debug(f"[Router] Topic match: '{query_topic}' ~ '{doc_topic}' (+2.0)")

        # 4. Document type matching
        query_types = set(t.lower() for t in query_meta.get("document_type_hints", []))
        doc_type = doc_meta.get("document_type", "").lower()

        if doc_type in query_types:
            score += 1.0
            logger.debug(f"[Router] Document type match: '{doc_type}' (+1.0)")

        # 5. Apply confidence multiplier
        confidence = doc_meta.get("confidence", 0.5)
        final_score = score * confidence

        logger.debug(
            f"[Router] Score for '{doc_meta.get('subject_name', 'Unknown')}': "
            f"{score:.2f} * {confidence:.2f} = {final_score:.2f}"
        )

        return final_score

