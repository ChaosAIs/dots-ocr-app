"""
Bridge to existing document routing functionality.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DocumentRouterBridge:
    """Bridge to existing document routing and query analysis services."""

    def __init__(self):
        self._unified_analyzer = None
        self._schema_service = None

    @property
    def unified_analyzer(self):
        """Lazy load unified query analyzer."""
        if self._unified_analyzer is None:
            try:
                from backend.analytics_service.unified_query_analyzer import UnifiedQueryAnalyzer
                self._unified_analyzer = UnifiedQueryAnalyzer()
            except ImportError:
                logger.warning("UnifiedQueryAnalyzer not available")
        return self._unified_analyzer

    @property
    def schema_service(self):
        """Lazy load schema service."""
        if self._schema_service is None:
            try:
                from backend.analytics_service.schema_service import SchemaService
                self._schema_service = SchemaService()
            except ImportError:
                logger.warning("SchemaService not available")
        return self._schema_service

    def analyze_query(
        self,
        query: str,
        workspace_id: str,
        chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Analyze query using unified analyzer.

        Args:
            query: User query
            workspace_id: Workspace context
            chat_history: Previous conversation

        Returns:
            Analysis results with intent, entities, etc.
        """
        if self.unified_analyzer:
            try:
                return self.unified_analyzer.analyze(
                    query=query,
                    workspace_id=workspace_id,
                    chat_history=chat_history or []
                )
            except Exception as e:
                logger.error(f"Query analysis failed: {e}")

        # Fallback: Basic analysis
        return {
            "intent": "data_analytics",
            "entities": [],
            "resolved_query": query
        }

    def get_relevant_documents(
        self,
        query: str,
        workspace_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get relevant documents for a query.

        Args:
            query: User query
            workspace_id: Workspace to search
            limit: Maximum documents to return

        Returns:
            List of relevant documents with metadata
        """
        try:
            from sqlalchemy.orm import Session
            from backend.database import get_db_session
            from backend.models import DocumentsData, DataSchema

            documents = []

            with get_db_session() as db:
                # Query documents
                doc_query = db.query(DocumentsData).filter(
                    DocumentsData.workspace_id == workspace_id
                ).limit(limit)

                for doc in doc_query:
                    # Get schema
                    schema_def = self._get_schema_definition(db, doc, workspace_id)

                    # Calculate relevance
                    relevance = self._calculate_relevance(query, doc, schema_def)

                    documents.append({
                        "document_id": str(doc.document_id),
                        "filename": doc.filename or f"document_{doc.document_id}",
                        "schema_type": doc.schema_type or "unknown",
                        "schema_definition": schema_def,
                        "relevance_score": relevance,
                        "data_location": "header"
                    })

            # Sort by relevance
            documents.sort(key=lambda x: x["relevance_score"], reverse=True)

            return documents

        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            return []

    def _get_schema_definition(
        self,
        db,
        doc,
        workspace_id: str
    ) -> Dict[str, Any]:
        """Get schema definition for a document."""
        try:
            from backend.models import DataSchema

            if not doc.schema_type:
                return {}

            schema = db.query(DataSchema).filter(
                DataSchema.schema_type == doc.schema_type,
                DataSchema.workspace_id == workspace_id
            ).first()

            if schema:
                return {
                    "header_schema": schema.header_schema or {},
                    "line_items_schema": schema.line_items_schema or {},
                    "fields": list((schema.header_schema or {}).keys())
                }

        except Exception as e:
            logger.warning(f"Error getting schema: {e}")

        return {}

    def _calculate_relevance(
        self,
        query: str,
        doc,
        schema_def: Dict
    ) -> float:
        """Calculate relevance score for a document."""
        score = 0.0
        query_lower = query.lower()

        # Filename match
        if doc.filename:
            filename_lower = doc.filename.lower()
            if any(word in filename_lower for word in query_lower.split()):
                score += 0.3

        # Schema type match
        if doc.schema_type:
            schema_lower = doc.schema_type.lower()
            if schema_lower in query_lower or query_lower in schema_lower:
                score += 0.2

        # Field name matches
        fields = schema_def.get("fields", [])
        for field in fields:
            if field.lower() in query_lower:
                score += 0.2
                break

        # Has data
        if doc.header_data or doc.summary_data:
            score += 0.2

        return min(score, 1.0)
