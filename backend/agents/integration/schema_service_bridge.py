"""
Bridge to existing schema service.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SchemaServiceBridge:
    """Bridge to existing SchemaService for schema operations."""

    def __init__(self):
        self._schema_service = None

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

    def get_schema_for_document(
        self,
        document_id: str,
        workspace_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get schema definition for a specific document.

        Args:
            document_id: Document ID
            workspace_id: Workspace context

        Returns:
            Schema definition or None
        """
        if self.schema_service:
            try:
                return self.schema_service.get_document_schema(
                    document_id=document_id,
                    workspace_id=workspace_id
                )
            except Exception as e:
                logger.error(f"Error getting document schema: {e}")

        return None

    def get_schema_by_type(
        self,
        schema_type: str,
        workspace_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get schema definition by type.

        Args:
            schema_type: Type of schema (e.g., 'invoice', 'tabular')
            workspace_id: Workspace context

        Returns:
            Schema definition or None
        """
        try:
            from backend.database import get_db_session
            from backend.models import DataSchema

            with get_db_session() as db:
                schema = db.query(DataSchema).filter(
                    DataSchema.schema_type == schema_type,
                    DataSchema.workspace_id == workspace_id
                ).first()

                if schema:
                    return {
                        "schema_type": schema.schema_type,
                        "header_schema": schema.header_schema or {},
                        "line_items_schema": schema.line_items_schema or {},
                        "summary_schema": schema.summary_schema or {},
                        "fields": list((schema.header_schema or {}).keys())
                    }

        except Exception as e:
            logger.error(f"Error getting schema by type: {e}")

        return None

    def get_field_mapping(
        self,
        schema_type: str,
        workspace_id: str
    ) -> Dict[str, str]:
        """Get field type mappings for a schema.

        Args:
            schema_type: Type of schema
            workspace_id: Workspace context

        Returns:
            Dictionary mapping field names to their types
        """
        if self.schema_service:
            try:
                return self.schema_service.get_field_mapping(
                    schema_type=schema_type,
                    workspace_id=workspace_id
                )
            except Exception as e:
                logger.error(f"Error getting field mapping: {e}")

        return {}

    def infer_schema(
        self,
        document_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Infer schema from document data.

        Args:
            document_data: Document data to analyze

        Returns:
            Inferred schema definition
        """
        if self.schema_service:
            try:
                return self.schema_service.infer_schema(document_data)
            except Exception as e:
                logger.error(f"Error inferring schema: {e}")

        # Fallback: Basic inference
        fields = []
        if isinstance(document_data, dict):
            fields = list(document_data.keys())

        return {
            "schema_type": "inferred",
            "fields": fields,
            "header_schema": {f: "unknown" for f in fields}
        }

    def analyze_schemas(
        self,
        document_ids: List[str],
        workspace_id: str
    ) -> Dict[str, Any]:
        """Analyze schemas of multiple documents.

        Args:
            document_ids: List of document IDs
            workspace_id: Workspace context

        Returns:
            Analysis with common fields and grouping recommendations
        """
        schemas = []
        all_fields = []

        for doc_id in document_ids:
            schema = self.get_schema_for_document(doc_id, workspace_id)
            if schema:
                schemas.append(schema)
                all_fields.append(set(schema.get("fields", [])))

        # Find common fields
        common_fields = set()
        if all_fields:
            common_fields = set.intersection(*all_fields) if len(all_fields) > 1 else all_fields[0]

        # Get unique schema types
        schema_types = list(set(s.get("schema_type") for s in schemas if s.get("schema_type")))

        return {
            "schemas": schemas,
            "common_fields": list(common_fields),
            "schema_types": schema_types,
            "groupable": len(schema_types) == 1 and len(common_fields) > 0
        }
