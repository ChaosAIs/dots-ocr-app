"""
Bridge to existing SQL generator service.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class SQLGeneratorBridge:
    """Bridge to existing LLMSQLGeneratorV2 for SQL operations."""

    def __init__(self):
        self._sql_generator = None
        self._sql_executor = None

    @property
    def sql_generator(self):
        """Lazy load SQL generator."""
        if self._sql_generator is None:
            try:
                from backend.analytics_service.llm_sql_generator_v2 import LLMSQLGeneratorV2
                self._sql_generator = LLMSQLGeneratorV2()
            except ImportError:
                logger.warning("LLMSQLGeneratorV2 not available")
        return self._sql_generator

    @property
    def sql_executor(self):
        """Lazy load SQL executor."""
        if self._sql_executor is None:
            try:
                from backend.analytics_service.sql_query_executor import SQLQueryExecutor
                self._sql_executor = SQLQueryExecutor()
            except ImportError:
                logger.warning("SQLQueryExecutor not available")
        return self._sql_executor

    def generate_sql(
        self,
        query: str,
        workspace_id: str,
        document_ids: List[str],
        schema_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate SQL query using existing generator.

        Args:
            query: Natural language query
            workspace_id: Workspace context
            document_ids: Documents to query
            schema_context: Schema information for field mapping

        Returns:
            Generated SQL with metadata
        """
        if self.sql_generator:
            try:
                result = self.sql_generator.generate_sql(
                    query=query,
                    workspace_id=workspace_id,
                    document_ids=document_ids,
                    context=schema_context
                )
                return {
                    "success": True,
                    "sql": result.get("sql", ""),
                    "explanation": result.get("explanation", ""),
                    "confidence": result.get("confidence", 0.8)
                }
            except Exception as e:
                logger.error(f"SQL generation failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "sql": ""
                }

        # Fallback: Template-based generation
        return self._fallback_generate(query, document_ids, schema_context)

    def execute_sql(
        self,
        sql: str,
        workspace_id: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Execute SQL query using existing executor.

        Args:
            sql: SQL query to execute
            workspace_id: Workspace context
            max_retries: Maximum retry attempts

        Returns:
            Query results
        """
        if self.sql_executor:
            try:
                return self.sql_executor.execute_query(
                    sql=sql,
                    workspace_id=workspace_id,
                    max_retries=max_retries
                )
            except Exception as e:
                logger.error(f"SQL execution failed: {e}")

        # Fallback: Direct execution
        return self._fallback_execute(sql)

    def _fallback_generate(
        self,
        query: str,
        document_ids: List[str],
        schema_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fallback SQL generation using templates."""
        doc_filter = ", ".join([f"'{d}'" for d in document_ids])

        # Detect aggregation type
        query_lower = query.lower()
        if "sum" in query_lower or "total" in query_lower:
            sql = f"""
SELECT SUM(CAST(header_data->>'amount' AS NUMERIC)) as total
FROM documents_data
WHERE document_id IN ({doc_filter})
"""
        elif "count" in query_lower:
            sql = f"""
SELECT COUNT(*) as count
FROM documents_data
WHERE document_id IN ({doc_filter})
"""
        elif "average" in query_lower or "avg" in query_lower:
            sql = f"""
SELECT AVG(CAST(header_data->>'amount' AS NUMERIC)) as average
FROM documents_data
WHERE document_id IN ({doc_filter})
"""
        else:
            sql = f"""
SELECT document_id, header_data, summary_data
FROM documents_data
WHERE document_id IN ({doc_filter})
"""

        return {
            "success": True,
            "sql": sql.strip(),
            "explanation": "Generated using fallback template",
            "confidence": 0.6
        }

    def _fallback_execute(self, sql: str) -> Dict[str, Any]:
        """Fallback SQL execution."""
        try:
            from sqlalchemy import text
            from backend.database import get_db_session
            import time

            start_time = time.time()

            with get_db_session() as db:
                result = db.execute(text(sql))
                rows = result.fetchall()
                columns = result.keys() if hasattr(result, 'keys') else []

                data = [dict(zip(columns, row)) for row in rows]
                execution_time = int((time.time() - start_time) * 1000)

                return {
                    "success": True,
                    "data": data,
                    "row_count": len(data),
                    "execution_time_ms": execution_time
                }

        except Exception as e:
            logger.error(f"Fallback execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
