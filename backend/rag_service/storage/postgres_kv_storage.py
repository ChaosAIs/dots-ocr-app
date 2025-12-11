"""
PostgreSQL Key-Value Storage implementation for GraphRAG.

Provides async KV storage operations on PostgreSQL tables for:
- graphrag_doc_full: Full document content
- graphrag_chunks: Document chunks
- graphrag_entities: Extracted entities
- graphrag_hyperedges: Relationships between entities
- graphrag_llm_cache: LLM response cache
"""

import logging
import hashlib
from typing import Optional, List, Dict, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from ..graph_rag.base import BaseKVStorage

logger = logging.getLogger(__name__)


class PostgresKVStorage(BaseKVStorage):
    """PostgreSQL-based Key-Value storage implementation."""

    # Valid table names for security
    VALID_TABLES = {
        "graphrag_doc_full",
        "graphrag_chunks",
        "graphrag_entities",
        "graphrag_hyperedges",
        "graphrag_llm_cache",
    }

    def __init__(
        self,
        table_name: str,
        workspace_id: str = "default",
        connection_string: str = None,
    ):
        """
        Initialize PostgreSQL KV storage.

        Args:
            table_name: Name of the table to use (must be in VALID_TABLES)
            workspace_id: Workspace ID for multi-tenant isolation
            connection_string: Async PostgreSQL connection string
        """
        super().__init__(workspace_id)

        if table_name not in self.VALID_TABLES:
            raise ValueError(f"Invalid table name: {table_name}. Must be one of {self.VALID_TABLES}")

        self.table_name = table_name
        self._engine = None
        self._session_factory = None
        self._connection_string = connection_string

    async def _get_session(self) -> AsyncSession:
        """Get an async database session."""
        if self._engine is None:
            if self._connection_string is None:
                import os
                from urllib.parse import quote_plus
                host = os.getenv("POSTGRES_HOST", "localhost")
                port = os.getenv("POSTGRES_PORT", "6400")
                db = os.getenv("POSTGRES_DB", "dots_ocr")
                user = os.getenv("POSTGRES_USER", "postgres")
                password = os.getenv("POSTGRES_PASSWORD", "")
                # URL-encode the password to handle special characters like @
                encoded_password = quote_plus(password)
                self._connection_string = f"postgresql+asyncpg://{user}:{encoded_password}@{host}:{port}/{db}"

            self._engine = create_async_engine(self._connection_string, echo=False)
            self._session_factory = sessionmaker(
                self._engine, class_=AsyncSession, expire_on_commit=False
            )

        return self._session_factory()

    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a record by ID."""
        async with await self._get_session() as session:
            query = text(f"""
                SELECT * FROM {self.table_name}
                WHERE id = :id AND workspace_id = :workspace_id
            """)
            result = await session.execute(
                query, {"id": id, "workspace_id": self.workspace_id}
            )
            row = result.mappings().first()
            return dict(row) if row else None

    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple records by IDs."""
        if not ids:
            return []

        async with await self._get_session() as session:
            query = text(f"""
                SELECT * FROM {self.table_name}
                WHERE id = ANY(:ids) AND workspace_id = :workspace_id
            """)
            result = await session.execute(
                query, {"ids": ids, "workspace_id": self.workspace_id}
            )
            return [dict(row) for row in result.mappings().all()]

    async def filter_keys(self, keys: List[str]) -> set:
        """Filter to return only keys that exist in storage."""
        if not keys:
            return set()

        async with await self._get_session() as session:
            query = text(f"""
                SELECT id FROM {self.table_name}
                WHERE id = ANY(:keys) AND workspace_id = :workspace_id
            """)
            result = await session.execute(
                query, {"keys": keys, "workspace_id": self.workspace_id}
            )
            return {row[0] for row in result.fetchall()}

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Upsert multiple records."""
        if not data:
            return

        async with await self._get_session() as session:
            for record_id, record_data in data.items():
                # Filter out 'id' from record_data to avoid duplicate column
                filtered_data = {k: v for k, v in record_data.items() if k != "id"}

                # Build dynamic column list and values
                columns = ["id", "workspace_id"] + list(filtered_data.keys())
                placeholders = [":id", ":workspace_id"] + [f":{k}" for k in filtered_data.keys()]
                update_set = ", ".join([f"{k} = EXCLUDED.{k}" for k in filtered_data.keys()])
                if update_set:
                    update_set += ", updated_at = CURRENT_TIMESTAMP"
                else:
                    update_set = "updated_at = CURRENT_TIMESTAMP"

                query = text(f"""
                    INSERT INTO {self.table_name} ({", ".join(columns)})
                    VALUES ({", ".join(placeholders)})
                    ON CONFLICT (id) DO UPDATE SET {update_set}
                """)

                params = {"id": record_id, "workspace_id": self.workspace_id}
                params.update(filtered_data)
                await session.execute(query, params)

            await session.commit()

    async def delete(self, ids: List[str]) -> None:
        """Delete records by IDs."""
        if not ids:
            return

        async with await self._get_session() as session:
            query = text(f"""
                DELETE FROM {self.table_name}
                WHERE id = ANY(:ids) AND workspace_id = :workspace_id
            """)
            await session.execute(
                query, {"ids": ids, "workspace_id": self.workspace_id}
            )
            await session.commit()

    async def drop(self) -> None:
        """Drop all data for this workspace."""
        async with await self._get_session() as session:
            query = text(f"""
                DELETE FROM {self.table_name}
                WHERE workspace_id = :workspace_id
            """)
            await session.execute(query, {"workspace_id": self.workspace_id})
            await session.commit()


class LLMCacheStorage(PostgresKVStorage):
    """Specialized storage for LLM response caching."""

    def __init__(self, workspace_id: str = "default", connection_string: str = None):
        super().__init__("graphrag_llm_cache", workspace_id, connection_string)

    @staticmethod
    def _hash_prompt(prompt: str) -> str:
        """Create a hash of the prompt for caching."""
        return hashlib.sha256(prompt.encode()).hexdigest()

    async def get_cached_response(self, prompt: str, model_name: str = None) -> Optional[str]:
        """Get cached LLM response for a prompt."""
        prompt_hash = self._hash_prompt(prompt)

        async with await self._get_session() as session:
            query = text("""
                SELECT response FROM graphrag_llm_cache
                WHERE workspace_id = :workspace_id
                  AND prompt_hash = :prompt_hash
                  AND (:model_name IS NULL OR model_name = :model_name)
                ORDER BY created_at DESC
                LIMIT 1
            """)
            result = await session.execute(
                query,
                {
                    "workspace_id": self.workspace_id,
                    "prompt_hash": prompt_hash,
                    "model_name": model_name,
                },
            )
            row = result.fetchone()
            return row[0] if row else None

    async def cache_response(
        self, prompt: str, response: str, model_name: str = None
    ) -> None:
        """Cache an LLM response."""
        import uuid
        prompt_hash = self._hash_prompt(prompt)
        record_id = str(uuid.uuid4())

        await self.upsert({
            record_id: {
                "prompt_hash": prompt_hash,
                "response": response,
                "model_name": model_name,
            }
        })

