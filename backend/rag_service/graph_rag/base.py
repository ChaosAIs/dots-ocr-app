"""
Base classes and data structures for GraphRAG.

This module defines the abstract base classes and common data structures
used throughout the GraphRAG implementation.

Following Graph-R1 paper design:
- LOCAL: Entity-focused retrieval
- GLOBAL: Relationship-focused retrieval
- HYBRID: Combined entity and relationship retrieval (default)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional, Any, List, Dict, Tuple


class QueryMode(str, Enum):
    """Query modes for GraphRAG retrieval (from Graph-R1 paper)."""
    LOCAL = "local"      # Entity-focused queries
    GLOBAL = "global"    # Relationship-focused queries
    HYBRID = "hybrid"    # Both entity and relationship queries (default)


@dataclass
class QueryParam:
    """
    Query parameters for GraphRAG (aligned with Graph-R1 paper).

    Attributes:
        mode: Query mode (local, global, or hybrid)
        top_k: Number of top entities/relationships to retrieve
        max_steps: Maximum iterations for think-query-retrieve-rethink cycle
        only_need_context: Return only context without generating answer
        response_type: Type of response to generate
        max_token_for_text_unit: Max tokens for document chunks
        max_token_for_global_context: Max tokens for relationship descriptions
        max_token_for_local_context: Max tokens for entity descriptions
    """
    mode: QueryMode = QueryMode.HYBRID
    top_k: int = 60
    max_steps: int = 5  # For iterative reasoning cycle (Graph-R1)
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 4000
    max_token_for_local_context: int = 4000


@dataclass
class Entity:
    """Represents an extracted entity."""
    id: str
    name: str
    entity_type: str
    description: str
    source_chunk_id: str
    key_score: int = 50
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """Represents a relationship (hyperedge) between entities."""
    id: str
    src_entity_id: str
    tgt_entity_id: str
    description: str
    keywords: str
    weight: float = 1.0
    source_chunk_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseKVStorage(ABC):
    """Abstract base class for Key-Value storage."""

    def __init__(self, workspace_id: str = "default"):
        self.workspace_id = workspace_id

    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a record by ID."""
        ...

    @abstractmethod
    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple records by IDs."""
        ...

    @abstractmethod
    async def filter_keys(self, keys: List[str]) -> set:
        """Filter to return only keys that exist in storage."""
        ...

    @abstractmethod
    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Upsert multiple records. Key is the record ID."""
        ...

    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """Delete records by IDs."""
        ...

    @abstractmethod
    async def drop(self) -> None:
        """Drop all data for this workspace."""
        ...


class BaseVectorStorage(ABC):
    """Abstract base class for Vector storage."""

    def __init__(self, workspace_id: str = "default"):
        self.workspace_id = workspace_id

    @abstractmethod
    async def query(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Query for similar vectors."""
        ...

    @abstractmethod
    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        """
        Upsert vectors with their data.
        Each entry should have 'content' for embedding and optional 'metadata'.
        """
        ...

    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """Delete vectors by IDs."""
        ...


class BaseGraphStorage(ABC):
    """Abstract base class for Graph storage."""

    def __init__(self, workspace_id: str = "default"):
        self.workspace_id = workspace_id

    @abstractmethod
    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""
        ...

    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        ...

    @abstractmethod
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]) -> None:
        """Upsert a node."""
        ...

    @abstractmethod
    async def upsert_edge(
        self,
        src_id: str,
        tgt_id: str,
        edge_data: Dict[str, Any]
    ) -> None:
        """Upsert an edge between two nodes."""
        ...

    @abstractmethod
    async def get_node_edges(
        self,
        node_id: str,
        direction: Literal["in", "out", "both"] = "both"
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Get edges connected to a node.
        Returns list of (source_id, target_id, edge_data) tuples.
        """
        ...

    @abstractmethod
    async def delete_node(self, node_id: str) -> None:
        """Delete a node and its edges."""
        ...

    @abstractmethod
    async def drop(self) -> None:
        """Drop all data for this workspace."""
        ...

