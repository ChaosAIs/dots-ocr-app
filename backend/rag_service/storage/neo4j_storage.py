"""
Neo4j Graph Storage implementation for GraphRAG.

Provides async graph operations for storing and querying:
- Entity nodes with properties
- Relationship edges between entities
- Graph traversal queries
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple, Literal

from ..graph_rag.base import BaseGraphStorage

logger = logging.getLogger(__name__)

# Neo4j driver will be imported on first use to avoid import errors if not installed
_neo4j_driver = None


def _get_neo4j_driver():
    """Get or create the Neo4j driver instance."""
    global _neo4j_driver
    if _neo4j_driver is None:
        from neo4j import AsyncGraphDatabase

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")

        _neo4j_driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")

    return _neo4j_driver


class Neo4jStorage(BaseGraphStorage):
    """Neo4j-based graph storage implementation."""

    def __init__(self, workspace_id: str = "default"):
        """
        Initialize Neo4j storage.

        Args:
            workspace_id: Workspace ID for multi-tenant isolation
        """
        super().__init__(workspace_id)
        self._driver = None

    def _get_driver(self):
        """Get the Neo4j driver."""
        if self._driver is None:
            self._driver = _get_neo4j_driver()
        return self._driver

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""
        driver = self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {id: $id, workspace_id: $workspace_id})
                RETURN count(e) > 0 as exists
                """,
                id=node_id,
                workspace_id=self.workspace_id,
            )
            record = await result.single()
            return record["exists"] if record else False

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        driver = self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {id: $id, workspace_id: $workspace_id})
                RETURN properties(e) as props
                """,
                id=node_id,
                workspace_id=self.workspace_id,
            )
            record = await result.single()
            return dict(record["props"]) if record else None

    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]) -> None:
        """Upsert a node."""
        driver = self._get_driver()
        # Ensure workspace_id is set
        node_data = {**node_data, "workspace_id": self.workspace_id}

        async with driver.session() as session:
            await session.run(
                """
                MERGE (e:Entity {id: $id, workspace_id: $workspace_id})
                SET e += $data
                """,
                id=node_id,
                workspace_id=self.workspace_id,
                data=node_data,
            )

    async def upsert_edge(
        self,
        src_id: str,
        tgt_id: str,
        edge_data: Dict[str, Any]
    ) -> None:
        """Upsert an edge between two nodes."""
        driver = self._get_driver()

        async with driver.session() as session:
            await session.run(
                """
                MATCH (src:Entity {id: $src_id, workspace_id: $workspace_id})
                MATCH (tgt:Entity {id: $tgt_id, workspace_id: $workspace_id})
                MERGE (src)-[r:RELATES_TO]->(tgt)
                SET r += $data
                """,
                src_id=src_id,
                tgt_id=tgt_id,
                workspace_id=self.workspace_id,
                data=edge_data,
            )

    async def get_node_edges(
        self,
        node_id: str,
        direction: Literal["in", "out", "both"] = "both"
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get edges connected to a node."""
        driver = self._get_driver()

        if direction == "out":
            pattern = "(e)-[r]->(related)"
        elif direction == "in":
            pattern = "(e)<-[r]-(related)"
        else:  # both
            pattern = "(e)-[r]-(related)"

        async with driver.session() as session:
            result = await session.run(
                f"""
                MATCH (e:Entity {{id: $id, workspace_id: $workspace_id}})
                MATCH {pattern}
                WHERE related.workspace_id = $workspace_id
                RETURN e.id as src, related.id as tgt, properties(r) as props
                """,
                id=node_id,
                workspace_id=self.workspace_id,
            )
            edges = []
            async for record in result:
                edges.append((record["src"], record["tgt"], dict(record["props"])))
            return edges

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and its edges."""
        driver = self._get_driver()
        async with driver.session() as session:
            await session.run(
                """
                MATCH (e:Entity {id: $id, workspace_id: $workspace_id})
                DETACH DELETE e
                """,
                id=node_id,
                workspace_id=self.workspace_id,
            )

    async def drop(self) -> None:
        """Drop all data for this workspace."""
        driver = self._get_driver()
        async with driver.session() as session:
            await session.run(
                """
                MATCH (e:Entity {workspace_id: $workspace_id})
                DETACH DELETE e
                """,
                workspace_id=self.workspace_id,
            )

    async def get_nodes_by_name(
        self,
        names: List[str],
        fuzzy: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get nodes by name.

        Args:
            names: List of entity names to search for
            fuzzy: If True, use CONTAINS for fuzzy matching

        Returns:
            List of node properties
        """
        if not names:
            return []

        driver = self._get_driver()
        async with driver.session() as session:
            if fuzzy:
                # Fuzzy match using CONTAINS
                result = await session.run(
                    """
                    MATCH (e:Entity {workspace_id: $workspace_id})
                    WHERE any(name IN $names WHERE toLower(e.name) CONTAINS toLower(name))
                    RETURN properties(e) as props
                    """,
                    workspace_id=self.workspace_id,
                    names=names,
                )
            else:
                # Exact match
                result = await session.run(
                    """
                    MATCH (e:Entity {workspace_id: $workspace_id})
                    WHERE e.name IN $names
                    RETURN properties(e) as props
                    """,
                    workspace_id=self.workspace_id,
                    names=names,
                )

            nodes = []
            async for record in result:
                nodes.append(dict(record["props"]))
            return nodes

    async def get_subgraph(
        self,
        node_ids: List[str],
        max_depth: int = 1
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, Dict[str, Any]]]]:
        """
        Get a subgraph starting from specified nodes.

        Args:
            node_ids: Starting node IDs
            max_depth: Maximum traversal depth

        Returns:
            Tuple of (nodes, edges)
        """
        if not node_ids:
            return [], []

        driver = self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                f"""
                MATCH (start:Entity {{workspace_id: $workspace_id}})
                WHERE start.id IN $node_ids
                CALL apoc.path.subgraphAll(start, {{maxLevel: $max_depth}})
                YIELD nodes, relationships
                UNWIND nodes as n
                WITH DISTINCT n, relationships
                WHERE n.workspace_id = $workspace_id
                RETURN collect(DISTINCT properties(n)) as nodes,
                       [r IN relationships | [startNode(r).id, endNode(r).id, properties(r)]] as edges
                """,
                workspace_id=self.workspace_id,
                node_ids=node_ids,
                max_depth=max_depth,
            )
            record = await result.single()
            if record:
                nodes = record["nodes"]
                edges = [(e[0], e[1], e[2]) for e in record["edges"]]
                return nodes, edges
            return [], []

    async def ensure_indexes(self) -> None:
        """Create indexes for better performance."""
        driver = self._get_driver()
        async with driver.session() as session:
            # Create unique constraint on entity id + workspace
            try:
                await session.run(
                    """
                    CREATE CONSTRAINT entity_workspace_id IF NOT EXISTS
                    FOR (e:Entity) REQUIRE (e.id, e.workspace_id) IS UNIQUE
                    """
                )
            except Exception as e:
                logger.debug(f"Index may already exist: {e}")

            # Create index on entity name for faster lookups
            try:
                await session.run(
                    """
                    CREATE INDEX entity_name IF NOT EXISTS
                    FOR (e:Entity) ON (e.name)
                    """
                )
            except Exception as e:
                logger.debug(f"Index may already exist: {e}")

