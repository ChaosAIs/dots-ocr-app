"""
Bridge to existing graph search service.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class GraphServiceBridge:
    """Bridge to existing Graph RAG (Neo4j) service."""

    def __init__(self):
        self._graph_service = None
        self._neo4j_driver = None

    @property
    def graph_service(self):
        """Lazy load graph service."""
        if self._graph_service is None:
            try:
                from backend.rag_service.graph_rag.graph_service import GraphRAGService
                self._graph_service = GraphRAGService()
            except ImportError:
                logger.warning("GraphRAGService not available")
        return self._graph_service

    @property
    def neo4j_driver(self):
        """Lazy load Neo4j driver."""
        if self._neo4j_driver is None:
            try:
                from neo4j import GraphDatabase
                import os

                uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
                user = os.getenv("NEO4J_USER", "neo4j")
                password = os.getenv("NEO4J_PASSWORD", "")

                if password:
                    self._neo4j_driver = GraphDatabase.driver(
                        uri, auth=(user, password)
                    )
            except ImportError:
                logger.warning("Neo4j driver not available")
            except Exception as e:
                logger.warning(f"Neo4j connection failed: {e}")

        return self._neo4j_driver

    def query_graph(
        self,
        query: str,
        workspace_id: str,
        entity_hints: Optional[List[str]] = None,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Query the knowledge graph.

        Args:
            query: Natural language query
            workspace_id: Workspace context
            entity_hints: Entity names to focus on
            max_depth: Maximum traversal depth

        Returns:
            Graph query results
        """
        if self.graph_service:
            try:
                return self.graph_service.query(
                    query=query,
                    workspace_id=workspace_id,
                    entity_hints=entity_hints or [],
                    max_depth=max_depth
                )
            except Exception as e:
                logger.error(f"Graph query failed: {e}")

        # Fallback: Direct Neo4j query
        return self._fallback_query(query, workspace_id, entity_hints, max_depth)

    def execute_cypher(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a Cypher query directly.

        Args:
            cypher: Cypher query string
            parameters: Query parameters

        Returns:
            Query results
        """
        if not self.neo4j_driver:
            return {
                "success": False,
                "error": "Neo4j not available",
                "entities": [],
                "relationships": []
            }

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(cypher, parameters or {})
                records = list(result)

                entities = []
                relationships = []

                for record in records:
                    for key, value in record.items():
                        if hasattr(value, 'labels'):  # Node
                            entities.append({
                                "id": value.id,
                                "labels": list(value.labels),
                                "properties": dict(value)
                            })
                        elif hasattr(value, 'type'):  # Relationship
                            relationships.append({
                                "id": value.id,
                                "type": value.type,
                                "start_node": value.start_node.id,
                                "end_node": value.end_node.id,
                                "properties": dict(value)
                            })

                return {
                    "success": True,
                    "entities": entities,
                    "relationships": relationships
                }

        except Exception as e:
            logger.error(f"Cypher execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "entities": [],
                "relationships": []
            }

    def find_entities(
        self,
        entity_names: List[str],
        workspace_id: str
    ) -> Dict[str, Any]:
        """Find entities by name.

        Args:
            entity_names: Entity names to search
            workspace_id: Workspace context

        Returns:
            Found entities
        """
        if not entity_names:
            return {"success": True, "entities": []}

        # Build UNION query for multiple names
        conditions = " OR ".join([
            f"toLower(n.name) CONTAINS toLower('{name}')"
            for name in entity_names
        ])

        cypher = f"""
        MATCH (n)
        WHERE n.workspace_id = $workspace_id
        AND ({conditions})
        RETURN n
        LIMIT 100
        """

        return self.execute_cypher(cypher, {"workspace_id": workspace_id})

    def find_relationships(
        self,
        entity_id: int,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Find relationships for an entity.

        Args:
            entity_id: Entity node ID
            max_depth: Maximum traversal depth

        Returns:
            Related entities and relationships
        """
        cypher = f"""
        MATCH (n)-[r*1..{max_depth}]-(m)
        WHERE ID(n) = $entity_id
        RETURN n, r, m
        LIMIT 100
        """

        return self.execute_cypher(cypher, {"entity_id": entity_id})

    def _fallback_query(
        self,
        query: str,
        workspace_id: str,
        entity_hints: Optional[List[str]],
        max_depth: int
    ) -> Dict[str, Any]:
        """Fallback graph query using direct Neo4j."""
        if not self.neo4j_driver:
            return {
                "success": False,
                "error": "Graph database not available",
                "entities": [],
                "relationships": []
            }

        # Build query based on hints
        if entity_hints:
            return self.find_entities(entity_hints, workspace_id)

        # Default: Get all nodes in workspace
        cypher = """
        MATCH (n)
        WHERE n.workspace_id = $workspace_id
        RETURN n
        LIMIT 50
        """

        return self.execute_cypher(cypher, {"workspace_id": workspace_id})

    def close(self):
        """Close Neo4j connection."""
        if self._neo4j_driver:
            self._neo4j_driver.close()
            self._neo4j_driver = None
