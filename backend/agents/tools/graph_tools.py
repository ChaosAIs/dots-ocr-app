"""
Graph search tools for the Graph Agent.

These tools handle:
- Cypher query execution against Neo4j
- Entity relationship traversal
- Result reporting
"""

import json
import logging
import time
from typing import Annotated, Optional

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agents.state.models import AgentOutput
from agents.config import AGENTIC_CONFIG

logger = logging.getLogger(__name__)


@tool
def cypher_query(
    query: str,
    entity_hints: str,
    max_depth: int,
    state: Annotated[dict, InjectedState]
) -> str:
    """Execute Cypher query against Neo4j knowledge graph.

    Searches for entities and relationships in the knowledge graph.

    Args:
        query: Natural language query or Cypher query
        entity_hints: JSON array of entity names/types to focus on
        max_depth: Maximum traversal depth for relationship queries

    Returns:
        JSON with query results including:
        - entities: Found entity nodes
        - relationships: Found relationships
        - paths: Traversal paths if applicable
    """
    try:
        hints = json.loads(entity_hints) if entity_hints else []
        workspace_id = state.get("workspace_id", "")

        start_time = time.time()

        # Check if graph search is enabled
        if not AGENTIC_CONFIG.get("enable_graph_search", True):
            return json.dumps({
                "success": False,
                "error": "Graph search is disabled",
                "entities": [],
                "relationships": []
            })

        # Try to use existing Graph RAG service
        try:
            from backend.rag_service.graph_rag.graph_service import GraphRAGService

            graph_service = GraphRAGService()

            # Execute query
            results = graph_service.query(
                query=query,
                workspace_id=workspace_id,
                entity_hints=hints,
                max_depth=max_depth
            )

            execution_time = int((time.time() - start_time) * 1000)

            return json.dumps({
                "success": True,
                "entities": results.get("entities", []),
                "relationships": results.get("relationships", []),
                "paths": results.get("paths", []),
                "execution_time_ms": execution_time
            })

        except ImportError:
            logger.warning("GraphRAGService not available")

        # Try direct Neo4j connection
        try:
            from neo4j import GraphDatabase
            import os

            neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "")

            if not neo4j_password:
                raise ValueError("Neo4j password not configured")

            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

            # Build a simple Cypher query based on the natural language
            # In production, use LLM to generate proper Cypher
            cypher = _build_cypher_query(query, hints, max_depth, workspace_id)

            with driver.session() as session:
                result = session.run(cypher)
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

            driver.close()

            execution_time = int((time.time() - start_time) * 1000)

            return json.dumps({
                "success": True,
                "entities": entities,
                "relationships": relationships,
                "paths": [],
                "execution_time_ms": execution_time
            })

        except Exception as neo4j_error:
            logger.warning(f"Neo4j query failed: {neo4j_error}")

        # Fallback: Return empty results with note
        execution_time = int((time.time() - start_time) * 1000)

        return json.dumps({
            "success": False,
            "error": "Graph database not available",
            "entities": [],
            "relationships": [],
            "paths": [],
            "execution_time_ms": execution_time,
            "note": "Neo4j connection not configured or unavailable"
        })

    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON input: {e}",
            "entities": [],
            "relationships": []
        })
    except Exception as e:
        logger.error(f"Error in cypher query: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "entities": [],
            "relationships": []
        })


def _build_cypher_query(
    query: str,
    hints: list,
    max_depth: int,
    workspace_id: str
) -> str:
    """Build a Cypher query from natural language (simple heuristic)."""
    # This is a simplified query builder
    # In production, use LLM to generate proper Cypher

    query_lower = query.lower()

    # Check for relationship patterns
    if "related" in query_lower or "connected" in query_lower:
        if hints:
            entity = hints[0]
            return f"""
            MATCH (n)-[r*1..{max_depth}]-(m)
            WHERE n.name CONTAINS '{entity}'
            AND n.workspace_id = '{workspace_id}'
            RETURN n, r, m
            LIMIT 50
            """

    # Check for entity search
    if hints:
        entity_filters = " OR ".join([f"n.name CONTAINS '{h}'" for h in hints])
        return f"""
        MATCH (n)
        WHERE ({entity_filters})
        AND n.workspace_id = '{workspace_id}'
        RETURN n
        LIMIT 100
        """

    # Default: Get all nodes in workspace
    return f"""
    MATCH (n)
    WHERE n.workspace_id = '{workspace_id}'
    RETURN n
    LIMIT 50
    """


@tool
def report_graph_result(
    task_id: str,
    entities: str,
    relationships: str,
    confidence: float,
    state: Annotated[dict, InjectedState]
) -> str:
    """Report graph search results back to the retrieval supervisor.

    Args:
        task_id: ID of the task being reported
        entities: JSON array of found entities
        relationships: JSON array of found relationships
        confidence: Overall confidence score (0-1)

    Returns:
        JSON string with the result report
    """
    try:
        parsed_entities = json.loads(entities) if entities else []
        parsed_relationships = json.loads(relationships) if relationships else []

        # Extract document IDs from entities if available
        doc_ids_used = list(set(
            e.get("properties", {}).get("document_id", "")
            for e in parsed_entities
            if e.get("properties", {}).get("document_id")
        ))

        # Determine status
        if confidence >= 0.7 and (parsed_entities or parsed_relationships):
            status = "success"
        elif confidence >= 0.5 or parsed_entities or parsed_relationships:
            status = "partial"
        else:
            status = "failed"

        logger.info(
            f"Graph Agent reporting result for task {task_id}: "
            f"status={status}, {len(parsed_entities)} entities, "
            f"{len(parsed_relationships)} relationships, confidence={confidence}"
        )

        return json.dumps({
            "success": True,
            "task_id": task_id,
            "agent_name": "graph_agent",
            "status": status,
            "data": {
                "entities": parsed_entities,
                "relationships": parsed_relationships,
                "entity_count": len(parsed_entities),
                "relationship_count": len(parsed_relationships)
            },
            "documents_used": doc_ids_used,
            "confidence": confidence,
            "message": f"Graph task {task_id} completed: {len(parsed_entities)} entities, {len(parsed_relationships)} relationships"
        })

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in report_graph_result: {e}")
        return json.dumps({
            "success": False,
            "task_id": task_id,
            "agent_name": "graph_agent",
            "status": "failed",
            "error": f"Failed to parse results: {e}"
        })
    except Exception as e:
        logger.error(f"Error reporting graph result: {e}")
        return json.dumps({
            "success": False,
            "task_id": task_id,
            "agent_name": "graph_agent",
            "status": "failed",
            "error": str(e)
        })
