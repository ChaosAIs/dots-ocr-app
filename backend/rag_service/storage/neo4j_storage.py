"""
Neo4j Graph Storage implementation for GraphRAG.

Provides async graph operations for storing and querying:
- Entity nodes with properties and embeddings
- Relationship edges between entities with embeddings
- Graph traversal queries
- Vector similarity search using Neo4j native vector indexes

Key Features:
- Document-level deduplication: Entities are unique per (name, entity_type, source_doc)
- Chunk-by-chunk saving: Entities/relationships saved immediately after each chunk
- Multi-page PDF support: Same entity appearing in multiple pages is stored once
- Thread-safe: Creates driver per event loop to avoid cross-loop issues
- Vector embeddings: Semantic search using Neo4j 5.11+ native vector indexes
"""

import os
import logging
import asyncio
import atexit
import warnings
from typing import Optional, List, Dict, Any, Tuple, Literal
import threading

from ..graph_rag.base import BaseGraphStorage

logger = logging.getLogger(__name__)

# Vector embedding configuration
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "2560"))  # Qwen3-Embedding-4B dimension

# Store drivers per thread AND event loop to handle different event loops
_thread_local = threading.local()

# Track all created drivers for cleanup
_all_drivers = []
_drivers_lock = threading.Lock()


def _cleanup_drivers():
    """Cleanup all Neo4j drivers on exit - sync version to avoid event loop issues."""
    with _drivers_lock:
        for driver in _all_drivers:
            try:
                # Use sync close if available, otherwise just let GC handle it
                if hasattr(driver, 'close'):
                    driver.close()
            except Exception:
                # Ignore errors during cleanup - the process is exiting anyway
                pass
        _all_drivers.clear()


# Register cleanup handler
atexit.register(_cleanup_drivers)


def _get_neo4j_driver():
    """
    Get or create Neo4j driver for current thread and event loop.

    Each thread gets its own driver instance to avoid event loop conflicts
    when async code runs in different threads. Additionally, we track the
    event loop ID to recreate the driver if the event loop changes (which
    happens when asyncio.run() is called multiple times in the same thread).
    """
    from neo4j import AsyncGraphDatabase

    # Get current event loop ID
    try:
        current_loop = asyncio.get_running_loop()
        current_loop_id = id(current_loop)
    except RuntimeError:
        # No running loop - will be created by asyncio.run()
        current_loop_id = None

    # Check if we have a driver for this thread with the same event loop
    has_driver = hasattr(_thread_local, 'neo4j_driver') and _thread_local.neo4j_driver is not None
    same_loop = hasattr(_thread_local, 'event_loop_id') and _thread_local.event_loop_id == current_loop_id

    if not has_driver or not same_loop:
        # Close old driver if exists (to avoid connection leaks)
        if has_driver:
            try:
                # Note: close() is async, but we can't await here
                # The driver will be garbage collected
                logger.debug(f"Recreating Neo4j driver for new event loop (thread: {threading.current_thread().name})")
            except Exception:
                pass

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")

        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        _thread_local.neo4j_driver = driver
        _thread_local.event_loop_id = current_loop_id

        # Track for cleanup
        with _drivers_lock:
            _all_drivers.append(driver)

        logger.info(f"Connected to Neo4j at {uri} (thread: {threading.current_thread().name}, loop_id: {current_loop_id})")

    return _thread_local.neo4j_driver


class Neo4jStorage(BaseGraphStorage):
    """Neo4j-based graph storage implementation."""

    def __init__(self, workspace_id: str = "default"):
        """
        Initialize Neo4j storage.

        Args:
            workspace_id: Workspace ID for multi-tenant isolation
        """
        super().__init__(workspace_id)

    def _get_driver(self):
        """Get the Neo4j driver for current thread."""
        return _get_neo4j_driver()

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

    async def upsert_entity_for_document(
        self,
        name: str,
        entity_type: str,
        description: str,
        source_doc: str,
        source_chunk_id: str,
        key_score: int = 50,
        embedding: List[float] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Upsert an entity with document-level deduplication and optional embedding.

        For multi-page PDFs, the same entity appearing in different pages
        will be merged into one node with combined descriptions.

        Args:
            name: Entity name
            entity_type: Entity type (e.g., person, organization)
            description: Entity description
            source_doc: Source document ID (for deduplication)
            source_chunk_id: Source chunk ID
            key_score: Importance score
            embedding: Optional embedding vector for semantic search
            metadata: Optional metadata dict (e.g., date properties for date entities)

        Returns:
            Entity ID used in the graph
        """
        driver = self._get_driver()

        # Create a consistent entity ID based on name, type, and source document
        # This ensures the same entity in different chunks of the same document is deduplicated
        import hashlib
        entity_key = f"{self.workspace_id}:{source_doc}:{entity_type.lower()}:{name.lower().strip()}"
        entity_id = hashlib.md5(entity_key.encode()).hexdigest()

        # Build metadata properties for Cypher query
        metadata = metadata or {}
        metadata_params = {}
        metadata_set_clauses = []

        for key, value in metadata.items():
            # Only include simple types (str, int, float, bool)
            if isinstance(value, (str, int, float, bool)):
                param_name = f"meta_{key}"
                metadata_params[param_name] = value
                metadata_set_clauses.append(f"e.{key} = ${param_name}")

        metadata_set_str = ", ".join(metadata_set_clauses) if metadata_set_clauses else ""

        async with driver.session() as session:
            if embedding:
                # MERGE with embedding - store as property for vector index
                query = f"""
                    MERGE (e:Entity {{id: $id, workspace_id: $workspace_id}})
                    ON CREATE SET
                        e.name = $name,
                        e.entity_type = $entity_type,
                        e.description = $description,
                        e.source_doc = $source_doc,
                        e.source_chunk_ids = [$source_chunk_id],
                        e.key_score = $key_score,
                        e.embedding = $embedding
                        {', ' + metadata_set_str if metadata_set_str else ''}
                    ON MATCH SET
                        e.description = CASE
                            WHEN e.description CONTAINS $description THEN e.description
                            ELSE e.description + ' | ' + $description
                        END,
                        e.source_chunk_ids = CASE
                            WHEN $source_chunk_id IN e.source_chunk_ids THEN e.source_chunk_ids
                            ELSE e.source_chunk_ids + $source_chunk_id
                        END,
                        e.key_score = CASE
                            WHEN $key_score > e.key_score THEN $key_score
                            ELSE e.key_score
                        END,
                        e.embedding = $embedding
                        {', ' + metadata_set_str if metadata_set_str else ''}
                    """

                await session.run(
                    query,
                    id=entity_id,
                    workspace_id=self.workspace_id,
                    name=name,
                    entity_type=entity_type,
                    description=description,
                    source_doc=source_doc,
                    source_chunk_id=source_chunk_id,
                    key_score=key_score,
                    embedding=embedding,
                    **metadata_params,
                )
            else:
                # MERGE without embedding
                query = f"""
                    MERGE (e:Entity {{id: $id, workspace_id: $workspace_id}})
                    ON CREATE SET
                        e.name = $name,
                        e.entity_type = $entity_type,
                        e.description = $description,
                        e.source_doc = $source_doc,
                        e.source_chunk_ids = [$source_chunk_id],
                        e.key_score = $key_score
                        {', ' + metadata_set_str if metadata_set_str else ''}
                    ON MATCH SET
                        e.description = CASE
                            WHEN e.description CONTAINS $description THEN e.description
                            ELSE e.description + ' | ' + $description
                        END,
                        e.source_chunk_ids = CASE
                            WHEN $source_chunk_id IN e.source_chunk_ids THEN e.source_chunk_ids
                            ELSE e.source_chunk_ids + $source_chunk_id
                        END,
                        e.key_score = CASE
                            WHEN $key_score > e.key_score THEN $key_score
                            ELSE e.key_score
                        END
                        {', ' + metadata_set_str if metadata_set_str else ''}
                    """

                await session.run(
                    query,
                    id=entity_id,
                    workspace_id=self.workspace_id,
                    name=name,
                    entity_type=entity_type,
                    description=description,
                    source_doc=source_doc,
                    source_chunk_id=source_chunk_id,
                    key_score=key_score,
                    **metadata_params,
                )

        logger.debug(f"Upserted entity: {name} ({entity_type}) for doc: {source_doc}")
        return entity_id

    async def upsert_relationship_for_document(
        self,
        src_name: str,
        tgt_name: str,
        src_entity_type: str,
        tgt_entity_type: str,
        description: str,
        keywords: str,
        source_doc: str,
        source_chunk_id: str,
        weight: float = 1.0,
        embedding: List[float] = None,
    ) -> None:
        """
        Upsert a relationship with document-level deduplication and optional embedding.

        For multi-page PDFs, the same relationship appearing in different pages
        will be merged with combined descriptions and weights.

        Args:
            src_name: Source entity name
            tgt_name: Target entity name
            src_entity_type: Source entity type
            tgt_entity_type: Target entity type
            description: Relationship description
            keywords: Relationship keywords
            source_doc: Source document ID (for deduplication)
            source_chunk_id: Source chunk ID
            weight: Relationship weight
            embedding: Optional embedding vector for semantic search
        """
        driver = self._get_driver()

        # Generate entity IDs consistent with upsert_entity_for_document
        import hashlib
        src_key = f"{self.workspace_id}:{source_doc}:{src_entity_type.lower()}:{src_name.lower().strip()}"
        src_id = hashlib.md5(src_key.encode()).hexdigest()
        tgt_key = f"{self.workspace_id}:{source_doc}:{tgt_entity_type.lower()}:{tgt_name.lower().strip()}"
        tgt_id = hashlib.md5(tgt_key.encode()).hexdigest()

        async with driver.session() as session:
            if embedding:
                # Use MERGE with embedding
                await session.run(
                    """
                    MATCH (src:Entity {id: $src_id, workspace_id: $workspace_id})
                    MATCH (tgt:Entity {id: $tgt_id, workspace_id: $workspace_id})
                    MERGE (src)-[r:RELATES_TO]->(tgt)
                    ON CREATE SET
                        r.description = $description,
                        r.keywords = $keywords,
                        r.weight = $weight,
                        r.source_doc = $source_doc,
                        r.source_chunk_ids = [$source_chunk_id],
                        r.embedding = $embedding
                    ON MATCH SET
                        r.description = CASE
                            WHEN r.description CONTAINS $description THEN r.description
                            ELSE r.description + ' | ' + $description
                        END,
                        r.keywords = CASE
                            WHEN r.keywords CONTAINS $keywords THEN r.keywords
                            ELSE r.keywords + ', ' + $keywords
                        END,
                        r.weight = CASE
                            WHEN $weight > r.weight THEN $weight
                            ELSE r.weight
                        END,
                        r.source_chunk_ids = CASE
                            WHEN $source_chunk_id IN r.source_chunk_ids THEN r.source_chunk_ids
                            ELSE r.source_chunk_ids + $source_chunk_id
                        END,
                        r.embedding = $embedding
                    """,
                    src_id=src_id,
                    tgt_id=tgt_id,
                    workspace_id=self.workspace_id,
                    description=description,
                    keywords=keywords,
                    weight=weight,
                    source_doc=source_doc,
                    source_chunk_id=source_chunk_id,
                    embedding=embedding,
                )
            else:
                # Use MERGE without embedding
                await session.run(
                    """
                    MATCH (src:Entity {id: $src_id, workspace_id: $workspace_id})
                    MATCH (tgt:Entity {id: $tgt_id, workspace_id: $workspace_id})
                    MERGE (src)-[r:RELATES_TO]->(tgt)
                    ON CREATE SET
                        r.description = $description,
                        r.keywords = $keywords,
                        r.weight = $weight,
                        r.source_doc = $source_doc,
                        r.source_chunk_ids = [$source_chunk_id]
                    ON MATCH SET
                        r.description = CASE
                            WHEN r.description CONTAINS $description THEN r.description
                            ELSE r.description + ' | ' + $description
                        END,
                        r.keywords = CASE
                            WHEN r.keywords CONTAINS $keywords THEN r.keywords
                            ELSE r.keywords + ', ' + $keywords
                        END,
                        r.weight = CASE
                            WHEN $weight > r.weight THEN $weight
                            ELSE r.weight
                        END,
                        r.source_chunk_ids = CASE
                            WHEN $source_chunk_id IN r.source_chunk_ids THEN r.source_chunk_ids
                            ELSE r.source_chunk_ids + $source_chunk_id
                        END
                    """,
                    src_id=src_id,
                    tgt_id=tgt_id,
                    workspace_id=self.workspace_id,
                    description=description,
                    keywords=keywords,
                    weight=weight,
                    source_doc=source_doc,
                    source_chunk_id=source_chunk_id,
                )

        logger.debug(f"Upserted relationship: {src_name} -> {tgt_name} for doc: {source_doc}")

    async def delete_by_document(self, source_doc: str) -> int:
        """
        Delete all entities and relationships for a specific document.

        Args:
            source_doc: Source document ID

        Returns:
            Number of deleted nodes
        """
        driver = self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {workspace_id: $workspace_id, source_doc: $source_doc})
                WITH e, count(e) as cnt
                DETACH DELETE e
                RETURN cnt
                """,
                workspace_id=self.workspace_id,
                source_doc=source_doc,
            )
            record = await result.single()
            count = record["cnt"] if record else 0
            if count > 0:
                logger.info(f"Deleted {count} entities from Neo4j for document: {source_doc}")
            return count

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

    async def delete_by_source(self, source_name: str) -> int:
        """
        Delete all nodes where source_chunk_id starts with source_name.

        This is used to delete all entities for a specific document.

        Args:
            source_name: Source document name (e.g., "my_document")

        Returns:
            Number of deleted nodes
        """
        driver = self._get_driver()
        async with driver.session() as session:
            # Delete nodes where source_chunk_id starts with source_name
            result = await session.run(
                """
                MATCH (e:Entity {workspace_id: $workspace_id})
                WHERE e.source_chunk_id STARTS WITH $source_pattern
                   OR e.source_chunk_id = $source_name
                WITH e, count(e) as cnt
                DETACH DELETE e
                RETURN cnt
                """,
                workspace_id=self.workspace_id,
                source_pattern=f"{source_name}_",
                source_name=source_name,
            )
            record = await result.single()
            count = record["cnt"] if record else 0
            if count > 0:
                logger.info(
                    f"Deleted {count} nodes from Neo4j for source: {source_name}"
                )
            return count

    async def get_nodes_by_name(
        self,
        name: str,
        limit: int = 10,
        fuzzy: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get nodes by name (single name search with fuzzy matching).

        Args:
            name: Entity name to search for
            limit: Maximum number of results to return
            fuzzy: If True, use CONTAINS for fuzzy matching (default True)

        Returns:
            List of node properties
        """
        if not name or not name.strip():
            return []

        name_lower = name.lower().strip()
        driver = self._get_driver()
        async with driver.session() as session:
            if fuzzy:
                # Fuzzy match using CONTAINS (case-insensitive)
                result = await session.run(
                    """
                    MATCH (e:Entity {workspace_id: $workspace_id})
                    WHERE toLower(e.name) CONTAINS $name_lower
                    RETURN properties(e) as props
                    LIMIT $limit
                    """,
                    workspace_id=self.workspace_id,
                    name_lower=name_lower,
                    limit=limit,
                )
            else:
                # Exact match (case-insensitive)
                result = await session.run(
                    """
                    MATCH (e:Entity {workspace_id: $workspace_id})
                    WHERE toLower(e.name) = $name_lower
                    RETURN properties(e) as props
                    LIMIT $limit
                    """,
                    workspace_id=self.workspace_id,
                    name_lower=name_lower,
                    limit=limit,
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
        """Create indexes for better performance including vector indexes."""
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

            # Create vector index for entity embeddings (Neo4j 5.11+)
            try:
                await session.run(
                    f"""
                    CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
                    FOR (e:Entity) ON (e.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {EMBEDDING_DIM},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                    """
                )
                logger.info(f"Created entity vector index (dim={EMBEDDING_DIM})")
            except Exception as e:
                logger.debug(f"Vector index may already exist or not supported: {e}")

            # Create vector index for relationship embeddings (Neo4j 5.18+)
            # Note: Relationship vector indexes require Neo4j 5.18+
            try:
                await session.run(
                    f"""
                    CREATE VECTOR INDEX relationship_embedding IF NOT EXISTS
                    FOR ()-[r:RELATES_TO]-() ON (r.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {EMBEDDING_DIM},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                    """
                )
                logger.info(f"Created relationship vector index (dim={EMBEDDING_DIM})")
            except Exception as e:
                logger.debug(f"Relationship vector index may not be supported: {e}")

    async def vector_search_entities(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_score: float = 0.7,
        source_names: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for entities using vector similarity.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            min_score: Minimum similarity score (0-1)
            source_names: Optional list of source document names to filter by

        Returns:
            List of entity dictionaries with similarity scores
        """
        driver = self._get_driver()
        async with driver.session() as session:
            try:
                # If source filtering is required, use KNN pre-filtered search
                # (Neo4j vector indexes don't support pre-filtering, so we filter first then compute similarity)
                if source_names and len(source_names) > 0:
                    query = """
                        MATCH (e:Entity)
                        WHERE e.workspace_id = $workspace_id
                          AND e.source_doc IN $source_names
                          AND e.embedding IS NOT NULL
                        WITH e, vector.similarity.cosine(e.embedding, $embedding) AS score
                        WHERE score >= $min_score
                        RETURN properties(e) as props, score
                        ORDER BY score DESC
                        LIMIT $limit
                        """
                    params = {
                        "embedding": query_embedding,
                        "workspace_id": self.workspace_id,
                        "source_names": source_names,
                        "min_score": min_score,
                        "limit": limit,
                    }
                else:
                    # Use vector index for unfiltered search
                    query = """
                        CALL db.index.vector.queryNodes('entity_embedding', $limit, $embedding)
                        YIELD node, score
                        WHERE node.workspace_id = $workspace_id AND score >= $min_score
                        RETURN properties(node) as props, score
                        ORDER BY score DESC
                        """
                    params = {
                        "embedding": query_embedding,
                        "limit": limit * 2,  # Fetch more to filter by workspace
                        "workspace_id": self.workspace_id,
                        "min_score": min_score,
                    }

                result = await session.run(query, **params)
                entities = []
                async for record in result:
                    entity = dict(record["props"])
                    entity["_score"] = record["score"]
                    entities.append(entity)
                    if not (source_names and len(source_names) > 0) and len(entities) >= limit:
                        break
                return entities
            except Exception as e:
                logger.warning(f"Vector search failed (index may not exist): {e}")
                return []

    async def vector_search_relationships(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_score: float = 0.7,
        source_names: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relationships using vector similarity.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            min_score: Minimum similarity score (0-1)
            source_names: Optional list of source document names to filter by

        Returns:
            List of relationship dictionaries with similarity scores
        """
        driver = self._get_driver()
        async with driver.session() as session:
            try:
                # If source filtering is required, use KNN pre-filtered search
                # (Neo4j vector indexes don't support pre-filtering, so we filter first then compute similarity)
                if source_names and len(source_names) > 0:
                    query = """
                        MATCH (src:Entity)-[r:RELATES_TO]->(tgt:Entity)
                        WHERE src.workspace_id = $workspace_id
                          AND r.source_doc IN $source_names
                          AND r.embedding IS NOT NULL
                        WITH r, src, tgt, vector.similarity.cosine(r.embedding, $embedding) AS score
                        WHERE score >= $min_score
                        RETURN properties(r) as props,
                               src.name as src_name,
                               tgt.name as tgt_name,
                               src.entity_type as src_type,
                               tgt.entity_type as tgt_type,
                               score
                        ORDER BY score DESC
                        LIMIT $limit
                        """
                    params = {
                        "embedding": query_embedding,
                        "workspace_id": self.workspace_id,
                        "source_names": source_names,
                        "min_score": min_score,
                        "limit": limit,
                    }
                else:
                    # Use vector index for unfiltered search
                    query = """
                        CALL db.index.vector.queryRelationships('relationship_embedding', $limit, $embedding)
                        YIELD relationship, score
                        WHERE relationship.source_doc IS NOT NULL AND score >= $min_score
                        WITH relationship, score,
                             startNode(relationship) as src,
                             endNode(relationship) as tgt
                        WHERE src.workspace_id = $workspace_id
                        RETURN properties(relationship) as props,
                               src.name as src_name,
                               tgt.name as tgt_name,
                               src.entity_type as src_type,
                               tgt.entity_type as tgt_type,
                               score
                        ORDER BY score DESC
                        """
                    params = {
                        "embedding": query_embedding,
                        "limit": limit * 2,
                        "workspace_id": self.workspace_id,
                        "min_score": min_score,
                    }

                result = await session.run(query, **params)
                relationships = []
                async for record in result:
                    rel = dict(record["props"])
                    rel["src_name"] = record["src_name"]
                    rel["tgt_name"] = record["tgt_name"]
                    rel["src_type"] = record["src_type"]
                    rel["tgt_type"] = record["tgt_type"]
                    rel["_score"] = record["score"]
                    relationships.append(rel)
                    if len(relationships) >= limit:
                        break
                return relationships
            except Exception as e:
                logger.warning(f"Relationship vector search failed: {e}")
                return []

    async def hybrid_search_entities(
        self,
        query_embedding: List[float],
        text_query: str = None,
        limit: int = 10,
        min_score: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and text matching.

        Args:
            query_embedding: Query embedding vector
            text_query: Optional text for name matching
            limit: Maximum number of results
            min_score: Minimum vector similarity score

        Returns:
            List of entity dictionaries with combined scores
        """
        # First get vector results
        vector_results = await self.vector_search_entities(
            query_embedding, limit=limit, min_score=min_score
        )

        # If text query provided, also get text matches
        if text_query:
            from .neo4j_storage import extract_entity_names_from_query
            text_results = await self.get_nodes_by_name(text_query, limit=limit)
            # Merge results, preferring vector scores
            seen_ids = {r.get("id") for r in vector_results}
            for entity in text_results:
                if entity.get("id") not in seen_ids:
                    entity["_score"] = 0.5  # Default score for text matches
                    vector_results.append(entity)

        return vector_results[:limit]

