"""
Qdrant Vector Store configuration for RAG.
Connects to Qdrant running at localhost:6333.
"""

import os
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Set, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from .local_qwen_embedding import LocalQwen3Embedding
from .timing_metrics import get_current_metrics

logger = logging.getLogger(__name__)

# Configuration for unified vector search
PARENT_CHUNK_AUTO_INCLUDE = os.getenv("PARENT_CHUNK_AUTO_INCLUDE", "true").lower() == "true"


@dataclass
class UnifiedSearchResult:
    """
    Unified result structure for vector search operations.

    This provides a consistent result format for both legacy single-shot
    and iterative reasoning search paths.
    """
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    parent_chunks: List[Dict[str, Any]] = field(default_factory=list)
    sources: Set[str] = field(default_factory=set)
    total_count: int = 0
    retrieval_stats: Dict[str, Any] = field(default_factory=dict)

# Collection names for document embeddings
COLLECTION_NAME = "documents"

# Singleton instances
_vectorstore: Optional[QdrantVectorStore] = None
_embeddings: Optional[LocalQwen3Embedding] = None
_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Get or create the Qdrant client instance."""
    global _client
    if _client is None:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        _client = QdrantClient(host=qdrant_host, port=qdrant_port, check_compatibility=False)
        logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
    return _client


def get_embeddings() -> LocalQwen3Embedding:
    """Get or create the embeddings instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = LocalQwen3Embedding()
    return _embeddings


def ensure_collection_exists(client: QdrantClient, embedding_dim: int = 2560):
    """
    Ensure the Qdrant collection exists, create if not.

    Args:
        client: The Qdrant client.
        embedding_dim: Dimension of the embedding vectors.
    """
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if COLLECTION_NAME not in collection_names:
        logger.info(f"Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"Collection {COLLECTION_NAME} created successfully")
    else:
        logger.info(f"Collection {COLLECTION_NAME} already exists")


def get_vectorstore() -> QdrantVectorStore:
    """
    Get or create the Qdrant vectorstore instance.

    Returns:
        Configured QdrantVectorStore instance.
    """
    global _vectorstore

    if _vectorstore is None:
        client = get_qdrant_client()
        embeddings = get_embeddings()

        # Ensure collection exists
        ensure_collection_exists(client)

        _vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
        logger.info("Qdrant vectorstore initialized")

    return _vectorstore



def get_chunks_by_ids(
    chunk_ids: List[str],
    source_names: List[str] = None,
    accessible_document_ids: List[str] = None
) -> List[Document]:
    """
    Get full chunk content by chunk IDs from the main document collection.

    Args:
        chunk_ids: List of chunk IDs to retrieve.
        source_names: Optional list of source names to filter.
        accessible_document_ids: Optional list of document IDs for access control filtering.
                                If provided, only chunks with document_id in this list are returned.
                                If empty list, returns empty results.

    Returns:
        List of Document objects with full chunk content.
    """
    # Access control: empty list means no access
    if accessible_document_ids is not None and len(accessible_document_ids) == 0:
        logger.warning("[Vectorstore] Access control: no accessible documents - returning empty chunks")
        return []

    client = get_qdrant_client()
    try:
        # Build filter conditions
        must_conditions = []
        should_conditions = []

        # Filter by chunk_id (stored in metadata)
        if chunk_ids:
            should_conditions = [
                models.FieldCondition(
                    key="metadata.chunk_id",
                    match=models.MatchValue(value=chunk_id),
                )
                for chunk_id in chunk_ids
            ]

        # Filter by source if specified
        if source_names and len(source_names) > 0:
            must_conditions.append(
                models.Filter(
                    should=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=source_name),
                        )
                        for source_name in source_names
                    ]
                )
            )

        # Add access control filter by document_id
        if accessible_document_ids is not None and len(accessible_document_ids) > 0:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.document_id",
                    match=models.MatchAny(any=accessible_document_ids),
                )
            )
            logger.info(f"[Vectorstore] get_chunks_by_ids: filtering to {len(accessible_document_ids)} accessible document IDs")

        filter_condition = models.Filter(
            must=must_conditions if must_conditions else None,
            should=should_conditions if should_conditions else None,
        )

        # Scroll through the collection to find matching documents
        result, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=filter_condition,
            limit=len(chunk_ids) if chunk_ids else 100,
            with_payload=True,
        )

        # Convert to Document objects
        docs = []
        for point in result:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            page_content = payload.get("page_content", "")
            docs.append(Document(page_content=page_content, metadata=metadata))

        return docs

    except Exception as e:
        logger.error(f"Error getting chunks by IDs: {e}")
        return []


def get_all_chunks_for_source(source_name: str) -> List[Document]:
    """
    Get all chunks for a specific source document from Qdrant.

    Used for metadata extraction re-indexing when vector indexing succeeded.

    Args:
        source_name: Source document name

    Returns:
        List of Document objects with all chunks from the source
    """
    client = get_qdrant_client()
    try:
        # Build filter for source
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=source_name),
                )
            ]
        )

        # Scroll through all chunks for this source
        result, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=filter_condition,
            with_payload=True,
            limit=10000,  # Large limit to get all chunks
        )

        # Convert to Document objects
        docs = []
        for point in result:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            page_content = payload.get("page_content", "")
            docs.append(Document(page_content=page_content, metadata=metadata))

        logger.info(f"Retrieved {len(docs)} chunks for source: {source_name}")
        return docs

    except Exception as e:
        logger.error(f"Error getting all chunks for source {source_name}: {e}")
        return []


def get_parent_chunks_for_children(
    child_chunk_ids: List[str],
    source_names: List[str] = None
) -> List[Document]:
    """
    Retrieve parent chunks for given child chunk IDs.

    This function implements parent context retrieval for the parent/child
    chunk design. It:
    1. Fetches child chunks to get their parent_chunk_id
    2. Retrieves parent chunks by those IDs

    Args:
        child_chunk_ids: List of child chunk IDs to look up parents for
        source_names: Optional list of source names to filter

    Returns:
        List of parent Document objects (deduplicated)
    """
    logger.debug(f"[ParentChunk] get_parent_chunks_for_children called with {len(child_chunk_ids) if child_chunk_ids else 0} child IDs")

    if not child_chunk_ids:
        logger.debug("[ParentChunk] No child chunk IDs provided, returning empty")
        return []

    client = get_qdrant_client()

    try:
        # Step 1: Get child chunks to extract parent_chunk_ids
        logger.debug(f"[ParentChunk] Step 1: Fetching child chunks by IDs: {child_chunk_ids[:5]}{'...' if len(child_chunk_ids) > 5 else ''}")
        child_docs = get_chunks_by_ids(child_chunk_ids, source_names)
        logger.debug(f"[ParentChunk] Retrieved {len(child_docs)} child documents from Qdrant")

        parent_ids = set()
        child_to_parent_map = {}  # Track which child maps to which parent
        for doc in child_docs:
            parent_id = doc.metadata.get("parent_chunk_id")
            chunk_id = doc.metadata.get("chunk_id", "unknown")
            chunk_type = doc.metadata.get("chunk_type", "unknown")
            is_parent = doc.metadata.get("is_parent_chunk", False)

            logger.debug(f"[ParentChunk] Child chunk '{chunk_id}': type={chunk_type}, is_parent={is_parent}, parent_id={parent_id}")

            if parent_id:
                parent_ids.add(parent_id)
                child_to_parent_map[chunk_id] = parent_id

        if not parent_ids:
            logger.info("[ParentChunk] No parent_chunk_ids found in any child chunks - chunks may be atomic (no parent)")
            return []

        logger.info(f"[ParentChunk] Step 2: Found {len(parent_ids)} unique parent IDs for {len(child_chunk_ids)} child chunks")
        logger.debug(f"[ParentChunk] Parent IDs to fetch: {list(parent_ids)[:5]}{'...' if len(parent_ids) > 5 else ''}")
        logger.debug(f"[ParentChunk] Child→Parent mapping: {dict(list(child_to_parent_map.items())[:5])}")

        # Step 2: Retrieve parent chunks
        parent_docs = get_chunks_by_ids(list(parent_ids), source_names)

        logger.info(f"[ParentChunk] Successfully retrieved {len(parent_docs)} parent chunks")
        for doc in parent_docs:
            parent_chunk_id = doc.metadata.get("chunk_id", "unknown")
            parent_source = doc.metadata.get("source", "unknown")
            content_preview = doc.page_content[:100].replace('\n', ' ') if doc.page_content else ""
            child_ids = doc.metadata.get("child_chunk_ids", [])
            logger.debug(f"[ParentChunk] Parent '{parent_chunk_id}' from '{parent_source}': {len(child_ids)} children, content='{content_preview}...'")

        return parent_docs

    except Exception as e:
        logger.error(f"Error getting parent chunks: {e}")
        return []


def get_parent_chunk_by_id(
    parent_chunk_id: str,
    source_names: List[str] = None
) -> Optional[Document]:
    """
    Retrieve a single parent chunk by its ID.

    Args:
        parent_chunk_id: The parent chunk ID to retrieve
        source_names: Optional list of source names to filter

    Returns:
        Document object for the parent chunk, or None if not found
    """
    docs = get_chunks_by_ids([parent_chunk_id], source_names)
    return docs[0] if docs else None


def unified_vector_search(
    query: str,
    document_ids: List[str],
    k: int = 18,
    fetch_k: int = 50,
    lambda_mult: float = 0.5,
    per_document_selection: bool = True,
    include_parent_chunks: bool = None,
) -> UnifiedSearchResult:
    """
    Unified vector search function combining the best features from legacy and iterative approaches.

    This function provides a consistent search experience regardless of whether iterative
    reasoning is enabled or disabled. It combines:
    - MMR search with diversity control (from legacy)
    - Per-document balanced selection (from legacy)
    - Parent chunk auto-inclusion (from iterative)
    - Deduplication and chunk type tracking (from iterative)

    Args:
        query: Search query string
        document_ids: List of document IDs the user can access (REQUIRED for access control).
                     If None or empty, returns empty results for security.
        k: Number of chunks to retrieve (default: 18)
        fetch_k: Number of candidates to fetch for MMR (default: 50)
        lambda_mult: MMR diversity parameter. 0.0 = max diversity, 1.0 = max relevance (default: 0.5)
        per_document_selection: If True, retrieve chunks from each document separately
                               to ensure all documents are represented (default: True)
        include_parent_chunks: If True, auto-fetch parent chunks for child chunks.
                              If None, uses PARENT_CHUNK_AUTO_INCLUDE env variable.

    Returns:
        UnifiedSearchResult with chunks, parent_chunks, sources, and stats
    """
    # Resolve include_parent_chunks from env if not specified
    if include_parent_chunks is None:
        include_parent_chunks = PARENT_CHUNK_AUTO_INCLUDE

    logger.info("=" * 80)
    logger.info("[VectorSearch] ========== UNIFIED VECTOR SEARCH START ==========")
    logger.info("=" * 80)
    logger.info(f"[VectorSearch] Query: {query[:80]}...")
    logger.info(f"[VectorSearch] Parameters:")
    logger.info(f"[VectorSearch]   - Document IDs: {len(document_ids) if document_ids else 0}")
    logger.info(f"[VectorSearch]   - K (chunks to retrieve): {k}")
    logger.info(f"[VectorSearch]   - Fetch K (MMR candidates): {fetch_k}")
    logger.info(f"[VectorSearch]   - Lambda Mult (diversity): {lambda_mult}")
    logger.info(f"[VectorSearch]   - Per-Document Selection: {per_document_selection}")
    logger.info(f"[VectorSearch]   - Include Parent Chunks: {include_parent_chunks}")
    logger.info("-" * 80)

    # Security check - block if no document_ids provided
    if document_ids is None or len(document_ids) == 0:
        logger.warning("[VectorSearch] BLOCKED: No accessible documents - returning empty results for security")
        logger.info("=" * 80)
        return UnifiedSearchResult(
            chunks=[],
            parent_chunks=[],
            sources=set(),
            total_count=0,
            retrieval_stats={
                "query": query,
                "blocked": True,
                "reason": "no_document_access"
            }
        )

    logger.info(f"[VectorSearch] Access Control: {len(document_ids)} documents accessible")

    # Get timing metrics context
    timing_metrics = get_current_metrics()
    vector_search_start = time.time()

    vectorstore = get_vectorstore()
    all_chunks = []
    seen_chunk_ids = set()
    sources = set()
    child_chunk_ids = []

    # Convert document_ids to strings
    doc_id_strings = [str(doc_id) for doc_id in document_ids]

    try:
        if per_document_selection and len(doc_id_strings) > 1:
            # Per-document balanced selection: retrieve from each document separately
            all_chunks, seen_chunk_ids, sources, child_chunk_ids = _search_per_document(
                vectorstore=vectorstore,
                query=query,
                document_ids=doc_id_strings,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
        else:
            # Single search across all documents
            all_chunks, seen_chunk_ids, sources, child_chunk_ids = _search_all_documents(
                vectorstore=vectorstore,
                query=query,
                document_ids=doc_id_strings,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )

        # Auto-include parent chunks for child matches
        parent_chunks = []
        if include_parent_chunks and child_chunk_ids:
            parent_chunks = _fetch_parent_chunks(
                child_chunk_ids=child_chunk_ids,
                seen_chunk_ids=seen_chunk_ids,
                sources=sources
            )

        # Build retrieval stats
        parent_count = sum(1 for c in all_chunks if c.get("is_parent"))
        child_count = len(child_chunk_ids)
        atomic_count = len(all_chunks) - parent_count

        stats = {
            "query": query,
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
            "per_document_selection": per_document_selection,
            "include_parent_chunks": include_parent_chunks,
            "document_count": len(doc_id_strings),
            "chunk_count": len(all_chunks),
            "parent_chunks_added": len(parent_chunks),
            "chunk_types": {
                "parent": parent_count,
                "child": child_count,
                "atomic": atomic_count
            },
            "source_count": len(sources)
        }

        total_count = len(all_chunks) + len(parent_chunks)

        logger.info("-" * 80)
        logger.info("[VectorSearch] SEARCH RESULTS:")
        logger.info(f"[VectorSearch]   - Total Chunks: {len(all_chunks)}")
        logger.info(f"[VectorSearch]   - Parent Chunks Added: {len(parent_chunks)}")
        logger.info(f"[VectorSearch]   - Sources: {len(sources)}")
        logger.info(f"[VectorSearch]   - Chunk Types:")
        logger.info(f"[VectorSearch]       - Parent: {parent_count}")
        logger.info(f"[VectorSearch]       - Child: {child_count}")
        logger.info(f"[VectorSearch]       - Atomic: {atomic_count}")
        if sources:
            logger.info(f"[VectorSearch]   - Source Names: {list(sources)[:5]}{'...' if len(sources) > 5 else ''}")
        logger.info("=" * 80)
        logger.info("[VectorSearch] ========== UNIFIED VECTOR SEARCH COMPLETE ==========")
        logger.info("=" * 80)

        # Record timing
        if timing_metrics:
            timing_metrics.record("vector_search", (time.time() - vector_search_start) * 1000)

        return UnifiedSearchResult(
            chunks=all_chunks,
            parent_chunks=parent_chunks,
            sources=sources,
            total_count=total_count,
            retrieval_stats=stats
        )

    except Exception as e:
        logger.error("=" * 80)
        logger.error("[VectorSearch] ========== SEARCH FAILED ==========")
        logger.error("=" * 80)
        logger.error(f"[VectorSearch] Error: {e}")
        logger.error("=" * 80)
        return UnifiedSearchResult(
            chunks=[],
            parent_chunks=[],
            sources=set(),
            total_count=0,
            retrieval_stats={
                "query": query,
                "error": str(e)
            }
        )


def _search_per_document(
    vectorstore: QdrantVectorStore,
    query: str,
    document_ids: List[str],
    k: int,
    fetch_k: int,
    lambda_mult: float
) -> tuple:
    """
    Search with per-document balanced selection using MMR.

    Retrieves chunks from each document separately to ensure all documents
    are represented in the results.

    Returns:
        Tuple of (chunks, seen_chunk_ids, sources, child_chunk_ids)
    """
    # Calculate chunks per document (distribute k evenly)
    chunks_per_doc = max(2, k // len(document_ids))
    fetch_per_doc = max(10, fetch_k // len(document_ids))

    logger.info(
        f"[UnifiedSearch] Per-document selection: {chunks_per_doc} chunks from each of "
        f"{len(document_ids)} documents"
    )

    all_chunks = []
    seen_chunk_ids = set()
    sources = set()
    child_chunk_ids = []

    for doc_id in document_ids:
        try:
            # Create filter for this document only
            doc_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.document_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            )

            # Check if document has any chunks
            test_docs = vectorstore.similarity_search(query, k=1, filter=doc_filter)
            if not test_docs:
                logger.debug(f"[UnifiedSearch] No chunks found for document_id: {doc_id}")
                continue

            # Search with MMR for this document
            docs = vectorstore.max_marginal_relevance_search(
                query,
                k=chunks_per_doc,
                fetch_k=fetch_per_doc,
                filter=doc_filter,
                lambda_mult=lambda_mult
            )

            source_name = docs[0].metadata.get('source', doc_id[:8]) if docs else doc_id[:8]
            logger.debug(f"[UnifiedSearch] Document '{source_name}': Retrieved {len(docs)} chunks")

            # Process and deduplicate chunks
            for doc in docs:
                chunk_data, chunk_id, is_child = _process_document_to_chunk(doc)

                if chunk_id and chunk_id in seen_chunk_ids:
                    continue

                all_chunks.append(chunk_data)
                if chunk_id:
                    seen_chunk_ids.add(chunk_id)
                sources.add(chunk_data["metadata"].get("source", "Unknown"))

                if is_child:
                    child_chunk_ids.append(chunk_id)

        except Exception as e:
            logger.error(f"[UnifiedSearch] Error retrieving from document {doc_id}: {e}")
            continue

    # Trim to k if we got more than requested
    if len(all_chunks) > k:
        logger.debug(f"[UnifiedSearch] Got {len(all_chunks)} chunks, trimming to {k}")
        all_chunks = all_chunks[:k]

    logger.info(
        f"[UnifiedSearch] Per-document result: {len(all_chunks)} chunks from "
        f"{len(set(c['metadata'].get('document_id', 'Unknown') for c in all_chunks))} documents"
    )

    return all_chunks, seen_chunk_ids, sources, child_chunk_ids


def _search_all_documents(
    vectorstore: QdrantVectorStore,
    query: str,
    document_ids: List[str],
    k: int,
    fetch_k: int,
    lambda_mult: float
) -> tuple:
    """
    Search across all documents with MMR.

    Used when per_document_selection is False or there's only one document.

    Returns:
        Tuple of (chunks, seen_chunk_ids, sources, child_chunk_ids)
    """
    # Build filter for all accessible documents
    doc_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.document_id",
                match=models.MatchAny(any=document_ids),
            )
        ]
    )

    # Perform MMR search
    docs = vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
        filter=doc_filter,
        lambda_mult=lambda_mult
    )

    logger.info(f"[UnifiedSearch] All-document search: Retrieved {len(docs)} chunks")

    all_chunks = []
    seen_chunk_ids = set()
    sources = set()
    child_chunk_ids = []

    for doc in docs:
        chunk_data, chunk_id, is_child = _process_document_to_chunk(doc)

        if chunk_id and chunk_id in seen_chunk_ids:
            continue

        all_chunks.append(chunk_data)
        if chunk_id:
            seen_chunk_ids.add(chunk_id)
        sources.add(chunk_data["metadata"].get("source", "Unknown"))

        if is_child:
            child_chunk_ids.append(chunk_id)

    return all_chunks, seen_chunk_ids, sources, child_chunk_ids


def _process_document_to_chunk(doc: Document) -> tuple:
    """
    Convert a Document to chunk dict format with metadata.

    Returns:
        Tuple of (chunk_dict, chunk_id, is_child)
    """
    chunk_id = doc.metadata.get("chunk_id", "")
    parent_chunk_id = doc.metadata.get("parent_chunk_id")
    is_parent_chunk = doc.metadata.get("is_parent_chunk", False)

    chunk_data = {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
        "chunk_id": chunk_id,
        "is_parent": is_parent_chunk,
    }

    # Track if this is a child chunk (has a parent)
    is_child = bool(parent_chunk_id)

    return chunk_data, chunk_id, is_child


def _fetch_parent_chunks(
    child_chunk_ids: List[str],
    seen_chunk_ids: set,
    sources: set
) -> List[Dict[str, Any]]:
    """
    Fetch parent chunks for child chunks.

    Args:
        child_chunk_ids: List of child chunk IDs that have parents
        seen_chunk_ids: Set of already seen chunk IDs (will be updated)
        sources: Set of sources (will be updated)

    Returns:
        List of parent chunk dicts
    """
    logger.info(f"[UnifiedSearch] Fetching parents for {len(child_chunk_ids)} child chunks")

    parent_docs = get_parent_chunks_for_children(child_chunk_ids, source_names=None)
    parent_chunks = []
    already_present_count = 0

    for doc in parent_docs:
        parent_id = doc.metadata.get("chunk_id", "")

        if parent_id in seen_chunk_ids:
            already_present_count += 1
            continue

        parent_chunk = {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "chunk_id": parent_id,
            "is_parent": True,
            "is_context_expansion": True,  # Mark as expanded context
        }
        parent_chunks.append(parent_chunk)
        seen_chunk_ids.add(parent_id)
        sources.add(doc.metadata.get("source", "Unknown"))

    if parent_chunks:
        logger.info(
            f"[UnifiedSearch] Added {len(parent_chunks)} parent chunks "
            f"({already_present_count} already present, skipped)"
        )

    return parent_chunks


def clear_collection():
    """Clear all documents from the collection."""
    client = get_qdrant_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted collection: {COLLECTION_NAME}")
        # Recreate empty collection
        ensure_collection_exists(client)
    except Exception as e:
        logger.error(f"Error clearing collection: {e}")


def get_collection_info() -> dict:
    """Get information about the collection."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(COLLECTION_NAME)
        return {
            "name": COLLECTION_NAME,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
        }
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        return {"name": COLLECTION_NAME, "error": str(e)}


def delete_documents_by_source(source_name: str) -> int:
    """
    Delete all documents with a specific source from the collection.

    Uses multiple matching strategies to ensure all related vectors are deleted:
    1. Exact match on source_name
    2. Prefix match for sources with _page_N suffixes
    3. Match with file extension variations

    Args:
        source_name: The source name to delete (e.g., document folder name without extension).

    Returns:
        Number of deleted documents.
    """
    client = get_qdrant_client()
    total_deleted = 0

    try:
        # Strategy 1: Exact match on source_name
        try:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.source",
                                match=models.MatchValue(value=source_name),
                            )
                        ]
                    )
                ),
            )
            logger.info(f"[VectorDelete] Deleted vectors with exact source match: '{source_name}'")
        except Exception as e:
            logger.warning(f"[VectorDelete] Exact match deletion failed: {e}")

        # Strategy 2: Scroll and delete vectors with source starting with source_name
        # This catches sources like "filename_page_0", "filename.pdf_page_1", etc.
        points_to_delete = []
        offset = None

        while True:
            result = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=["metadata"],
                with_vectors=False
            )

            points, offset = result
            if not points:
                break

            for point in points:
                source = point.payload.get("metadata", {}).get("source", "")
                # Match if source starts with source_name (handles _page_N suffix)
                # Also match if source starts with source_name + extension (handles .pdf_page_N, etc.)
                if source == source_name or source.startswith(f"{source_name}_page_") or source.startswith(f"{source_name}."):
                    points_to_delete.append(point.id)

            if offset is None:
                break

        if points_to_delete:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.PointIdsList(points=points_to_delete)
            )
            total_deleted = len(points_to_delete)
            logger.info(f"[VectorDelete] Deleted {total_deleted} vectors with source prefix: '{source_name}'")
        else:
            logger.info(f"[VectorDelete] No vectors found with source prefix: '{source_name}'")

        return total_deleted

    except Exception as e:
        logger.error(f"[VectorDelete] Error deleting documents by source '{source_name}': {e}")
        return 0


def delete_documents_by_file_path(file_path: str) -> int:
    """
    Delete all documents with a specific file_path from the collection.

    Uses prefix matching to catch all related files in the same directory.

    Args:
        file_path: The file path to delete (can be a directory path or file path).

    Returns:
        Number of deleted documents.
    """
    client = get_qdrant_client()
    total_deleted = 0

    try:
        # Strategy 1: Exact match
        try:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.file_path",
                                match=models.MatchValue(value=file_path),
                            )
                        ]
                    )
                ),
            )
            logger.info(f"[VectorDelete] Deleted vectors with exact file_path: '{file_path}'")
        except Exception as e:
            logger.warning(f"[VectorDelete] Exact file_path match failed: {e}")

        # Strategy 2: Scroll and delete by file_path prefix (catches all files in directory)
        points_to_delete = []
        offset = None

        while True:
            result = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=["metadata"],
                with_vectors=False
            )

            points, offset = result
            if not points:
                break

            for point in points:
                point_file_path = point.payload.get("metadata", {}).get("file_path", "")
                # Match if file_path starts with the given path (directory matching)
                if point_file_path and (point_file_path == file_path or point_file_path.startswith(f"{file_path}/")):
                    points_to_delete.append(point.id)

            if offset is None:
                break

        if points_to_delete:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.PointIdsList(points=points_to_delete)
            )
            total_deleted = len(points_to_delete)
            logger.info(f"[VectorDelete] Deleted {total_deleted} vectors with file_path prefix: '{file_path}'")

        return total_deleted

    except Exception as e:
        logger.error(f"[VectorDelete] Error deleting documents by file_path '{file_path}': {e}")
        return 0


def delete_documents_by_document_id(document_id: str) -> int:
    """
    Delete all documents with a specific document_id from the collection.

    This is the most reliable deletion method since document_id is always
    stored in the vector metadata during indexing.

    Args:
        document_id: The document UUID as string.

    Returns:
        Number of deleted documents.
    """
    client = get_qdrant_client()
    total_deleted = 0

    try:
        # First scroll to find all points with this document_id
        points_to_delete = []
        offset = None

        while True:
            result = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False
            )

            points, offset = result
            if not points:
                break

            points_to_delete.extend([p.id for p in points])

            if offset is None:
                break

        if points_to_delete:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.PointIdsList(points=points_to_delete)
            )
            total_deleted = len(points_to_delete)
            logger.info(f"[VectorDelete] Deleted {total_deleted} vectors by document_id: '{document_id}'")
        else:
            logger.info(f"[VectorDelete] No vectors found for document_id: '{document_id}'")

        return total_deleted

    except Exception as e:
        logger.error(f"[VectorDelete] Error deleting documents by document_id '{document_id}': {e}")
        return 0


def is_document_indexed(source_name: str) -> bool:
    """
    Check if a document has any embeddings in the vector database.

    Args:
        source_name: The source name to check (e.g., document folder name).

    Returns:
        True if the document has embeddings, False otherwise.
    """
    client = get_qdrant_client()
    try:
        # Use scroll with limit 1 to check if any points exist with this source
        result = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=source_name),
                    )
                ]
            ),
            limit=1,
        )
        points, _ = result
        return len(points) > 0
    except Exception as e:
        logger.error(f"Error checking if document '{source_name}' is indexed: {e}")
        return False


# =============================================================================
# Document Metadata Search (via documents collection summary chunks)
# =============================================================================
#
# DATA SOURCES OF TRUTH:
# - Qdrant chunk metadata is a READ-ONLY CACHE populated during indexing
# - Authoritative sources are in PostgreSQL:
#   - schema_type: documents_data.schema_type
#   - vendor_name/customer_name: documents_data.header_data
#   - document_types: documents.document_metadata['document_types']
# - Qdrant metadata is used for semantic search/discovery, not as source of truth
#
# =============================================================================

# Configuration for document routing search
METADATA_SEARCH_TOP_K = int(os.getenv("METADATA_SEARCH_TOP_K", "15"))
METADATA_SCORE_THRESHOLD = float(os.getenv("METADATA_SCORE_THRESHOLD", "0.4"))


def search_document_metadata(
    query_text: str,
    top_k: int = None,
    score_threshold: float = None,
    accessible_document_ids: List[str] = None
) -> List[dict]:
    """
    Search for relevant documents by querying summary chunks in the documents collection.

    This function searches the main documents collection for tabular_summary chunks
    which contain rich metadata about each document (vendor, customer, totals, etc.).

    Args:
        query_text: Query text to search for (will be embedded)
        top_k: Number of results to return (default: METADATA_SEARCH_TOP_K)
        score_threshold: Minimum score threshold (default: METADATA_SCORE_THRESHOLD)
        accessible_document_ids: List of document IDs the user can access (for access control).
                                 If provided, only documents with these IDs will be returned.
                                 If empty list, returns empty results.
                                 If None, returns all matching documents.

    Returns:
        List of dictionaries with document info and scores:
        [{"document_id": str, "source_name": str, "score": float, "payload": dict}, ...]
    """
    if top_k is None:
        top_k = METADATA_SEARCH_TOP_K
    if score_threshold is None:
        score_threshold = METADATA_SCORE_THRESHOLD

    # Access control: empty list means no access
    if accessible_document_ids is not None and len(accessible_document_ids) == 0:
        logger.warning("[DocumentMetadataSearch] Access control: no accessible documents - returning empty")
        return []

    try:
        client = get_qdrant_client()
        embeddings = get_embeddings()

        # Embed the query
        query_vector = embeddings.embed_query(query_text)

        # Build filter conditions
        must_conditions = []

        # Filter for summary chunks which contain document metadata
        must_conditions.append(
            models.FieldCondition(
                key="metadata.chunk_type",
                match=models.MatchValue(value="tabular_summary")
            )
        )

        # Add access control filter by document_id
        if accessible_document_ids is not None and len(accessible_document_ids) > 0:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.document_id",
                    match=models.MatchAny(any=accessible_document_ids)
                )
            )
            logger.debug(f"[DocumentMetadataSearch] Access control: filtering to {len(accessible_document_ids)} document IDs")

        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        # Search using query_points
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True
        )

        # Format results - extract metadata from the chunk's metadata field
        formatted_results = []
        seen_doc_ids = set()  # Deduplicate by document_id

        for point in results.points:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})

            doc_id = metadata.get("document_id", "")
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            # Extract document info from chunk metadata
            result = {
                "document_id": doc_id,
                "source_name": metadata.get("source", ""),
                "filename": metadata.get("filename", ""),
                "score": point.score,
                "document_types": [metadata.get("schema_type", "document")],
                "subject_name": metadata.get("source", ""),
                "confidence": 0.95,  # Summary chunks have high confidence
                "payload": {
                    # Include key metadata fields for entity matching
                    "embedding_text": payload.get("page_content", ""),
                    "vendor_name": metadata.get("vendor_name"),
                    "customer_name": metadata.get("customer_name"),
                    "invoice_number": metadata.get("invoice_number"),
                    "invoice_date": metadata.get("invoice_date"),
                    "total_amount": metadata.get("total_amount"),
                    "schema_type": metadata.get("schema_type"),
                    "is_tabular": metadata.get("is_tabular", False),
                    "row_count": metadata.get("row_count", 0),
                    "columns": metadata.get("columns", []),
                }
            }

            # Include tabular-specific fields at top level for easier access
            if metadata.get("is_tabular"):
                result["is_tabular"] = True
                result["columns"] = metadata.get("columns", [])
                result["row_count"] = metadata.get("row_count", 0)
                result["schema_type"] = metadata.get("schema_type")

            formatted_results.append(result)

        logger.info(
            f"[DocumentMetadataSearch] Search returned {len(formatted_results)} results "
            f"(query: '{query_text[:50]}...', threshold: {score_threshold})"
        )

        return formatted_results

    except Exception as e:
        logger.error(f"[DocumentMetadataSearch] Error searching: {e}", exc_info=True)
        return []


def format_query_for_metadata_search(
    query_text: str,
    entities: List[str] = None,
    topics: List[str] = None,
    document_type_hints: List[str] = None
) -> str:
    """
    Format a user query into text suitable for document metadata search.

    Args:
        query_text: Original user query
        entities: Extracted entities from query
        topics: Extracted topics from query
        document_type_hints: Hints about document type

    Returns:
        Formatted query text for embedding
    """
    parts = []

    # Add document type hints first (high priority)
    if document_type_hints:
        parts.append(f"Document type: {', '.join(document_type_hints)}")

    # Add the original query
    parts.append(query_text)

    # Add topics
    if topics:
        parts.append(f"Topics: {', '.join(topics[:5])}")

    # Add entities
    if entities:
        parts.append(f"Entities: {', '.join(entities[:5])}")

    return ". ".join(parts)


# =============================================================================
# Query Relevance Scoring for Chunks
# =============================================================================

# Configuration for query relevance scoring
CONTEXT_ENABLE_QUERY_SCORING = os.getenv("CONTEXT_ENABLE_QUERY_SCORING", "true").lower() == "true"


def _compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between 0 and 1
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def score_chunks_by_query(
    chunks: List[Dict[str, Any]],
    query: str,
    query_embedding: List[float] = None
) -> List[Dict[str, Any]]:
    """
    Score chunks by query relevance using cosine similarity.

    This function computes relevance scores for a list of chunks against a query
    by comparing the query embedding with stored chunk embeddings in Qdrant.

    Args:
        chunks: List of chunk dictionaries with chunk_id in metadata
        query: The query string (used if query_embedding not provided)
        query_embedding: Optional pre-computed query embedding

    Returns:
        Same chunks with _score field added to metadata, sorted by score descending
    """
    if not chunks:
        return chunks

    if not CONTEXT_ENABLE_QUERY_SCORING:
        logger.debug("[QueryScoring] Disabled via CONTEXT_ENABLE_QUERY_SCORING")
        return chunks

    client = get_qdrant_client()
    embeddings = get_embeddings()

    # Get query embedding if not provided
    if query_embedding is None:
        if not query:
            logger.warning("[QueryScoring] No query or embedding provided")
            return chunks
        try:
            query_embedding = embeddings.embed_query(query)
            logger.debug(f"[QueryScoring] Embedded query: '{query[:50]}...' -> {len(query_embedding)} dims")
        except Exception as e:
            logger.error(f"[QueryScoring] Failed to embed query: {e}")
            return chunks

    # Collect chunk IDs for batch retrieval
    chunk_id_to_idx = {}
    for idx, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id") or chunk.get("metadata", {}).get("chunk_id", "")
        if chunk_id:
            chunk_id_to_idx[chunk_id] = idx

    if not chunk_id_to_idx:
        logger.warning("[QueryScoring] No valid chunk IDs found")
        return chunks

    logger.info(f"[QueryScoring] Scoring {len(chunk_id_to_idx)} chunks against query")

    try:
        # Build filter to retrieve specific chunks with their vectors
        should_conditions = [
            models.FieldCondition(
                key="metadata.chunk_id",
                match=models.MatchValue(value=chunk_id),
            )
            for chunk_id in chunk_id_to_idx.keys()
        ]

        filter_condition = models.Filter(should=should_conditions)

        # Scroll through chunks with vectors
        result, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=filter_condition,
            limit=len(chunk_id_to_idx),
            with_payload=True,
            with_vectors=True,  # Get the stored embeddings
        )

        # Calculate scores for each chunk
        scores_computed = 0
        for point in result:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            chunk_id = metadata.get("chunk_id", "")

            if chunk_id not in chunk_id_to_idx:
                continue

            # Get stored vector
            stored_vector = point.vector
            if stored_vector is None:
                continue

            # Compute cosine similarity
            score = _compute_cosine_similarity(query_embedding, stored_vector)

            # Add score to the chunk
            idx = chunk_id_to_idx[chunk_id]
            if "metadata" not in chunks[idx]:
                chunks[idx]["metadata"] = {}
            chunks[idx]["metadata"]["_score"] = score
            chunks[idx]["_score"] = score  # Also at top level for convenience
            scores_computed += 1

        logger.info(f"[QueryScoring] Computed scores for {scores_computed}/{len(chunk_id_to_idx)} chunks")

        # Sort chunks by score (highest first)
        chunks_with_scores = [(c, c.get("_score", 0.0)) for c in chunks]
        chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_chunks = [c for c, _ in chunks_with_scores]

        # Log score distribution
        if sorted_chunks:
            scores = [c.get("_score", 0.0) for c in sorted_chunks]
            if scores:
                logger.info(
                    f"[QueryScoring] Score distribution: max={max(scores):.3f}, "
                    f"min={min(scores):.3f}, avg={sum(scores)/len(scores):.3f}"
                )

        return sorted_chunks

    except Exception as e:
        logger.error(f"[QueryScoring] Failed to score chunks: {e}")
        return chunks


def score_and_merge_chunks(
    chunk_lists: List[List[Dict[str, Any]]],
    query: str,
    deduplicate: bool = True
) -> List[Dict[str, Any]]:
    """
    Merge multiple chunk lists, deduplicate, and score by query relevance.

    This is the main entry point for centralized scoring. It:
    1. Merges all chunk lists
    2. Deduplicates by chunk_id
    3. Scores all unique chunks against the query
    4. Returns sorted by relevance score

    Args:
        chunk_lists: List of chunk lists (e.g., [vector_chunks, graph_chunks, parent_chunks])
        query: The query string for scoring
        deduplicate: Whether to deduplicate by chunk_id (default: True)

    Returns:
        Merged, deduplicated, scored, and sorted chunks
    """
    if not chunk_lists:
        return []

    # Merge all chunks
    all_chunks = []
    seen_ids = set()

    for chunk_list in chunk_lists:
        for chunk in chunk_list:
            if deduplicate:
                chunk_id = chunk.get("chunk_id") or chunk.get("metadata", {}).get("chunk_id", "")
                if chunk_id and chunk_id in seen_ids:
                    continue
                if chunk_id:
                    seen_ids.add(chunk_id)
            all_chunks.append(chunk)

    logger.info(
        f"[ScoreAndMerge] Merged {sum(len(cl) for cl in chunk_lists)} chunks "
        f"into {len(all_chunks)} unique chunks"
    )

    # Score and sort
    return score_chunks_by_query(all_chunks, query)
