"""
Qdrant Vector Store configuration for RAG.
Connects to Qdrant running at localhost:6333.
"""

import os
import logging
from typing import Optional, List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from .local_qwen_embedding import LocalQwen3Embedding

logger = logging.getLogger(__name__)

# Collection names for document embeddings
COLLECTION_NAME = "documents"
METADATA_COLLECTION_NAME = "metadatas"

# Configuration for metadata vector search
METADATA_SEARCH_TOP_K = int(os.getenv("METADATA_SEARCH_TOP_K", "15"))
METADATA_SCORE_THRESHOLD = float(os.getenv("METADATA_SCORE_THRESHOLD", "0.4"))  # Lower threshold for better recall
METADATA_FALLBACK_MIN_RESULTS = int(os.getenv("METADATA_FALLBACK_MIN_RESULTS", "3"))

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


def _create_per_source_retriever(
    vectorstore,
    source_names: List[str],
    k: int,
    fetch_k: int,
    lambda_mult: float = None,
    document_ids: list = None
):
    """
    Create a custom retriever that retrieves chunks from each source separately,
    then combines them to ensure all sources are represented.

    This prevents MMR from excluding entire sources due to diversity filtering.

    Args:
        vectorstore: The Qdrant vectorstore instance
        source_names: List of source names to retrieve from
        k: Total number of chunks to retrieve
        fetch_k: Number of candidates to fetch per source for MMR
        lambda_mult: MMR diversity parameter
        document_ids: Optional list of document IDs for access control filtering

    Returns:
        A custom retriever that ensures per-source representation
    """
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from typing import List as TypingList, Optional as TypingOptional

    class PerDocumentRetriever(BaseRetriever):
        """Custom retriever that ensures all documents are represented using document_id filter only."""

        vectorstore: QdrantVectorStore
        document_ids: TypingList[str]  # Required - list of accessible document IDs
        k: int
        fetch_k: int
        lambda_mult: float

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> TypingList[Document]:
            """Retrieve documents with per-document selection using document_id filter only."""

            # Calculate chunks per document (distribute k evenly)
            chunks_per_doc = max(2, self.k // len(self.document_ids))
            fetch_per_doc = max(10, self.fetch_k // len(self.document_ids))

            logger.info(
                f"[PerDocumentRetriever] Retrieving {chunks_per_doc} chunks from each of "
                f"{len(self.document_ids)} documents (total target: {self.k})"
            )

            all_docs = []
            seen_chunk_ids = set()

            # Retrieve from each document separately using document_id filter only
            for doc_id in self.document_ids:
                try:
                    # Create filter using only document_id
                    doc_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.document_id",
                                match=models.MatchValue(value=doc_id),
                            )
                        ]
                    )

                    # First, check how many chunks exist for this document
                    try:
                        test_docs = self.vectorstore.similarity_search(
                            query,
                            k=1,
                            filter=doc_filter
                        )
                        if not test_docs:
                            logger.debug(
                                f"[PerDocumentRetriever] No chunks found for document_id: {doc_id}"
                            )
                            continue
                    except Exception as e:
                        logger.warning(f"[PerDocumentRetriever] Cannot access document {doc_id}: {e}")
                        continue

                    # Search with MMR for this document only
                    docs = self.vectorstore.max_marginal_relevance_search(
                        query,
                        k=chunks_per_doc,
                        fetch_k=fetch_per_doc,
                        filter=doc_filter,
                        lambda_mult=self.lambda_mult if self.lambda_mult is not None else 0.5
                    )

                    source_name = docs[0].metadata.get('source', doc_id[:8]) if docs else doc_id[:8]
                    logger.info(
                        f"[PerDocumentRetriever] Document '{source_name}' ({doc_id[:8]}...): Retrieved {len(docs)} chunks"
                    )

                    # Deduplicate by chunk_id
                    added_count = 0
                    for doc in docs:
                        chunk_id = doc.metadata.get("chunk_id")
                        if chunk_id and chunk_id not in seen_chunk_ids:
                            all_docs.append(doc)
                            seen_chunk_ids.add(chunk_id)
                            added_count += 1
                        elif not chunk_id:
                            all_docs.append(doc)
                            added_count += 1

                    logger.debug(
                        f"[PerDocumentRetriever] Document '{source_name}': Added {added_count}/{len(docs)} chunks"
                    )

                except Exception as e:
                    logger.error(f"[PerDocumentRetriever] Error retrieving from document {doc_id}: {e}")
                    continue

            # Trim to k if needed
            if len(all_docs) > self.k:
                logger.info(f"[PerDocumentRetriever] Got {len(all_docs)} chunks, trimming to {self.k}")
                all_docs = all_docs[:self.k]

            logger.info(
                f"[PerDocumentRetriever] Final result: {len(all_docs)} chunks from "
                f"{len(set(doc.metadata.get('document_id', 'Unknown') for doc in all_docs))} documents"
            )

            return all_docs

    # Convert document_ids to strings for the retriever
    doc_id_strings = [str(doc_id) for doc_id in document_ids] if document_ids else []

    return PerDocumentRetriever(
        vectorstore=vectorstore,
        document_ids=doc_id_strings,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult if lambda_mult is not None else 0.5
    )


def get_retriever_with_sources(
    k: int = 8,
    fetch_k: int = 30,
    source_names: List[str] = None,  # Deprecated - kept for backward compatibility
    lambda_mult: float = None,
    per_source_selection: bool = True,  # Renamed semantically to per_document_selection
    document_ids: list = None
):
    """
    Get a retriever from the vectorstore with document_id filtering.

    IMPORTANT: This function now uses document_id as the primary filter.
    source_names is deprecated and ignored.

    Args:
        k: Number of documents to retrieve (total across all documents).
        fetch_k: Number of documents to fetch for MMR.
        source_names: DEPRECATED - ignored. Use document_ids instead.
        lambda_mult: MMR diversity parameter (0.0 = max diversity, 1.0 = max relevance).
                     If None, uses default (0.5).
        per_source_selection: If True and document_ids provided, retrieve chunks
                              from each document separately to ensure all documents
                              are represented.
        document_ids: List of document IDs to filter results (required for access control).

    Returns:
        A LangChain retriever configured for MMR search with document_id filtering.
    """
    vectorstore = get_vectorstore()

    # If per-document selection is enabled and we have document_ids, use custom retriever
    if per_source_selection and document_ids and len(document_ids) > 1:
        logger.info(
            f"[Retriever] Using per-document selection for {len(document_ids)} documents "
            f"to ensure all documents are represented"
        )
        return _create_per_source_retriever(
            vectorstore=vectorstore,
            source_names=None,  # Not used anymore
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            document_ids=document_ids
        )

    # Otherwise, use standard MMR retriever with document_id filter only
    search_kwargs = {"k": k, "fetch_k": fetch_k}

    # Build filter using document_id only
    if document_ids is not None and len(document_ids) > 0:
        doc_id_strings = [str(doc_id) for doc_id in document_ids]
        search_kwargs["filter"] = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.document_id",
                    match=models.MatchAny(any=doc_id_strings),
                )
            ]
        )

    # Add lambda_mult if specified (controls relevance vs diversity tradeoff)
    if lambda_mult is not None:
        search_kwargs["lambda_mult"] = lambda_mult

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )


def get_retriever(
    k: int = 8,
    fetch_k: int = 30,
    source_filter: str = None,  # DEPRECATED - ignored
    document_ids: list = None
):
    """
    Get a retriever from the vectorstore using document_id filter only.

    Args:
        k: Number of documents to retrieve.
        fetch_k: Number of documents to fetch for MMR.
        source_filter: DEPRECATED - ignored. Use document_ids instead.
        document_ids: List of document IDs to filter results (for access control).

    Returns:
        A LangChain retriever configured for MMR search.
    """
    vectorstore = get_vectorstore()
    search_kwargs = {"k": k, "fetch_k": fetch_k}

    # Build filter using document_id only
    filter_conditions = []

    # Add document ID filter for access control
    if document_ids is not None and len(document_ids) > 0:
        # Convert UUIDs to strings if necessary
        doc_id_strings = [str(doc_id) for doc_id in document_ids]
        filter_conditions.append(
            models.FieldCondition(
                key="metadata.document_id",
                match=models.MatchAny(any=doc_id_strings),
            )
        )

    # Apply filter if any conditions exist
    if filter_conditions:
        search_kwargs["filter"] = models.Filter(must=filter_conditions)

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )


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

    Args:
        source_name: The source name to delete (e.g., document folder name).

    Returns:
        Number of deleted documents.
    """
    client = get_qdrant_client()
    try:
        # Use scroll to find points with matching source metadata
        # Qdrant filter by metadata
        result = client.delete(
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
        logger.info(f"Deleted documents with source '{source_name}' from collection")
        return result
    except Exception as e:
        logger.error(f"Error deleting documents by source '{source_name}': {e}")
        return 0


def delete_documents_by_file_path(file_path: str) -> int:
    """
    Delete all documents with a specific file_path from the collection.

    Args:
        file_path: The file path to delete.

    Returns:
        Number of deleted documents.
    """
    client = get_qdrant_client()
    try:
        result = client.delete(
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
        logger.info(f"Deleted documents with file_path '{file_path}' from collection")
        return result
    except Exception as e:
        logger.error(f"Error deleting documents by file_path '{file_path}': {e}")
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
# Document Metadata Vector Collection Functions
# =============================================================================

def ensure_metadata_collection_exists(client: QdrantClient = None, embedding_dim: int = 2560):
    """
    Ensure the document metadata collection exists, create if not.

    Args:
        client: The Qdrant client. If None, will get the singleton.
        embedding_dim: Dimension of the embedding vectors.
    """
    if client is None:
        client = get_qdrant_client()

    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if METADATA_COLLECTION_NAME not in collection_names:
        logger.info(f"Creating metadata collection: {METADATA_COLLECTION_NAME}")
        client.create_collection(
            collection_name=METADATA_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"Metadata collection {METADATA_COLLECTION_NAME} created successfully")
    else:
        logger.debug(f"Metadata collection {METADATA_COLLECTION_NAME} already exists")


def format_metadata_for_embedding(metadata: dict, source_name: str) -> str:
    """
    Format document metadata into a text string suitable for embedding.

    IMPORTANT: Keep it SHORT and FOCUSED for better vector search accuracy!
    - Source name and document type are the most distinctive identifiers
    - Key entities provide specific names/terms for matching
    - Topics categorize the document type
    - Summary is EXCLUDED to avoid diluting the semantic signal with generic words

    Args:
        metadata: Document metadata dictionary
        source_name: Document source name

    Returns:
        Formatted text string for embedding (concise, ~200-300 chars)
    """
    document_type = metadata.get("document_type", "document")
    subject_name = metadata.get("subject_name", source_name)

    # Join key entities - limit to top 8 for conciseness
    key_entities = metadata.get("key_entities", [])
    entities_str = ", ".join([
        e.get("name", "") for e in key_entities[:8]
        if isinstance(e, dict) and e.get("name")
    ]) if key_entities else ""

    # Join topics - limit to top 5
    topics = metadata.get("topics", [])
    topics_str = ", ".join(topics[:5]) if topics else ""

    # Format the embedding text - CONCISE for better semantic matching
    # Format: "source_name. Key entities: ... document_type: subject. Topics: ..."
    # NO SUMMARY - summaries contain generic words that dilute entity matching
    parts = []

    # 1. Source name FIRST (most distinctive)
    parts.append(source_name)

    # 2. Entities SECOND (key identifiers)
    if entities_str:
        parts.append(f"Key entities: {entities_str}")

    # 3. Document type and subject
    parts.append(f"{document_type}: {subject_name}")

    # 4. Topics
    if topics_str:
        parts.append(f"Topics: {topics_str}")

    # NOTE: Summary is intentionally excluded to keep embedding text focused
    # Long summaries introduce generic words that confuse vector similarity

    return ". ".join(parts)


def upsert_document_metadata_embedding(
    document_id: str,
    source_name: str,
    filename: str,
    metadata: dict
) -> bool:
    """
    Embed document metadata and upsert to the metadata collection.

    Args:
        document_id: Document UUID as string
        source_name: Document source name (filename without extension)
        filename: Original filename with extension
        metadata: Document metadata dictionary

    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_qdrant_client()
        embeddings = get_embeddings()

        # Ensure collection exists
        ensure_metadata_collection_exists(client)

        # Format metadata for embedding
        embedding_text = format_metadata_for_embedding(metadata, source_name)
        logger.debug(f"[MetadataVector] Embedding text for {source_name}: {embedding_text[:200]}...")

        # Generate embedding
        embedding_vector = embeddings.embed_query(embedding_text)

        # Prepare payload
        payload = {
            "document_id": document_id,
            "source_name": source_name,
            "filename": filename,
            "document_type": metadata.get("document_type", "unknown"),
            "subject_name": metadata.get("subject_name", source_name),
            "confidence": metadata.get("confidence", 0.0),
            "topics": metadata.get("topics", []),
            "embedding_text": embedding_text,  # Store for debugging
        }

        # Add date information if available
        dates = metadata.get("dates", {})
        if dates:
            primary_date = dates.get("primary_date", {})
            if primary_date:
                payload["primary_date"] = primary_date.get("normalized")
                payload["date_year"] = primary_date.get("year")
                payload["date_month"] = primary_date.get("month")

        # Upsert to collection (use document_id as point ID for idempotent updates)
        client.upsert(
            collection_name=METADATA_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=document_id,
                    vector=embedding_vector,
                    payload=payload
                )
            ]
        )

        logger.info(
            f"[MetadataVector] Upserted metadata embedding for {source_name} "
            f"(type={metadata.get('document_type')}, confidence={metadata.get('confidence', 0):.2f})"
        )
        return True

    except Exception as e:
        logger.error(f"[MetadataVector] Error upserting metadata for {source_name}: {e}", exc_info=True)
        return False


def delete_document_metadata_embedding(document_id: str) -> bool:
    """
    Delete document metadata embedding from the collection.

    Args:
        document_id: Document UUID as string

    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_qdrant_client()

        client.delete(
            collection_name=METADATA_COLLECTION_NAME,
            points_selector=models.PointIdsList(
                points=[document_id]
            )
        )

        logger.info(f"[MetadataVector] Deleted metadata embedding for document {document_id}")
        return True

    except Exception as e:
        logger.error(f"[MetadataVector] Error deleting metadata for {document_id}: {e}")
        return False


def delete_metadata_by_source_name(source_name: str) -> int:
    """
    Delete all document metadata entries with a specific source_name from the collection.

    This is useful for cleaning up orphaned metadata records that weren't properly
    deleted when the document was removed.

    Args:
        source_name: The source name to delete (e.g., document name without extension)

    Returns:
        Number of records deleted
    """
    try:
        client = get_qdrant_client()

        # First, scroll to find all points with this source_name
        points_to_delete = []
        offset = None

        while True:
            result = client.scroll(
                collection_name=METADATA_COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_name",
                            match=models.MatchValue(value=source_name)
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

        if not points_to_delete:
            logger.info(f"[MetadataVector] No metadata found for source_name: {source_name}")
            return 0

        # Delete all found points
        client.delete(
            collection_name=METADATA_COLLECTION_NAME,
            points_selector=models.PointIdsList(points=points_to_delete)
        )

        logger.info(f"[MetadataVector] Deleted {len(points_to_delete)} metadata entries for source_name: {source_name}")
        return len(points_to_delete)

    except Exception as e:
        logger.error(f"[MetadataVector] Error deleting metadata by source_name '{source_name}': {e}")
        return 0


def cleanup_orphaned_metadata(valid_document_ids: List[str]) -> dict:
    """
    Remove metadata records that don't have corresponding documents in the database.

    Args:
        valid_document_ids: List of valid document IDs from the database

    Returns:
        Dictionary with cleanup statistics
    """
    try:
        client = get_qdrant_client()

        # Get all metadata points
        all_points = []
        offset = None

        while True:
            result = client.scroll(
                collection_name=METADATA_COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            points, offset = result
            if not points:
                break

            all_points.extend(points)

            if offset is None:
                break

        # Find orphaned points (those not in valid_document_ids)
        valid_ids_set = set(valid_document_ids)
        orphaned_points = []

        for point in all_points:
            # Point ID is the document_id
            point_id = str(point.id) if not isinstance(point.id, str) else point.id
            if point_id not in valid_ids_set:
                source_name = point.payload.get("source_name", "unknown") if point.payload else "unknown"
                orphaned_points.append({
                    "id": point_id,
                    "source_name": source_name
                })

        if not orphaned_points:
            logger.info("[MetadataVector] No orphaned metadata records found")
            return {"total": len(all_points), "orphaned": 0, "deleted": 0}

        # Delete orphaned points
        orphaned_ids = [p["id"] for p in orphaned_points]
        client.delete(
            collection_name=METADATA_COLLECTION_NAME,
            points_selector=models.PointIdsList(points=orphaned_ids)
        )

        logger.info(f"[MetadataVector] Cleaned up {len(orphaned_points)} orphaned metadata records")
        for p in orphaned_points[:10]:  # Log first 10
            logger.debug(f"[MetadataVector]   - Deleted orphaned: {p['source_name']} (id={p['id']})")

        return {
            "total": len(all_points),
            "orphaned": len(orphaned_points),
            "deleted": len(orphaned_points),
            "orphaned_sources": list(set(p["source_name"] for p in orphaned_points))
        }

    except Exception as e:
        logger.error(f"[MetadataVector] Error cleaning up orphaned metadata: {e}")
        return {"error": str(e)}


def search_document_metadata(
    query_text: str,
    top_k: int = None,
    score_threshold: float = None,
    accessible_document_ids: List[str] = None
) -> List[dict]:
    """
    Search for relevant documents by querying the metadata vector collection.

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
        logger.warning("[MetadataVector] Access control: no accessible documents - returning empty")
        return []

    try:
        client = get_qdrant_client()
        embeddings = get_embeddings()

        # Ensure collection exists
        ensure_metadata_collection_exists(client)

        # Check if collection has any points
        collection_info = client.get_collection(METADATA_COLLECTION_NAME)
        if collection_info.points_count == 0:
            logger.warning("[MetadataVector] Metadata collection is empty")
            return []

        # Embed the query
        query_vector = embeddings.embed_query(query_text)

        # Build filter for access control
        query_filter = None
        if accessible_document_ids is not None and len(accessible_document_ids) > 0:
            # Filter by document_id (the point ID in metadata collection is the document UUID)
            query_filter = models.Filter(
                must=[
                    models.HasIdCondition(
                        has_id=accessible_document_ids
                    )
                ]
            )
            logger.info(f"[MetadataVector] Access control: filtering to {len(accessible_document_ids)} document IDs")

        # Search using query_points (qdrant-client >= 1.7.0)
        results = client.query_points(
            collection_name=METADATA_COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True
        )

        # Format results
        formatted_results = []
        for point in results.points:
            formatted_results.append({
                "document_id": str(point.id),
                "source_name": point.payload.get("source_name"),
                "filename": point.payload.get("filename"),
                "score": point.score,
                "document_type": point.payload.get("document_type"),
                "subject_name": point.payload.get("subject_name"),
                "confidence": point.payload.get("confidence", 0),
                "payload": point.payload
            })

        logger.info(
            f"[MetadataVector] Search returned {len(formatted_results)} results "
            f"(query: '{query_text[:50]}...', threshold: {score_threshold})"
        )

        return formatted_results

    except Exception as e:
        logger.error(f"[MetadataVector] Error searching metadata: {e}", exc_info=True)
        return []


def format_query_for_metadata_search(
    query_text: str,
    entities: List[str] = None,
    topics: List[str] = None,
    document_type_hints: List[str] = None
) -> str:
    """
    Format a user query into text suitable for metadata vector search.

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


def get_metadata_collection_info() -> dict:
    """Get information about the metadata collection."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(METADATA_COLLECTION_NAME)
        return {
            "name": METADATA_COLLECTION_NAME,
            "points_count": info.points_count,
            "status": info.status.value,
        }
    except Exception as e:
        logger.error(f"Error getting metadata collection info: {e}")
        return {"name": METADATA_COLLECTION_NAME, "error": str(e)}
