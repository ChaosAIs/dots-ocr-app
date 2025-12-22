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
COLLECTION_NAME = "dots_ocr_documents"
METADATA_COLLECTION_NAME = "document_metadatas"

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



def get_chunks_by_ids(chunk_ids: List[str], source_names: List[str] = None) -> List[Document]:
    """
    Get full chunk content by chunk IDs from the main document collection.

    Args:
        chunk_ids: List of chunk IDs to retrieve.
        source_names: Optional list of source names to filter.

    Returns:
        List of Document objects with full chunk content.
    """
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


def _create_per_source_retriever(
    vectorstore,
    source_names: List[str],
    k: int,
    fetch_k: int,
    lambda_mult: float = None
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

    Returns:
        A custom retriever that ensures per-source representation
    """
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from typing import List as TypingList

    class PerSourceRetriever(BaseRetriever):
        """Custom retriever that ensures all sources are represented."""

        vectorstore: QdrantVectorStore
        source_names: TypingList[str]
        k: int
        fetch_k: int
        lambda_mult: float

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> TypingList[Document]:
            """Retrieve documents with per-source selection."""

            # Calculate chunks per source (distribute k evenly)
            chunks_per_source = max(2, k // len(self.source_names))
            fetch_per_source = max(10, self.fetch_k // len(self.source_names))

            logger.info(
                f"[PerSourceRetriever] Retrieving {chunks_per_source} chunks from each of "
                f"{len(self.source_names)} sources (total target: {k})"
            )

            all_docs = []
            seen_chunk_ids = set()

            # Retrieve from each source separately
            for source_name in self.source_names:
                try:
                    # Create filter for this specific source
                    source_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.source",
                                match=models.MatchValue(value=source_name),
                            )
                        ]
                    )

                    # First, check how many chunks exist for this source
                    try:
                        test_docs = self.vectorstore.similarity_search(
                            query,
                            k=1,
                            filter=source_filter
                        )
                        if not test_docs:
                            logger.warning(
                                f"[PerSourceRetriever] No chunks found for source: {source_name} "
                                f"(source may not exist in vectorstore or has no matching chunks)"
                            )
                            continue
                    except Exception as e:
                        logger.warning(f"[PerSourceRetriever] Cannot access source {source_name}: {e}")
                        continue

                    # Search with MMR for this source only
                    docs = self.vectorstore.max_marginal_relevance_search(
                        query,
                        k=chunks_per_source,
                        fetch_k=fetch_per_source,
                        filter=source_filter,
                        lambda_mult=self.lambda_mult if self.lambda_mult is not None else 0.5
                    )

                    # Log details about retrieved chunks
                    logger.info(
                        f"[PerSourceRetriever] Source '{source_name}': Retrieved {len(docs)} chunks"
                    )

                    # Deduplicate by chunk_id and log chunk details
                    added_count = 0
                    for i, doc in enumerate(docs, 1):
                        chunk_id = doc.metadata.get("chunk_id")
                        heading = doc.metadata.get("heading_path", "")
                        content_preview = doc.page_content[:100].replace('\n', ' ')

                        logger.debug(
                            f"[PerSourceRetriever]   Chunk {i}: {source_name} | "
                            f"heading='{heading}' | preview='{content_preview}...'"
                        )

                        if chunk_id and chunk_id not in seen_chunk_ids:
                            all_docs.append(doc)
                            seen_chunk_ids.add(chunk_id)
                            added_count += 1
                        elif not chunk_id:
                            # No chunk_id, add anyway
                            all_docs.append(doc)
                            added_count += 1
                        else:
                            logger.debug(f"[PerSourceRetriever]   Skipped duplicate chunk_id: {chunk_id}")

                    logger.info(
                        f"[PerSourceRetriever] Source '{source_name}': Added {added_count}/{len(docs)} chunks "
                        f"(after deduplication)"
                    )

                except Exception as e:
                    logger.error(f"[PerSourceRetriever] Error retrieving from source {source_name}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue

            # If we got more than k chunks, trim to k (keep highest scoring)
            if len(all_docs) > self.k:
                logger.info(
                    f"[PerSourceRetriever] Got {len(all_docs)} chunks, trimming to {self.k}"
                )
                all_docs = all_docs[:self.k]

            logger.info(
                f"[PerSourceRetriever] Final result: {len(all_docs)} chunks from "
                f"{len(set(doc.metadata.get('source', 'Unknown') for doc in all_docs))} sources"
            )

            return all_docs

    # Create and return the custom retriever
    return PerSourceRetriever(
        vectorstore=vectorstore,
        source_names=source_names,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult if lambda_mult is not None else 0.5
    )


def get_retriever_with_sources(
    k: int = 8,
    fetch_k: int = 30,
    source_names: List[str] = None,
    lambda_mult: float = None,
    per_source_selection: bool = True
):
    """
    Get a retriever from the vectorstore with optional source filtering.

    Args:
        k: Number of documents to retrieve (total across all sources).
        fetch_k: Number of documents to fetch for MMR.
        source_names: Optional list of source names to filter results.
        lambda_mult: MMR diversity parameter (0.0 = max diversity, 1.0 = max relevance).
                     If None, uses default (0.5).
        per_source_selection: If True and source_names is provided, retrieve chunks
                              from each source separately to ensure all sources are
                              represented. If False, use global MMR across all sources.

    Returns:
        A LangChain retriever configured for MMR search, or a custom retriever
        that ensures per-source representation.
    """
    vectorstore = get_vectorstore()

    # If per-source selection is enabled and we have multiple sources, use custom retriever
    if per_source_selection and source_names and len(source_names) > 1:
        logger.info(
            f"[Retriever] Using per-source selection for {len(source_names)} sources "
            f"to ensure all sources are represented"
        )
        return _create_per_source_retriever(
            vectorstore=vectorstore,
            source_names=source_names,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )

    # Otherwise, use standard MMR retriever
    search_kwargs = {"k": k, "fetch_k": fetch_k}

    # Add source filter if specified
    if source_names and len(source_names) > 0:
        # Use 'should' for multiple sources (OR logic) - match ANY of the specified sources
        # But wrap in 'must' to ensure the filter is applied (not optional)
        search_kwargs["filter"] = models.Filter(
            must=[
                models.Filter(
                    should=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=source_name),
                        )
                        for source_name in source_names
                    ]
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


def get_retriever(k: int = 8, fetch_k: int = 30, source_filter: str = None):
    """
    Get a retriever from the vectorstore.

    Args:
        k: Number of documents to retrieve.
        fetch_k: Number of documents to fetch for MMR.
        source_filter: Optional source name to filter results (e.g., document folder name).

    Returns:
        A LangChain retriever configured for MMR search.
    """
    vectorstore = get_vectorstore()
    search_kwargs = {"k": k, "fetch_k": fetch_k}

    # Add source filter if specified
    if source_filter:
        search_kwargs["filter"] = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=source_filter),
                )
            ]
        )

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

    Args:
        metadata: Document metadata dictionary
        source_name: Document source name

    Returns:
        Formatted text string for embedding
    """
    document_type = metadata.get("document_type", "document")
    subject_name = metadata.get("subject_name", source_name)

    # Join topics
    topics = metadata.get("topics", [])
    topics_str = ", ".join(topics[:10]) if topics else ""

    # Join key entities
    key_entities = metadata.get("key_entities", [])
    entities_str = ", ".join([
        e.get("name", "") for e in key_entities[:10]
        if isinstance(e, dict) and e.get("name")
    ]) if key_entities else ""

    # Get summary (truncate if too long)
    summary = metadata.get("summary", "")
    if len(summary) > 500:
        summary = summary[:500] + "..."

    # Format the embedding text
    parts = [f"{document_type}: {subject_name}"]

    if topics_str:
        parts.append(f"Topics: {topics_str}")

    if entities_str:
        parts.append(f"Entities: {entities_str}")

    if summary:
        parts.append(f"Summary: {summary}")

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


def search_document_metadata(
    query_text: str,
    top_k: int = None,
    score_threshold: float = None
) -> List[dict]:
    """
    Search for relevant documents by querying the metadata vector collection.

    Args:
        query_text: Query text to search for (will be embedded)
        top_k: Number of results to return (default: METADATA_SEARCH_TOP_K)
        score_threshold: Minimum score threshold (default: METADATA_SCORE_THRESHOLD)

    Returns:
        List of dictionaries with document info and scores:
        [{"document_id": str, "source_name": str, "score": float, "payload": dict}, ...]
    """
    if top_k is None:
        top_k = METADATA_SEARCH_TOP_K
    if score_threshold is None:
        score_threshold = METADATA_SCORE_THRESHOLD

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

        # Search using query_points (qdrant-client >= 1.7.0)
        results = client.query_points(
            collection_name=METADATA_COLLECTION_NAME,
            query=query_vector,
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
