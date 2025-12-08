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
FILE_SUMMARY_COLLECTION_NAME = "dots_ocr_file_summaries"
CHUNK_SUMMARY_COLLECTION_NAME = "dots_ocr_chunk_summaries"

# Singleton instances
_vectorstore: Optional[QdrantVectorStore] = None
_file_summary_vectorstore: Optional[QdrantVectorStore] = None
_chunk_summary_vectorstore: Optional[QdrantVectorStore] = None
_embeddings: Optional[LocalQwen3Embedding] = None
_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Get or create the Qdrant client instance."""
    global _client
    if _client is None:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        _client = QdrantClient(host=qdrant_host, port=qdrant_port)
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


def ensure_file_summary_collection_exists(client: QdrantClient, embedding_dim: int = 2560):
    """
    Ensure the file summary Qdrant collection exists, create if not.

    Args:
        client: The Qdrant client.
        embedding_dim: Dimension of the embedding vectors.
    """
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if FILE_SUMMARY_COLLECTION_NAME not in collection_names:
        logger.info(f"Creating collection: {FILE_SUMMARY_COLLECTION_NAME}")
        client.create_collection(
            collection_name=FILE_SUMMARY_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"Collection {FILE_SUMMARY_COLLECTION_NAME} created successfully")
    else:
        logger.debug(f"Collection {FILE_SUMMARY_COLLECTION_NAME} already exists")


def get_file_summary_vectorstore() -> QdrantVectorStore:
    """
    Get or create the file summary Qdrant vectorstore instance.

    Returns:
        Configured QdrantVectorStore instance for file summaries.
    """
    global _file_summary_vectorstore

    if _file_summary_vectorstore is None:
        client = get_qdrant_client()
        embeddings = get_embeddings()

        # Ensure collection exists
        ensure_file_summary_collection_exists(client)

        _file_summary_vectorstore = QdrantVectorStore(
            client=client,
            collection_name=FILE_SUMMARY_COLLECTION_NAME,
            embedding=embeddings,
        )
        logger.info("File summary vectorstore initialized")

    return _file_summary_vectorstore


def ensure_chunk_summary_collection_exists(client: QdrantClient, embedding_dim: int = 2560):
    """
    Ensure the chunk summary Qdrant collection exists, create if not.

    Args:
        client: The Qdrant client.
        embedding_dim: Dimension of the embedding vectors.
    """
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if CHUNK_SUMMARY_COLLECTION_NAME not in collection_names:
        logger.info(f"Creating collection: {CHUNK_SUMMARY_COLLECTION_NAME}")
        client.create_collection(
            collection_name=CHUNK_SUMMARY_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"Collection {CHUNK_SUMMARY_COLLECTION_NAME} created successfully")
    else:
        logger.debug(f"Collection {CHUNK_SUMMARY_COLLECTION_NAME} already exists")


def get_chunk_summary_vectorstore() -> QdrantVectorStore:
    """
    Get or create the chunk summary Qdrant vectorstore instance.

    Returns:
        Configured QdrantVectorStore instance for chunk summaries.
    """
    global _chunk_summary_vectorstore

    if _chunk_summary_vectorstore is None:
        client = get_qdrant_client()
        embeddings = get_embeddings()

        # Ensure collection exists
        ensure_chunk_summary_collection_exists(client)

        _chunk_summary_vectorstore = QdrantVectorStore(
            client=client,
            collection_name=CHUNK_SUMMARY_COLLECTION_NAME,
            embedding=embeddings,
        )
        logger.info("Chunk summary vectorstore initialized")

    return _chunk_summary_vectorstore


def add_file_summary(source_name: str, file_path: str, summary: str) -> None:
    """
    Add a file summary to the file summary collection.

    Args:
        source_name: The document source name.
        file_path: Path to the original file.
        summary: The generated file summary.
    """
    try:
        vectorstore = get_file_summary_vectorstore()
        doc = Document(
            page_content=summary,
            metadata={
                "source": source_name,
                "file_path": file_path,
                "type": "file_summary",
            }
        )
        vectorstore.add_documents([doc])
        logger.info(f"Added file summary for source: {source_name}")
    except Exception as e:
        logger.error(f"Error adding file summary for {source_name}: {e}")


def search_file_summaries(
    query: str, k: int = 5, score_threshold: float = 0.25
) -> List[Document]:
    """
    Search file summaries to find relevant documents.

    Uses similarity_search_with_score to filter results by relevance threshold.
    This prevents returning irrelevant documents when the collection contains
    many documents that don't match the query.

    Note: Since each document may have multiple file summaries (one per page),
    we deduplicate by source and keep only the highest scoring summary per source.

    Args:
        query: The search query.
        k: Number of unique sources to return.
        score_threshold: Minimum similarity score (0-1) to include a document.
                        Higher values = stricter matching. Default 0.25.

    Returns:
        List of Document objects with file summaries that meet the score threshold.
    """
    try:
        vectorstore = get_file_summary_vectorstore()
        # Fetch more results initially to account for multiple summaries per source
        # We want k unique sources, so fetch k*5 to ensure we get enough variety
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=k * 5)

        # Log all scores for debugging
        logger.info(f"File summary search for query: '{query[:50]}...'")
        for doc, score in docs_with_scores:
            source = doc.metadata.get("source", "unknown")
            logger.info(f"  File summary score: {source} = {score:.4f}")

        # Deduplicate by source - keep only the highest scoring summary per source
        # This is important because each document may have multiple file summaries
        source_best_scores: dict = {}  # source -> (doc, score)
        for doc, score in docs_with_scores:
            source = doc.metadata.get("source", "unknown")
            if source not in source_best_scores or score > source_best_scores[source][1]:
                source_best_scores[source] = (doc, score)

        logger.info(f"Deduplicated to {len(source_best_scores)} unique sources:")
        for source, (doc, score) in source_best_scores.items():
            logger.info(f"  Best score for '{source}': {score:.4f}")

        # Filter by score threshold and limit to k results
        # NOTE: Qdrant with Cosine distance returns scores where HIGHER is better (similarity)
        # Values range from 0 (completely different) to 1 (identical)
        filtered_docs = []
        # Sort by score descending to get top k
        sorted_sources = sorted(
            source_best_scores.items(), key=lambda x: x[1][1], reverse=True
        )
        for source, (doc, score) in sorted_sources[:k]:
            if score >= score_threshold:
                filtered_docs.append(doc)
                logger.info(f"Including file '{source}' with score {score:.4f}")
            else:
                logger.info(
                    f"Filtered out file '{source}' with low score {score:.4f} < {score_threshold}"
                )

        return filtered_docs
    except Exception as e:
        logger.error(f"Error searching file summaries: {e}")
        return []


def add_chunk_summaries(
    source_name: str,
    file_path: str,
    chunk_summaries: List[dict],
) -> None:
    """
    Add chunk summaries to the chunk summary collection.

    Args:
        source_name: The document source name.
        file_path: Path to the original file.
        chunk_summaries: List of dicts with keys:
            - chunk_indices: List[int] - indices of chunks covered by this summary
            - chunk_ids: List[str] - IDs of chunks covered by this summary
            - summary: str - the summary text
            - heading_path: str - combined heading path
    """
    try:
        vectorstore = get_chunk_summary_vectorstore()
        docs = []
        for cs in chunk_summaries:
            # Support both single chunk_id (legacy) and multiple chunk_ids (new)
            chunk_ids = cs.get("chunk_ids", [])
            if not chunk_ids and cs.get("chunk_id"):
                chunk_ids = [cs["chunk_id"]]

            chunk_indices = cs.get("chunk_indices", [])
            if not chunk_indices and "chunk_index" in cs:
                chunk_indices = [cs["chunk_index"]]

            doc = Document(
                page_content=cs["summary"],
                metadata={
                    "source": source_name,
                    "file_path": file_path,
                    "chunk_indices": chunk_indices,
                    "heading_path": cs.get("heading_path", ""),
                    "chunk_ids": chunk_ids,
                    "type": "chunk_summary",
                }
            )
            docs.append(doc)
        if docs:
            vectorstore.add_documents(docs)
            logger.info(f"Added {len(docs)} chunk summaries for source: {source_name}")
    except Exception as e:
        logger.error(f"Error adding chunk summaries for {source_name}: {e}")


def search_chunk_summaries(
    query: str,
    k: int = 10,
    source_names: List[str] = None,
    score_threshold: float = 0.25,
) -> List[Document]:
    """
    Search chunk summaries to find relevant chunks.

    Uses similarity_search_with_score to filter results by relevance threshold.

    Args:
        query: The search query.
        k: Number of results to return.
        source_names: Optional list of source names to filter results.
        score_threshold: Minimum similarity score (0-1) to include a chunk.
                        Default 0.25 (slightly lower than file summaries).

    Returns:
        List of Document objects with chunk summaries that meet the score threshold.
    """
    try:
        vectorstore = get_chunk_summary_vectorstore()

        if source_names and len(source_names) > 0:
            # Search with source filter
            filter_condition = models.Filter(
                should=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=source_name),
                    )
                    for source_name in source_names
                ]
            )
            docs_with_scores = vectorstore.similarity_search_with_score(
                query, k=k, filter=filter_condition
            )
        else:
            docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)

        # Filter by score threshold
        filtered_docs = []
        for doc, score in docs_with_scores:
            source = doc.metadata.get("source", "unknown")
            if score >= score_threshold:
                filtered_docs.append(doc)
            else:
                logger.debug(
                    f"Filtered out chunk from '{source}' with low score {score:.4f} < {score_threshold}"
                )

        return filtered_docs
    except Exception as e:
        logger.error(f"Error searching chunk summaries: {e}")
        return []


def delete_chunk_summaries_by_source(source_name: str) -> int:
    """
    Delete chunk summaries with a specific source from the chunk summary collection.

    Args:
        source_name: The source name to delete (e.g., document folder name).

    Returns:
        Number of deleted summaries.
    """
    client = get_qdrant_client()
    try:
        # Ensure collection exists before trying to delete
        ensure_chunk_summary_collection_exists(client)

        result = client.delete(
            collection_name=CHUNK_SUMMARY_COLLECTION_NAME,
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
        logger.info(f"Deleted chunk summaries with source '{source_name}'")
        return result
    except Exception as e:
        logger.error(f"Error deleting chunk summaries for '{source_name}': {e}")
        return 0


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


def get_retriever_with_sources(
    k: int = 8,
    fetch_k: int = 30,
    source_names: List[str] = None
):
    """
    Get a retriever from the vectorstore with optional source filtering.

    Args:
        k: Number of documents to retrieve.
        fetch_k: Number of documents to fetch for MMR.
        source_names: Optional list of source names to filter results.

    Returns:
        A LangChain retriever configured for MMR search.
    """
    vectorstore = get_vectorstore()
    search_kwargs = {"k": k, "fetch_k": fetch_k}

    # Add source filter if specified
    if source_names and len(source_names) > 0:
        search_kwargs["filter"] = models.Filter(
            should=[
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=source_name),
                )
                for source_name in source_names
            ]
        )

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


def delete_file_summary_by_source(source_name: str) -> int:
    """
    Delete file summary with a specific source from the file summary collection.

    Args:
        source_name: The source name to delete (e.g., document folder name).

    Returns:
        Number of deleted summaries.
    """
    client = get_qdrant_client()
    try:
        # Ensure collection exists before trying to delete
        ensure_file_summary_collection_exists(client)

        result = client.delete(
            collection_name=FILE_SUMMARY_COLLECTION_NAME,
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
        logger.info(f"Deleted file summary with source '{source_name}'")
        return result
    except Exception as e:
        logger.error(f"Error deleting file summary for '{source_name}': {e}")
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
