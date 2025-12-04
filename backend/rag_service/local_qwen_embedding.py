"""
Local Qwen3 Embedding wrapper for LangChain compatibility.
Connects to Qwen3-Embedding-4B running at localhost:8003.
"""

import os
import logging
from typing import List
import requests
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class LocalQwen3Embedding(Embeddings):
    """
    LangChain-compatible embedding class that connects to a local
    Qwen3-Embedding-4B server via OpenAI-compatible API.
    """

    def __init__(
        self,
        url: str = None,
        timeout: int = 60,
        batch_size: int = 32,
    ):
        """
        Initialize the embedding client.

        Args:
            url: The URL of the embedding server. Defaults to localhost:8003.
            timeout: Request timeout in seconds.
            batch_size: Maximum number of texts to embed in a single request.
        """
        # Get URL from environment or use default
        if url is None:
            embedding_host = os.getenv("EMBEDDING_HOST", "localhost")
            embedding_port = os.getenv("EMBEDDING_PORT", "8003")
            url = f"http://{embedding_host}:{embedding_port}/v1/embeddings"

        self.url = url
        self.timeout = timeout
        self.batch_size = batch_size
        logger.info(f"Initialized LocalQwen3Embedding with URL: {self.url}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                response = requests.post(
                    self.url,
                    json={"input": batch},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                # Extract embeddings from response
                batch_embeddings = [d["embedding"] for d in data["data"]]
                all_embeddings.extend(batch_embeddings)

            except requests.exceptions.RequestException as e:
                logger.error(f"Error calling embedding API: {e}")
                raise RuntimeError(f"Failed to get embeddings: {e}")

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: The query text to embed.

        Returns:
            The embedding vector.
        """
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents (falls back to sync)."""
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query (falls back to sync)."""
        return self.embed_query(text)

