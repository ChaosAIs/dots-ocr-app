"""
Local Qwen3 Embedding wrapper for LangChain compatibility.
Connects to Qwen3-Embedding-4B running at localhost:8003.
"""

import os
import logging
import time
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
        timeout: int = 120,
        batch_size: int = 16,
        max_retries: int = 3,
    ):
        """
        Initialize the embedding client.

        Args:
            url: The URL of the embedding server. Defaults to localhost:8003.
            timeout: Request timeout in seconds (increased to 120 for large texts).
            batch_size: Maximum number of texts to embed in a single request (reduced to 16 for stability).
            max_retries: Number of retries for failed requests.
        """
        # Get URL from environment or use default
        if url is None:
            embedding_host = os.getenv("EMBEDDING_HOST", "localhost")
            embedding_port = os.getenv("EMBEDDING_PORT", "8003")
            url = f"http://{embedding_host}:{embedding_port}/v1/embeddings"

        self.url = url
        self.timeout = int(os.getenv("EMBEDDING_TIMEOUT", str(timeout)))
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", str(batch_size)))
        self.max_retries = max_retries
        logger.info(f"Initialized LocalQwen3Embedding with URL: {self.url}, timeout: {self.timeout}s, batch_size: {self.batch_size}")

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
            batch_embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _embed_batch_with_retry(self, batch: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts with retry logic.

        Args:
            batch: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Log batch info for debugging
                total_chars = sum(len(t) for t in batch)
                logger.debug(f"Embedding batch: {len(batch)} texts, {total_chars} chars (attempt {attempt + 1}/{self.max_retries})")

                response = requests.post(
                    self.url,
                    json={"input": batch},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                # Extract embeddings from response
                batch_embeddings = [d["embedding"] for d in data["data"]]
                return batch_embeddings

            except requests.exceptions.ReadTimeout as e:
                last_error = e
                logger.warning(f"Embedding API timeout (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)

            except requests.exceptions.RequestException as e:
                last_error = e
                logger.error(f"Error calling embedding API (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)

        logger.error(f"All {self.max_retries} embedding attempts failed")
        raise RuntimeError(f"Failed to get embeddings after {self.max_retries} attempts: {last_error}")

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

