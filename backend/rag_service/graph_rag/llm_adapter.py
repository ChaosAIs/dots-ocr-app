"""
LLM Adapter for GraphRAG Agent.

Provides a simple async interface for the GraphRAG agent to use LLM services.
"""

import asyncio
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class LLMAdapter:
    """
    Adapter to provide async generate() method for GraphRAG agent.
    
    Wraps the existing LLM service with an async interface.
    """
    
    def __init__(self, llm_service=None):
        """
        Initialize the LLM adapter.
        
        Args:
            llm_service: Optional LLM service instance. If not provided,
                        will use get_llm_service() from llm_service module.
        """
        self._llm_service = llm_service
        self._chat_model = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        
    def _init_service(self):
        """Lazily initialize the LLM service."""
        if self._llm_service is None:
            from ..llm_service import get_llm_service
            self._llm_service = get_llm_service()
            
        if self._chat_model is None:
            self._chat_model = self._llm_service.get_chat_model(
                temperature=0.3,
                num_ctx=8192,
            )
            logger.info("[LLM Adapter] Chat model initialized")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> str:
        """
        Generate text response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        self._init_service()
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        # Run the sync LLM call in a thread pool to make it async
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                self._executor,
                lambda: self._chat_model.invoke(messages)
            )
            return response.content
        except Exception as e:
            logger.error(f"[LLM Adapter] Error generating response: {e}")
            raise


class EmbeddingAdapter:
    """
    Adapter to provide async embed() method for GraphRAG agent.
    
    Wraps the existing embedding service with an async interface.
    """
    
    def __init__(self, embedding_service=None):
        """
        Initialize the embedding adapter.
        
        Args:
            embedding_service: Optional embedding service instance.
        """
        self._embedding_service = embedding_service
        self._executor = ThreadPoolExecutor(max_workers=2)
        
    def _init_service(self):
        """Lazily initialize the embedding service."""
        if self._embedding_service is None:
            from ..local_qwen_embedding import LocalQwen3Embedding
            self._embedding_service = LocalQwen3Embedding()
            logger.info("[Embedding Adapter] Embedding service initialized")
    
    async def embed(self, text: str) -> list:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        self._init_service()
        
        # Run the sync embedding call in a thread pool to make it async
        loop = asyncio.get_event_loop()
        try:
            embedding = await loop.run_in_executor(
                self._executor,
                lambda: self._embedding_service.embed_query(text)
            )
            return embedding
        except Exception as e:
            logger.error(f"[Embedding Adapter] Error generating embedding: {e}")
            raise

