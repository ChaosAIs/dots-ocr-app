"""
LLM Service Module for RAG Agent.
Provides abstraction for different LLM backends (Ollama, vLLM).
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class BaseLLMService(ABC):
    """Abstract base class for LLM services."""

    @abstractmethod
    def get_chat_model(
        self,
        temperature: float = 0.2,
        num_ctx: int = 8192,
        num_predict: Optional[int] = None,
    ) -> BaseChatModel:
        """Get a chat model instance for inference."""
        pass

    @abstractmethod
    def get_query_model(
        self,
        temperature: float = 0.1,
        num_ctx: int = 2048,
        num_predict: int = 256,
    ) -> BaseChatModel:
        """Get a smaller/faster model for query analysis."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        pass

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """
        Simple text-in/text-out generation interface.

        This method provides a simple interface for code that expects
        a generate(prompt) -> response pattern, wrapping the LangChain
        chat model invocation.

        Args:
            prompt: The prompt text to send to the LLM
            temperature: Temperature for generation (default 0.2)

        Returns:
            The generated text response
        """
        from langchain_core.messages import HumanMessage

        chat_model = self.get_chat_model(temperature=temperature)
        response = chat_model.invoke([HumanMessage(content=prompt)])

        # Extract text content from response
        if hasattr(response, 'content'):
            return response.content
        return str(response)


class OllamaLLMService(BaseLLMService):
    """Ollama LLM Service implementation."""

    def __init__(self):
        self.host = os.getenv("RAG_OLLAMA_HOST", os.getenv("OLLAMA_HOST", "localhost"))
        self.port = os.getenv("RAG_OLLAMA_PORT", os.getenv("OLLAMA_PORT", "11434"))
        self.model = os.getenv("RAG_OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:latest"))
        self.query_model = os.getenv("RAG_OLLAMA_QUERY_MODEL", os.getenv("OLLAMA_QUERY_MODEL", "qwen2.5:latest"))
        self.base_url = f"http://{self.host}:{self.port}"
        logger.info(f"OllamaLLMService initialized: {self.base_url}, model={self.model}")

    def get_chat_model(
        self,
        temperature: float = 0.2,
        num_ctx: int = 8192,
        num_predict: Optional[int] = None,
    ) -> BaseChatModel:
        """Get Ollama chat model for main inference."""
        kwargs = {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": temperature,
            "num_ctx": num_ctx,
        }
        if num_predict is not None:
            kwargs["num_predict"] = num_predict
        return ChatOllama(**kwargs)

    def get_query_model(
        self,
        temperature: float = 0.1,
        num_ctx: int = 2048,
        num_predict: int = 256,
    ) -> BaseChatModel:
        """Get Ollama query model for query analysis."""
        return ChatOllama(
            base_url=self.base_url,
            model=self.query_model,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
        )

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        import requests
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False


class VLLMLLMService(BaseLLMService):
    """vLLM LLM Service implementation using OpenAI-compatible API.

    Supports Qwen3 models with configurable reasoning mode:
    - "thinking": Enable chain-of-thought reasoning with <think>...</think> blocks
    - "instruction": Direct answers without reasoning blocks (default)

    Set via RAG_VLLM_REASONING_MODE environment variable.
    """

    def __init__(self):
        self.host = os.getenv("RAG_VLLM_HOST", "localhost")
        self.port = os.getenv("RAG_VLLM_PORT", "8004")
        self.model = os.getenv("RAG_VLLM_MODEL", "Qwen/Qwen3-14B")
        self.api_key = os.getenv("RAG_VLLM_API_KEY", "EMPTY")
        self.base_url = f"http://{self.host}:{self.port}/v1"

        # Reasoning mode: "thinking" or "instruction" (default: instruction)
        reasoning_mode = os.getenv("RAG_VLLM_REASONING_MODE", "instruction").strip().lower()
        if reasoning_mode not in {"thinking", "instruction"}:
            reasoning_mode = "instruction"
        self.reasoning_mode = reasoning_mode
        self.enable_thinking = (reasoning_mode == "thinking")

        logger.info(f"VLLMLLMService initialized: {self.base_url}, model={self.model}, reasoning_mode={self.reasoning_mode}")

    def _get_extra_body(self) -> dict:
        """Get extra_body parameters for Qwen3 chat template."""
        return {
            "chat_template_kwargs": {
                "enable_thinking": self.enable_thinking
            }
        }

    def get_chat_model(
        self,
        temperature: float = 0.2,
        num_ctx: int = 8192,
        num_predict: Optional[int] = None,
    ) -> BaseChatModel:
        """Get vLLM chat model for main inference."""
        kwargs = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "temperature": temperature,
            "extra_body": self._get_extra_body(),
        }
        if num_predict is not None:
            kwargs["max_tokens"] = num_predict
        return ChatOpenAI(**kwargs)

    def get_query_model(
        self,
        temperature: float = 0.1,
        num_ctx: int = 2048,
        num_predict: int = 256,
    ) -> BaseChatModel:
        """Get vLLM model for query analysis (uses same model)."""
        return ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            temperature=temperature,
            max_tokens=num_predict,
            extra_body=self._get_extra_body(),
        )

    def is_available(self) -> bool:
        """Check if vLLM server is available."""
        import requests
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"vLLM not available: {e}")
            return False


# LLM Backend selection
RAG_LLM_BACKEND = os.getenv("RAG_LLM_BACKEND", "ollama").lower()

# Singleton instance
_llm_service: Optional[BaseLLMService] = None


def get_llm_service() -> BaseLLMService:
    """Get the configured LLM service instance (singleton)."""
    global _llm_service
    if _llm_service is None:
        if RAG_LLM_BACKEND == "vllm":
            _llm_service = VLLMLLMService()
            logger.info("Using vLLM LLM backend for RAG")
        else:
            _llm_service = OllamaLLMService()
            logger.info("Using Ollama LLM backend for RAG")
    return _llm_service

