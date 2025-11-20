#!/usr/bin/env python3
"""Gemma3 OCR / Image Analysis Converter.

This module provides a small wrapper around a local Ollama instance running a
Gemma3 model. It exposes a helper to send a base64-encoded image to the model
and get markdown analysis back.
"""

import os
import logging
from typing import Optional
from utils.image_utils import resize_image_if_needed

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    requests = None  # type: ignore


class Gemma3OCRConverter:
    """Helper for calling a local Gemma3 vision model via Ollama.

    The converter expects a base64-encoded image string (no data URL prefix)
    and returns markdown text describing/analyzing the image.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        # Resolve configuration from explicit args or environment variables.
        # Environment variable precedence for base URL:
        #   1) GEMMA_OLLAMA_BASE_URL / GEMMA3_OLLAMA_BASE_URL
        #   2) OLLAMA_BASE_URL
        #   3) default http://127.0.0.1:11434
        resolved_base_url = (
            base_url
            or os.getenv("GEMMA_OLLAMA_BASE_URL")
            or os.getenv("GEMMA3_OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://127.0.0.1:11434"
        )
        self.base_url = resolved_base_url.rstrip("/")

        # Model name precedence:
        #   GEMMA_MODEL / GEMMA3_MODEL / OLLAMA_MODEL / default gemma3:27b
        self.model_name = (
            model_name
            or os.getenv("GEMMA_MODEL")
            or os.getenv("GEMMA3_MODEL")
            or os.getenv("OLLAMA_MODEL")
            or "gemma3:27b"
        )

        # Temperature precedence:
        #   GEMMA_TEMPERATURE / GEMMA3_TEMPERATURE / OLLAMA_TEMPERATURE / default 0.1
        if temperature is not None:
            self.temperature = float(temperature)
        else:
            temp_env = (
                os.getenv("GEMMA_TEMPERATURE")
                #or os.getenv("GEMMA3_TEMPERATURE")
                #or os.getenv("OLLAMA_TEMPERATURE")
                or "0.1"
            )
            self.temperature = float(temp_env)

        # Timeout precedence:
        #   GEMMA_TIMEOUT / GEMMA3_TIMEOUT / OLLAMA_TIMEOUT / default 120
        if timeout is not None:
            self.timeout = int(timeout)
        else:
            timeout_env = (
                os.getenv("GEMMA_TIMEOUT")
                #or os.getenv("GEMMA3_TIMEOUT")
                #or os.getenv("OLLAMA_TIMEOUT")
                or "120"
            )
            self.timeout = int(timeout_env)

        self.generate_endpoint = f"{self.base_url}/api/generate"

    def _build_default_prompt(self) -> str:
        """Build the default prompt used for Gemma3 image analysis."""
        return (
            "You are an expert document analysis assistant.\n"
            "You will be given a single image.\n\n"
            "Analyze the image and return a clear, well-structured Markdown description.\n"
            "- If it contains charts, dashboards, or diagrams, describe the structure and key insights.\n"
            "- If it contains tables or forms, summarize the important fields and relationships.\n"
            "- If it contains text, transcribe important text and summarize the main points.\n"
            "- Use Markdown headings, bullet lists and tables when helpful.\n"
            "- Focus on factual description and interpretation of the content.\n"
            "- Do NOT mention that you received an image or are an AI model; just provide the analysis.\n"
        )

    def convert_image_base64_to_markdown(
        self,
        image_base64: str,
        prompt: Optional[str] = None,
    ) -> str:
        """Call Gemma3 (via Ollama) to analyze a base64-encoded image.

        Args:
            image_base64: Raw base64 string of the image content. Do NOT include a
                data URL prefix like 'data:image/png;base64,'.
            prompt: Optional custom prompt. If omitted, a default analysis prompt is used.

        Returns:
            Markdown text with the analysis results. If the call fails, a short
            markdown-formatted error message is returned instead of raising.
        """
        if not image_base64:
            return "Image analysis error: no image data was provided for analysis.\n"

        # Resize image if needed to prevent processing issues
        # Get max area from environment or use default 2.56M pixels (e.g., 1600x1600)
        max_area = int(os.getenv("GEMMA_TRANSFORMERS_MAX_IMAGE_AREA", "2560000"))
        image_base64 = resize_image_if_needed(image_base64, max_area=max_area, logger_instance=self.logger)

        if requests is None:  # pragma: no cover - hit only when requests missing
            self.logger.error("requests library not installed - cannot call Ollama/Gemma3")
            return (
                "Image analysis error: the `requests` library is not available, "
                "so Gemma3 analysis cannot be performed.\n"
            )

        final_prompt = prompt or self._build_default_prompt()

        payload = {
            "model": self.model_name,
            "prompt": final_prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        try:
            self.logger.info(
                "Sending image analysis request to Gemma3 via Ollama: model=%s, endpoint=%s",
                self.model_name,
                self.generate_endpoint,
            )
            response = requests.post(
                self.generate_endpoint,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
        except Exception as exc:  # pragma: no cover - network/IO errors
            self.logger.error("Error calling Gemma3 via Ollama: %s", exc)
            return (
                "Image analysis error: failed to call Gemma3 via Ollama: "
                f"{exc}\n"
            )

        if response.status_code != 200:
            self.logger.error(
                "Gemma3/Ollama returned non-200 status: %s - %s",
                response.status_code,
                response.text,
            )
            return (
                "Image analysis error: Ollama API error: "
                f"{response.status_code} - {response.text}\n"
            )

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - invalid JSON
            self.logger.error("Failed to parse Gemma3/Ollama JSON response: %s", exc)
            return (
                "Image analysis error: could not parse the response from "
                "Gemma3/Ollama as JSON.\n"
            )

        result = (data.get("response") or "").strip()
        if not result:
            self.logger.warning("Gemma3/Ollama response contained no 'response' text")
            return "No analysis text was returned by the Gemma3 model.\n"

        return result


