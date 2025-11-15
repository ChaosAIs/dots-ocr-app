#!/usr/bin/env python3
"""Qwen3-VL OCR / Image Analysis Converter.

This module provides a small wrapper around a local Ollama instance running a
Qwen3-VL model. It exposes a helper to send a base64-encoded image to the
model and get markdown analysis back.
"""

import os
import logging
import re
from typing import Optional

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    requests = None  # type: ignore


class Qwen3OCRConverter:
    """Helper for calling a local Qwen3-VL vision model via Ollama.

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
        #   1) QWEN_OLLAMA_BASE_URL / QWEN3_OLLAMA_BASE_URL
        #   2) OLLAMA_BASE_URL
        #   3) default http://127.0.0.1:11434
        resolved_base_url = (
            base_url
            or os.getenv("QWEN_OLLAMA_BASE_URL")
           #or os.getenv("QWEN3_OLLAMA_BASE_URL")
           #or os.getenv("OLLAMA_BASE_URL")
            or "http://127.0.0.1:11434"
        )
        self.base_url = resolved_base_url.rstrip("/")

        # Model name precedence:
        #   QWEN_MODEL / QWEN3_MODEL / OLLAMA_MODEL / default qwen3-vl:32b
        self.model_name = (
            model_name
            or os.getenv("QWEN_MODEL")
            #or os.getenv("QWEN3_MODEL")
            #or os.getenv("OLLAMA_MODEL")
            or "qwen3-vl:32b"
        )

        # Temperature precedence:
        #   QWEN_TEMPERATURE / QWEN3_TEMPERATURE / OLLAMA_TEMPERATURE / default 0.1
        if temperature is not None:
            self.temperature = float(temperature)
        else:
            temp_env = (
                os.getenv("QWEN_TEMPERATURE")
                #or os.getenv("QWEN3_TEMPERATURE")
                #or os.getenv("OLLAMA_TEMPERATURE")
                or "0.1"
            )
            self.temperature = float(temp_env)

        # Timeout precedence:
        #   QWEN_TIMEOUT / QWEN3_TIMEOUT / OLLAMA_TIMEOUT / default 180
        if timeout is not None:
            self.timeout = int(timeout)
        else:
            timeout_env = (
                os.getenv("QWEN_TIMEOUT")
                or os.getenv("QWEN3_TIMEOUT")
                or os.getenv("OLLAMA_TIMEOUT")
                or "180"
            )
            self.timeout = int(timeout_env)

        self.generate_endpoint = f"{self.base_url}/api/generate"

    def _build_default_prompt(self) -> str:
        """Build the default prompt used for Qwen3-VL image analysis."""
        return (
            "You are an expert multimodal reasoning assistant.\n"
            "You will be given a single image from a larger document.\n\n"
            "Carefully analyze the image step by step, but only output the final "
            "answer, formatted as Markdown.\n"
            "- If the image contains tables or forms, reconstruct them as Markdown tables.\n"
            "- If it contains diagrams, dashboards, or charts, describe the structure and key insights.\n"
            "- If it contains text, transcribe important text and preserve the layout when possible.\n"
            "- Use Markdown headings, lists, and tables when helpful.\n"
            "- Do NOT mention that you are reasoning or that you received an image; "
            "just provide the final analysis.\n"
        )

    def _extract_markdown_from_reasoning_output(self, raw: str) -> str:
        """Strip Qwen3 `<think>` reasoning blocks and return only the final answer.

        Qwen3 reasoning models (including the Ollama variants) typically wrap their
        chain-of-thought in `<think>...</think>` tags followed by the final answer.
        For our use case we *never* want to expose the reasoning text, only the
        final markdown answer.
        """
        if not raw:
            return ""

        text = raw.strip()

        # If the model emitted `<think>...</think>` blocks, remove them entirely.
        # We operate case-insensitively and across newlines.
        if "<think>" in text.lower():
            cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            if cleaned:
                return cleaned

        # Fallback: return the original text trimmed. This covers non-reasoning
        # variants or future formats that do not use `<think>` tags.
        return text

    def convert_image_base64_to_markdown(
        self,
        image_base64: str,
        prompt: Optional[str] = None,
    ) -> str:
        """Call Qwen3-VL (via Ollama) to analyze a base64-encoded image.

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

        if requests is None:  # pragma: no cover - hit only when requests missing
            self.logger.error("requests library not installed - cannot call Ollama/Qwen3")
            return (
                "Image analysis error: the `requests` library is not available, "
                "so Qwen3 analysis cannot be performed.\n"
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
                "Sending image analysis request to Qwen3 via Ollama: model=%s, endpoint=%s",
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
            self.logger.error("Error calling Qwen3 via Ollama: %s", exc)
            return (
                "Image analysis error: failed to call Qwen3 via Ollama: "
                f"{exc}\n"
            )

        if response.status_code != 200:
            self.logger.error(
                "Qwen3/Ollama returned non-200 status: %s - %s",
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
            self.logger.error("Failed to parse Qwen3/Ollama JSON response: %s", exc)
            return (
                "Image analysis error: could not parse the response from "
                "Qwen3/Ollama as JSON.\n"
            )

        raw_result = (data.get("response") or "").strip()
        if not raw_result:
            self.logger.warning("Qwen3/Ollama response contained no 'response' text")
            return "No analysis text was returned by the Qwen3 model.\n"

        # Strip any `<think>` reasoning blocks so that only the final markdown
        # answer is returned to the caller.
        result = self._extract_markdown_from_reasoning_output(raw_result)
        if not result:
            self.logger.warning("Qwen3 reasoning output became empty after stripping <think> blocks")
            return "No analysis text was returned by the Qwen3 model.\n"

        return result

