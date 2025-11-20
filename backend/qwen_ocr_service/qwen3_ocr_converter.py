#!/usr/bin/env python3
"""Qwen3-VL OCR / Image Analysis Converter.

This module provides a small wrapper around a local Ollama instance running a
Qwen3-VL model. It exposes a helper to send a base64-encoded image to the
model and get markdown analysis back.
"""

import os
import logging
import re
import base64
import io
from typing import Optional, Tuple
from utils.image_utils import resize_image_if_needed

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    requests = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    OpenAI = None  # type: ignore

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor  # type: ignore
    import torch  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    Qwen3VLForConditionalGeneration = None  # type: ignore
    AutoProcessor = None  # type: ignore
    torch = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    Image = None  # type: ignore


class Qwen3OCRConverter:
    """Helper for calling a local Qwen3-VL vision model.

    The converter expects a base64-encoded image string (no data URL prefix)
    and returns markdown text describing/analyzing the image.

    It supports three backends:
      - "ollama" (default): calls a local Ollama server's `/api/generate` endpoint.
      - "vllm": calls a vLLM OpenAI-compatible server's `/v1/chat/completions` endpoint.
      - "transformers": uses HuggingFace transformers library to run the model locally.

    Backend selection is controlled via the ``QWEN_BACKEND`` environment variable.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        # Determine backend: "ollama" (default), "vllm" (OpenAI-compatible API), or "transformers"
        backend_env = (os.getenv("QWEN_BACKEND", "ollama") or "ollama").strip().lower()
        if backend_env not in {"ollama", "vllm", "transformers"}:
            backend_env = "ollama"
        self.backend = backend_env

        # Determine reasoning mode: "thinking" (chain-of-thought with <think> tags)
        # or "instruction" (direct answers, no special reasoning blocks).
        mode_env = (os.getenv("QWEN_REASONING_MODE", "thinking") or "thinking").strip().lower()
        if mode_env not in {"thinking", "instruction"}:
            mode_env = "thinking"
        self.reasoning_mode = mode_env

        # Model name precedence (for Ollama/vLLM backends):
        #   QWEN_MODEL / QWEN3_MODEL / OLLAMA_MODEL / default qwen3-vl:32b
        self.model_name = (
            model_name
            or os.getenv("QWEN_MODEL")
            #or os.getenv("QWEN3_MODEL")
            #or os.getenv("OLLAMA_MODEL")
            or "qwen3-vl:32b"
        )

        # Transformers model name (set later if using transformers backend)
        self.transformers_model_name = None

        # Image size detection threshold (configurable via environment variable)
        # Images with area below this threshold use simple prompt, above use complex prompt
        self.pixel_area_threshold = int(os.getenv("QWEN_IMAGE_PIXEL_AREA_THRESHOLD", "40000"))

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

        # Timeout precedence (used by both backends):
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

        # Optional max tokens hint for vLLM/OpenAI backend
        max_tokens_env = os.getenv("QWEN_MAX_TOKENS")
        self.max_tokens = int(max_tokens_env) if max_tokens_env is not None else 2048

        # Configure backend-specific endpoints/clients
        if self.backend == "transformers":
            # For transformers backend, load the model and processor directly
            # using HuggingFace transformers library
            if Qwen3VLForConditionalGeneration is None or AutoProcessor is None or torch is None:
                self.logger.error(
                    "transformers/torch libraries are not installed - cannot use transformers backend for Qwen3.",
                )
                self.model = None
                self.processor = None
                self.client = None
            else:
                # Get GPU device configuration from environment
                # Default to GPUs 4,5,6,7 as specified in the task
                gpu_devices_str = os.getenv("QWEN_TRANSFORMERS_GPU_DEVICES", "4,5,6,7")
                gpu_devices = [int(x.strip()) for x in gpu_devices_str.split(",") if x.strip()]

                # Set CUDA_VISIBLE_DEVICES to restrict to specified GPUs
                # This remaps the GPUs so cuda:0 refers to the first GPU in the list
                if len(gpu_devices) > 0:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))

                # Get model path/name from environment
                # Default to Qwen/Qwen3-VL-8B-Instruct as specified in the task
                transformers_model = os.getenv("QWEN_TRANSFORMERS_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
                # Store for later use in logging
                self.transformers_model_name = transformers_model

                # Get dtype configuration
                dtype_str = os.getenv("QWEN_TRANSFORMERS_DTYPE", "bfloat16")
                if dtype_str == "bfloat16":
                    dtype = torch.bfloat16
                elif dtype_str == "float16":
                    dtype = torch.float16
                elif dtype_str == "float32":
                    dtype = torch.float32
                else:
                    # Default to bfloat16 for better performance and memory efficiency
                    dtype = torch.bfloat16

                # Get attention implementation
                # Default to "eager" for compatibility; use "flash_attention_2" if installed
                attn_impl = os.getenv("QWEN_TRANSFORMERS_ATTN_IMPL", "eager")

                try:
                    self.logger.info(
                        f"Loading Qwen3-VL model via transformers: {transformers_model} on GPUs {gpu_devices}"
                    )

                    # Prepare kwargs for from_pretrained
                    model_kwargs = {
                        "torch_dtype": dtype,
                        "local_files_only": True,  # Use only local cache to avoid network calls
                    }

                    # Only add device_map if we have multiple GPUs
                    # For single GPU or specific GPU, we'll use .to() after loading
                    if len(gpu_devices) > 1:
                        model_kwargs["device_map"] = "auto"

                    # Only add attn_implementation if not using default eager
                    if attn_impl and attn_impl != "eager":
                        model_kwargs["attn_implementation"] = attn_impl

                    # Load model with specified configuration
                    # Uses default HuggingFace cache directory (~/.cache/huggingface)
                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        transformers_model,
                        **model_kwargs
                    )

                    # Move to specific GPU if single GPU setup
                    if len(gpu_devices) == 1:
                        self.model = self.model.to(f"cuda:0")  # cuda:0 after CUDA_VISIBLE_DEVICES remapping

                    # Load processor
                    # Uses default HuggingFace cache directory (~/.cache/huggingface)
                    self.processor = AutoProcessor.from_pretrained(
                        transformers_model,
                        local_files_only=True,  # Use only local cache to avoid network calls
                    )

                    self.logger.info(f"Qwen3-VL model loaded successfully via transformers")
                    self.client = None
                except Exception as exc:
                    self.logger.error(f"Failed to load Qwen3-VL model via transformers: {exc}")
                    self.model = None
                    self.processor = None
                    self.client = None

            self.base_url = None
            self.generate_endpoint = None

        elif self.backend == "vllm":
            # For vLLM, ``base_url`` / ``QWEN_VLLM_API_BASE`` should point to the
            # OpenAI-compatible base URL, typically ``http://host:port/v1``.
            api_base = (
                base_url
                or os.getenv("QWEN_VLLM_API_BASE")
                or "http://127.0.0.1:8006/v1"
            )
            self.api_base = api_base.rstrip("/")
            self.base_url = self.api_base

            if OpenAI is None:  # pragma: no cover - handled gracefully at runtime
                self.logger.error(
                    "OpenAI client is not installed - cannot use vLLM backend for Qwen3.",
                )
                self.client = None
            else:
                self.client = OpenAI(
                    api_key=os.getenv("QWEN_API_KEY", "EMPTY"),
                    base_url=self.api_base,
                    timeout=self.timeout,
                )

            # For logging / debugging only; actual calls go through the OpenAI client
            self.generate_endpoint = f"{self.api_base}/chat/completions"
            self.model = None
            self.processor = None

        else:
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
            self.client = None
            self.generate_endpoint = f"{self.base_url}/api/generate"
            self.model = None
            self.processor = None

    def _get_image_dimensions(self, image_base64: str) -> Tuple[int, int]:
        """Get image dimensions (width, height) from base64 string.

        Args:
            image_base64: Raw base64 string of the image content.

        Returns:
            Tuple of (width, height) in pixels. Returns (0, 0) if unable to determine.
        """
        if Image is None:  # pragma: no cover - PIL not available
            self.logger.warning("PIL not available - cannot determine image dimensions")
            return (0, 0)

        try:
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_base64)

            # Open image with PIL
            image = Image.open(io.BytesIO(image_bytes))

            return image.size  # Returns (width, height)
        except Exception as e:  # pragma: no cover - invalid image data
            self.logger.warning(f"Failed to get image dimensions: {e}")
            return (0, 0)

    def _is_simple_image(self, image_base64: str) -> bool:
        """Determine if an image is simple (small) or complex (large) based on pixel area.

        Simple images are typically:
        - Small icons, logos, labels
        - Single words or short phrases
        - Total pixel area < pixel_area_threshold

        Complex images are typically:
        - Charts, tables, diagrams
        - Multi-paragraph documents
        - Total pixel area >= pixel_area_threshold

        Threshold is configurable via environment variable:
        - QWEN_IMAGE_PIXEL_AREA_THRESHOLD (default: 40000, approximately 200x200)

        Args:
            image_base64: Raw base64 string of the image content.

        Returns:
            True if image is simple (use simple analysis), False if complex (use deep analysis).
        """
        width, height = self._get_image_dimensions(image_base64)

        # If we couldn't get dimensions, default to complex analysis (safer)
        if width == 0 or height == 0:
            return False

        # Calculate total pixel area
        pixel_area = width * height

        self.logger.info(f"Image analysis: dimensions={width}x{height}, area={pixel_area:,}")

        # Image is simple if total area is small
        is_simple = pixel_area < self.pixel_area_threshold

        self.logger.info(
            f"Image analysis: dimensions={width}x{height}, area={pixel_area:,}, "
            f"classification={'SIMPLE' if is_simple else 'COMPLEX'} "
            f"(area_threshold={self.pixel_area_threshold:,})"
        )

        return is_simple

    def _build_simple_prompt(self) -> str:
        """Build a simple prompt for small/basic images (icons, labels, single words)."""
        return (
            "You are a helpful assistant that can read and understand images.\n"
            "This is a small image, likely containing simple content.\n\n"
            "Extract the content directly without extra analysis:\n"
            "- If it's a single word or short phrase: output just the text.\n"
            "- If it's a table or form: output the text and layout as original format with markdown.\n"
            "- If it's a logo or icon: output the name/label only.\n"
            "- If it's a label: output the text content.\n"
            "- If it's a diagram: briefly describe its workflow.\n\n"
            "- If it's an object: briefly describe what is it.\n\n"
            "IMPORTANT: Preserve the original language of the content in the image. "
            "If the text is in Chinese, output in Chinese. If it's in English, output in English. If it's in French, output in French."
            "Do NOT translate the extracted content to another language.\n\n"
            "Note: Keep your response concise and direct. No need for sections or detailed analysis.\n"
        )

    def _build_complex_prompt(self) -> str:
        """Build a detailed prompt for complex images (charts, tables, diagrams, forms)."""
        return (
            "CRITICAL INSTRUCTION - READ FIRST:\n"
            "You MUST preserve the ORIGINAL LANGUAGE of the text in the image.\n"
            "- If the image contains Chinese text, respond ENTIRELY in Chinese\n"
            "- If the image contains English text, respond ENTIRELY in English\n"
            "- If the image contains other languages, respond in that language\n"
            "- Do NOT translate the content\n"
            "- All section headers, descriptions, and analysis MUST be in the same language as the image content\n\n"
            "You are analyzing a complex image containing detailed information.\n\n"
            "Structure your output with these sections (use the appropriate language for section headers):\n\n"
            "Section 1: Document Analysis\n"
            "Provide a brief overview of what this image contains.\n\n"
            "Section 2: Content\n"
            "Extract and present the content using appropriate markdown formatting:\n"
            "- **Tables/Forms**: Reconstruct as properly formatted Markdown tables\n"
            "- **Charts/Dashboards**: List all data points, values, labels, trends, and insights\n"
            "- **Diagrams/Flowcharts**: Describe the workflow, process flow, and relationships between components\n"
            "- **Text Content**: Transcribe all text accurately, preserve hierarchical structure, use markdown headers and lists\n\n"
            "Section 3: Key Information\n"
            "Highlight the most important information:\n"
            "- Critical numbers, amounts, percentages\n"
            "- Dates and time periods\n"
            "- Titles, labels, and categories\n"
            "- Any warnings, notes, or special annotations\n\n"
            "Section 4: Insights\n"
            "Provide brief analytical observations:\n"
            "- What does this data show or indicate?\n"
            "- Are there any notable trends, comparisons, or outliers?\n"
            "- What is the main purpose or message of this content?\n\n"
            "REMEMBER: Use the SAME LANGUAGE as the text in the image for ALL sections and content.\n"
        )

    def _build_default_prompt(self) -> str:
        """Build the default prompt used for Qwen3-VL image analysis.

        This is an alias for _build_complex_prompt() for backward compatibility.
        """
        return self._build_complex_prompt()

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
        """Call Qwen3-VL to analyze a base64-encoded image.

        This dispatches to the Ollama, vLLM (OpenAI-compatible), or transformers
        backend depending on ``self.backend`` / ``QWEN_BACKEND``.

        The method automatically detects image size and applies:
        - Automatic resizing to half size if either dimension > 2000px
        - Simple analysis for small images (< 200px dimension or < 50k pixels)
        - Complex/deep analysis for larger images (charts, tables, diagrams)

        Args:
            image_base64: Raw base64 string of the image content. Do NOT include a
                data URL prefix like 'data:image/png;base64,'.
            prompt: Optional custom prompt. If omitted, automatically selects simple
                or complex prompt based on image dimensions.

        Returns:
            Markdown text with the analysis results. If the call fails, a short
            markdown-formatted error message is returned instead of raising.
        """
        if not image_base64:
            return "Image analysis error: no image data was provided for analysis.\n"

        # If no custom prompt provided, choose based on ORIGINAL image size
        # This ensures classification is based on content complexity, not GPU constraints
        if prompt is None:
            if self._is_simple_image(image_base64):
                final_prompt = self._build_simple_prompt()
            else:
                final_prompt = self._build_complex_prompt()
        else:
            final_prompt = prompt

        # Resize image if needed to prevent CUDA timeout errors
        # This happens AFTER classification to avoid affecting prompt selection
        # Get max area from environment or use default 2.56M pixels (e.g., 1600x1600)
        max_area = int(os.getenv("QWEN_TRANSFORMERS_MAX_IMAGE_AREA", "2560000"))
        image_base64 = resize_image_if_needed(image_base64, max_area=max_area, logger_instance=self.logger)

        if self.backend == "transformers":
            return self._convert_with_transformers(image_base64, final_prompt)
        elif self.backend == "vllm":
            return self._convert_with_vllm(image_base64, final_prompt)

        # Default: Ollama backend
        return self._convert_with_ollama(image_base64, final_prompt)

    def _convert_with_ollama(self, image_base64: str, final_prompt: str) -> str:
        """Call a local Ollama server running Qwen3-VL."""
        if requests is None:  # pragma: no cover - hit only when requests missing
            self.logger.error("requests library not installed - cannot call Ollama/Qwen3")
            return (
                "Image analysis error: the `requests` library is not available, "
                "so Qwen3 analysis cannot be performed.\n"
            )

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

        if self.reasoning_mode == "thinking":
            # Strip any `<think>` reasoning blocks so that only the final markdown
            # answer is returned to the caller.
            result = self._extract_markdown_from_reasoning_output(raw_result)
        else:
            # Instruction-only models do not emit `<think>` blocks; use the raw text.
            result = raw_result

        if not result:
            self.logger.warning(
                "Qwen3 response contained no analysis text after applying reasoning_mode=%s",
                self.reasoning_mode,
            )
            return "No analysis text was returned by the Qwen3 model.\n"

        return result

    def _convert_with_vllm(self, image_base64: str, final_prompt: str) -> str:
        """Call a vLLM OpenAI-compatible server running Qwen2.5-VL/Qwen3-VL.

        The server is expected to expose a `/v1/chat/completions` endpoint that
        accepts multi-modal messages with an `image_url` and `text` content.
        """
        if OpenAI is None or self.client is None:  # pragma: no cover - runtime safety
            self.logger.error("OpenAI client is not available - cannot call Qwen3 via vLLM")
            return (
                "Image analysis error: the OpenAI/vLLM client is not available, "
                "so Qwen3 analysis cannot be performed.\n"
            )

        # vLLM expects a full data URL for base64 images. If the caller provided a
        # raw base64 string (our typical case), wrap it in a PNG data URL prefix.
        image_url = image_base64
        if not image_url.startswith("data:image"):
            image_url = f"data:image/png;base64,{image_base64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": final_prompt,
                    },
                ],
            }
        ]

        # Use a conservative max_tokens to avoid extremely long generations.
        safe_max_tokens = max(256, min(self.max_tokens, 4096))

        try:
            self.logger.info(
                "Sending image analysis request to Qwen3 via vLLM/OpenAI: model=%s, endpoint=%s",
                self.model_name,
                self.generate_endpoint,
            )
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=safe_max_tokens,
                temperature=self.temperature,
                top_p=0.95,
            )
        except Exception as exc:  # pragma: no cover - network/IO errors
            self.logger.error("Error calling Qwen3 via vLLM/OpenAI: %s", exc)
            return (
                "Image analysis error: failed to call Qwen3 via vLLM/OpenAI: "
                f"{exc}\n"
            )

        # Extract the text content from the first choice.
        if not getattr(response, "choices", None):
            self.logger.warning("Qwen3/vLLM response contained no choices")
            return "No analysis text was returned by the Qwen3 model.\n"

        first_choice = response.choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", "") if message is not None else ""

        # OpenAI-style clients may return content as a plain string or as a
        # list of content parts. Handle both for robustness.
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text_part = item.get("text")
                    if text_part:
                        parts.append(text_part)
            raw_result = "\n".join(parts).strip()
        else:
            raw_result = (content or "").strip()

        if not raw_result:
            self.logger.warning("Qwen3/vLLM response contained empty content")
            return "No analysis text was returned by the Qwen3 model.\n"

        if self.reasoning_mode == "thinking":
            # Strip any `<think>` reasoning blocks so that only the final markdown
            # answer is returned to the caller.
            result = self._extract_markdown_from_reasoning_output(raw_result)
        else:
            # Instruction-only models do not emit `<think>` blocks; use the raw text.
            result = raw_result

        if not result:
            self.logger.warning(
                "Qwen3 response contained no analysis text after applying reasoning_mode=%s (vLLM backend)",
                self.reasoning_mode,
            )
            return "No analysis text was returned by the Qwen3 model.\n"

        return result

    def _convert_with_transformers(self, image_base64: str, final_prompt: str) -> str:
        """Call Qwen3-VL using HuggingFace transformers library.

        This method uses the transformers library to run the model locally on GPU(s).
        The model and processor should have been initialized in __init__.
        """
        if self.model is None or self.processor is None:
            self.logger.error("Transformers model/processor not available - cannot call Qwen3 via transformers")
            return (
                "Image analysis error: the transformers model/processor is not available, "
                "so Qwen3 analysis cannot be performed.\n"
            )

        try:
            import base64
            from io import BytesIO
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - runtime safety
            self.logger.error(f"Required libraries not available: {exc}")
            return (
                "Image analysis error: required libraries (PIL/base64) are not available.\n"
            )

        # Decode base64 image to PIL Image
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
        except Exception as exc:
            self.logger.error(f"Failed to decode base64 image: {exc}")
            return (
                "Image analysis error: failed to decode the base64 image data.\n"
            )

        # Prepare messages in the format expected by Qwen3-VL
        # Based on the reference code from the task
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,  # PIL Image object
                    },
                    {
                        "type": "text",
                        "text": final_prompt,
                    },
                ],
            }
        ]

        try:
            self.logger.info(
                "Sending image analysis request to Qwen3 via transformers: model=%s",
                self.transformers_model_name,
            )

            # Apply chat template and prepare inputs
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)

            # Get generation hyperparameters from environment or use defaults
            # Based on VL hyperparameters from the task reference
            max_new_tokens = int(os.getenv("QWEN_TRANSFORMERS_MAX_NEW_TOKENS", "2048"))
            top_p = float(os.getenv("QWEN_TRANSFORMERS_TOP_P", "0.8"))
            top_k = int(os.getenv("QWEN_TRANSFORMERS_TOP_K", "20"))
            repetition_penalty = float(os.getenv("QWEN_TRANSFORMERS_REPETITION_PENALTY", "1.0"))

            # Use do_sample=False for faster deterministic generation (reduces GPU time)
            # This helps prevent CUDA watchdog timeouts
            do_sample = os.getenv("QWEN_TRANSFORMERS_DO_SAMPLE", "false").lower() == "true"

            # Generate output with timeout-friendly settings
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
            }

            # Only add sampling parameters if do_sample=True
            if do_sample:
                generate_kwargs["temperature"] = self.temperature
                generate_kwargs["top_p"] = top_p
                generate_kwargs["top_k"] = top_k
                generate_kwargs["do_sample"] = True
            else:
                # Deterministic generation (faster, no sampling overhead)
                generate_kwargs["do_sample"] = False

            generated_ids = self.model.generate(
                **inputs,
                **generate_kwargs
            )

            # Trim the input tokens from the generated output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # Decode the output
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            # Extract the first result (batch_decode returns a list)
            raw_result = output_text[0].strip() if output_text else ""

            if not raw_result:
                self.logger.warning("Qwen3/transformers response contained no text")
                return "No analysis text was returned by the Qwen3 model.\n"

            if self.reasoning_mode == "thinking":
                # Strip any `<think>` reasoning blocks so that only the final markdown
                # answer is returned to the caller.
                result = self._extract_markdown_from_reasoning_output(raw_result)
            else:
                # Instruction-only models do not emit `<think>` blocks; use the raw text.
                result = raw_result

            if not result:
                self.logger.warning(
                    "Qwen3 response contained no analysis text after applying reasoning_mode=%s (transformers backend)",
                    self.reasoning_mode,
                )
                return "No analysis text was returned by the Qwen3 model.\n"

            return result

        except Exception as exc:  # pragma: no cover - runtime errors
            self.logger.error(f"Error calling Qwen3 via transformers: {exc}")
            return (
                "Image analysis error: failed to call Qwen3 via transformers: "
                f"{exc}\n"
            )

