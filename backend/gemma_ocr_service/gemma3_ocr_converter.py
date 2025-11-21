#!/usr/bin/env python3
"""Gemma3 OCR / Image Analysis Converter.

This module provides a small wrapper around a local Ollama instance running a
Gemma3 model. It exposes a helper to send a base64-encoded image to the model
and get markdown analysis back.
"""

import os
import logging
import base64
import io
from typing import Optional, Tuple
from utils.image_utils import resize_image_if_needed, get_image_dimensions

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    requests = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    Image = None  # type: ignore


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

        # Pixel area threshold for simple vs complex image classification
        # Default: 40000 pixels (approximately 200x200)
        pixel_threshold_env = os.getenv("GEMMA_IMAGE_PIXEL_AREA_THRESHOLD", "40000")
        self.pixel_area_threshold = int(pixel_threshold_env)

        self.generate_endpoint = f"{self.base_url}/api/generate"

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

    def _build_simple_prompt(self) -> str:
        """Build a simple prompt for small/basic images (icons, labels, single words)."""
        return (
            "You are a helpful assistant that can read and understand images.\n"
            "This is a small image, likely containing simple content.\n\n"
            "CRITICAL RULES:\n"
            "1. If the image is too blurry, corrupted, empty, or contains no readable text/content:\n"
            "   - Return absolutely NOTHING (completely empty response)\n"
            "   - Do NOT return any messages like 'The image is too blurry' or 'No content found'\n"
            "   - Do NOT return any base64 strings, encoded data, or technical information\n"
            "   - Do NOT return any explanations or warnings\n"
            "   - Just return empty/blank response\n\n"
            "2. NEVER RETURN ENCODED DATA:\n"
            "   - Do NOT return base64 strings (long strings of random characters)\n"
            "   - Do NOT return binary data or encoded image data\n"
            "   - Do NOT return data URLs (data:image/png;base64,...)\n"
            "   - Do NOT return any technical metadata or file information\n"
            "   - ONLY return human-readable text content extracted from the image\n\n"
            "3. If you CAN extract content, follow these rules:\n"
            "   - If it's a single word or short phrase: output just the text.\n"
            "   - If it's a table or form: output the text and layout as original format with markdown.\n"
            "   - If it's a logo or icon: output the name/label only.\n"
            "   - If it's a label: output the text content.\n"
            "   - If it's a diagram: briefly describe its workflow.\n"
            "   - If it's an object: briefly describe what is it.\n\n"
            "IMPORTANT: Preserve the original language of the content in the image. "
            "If the text is in Chinese, output in Chinese. If it's in English, output in English. "
            "Do NOT translate the extracted content to another language.\n\n"
            "Note: Keep your response concise and direct. No need for sections or detailed analysis.\n"
        )

    def _build_complex_prompt(self) -> str:
        """Build a detailed prompt for complex images (charts, tables, diagrams, forms)."""
        return (
            "CRITICAL RULES - READ FIRST:\n\n"
            "1. If the image is too blurry, corrupted, empty, or contains no readable text/content:\n"
            "   - Return absolutely NOTHING (completely empty response)\n"
            "   - Do NOT return any messages like 'The image is too blurry' or 'No content found'\n"
            "   - Do NOT return any base64 strings, encoded data, or technical information\n"
            "   - Do NOT return any explanations or warnings\n"
            "   - Just return empty/blank response\n\n"
            "2. NEVER RETURN ENCODED DATA:\n"
            "   - Do NOT return base64 strings (long strings of random characters)\n"
            "   - Do NOT return binary data or encoded image data\n"
            "   - Do NOT return data URLs (data:image/png;base64,...)\n"
            "   - Do NOT return any technical metadata or file information\n"
            "   - ONLY return human-readable text content extracted from the image\n\n"
            "3. LANGUAGE PRESERVATION:\n"
            "   - You MUST preserve the ORIGINAL LANGUAGE of the text in the image\n"
            "   - If the image contains Chinese text, respond ENTIRELY in Chinese\n"
            "   - If the image contains English text, respond ENTIRELY in English\n"
            "   - If the image contains other languages, respond in that language\n"
            "   - Do NOT translate the content\n"
            "   - All section headers, descriptions, and analysis MUST be in the same language as the image content\n\n"
            "4. If you CAN extract content, you are analyzing a complex image containing detailed information.\n\n"
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
        """Build the default prompt used for Gemma3 image analysis.

        This is an alias for _build_complex_prompt() for backward compatibility.
        """
        return self._build_complex_prompt()

    def _filter_encoded_data(self, text: str) -> str:
        """Remove base64 strings and other encoded data from the output.

        This is a safety filter to catch any base64-encoded image data or other
        binary/encoded content that the LLM might accidentally include in its response.

        Args:
            text: The raw text output from the LLM

        Returns:
            Cleaned text with encoded data removed
        """
        if not text:
            return ""

        # Pattern to detect base64-like strings (long sequences of alphanumeric + / + =)
        # We look for strings that are:
        # 1. At least 100 characters long (to avoid false positives)
        # 2. Contain mostly base64 characters (A-Za-z0-9+/=)
        # 3. Have very few spaces or newlines (typical of base64)

        lines = text.split('\n')
        filtered_lines = []

        for line in lines:
            line_stripped = line.strip()

            # Skip lines that look like base64 data:
            # - Very long lines (>200 chars) with mostly base64 characters
            # - Lines that start with common data URL prefixes
            if len(line_stripped) > 200:
                # Count base64-like characters
                base64_chars = sum(1 for c in line_stripped if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
                total_chars = len(line_stripped)

                # If more than 90% of characters are base64-like, skip this line
                if total_chars > 0 and (base64_chars / total_chars) > 0.9:
                    self.logger.warning(f"Filtered out suspected base64 data (line length: {len(line_stripped)})")
                    continue

            # Skip lines that start with data URL prefixes
            if line_stripped.startswith('data:image/') or line_stripped.startswith('iVBORw0KGgo'):
                self.logger.warning("Filtered out data URL or base64 image data")
                continue

            filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def _classify_image_content_with_llm(self, image_base64: str) -> bool:
        """Use LLM to determine if image contains complex content requiring detailed analysis.

        This method sends a lightweight classification prompt to the LLM to detect:
        - Statistical reports, charts, graphs with numbers
        - Diagrams, flowcharts, process flows
        - Mechanical or electrical components/schematics
        - Tables with data, forms with multiple fields
        - Technical drawings or blueprints

        Args:
            image_base64: Raw base64 string of the image content.

        Returns:
            True if image needs complex analysis, False if simple analysis is sufficient.
        """
        classification_prompt = (
            "You are an image content classifier. Analyze this image and determine if it requires COMPLEX or SIMPLE analysis.\n\n"
            "Return COMPLEX if the image contains ANY of the following:\n"
            "- Statistical reports, charts, graphs, or dashboards with numbers\n"
            "- Diagrams with flows, processes, or relationships\n"
            "- Mechanical components, electrical circuits, or technical schematics\n"
            "- Tables with multiple rows/columns of data\n"
            "- Forms with many fields or complex layouts\n"
            "- Technical drawings, blueprints, or architectural plans\n"
            "- Multi-section documents with structured information\n\n"
            "Return SIMPLE if the image contains:\n"
            "- Single words, short phrases, or labels\n"
            "- Simple icons or logos\n"
            "- Basic objects without technical details\n"
            "- Plain text without complex structure\n\n"
            "Respond with ONLY one word: either 'COMPLEX' or 'SIMPLE' (no explanation needed)."
        )

        try:
            # Use a lightweight call with minimal tokens for fast classification
            result = self._classify_with_ollama(image_base64, classification_prompt)

            # Parse the response
            result_upper = result.strip().upper()
            is_complex = "COMPLEX" in result_upper

            self.logger.info(
                f"LLM content classification: result='{result.strip()}', "
                f"classification={'COMPLEX' if is_complex else 'SIMPLE'}"
            )

            return is_complex

        except Exception as e:
            self.logger.warning(f"LLM classification failed: {e}, defaulting to area-based classification")
            # If LLM classification fails, fall back to area-based classification
            return not self._is_simple_image_by_area(image_base64)

    def _classify_with_ollama(self, image_base64: str, prompt: str) -> str:
        """Lightweight classification call using Ollama backend."""
        if requests is None:
            raise RuntimeError("requests library not available")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": 0.0,  # Use deterministic output for classification
                "num_predict": 10,   # Only need one word response
            },
        }

        response = requests.post(
            self.generate_endpoint,
            json=payload,
            timeout=30,  # Short timeout for classification
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.status_code}")

        data = response.json()
        return (data.get("response") or "").strip()

    def _is_simple_image_by_area(self, image_base64: str) -> bool:
        """Determine if an image is simple based on pixel area only.

        This is the original area-based classification method.

        Args:
            image_base64: Raw base64 string of the image content.

        Returns:
            True if image is simple (small area), False if complex (large area).
        """
        width, height = self._get_image_dimensions(image_base64)

        # If we couldn't get dimensions, default to complex analysis (safer)
        if width == 0 or height == 0:
            return False

        # Calculate total pixel area
        pixel_area = width * height

        # Image is simple if total area is small
        is_simple = pixel_area < self.pixel_area_threshold

        self.logger.info(
            f"Area-based classification: dimensions={width}x{height}, area={pixel_area:,}, "
            f"classification={'SIMPLE' if is_simple else 'COMPLEX'} "
            f"(area_threshold={self.pixel_area_threshold:,})"
        )

        return is_simple

    def _is_simple_image(self, image_base64: str) -> bool:
        """Determine if an image is simple or complex using both area and LLM-based classification.

        This method combines two classification approaches:
        1. Area-based: Small images (< threshold pixels) are typically simple
        2. LLM-based: Content analysis to detect charts, diagrams, technical content, etc.

        An image is considered COMPLEX if:
        - It's large (area >= threshold), OR
        - LLM detects complex content (charts, diagrams, statistics, technical drawings)

        An image is considered SIMPLE only if:
        - It's small (area < threshold), AND
        - LLM confirms simple content (labels, icons, plain text)

        Environment variables:
        - GEMMA_IMAGE_PIXEL_AREA_THRESHOLD: Pixel area threshold (default: 40000)
        - GEMMA_USE_LLM_CLASSIFICATION: Enable/disable LLM classification (default: true)

        Args:
            image_base64: Raw base64 string of the image content.

        Returns:
            True if image is simple (use simple analysis), False if complex (use deep analysis).
        """
        # Check if LLM classification is enabled (default: true)
        use_llm_classification = os.getenv("GEMMA_USE_LLM_CLASSIFICATION", "true").lower() in ("true", "1", "yes")

        # First check: area-based classification
        is_simple_by_area = self._is_simple_image_by_area(image_base64)

        # If LLM classification is disabled, use only area-based classification
        if not use_llm_classification:
            self.logger.info(
                f"Final classification (area-only): {'SIMPLE' if is_simple_by_area else 'COMPLEX'}"
            )
            return is_simple_by_area

        # If image is large by area, it's definitely complex - skip LLM call for efficiency
        if not is_simple_by_area:
            self.logger.info("Image is large by area - classified as COMPLEX (skipping LLM check)")
            return False

        # If image is extremely small (< 5000 pixels), it's definitely simple - skip LLM call
        # Examples: small icons, labels, single characters (e.g., 27x8=216, 100x50=5000)
        width, height = self._get_image_dimensions(image_base64)
        pixel_area = width * height
        min_llm_check_threshold = 5000

        if pixel_area < min_llm_check_threshold:
            self.logger.info(
                f"Image is very small ({width}x{height}={pixel_area:,} pixels < {min_llm_check_threshold:,}) "
                f"- classified as SIMPLE (skipping LLM check)"
            )
            return True

        # Image is small by area, but use LLM to check if content is complex
        # (e.g., small chart, small technical diagram)
        is_complex_by_content = self._classify_image_content_with_llm(image_base64)

        # Final decision: simple only if small area AND simple content
        is_simple = is_simple_by_area and not is_complex_by_content

        self.logger.info(
            f"Final classification: area={'SIMPLE' if is_simple_by_area else 'COMPLEX'}, "
            f"content={'COMPLEX' if is_complex_by_content else 'SIMPLE'}, "
            f"final={'SIMPLE' if is_simple else 'COMPLEX'}"
        )

        return is_simple

    def convert_image_base64_to_markdown(
        self,
        image_base64: str,
        prompt: Optional[str] = None,
    ) -> str:
        """Call Gemma3 (via Ollama) to analyze a base64-encoded image.

        This method automatically detects image size and content to apply:
        - Simple analysis for small images with simple content (< 200px dimension or < 50k pixels)
        - Complex/deep analysis for larger images or images with complex content (charts, tables, diagrams)

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

        # Check if image is extremely small (< 1000 pixels) - likely unreadable
        # Skip OCR entirely to avoid LLM returning base64 strings or warning messages
        width, height = self._get_image_dimensions(image_base64)
        pixel_area = width * height
        min_ocr_threshold = 1000

        if pixel_area < min_ocr_threshold:
            self.logger.info(
                f"⚠️ Image too small for OCR ({width}x{height}={pixel_area:,} pixels < {min_ocr_threshold:,}) "
                f"- returning empty result"
            )
            return ""

        # If no custom prompt provided, choose based on ORIGINAL image size and content
        # This ensures classification is based on content complexity, not GPU constraints
        if prompt is None:
            if self._is_simple_image(image_base64):
                final_prompt = self._build_simple_prompt()
            else:
                final_prompt = self._build_complex_prompt()
        else:
            final_prompt = prompt

        # Resize image if needed to prevent processing issues
        # This happens AFTER classification to avoid affecting prompt selection
        # Get max area from environment or use default 2.56M pixels (e.g., 1600x1600)
        max_area = int(os.getenv("GEMMA_TRANSFORMERS_MAX_IMAGE_AREA", "2560000"))

        # Get original image dimensions for debug logging
        original_dims = get_image_dimensions(image_base64)
        original_area = original_dims[0] * original_dims[1] if original_dims else 0

        # Resize if needed
        resized_image_base64 = resize_image_if_needed(image_base64, max_area=max_area, logger_instance=self.logger)

        # Debug log: show resize result
        if resized_image_base64 != image_base64:
            new_dims = get_image_dimensions(resized_image_base64)
            new_area = new_dims[0] * new_dims[1] if new_dims else 0
            self.logger.debug(
                f"🔄 Image resize completed: {original_dims[0]}x{original_dims[1]} ({original_area:,} pixels) "
                f"→ {new_dims[0]}x{new_dims[1]} ({new_area:,} pixels), "
                f"reduction: {((1 - new_area/original_area) * 100):.1f}%"
            )
        else:
            self.logger.debug(
                f"✓ Image resize skipped: {original_dims[0]}x{original_dims[1]} ({original_area:,} pixels) "
                f"≤ max_area ({max_area:,} pixels)"
            )

        image_base64 = resized_image_base64

        if requests is None:  # pragma: no cover - hit only when requests missing
            self.logger.error("requests library not installed - cannot call Ollama/Gemma3")
            return (
                "Image analysis error: the `requests` library is not available, "
                "so Gemma3 analysis cannot be performed.\n"
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
            return ""

        # Filter out any base64 or encoded data that might have slipped through
        result = self._filter_encoded_data(result)

        return result


