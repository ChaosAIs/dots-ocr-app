"""
DeepSeek OCR Converter

This module provides functionality to convert images to markdown using DeepSeek OCR API.

The DeepSeek-OCR model is served via vLLM and provides an OpenAI-compatible API endpoint.
This converter communicates with the vLLM server to perform OCR on images.

Reference Implementation:
    - DeepSeek-OCR vLLM server: DeekSeek-OCR--Dockerized-API/start_server.py
    - Image processing: DeekSeek-OCR--Dockerized-API/custom_run_dpsk_ocr_image.py
    - Configuration: DeekSeek-OCR--Dockerized-API/custom_config.py

Default Prompt Format:
    The default prompt is: "<|grounding|>Convert the document to markdown."

    Note: When using the OpenAI-compatible API endpoint (/v1/chat/completions),
    the <image> token is NOT included in the prompt. The image is sent separately
    via the image_url field, and vLLM handles the image placement internally.

    For direct vLLM API calls (llm.generate()), the <image> token would be required.

    Common prompt variations (without <image> token for OpenAI API):
    - Document conversion: "<|grounding|>Convert the document to markdown."
    - General OCR: "<|grounding|>OCR this image."
    - Free OCR (no layout): "Free OCR."
    - Figure parsing: "Parse the figure."
    - General description: "Describe this image in detail."

Usage:
    converter = DeepSeekOCRConverter(
        api_base="http://localhost:8005/v1",
        model_name="deepseek-ai/DeepSeek-OCR"
    )

    # Convert with default prompt
    markdown = converter.convert_image_to_markdown(Path("image.png"))

    # Convert with custom prompt (no <image> token needed)
    markdown = converter.convert_image_to_markdown(
        Path("image.png"),
        prompt="Free OCR."
    )
"""

import os
import logging
import base64
from pathlib import Path
from typing import Optional, Dict, Any
from openai import OpenAI
from PIL import Image


class DeepSeekOCRConverter:
    """
    Converter class for converting images to markdown using DeepSeek OCR API.

    This converter uses the OpenAI-compatible API provided by vLLM serving DeepSeek-OCR model.
    The DeepSeek-OCR model is a vision-language model specifically designed for OCR tasks,
    supporting document layout analysis, table recognition, and markdown conversion.

    Key Features:
        - High-quality OCR with layout preservation
        - Table recognition and markdown formatting
        - Support for various image formats (PNG, JPG, GIF, BMP, TIFF, WEBP)
        - Customizable prompts for different OCR tasks
        - Grounding mode for element localization

    Attributes:
        SUPPORTED_EXTENSIONS: List of supported image file extensions
        client: OpenAI client for API communication
        model_name: Name of the DeepSeek-OCR model
        temperature: Sampling temperature (0.0 for deterministic output)
        max_tokens: Maximum tokens to generate
    """
    
    # Supported image extensions
    SUPPORTED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']
    
    def __init__(
        self,
        api_base: str = "http://localhost:8005/v1",
        model_name: str = "deepseek-ai/DeepSeek-OCR",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        ngram_size: int = 30,
        window_size: int = 90,
        whitelist_token_ids: list = None,
        timeout: int = 3600,
        use_vllm_xargs: bool = False
    ):
        """
        Initialize DeepSeek OCR Converter.

        Args:
            api_base: Base URL for the DeepSeek OCR API (vLLM server)
            model_name: Model name to use
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            ngram_size: N-gram size for logits processor
            window_size: Window size for logits processor
            whitelist_token_ids: List of token IDs to whitelist (e.g., for <td>, </td>)
            timeout: Request timeout in seconds
            use_vllm_xargs: Whether to use vllm_xargs in extra_body (default: False)
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or [128821, 128822]
        self.use_vllm_xargs = use_vllm_xargs
        
        # Initialize OpenAI client with vLLM endpoint
        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require API key
            base_url=api_base,
            timeout=timeout
        )
        
        self.logger.info(f"DeepSeek OCR Converter initialized with API base: {api_base}")
        self.logger.info(f"Model: {model_name}")
    
    def is_supported_file(self, file_path: Path) -> bool:
        """
        Check if the file is a supported image format.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is supported, False otherwise
        """
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def _encode_image_to_base64(self, image_path: Path) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string with data URI prefix
        """
        try:
            # Open and convert image to RGB
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to bytes
                import io
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                
                # Encode to base64
                image_bytes = buffer.read()
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                
                return f"data:image/png;base64,{base64_image}"
        except Exception as e:
            self.logger.error(f"Error encoding image {image_path}: {e}")
            raise
    
    def convert_image_to_markdown(
        self,
        image_path: Path,
        prompt: str = None
    ) -> str:
        """
        Convert an image to markdown using DeepSeek OCR API.

        Args:
            image_path: Path to the image file
            prompt: Prompt to use for OCR. If None, uses default prompt.
                   Note: Do NOT include <image> token when using OpenAI-compatible API.
                   The image is sent separately via image_url field.
                   Default: "<|grounding|>Convert the document to markdown."

        Returns:
            Markdown content as string

        Raises:
            ValueError: If file is not supported
            Exception: If API call fails
        """
        if not self.is_supported_file(image_path):
            raise ValueError(
                f"Unsupported file type: {image_path.suffix}. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Use default prompt if none provided
        # Note: When using OpenAI-compatible API, we don't include <image> token
        # The API handles image placement automatically via the image_url field
        # For direct vLLM API, <image> token would be required in the prompt
        if prompt is None:
            # Default instruction for document conversion
            prompt = "<|grounding|>Convert the document to markdown."

        # Remove <image> token if present (not needed for OpenAI-compatible API)
        # The image is sent separately via image_url field
        if "<image>" in prompt:
            self.logger.info("Removing <image> token from prompt (handled by API)")
            prompt = prompt.replace("<image>", "").strip()

        self.logger.info(f"Converting image to markdown: {image_path}")
        self.logger.info(f"Using prompt: {prompt[:100]}...")  # Log first 100 chars

        try:
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image_path)

            # Prepare messages for OpenAI-compatible API
            # The image is sent via image_url, and the prompt is sent as text
            # vLLM will internally handle the image processing and merge with the prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            # Call DeepSeek OCR API
            self.logger.info(f"Calling DeepSeek OCR API for {image_path.name}...")

            # Prepare extra_body parameters
            extra_body = {"skip_special_tokens": False}
            if self.use_vllm_xargs:
                extra_body["vllm_xargs"] = {
                    "ngram_size": self.ngram_size,
                    "window_size": self.window_size,
                    "whitelist_token_ids": self.whitelist_token_ids,
                }

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_body=extra_body,
            )

            # Extract markdown content
            markdown_content = response.choices[0].message.content

            # Clean up the result - remove end-of-sentence tokens if present
            if '<｜end▁of▁sentence｜>' in markdown_content:
                markdown_content = markdown_content.replace('<｜end▁of▁sentence｜>', '')
                self.logger.info("Removed end-of-sentence tokens from output")

            self.logger.info(f"Successfully converted {image_path.name} to markdown")
            self.logger.info(f"Output length: {len(markdown_content)} characters")

            return markdown_content

        except Exception as e:
            self.logger.error(f"Error converting image {image_path}: {e}")
            raise
    
    def convert_file(
        self,
        input_path: Path,
        output_path: Path,
        prompt: str = None
    ) -> bool:
        """
        Convert an image file to markdown and save to output path.

        Args:
            input_path: Path to input image file
            output_path: Path to output markdown file
            prompt: Prompt to use for OCR. If None, uses default prompt.
                   Note: Do NOT include <image> token when using OpenAI-compatible API.
                   Default: "<|grounding|>Convert the document to markdown."

        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Convert image to markdown
            markdown_content = self.convert_image_to_markdown(input_path, prompt)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write markdown to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            self.logger.info(f"Markdown saved to: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to convert {input_path}: {e}")
            return False
    
    def get_supported_extensions(self) -> list:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported extensions
        """
        return self.SUPPORTED_EXTENSIONS.copy()

