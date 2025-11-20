"""
Image utility functions for OCR services.

This module provides shared image processing utilities used across different OCR converters.
"""

import io
import base64
import logging
from typing import Optional

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)


def resize_image_if_needed(
    image_base64: str,
    max_area: int = 4000000,
    logger_instance: Optional[logging.Logger] = None
) -> str:
    """
    Resize image based on max_area while maintaining aspect ratio.

    Resizes the image if total pixel area exceeds max_area.
    This helps prevent CUDA timeout errors during inference and reduces memory usage.

    Args:
        image_base64: Base64-encoded image string
        max_area: Maximum allowed total pixel area (width × height) in pixels
                 Default: 4,000,000 pixels (equivalent to 2000×2000)
        logger_instance: Optional logger instance for logging messages

    Returns:
        Base64-encoded image string (resized if needed, original if not)

    Example:
        >>> resized = resize_image_if_needed(image_b64, max_area=2560000)
        >>> # 3068x3835 image (11.8M pixels) becomes ~1431x1788 (2.56M pixels)
    """
    log = logger_instance or logger

    if Image is None:
        log.warning("PIL not available - cannot resize image")
        return image_base64

    try:
        # Decode base64 to image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        current_area = width * height

        # Check if resize is needed
        if current_area <= max_area:
            return image_base64

        # Calculate scale factor to achieve target area
        # new_area = (width * scale) * (height * scale) = max_area
        # scale = sqrt(max_area / current_area)
        import math
        scale_factor = math.sqrt(max_area / current_area)

        # Calculate new dimensions maintaining aspect ratio
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        new_area = new_width * new_height

        log.info(
            f"Resizing image from {width}x{height} (area={current_area:,}) to "
            f"{new_width}x{new_height} (area={new_area:,}) - "
            f"scale={scale_factor:.3f}, reason=area exceeds max_area={max_area:,}"
        )

        # Resize using high-quality Lanczos resampling
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Encode back to base64
        output_buffer = io.BytesIO()
        image_format = image.format if image.format else 'PNG'
        resized_image.save(output_buffer, format=image_format)
        output_buffer.seek(0)

        resized_base64 = base64.b64encode(output_buffer.read()).decode('utf-8')

        return resized_base64

    except Exception as e:
        log.warning(f"Failed to resize image: {e}, using original")
        return image_base64


def get_image_dimensions(image_base64: str) -> Optional[tuple[int, int]]:
    """
    Get dimensions of a base64-encoded image.
    
    Args:
        image_base64: Base64-encoded image string
        
    Returns:
        Tuple of (width, height) or None if unable to determine
    """
    if Image is None:
        return None
    
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        return image.size
    except Exception:
        return None

