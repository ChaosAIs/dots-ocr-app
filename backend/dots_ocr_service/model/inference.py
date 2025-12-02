import json
import io
import base64
import math
from PIL import Image
import requests
from dots_ocr.utils.image_utils import PILimage_to_base64
from openai import OpenAI
import os


def inference_with_vllm(
        image,
        prompt,
        ip="localhost",
        port=8000,
        temperature=0.1,
        top_p=0.9,
        max_completion_tokens=32768,
        model_name='model',
        ):
    import logging
    logger = logging.getLogger(__name__)

    addr = f"http://{ip}:{port}/v1"

    # Log image details for debugging
    image_width, image_height = image.size
    image_pixels = image_width * image_height
    logger.info(f"🖼️  Image dimensions: {image_width}x{image_height} = {image_pixels:,} pixels")
    logger.info(f"🔗 vLLM server: {addr}")
    logger.info(f"🎯 Model: {model_name}")
    logger.info(f"📝 Prompt length: {len(prompt)} chars")
    logger.info(f"🎲 Temperature: {temperature}, Top-p: {top_p}")
    logger.info(f"📊 Max completion tokens: {max_completion_tokens}")

    # Convert image to base64 and log size
    base64_image = PILimage_to_base64(image)
    base64_size_mb = len(base64_image) / (1024 * 1024)
    logger.info(f"📦 Base64 image size: {base64_size_mb:.2f} MB")

    # Set timeout to 20 minutes (1200 seconds) for large images
    # Set max_retries to 2 to allow some retries for transient connection issues
    client = OpenAI(
        api_key="{}".format(os.environ.get("API_KEY", "0")),
        base_url=addr,
        timeout=1200.0,  # 20 minutes timeout per request for large images
        max_retries=2   # Allow 2 retries for connection errors
    )

    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url":  base64_image},
                },
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"}  # if no "<|img|><|imgpad|><|endofimg|>" here,vllm v1 will add "\n" here
            ],
        }
    )

    try:
        logger.info(f"🚀 Sending request to vLLM server...")
        response = client.chat.completions.create(
            messages=messages,
            model=model_name,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p)
        response_text = response.choices[0].message.content
        logger.info(f"✅ Inference successful! Response length: {len(response_text)} chars")
        return response_text
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Request error: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        logger.error(f"   Image size: {image_width}x{image_height} ({image_pixels:,} pixels)")
        logger.error(f"   Base64 size: {base64_size_mb:.2f} MB")
        print(f"Request error: {e}")
        raise Exception(f"Inference API request failed: {e}")
    except Exception as e:
        logger.error(f"❌ Inference error: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        logger.error(f"   Image size: {image_width}x{image_height} ({image_pixels:,} pixels)")
        logger.error(f"   Base64 size: {base64_size_mb:.2f} MB")

        # Try to extract more details from the error
        if hasattr(e, 'response'):
            logger.error(f"   Response status: {e.response.status_code if hasattr(e.response, 'status_code') else 'N/A'}")
            logger.error(f"   Response body: {e.response.text if hasattr(e.response, 'text') else 'N/A'}")

        print(f"Inference error: {e}")
        raise Exception(f"Inference failed: {e}")

