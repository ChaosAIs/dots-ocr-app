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

    addr = f"http://{ip}:{port}/v1"
    # Set timeout to 3 minutes (180 seconds) to prevent indefinite hanging
    # Set max_retries to 0 to disable automatic retries (we handle retries at a higher level)
    # This prevents the "never-ending" behavior where requests retry indefinitely
    client = OpenAI(
        api_key="{}".format(os.environ.get("API_KEY", "0")),
        base_url=addr,
        timeout=180.0,  # 3 minutes timeout per request
        max_retries=0   # Disable automatic retries
    )
    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url":  PILimage_to_base64(image)},
                },
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"}  # if no "<|img|><|imgpad|><|endofimg|>" here,vllm v1 will add "\n" here
            ],
        }
    )
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=model_name,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p)
        response = response.choices[0].message.content
        return response
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        raise Exception(f"Inference API request failed: {e}")
    except Exception as e:
        print(f"Inference error: {e}")
        raise Exception(f"Inference failed: {e}")

