#!/usr/bin/env python3
"""
Diagnostic script to check the vLLM inference server status
"""
import requests
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Dots OCR vLLM server configuration
DOTS_OCR_VLLM_HOST = os.getenv('DOTS_OCR_VLLM_HOST', 'localhost')
DOTS_OCR_VLLM_PORT = os.getenv('DOTS_OCR_VLLM_PORT', '8001')
DOTS_OCR_VLLM_MODEL = os.getenv('DOTS_OCR_VLLM_MODEL', 'dots_ocr')

base_url = f"http://{DOTS_OCR_VLLM_HOST}:{DOTS_OCR_VLLM_PORT}"

print("=" * 60)
print("Dots OCR vLLM Inference Server Diagnostic")
print("=" * 60)
print(f"Server URL: {base_url}")
print(f"Model Name: {DOTS_OCR_VLLM_MODEL}")
print("=" * 60)

# Test 1: Check if server is reachable
print("\n[1] Checking if server is reachable...")
try:
    response = requests.get(f"{base_url}/health", timeout=5)
    if response.status_code == 200:
        print("✓ Server is reachable (health check passed)")
    else:
        print(f"✗ Server returned status code: {response.status_code}")
except requests.exceptions.ConnectionError:
    print(f"✗ Cannot connect to server at {base_url}")
    print("  → Make sure the Dots OCR vLLM server is running")
    print("  → Check DOTS_OCR_VLLM_HOST and DOTS_OCR_VLLM_PORT in .env file")
    sys.exit(1)
except requests.exceptions.Timeout:
    print("✗ Connection timeout")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 2: Check available models
print("\n[2] Checking available models...")
try:
    response = requests.get(f"{base_url}/v1/models", timeout=5)
    if response.status_code == 200:
        models = response.json()
        print(f"✓ Models endpoint accessible")
        if 'data' in models:
            print(f"  Available models:")
            for model in models['data']:
                model_id = model.get('id', 'unknown')
                print(f"    - {model_id}")
                if model_id == DOTS_OCR_VLLM_MODEL:
                    print(f"      ✓ Configured model '{DOTS_OCR_VLLM_MODEL}' is available")
        else:
            print(f"  Response: {models}")
    else:
        print(f"✗ Models endpoint returned status code: {response.status_code}")
except Exception as e:
    print(f"✗ Error checking models: {e}")

# Test 3: Check server load/stats (if available)
print("\n[3] Checking server statistics...")
try:
    # Some vLLM servers expose stats at /stats or /metrics
    for endpoint in ['/stats', '/metrics', '/v1/stats']:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✓ Stats available at {endpoint}")
                # Try to parse and display relevant info
                try:
                    stats = response.json()
                    print(f"  Stats: {stats}")
                except:
                    print(f"  (Raw response, not JSON)")
                break
        except:
            continue
    else:
        print("  ℹ No stats endpoint found (this is normal for some vLLM versions)")
except Exception as e:
    print(f"  ℹ Could not retrieve stats: {e}")

# Test 4: Simple inference test
print("\n[4] Testing simple inference...")
print("  (This will send a minimal test request)")
try:
    from openai import OpenAI
    
    client = OpenAI(
        api_key=os.environ.get("API_KEY", "0"),
        base_url=f"{base_url}/v1"
    )
    
    # Simple text-only test (no image)
    print("  Sending test request...")
    response = client.chat.completions.create(
        model=DOTS_OCR_VLLM_MODEL,
        messages=[
            {"role": "user", "content": "Hello"}
        ],
        max_tokens=10,
        temperature=0.1,
    )
    
    print("✓ Inference test successful")
    print(f"  Response: {response.choices[0].message.content[:100]}")
    
except Exception as e:
    print(f"✗ Inference test failed: {e}")
    print("\n  Possible causes:")
    print("  1. Server is overloaded or out of memory")
    print("  2. Model is not loaded correctly")
    print("  3. Server configuration issue")
    print("\n  Recommendations:")
    print("  - Check vLLM server logs for errors")
    print("  - Restart the vLLM server")
    print("  - Reduce NUM_WORKERS in backend/.env to 1")
    print("  - Check GPU/CPU memory usage")

print("\n" + "=" * 60)
print("Diagnostic complete")
print("=" * 60)

