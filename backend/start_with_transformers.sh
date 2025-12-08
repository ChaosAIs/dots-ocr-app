#!/bin/bash
# Startup script for running the backend with Qwen3 transformers backend

# Change to the script's directory
cd "$(dirname "$0")"

# Activate the virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Activated virtual environment: $(which python)"
else
    echo "Warning: Virtual environment not found at venv/bin/activate"
    echo "Using system Python: $(which python)"
fi

# Set environment variables (these can also be in .env)
export QWEN_BACKEND=transformers
export QWEN_TRANSFORMERS_ATTN_IMPL=eager
export QWEN_TRANSFORMERS_DTYPE=bfloat16
export QWEN_TRANSFORMERS_GPU_DEVICES=4,5,6,7

echo "=========================================="
echo "Starting Backend with Transformers Backend"
echo "=========================================="
echo "QWEN_BACKEND: $QWEN_BACKEND"
echo "QWEN_TRANSFORMERS_GPU_DEVICES: $QWEN_TRANSFORMERS_GPU_DEVICES"
echo "QWEN_TRANSFORMERS_DTYPE: $QWEN_TRANSFORMERS_DTYPE"
echo "=========================================="

# Start the backend
python main.py

