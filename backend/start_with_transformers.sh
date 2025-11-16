#!/bin/bash
# Startup script for running the backend with Qwen3 transformers backend

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

