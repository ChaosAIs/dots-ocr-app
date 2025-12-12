#!/bin/bash
# Startup script for running the backend
# Supports both Ollama and vLLM backends for RAG (configurable via RAG_LLM_BACKEND in .env)

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

# Load RAG LLM backend setting from .env
RAG_LLM_BACKEND="ollama"
if [ -f ".env" ]; then
    RAG_LLM_BACKEND=$(grep '^RAG_LLM_BACKEND=' .env | cut -d'=' -f2 | tr -d ' ')
    if [ -z "$RAG_LLM_BACKEND" ]; then
        RAG_LLM_BACKEND="ollama"
    fi
fi

echo "=========================================="
echo "Starting Backend"
echo "=========================================="
echo ""
echo "Dots OCR Service (Main Document Conversion):"
if [ -f ".env" ]; then
    echo "  DOTS_OCR_VLLM_HOST: $(grep '^DOTS_OCR_VLLM_HOST=' .env | cut -d'=' -f2)"
    echo "  DOTS_OCR_VLLM_PORT: $(grep '^DOTS_OCR_VLLM_PORT=' .env | cut -d'=' -f2)"
    echo "  DOTS_OCR_VLLM_MODEL: $(grep '^DOTS_OCR_VLLM_MODEL=' .env | cut -d'=' -f2)"
fi
echo ""
echo "RAG LLM Backend: $RAG_LLM_BACKEND"
if [ "$RAG_LLM_BACKEND" = "vllm" ]; then
    echo "  vLLM Service: vllm_qwen3_14b (port 8004)"
    if [ -f ".env" ]; then
        echo "  RAG_VLLM_HOST: $(grep '^RAG_VLLM_HOST=' .env | cut -d'=' -f2)"
        echo "  RAG_VLLM_PORT: $(grep '^RAG_VLLM_PORT=' .env | cut -d'=' -f2)"
        echo "  RAG_VLLM_MODEL: $(grep '^RAG_VLLM_MODEL=' .env | cut -d'=' -f2)"
    fi
else
    echo "  Ollama Service: ollama-llm (port 11434)"
    if [ -f ".env" ]; then
        echo "  RAG_OLLAMA_HOST: $(grep '^RAG_OLLAMA_HOST=' .env | cut -d'=' -f2)"
        echo "  RAG_OLLAMA_PORT: $(grep '^RAG_OLLAMA_PORT=' .env | cut -d'=' -f2)"
        echo "  RAG_OLLAMA_MODEL: $(grep '^RAG_OLLAMA_MODEL=' .env | cut -d'=' -f2)"
    fi
fi
echo ""

# Load and display OCR configuration from .env
if [ -f ".env" ]; then
    echo "Image Analysis Configuration:"
    echo "  IMAGE_ANALYSIS_BACKEND: $(grep '^IMAGE_ANALYSIS_BACKEND=' .env | cut -d'=' -f2)"
    echo "  OCR_OLLAMA_BASE_URL: $(grep '^OCR_OLLAMA_BASE_URL=' .env | cut -d'=' -f2)"
    echo "  OCR_OLLAMA_MODEL: $(grep '^OCR_OLLAMA_MODEL=' .env | cut -d'=' -f2)"
fi
echo "=========================================="

# Start the backend
python main.py

