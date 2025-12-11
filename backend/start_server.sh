#!/bin/bash
# Startup script for running the backend with Ollama backend
# This is the recommended approach - simpler and more resource-efficient

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

echo "=========================================="
echo "Starting Backend with Ollama Backend"
echo "=========================================="
echo ""
echo "Ollama Services:"
echo "  OCR Service:  ollama-qwen3-vl (port 11435)"
echo "  RAG Service:  ollama-llm (port 11434)"
echo ""

# Load and display key configuration from .env
if [ -f ".env" ]; then
    echo "Configuration from .env:"
    echo "  OCR_OLLAMA_BASE_URL: $(grep '^OCR_OLLAMA_BASE_URL=' .env | cut -d'=' -f2)"
    echo "  OCR_OLLAMA_MODEL: $(grep '^OCR_OLLAMA_MODEL=' .env | cut -d'=' -f2)"
    echo "  RAG_OLLAMA_HOST: $(grep '^RAG_OLLAMA_HOST=' .env | cut -d'=' -f2)"
    echo "  RAG_OLLAMA_PORT: $(grep '^RAG_OLLAMA_PORT=' .env | cut -d'=' -f2)"
    echo "  QWEN_BACKEND: $(grep '^QWEN_BACKEND=' .env | cut -d'=' -f2)"
fi
echo "=========================================="

# Start the backend
python main.py

