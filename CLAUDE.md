# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Airgapped Offline RAG is a Retrieval-Augmented Generation (RAG) system for documents using local LLM inference. It uses Llama C++ for model inference, LangChain for orchestration, ChromaDB for vector storage, and Streamlit for the UI. The system is designed to work completely offline with GGUF quantized models.

## Development Commands

### Setup and Installation
```bash
make setup        # Create virtual environment and install dependencies
make install      # Install requirements only
make clean        # Clean up virtual environment
```

### Running the Application
```bash
make run          # Run Streamlit app locally (runs: streamlit run app/main.py)
make docker-build # Build Docker image
make docker-run   # Run Docker container
```

### Testing and Code Quality
```bash
make test         # Run pytest tests (PYTHONPATH=$(PWD) pytest tests/)
make precommit    # Run pre-commit hooks (security checks, linting, formatting)
```

## Key Architecture Components

### Core Application Structure
- **app/main.py**: Streamlit UI entry point with custom dark theme CSS
- **app/rag.py**: RAG implementation with ChromaDB vector store and context retrieval
- **app/document_processor.py**: PDF processing and document chunking logic
- **app/model_handler.py**: LLM model loading and inference using Llama C++
- **app/utils.py**: Configuration loading and utilities

### Configuration
- **config.yaml**: Main configuration file with model paths, RAG settings, and embedding configuration
  - Model paths use environment variable substitution (e.g., `${LLAMA_MODEL_PATH:-"default/path"}`)
  - Supports Llama, Mistral, and Gemma models in GGUF format
  - Configurable chunk size, overlap, and retrieval parameters

### Testing
Tests use pytest and are located in `tests/` directory. Each main module has a corresponding test file.

### Dependencies and Environment
- Python 3.9 required (specified in runtime.txt)
- Key dependencies: llama-cpp-python, langchain, chromadb, streamlit, sentence-transformers
- Pre-commit hooks configured for security (gitleaks, detect-secrets), linting (flake8, black), and formatting

### Models
Models should be placed in the `models/` directory in GGUF format. The system supports:
- Llama 3 (8B) models
- Mistral (7B) models  
- Gemma (2B) models