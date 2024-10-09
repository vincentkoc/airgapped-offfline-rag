# Offline Airgapped RAG

## Lightweight Local Document RAG

This project implements a Retrieval-Augmented Generation (RAG) based Question-Answering system for documents. It uses Llama 3, Mistral, and Gemini models for local inference with LlaMa c++, langchain for orchestration, chromadb for vector storage, and Streamlit for the user interface.

## Setup

1. Ensure you have Python 3.9 installed. You can use `pyenv`:

   ```
   pyenv install 3.9.16
   pyenv local 3.9.16
   pyenv rehash
   ```

2. Create a virtual environment and install dependencies:

   ```
   make setup
   ```

3. Download the Llama 3 (8B) and Mistral (7B) models in GGUF format and place them in the `models/` directory. `TheBloke` on Hugging Face has shared the models [here](https://huggingface.co/TheBloke):
- https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q3_K_L.gguf
- https://huggingface.co/TheBloke/LLaMA-Pro-8B-Instruct-GGUF/blob/main/llama-pro-8b-instruct.Q3_K_L.gguf

1. Qdrant sentence transformer model will be downloaded automatically on first run. If you are running the airgapped RAG locally then best to run the codebase locally first with internet access and then run the codebase locally without internet access once the model has been downloaded to the models directory.

## Running the Application

### Locally

```
make run
```

### Using Docker

```
make docker-build
make docker-run
```

## Usage

1. Upload PDF documents using the file uploader.
2. Select the model you want to use (i.e Mistral).
3. Enter your question in the text input.
4. Click "Generate Answer" to get a response based on the document content.

## Configuration

Adjust settings in `config.yaml` to modify model paths, chunk sizes, and other parameters.

## Note

Ensure you have the necessary permissions and licenses to use the third-party models.
