from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from fastembed import TextEmbedding
import os
from utils import load_config

config = load_config()

def get_embedding_function():
    if config['use_fast_embed']:
        return TextEmbedding()
    else:
        return HuggingFaceEmbeddings(model_name=config['embedding_model'])

def process_documents(uploaded_files):
    documents = []
    for file in uploaded_files:
        temp_file_path = f"temp_{file.name}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.getvalue())
        loader = PyPDFLoader(temp_file_path)
        documents.extend(loader.load())
        os.remove(temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap']
    )
    texts = text_splitter.split_documents(documents)

    embeddings = get_embedding_function()
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    vectorstore.persist()

    return len(texts)
