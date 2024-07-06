from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import os
import shutil
from .utils import load_config
import streamlit as st

config = load_config()

@st.cache_resource
def get_embedding_function():
    try:
        return FastEmbedEmbeddings(
            model_name=config['embedding_model'],
            max_length=512,
            doc_embed_type="passage"
        )
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        st.info("Using fallback embedding method.")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings()

def process_documents(uploaded_files, rebuild=False):
    if rebuild:
        # Remove existing Chroma database
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")

    documents = []
    for file in uploaded_files:
        temp_file_path = f"temp_{file.name}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.getvalue())
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        # Update the metadata with the correct filename
        for doc in docs:
            doc.metadata['source'] = file.name
        documents.extend(docs)
        os.remove(temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap']
    )
    texts = text_splitter.split_documents(documents)

    embeddings = get_embedding_function()

    vectorstore = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory="./chroma_db"
    )

    return len(texts)

def get_existing_documents():
    try:
        if os.path.exists("./chroma_db"):
            embeddings = get_embedding_function()
            vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
            docs = vectorstore.get()
            unique_sources = set(metadata['source'] for metadata in docs['metadatas'])
            return list(unique_sources)
    except Exception as e:
        st.error(f"Error retrieving existing documents: {str(e)}")
    return []

def clear_vectorstore():
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        return True
    return False
