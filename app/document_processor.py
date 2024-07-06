from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import os
import shutil
from .utils import load_config
import streamlit as st
import logging

config = load_config()
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = "./documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

@st.cache_resource
def get_embedding_function():
    try:
        return FastEmbedEmbeddings(
            model_name=config['embedding_model'],
            max_length=512,
            doc_embed_type="passage",
            cache_dir="./models"
        )
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        st.error(f"Error loading embedding model: {str(e)}")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(cache_folder="./models")

@st.cache_resource
def get_vectorstore():
    embeddings = get_embedding_function()
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def process_documents(uploaded_files, rebuild=False):
    if rebuild:
        clear_vectorstore()

    documents = []
    new_files = []
    existing_files = set(f for f in os.listdir(DOCUMENTS_DIR) if f.endswith('.pdf'))

    for file in uploaded_files:
        if file.name not in existing_files:
            new_files.append(file)
            file_path = os.path.join(DOCUMENTS_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())

            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = file.name
            documents.extend(docs)

    if not documents and not rebuild:
        logger.info("No new documents to process.")
        return 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap']
    )
    texts = text_splitter.split_documents(documents)

    vectorstore = get_vectorstore()
    if texts:
        vectorstore.add_documents(texts)
        vectorstore.persist()

    return len(texts)

def get_existing_documents():
    try:
        vectorstore = get_vectorstore()

        # Get all documents from the vectorstore
        all_docs = vectorstore.get()

        # Check which documents actually exist in the file system
        existing_files = set(f for f in os.listdir(DOCUMENTS_DIR) if f.endswith('.pdf'))

        # Filter out documents that no longer exist in the file system
        existing_docs = [
            doc for doc, metadata in zip(all_docs['documents'], all_docs['metadatas'])
            if metadata['source'] in existing_files
        ]

        # Update the vectorstore to remove documents that no longer exist
        if len(existing_docs) != len(all_docs['documents']):
            vectorstore.delete(ids=[id for id, metadata in zip(all_docs['ids'], all_docs['metadatas'])
                                    if metadata['source'] not in existing_files])
            vectorstore.persist()

        return list(existing_files)
    except Exception as e:
        logger.error(f"Error retrieving existing documents: {str(e)}")
        st.error(f"Error retrieving existing documents: {str(e)}")
        return []

def clear_vectorstore():
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        logger.info("Cleared Chroma vectorstore.")
        # Clear the cached vectorstore
        get_vectorstore.clear()

    # Remove all files from the documents directory
    for file in os.listdir(DOCUMENTS_DIR):
        file_path = os.path.join(DOCUMENTS_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    logger.info("Cleared documents directory.")
    return True

def remove_document(document_name):
    file_path = os.path.join(DOCUMENTS_DIR, document_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Removed document: {document_name}")

        # Remove the document from the vectorstore
        vectorstore = get_vectorstore()
        all_docs = vectorstore.get()
        ids_to_delete = [id for id, metadata in zip(all_docs['ids'], all_docs['metadatas'])
                         if metadata['source'] == document_name]
        if ids_to_delete:
            vectorstore.delete(ids=ids_to_delete)
            vectorstore.persist()

        return True
    else:
        logger.warning(f"Document not found: {document_name}")
        return False
