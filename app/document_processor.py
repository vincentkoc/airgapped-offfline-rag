# This file is part of airgapped-offline-rag.
#
# Airgapped Offline RAG is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Airgapped Offline RAG is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Airgapped Offline RAG. If not, see <https://www.gnu.org/licenses/>.
#
# Copyright (C) 2024 Vincent Koc (https://github.com/vincentkoc)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.schema import Document
import os
import shutil
import mimetypes
from typing import List, Optional, Union, BinaryIO
import pathlib
from .utils import load_config
from .document_handlers import DocumentHandlerFactory
from .telemetry import telemetry, track_document_processing
import streamlit as st
import logging

config = load_config()
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = "./documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# Initialize document handler factory
document_factory = DocumentHandlerFactory()

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

@track_document_processing("batch", 0)
def process_documents(uploaded_files, rebuild=False):
    """Process uploaded documents using the new document handler system"""
    if rebuild:
        clear_vectorstore()

    documents = []
    processed_files = []
    
    for file in uploaded_files:
        try:
            file_path = os.path.join(DOCUMENTS_DIR, file.name)
            
            # Save uploaded file
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            
            # Log user interaction
            telemetry.log_user_interaction(
                "document_upload",
                {
                    "filename": file.name,
                    "file_size": len(file.getvalue()),
                    "file_type": file.type if hasattr(file, 'type') else 'unknown'
                }
            )
            
            # Process document using new handler system
            docs = process_single_document(file_path)
            
            if docs:
                # Convert DocumentChunks to LangChain Documents
                langchain_docs = []
                for chunk in docs:
                    # Ensure metadata includes source
                    chunk.metadata['source'] = file.name
                    chunk.metadata['original_filename'] = file.name
                    
                    # Create LangChain Document
                    doc = Document(
                        page_content=chunk.content,
                        metadata=chunk.metadata
                    )
                    langchain_docs.append(doc)
                
                documents.extend(langchain_docs)
                processed_files.append({
                    'name': file.name,
                    'chunks': len(langchain_docs),
                    'handler': chunk.metadata.get('processor', 'unknown')
                })
                
                logger.info(f"Processed {len(langchain_docs)} chunks from {file.name} using {chunk.metadata.get('processor', 'unknown')} handler")
            else:
                logger.warning(f"No content extracted from {file.name}")
                
        except Exception as e:
            logger.error(f"Error processing {file.name}: {e}")
            telemetry.log_error(e, {"filename": file.name, "operation": "document_processing"})
            continue
    
    if not documents:
        logger.warning("No documents were loaded.")
        return 0, processed_files

    # Add to vector store
    vectorstore = get_vectorstore()
    try:
        # Filter complex metadata before adding to ChromaDB
        filtered_documents = filter_complex_metadata(documents)
        vectorstore.add_documents(filtered_documents)
        vectorstore.persist()
        logger.info(f"Added {len(filtered_documents)} chunks to the vector store")
        
        # Log successful processing
        telemetry.log_user_interaction(
            "documents_processed",
            {
                "total_files": len(uploaded_files),
                "successful_files": len(processed_files),
                "total_chunks": len(documents),
                "handlers_used": list(set(f['handler'] for f in processed_files))
            }
        )
        
    except Exception as e:
        logger.error(f"Error adding documents to vector store: {str(e)}")
        telemetry.log_error(e, {"operation": "vector_store_add", "chunk_count": len(documents)})
        raise

    return len(documents), processed_files

def process_single_document(file_path: Union[str, pathlib.Path], **kwargs):
    """Process a single document using appropriate handler"""
    try:
        file_path = pathlib.Path(file_path)
        
        # Get appropriate handler
        handler = document_factory.get_handler(file_path)
        
        if not handler:
            logger.warning(f"No handler found for {file_path}")
            return None
        
        # Validate file
        if not handler.validate_file(file_path):
            logger.warning(f"File validation failed for {file_path}")
            return None
        
        # Extract structured content
        chunk_size = kwargs.get('chunk_size', int(config['chunk_size']))
        chunk_overlap = kwargs.get('chunk_overlap', min(int(config['chunk_overlap']), chunk_size - 1))
        
        chunks = handler.extract_structured_content(
            file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error processing single document {file_path}: {e}")
        return None

def get_supported_formats():
    """Get all supported document formats"""
    return document_factory.get_supported_formats()

def get_supported_extensions():
    """Get all supported file extensions"""
    return document_factory.get_supported_extensions()

def detect_document_format(file_path_or_buffer: Union[str, pathlib.Path, BinaryIO], mimetype: Optional[str] = None) -> Optional[str]:
    """Detect document format"""
    return document_factory.detect_format(file_path_or_buffer, mimetype)

def get_existing_documents():
    try:
        vectorstore = get_vectorstore()
        # Retrieve all documents
        results = vectorstore.get()
        # Extract unique source names from the results
        documents = list(set(metadata['source'] for metadata in results['metadatas']))

        # Persist the vectorstore
        vectorstore.persist()

        return documents
    except Exception as e:
        logging.error(f"Error retrieving existing documents: {e}")
        return []

def clear_vectorstore():
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        logger.info("Cleared Chroma vectorstore.")
        get_vectorstore.clear()

    for file in os.listdir(DOCUMENTS_DIR):
        file_path = os.path.join(DOCUMENTS_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    logger.info("Cleared documents directory.")
    return True

def remove_document(document_name):
    try:
        vectorstore = get_vectorstore()

        # Check if the document file exists
        document_path = os.path.join(DOCUMENTS_DIR, document_name)
        if os.path.exists(document_path):
            # Remove the document file
            os.remove(document_path)
            logging.info(f"Removed file: {document_path}")
        else:
            logging.warning(f"Document file not found: {document_path}")

        # Get the ids of the documents to delete from vectorstore
        results = vectorstore.get(where={"source": document_name})
        if results and results['ids']:
            # Delete the documents from vectorstore
            vectorstore.delete(ids=results['ids'])
            logging.info(f"Removed {len(results['ids'])} embeddings for document: {document_name}")

            # Persist the changes
            vectorstore.persist()

            return True
        else:
            logging.warning(f"No embeddings found in vectorstore for document: {document_name}")
            return False
    except Exception as e:
        logging.error(f"Error removing document {document_name}: {e}")
        return False
