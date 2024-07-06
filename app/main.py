import streamlit as st
import sys
import os
from typing import List
import logging
import argparse

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.document_processor import process_documents, get_existing_documents, clear_vectorstore
from app.model_handler import ModelHandler
from app.rag import retrieve_context
from app.utils import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Document QnA System")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()

# Load configuration
config = load_config()

# Initialize ModelHandler
@st.cache_resource
def get_model_handler():
    return ModelHandler(config)

model_handler = get_model_handler()

st.title("Document QnA System")

# Sidebar
st.sidebar.title("Settings")

# File uploader in sidebar
uploaded_files = st.sidebar.file_uploader("Upload PDF documents", accept_multiple_files=True, type=['pdf'])

# Get existing documents
try:
    existing_docs = get_existing_documents()
    if existing_docs:
        st.sidebar.write("Existing documents:")
        for doc in existing_docs:
            st.sidebar.write(f"- {doc}")

        if st.sidebar.button("Clear Vector Store"):
            if clear_vectorstore():
                st.sidebar.success("Vector store cleared successfully.")
                st.experimental_rerun()
            else:
                st.sidebar.error("Failed to clear vector store.")
    else:
        st.sidebar.write("No existing documents found.")
except Exception as e:
    logger.error(f"Error loading existing documents: {str(e)}")
    st.sidebar.error(f"Error loading existing documents: {str(e)}")

if uploaded_files:
    rebuild = st.sidebar.checkbox("Rebuild Vector Store", value=False)
    if st.sidebar.button("Process Documents"):
        with st.spinner("Processing documents..."):
            try:
                num_chunks = process_documents(uploaded_files, rebuild)
                st.sidebar.success(f"Processed {num_chunks} chunks from {len(uploaded_files)} documents")
                st.experimental_rerun()
            except Exception as e:
                logger.error(f"Error processing documents: {str(e)}")
                st.sidebar.error(f"Error processing documents: {str(e)}")

# Model selection
model_choice = st.sidebar.selectbox("Choose a model", ["Llama 3", "Mistral"], index=0 if config['default_model'] == 'llama' else 1)

# Debug mode
debug_mode = st.sidebar.checkbox("Debug Mode") or args.debug

# Use RAG
use_rag = st.sidebar.checkbox("Use RAG", value=True)

# Main chat interface
st.header("Chat Interface")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Retrieve context if using RAG
        if use_rag:
            try:
                context = retrieve_context(prompt, top_k=config['top_k'])
                if debug_mode:
                    logger.info(f"RAG Context: {context}")
                    st.write("RAG Context:")
                    st.write(context)
            except Exception as e:
                logger.error(f"Error retrieving context: {str(e)}")
                st.error(f"Error retrieving context: {str(e)}")
                context = ""
        else:
            context = ""

        # Generate response
        prompt_template = f"Context: {context}\n\nHuman: {prompt}\n\nAssistant:"

        try:
            for chunk in model_handler.generate_stream(prompt_template, "llama" if model_choice == "Llama 3" else "mistral"):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            st.error(f"Error generating response: {str(e)}")
            full_response = "I apologize, but I encountered an error while generating the response."

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# About section in sidebar
st.sidebar.title("About")
st.sidebar.info("This is a RAG-based Document QnA system. Upload PDF documents and ask questions about their content.")
if os.path.exists("./chroma_db"):
    st.sidebar.info(f"RAG database: {os.path.abspath('./chroma_db')}")

if debug_mode:
    logger.info("Debug mode is enabled")
    st.sidebar.write("Debug mode is enabled")
