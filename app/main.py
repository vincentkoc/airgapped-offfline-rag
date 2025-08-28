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

import streamlit as st
import sys
import os
import logging
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64

# Set page config at the very beginning
st.set_page_config(
    layout="wide",
    page_title="Airgapped Offline RAG",
    page_icon="assets/airgapped_offline_rag_icon.png",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Custom CSS to inject into the Streamlit app
st.markdown("""
<style>
/* Overall theme */
body {
    color: #e0e0e0;
    background-color: #0a0a0a;
}

/* Adjust title and header */
.main > div:first-child h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    color: #f4298a;
}

/* Adjust column gap and padding */
.row-widget.stHorizontal {
    gap: 2rem;
    padding: 1rem 0;
}

/* Style for existing documents */
.existing-docs {
    max-height: 200px;
    overflow-y: auto;
    border: 1px solid #333;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 1rem;
    background-color: #111;
}

/* Footer styling */
.footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #333;
    color: #888;
    font-size: 0.8rem;
}

/* GitHub icon */
.github-icon {
    height: 20px;
    vertical-align: middle;
    margin-left: 5px;
    filter: invert(1);
}

/* Adjust spacing for settings and chat interface */
.stColumn > div {
    padding: 1rem;
    background-color: #111;
    border-radius: 10px;
    margin-bottom: 1rem;
}

/* Adjust font sizes */
body {
    font-size: 16px;
}

.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stCheckbox > label > div {
    font-size: 16px;
    background-color: #222;
    color: #e0e0e0;
    border-color: #444;
}

/* Remove box above filenames */
.css-1kyxreq {
    display: none;
}

/* Adjust sidebar width when chat is active */
@media (min-width: 768px) {
    .main .block-container {
        max-width: 90%;
        padding-left: 5rem;
        padding-right: 5rem;
    }
}

/* Reduce file name font size */
.existing-docs p {
    font-size: 0.9rem;
    color: #26f6cb;
}

/* Right align Process Documents button */
.stButton > button:last-child {
    float: right;
}

/* Add vertical line between settings and chat */
.main .block-container > div > div > div:nth-child(2) {
    border-left: 1px solid #333;
    padding-left: 2rem;
}

/* Styling for info and success messages */
.stAlert {
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
    background-color: transparent !important;
    border: none !important;
}

/* Adjust button layout */
.settings-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 1rem;
}

/* Style for processing logs */
.processing-logs {
    background-color: #111;
    border: 1px solid #444;
    border-radius: 5px;
    padding: 10px;
    font-family: monospace;
    font-size: 0.9rem;
    white-space: pre-wrap;
    max-height: 200px;
    overflow-y: auto;
    color: #26f6cb;
}

/* Custom styles for Streamlit components */
.stSelectbox > div[data-baseweb="select"] > div {
    background-color: #222;
    border-color: #444;
}

.stCheckbox > label > div[data-testid="stMarkdownContainer"] > p {
    color: #e0e0e0;
}

/* Remove highlight behind checkboxes */
.stCheckbox > label {
    background-color: transparent !important;
}

.stButton > button {
    background-color: #f4298a;
    color: #fff;
    border: none;
    transition: background-color 0.3s ease;
}

.stButton > button:hover {
    background-color: #d54d8e;
    color: #fff;
}

/* Chat message styling */
.stChatMessage {
    background-color: #111;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}

.stChatMessage [data-testid="stChatMessageContent"] {
    color: #e0e0e0;
}

/* Expander styling */
.streamlit-expanderHeader {
    background-color: #222;
    color: #f4298a;
    border: 1px solid #444;
}

.streamlit-expanderContent {
    background-color: #111;
    border: 1px solid #444;
    border-top: none;
}

/* File uploader styling */
.stFileUploader > div > div {
    background-color: #222;
    border-color: #444;
}

/* Improve readability of select dropdown */
.stSelectbox > div[data-baseweb="select"] > div > div {
    background-color: #222;
    color: #e0e0e0;
}

.stSelectbox > div[data-baseweb="select"] > div > div:hover {
    background-color: #333;
}

/* Chat container styling */
.chat-container {
    border: 1px solid #333;
    border-radius: 10px;
    padding: 1rem;
    background-color: #111;
}

/* Remove border from chat interface message */
.chat-container > div:first-child {
    border: none !important;
    background-color: transparent !important;
}

/* Remove border and background from info messages */
.stAlert.info {
    border: none !important;
    background-color: transparent !important;
}

/* Remove background from checkboxes */
.stCheckbox {
    background-color: transparent !important;
}

.stCheckbox > label {
    background-color: transparent !important;
}

.stCheckbox > label > div[data-testid="stMarkdownContainer"] {
    background-color: transparent !important;
}

</style>
""", unsafe_allow_html=True)

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.document_processor import process_documents, get_existing_documents, clear_vectorstore, get_embedding_function, remove_document
from app.model_handler import ModelHandler
from app.rag import retrieve_context
from app.utils import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

@st.cache_resource
def get_model_handler():
    model_handler = ModelHandler(config)
    # st.write(f"Available models: {model_handler.available_models}")  # Debug info
    return model_handler

def load_models():
    with st.spinner("Loading models... This may take a few minutes."):
        model_handler = get_model_handler()

        if not model_handler.available_models:
            st.warning("🤖 No AI models detected! Download GGUF model files and place them in the `models/` directory. 📖 [Setup guide](https://github.com/vincentkoc/airgapped-offfline-rag#model-setup)")
        else:
            # st.write(f"Debug: Available models: {model_handler.available_models}")  # More debug info
            alert = st.success(f"Models loaded successfully! Available models: {', '.join(model_handler.available_models)}")
            time.sleep(2)
            alert.empty()

def main():
    # Custom CSS for responsive layout
    st.markdown("""
    <style>
    .header-container {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
    }
    .logo-container {
        width: 60px;
        margin-right: 20px;
    }
    .title-container {
        flex-grow: 1;
    }
    .title-container h1 {
        margin: 0;
        padding: 0;
        font-size: 2.5rem;
        white-space: normal;
        word-wrap: break-word;
    }
    @media (max-width: 768px) {
        .title-container h1 {
            font-size: 1.8rem;
        }
        .logo-container {
            width: 40px;
            margin-right: 10px;
        }
    }
    .stColumns {
        gap: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create a container for the header
    header = st.container()
    with header:
        st.markdown("""
        <div class="header-container">
            <div class="logo-container">
                <img src="data:image/png;base64,{}" width="100%">
            </div>
            <div class="title-container">
                <h1>Airgapped Offline RAG</h1>
            </div>
        </div>
        """.format(get_base64_of_image("assets/airgapped_offline_rag_icon.png")), unsafe_allow_html=True)

    # Expanded blurb
    st.markdown("""
    An offline RAG (Retrieval-Augmented Generation) system for document analysis and Q&A.
    This tool allows you to process and query your documents locally, without relying on external APIs or internet connection.
    Perfect for handling sensitive information or working in air-gapped environments.
    """)

    # Initialize session state variables
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'chat_enabled' not in st.session_state:
        st.session_state.chat_enabled = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'use_rag' not in st.session_state:
        st.session_state.use_rag = True
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

    if not st.session_state.models_loaded:
        load_models()
        st.session_state.models_loaded = True

    # Use st.columns with equal width for settings and chat
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.markdown('<div class="settings-column">', unsafe_allow_html=True)
            settings_section()
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown('<div class="chat-column">', unsafe_allow_html=True)
            chat_interface()
            st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
        <div class="footer">
            © 2023-2025 Airgapped Offline Local Document RAG by <a href="https://x.com/vincent_koc" target="_blank">Vincent Koc</a> |
            <a href="https://github.com/vincentkoc/airgapped-offfline-rag" target="_blank">
                Open Source on GitHub <img src="https://github.com/fluidicon.png" class="github-icon">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

def settings_section():
    with st.container():
        st.subheader("Settings")

        uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type=['pdf'])

        existing_docs = get_existing_documents()
        if existing_docs:
            st.write("Existing documents:")
            with st.container():
                for doc in existing_docs:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"<p>- {doc}</p>", unsafe_allow_html=True)
                    with col2:
                        if st.button(f"Remove", key=f"remove_{doc}"):
                            if remove_document(doc):
                                st.success(f"Removed {doc}")
                                st.rerun()
                            else:
                                st.error(f"Failed to remove {doc}")
        else:
            st.write("No existing documents found.")

        model_handler = get_model_handler()
        if model_handler.available_models:
            model_choice = st.selectbox("Choose a model", model_handler.available_models)
            st.session_state.model_choice = model_choice
        else:
            return

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.session_state.debug_mode = st.checkbox("Debug")
        with col2:
            st.session_state.use_rag = st.checkbox("Use RAG", value=True)
        with col3:
            if st.button("Process"):
                if uploaded_files or existing_docs:
                    process_and_enable_chat(uploaded_files)
                elif st.session_state.use_rag:
                    st.warning("No documents found. Please upload documents to use RAG or disable RAG.")
                else:
                    st.success("Chat enabled without RAG.")
                    st.session_state.chat_enabled = True

        if 'processing_result' in st.session_state:
            st.markdown(st.session_state.processing_result, unsafe_allow_html=True)
            if st.session_state.debug_mode and 'processing_logs' in st.session_state:
                with st.expander("Processing Logs"):
                    st.markdown(f'<div class="processing-logs">{st.session_state.processing_logs}</div>', unsafe_allow_html=True)

def process_and_enable_chat(uploaded_files):
    with st.spinner("Processing documents..."):
        try:
            log_capture = io.StringIO()
            log_handler = logging.StreamHandler(log_capture)
            logger.addHandler(log_handler)

            result = process_documents(uploaded_files)
            num_chunks, file_info = result

            logger.removeHandler(log_handler)
            log_contents = log_capture.getvalue()

            if num_chunks > 0:
                st.session_state.processing_result = f'<div class="stAlert success">Processed {num_chunks} chunks from {len(file_info)} documents</div>'
                st.session_state.chat_enabled = True
            else:
                st.session_state.processing_result = '<div class="stAlert info fade-out">No new documents to process.</div>'

            if st.session_state.debug_mode:
                st.session_state.processing_logs = log_contents
            st.session_state.chat_enabled = True
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            st.session_state.processing_result = f'<div class="stAlert error">Error processing documents: {str(e)}</div>'
            st.session_state.chat_enabled = False

def chat_interface():
    st.subheader("Chat Interface")

    if st.session_state.chat_enabled or not st.session_state.use_rag:
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
                model_handler = ModelHandler(config)
                model_choice = st.session_state.model_choice

                if st.session_state.use_rag:
                    context = retrieve_context(prompt)
                    system_prompt = get_system_prompt()
                    full_prompt = f"{system_prompt}\n\nContext: {context}\n\nHuman: {prompt}\n\nAssistant:"
                else:
                    full_prompt = prompt

                if st.session_state.debug_mode:
                    with st.expander("LLM Prompt"):
                        st.code(full_prompt)

                try:
                    for response in model_handler.generate_stream(full_prompt, model_choice=model_choice):
                        full_response += response
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}")
                    st.error(f"Error generating response: {str(e)}")
                    full_response = "I apologize, but I encountered an error while generating the response."

            st.session_state.messages.append({"role": "assistant", "content": full_response})

    elif not st.session_state.use_rag:
        st.info("RAG is disabled. You can start chatting without document context.")
    else:
        st.info("Please process documents or disable RAG to start chatting.")

def get_system_prompt():
    return """You are a helpful AI assistant. Your task is to answer questions based solely on the provided context.
    If the context doesn't contain enough information to answer the question, say so.
    Do not use any external knowledge or make assumptions beyond what's given in the context.
    If asked about your capabilities or identity, refer only to being an AI assistant without mentioning specific models or companies."""

def handle_chat_input():
    if prompt := st.chat_input("What is your question?"):
        system_prompt = get_system_prompt()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            model_handler = ModelHandler(config)
            model_choice = st.session_state.model_choice
            message_placeholder = st.empty()
            full_response = ""

            if st.session_state.use_rag:
                context = retrieve_context(prompt)
                system_prompt = get_system_prompt()
                full_prompt = f"{system_prompt}\n\nContext: {context}\n\nHuman: {prompt}\n\nAssistant:"
            else:
                full_prompt = prompt

            if st.session_state.debug_mode:
                with st.expander("LLM Prompt"):
                    st.code(full_prompt)

            try:
                for response in model_handler.generate_stream(full_prompt, model_choice=model_choice):
                    full_response += response
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                st.error(f"Error generating response: {str(e)}")
                full_response = "I apologize, but I encountered an error while generating the response."

        st.session_state.messages.append({"role": "assistant", "content": full_response})

def get_rag_context(prompt):
    try:
        context = retrieve_context(prompt, top_k=config['top_k'])
        if st.session_state.debug_mode:
            logger.info(f"RAG Context: {context}")
            with st.expander("RAG Debug Information"):
                st.write("RAG Context:")
                st.code(context)
        return context
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        st.error(f"Error retrieving context: {str(e)}")
        return ""

def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

if __name__ == "__main__":
    main()
