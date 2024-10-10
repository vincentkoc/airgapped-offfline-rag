import streamlit as st
import sys
import os
import logging
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set page config at the very beginning
st.set_page_config(
    layout="wide",
    page_title="Offline RAG",
    page_icon="assets/offline_rag_icon.png",
    initial_sidebar_state="collapsed",
)

# Custom CSS to inject into the Streamlit app
st.markdown("""
<style>
/* Adjust title and header */
.main > div:first-child h1 {
    font-size: 2.2rem;
    margin-bottom: 0.5rem;
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
    border: 1px solid #e0e0e0;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 1rem;
}

/* Footer styling */
.footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #e0e0e0;
    color: #666;
    font-size: 0.8rem;
}

/* GitHub icon */
.github-icon {
    height: 20px;
    vertical-align: middle;
    margin-left: 5px;
}

/* Adjust spacing for settings and chat interface */
.stColumn > div {
    padding: 1rem;
    background-color: #f9f9f9;
    border-radius: 10px;
    margin-bottom: 1rem;
}

/* Adjust font sizes */
body {
    font-size: 16px;
}

.stTextInput > div > div > input {
    font-size: 16px;
}

.stSelectbox > div > div > div {
    font-size: 16px;
}

.stCheckbox > label > div {
    font-size: 16px;
}

/* Remove box above filenames */
.css-1kyxreq {
    display: none;
}

/* Adjust sidebar width when chat is active */
@media (min-width: 768px) {
    .main .block-container {
        max-width: 80%;
        padding-left: 5rem;
        padding-right: 5rem;
    }
}

/* Reduce file name font size */
.existing-docs p {
    font-size: 0.9rem;
}

/* Right align Process Documents button */
.stButton > button:last-child {
    float: right;
}

/* Add vertical line between settings and chat */
.main .block-container > div > div > div:nth-child(2) {
    border-left: 1px solid #e0e0e0;
    padding-left: 2rem;
}

/* Styling for info and success messages */
.stAlert {
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
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
    background-color: #f0f0f0;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 10px;
    font-family: monospace;
    font-size: 0.9rem;
    white-space: pre-wrap;
    max-height: 200px;
    overflow-y: auto;
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
            st.error("No models are available. Please check your configuration and model files.")
        else:
            # st.write(f"Debug: Available models: {model_handler.available_models}")  # More debug info
            alert = st.success(f"Models loaded successfully! Available models: {', '.join(model_handler.available_models)}")
            time.sleep(2)
            alert.empty()

def main():
    st.title("Lightweight Offline Document RAG")
    st.write("An offline RAG system for document analysis and Q&A.")

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

    col1, col2 = st.columns([1, 1])

    with col1:
        settings_section()

    with col2:
        chat_interface()

    # Footer
    st.markdown(
        """
        <div class="footer">
            © 2023 Lightweight Offline Document RAG |
            <a href="https://github.com/yourusername/your-repo" target="_blank">
                GitHub <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" class="github-icon">
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
                                st.experimental_rerun()
                            else:
                                st.error(f"Failed to remove {doc}")
        else:
            st.write("No existing documents found.")

        model_handler = get_model_handler()
        if model_handler.available_models:
            model_choice = st.selectbox("Choose a model", model_handler.available_models)
            st.session_state.model_choice = model_choice
        else:
            st.error("No models available. Please check your configuration and model files.")
            return

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.session_state.debug_mode = st.checkbox("Debug Mode")
        with col2:
            st.session_state.use_rag = st.checkbox("Use RAG", value=True)
        with col3:
            if st.button("Process Documents"):
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

            num_chunks = process_documents(uploaded_files)

            logger.removeHandler(log_handler)
            log_contents = log_capture.getvalue()

            if num_chunks > 0:
                st.session_state.processing_result = f'<div class="stAlert success">Processed {num_chunks} chunks from {len(uploaded_files)} documents</div>'
                st.session_state.chat_enabled = True
            else:
                st.session_state.processing_result = '<div class="stAlert info">No new documents to process.</div>'

            if st.session_state.debug_mode:
                st.session_state.processing_logs = log_contents
            st.session_state.chat_enabled = True
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            st.session_state.processing_result = f'<div class="stAlert error">Error processing documents: {str(e)}</div>'
            st.session_state.chat_enabled = False

def chat_interface():
    st.subheader("Chat Interface")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if st.session_state.chat_enabled or not st.session_state.use_rag:
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

if __name__ == "__main__":
    main()
