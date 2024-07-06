import streamlit as st
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.document_processor import process_documents
from app.model_handler import ModelHandler
from app.rag import retrieve_context
from app.utils import load_config

# Load configuration
config = load_config()

# Initialize ModelHandler
model_handler = ModelHandler(config)

st.title("Document QnA System")

# File uploader
uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type=['pdf'])
if uploaded_files:
    with st.spinner("Processing documents..."):
        process_documents(uploaded_files)
    st.success("Documents processed successfully!")

# Model selection
model_choice = st.selectbox("Choose a model", ["Llama 3", "Mistral"], index=0 if config['default_model'] == 'llama' else 1)

# Query input
query = st.text_input("Enter your question")

if st.button("Generate Answer"):
    if query:
        with st.spinner("Generating answer..."):
            context = retrieve_context(query, top_k=config['top_k'])
            prompt = f"Context: {context}\n\nHuman: {query}\n\nAssistant:"
            response = model_handler.generate(prompt, "llama" if model_choice == "Llama 3" else "mistral")
        st.write("Answer:", response)
    else:
        st.warning("Please enter a question.")

st.sidebar.title("About")
st.sidebar.info("This is a RAG-based Document QnA system. Upload PDF documents and ask questions about their content.")
