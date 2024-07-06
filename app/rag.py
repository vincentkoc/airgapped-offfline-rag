from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from .utils import load_config
from .document_processor import initialize_chroma
import streamlit as st
import logging

config = load_config()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_embedding_function():
    try:
        return FastEmbedEmbeddings(
            model_name=config['embedding_model'],
            max_length=512,
            doc_embed_type="passage"
        )
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        st.error(f"Error loading embedding model: {str(e)}")
        return None

def retrieve_context(query, top_k=3):
    embeddings = get_embedding_function()
    if embeddings is None:
        logger.error("Failed to initialize embeddings.")
        return ""

    try:
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )

        # Log the number of documents in the vectorstore
        logger.info(f"Number of documents in vectorstore: {vectorstore._collection.count()}")

        docs = vectorstore.similarity_search(query, k=top_k)
        context = "\n".join([doc.page_content for doc in docs])

        logger.info(f"Retrieved {len(docs)} documents for query: {query}")
        logger.info(f"Context: {context[:500]}...")  # Log first 500 characters of context

        return context
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        st.error(f"Error retrieving context: {str(e)}")
        return ""
