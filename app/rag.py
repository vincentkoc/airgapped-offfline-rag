from .utils import load_config
from .document_processor import get_vectorstore
import streamlit as st
import logging

config = load_config()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retrieve_context(query, top_k=3):
    try:
        vectorstore = get_vectorstore()

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
