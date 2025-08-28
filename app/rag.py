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

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from typing import List, Dict, Any, Tuple
from .utils import load_config
from .document_processor import get_embedding_function, get_vectorstore
from .telemetry import telemetry, track_rag
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

@track_rag("context_retrieval")
def retrieve_context(query: str, top_k: int = 3) -> str:
    """Retrieve relevant context for a query with telemetry"""
    embeddings = get_embedding_function()
    if embeddings is None:
        logger.error("Failed to initialize embeddings.")
        telemetry.log_error(
            Exception("Failed to initialize embeddings"),
            {"operation": "embedding_initialization", "query": query}
        )
        return ""

    try:
        vectorstore = get_vectorstore()

        # Get collection info for telemetry
        try:
            collection_count = vectorstore._collection.count()
            logger.info(f"Number of documents in vectorstore: {collection_count}")
        except:
            collection_count = -1

        # Log retrieval start
        telemetry.log_user_interaction(
            "context_retrieval_start",
            {
                "query": query[:100],  # Truncate for privacy
                "query_length": len(query),
                "top_k": top_k,
                "collection_size": collection_count
            }
        )

        # Perform similarity search
        docs = vectorstore.similarity_search(query, k=top_k)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Extract metadata for telemetry
        doc_sources = [doc.metadata.get('source', 'unknown') for doc in docs]
        doc_types = [doc.metadata.get('document_type', 'unknown') for doc in docs]
        
        logger.info(f"Retrieved {len(docs)} documents for query: {query[:50]}...")
        logger.info(f"Context length: {len(context)} characters")

        # Log successful retrieval
        telemetry.log_user_interaction(
            "context_retrieval_success",
            {
                "query_length": len(query),
                "documents_retrieved": len(docs),
                "context_length": len(context),
                "document_sources": doc_sources,
                "document_types": doc_types,
                "top_k_requested": top_k
            }
        )

        return context
        
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        telemetry.log_error(e, {
            "operation": "context_retrieval",
            "query": query[:100],
            "top_k": top_k
        })
        st.error(f"Error retrieving context: {str(e)}")
        return ""

@track_rag("similarity_search_with_scores")
def retrieve_context_with_scores(query: str, top_k: int = 3, score_threshold: float = 0.0) -> Tuple[List[str], List[float]]:
    """Retrieve context with similarity scores"""
    embeddings = get_embedding_function()
    if embeddings is None:
        logger.error("Failed to initialize embeddings.")
        return [], []

    try:
        vectorstore = get_vectorstore()
        
        # Perform similarity search with scores
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)
        
        # Filter by score threshold
        filtered_docs = [(doc, score) for doc, score in docs_with_scores if score >= score_threshold]
        
        contexts = [doc.page_content for doc, score in filtered_docs]
        scores = [score for doc, score in filtered_docs]
        
        # Log retrieval with scores
        telemetry.log_user_interaction(
            "scored_retrieval",
            {
                "query_length": len(query),
                "documents_found": len(docs_with_scores),
                "documents_after_threshold": len(filtered_docs),
                "score_threshold": score_threshold,
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0
            }
        )
        
        return contexts, scores
        
    except Exception as e:
        logger.error(f"Error in scored retrieval: {str(e)}")
        telemetry.log_error(e, {"operation": "scored_retrieval", "query": query[:100]})
        return [], []

@track_rag("rag_pipeline")
def rag_query(query: str, model_choice: str = "Mistral", top_k: int = 3, 
              include_sources: bool = True) -> Dict[str, Any]:
    """Complete RAG pipeline with telemetry"""
    try:
        # Log RAG query start
        telemetry.log_user_interaction(
            "rag_query_start",
            {
                "query": query[:100],  # Truncated for privacy
                "query_length": len(query),
                "model_choice": model_choice,
                "top_k": top_k,
                "include_sources": include_sources
            }
        )
        
        # Retrieve context
        context = retrieve_context(query, top_k=top_k)
        
        if not context:
            logger.warning("No context retrieved for query")
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "context": "",
                "sources": [],
                "success": False,
                "error": "No relevant context found"
            }
        
        # Create RAG prompt
        rag_prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        # Log successful RAG completion
        result = {
            "context": context,
            "prompt": rag_prompt,
            "context_length": len(context),
            "success": True
        }
        
        if include_sources:
            # Extract sources from context (this would need more sophisticated implementation)
            result["sources"] = ["Retrieved from document"]
        
        telemetry.log_user_interaction(
            "rag_query_complete",
            {
                "query_length": len(query),
                "context_length": len(context),
                "prompt_length": len(rag_prompt),
                "model_choice": model_choice,
                "success": True
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        telemetry.log_error(e, {
            "operation": "rag_pipeline",
            "query": query[:100],
            "model_choice": model_choice
        })
        return {
            "answer": "An error occurred while processing your question.",
            "context": "",
            "sources": [],
            "success": False,
            "error": str(e)
        }
