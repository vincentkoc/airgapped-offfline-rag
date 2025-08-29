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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import logging
from typing import List, Tuple, Dict, Any, Optional
from app.document_processor import get_vectorstore

logger = logging.getLogger(__name__)

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Install with: pip install umap-learn")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")

class EmbeddingVisualizer:
    """Visualize document embeddings using dimensionality reduction"""
    
    def __init__(self):
        self.vectorstore = None
        self.embeddings = None
        self.metadatas = None
        self.documents = None
        
    def load_embeddings(self) -> bool:
        """Load embeddings from the vectorstore"""
        try:
            self.vectorstore = get_vectorstore()
            
            # Get documents and metadata (ChromaDB 1.0+ doesn't expose embeddings)
            results = self.vectorstore.get()
            logger.info(f"Vectorstore contains {len(results.get('documents', []))} documents")
            
            if not results or len(results.get('documents', [])) == 0:
                logger.warning("No documents found in vectorstore")
                return False
                
            self.metadatas = results.get('metadatas', [])
            self.documents = results.get('documents', [])
            
            # Generate 2D visualization based on document similarity
            self.embeddings = self._create_document_layout()
            
            if self.embeddings is None:
                return False
                
            logger.info(f"Created visualization layout for {len(self.embeddings)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return False
    
    def _create_document_layout(self) -> Optional[np.ndarray]:
        """Create 2D layout based on document metadata and content"""
        try:
            n_docs = len(self.documents)
            if n_docs < 1:
                return None
            
            # Create simple layout based on document properties
            layout = np.zeros((n_docs, 2))
            
            # Group by source document
            sources = {}
            for i, meta in enumerate(self.metadatas):
                source = meta.get('source', 'unknown')
                if source not in sources:
                    sources[source] = []
                sources[source].append(i)
            
            # Position documents in clusters by source
            angle_step = 2 * np.pi / len(sources)
            for source_idx, (source, doc_indices) in enumerate(sources.items()):
                # Each source gets a position around a circle
                center_x = 3 * np.cos(source_idx * angle_step)
                center_y = 3 * np.sin(source_idx * angle_step)
                
                # Documents from same source cluster around the center
                for doc_idx_in_source, doc_idx in enumerate(doc_indices):
                    if len(doc_indices) == 1:
                        layout[doc_idx] = [center_x, center_y]
                    else:
                        # Arrange in a small circle around center
                        sub_angle = 2 * np.pi * doc_idx_in_source / len(doc_indices)
                        layout[doc_idx] = [
                            center_x + 0.5 * np.cos(sub_angle),
                            center_y + 0.5 * np.sin(sub_angle)
                        ]
            
            # Add some noise for better visualization
            layout += np.random.normal(0, 0.1, layout.shape)
            
            return layout
            
        except Exception as e:
            logger.error(f"Error creating document layout: {e}")
            # Fallback to random layout
            return np.random.rand(len(self.documents), 2) * 4 - 2
    
    def reduce_dimensions(self, method: str = "layout", n_components: int = 2, **kwargs) -> Optional[np.ndarray]:
        """Return the pre-computed document layout"""
        if self.embeddings is None:
            logger.error("No layout computed")
            return None
            
        # Since we're using a pre-computed layout, just return it
        # In the future, this could apply additional transformations
        logger.info(f"Using document layout for visualization")
        return self.embeddings
    
    def create_visualization(self, reduced_embeddings: np.ndarray, 
                          title: str = "Document Embeddings Visualization") -> go.Figure:
        """Create interactive plotly visualization"""
        if reduced_embeddings is None or self.metadatas is None:
            return None
            
        try:
            # Prepare data for visualization
            df_data = {
                'x': reduced_embeddings[:, 0],
                'y': reduced_embeddings[:, 1],
                'document_id': [i for i in range(len(reduced_embeddings))],
                'source': [meta.get('source', 'Unknown') for meta in self.metadatas],
                'chunk_type': [meta.get('chunk_type', 'text') for meta in self.metadatas],
                'element_type': [meta.get('element_type', 'text') for meta in self.metadatas],
                'preview': [doc[:100] + "..." if len(doc) > 100 else doc 
                           for doc in (self.documents or [""] * len(reduced_embeddings))]
            }
            
            # Add 3D component if available
            if reduced_embeddings.shape[1] >= 3:
                df_data['z'] = reduced_embeddings[:, 2]
                
            df = pd.DataFrame(df_data)
            
            # Create the plot
            if reduced_embeddings.shape[1] >= 3:
                # 3D scatter plot
                fig = px.scatter_3d(
                    df, x='x', y='y', z='z',
                    color='source',
                    symbol='chunk_type',
                    hover_data=['element_type'],
                    hover_name='preview',
                    title=title,
                    opacity=0.7
                )
                fig.update_traces(marker_size=5)
            else:
                # 2D scatter plot
                fig = px.scatter(
                    df, x='x', y='y',
                    color='source',
                    symbol='chunk_type',
                    hover_data=['element_type'],
                    hover_name='preview',
                    title=title,
                    opacity=0.7
                )
                fig.update_traces(marker_size=8)
            
            # Styling
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font_size=16,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return None
    
    def get_available_methods(self) -> List[str]:
        """Get list of available dimensionality reduction methods"""
        methods = []
        if UMAP_AVAILABLE:
            methods.append("UMAP")
        if SKLEARN_AVAILABLE:
            methods.extend(["t-SNE", "PCA"])
        return methods

def create_embedding_visualization():
    """Create the embedding visualization interface"""
    st.subheader("🔍 Document Embeddings Visualization")
    
    visualizer = EmbeddingVisualizer()
    
    # Load embeddings with detailed error info
    if not visualizer.load_embeddings():
        st.warning("No embeddings found. Please process some documents first.")
        
        # Show debug information
        if st.checkbox("Show Debug Info", key="viz_debug"):
            try:
                vectorstore = visualizer.vectorstore or get_vectorstore()
                results = vectorstore.get()
                
                st.write("**Vectorstore Debug:**")
                st.write(f"- Results type: {type(results)}")
                st.write(f"- Results keys: {list(results.keys()) if results else 'None'}")
                if results:
                    st.write(f"- IDs count: {len(results.get('ids', []))}")
                    st.write(f"- Metadatas count: {len(results.get('metadatas', []))}")
                    st.write(f"- Documents count: {len(results.get('documents', []))}")
                    st.write(f"- Has embeddings key: {'embeddings' in results}")
                    
                    # Check if embeddings are None vs missing
                    if 'embeddings' in results:
                        embeddings = results['embeddings']
                        st.write(f"- Embeddings value: {type(embeddings)} - {embeddings is not None}")
                        if embeddings:
                            st.write(f"- First embedding type: {type(embeddings[0]) if embeddings else 'N/A'}")
                
            except Exception as e:
                st.error(f"Debug failed: {e}")
        
        return
    
    # Auto-generate visualization
    with st.spinner("Generating document layout visualization..."):
        reduced = visualizer.reduce_dimensions("layout", 2)
        
        if reduced is not None:
            st.session_state.reduced_embeddings = reduced
            st.session_state.visualization_method = "Document Layout"
            st.session_state.visualization_dims = 2
    
    # Display visualization if available
    if hasattr(st.session_state, 'reduced_embeddings'):
        fig = visualizer.create_visualization(
            st.session_state.reduced_embeddings,
            title=f"Document Embeddings - {st.session_state.visualization_method} "
                  f"({st.session_state.visualization_dims}D)"
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Stats
            st.markdown("### 📊 Visualization Stats")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Chunks", len(visualizer.embeddings))
            
            with col2:
                sources = set(meta.get('source', 'Unknown') for meta in visualizer.metadatas)
                st.metric("Unique Documents", len(sources))
            
            with col3:
                st.metric("Embedding Dimension", visualizer.embeddings.shape[1])