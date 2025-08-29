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

import logging
from typing import List, Dict, Any, Union, BinaryIO
import pathlib
from .base import BaseDocumentHandler, DocumentChunk
from ..telemetry import track_document_processing

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

logger = logging.getLogger(__name__)

class DoclingHandler(BaseDocumentHandler):
    """Universal document handler using Docling for advanced document processing"""
    
    def __init__(self):
        self.converter = None
        if DOCLING_AVAILABLE:
            try:
                self.converter = DocumentConverter()
                logger.info("Docling DocumentConverter initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Docling: {e}")
        else:
            logger.warning("Docling not available. Advanced document processing disabled.")
        super().__init__()
    
    def get_supported_formats(self) -> List[str]:
        if not DOCLING_AVAILABLE or not self.converter:
            return []
        
        return [
            "docx", "doc", "xlsx", "xls", "pptx", "ppt", 
            "pdf", "html", "md", "txt", "rtf", "odt", 
            "ods", "odp", "epub", "latex", "tex"
        ]
    
    def get_supported_extensions(self) -> List[str]:
        if not DOCLING_AVAILABLE or not self.converter:
            return []
            
        return [
            ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
            ".pdf", ".html", ".htm", ".md", ".markdown", ".txt", 
            ".rtf", ".odt", ".ods", ".odp", ".epub", 
            ".latex", ".tex"
        ]
    
    def get_supported_mimetypes(self) -> List[str]:
        if not DOCLING_AVAILABLE or not self.converter:
            return []
            
        return [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.ms-powerpoint",
            "application/pdf",
            "text/html",
            "text/markdown",
            "text/plain",
            "application/rtf",
            "application/vnd.oasis.opendocument.text",
            "application/vnd.oasis.opendocument.spreadsheet",
            "application/vnd.oasis.opendocument.presentation",
            "application/epub+zip",
            "application/x-latex",
            "text/x-tex"
        ]
    
    def can_handle(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO], 
                  mimetype: str = None) -> bool:
        """Check if Docling can handle this file"""
        if not DOCLING_AVAILABLE or not self.converter:
            return False
        return super().can_handle(file_path_or_buffer, mimetype)
    
    @track_document_processing("docling")
    def extract_text(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO], 
                    **kwargs) -> str:
        """Extract raw text using Docling"""
        if not self.converter:
            raise RuntimeError("Docling not available")
        
        try:
            # Convert file path to string for Docling
            if isinstance(file_path_or_buffer, pathlib.Path):
                file_path = str(file_path_or_buffer)
            elif isinstance(file_path_or_buffer, str):
                file_path = file_path_or_buffer
            else:
                raise ValueError("Docling requires file path, not file buffer")
            
            # Convert document
            result = self.converter.convert(file_path)
            
            # Extract text content
            if hasattr(result, 'document') and result.document:
                text = result.document.export_to_text()
                return text
            else:
                logger.warning("No document content found in Docling result")
                return ""
                
        except Exception as e:
            logger.error(f"Error extracting text with Docling: {e}")
            raise
    
    @track_document_processing("docling")
    def extract_structured_content(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO], 
                                 **kwargs) -> List[DocumentChunk]:
        """Extract structured content using Docling's advanced parsing"""
        if not self.converter:
            raise RuntimeError("Docling not available")
        
        chunk_size = kwargs.get('chunk_size', 1000)
        chunk_overlap = kwargs.get('chunk_overlap', 100)
        
        try:
            # Convert file path to string for Docling
            if isinstance(file_path_or_buffer, pathlib.Path):
                file_path = str(file_path_or_buffer)
            elif isinstance(file_path_or_buffer, str):
                file_path = file_path_or_buffer
            else:
                raise ValueError("Docling requires file path, not file buffer")
            
            # Convert document
            result = self.converter.convert(file_path)
            
            if not (hasattr(result, 'document') and result.document):
                logger.warning("No document content found in Docling result")
                return []
            
            document = result.document
            chunks = []
            base_metadata = self.get_metadata(file_path_or_buffer)
            
            # Add Docling-specific metadata
            base_metadata.update({
                "document_type": "docling_processed",
                "processor": "docling"
            })
            
            # Try to extract structured elements from document texts and other content
            try:
                # Process text items directly from document.texts
                if hasattr(document, 'texts') and document.texts:
                    for element_idx, text_item in enumerate(document.texts):
                        element_text = ""
                        element_type = "text"
                        element_metadata = base_metadata.copy()
                        
                        # Extract text content using .text attribute
                        if hasattr(text_item, 'text') and text_item.text:
                            element_text = text_item.text.strip()
                        else:
                            continue  # Skip items without text content
                        
                        # Skip empty or near-empty text
                        if len(element_text) < 5:
                            continue
                        
                        # Determine element type from label
                        if hasattr(text_item, 'label') and text_item.label:
                            element_type = str(text_item.label).lower().replace('docitemlabel.', '')
                        
                        # Add element-specific metadata
                        element_metadata.update({
                            "element_index": element_idx,
                            "element_type": element_type,
                            "content_layer": str(text_item.content_layer) if hasattr(text_item, 'content_layer') else None
                        })
                        
                        # Handle different element types
                        if element_type in ["table", "figure", "image"]:
                            # Special handling for tables and figures
                            chunk = DocumentChunk(
                                content=element_text,
                                metadata=element_metadata,
                                chunk_id=f"text_{element_idx}_{element_type}",
                                chunk_type=element_type
                            )
                            chunks.append(chunk)
                        else:
                            # Regular text content - chunk if needed
                            if len(element_text) <= chunk_size:
                                chunk = DocumentChunk(
                                    content=element_text,
                                    metadata=element_metadata,
                                    chunk_id=f"text_{element_idx}",
                                    chunk_type=element_type
                                )
                                chunks.append(chunk)
                            else:
                                # Split long content
                                text_chunks = self.chunk_text(
                                    element_text,
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap,
                                    preserve_paragraphs=True
                                )
                                
                                for chunk_idx, chunk_text in enumerate(text_chunks):
                                    chunk_metadata = element_metadata.copy()
                                    chunk_metadata["chunk_index"] = chunk_idx
                                    
                                    chunk = DocumentChunk(
                                        content=chunk_text,
                                        metadata=chunk_metadata,
                                        chunk_id=f"text_{element_idx}_chunk_{chunk_idx}",
                                        chunk_type=element_type
                                    )
                                    chunks.append(chunk)
                
                else:
                    # Fallback: extract as plain text and chunk
                    full_text = document.export_to_text()
                    text_chunks = self.chunk_text(
                        full_text,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        preserve_paragraphs=True
                    )
                    
                    for chunk_idx, chunk_text in enumerate(text_chunks):
                        chunk_metadata = base_metadata.copy()
                        chunk_metadata["chunk_index"] = chunk_idx
                        
                        chunk = DocumentChunk(
                            content=chunk_text,
                            metadata=chunk_metadata,
                            chunk_id=f"chunk_{chunk_idx}",
                            chunk_type="text"
                        )
                        chunks.append(chunk)
            
            except Exception as e:
                logger.warning(f"Error processing structured content, falling back to text extraction: {e}")
                # Fallback to plain text extraction
                full_text = document.export_to_text()
                text_chunks = self.chunk_text(
                    full_text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    preserve_paragraphs=True
                )
                
                for chunk_idx, chunk_text in enumerate(text_chunks):
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata["chunk_index"] = chunk_idx
                    
                    chunk = DocumentChunk(
                        content=chunk_text,
                        metadata=chunk_metadata,
                        chunk_id=f"fallback_chunk_{chunk_idx}",
                        chunk_type="text"
                    )
                    chunks.append(chunk)
            
            logger.info(f"Extracted {len(chunks)} chunks using Docling")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting structured content with Docling: {e}")
            raise
    
    def get_metadata(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO]) -> Dict[str, Any]:
        """Extract enhanced metadata using Docling"""
        metadata = super().get_metadata(file_path_or_buffer)
        
        if not self.converter:
            return metadata
        
        try:
            # Convert file path to string for Docling
            if isinstance(file_path_or_buffer, pathlib.Path):
                file_path = str(file_path_or_buffer)
            elif isinstance(file_path_or_buffer, str):
                file_path = file_path_or_buffer
            else:
                return metadata
            
            # Convert document to extract metadata
            result = self.converter.convert(file_path)
            
            if hasattr(result, 'document') and result.document:
                document = result.document
                
                # Add document-level metadata
                metadata.update({
                    "processor": "docling",
                    "has_structured_content": True
                })
                
                # Extract document properties if available
                if hasattr(document, 'metadata') and document.metadata:
                    doc_metadata = document.metadata
                    if hasattr(doc_metadata, 'title') and doc_metadata.title:
                        metadata["document_title"] = doc_metadata.title
                    if hasattr(doc_metadata, 'authors') and doc_metadata.authors:
                        metadata["document_authors"] = doc_metadata.authors
                    if hasattr(doc_metadata, 'creation_date') and doc_metadata.creation_date:
                        metadata["document_creation_date"] = doc_metadata.creation_date
                
                # Count different element types if available
                if hasattr(document, 'body') and document.body:
                    element_types = {}
                    for element in document.body:
                        element_type = "text"
                        if hasattr(element, 'element_type'):
                            element_type = element.element_type
                        elif hasattr(element, 'type'):
                            element_type = element.type
                        
                        element_types[element_type] = element_types.get(element_type, 0) + 1
                    
                    metadata["element_counts"] = element_types
                    metadata["total_elements"] = len(document.body)
                
        except Exception as e:
            logger.warning(f"Error extracting Docling metadata: {e}")
        
        return metadata