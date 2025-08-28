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
from pypdf import PdfReader
from .base import BaseDocumentHandler, DocumentChunk
from ..telemetry import track_document_processing

logger = logging.getLogger(__name__)

class PDFHandler(BaseDocumentHandler):
    """Handler for PDF documents using pypdf"""
    
    def get_supported_formats(self) -> List[str]:
        return ["pdf"]
    
    def get_supported_extensions(self) -> List[str]:
        return [".pdf"]
    
    def get_supported_mimetypes(self) -> List[str]:
        return ["application/pdf"]
    
    @track_document_processing("pdf")
    def extract_text(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO], 
                    **kwargs) -> str:
        """Extract raw text from PDF"""
        try:
            reader = PdfReader(file_path_or_buffer)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n\n--- Page {page_num + 1} ---\n\n"
                        text += page_text
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                    continue
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    @track_document_processing("pdf")
    def extract_structured_content(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO], 
                                 **kwargs) -> List[DocumentChunk]:
        """Extract structured content from PDF preserving page boundaries"""
        chunk_size = kwargs.get('chunk_size', 1000)
        chunk_overlap = kwargs.get('chunk_overlap', 100)
        
        try:
            reader = PdfReader(file_path_or_buffer)
            chunks = []
            base_metadata = self.get_metadata(file_path_or_buffer)
            
            # Add PDF-specific metadata
            base_metadata.update({
                "total_pages": len(reader.pages),
                "document_type": "pdf"
            })
            
            # Extract metadata from PDF if available
            if reader.metadata:
                pdf_metadata = {
                    "title": reader.metadata.get("/Title"),
                    "author": reader.metadata.get("/Author"),
                    "subject": reader.metadata.get("/Subject"),
                    "creator": reader.metadata.get("/Creator"),
                    "producer": reader.metadata.get("/Producer"),
                    "creation_date": reader.metadata.get("/CreationDate"),
                    "modification_date": reader.metadata.get("/ModDate")
                }
                # Filter out None values
                pdf_metadata = {k: v for k, v in pdf_metadata.items() if v is not None}
                base_metadata.update(pdf_metadata)
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    
                    if not page_text or not page_text.strip():
                        continue
                    
                    # Create page-level metadata
                    page_metadata = base_metadata.copy()
                    page_metadata.update({
                        "page_number": page_num + 1,
                        "source_page": page_num + 1
                    })
                    
                    # If page text is small enough, create single chunk
                    if len(page_text) <= chunk_size:
                        chunk = DocumentChunk(
                            content=page_text.strip(),
                            metadata=page_metadata,
                            page_number=page_num + 1,
                            chunk_id=f"page_{page_num + 1}",
                            chunk_type="text"
                        )
                        chunks.append(chunk)
                    else:
                        # Split page text into smaller chunks
                        text_chunks = self.chunk_text(
                            page_text, 
                            chunk_size=chunk_size, 
                            chunk_overlap=chunk_overlap,
                            preserve_paragraphs=True
                        )
                        
                        for chunk_idx, chunk_text in enumerate(text_chunks):
                            chunk_metadata = page_metadata.copy()
                            chunk_metadata["chunk_index"] = chunk_idx
                            
                            chunk = DocumentChunk(
                                content=chunk_text,
                                metadata=chunk_metadata,
                                page_number=page_num + 1,
                                chunk_id=f"page_{page_num + 1}_chunk_{chunk_idx}",
                                chunk_type="text"
                            )
                            chunks.append(chunk)
                
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {e}")
                    continue
            
            logger.info(f"Extracted {len(chunks)} chunks from PDF with {len(reader.pages)} pages")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting structured content from PDF: {e}")
            raise
    
    def get_metadata(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO]) -> Dict[str, Any]:
        """Extract enhanced metadata from PDF"""
        metadata = super().get_metadata(file_path_or_buffer)
        
        try:
            reader = PdfReader(file_path_or_buffer)
            
            # Basic PDF info
            metadata.update({
                "total_pages": len(reader.pages),
                "document_type": "pdf"
            })
            
            # PDF metadata
            if reader.metadata:
                pdf_metadata = {
                    "pdf_title": reader.metadata.get("/Title"),
                    "pdf_author": reader.metadata.get("/Author"),
                    "pdf_subject": reader.metadata.get("/Subject"),
                    "pdf_creator": reader.metadata.get("/Creator"),
                    "pdf_producer": reader.metadata.get("/Producer"),
                    "pdf_creation_date": reader.metadata.get("/CreationDate"),
                    "pdf_modification_date": reader.metadata.get("/ModDate")
                }
                # Filter out None values
                pdf_metadata = {k: v for k, v in pdf_metadata.items() if v is not None}
                metadata.update(pdf_metadata)
            
            # Try to detect if PDF is encrypted
            metadata["is_encrypted"] = reader.is_encrypted
            
        except Exception as e:
            logger.warning(f"Error extracting PDF metadata: {e}")
        
        return metadata