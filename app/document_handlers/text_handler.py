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
import chardet
from .base import BaseDocumentHandler, DocumentChunk
from ..telemetry import track_document_processing

logger = logging.getLogger(__name__)

class TextHandler(BaseDocumentHandler):
    """Handler for plain text documents"""
    
    def get_supported_formats(self) -> List[str]:
        return ["text", "txt"]
    
    def get_supported_extensions(self) -> List[str]:
        return [".txt", ".text", ".log", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml"]
    
    def get_supported_mimetypes(self) -> List[str]:
        return [
            "text/plain", 
            "text/csv", 
            "text/tab-separated-values",
            "application/json",
            "application/xml",
            "text/xml",
            "application/x-yaml",
            "text/yaml"
        ]
    
    def _detect_encoding(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO]) -> str:
        """Detect file encoding"""
        try:
            if isinstance(file_path_or_buffer, (str, pathlib.Path)):
                with open(file_path_or_buffer, 'rb') as f:
                    raw_data = f.read(10000)  # Read first 10KB for detection
            else:
                current_pos = file_path_or_buffer.tell()
                raw_data = file_path_or_buffer.read(10000)
                file_path_or_buffer.seek(current_pos)  # Reset position
            
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            
            # Fallback to utf-8 if confidence is too low
            if confidence < 0.7:
                encoding = 'utf-8'
                
            logger.debug(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            return encoding
            
        except Exception as e:
            logger.warning(f"Error detecting encoding, defaulting to utf-8: {e}")
            return 'utf-8'
    
    @track_document_processing("text")
    def extract_text(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO], 
                    **kwargs) -> str:
        """Extract text from file"""
        encoding = kwargs.get('encoding', None)
        
        try:
            if isinstance(file_path_or_buffer, (str, pathlib.Path)):
                if not encoding:
                    encoding = self._detect_encoding(file_path_or_buffer)
                
                with open(file_path_or_buffer, 'r', encoding=encoding, errors='replace') as f:
                    text = f.read()
            else:
                # File-like object
                if hasattr(file_path_or_buffer, 'read'):
                    if hasattr(file_path_or_buffer, 'mode') and 'b' in file_path_or_buffer.mode:
                        # Binary mode - need to decode
                        content = file_path_or_buffer.read()
                        if not encoding:
                            # Try to detect encoding from content
                            result = chardet.detect(content[:10000])
                            encoding = result.get('encoding', 'utf-8')
                        text = content.decode(encoding, errors='replace')
                    else:
                        # Text mode
                        text = file_path_or_buffer.read()
                else:
                    raise ValueError("Invalid file buffer")
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            raise
    
    @track_document_processing("text")
    def extract_structured_content(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO], 
                                 **kwargs) -> List[DocumentChunk]:
        """Extract structured content from text file"""
        chunk_size = kwargs.get('chunk_size', 1000)
        chunk_overlap = kwargs.get('chunk_overlap', 100)
        preserve_structure = kwargs.get('preserve_structure', True)
        
        try:
            text = self.extract_text(file_path_or_buffer, **kwargs)
            
            if not text:
                return []
            
            chunks = []
            base_metadata = self.get_metadata(file_path_or_buffer)
            base_metadata.update({
                "document_type": "text",
                "total_characters": len(text)
            })
            
            # Detect file type for special handling
            file_type = self._detect_text_type(file_path_or_buffer, text)
            base_metadata["text_type"] = file_type
            
            if file_type == "structured" and preserve_structure:
                # Try to preserve structure for structured text
                chunks = self._extract_structured_chunks(text, base_metadata, chunk_size, chunk_overlap)
            else:
                # Regular text chunking
                text_chunks = self.chunk_text(
                    text,
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
                        chunk_id=f"text_chunk_{chunk_idx}",
                        chunk_type="text"
                    )
                    chunks.append(chunk)
            
            logger.info(f"Extracted {len(chunks)} chunks from text file")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting structured content from text: {e}")
            raise
    
    def _detect_text_type(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO], text: str) -> str:
        """Detect the type of text content"""
        # Check by file extension first
        if isinstance(file_path_or_buffer, (str, pathlib.Path)):
            ext = pathlib.Path(file_path_or_buffer).suffix.lower()
            if ext in ['.json']:
                return 'json'
            elif ext in ['.xml']:
                return 'xml'
            elif ext in ['.csv']:
                return 'csv'
            elif ext in ['.yaml', '.yml']:
                return 'yaml'
            elif ext in ['.log']:
                return 'log'
        
        # Try to detect from content
        text_sample = text[:1000].strip()
        
        if text_sample.startswith('{') or text_sample.startswith('['):
            return 'json'
        elif text_sample.startswith('<?xml') or text_sample.startswith('<'):
            return 'xml'
        elif ',' in text_sample and '\n' in text_sample:
            # Might be CSV
            lines = text_sample.split('\n')[:5]
            if all(',' in line for line in lines if line.strip()):
                return 'csv'
        elif any(text_sample.startswith(prefix) for prefix in ['---\n', '- ', '  - ']):
            return 'yaml'
        elif any(keyword in text_sample.lower() for keyword in ['error:', 'warning:', 'info:', 'debug:']):
            return 'log'
        
        # Check for structured patterns
        lines = text_sample.split('\n')
        if len(lines) > 5:
            # Check for consistent structure
            non_empty_lines = [line for line in lines if line.strip()]
            if len(non_empty_lines) > 3:
                # Look for patterns like bullets, numbers, or consistent indentation
                if any(line.strip().startswith(('- ', '* ', '+ ')) for line in non_empty_lines):
                    return 'structured'
                elif any(line.strip()[0].isdigit() and '.' in line[:10] for line in non_empty_lines):
                    return 'structured'
        
        return 'plain'
    
    def _extract_structured_chunks(self, text: str, base_metadata: Dict[str, Any], 
                                 chunk_size: int, chunk_overlap: int) -> List[DocumentChunk]:
        """Extract chunks while preserving text structure"""
        chunks = []
        
        # Split by double newlines first (paragraphs)
        sections = text.split('\n\n')
        current_chunk = ""
        chunk_idx = 0
        
        for section_idx, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            
            # If adding this section would exceed chunk size
            if current_chunk and len(current_chunk) + len(section) + 2 > chunk_size:
                # Save current chunk
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": chunk_idx,
                    "section_start": section_idx - (current_chunk.count('\n\n') + 1),
                    "section_end": section_idx - 1
                })
                
                chunk = DocumentChunk(
                    content=current_chunk,
                    metadata=chunk_metadata,
                    chunk_id=f"structured_chunk_{chunk_idx}",
                    chunk_type="structured_text"
                )
                chunks.append(chunk)
                chunk_idx += 1
                
                # Start new chunk with overlap if needed
                if chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-chunk_overlap:]
                    current_chunk = overlap_text + "\n\n" + section
                else:
                    current_chunk = section
            else:
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_index": chunk_idx,
                "section_start": len(sections) - current_chunk.count('\n\n') - 1,
                "section_end": len(sections) - 1
            })
            
            chunk = DocumentChunk(
                content=current_chunk,
                metadata=chunk_metadata,
                chunk_id=f"structured_chunk_{chunk_idx}",
                chunk_type="structured_text"
            )
            chunks.append(chunk)
        
        return chunks