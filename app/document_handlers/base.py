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

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, BinaryIO, Union
from dataclasses import dataclass
import mimetypes
import pathlib

@dataclass
class DocumentChunk:
    """Represents a chunk of processed document content"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    chunk_type: str = "text"  # text, table, image, code, etc.

class BaseDocumentHandler(ABC):
    """Base class for all document handlers"""
    
    def __init__(self):
        self.supported_formats = self.get_supported_formats()
        self.supported_extensions = self.get_supported_extensions()
        self.supported_mimetypes = self.get_supported_mimetypes()
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported format names"""
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions (with dots)"""
        pass
    
    @abstractmethod
    def get_supported_mimetypes(self) -> List[str]:
        """Return list of supported MIME types"""
        pass
    
    @abstractmethod
    def extract_text(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO], 
                    **kwargs) -> str:
        """Extract raw text from document"""
        pass
    
    @abstractmethod
    def extract_structured_content(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO], 
                                 **kwargs) -> List[DocumentChunk]:
        """Extract structured content preserving document hierarchy"""
        pass
    
    def can_handle(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO], 
                  mimetype: Optional[str] = None) -> bool:
        """Check if this handler can process the given file"""
        if isinstance(file_path_or_buffer, (str, pathlib.Path)):
            file_path = pathlib.Path(file_path_or_buffer)
            extension = file_path.suffix.lower()
            
            # Check by extension first
            if extension in self.supported_extensions:
                return True
            
            # Check by mimetype if provided
            if mimetype and mimetype in self.supported_mimetypes:
                return True
            
            # Try to guess mimetype from file
            guessed_mimetype, _ = mimetypes.guess_type(str(file_path))
            if guessed_mimetype and guessed_mimetype in self.supported_mimetypes:
                return True
        
        elif hasattr(file_path_or_buffer, 'name'):
            # File-like object with name attribute
            return self.can_handle(file_path_or_buffer.name, mimetype)
        
        elif mimetype:
            # Only mimetype available
            return mimetype in self.supported_mimetypes
        
        return False
    
    def validate_file(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO]) -> bool:
        """Validate that the file can be processed"""
        try:
            if isinstance(file_path_or_buffer, (str, pathlib.Path)):
                file_path = pathlib.Path(file_path_or_buffer)
                if not file_path.exists():
                    return False
                if file_path.stat().st_size == 0:
                    return False
            return True
        except Exception:
            return False
    
    def get_metadata(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO]) -> Dict[str, Any]:
        """Extract basic metadata from document"""
        metadata = {}
        
        if isinstance(file_path_or_buffer, (str, pathlib.Path)):
            file_path = pathlib.Path(file_path_or_buffer)
            stat = file_path.stat()
            
            metadata.update({
                "filename": file_path.name,
                "file_size": stat.st_size,
                "file_extension": file_path.suffix.lower(),
                "created_time": stat.st_ctime,
                "modified_time": stat.st_mtime
            })
            
            # Try to guess mimetype
            mimetype, encoding = mimetypes.guess_type(str(file_path))
            if mimetype:
                metadata["mimetype"] = mimetype
            if encoding:
                metadata["encoding"] = encoding
        
        elif hasattr(file_path_or_buffer, 'name'):
            metadata["filename"] = file_path_or_buffer.name
        
        return metadata
    
    def chunk_text(self, text: str, chunk_size: int = 1000, 
                  chunk_overlap: int = 100, preserve_paragraphs: bool = True) -> List[str]:
        """Split text into chunks with optional paragraph preservation"""
        if not text:
            return []
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        
        if preserve_paragraphs:
            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            current_chunk = ""
            
            for paragraph in paragraphs:
                # If adding this paragraph would exceed chunk size
                if len(current_chunk) + len(paragraph) + 2 > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        # Start new chunk with overlap from previous
                        if chunk_overlap > 0 and chunks:
                            overlap_text = current_chunk[-chunk_overlap:]
                            current_chunk = overlap_text + "\n\n" + paragraph
                        else:
                            current_chunk = paragraph
                    else:
                        # Single paragraph is too long, split it
                        para_chunks = self._split_long_text(paragraph, chunk_size, chunk_overlap)
                        chunks.extend(para_chunks[:-1])
                        current_chunk = para_chunks[-1] if para_chunks else ""
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            chunks = self._split_long_text(text, chunk_size, chunk_overlap)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _split_long_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split long text without paragraph preservation"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
                else:
                    # Look for word boundary
                    for i in range(end, max(start + chunk_size // 2, end - 50), -1):
                        if text[i].isspace():
                            end = i
                            break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - chunk_overlap)
        
        return chunks