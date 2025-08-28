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
from typing import List, Dict, Optional, Union, BinaryIO
import pathlib
import mimetypes
from .base import BaseDocumentHandler

logger = logging.getLogger(__name__)

class DocumentHandlerFactory:
    """Factory for creating appropriate document handlers"""
    
    def __init__(self):
        self._handlers: Dict[str, BaseDocumentHandler] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register all available document handlers"""
        try:
            from .pdf_handler import PDFHandler
            self.register_handler(PDFHandler())
        except ImportError as e:
            logger.warning(f"PDF handler not available: {e}")
        
        try:
            from .docx_handler import DOCXHandler
            self.register_handler(DOCXHandler())
        except ImportError as e:
            logger.warning(f"DOCX handler not available: {e}")
        
        try:
            from .xlsx_handler import XLSXHandler
            self.register_handler(XLSXHandler())
        except ImportError as e:
            logger.warning(f"XLSX handler not available: {e}")
        
        try:
            from .pptx_handler import PPTXHandler
            self.register_handler(PPTXHandler())
        except ImportError as e:
            logger.warning(f"PPTX handler not available: {e}")
        
        try:
            from .markdown_handler import MarkdownHandler
            self.register_handler(MarkdownHandler())
        except ImportError as e:
            logger.warning(f"Markdown handler not available: {e}")
        
        try:
            from .html_handler import HTMLHandler
            self.register_handler(HTMLHandler())
        except ImportError as e:
            logger.warning(f"HTML handler not available: {e}")
        
        try:
            from .text_handler import TextHandler
            self.register_handler(TextHandler())
        except ImportError as e:
            logger.warning(f"Text handler not available: {e}")
        
        try:
            from .docling_handler import DoclingHandler
            self.register_handler(DoclingHandler())
        except ImportError as e:
            logger.warning(f"Docling handler not available: {e}")
    
    def register_handler(self, handler: BaseDocumentHandler):
        """Register a document handler"""
        for format_name in handler.supported_formats:
            self._handlers[format_name] = handler
        logger.info(f"Registered handler for formats: {handler.supported_formats}")
    
    def get_handler(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO], 
                   mimetype: Optional[str] = None) -> Optional[BaseDocumentHandler]:
        """Get the appropriate handler for a file"""
        
        # Try each registered handler to see if it can handle the file
        for handler in set(self._handlers.values()):
            if handler.can_handle(file_path_or_buffer, mimetype):
                logger.debug(f"Selected handler {type(handler).__name__} for file")
                return handler
        
        logger.warning(f"No handler found for file: {file_path_or_buffer}")
        return None
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get all supported formats and their extensions"""
        formats = {}
        for handler in set(self._handlers.values()):
            for format_name in handler.supported_formats:
                formats[format_name] = handler.supported_extensions
        return formats
    
    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions"""
        extensions = set()
        for handler in set(self._handlers.values()):
            extensions.update(handler.supported_extensions)
        return sorted(list(extensions))
    
    def get_supported_mimetypes(self) -> List[str]:
        """Get all supported MIME types"""
        mimetypes_set = set()
        for handler in set(self._handlers.values()):
            mimetypes_set.update(handler.supported_mimetypes)
        return sorted(list(mimetypes_set))
    
    def detect_format(self, file_path_or_buffer: Union[str, pathlib.Path, BinaryIO], 
                     mimetype: Optional[str] = None) -> Optional[str]:
        """Detect the format of a file"""
        handler = self.get_handler(file_path_or_buffer, mimetype)
        if handler:
            return handler.supported_formats[0]  # Return primary format
        return None

# Global factory instance
document_factory = DocumentHandlerFactory()