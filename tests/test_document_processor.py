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

import pytest
from unittest.mock import patch, MagicMock
from app.document_processor import get_embedding_function, process_documents

@patch('app.document_processor.config')
@patch('app.document_processor.FastEmbedEmbeddings')
def test_get_embedding_function(mock_fastembed, mock_config):
    mock_config.__getitem__.return_value = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_func = get_embedding_function()
    assert isinstance(embedding_func, MagicMock)
    mock_fastembed.assert_called_once_with(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        max_length=512,
        doc_embed_type="passage",
        cache_dir="./models"
    )

@patch('app.document_processor.process_single_document')
@patch('app.document_processor.get_vectorstore')
@patch('app.document_processor.config')
def test_process_documents(mock_config, mock_get_vectorstore, mock_process_single):
    mock_config.__getitem__.side_effect = lambda key: {'chunk_size': 1000, 'chunk_overlap': 200}.get(key, None)

    # Mock file
    mock_file = MagicMock()
    mock_file.name = 'test.pdf'
    mock_file.getvalue.return_value = b'mock pdf content'
    
    # Mock document processing to return 3 chunks
    from app.document_handlers.base import DocumentChunk
    mock_chunks = [
        DocumentChunk(content="chunk1", metadata={"source": "test.pdf"}),
        DocumentChunk(content="chunk2", metadata={"source": "test.pdf"}), 
        DocumentChunk(content="chunk3", metadata={"source": "test.pdf"})
    ]
    mock_process_single.return_value = mock_chunks

    # Mock Chroma
    mock_vectorstore = MagicMock()
    mock_get_vectorstore.return_value = mock_vectorstore

    # Call the function
    result = process_documents([mock_file])

    # Assertions
    assert result[0] == 3  # Number of chunks (first element of tuple)
    assert len(result[1]) == 1  # One file processed
    assert result[1][0]['name'] == 'test.pdf'
    assert result[1][0]['chunks'] == 3
    mock_process_single.assert_called_once()
    mock_get_vectorstore.assert_called_once()
    mock_vectorstore.add_documents.assert_called_once()
    mock_vectorstore.persist.assert_called_once()
