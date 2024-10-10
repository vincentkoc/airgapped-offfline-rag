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
from app.rag import retrieve_context, get_embedding_function

@pytest.fixture
def mock_config():
    return {
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'top_k': 3
    }

@patch('app.rag.FastEmbedEmbeddings')
def test_get_embedding_function(mock_fastembed, mock_config):
    with patch('app.rag.config', mock_config):
        embedding_func = get_embedding_function()
        assert isinstance(embedding_func, MagicMock)
        mock_fastembed.assert_called_once_with(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            max_length=512,
            doc_embed_type="passage"
        )

@patch('app.rag.get_vectorstore')
@patch('app.rag.get_embedding_function')
def test_retrieve_context(mock_get_embedding, mock_get_vectorstore, mock_config):
    with patch('app.rag.config', mock_config):
        # Mock vectorstore and its methods
        mock_vectorstore = MagicMock()
        mock_get_vectorstore.return_value = mock_vectorstore
        mock_docs = [
            MagicMock(page_content='content1'),
            MagicMock(page_content='content2'),
            MagicMock(page_content='content3')
        ]
        mock_vectorstore.similarity_search.return_value = mock_docs

        # Call the function
        result = retrieve_context("test query")

        # Assertions
        mock_get_embedding.assert_called_once()
        mock_get_vectorstore.assert_called_once()
        mock_vectorstore.similarity_search.assert_called_once_with("test query", k=3)
        assert result == 'content1\ncontent2\ncontent3'
