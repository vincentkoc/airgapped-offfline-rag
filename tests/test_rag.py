import pytest
from unittest.mock import patch, MagicMock
from app.rag import retrieve_context, get_embedding_function

@pytest.fixture
def mock_config():
    return {
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'top_k': 3
    }

@patch('app.rag.TextEmbedding')
def test_get_embedding_function(mock_text_embedding, mock_config):
    mock_text_embedding.list_supported_models.return_value = ['sentence-transformers/all-MiniLM-L6-v2']
    with patch('app.rag.config', mock_config):
        embedding_func = get_embedding_function()
        assert isinstance(embedding_func, MagicMock)
        mock_text_embedding.assert_called_once_with('sentence-transformers/all-MiniLM-L6-v2')

@patch('app.rag.Chroma')
@patch('app.rag.get_embedding_function')
def test_retrieve_context(mock_get_embedding, mock_chroma, mock_config):
    with patch('app.rag.config', mock_config):
        # Mock Chroma and its methods
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
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
        mock_chroma.assert_called_once_with(persist_directory="./chroma_db", embedding_function=mock_get_embedding.return_value)
        mock_vectorstore.similarity_search.assert_called_once_with("test query", k=3)
        assert result == 'content1\ncontent2\ncontent3'
