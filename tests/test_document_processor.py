import pytest
from unittest.mock import patch, MagicMock
from app.document_processor import process_documents, get_embedding_function

@pytest.fixture
def mock_config():
    return {
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'use_fast_embed': True,
        'embedding_model': 'test-model'
    }

@patch('app.document_processor.load_config')
def test_get_embedding_function(mock_load_config, mock_config):
    mock_load_config.return_value = mock_config
    with patch('app.document_processor.TextEmbedding') as mock_fast_embed, \
         patch('app.document_processor.HuggingFaceEmbeddings') as mock_hf_embed:

        # Test with FastEmbed
        mock_config['use_fast_embed'] = True
        embedding_func = get_embedding_function()
        assert isinstance(embedding_func, MagicMock)
        mock_fast_embed.assert_called_once()
        mock_hf_embed.assert_not_called()

        # Test with HuggingFaceEmbeddings
        mock_config['use_fast_embed'] = False
        embedding_func = get_embedding_function()
        assert isinstance(embedding_func, MagicMock)
        mock_hf_embed.assert_called_once_with(model_name='test-model')

@patch('app.document_processor.load_config')
@patch('app.document_processor.PyPDFLoader')
@patch('app.document_processor.RecursiveCharacterTextSplitter')
@patch('app.document_processor.Chroma')
@patch('app.document_processor.get_embedding_function')
def test_process_documents(mock_get_embedding, mock_chroma, mock_splitter, mock_loader, mock_load_config, mock_config):
    mock_load_config.return_value = mock_config
    # Mock file and loader
    mock_file = MagicMock()
    mock_file.name = 'test.pdf'
    mock_loader.return_value.load.return_value = ['doc1', 'doc2']

    # Mock text splitter
    mock_splitter.return_value.split_documents.return_value = ['chunk1', 'chunk2', 'chunk3']

    # Mock Chroma
    mock_vectorstore = MagicMock()
    mock_chroma.from_documents.return_value = mock_vectorstore

    # Call the function
    result = process_documents([mock_file])

    # Assertions
    assert result == 3  # Number of chunks
    mock_loader.assert_called_once()
    mock_splitter.assert_called_once_with(chunk_size=1000, chunk_overlap=200)
    mock_get_embedding.assert_called_once()
    mock_chroma.from_documents.assert_called_once()
    mock_vectorstore.persist.assert_called_once()
