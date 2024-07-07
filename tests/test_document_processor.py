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

@patch('app.document_processor.os.path.exists', return_value=False)
@patch('app.document_processor.os.makedirs')
@patch('app.document_processor.PyPDFLoader')
@patch('app.document_processor.RecursiveCharacterTextSplitter')
@patch('app.document_processor.get_vectorstore')
@patch('app.document_processor.config')
def test_process_documents(mock_config, mock_get_vectorstore, mock_splitter, mock_loader, mock_makedirs, mock_exists):
    mock_config.__getitem__.side_effect = lambda key: {'chunk_size': 1000, 'chunk_overlap': 200}.get(key, None)

    # Mock file and loader
    mock_file = MagicMock()
    mock_file.name = 'test.pdf'
    mock_file.getvalue.return_value = b'mock pdf content'
    mock_loader.return_value.load.return_value = [MagicMock(metadata={})] * 2  # Two pages with metadata

    # Mock text splitter
    mock_splitter.return_value.split_documents.return_value = ['chunk1', 'chunk2', 'chunk3']

    # Mock Chroma
    mock_vectorstore = MagicMock()
    mock_get_vectorstore.return_value = mock_vectorstore

    # Call the function
    result = process_documents([mock_file])

    # Assertions
    assert result == 3  # Number of chunks
    mock_loader.assert_called_once()
    mock_splitter.assert_called_once_with(chunk_size=1000, chunk_overlap=200)
    mock_get_vectorstore.assert_called_once()
    mock_vectorstore.add_documents.assert_called_once()
    mock_vectorstore.persist.assert_called_once()
