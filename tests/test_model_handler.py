import pytest
from unittest.mock import patch, MagicMock
from app.model_handler import ModelHandler

@pytest.fixture
def mock_config():
    return {
        'llama_model_path': './models/llama-3-8b.gguf',
        'mistral_model_path': './models/mistral-7b-v0.1.gguf',
        'model_n_ctx': 2048,
        'model_n_batch': 512,
        'max_input_length': 512
    }

@pytest.fixture
def model_handler(mock_config):
    return ModelHandler(mock_config)

@patch('app.model_handler.Llama')
def test_load_llama(mock_llama, model_handler):
    model_handler.load_llama()
    mock_llama.assert_called_once_with(
        model_path='./models/llama-3-8b.gguf',
        n_ctx=2048,
        n_batch=512,
        n_gpu_layers=-1
    )

@patch('app.model_handler.Llama')
def test_load_mistral(mock_llama, model_handler):
    model_handler.load_mistral()
    mock_llama.assert_called_once_with(
        model_path='./models/mistral-7b-v0.1.gguf',
        n_ctx=2048,
        n_batch=512,
        n_gpu_layers=-1
    )

@patch('app.model_handler.Llama')
def test_generate_stream_llama(mock_llama, model_handler):
    mock_llama_instance = MagicMock()
    mock_llama_instance.return_value = [{'choices': [{'text': 'Generated text'}]}]
    mock_llama.return_value = mock_llama_instance

    result = list(model_handler.generate_stream("Test prompt", model_choice="Llama 3"))

    mock_llama_instance.assert_called_once_with(
        "Test prompt",
        max_tokens=512,
        stop=["Human:", "\n"],
        echo=False,
        stream=True
    )
    assert result == ['Generated text']

@patch('app.model_handler.Llama')
def test_generate_stream_mistral(mock_llama, model_handler):
    mock_mistral_instance = MagicMock()
    mock_mistral_instance.return_value = [{'choices': [{'text': 'Generated text'}]}]
    mock_llama.return_value = mock_mistral_instance

    result = list(model_handler.generate_stream("Test prompt", model_choice="Mistral"))

    mock_mistral_instance.assert_called_once_with(
        "Test prompt",
        max_tokens=512,
        stop=["Human:", "\n"],
        echo=False,
        stream=True
    )
    assert result == ['Generated text']
