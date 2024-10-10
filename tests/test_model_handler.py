import pytest
from unittest.mock import patch, MagicMock
from app.model_handler import ModelHandler

@pytest.fixture
def mock_config():
    return {
        'llama_model_path': './models/llama-3-8b.gguf',
        'mistral_model_path': './models/mistral-7b-v0.1.gguf',
        'gemma_model_path': './models/gemma-2b.gguf',
        'max_input_length': 512,
        'model_n_batch': 512,
        'model_n_ctx': 2048,
        'model_temperature': 0.7,
        'model_top_k': 40,
        'model_top_p': 0.5,
        'model_repeat_penalty': 1.1
    }

@pytest.fixture
def model_handler(mock_config):
    return ModelHandler(mock_config)

@patch('app.model_handler.Llama')
@patch('app.model_handler.torch.cuda.is_available', return_value=False)
def test_load_model(mock_cuda, mock_llama, model_handler):
    model_handler.load_model('./models/llama-3-8b.gguf')
    mock_llama.assert_called_once_with(
        model_path='./models/llama-3-8b.gguf',
        n_ctx=2048,
        n_batch=512,
        n_gpu_layers=0,
        f16_kv=True,
        use_mmap=True,
        verbose=False
    )

@patch('app.model_handler.Llama')
def test_generate_stream(mock_llama, model_handler):
    mock_llama_instance = MagicMock()
    mock_llama_instance.return_value = [{'choices': [{'text': 'Generated text'}]}]
    mock_llama.return_value = mock_llama_instance

    with patch.object(model_handler, 'get_model', return_value=mock_llama_instance):
        result = list(model_handler.generate_stream("Test prompt", model_choice="Llama 3"))

    mock_llama_instance.assert_called_once_with(
        "Test prompt",
        max_tokens=512,
        stop=["Human:", "\n"],
        echo=False,
        stream=True,
        temperature=0.7,
        top_p=0.95,
        repeat_penalty=1.1
    )
    assert result == ['Generated text']

def test_get_quantization_from_filename(model_handler):
    assert model_handler._get_quantization_from_filename('model_q4.gguf') == 'q4'
    assert model_handler._get_quantization_from_filename('model_Q8.bin') == 'q8'
    assert model_handler._get_quantization_from_filename('model.bin') == 'default'

def test_get_quantization_params(model_handler):
    assert model_handler._get_quantization_params('q4') == {'n_gqa': 4}
    assert model_handler._get_quantization_params('q8') == {'n_gqa': 8}
    assert model_handler._get_quantization_params('default') == {}

def test_check_available_models(model_handler, mock_config):
    with patch('os.path.exists', return_value=True):
        model_handler.check_available_models()
        assert set(model_handler.available_models) == set(["Llama 3", "Mistral", "Gemma"])

def test_get_dynamic_max_tokens(model_handler):
    assert model_handler._get_dynamic_max_tokens("Short prompt") == 512
    long_prompt = " ".join(["word"] * 2000)
    assert model_handler._get_dynamic_max_tokens(long_prompt) == 48  # 2048 - 2000

@patch('app.model_handler.logger')
def test_log_performance_metrics(mock_logger, model_handler):
    model_handler._log_performance_metrics(0, 10, 100)
    mock_logger.info.assert_called_once_with("Generated 100 tokens in 10.00 seconds (10.00 tokens/sec)")
