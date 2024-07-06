import pytest
import yaml
from app.utils import load_config

def test_load_config(tmp_path):
    config_content = {
        "llama_model_path": "./models/llama-3-8b.gguf",
        "mistral_model_path": "./models/mistral-7b-v0.1.gguf",
        "model_n_ctx": 2048,
        "model_n_batch": 512,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 3,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "use_fast_embed": True,
        "default_model": "llama",
        "max_input_length": 512
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    # Monkeypatch the config file path
    monkeypatch.setattr('app.utils.CONFIG_PATH', str(config_file))

    loaded_config = load_config()
    assert loaded_config == config_content
