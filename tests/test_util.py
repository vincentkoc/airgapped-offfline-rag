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
import yaml
from app.utils import load_config, CONFIG_PATH

def test_load_config(tmp_path, monkeypatch):
    config_content = {
        "llama_model_path": "./models/llama-3-8b.gguf",
        "mistral_model_path": "./models/mistral-7b-v0.1.gguf",
        "model_n_ctx": 2048,
        "model_n_batch": 512,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 3,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
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
