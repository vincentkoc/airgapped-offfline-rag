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

from llama_cpp import Llama
import torch
import os
import streamlit as st
import time
import logging
from typing import Dict, List, Optional, Iterator, Any
from .telemetry import telemetry

logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self, config):
        self.config = config
        self.loaded_models = {}
        self.check_available_models()

    def check_available_models(self):
        self.available_models = []
        model_paths = {
            "Llama": self.config.get('llama_model_path'),
            "Mistral": self.config.get('mistral_model_path'),
            "Gemma": self.config.get('gemma_model_path'),
            "DeepSeek": self.config.get('deepseek_model_path'),
            "Phi": self.config.get('phi_model_path'),
            "Qwen": self.config.get('qwen_model_path')
        }
        for model_name, path in model_paths.items():
            if path and os.path.exists(path):
                self.available_models.append(model_name)

    @st.cache_resource
    def load_model(_self, model_path):
        try:
            quantization = _self._get_quantization_from_filename(model_path)
            return Llama(
                model_path=model_path,
                n_ctx=_self.config['model_n_ctx'],
                n_batch=_self.config['model_n_batch'],
                n_gpu_layers=-1 if torch.cuda.is_available() else 0,
                f16_kv=True,
                use_mmap=True,
                verbose=False,
                **_self._get_quantization_params(quantization)
            )
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise

    def _get_quantization_from_filename(self, filename):
        filename_lower = filename.lower()
        for q in ['q2', 'q3', 'q4', 'q5', 'q6', 'q8']:
            if q in filename_lower:
                return q
        return 'default'

    def _get_quantization_params(self, quantization):
        return {'n_gqa': int(quantization[1])} if quantization.startswith('q') else {}

    def get_model(self, model_choice):
        if model_choice not in self.loaded_models:
            model_paths = {
                "Llama": self.config['llama_model_path'],
                "Mistral": self.config['mistral_model_path'],
                "Gemma": self.config['gemma_model_path'],
                "DeepSeek": self.config['deepseek_model_path'],
                "Phi": self.config['phi_model_path'],
                "Qwen": self.config['qwen_model_path']
            }
            if model_choice not in model_paths:
                raise ValueError(f"Model {model_choice} is not available. Available models: {', '.join(self.available_models)}")
            self.loaded_models[model_choice] = self.load_model(model_paths[model_choice])
        return self.loaded_models[model_choice]

    def generate_stream(self, prompt, model_choice="Mistral"):
        model = self.get_model(model_choice)
        start_time = time.time()
        tokens_generated = 0

        for output in model(
            prompt,
            max_tokens=self._get_dynamic_max_tokens(prompt),
            stop=["Question:", "Human:", "\n\nQuestion:", "\n\nHuman:"],
            echo=False,
            stream=True,
            temperature=0.7,
            top_p=0.95,
            repeat_penalty=1.1
        ):
            tokens_generated += 1
            yield output['choices'][0]['text']

        end_time = time.time()
        self._log_performance_metrics(start_time, end_time, tokens_generated)

    def _get_dynamic_max_tokens(self, prompt):
        prompt_tokens = len(prompt.split())
        max_tokens = min(
            int(self.config['max_input_length']),
            self.config['model_n_ctx'] - prompt_tokens
        )
        return max(1, max_tokens)  # Ensure at least 1 token is generated
    
    def get_model_info(self, model_choice: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        model_paths = {
            "Llama": self.config['llama_model_path'],
            "Mistral": self.config['mistral_model_path'],
            "Gemma": self.config['gemma_model_path'],
            "DeepSeek": os.getenv('DEEPSEEK_R1_7B_PATH'),
            "Phi": os.getenv('PHI4_14B_PATH'),
            "Qwen": os.getenv('QWEN25_7B_PATH')
        }
        
        if model_choice not in model_paths:
            return {}
        
        model_path = model_paths[model_choice]
        info = {
            "name": model_choice,
            "path": model_path,
            "available": model_path and os.path.exists(model_path),
            "loaded": model_choice in self.loaded_models
        }
        
        if info["available"]:
            try:
                stat = os.stat(model_path)
                info.update({
                    "file_size_mb": stat.st_size / (1024 * 1024),
                    "quantization": self._get_quantization_from_filename(model_path),
                    "modified_time": stat.st_mtime
                })
            except Exception as e:
                logger.warning(f"Error getting model file info: {e}")
        
        return info
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models with their information"""
        return [self.get_model_info(model) for model in self.available_models]
    
    def get_model_prompt_template(self, model_choice: str) -> str:
        """Get the appropriate prompt template for the model"""
        templates = {
            "Llama": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "Mistral": "<s>[INST] {prompt} [/INST]",
            "Gemma": "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
            "DeepSeek": "<￥begin₁of₁sentence￥>User: {prompt}\n\nAssistant: ",
            "Phi": "<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
            "Qwen": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        }
        
        return templates.get(model_choice, "{prompt}")
    
    def format_prompt(self, prompt: str, model_choice: str) -> str:
        """Format prompt using model-specific template"""
        template = self.get_model_prompt_template(model_choice)
        return template.format(prompt=prompt)

    def _log_performance_metrics(self, start_time, end_time, tokens_generated):
        total_time = end_time - start_time
        tokens_per_second = tokens_generated / total_time
        logger.info(f"Generated {tokens_generated} tokens in {total_time:.2f} seconds ({tokens_per_second:.2f} tokens/sec)")
