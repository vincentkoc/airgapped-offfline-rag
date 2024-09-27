from llama_cpp import Llama
import torch
import os
import streamlit as st
import time
import logging

logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self, config):
        self.config = config
        self.available_models = []
        self.check_available_models()

    def check_available_models(self):
        self.available_models = []
        model_paths = {
            "Llama 3": self.config['llama_model_path'],
            "Mistral": self.config['mistral_model_path'],
            "Gemma": self.config['gemma_model_path']
        }
        for name, path in model_paths.items():
            if os.path.exists(path):
                self.available_models.append(name)
        return self.available_models

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
        if 'q2' in filename_lower:
            return 'q2'
        elif 'q3' in filename_lower:
            return 'q3'
        elif 'q4' in filename_lower:
            return 'q4'
        elif 'q5' in filename_lower:
            return 'q5'
        elif 'q6' in filename_lower:
            return 'q6'
        elif 'q8' in filename_lower:
            return 'q8'
        else:
            return 'default'

    def _get_quantization_params(self, quantization):
        params = {
            'q2': {'n_gqa': 2},
            'q3': {'n_gqa': 3},
            'q4': {'n_gqa': 4},
            'q5': {'n_gqa': 5},
            'q6': {'n_gqa': 6},
            'q8': {'n_gqa': 8},
            'default': {}
        }
        return params.get(quantization, {})

    def get_model(self, model_choice):
        model_paths = {
            "Llama 3": self.config['llama_model_path'],
            "Mistral": self.config['mistral_model_path'],
            "Gemma": self.config['gemma_model_path']
        }
        if model_choice not in model_paths:
            raise ValueError(f"Model {model_choice} is not available. Available models: {', '.join(self.available_models)}")

        return self.load_model(model_paths[model_choice])

    def generate_stream(self, prompt, model_choice="Mistral"):
        model = self.get_model(model_choice)
        start_time = time.time()
        tokens_generated = 0

        for output in model(
            prompt,
            max_tokens=self._get_dynamic_max_tokens(prompt),
            stop=["Human:", "\n"],
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

    def _log_performance_metrics(self, start_time, end_time, tokens_generated):
        total_time = end_time - start_time
        tokens_per_second = tokens_generated / total_time
        logger.info(f"Generated {tokens_generated} tokens in {total_time:.2f} seconds ({tokens_per_second:.2f} tokens/sec)")
