from llama_cpp import Llama
import torch
import os
import streamlit as st

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
        return Llama(
            model_path=model_path,
            n_ctx=_self.config['model_n_ctx'],
            n_batch=_self.config['model_n_batch'],
            n_gpu_layers=-1 if torch.cuda.is_available() else 0,
            f16_kv=True,
            use_mmap=True,
            n_gqa=8,
            verbose=False
        )

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

        for output in model(
            prompt,
            max_tokens=int(self.config['max_input_length']),
            stop=["Human:", "\n"],
            echo=False,
            stream=True,
            temperature=0.7,
            top_p=0.95,
            repeat_penalty=1.1
        ):
            yield output['choices'][0]['text']
