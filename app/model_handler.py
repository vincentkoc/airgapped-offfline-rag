from llama_cpp import Llama
import torch
import os
import streamlit as st

class ModelHandler:
    def __init__(self, config):
        self.config = config
        self.llama_model = None
        self.mistral_model = None
        self.available_models = []
        self.check_available_models()

    def check_available_models(self):
        self.available_models = []
        if os.path.exists(self.config['llama_model_path']):
            self.available_models.append("Llama 3")
        if os.path.exists(self.config['mistral_model_path']):
            self.available_models.append("Mistral")
        return self.available_models

    @st.cache_resource
    def load_llama(_self):
        return Llama(
            model_path=_self.config['llama_model_path'],
            n_ctx=_self.config['model_n_ctx'],
            n_batch=_self.config['model_n_batch'],
            n_gpu_layers=-1 if torch.cuda.is_available() else 0
        )

    @st.cache_resource
    def load_mistral(_self):
        return Llama(
            model_path=_self.config['mistral_model_path'],
            n_ctx=_self.config['model_n_ctx'],
            n_batch=_self.config['model_n_batch'],
            n_gpu_layers=-1 if torch.cuda.is_available() else 0
        )

    def get_model(self, model_choice):
        if model_choice == "Llama 3":
            return self.load_llama()
        elif model_choice == "Mistral":
            return self.load_mistral()
        else:
            raise ValueError(f"Model {model_choice} is not available. Available models: {', '.join(self.available_models)}")

    def generate_stream(self, prompt, model_choice="Llama 3"):
        model = self.get_model(model_choice)

        for output in model(
            prompt,
            max_tokens=self.config['max_input_length'],
            stop=["Human:", "\n"],
            echo=False,
            stream=True
        ):
            yield output['choices'][0]['text']
