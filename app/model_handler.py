from llama_cpp import Llama
import torch
import os

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

    def load_llama(self):
        if "Llama 3" in self.available_models and not self.llama_model:
            self.llama_model = Llama(
                model_path=self.config['llama_model_path'],
                n_ctx=self.config['model_n_ctx'],
                n_batch=self.config['model_n_batch'],
                n_gpu_layers=-1 if torch.cuda.is_available() else 0
            )

    def load_mistral(self):
        if "Mistral" in self.available_models and not self.mistral_model:
            self.mistral_model = Llama(
                model_path=self.config['mistral_model_path'],
                n_ctx=self.config['model_n_ctx'],
                n_batch=self.config['model_n_batch'],
                n_gpu_layers=-1 if torch.cuda.is_available() else 0
            )

    def generate_stream(self, prompt, model_choice="Llama 3"):
        if model_choice not in self.available_models:
            raise ValueError(f"Model {model_choice} is not available. Available models: {', '.join(self.available_models)}")

        if model_choice == "Llama 3":
            self.load_llama()
            model = self.llama_model
        else:
            self.load_mistral()
            model = self.mistral_model

        for output in model(
            prompt,
            max_tokens=self.config['max_input_length'],
            stop=["Human:", "\n"],
            echo=False,
            stream=True
        ):
            yield output['choices'][0]['text']
