from llama_cpp import Llama

class ModelHandler:
    def __init__(self, config):
        self.config = config
        self.llama_model = None
        self.mistral_model = None

    def load_llama(self):
        if not self.llama_model:
            self.llama_model = Llama(
                model_path=self.config['llama_model_path'],
                n_ctx=self.config['model_n_ctx'],
                n_batch=self.config['model_n_batch'],
            )

    def load_mistral(self):
        if not self.mistral_model:
            self.mistral_model = Llama(
                model_path=self.config['mistral_model_path'],
                n_ctx=self.config['model_n_ctx'],
                n_batch=self.config['model_n_batch'],
            )

    def generate_stream(self, prompt, model_choice="llama"):
        if model_choice == "llama":
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
