import yaml
import os
from dotenv import load_dotenv

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')

def load_config():
    load_dotenv()  # Load environment variables from .env file if it exists
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Process environment variables and convert numeric values to integers
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]  # Remove ${ and }
            default_value = None
            if ":-" in env_var:
                env_var, default_value = env_var.split(":-")
            config[key] = os.getenv(env_var, default_value)

        # Convert numeric values to integers
        if isinstance(config[key], str) and config[key].isdigit():
            config[key] = int(config[key])

    return config
