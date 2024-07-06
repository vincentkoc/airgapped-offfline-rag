import yaml
import os
from dotenv import load_dotenv

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')

def load_config():
    load_dotenv()  # Load environment variables from .env file if it exists
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Override config values with environment variables if they exist
    for key in config:
        env_value = os.getenv(key.upper())
        if env_value:
            config[key] = env_value

    return config
