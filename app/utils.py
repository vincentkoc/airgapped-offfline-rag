import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)
