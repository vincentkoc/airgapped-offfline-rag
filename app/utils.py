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
