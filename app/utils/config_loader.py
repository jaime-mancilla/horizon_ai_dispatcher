# app/utils/config_loader.py

import os
import yaml
from dotenv import load_dotenv

load_dotenv()

def load_config(path: str = "config/default.yaml") -> dict:
    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Expand env vars (e.g. ${ELEVENLABS_API_KEY})
    def expand_env_vars(obj):
        if isinstance(obj, dict):
            return {k: expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_env_vars(i) for i in obj]
        elif isinstance(obj, str) and "${" in obj:
            return os.path.expandvars(obj)
        return obj

    return expand_env_vars(raw_config)
