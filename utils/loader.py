import yaml
from typing import Dict


def load_yaml(config_file: str) -> Dict:
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config