import os
import yaml
from pathlib import Path
from types import SimpleNamespace

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

root = Path(__file__).resolve().parent
file_path = os.path.join(root, 'data/VisDrone.yaml')

with open(file_path, 'r') as file:
    config_dict = yaml.safe_load(file)

cfg = config_dict