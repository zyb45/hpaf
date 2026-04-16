import json
import os
from typing import Any, Dict
import yaml
from .config import AppConfig


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str) -> AppConfig:
    data = load_yaml(path)
    return AppConfig(**data)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
