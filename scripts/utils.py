"""
Hlokk - Shared Utilities
Config loading, JSON I/O, and common helpers.
"""
import json
import hashlib
from datetime import datetime
from pathlib import Path

import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(data: dict, path: str, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def paper_id_from_paths(pdf_paths: list[str]) -> str:
    """Generate a stable paper ID from sorted PDF paths."""
    combined = "|".join(sorted(pdf_paths))
    return hashlib.md5(combined.encode()).hexdigest()[:16]


def truncate_text(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
