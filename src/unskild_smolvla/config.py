from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_INTERFACE_SCHEMA_PATH = PACKAGE_DIR / "interface_schema.yaml"
DEFAULT_MODEL_CONFIG_PATH = PACKAGE_DIR / "model_config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_interface_schema(path: str | Path | None = None) -> dict[str, Any]:
    schema_path = Path(path) if path is not None else DEFAULT_INTERFACE_SCHEMA_PATH
    data = _load_yaml(schema_path)
    if "observation" not in data or "action" not in data:
        raise ValueError(f"Interface schema must define observation and action: {schema_path}")
    return data


def load_model_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_MODEL_CONFIG_PATH
    data = _load_yaml(config_path)
    required = {
        "state_dim",
        "prompt_embedding_dim",
        "hidden_dims",
        "action_horizon",
        "action_dim",
    }
    missing = sorted(required - set(data.keys()))
    if missing:
        raise ValueError(f"Missing required model config keys in {config_path}: {missing}")
    return data
