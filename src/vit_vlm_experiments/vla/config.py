from __future__ import annotations

from pathlib import Path
import logging

import yaml
import torch


LOGGER = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    LOGGER.info("Loaded configuration from %s", path)
    return config


def resolve_device(value: str) -> str:
    if value != "auto":
        return value

    if torch.cuda.is_available():
        return "cuda"

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"

    return "cpu"
