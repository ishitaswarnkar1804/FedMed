"""Load and expose FedMed configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(relative: str, config: dict[str, Any] | None = None) -> Path:
    root = PROJECT_ROOT
    if relative.startswith("data/") or relative.startswith("logs/"):
        return root / relative
    return root / relative
