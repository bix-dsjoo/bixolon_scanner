"""Version-controlled JSON configuration loading with legacy redirects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REDIRECT_KEY = "$redirect"


def resolve_config_path(path: Path) -> Path:
    """Resolve a legacy redirect file to its canonical JSON configuration."""

    current = path.resolve()
    visited: set[Path] = set()
    while True:
        if current in visited:
            raise ValueError(f"configuration redirect cycle: {current}")
        visited.add(current)
        if not current.is_file():
            raise FileNotFoundError(current)
        value = json.loads(current.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or REDIRECT_KEY not in value:
            return current
        if set(value) != {REDIRECT_KEY} or not isinstance(value[REDIRECT_KEY], str):
            raise ValueError(f"invalid configuration redirect: {current}")
        current = (current.parent / value[REDIRECT_KEY]).resolve()


def load_json_config(path: Path) -> dict[str, Any]:
    canonical = resolve_config_path(path)
    value = json.loads(canonical.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be an object: {canonical}")
    return value
