from __future__ import annotations

import tomllib
from pathlib import Path

import bixolon_scanner


def test_runtime_version_matches_project_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert bixolon_scanner.__version__ == project["version"]
