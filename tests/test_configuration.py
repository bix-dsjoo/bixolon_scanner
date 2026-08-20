from __future__ import annotations

import json
from pathlib import Path

import pytest

from bixolon_scanner.configuration import load_json_config, resolve_config_path

ROOT = Path(__file__).resolve().parents[1]


def test_config_redirect_resolves_relative_to_alias(tmp_path) -> None:
    canonical = tmp_path / "experiments" / "candidate.json"
    canonical.parent.mkdir()
    canonical.write_text(json.dumps({"experiment": {"version": "1.2.3"}}), encoding="utf-8")
    alias = tmp_path / "candidate.json"
    alias.write_text(json.dumps({"$redirect": "experiments/candidate.json"}), encoding="utf-8")

    assert resolve_config_path(alias) == canonical.resolve()
    assert load_json_config(alias)["experiment"]["version"] == "1.2.3"


def test_config_redirect_rejects_cycles_and_missing_targets(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"$redirect": "second.json"}), encoding="utf-8")
    second.write_text(json.dumps({"$redirect": "first.json"}), encoding="utf-8")

    with pytest.raises(ValueError, match="redirect cycle"):
        resolve_config_path(first)

    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"$redirect": "absent.json"}), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        resolve_config_path(missing)


def test_repository_legacy_config_redirects_are_valid() -> None:
    config_root = (ROOT / "configs").resolve()
    for path in (ROOT / "configs").glob("*.json"):
        value = load_json_config(path)
        canonical = resolve_config_path(path)
        assert value, path.name
        assert canonical != path.resolve()
        assert canonical.is_relative_to(config_root), path.name
