"""Block Python file access to held-out data during development and subprocesses."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..configuration import load_json_config

ENVIRONMENT_KEY = "SCANNER_EXPERIMENT_GUARD_CONFIG"
_active_roots: tuple[Path, ...] = ()
_installed = False


def _audit(event: str, args: tuple) -> None:
    if event not in {"open", "os.listdir", "os.scandir", "os.chdir"} or not _active_roots:
        return
    value = args[0] if args else None
    if value is None and event in {"os.listdir", "os.scandir"}:
        value = Path.cwd()
    if value is None or isinstance(value, int):
        return
    path = Path(os.fsdecode(value)).resolve()
    if any(path == root or root in path.parents for root in _active_roots):
        raise PermissionError("held-out benchmark access is prohibited before final freeze")


def install(config_path: Path) -> None:
    global _active_roots, _installed
    config = load_json_config(config_path)
    _active_roots = tuple(
        Path(root).resolve()
        for root in config.get("input_isolation", {}).get("forbidden_roots", [])
    )
    if not _installed:
        sys.addaudithook(_audit)
        _installed = True


def install_from_environment() -> None:
    if os.environ.get(ENVIRONMENT_KEY):
        install(Path(os.environ[ENVIRONMENT_KEY]))


def protect_development(config_path: Path, work: Path) -> None:
    """Install a Python audit guard; this does not claim OS-level sandboxing."""
    if not load_json_config(config_path).get("input_isolation"):
        return
    directory = work / "input-guard"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sitecustomize.py").write_text(
        "try:\n"
        "    from bixolon_scanner.training.input_isolation import install_from_environment\n"
        "    install_from_environment()\n"
        "except Exception as exc:\n"
        "    raise SystemExit('Experiment input guard initialization failed') from exc\n",
        encoding="utf-8",
    )
    os.environ[ENVIRONMENT_KEY] = str(config_path.resolve())
    paths = [str(directory.resolve()), str(Path(__file__).resolve().parents[2])]
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    install(config_path)
