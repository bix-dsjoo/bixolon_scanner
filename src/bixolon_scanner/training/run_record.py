from __future__ import annotations

import json
import platform
import sys
from argparse import Namespace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


RECORDED_PACKAGES = ("torch", "torchvision", "transformers", "onnx", "onnxruntime-gpu")


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in RECORDED_PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _safe_arguments(args: Namespace) -> dict[str, Any]:
    omitted_paths = {"dataset_root", "manifest", "output_dir"}
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in omitted_paths:
            continue
        if key == "weights" and value is not None:
            result[key] = Path(value).name
        elif isinstance(value, Path):
            result[key] = value.name
        else:
            result[key] = value
    return result


def write_run_record(
    output_dir: Path,
    *,
    task: str,
    args: Namespace,
    device: str,
    dataset_sizes: dict[str, int],
) -> None:
    dataset: dict[str, str] = {}
    manifest_path = getattr(args, "manifest", None)
    if manifest_path is not None:
        metadata_path = Path(manifest_path).parent / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            dataset = {
                "version": metadata["dataset_version"],
                "manifest_sha256": metadata["manifest_sha256"],
            }
    record = {
        "task": task,
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device,
        "arguments": _safe_arguments(args),
        "dataset_sizes": dataset_sizes,
        "dataset": dataset,
        "dependencies": _package_versions(),
    }
    (output_dir / "run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
