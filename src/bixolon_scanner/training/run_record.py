from __future__ import annotations

import hashlib
import json
import platform
import sys
from argparse import Namespace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

RECORDED_PACKAGES = ("torch", "torchvision", "transformers", "onnx", "onnxruntime-gpu")


def _sha256_path(path: Path, *, suffix: str | None = None) -> str:
    digest = hashlib.sha256()
    paths = (
        [path]
        if path.is_file()
        else sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and (suffix is None or candidate.suffix == suffix)
        )
    )
    for candidate in paths:
        relative = candidate.name if path.is_file() else candidate.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


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
        "source_sha256": _sha256_path(Path(__file__).parent, suffix=".py"),
    }
    pipeline_contract_path = getattr(args, "pipeline_contract", None)
    if pipeline_contract_path is not None:
        from .pipeline_contract import canonical_contract_sha256, load_pipeline_contract

        contract = load_pipeline_contract(Path(pipeline_contract_path))
        if contract.component != task.removesuffix("_training"):
            raise ValueError("pipeline contract component does not match the training task")
        record["training_pipeline"] = {
            "component": contract.component,
            "version": contract.pipeline_version,
            "contract_sha256": canonical_contract_sha256(contract),
        }
    pretrained = getattr(args, "pretrained_name", None)
    if pretrained is not None and Path(str(pretrained)).exists():
        checkpoint_path = Path(str(pretrained)).resolve()
        record["starting_checkpoint"] = {
            "name": checkpoint_path.name,
            "sha256": _sha256_path(checkpoint_path),
        }
    try:
        import torch

        record["accelerator"] = {
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        record["accelerator"] = None
    (output_dir / "run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
