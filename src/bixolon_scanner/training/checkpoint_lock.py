from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..contracts.model_package import sha256_file


def lock_detector_checkpoint(
    source: Path,
    output: Path,
    *,
    dataset_version: str,
    manifest: Path,
    source_revision: str,
    source_weight_sha256: str,
    synthetic_manifest: Path,
    coco_provenance: Path,
    training_config: Path,
    epochs: int,
    batch_size: int,
    backbone_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
    synthetic_seed: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"locked checkpoint already exists: {output}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Detector checkpoint lock requires PyTorch") from exc
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not any(key in payload for key in ("model", "ema")):
        raise ValueError("source is not a D-FINE training checkpoint")
    provenance = {
        "dataset_version": dataset_version,
        "manifest_sha256": sha256_file(manifest),
        "source_revision": source_revision,
        "source_weight_sha256": source_weight_sha256,
        "synthetic_manifest_sha256": sha256_file(synthetic_manifest),
        "coco_provenance_sha256": sha256_file(coco_provenance),
        "training_config_sha256": sha256_file(training_config),
        "epochs": epochs,
        "batch_size": batch_size,
        "backbone_learning_rate": backbone_learning_rate,
        "head_learning_rate": head_learning_rate,
        "weight_decay": weight_decay,
        "synthetic_seed": synthetic_seed,
    }
    payload["bixolon_training_provenance"] = provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "checkpoint": str(output),
        "sha256": sha256_file(output),
        "source_checkpoint_sha256": sha256_file(source),
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock native D-FINE checkpoint provenance")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-weight-sha256", required=True)
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--coco-provenance", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--backbone-learning-rate", type=float, required=True)
    parser.add_argument("--head-learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--synthetic-seed", type=int, required=True)
    args = parser.parse_args()
    result = lock_detector_checkpoint(
        args.source,
        args.output,
        dataset_version=args.dataset_version,
        manifest=args.manifest,
        source_revision=args.source_revision,
        source_weight_sha256=args.source_weight_sha256,
        synthetic_manifest=args.synthetic_manifest,
        coco_provenance=args.coco_provenance,
        training_config=args.training_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        backbone_learning_rate=args.backbone_learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        synthetic_seed=args.synthetic_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
