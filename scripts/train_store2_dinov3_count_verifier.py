from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.training.count_verifier import (
    export_count_verifier,
    extract_convnext_features,
    load_source_only_count_records,
    select_count_head,
    source_revision,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a source-only DINOv3 exact-count verifier")
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--regularization-c", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0]
    )
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = load_source_only_count_records(
        args.training_manifest,
        args.training_root,
        expected_dataset_version=args.dataset_version,
    )
    validation = load_source_only_count_records(
        args.validation_manifest,
        args.validation_root,
        expected_dataset_version=args.dataset_version,
    )
    training_hashes = {row["image_sha256"] for row in training}
    if training_hashes & {row["image_sha256"] for row in validation}:
        raise ValueError("count verifier training and validation images overlap")
    train_features, train_seconds = extract_convnext_features(
        training,
        args.weights,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
    )
    validation_features, validation_seconds = extract_convnext_features(
        validation,
        args.weights,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
    )
    train_labels = np.asarray([row["count_label"] for row in training], dtype=np.int64)
    validation_labels = np.asarray([row["count_label"] for row in validation], dtype=np.int64)
    head, temperature, validation_metrics, candidates = select_count_head(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        regularization_values=args.regularization_c,
        seed=args.seed,
    )
    model_path = args.output_dir / "count-verifier.onnx"
    export_count_verifier(
        args.weights,
        model_path,
        head,
        image_size=args.image_size,
        opset=args.opset,
    )
    report = {
        "schema_version": "1.0",
        "experiment": "store2_source_only_dinov3_convnext_exact_count",
        "model_role": "exact_count_verifier",
        "comparison_mode": "exact_count",
        "active_runtime_modified": False,
        "dataset_version": args.dataset_version,
        "training_source_policy": "bix_bakery_dataset-only",
        "training_manifest": str(args.training_manifest),
        "training_manifest_sha256": sha256_file(args.training_manifest),
        "training_image_count": len(training),
        "validation_manifest": str(args.validation_manifest),
        "validation_manifest_sha256": sha256_file(args.validation_manifest),
        "validation_image_count": len(validation),
        "source_revision": source_revision(),
        "source_weight_filename": args.weights.name,
        "source_weight_sha256": sha256_file(args.weights),
        "backbone_kind": "dinov3_convnext_tiny",
        "image_size": args.image_size,
        "count_labels": head["classes"].tolist(),
        "temperature": temperature,
        "confidence_threshold": 0.5,
        "validation": validation_metrics,
        "selection_candidates": candidates,
        "feature_extraction_seconds": {
            "training": train_seconds,
            "validation": validation_seconds,
        },
        "onnx_sha256": sha256_file(model_path),
        "opset": args.opset,
        "evaluation_images_used": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
