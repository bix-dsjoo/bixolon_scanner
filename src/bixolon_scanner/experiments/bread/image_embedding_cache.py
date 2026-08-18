from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...contracts.model_package import load_model_package, sha256_file
from ...training.data import read_manifest
from ...training.models import require_torch
from .proposal_embedding_verifier import collect_embeddings, validate_classifier_source


def full_image_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "image_id": record["image_id"],
            "boxes_xyxy": [[0.0, 0.0, float(record["width"]), float(record["height"])]],
            "scores": [1.0],
            "class_ids": [0],
        }
        for record in records
    ]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    folds = set(args.folds)
    expected_statuses = set(args.expected_statuses)
    difficulties = set(args.difficulties) if args.difficulties else None
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) in folds
        and row.get("expected_image_status") in expected_statuses
        and (difficulties is None or row.get("difficulty") in difficulties)
    ]
    package = load_model_package(args.package)
    torch = require_torch()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    manifest_metadata = json.loads(args.manifest_metadata.read_text(encoding="utf-8"))
    classifier_source = validate_classifier_source(checkpoint, manifest_metadata)
    raw, adapted, counts = collect_embeddings(
        records,
        full_image_predictions(records),
        dataset_root=args.dataset_root,
        package=package,
        checkpoint=checkpoint,
        batch_size=args.batch_size,
        cpu=args.cpu,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        raw_embeddings=raw.astype(np.float16),
        adapted_embeddings=adapted.astype(np.float16),
        counts=counts,
        image_ids=np.asarray([int(record["image_id"]) for record in records], dtype=np.int64),
        folds=np.asarray([int(record["fold"]) for record in records], dtype=np.int8),
        expected_image_statuses=np.asarray(
            [str(record["expected_image_status"]) for record in records]
        ),
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_full_image_embedding_cache",
        "folds": sorted(folds),
        "expected_statuses": sorted(expected_statuses),
        "difficulties": sorted(difficulties) if difficulties else None,
        "image_count": len(records),
        "embedding_dimension": int(raw.shape[1]),
        "classifier_source": classifier_source,
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "output": args.output.name,
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache DINOv3 full-image embeddings")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-metadata", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--expected-statuses",
        nargs="+",
        default=["ANNOTATED"],
    )
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
