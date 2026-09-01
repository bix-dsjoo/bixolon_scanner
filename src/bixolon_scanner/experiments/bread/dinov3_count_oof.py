from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dinov3_exact_count_probe import (
    configure_cuda_runtime,
    count_metrics,
    cross_validated_count_probe,
    extract_features,
    load_manifest_records,
    softmax,
)


def oof_prediction_rows(
    records: list[dict],
    logits: np.ndarray,
    classes: np.ndarray,
    *,
    temperature: float,
) -> list[dict]:
    probabilities = softmax(logits, temperature=temperature)
    indices = np.argmax(probabilities, axis=1)
    return [
        {
            "image_id": int(record["image_id"]),
            "image_path": str(record["image_path"]),
            "fold": int(record["fold"]),
            "expected_count": int(record["count_label"]),
            "predicted_count": int(classes[index]),
            "confidence": float(probability[index]),
            "probabilities": {
                str(int(label)): float(probability[class_index])
                for class_index, label in enumerate(classes)
            },
        }
        for record, probability, index in zip(records, probabilities, indices, strict=True)
    ]


def run(args: argparse.Namespace) -> dict:
    configure_cuda_runtime(args.provider, args.cuda_dll_dir)
    records = load_manifest_records(
        args.manifest,
        args.dataset_root,
        require_folds=True,
    )
    features, elapsed = extract_features(
        records,
        args.source_model,
        image_size=args.image_size,
        provider=args.provider,
        feature_mode=args.feature_mode,
    )
    labels = np.asarray([record["count_label"] for record in records], dtype=np.int64)
    folds = np.asarray([record["fold"] for record in records], dtype=np.int64)
    result = cross_validated_count_probe(
        features,
        labels,
        folds,
        regularization_values=args.regularization_c,
        seed=args.seed,
    )
    rows = oof_prediction_rows(
        records,
        result["selected_logits"],
        result["classes"],
        temperature=float(result["selected_temperature"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "dinov3_exact_count_true_group_oof",
        "image_count": len(records),
        "feature_mode": args.feature_mode,
        "feature_extraction_seconds": elapsed,
        "selected_regularization_c": result["selected_regularization_c"],
        "selected_temperature": result["selected_temperature"],
        "metrics": count_metrics(
            labels,
            result["selected_logits"],
            result["classes"],
            temperature=float(result["selected_temperature"]),
        ),
        "prediction_path": args.output.resolve().as_posix(),
        "in_sample_predictions_used": False,
        "independent_test": False,
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write true group-OOF DINOv3 count predictions")
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--feature-mode", choices=("cls", "patch_mean"), default="patch_mean")
    parser.add_argument(
        "--regularization-c",
        type=float,
        nargs="+",
        default=[0.0001, 0.001, 0.01, 0.1, 1.0],
    )
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
