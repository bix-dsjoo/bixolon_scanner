from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ...contracts.model_package import sha256_file
from ...evaluation.onnx_detector import raw_outputs_to_prediction
from ...runtime.imaging import image_original_size
from ...runtime.onnx import OrtRunner, prepare_rgb
from ...training.data import read_manifest


def _records(args: argparse.Namespace) -> list[dict[str, Any]]:
    folds = set(args.folds)
    statuses = set(args.expected_statuses)
    difficulties = set(args.difficulties) if args.difficulties else None
    return [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) in folds
        and row.get("expected_image_status") in statuses
        and (difficulties is None or row.get("difficulty") in difficulties)
    ]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    records = _records(args)
    model_paths = {
        0: args.fold0_model,
        1: args.fold1_model,
        2: args.fold2_model,
    }
    required_folds = sorted({int(record["fold"]) for record in records})
    missing = [fold for fold in required_folds if model_paths.get(fold) is None]
    if missing:
        raise ValueError(f"missing ONNX model routes for folds: {missing}")
    runners = {
        fold: OrtRunner(model_paths[fold], args.provider, args.cuda_dll_dir)
        for fold in required_folds
    }
    root = args.dataset_root.resolve()
    predictions = []
    route_counts = {fold: 0 for fold in required_folds}
    for record in records:
        fold = int(record["fold"])
        image_path = (root / str(record["image_path"])).resolve()
        image_path.relative_to(root)
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image_original_size(image)
            tensor = prepare_rgb(
                image,
                (args.input_height, args.input_width),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                reducing_gap=1.0,
            )[None]
        logits, boxes = runners[fold].run(
            [args.logits_output, args.boxes_output], args.input_name, tensor
        )
        prediction = raw_outputs_to_prediction(
            np.asarray(logits)[0],
            np.asarray(boxes)[0],
            image_width=width,
            image_height=height,
        )
        prediction.update(
            {
                "image_id": int(record["image_id"]),
                "outer_fold_model": fold,
            }
        )
        predictions.append(prediction)
        route_counts[fold] += 1
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_fold_routed_onnx_predictions",
        "image_count": len(records),
        "folds": required_folds,
        "expected_statuses": sorted(set(args.expected_statuses)),
        "difficulties": sorted(set(args.difficulties)) if args.difficulties else None,
        "provider": args.provider,
        "route_counts": {str(fold): count for fold, count in route_counts.items()},
        "model_sha256": {str(fold): sha256_file(model_paths[fold]) for fold in required_folds},
        "output": args.predictions_output.name,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run each image through its unseen outer-fold ONNX detector"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--expected-statuses", nargs="+", required=True)
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--fold0-model", type=Path)
    parser.add_argument("--fold1-model", type=Path)
    parser.add_argument("--fold2-model", type=Path)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--input-name", default="pixel_values")
    parser.add_argument("--logits-output", default="logits")
    parser.add_argument("--boxes-output", default="pred_boxes")
    parser.add_argument("--input-height", type=int, default=640)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
