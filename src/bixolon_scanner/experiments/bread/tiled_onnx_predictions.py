from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ...contracts.model_package import sha256_file
from ...evaluation.onnx_detector import raw_outputs_to_prediction
from ...runtime.onnx import OrtRunner, prepare_rgb
from ...training.data import read_manifest


def tile_windows(
    width: int,
    height: int,
    *,
    rows: int,
    columns: int,
    width_fraction: float,
    height_fraction: float,
) -> list[tuple[int, int, int, int]]:
    if rows < 1 or columns < 1:
        raise ValueError("tile rows and columns must be positive")
    if not 0.0 < width_fraction <= 1.0 or not 0.0 < height_fraction <= 1.0:
        raise ValueError("tile fractions must be in (0, 1]")
    tile_width = max(1, min(width, round(width * width_fraction)))
    tile_height = max(1, min(height, round(height * height_fraction)))
    x_starts = np.linspace(0, width - tile_width, columns).round().astype(int)
    y_starts = np.linspace(0, height - tile_height, rows).round().astype(int)
    return [
        (int(x), int(y), int(x + tile_width), int(y + tile_height))
        for y in y_starts
        for x in x_starts
    ]


def map_tile_prediction(
    prediction: dict[str, Any], *, x_offset: int, y_offset: int
) -> dict[str, Any]:
    return {
        **prediction,
        "boxes_xyxy": [
            [x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset]
            for x1, y1, x2, y2 in prediction["boxes_xyxy"]
        ],
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.tile_batch_size < 1:
        raise ValueError("tile batch size must be positive")
    folds = set(args.folds)
    difficulties = set(args.difficulties) if args.difficulties else None
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) in folds
        and row.get("expected_image_status") == "ANNOTATED"
        and (difficulties is None or row.get("difficulty") in difficulties)
    ]
    runner = OrtRunner(args.model, args.provider, args.cuda_dll_dir)
    root = args.dataset_root.resolve()
    outputs = []
    tile_count = 0
    for record in records:
        image_path = (root / str(record["image_path"])).resolve()
        image_path.relative_to(root)
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            windows = tile_windows(
                width,
                height,
                rows=args.rows,
                columns=args.columns,
                width_fraction=args.tile_width_fraction,
                height_fraction=args.tile_height_fraction,
            )
            tensors = np.stack(
                [
                    prepare_rgb(
                        image.crop(window),
                        (args.input_height, args.input_width),
                        (0.0, 0.0, 0.0),
                        (1.0, 1.0, 1.0),
                        reducing_gap=1.0,
                    )
                    for window in windows
                ]
            )
        logits_parts = []
        box_parts = []
        for start in range(0, len(tensors), args.tile_batch_size):
            logits, boxes = runner.run(
                [args.logits_output, args.boxes_output],
                args.input_name,
                tensors[start : start + args.tile_batch_size],
            )
            logits_parts.append(np.asarray(logits))
            box_parts.append(np.asarray(boxes))
        logits = np.concatenate(logits_parts)
        boxes = np.concatenate(box_parts)
        mapped = []
        for index, (x1, y1, x2, y2) in enumerate(windows):
            prediction = raw_outputs_to_prediction(
                np.asarray(logits)[index],
                np.asarray(boxes)[index],
                image_width=x2 - x1,
                image_height=y2 - y1,
            )
            mapped.append(map_tile_prediction(prediction, x_offset=x1, y_offset=y1))
        outputs.append(
            {
                "image_id": int(record["image_id"]),
                "boxes_xyxy": [box for row in mapped for box in row["boxes_xyxy"]],
                "scores": [score for row in mapped for score in row["scores"]],
                "class_ids": [class_id for row in mapped for class_id in row["class_ids"]],
                "top3_class_ids": [
                    class_ids for row in mapped for class_ids in row.get("top3_class_ids", [])
                ],
                "tile_windows": [list(window) for window in windows],
            }
        )
        tile_count += len(windows)
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_tiled_onnx_predictions",
        "selection_scope": "fixed normalized tile layout without labels or image identifiers",
        "folds": sorted(folds),
        "difficulties": sorted(difficulties) if difficulties is not None else None,
        "provider": args.provider,
        "image_count": len(records),
        "tile_count": tile_count,
        "tiles_per_image": args.rows * args.columns,
        "tile_batch_size": args.tile_batch_size,
        "tile_layout": {
            "rows": args.rows,
            "columns": args.columns,
            "width_fraction": args.tile_width_fraction,
            "height_fraction": args.tile_height_fraction,
        },
        "input_size": [args.input_height, args.input_width],
        "model": args.model.name,
        "model_sha256": sha256_file(args.model),
        "output": args.predictions_output.name,
    }
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.write_text(
        "".join(json.dumps(row) + "\n" for row in outputs), encoding="utf-8"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fixed tiled D-FINE ONNX detector")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--input-name", default="pixel_values")
    parser.add_argument("--logits-output", default="logits")
    parser.add_argument("--boxes-output", default="pred_boxes")
    parser.add_argument("--input-height", type=int, default=640)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--tile-width-fraction", type=float, default=0.65)
    parser.add_argument("--tile-height-fraction", type=float, default=0.65)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
