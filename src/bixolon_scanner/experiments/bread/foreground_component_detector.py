from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from ...evaluation.detector import _metrics, detection_error_rows
from ...training.data import read_manifest


def foreground_component_boxes(
    image: np.ndarray,
    *,
    color_distance: float,
    minimum_area_ratio: float,
    maximum_area_ratio: float,
    opening_size: int,
    closing_size: int,
    padding_ratio: float,
) -> list[list[float]]:
    """Return foreground component boxes relative to a robust border background color."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("foreground component input must be an RGB image")
    if not 0.0 < minimum_area_ratio < maximum_area_ratio <= 1.0:
        raise ValueError("foreground component area ratios are invalid")
    if opening_size < 1 or closing_size < 1:
        raise ValueError("foreground morphology sizes must be positive")
    height, width = image.shape[:2]
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]), axis=0)
    background = np.median(border, axis=0).astype(np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    background_lab = cv2.cvtColor(background.reshape(1, 1, 3), cv2.COLOR_RGB2LAB).astype(
        np.float32
    )[0, 0]
    distance = np.linalg.norm(lab - background_lab, axis=2)
    mask = (distance >= color_distance).astype(np.uint8)
    if opening_size > 1:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((opening_size, opening_size), dtype=np.uint8),
        )
    if closing_size > 1:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((closing_size, closing_size), dtype=np.uint8),
        )
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    image_area = float(height * width)
    boxes = []
    for label in range(1, count):
        x, y, component_width, component_height, area = [int(value) for value in stats[label]]
        area_ratio = area / image_area
        if not minimum_area_ratio <= area_ratio <= maximum_area_ratio:
            continue
        padding = padding_ratio * max(component_width, component_height)
        boxes.append(
            [
                max(0.0, x - padding),
                max(0.0, y - padding),
                min(float(width), x + component_width + padding),
                min(float(height), y + component_height + padding),
            ]
        )
    return boxes


def _records(args: argparse.Namespace) -> list[dict[str, Any]]:
    folds = set(args.folds)
    difficulties = set(args.difficulties) if args.difficulties else None
    return [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) in folds
        and row.get("expected_image_status") == "ANNOTATED"
        and (difficulties is None or row.get("difficulty") in difficulties)
    ]


def _load_resized_images(
    records: list[dict[str, Any]], dataset_root: Path, maximum_side: int
) -> list[tuple[np.ndarray, float, float]]:
    root = dataset_root.resolve()
    loaded = []
    for record in records:
        path = (root / str(record["image_path"])).resolve()
        path.relative_to(root)
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            original_width, original_height = image.size
            scale = min(1.0, maximum_side / max(original_width, original_height))
            resized = image.resize(
                (max(1, round(original_width * scale)), max(1, round(original_height * scale))),
                Image.Resampling.BILINEAR,
            )
        values = np.asarray(resized, dtype=np.uint8)
        loaded.append((values, original_width / values.shape[1], original_height / values.shape[0]))
    return loaded


def _predictions(
    records: list[dict[str, Any]],
    images: list[tuple[np.ndarray, float, float]],
    *,
    color_distance: float,
    minimum_area_ratio: float,
    maximum_area_ratio: float,
    opening_size: int,
    closing_size: int,
    padding_ratio: float,
) -> list[dict[str, Any]]:
    output = []
    for record, (image, scale_x, scale_y) in zip(records, images):
        boxes = foreground_component_boxes(
            image,
            color_distance=color_distance,
            minimum_area_ratio=minimum_area_ratio,
            maximum_area_ratio=maximum_area_ratio,
            opening_size=opening_size,
            closing_size=closing_size,
            padding_ratio=padding_ratio,
        )
        scaled = [
            [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y]
            for box in boxes
        ]
        output.append(
            {
                "image_id": int(record["image_id"]),
                "boxes_xyxy": scaled,
                "scores": [1.0] * len(scaled),
                "class_ids": [0] * len(scaled),
            }
        )
    return output


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    records = _records(args)
    images = _load_resized_images(records, args.dataset_root, args.maximum_side)
    candidates = []
    cache: dict[tuple[float, float, int, int, float], list[dict[str, Any]]] = {}
    for distance, minimum_area, opening, closing, padding in product(
        args.color_distances,
        args.minimum_area_ratios,
        args.opening_sizes,
        args.closing_sizes,
        args.padding_ratios,
    ):
        key = (distance, minimum_area, opening, closing, padding)
        predictions = _predictions(
            records,
            images,
            color_distance=distance,
            minimum_area_ratio=minimum_area,
            maximum_area_ratio=args.maximum_area_ratio,
            opening_size=opening,
            closing_size=closing,
            padding_ratio=padding,
        )
        metrics = _metrics(
            records,
            predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
            max_queries=600,
        )
        cache[key] = predictions
        candidates.append(
            {
                "color_distance": distance,
                "minimum_area_ratio": minimum_area,
                "opening_size": opening,
                "closing_size": closing,
                "padding_ratio": padding,
                "metrics": metrics,
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            -row["metrics"]["false_positive_count"] - row["metrics"]["false_negative_count"],
            row["metrics"]["exact_image_rate"],
            -row["metrics"]["false_negative_count"],
        ),
    )
    selected_key = (
        selected["color_distance"],
        selected["minimum_area_ratio"],
        selected["opening_size"],
        selected["closing_size"],
        selected["padding_ratio"],
    )
    selected_predictions = cache[selected_key]
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_foreground_component_detector",
        "selection_scope": "same border-background and connected-component policy for every image",
        "folds": sorted(set(args.folds)),
        "difficulties": sorted(set(args.difficulties)) if args.difficulties else None,
        "image_count": len(records),
        "maximum_side": args.maximum_side,
        "maximum_area_ratio": args.maximum_area_ratio,
        "candidate_count": len(candidates),
        "zero_error_candidate_count": sum(
            row["metrics"]["false_positive_count"] == 0
            and row["metrics"]["false_negative_count"] == 0
            for row in candidates
        ),
        "selected": selected,
        "error_images": detection_error_rows(
            records,
            selected_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_output.write_text(
            "".join(json.dumps(row) + "\n" for row in selected_predictions),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate label-independent foreground components as detector proposals"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--maximum-side", type=int, default=1024)
    parser.add_argument("--color-distances", type=float, nargs="+", required=True)
    parser.add_argument("--minimum-area-ratios", type=float, nargs="+", required=True)
    parser.add_argument("--maximum-area-ratio", type=float, default=0.5)
    parser.add_argument("--opening-sizes", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--closing-sizes", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--padding-ratios", type=float, nargs="+", default=[0.0, 0.01, 0.02])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
