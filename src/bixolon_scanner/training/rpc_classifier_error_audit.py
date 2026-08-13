from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .calibration import softmax
from .rpc_class_aware_nms import _keep_indices
from .rpc_context_rejector import _geometry_features, _read_jsonl
from .rpc_worker_gate import postprocess_worker_gate


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def _intersection_over_detection(left: list[float], right: list[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    area = max((left[2] - left[0]) * (left[3] - left[1]), 1e-9)
    return width * height / area


def _neighbor_overlap(left: list[float], right: list[float]) -> dict[str, float]:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    left_area = max((left[2] - left[0]) * (left[3] - left[1]), 1e-9)
    right_area = max((right[2] - right[0]) * (right[3] - right[1]), 1e-9)
    return {
        "intersection_over_self": intersection / left_area,
        "intersection_over_other": intersection / right_area,
        "containment": intersection / min(left_area, right_area),
    }


def _crop(image: Image.Image, box: list[float], size: tuple[int, int]) -> Image.Image:
    width, height = image.size
    clipped = (
        max(0, int(box[0])),
        max(0, int(box[1])),
        min(width, int(np.ceil(box[2]))),
        min(height, int(np.ceil(box[3]))),
    )
    return image.crop(clipped).resize(size, Image.Resampling.BILINEAR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--role", choices=("calibration", "selection"), default="selection")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.output_dir
    run_dir = root / "runs" / "full" / f"seed{args.seed}"
    detector_dir = root / "detector"
    records = {
        int(row["image_id"]): row
        for row in _read_jsonl(detector_dir / "manifest" / "manifest.jsonl")
        if row["role"] == args.role
    }
    predictions = {
        str(row["sample_key"]): row
        for row in _read_jsonl(detector_dir / "predictions" / "val_oof.jsonl")
    }
    threshold = json.loads(
        (detector_dir / "threshold.json").read_text(encoding="utf-8")
    )["selected_score_threshold"]
    options = dict(config["detector"], score_threshold=float(threshold))
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    policy = json.loads(
        (run_dir / "context-rejector" / "report.json").read_text(encoding="utf-8")
    )["models"]["logistic"]["policy"]
    archive = np.load(
        run_dir
        / (
            "partial_calibration_predictions.npz"
            if args.role == "calibration"
            else "selection_predictions.npz"
        )
    )
    probabilities = softmax(archive["logits"], float(calibration["temperature"]))
    quality_key = "calibration_oof" if args.role == "calibration" else "selection"
    quality = np.load(run_dir / "context-rejector" / "logistic_scores.npz")[quality_key]
    sample_index = {
        str(sample_id): index
        for index, sample_id in enumerate(archive["sample_ids"])
    }
    errors: list[dict[str, Any]] = []
    ambiguity_rows: list[tuple[float, bool]] = []
    for image_id, record in records.items():
        result = postprocess_worker_gate(
            record,
            predictions[f"{record['source']}:{image_id}"],
            options,
        )
        if result["recapture_reasons"]:
            continue
        detections = result["detections"]
        ids = [f"val:{image_id}:det{index}" for index in range(len(detections))]
        if not all(sample_id in sample_index for sample_id in ids):
            continue
        indices = [sample_index[sample_id] for sample_id in ids]
        classes = [int(probabilities[index].argmax()) for index in indices]
        kept = _keep_indices(detections, classes, 0.55)
        for detection_index in kept:
            archive_index = indices[detection_index]
            target = int(archive["targets"][archive_index])
            predicted = classes[detection_index]
            approved = (
                probabilities[archive_index].max()
                >= float(policy["classifier_threshold"])
                and quality[archive_index] >= float(policy["quality_threshold"])
            )
            match = result["matches"].get(str(detection_index))
            if match is None:
                continue
            gt_intersections = sorted(
                (
                    _intersection_over_detection(
                        detections[detection_index]["bbox_xyxy"],
                        _xywh_to_xyxy(other["bbox_xywh"]),
                    )
                    for other in record["annotations"]
                ),
                reverse=True,
            )
            geometry = _geometry_features(
                detections,
                float(record["width"]),
                float(record["height"]),
                detection_index,
            )
            ambiguity_rows.append(
                (
                    float(geometry[9]),
                    len(gt_intersections) > 1 and gt_intersections[1] >= 0.4,
                )
            )
            if target < 0 or predicted == target or not approved:
                continue
            annotation = record["annotations"][int(match[0])]
            errors.append(
                {
                    "sample_id": ids[detection_index],
                    "image_path": record["image_path"],
                    "level": record["level"],
                    "predicted": predicted,
                    "target": target,
                    "confidence": float(probabilities[archive_index].max()),
                    "quality": float(quality[archive_index]),
                    "iou": float(match[1]),
                    "detection_box": detections[detection_index]["bbox_xyxy"],
                    "ground_truth_box": _xywh_to_xyxy(annotation["bbox_xywh"]),
                    "geometry_features": geometry,
                    "gt_intersection_over_detection": gt_intersections,
                    "neighbors": [
                        {
                            "index": other_index,
                            "predicted": classes[other_index],
                            "detector_score": float(detections[other_index]["score"]),
                            **_neighbor_overlap(
                                detections[detection_index]["bbox_xyxy"],
                                detections[other_index]["bbox_xyxy"],
                            ),
                        }
                        for other_index in range(len(detections))
                        if other_index != detection_index
                    ],
                }
            )

    cell_width, cell_height = 900, 420
    sheet = Image.new("RGB", (cell_width, cell_height * len(errors)), "white")
    for position, row in enumerate(errors):
        with Image.open(args.dataset_root / row["image_path"]) as source:
            image = source.convert("RGB")
        display = image.copy()
        display.thumbnail((500, 350), Image.Resampling.BILINEAR)
        scale_x = display.width / image.width
        scale_y = display.height / image.height
        draw = ImageDraw.Draw(display)
        draw.rectangle(
            [
                row["ground_truth_box"][0] * scale_x,
                row["ground_truth_box"][1] * scale_y,
                row["ground_truth_box"][2] * scale_x,
                row["ground_truth_box"][3] * scale_y,
            ],
            outline="lime",
            width=4,
        )
        draw.rectangle(
            [
                row["detection_box"][0] * scale_x,
                row["detection_box"][1] * scale_y,
                row["detection_box"][2] * scale_x,
                row["detection_box"][3] * scale_y,
            ],
            outline="red",
            width=4,
        )
        cell = Image.new("RGB", (cell_width, cell_height), "white")
        cell.paste(display, (0, 0))
        cell.paste(_crop(image, row["detection_box"], (180, 180)), (510, 0))
        cell.paste(_crop(image, row["ground_truth_box"], (180, 180)), (700, 0))
        label = ImageDraw.Draw(cell)
        label.text((510, 195), "detector crop", fill="red")
        label.text((700, 195), "GT crop", fill="green")
        label.text(
            (510, 235),
            (
                f"{row['sample_id']} {row['level']}\n"
                f"target={row['target']} predicted={row['predicted']}\n"
                f"confidence={row['confidence']:.6f} quality={row['quality']:.6f}\n"
                f"match IoU={row['iou']:.4f}"
            ),
            fill="black",
            spacing=8,
        )
        sheet.paste(cell, (0, position * cell_height))
    args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.contact_sheet, quality=94)
    sweep = []
    for threshold_value in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        tp = sum(score >= threshold_value and label for score, label in ambiguity_rows)
        fn = sum(score < threshold_value and label for score, label in ambiguity_rows)
        fp = sum(score >= threshold_value and not label for score, label in ambiguity_rows)
        sweep.append(
            {
                "threshold": threshold_value,
                "severe_ambiguity_recall": tp / max(tp + fn, 1),
                "normal_rejected": fp,
                "severe_count": tp + fn,
            }
        )
    print(
        json.dumps(
            {"count": len(errors), "errors": errors, "ambiguity_sweep": sweep},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
