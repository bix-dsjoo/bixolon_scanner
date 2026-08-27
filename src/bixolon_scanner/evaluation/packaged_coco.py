from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..contracts.catalog import sha256_file
from ..pipeline.ports import Detection
from ..runtime.onnx import box_iou


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("packaged COCO evidence is empty")
    return rows


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _box(values: list[float] | dict[str, float]) -> Detection:
    if isinstance(values, dict):
        x = float(values["x"])
        y = float(values["y"])
        width = float(values["width"])
        height = float(values["height"])
    else:
        x, y, width, height = (float(value) for value in values)
    return Detection(x, y, x + width, y + height, 1.0)


def _match(
    predictions: list[Detection], targets: list[Detection], threshold: float
) -> dict[int, int]:
    candidates = sorted(
        (
            (box_iou(prediction, target), prediction_index, target_index)
            for prediction_index, prediction in enumerate(predictions)
            for target_index, target in enumerate(targets)
        ),
        reverse=True,
    )
    matches: dict[int, int] = {}
    used_targets: set[int] = set()
    for overlap, prediction_index, target_index in candidates:
        if overlap < threshold:
            break
        if prediction_index not in matches and target_index not in used_targets:
            matches[prediction_index] = target_index
            used_targets.add(target_index)
    return matches


def evaluate_packaged_coco(
    manifest_path: Path,
    trace_path: Path,
    *,
    match_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    manifest_values = _jsonl(manifest_path)
    trace_values = _jsonl(trace_path)
    manifest_rows = {int(row["image_id"]): row for row in manifest_values}
    trace_rows = {int(row["image_id"]): row for row in trace_values}
    if len(manifest_rows) != len(manifest_values) or len(trace_rows) != len(trace_values):
        raise ValueError("packaged COCO evidence contains duplicate image IDs")
    if set(manifest_rows) != set(trace_rows):
        raise ValueError("packaged COCO manifest and trace image IDs differ")

    counts = {
        "image_count": len(manifest_rows),
        "empty_image_count": 0,
        "nonempty_image_count": 0,
        "empty_image_recapture_count": 0,
        "nonempty_image_recapture_count": 0,
        "ground_truth_count": 0,
        "prediction_count": 0,
        "matched_count": 0,
        "false_negative_count": 0,
        "false_positive_count": 0,
        "approved_count": 0,
        "approved_correct_count": 0,
        "approved_wrong_count": 0,
        "unknown_count": 0,
        "unknown_top3_hit_count": 0,
        "unknown_top3_miss_count": 0,
        "segment_recapture_count": 0,
    }
    for image_id in sorted(manifest_rows):
        record = manifest_rows[image_id]
        response = trace_rows[image_id]["response"]
        annotations = record.get("annotations", [])
        counts["ground_truth_count"] += len(annotations)
        empty = not annotations
        counts["empty_image_count" if empty else "nonempty_image_count"] += 1
        if response["status"] == "IMAGE_RECAPTURE":
            counts[
                "empty_image_recapture_count" if empty else "nonempty_image_recapture_count"
            ] += 1
            continue
        if response["status"] != "SEGMENTATION":
            raise ValueError("packaged COCO trace contains an unsupported response status")

        segmentations = response["segmentations"]
        predictions = [_box(segmentation["bbox"]) for segmentation in segmentations]
        targets = [
            _box(annotation.get("bbox_xywh", annotation.get("bbox"))) for annotation in annotations
        ]
        matches = _match(predictions, targets, match_iou_threshold)
        counts["prediction_count"] += len(predictions)
        counts["matched_count"] += len(matches)
        counts["false_negative_count"] += len(targets) - len(matches)
        counts["false_positive_count"] += len(predictions) - len(matches)
        for prediction_index, segmentation in enumerate(segmentations):
            status = segmentation["status"]
            if status == "APPROVED":
                counts["approved_count"] += 1
            elif status == "UNKNOWN":
                counts["unknown_count"] += 1
            elif status == "SEGMENT_RECAPTURE":
                counts["segment_recapture_count"] += 1
            else:
                raise ValueError("packaged COCO trace contains an unsupported item status")
            target_index = matches.get(prediction_index)
            if target_index is None:
                continue
            expected = f"bread_{int(annotations[target_index]['category_id']):02d}"
            if status == "APPROVED":
                correct = segmentation["prediction"]["class_id"] == expected
                counts["approved_correct_count"] += int(correct)
                counts["approved_wrong_count"] += int(not correct)
            elif status == "UNKNOWN":
                hit = expected in {candidate["class_id"] for candidate in segmentation["top3"]}
                counts["unknown_top3_hit_count"] += int(hit)
                counts["unknown_top3_miss_count"] += int(not hit)

    ground_truth_count = counts["ground_truth_count"]
    return {
        "schema_version": "1.0",
        "evaluation": "packaged_worker_coco_outcomes",
        "manifest": {
            "path": manifest_path.resolve().as_posix(),
            "sha256": sha256_file(manifest_path),
        },
        "trace": {
            "path": trace_path.resolve().as_posix(),
            "sha256": sha256_file(trace_path),
        },
        "match_iou_threshold": match_iou_threshold,
        "counts": counts,
        "rates": {
            "matched_ground_truth_rate": _rate(counts["matched_count"], ground_truth_count),
            "false_positive_per_ground_truth": _rate(
                counts["false_positive_count"], ground_truth_count
            ),
            "approved_accuracy": _rate(counts["approved_correct_count"], counts["approved_count"]),
            "unknown_top3_accuracy": _rate(
                counts["unknown_top3_hit_count"], counts["unknown_count"]
            ),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score a packaged Worker trace against COCO GT")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    report = evaluate_packaged_coco(
        args.manifest,
        args.trace,
        match_iou_threshold=args.match_iou_threshold,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
