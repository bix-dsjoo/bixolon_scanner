from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from bixolon_scanner.training.models import require_torch


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _truth_boxes(record: dict):
    torch = require_torch()
    return torch.as_tensor(
        [
            [
                row["bbox_xywh"][0],
                row["bbox_xywh"][1],
                row["bbox_xywh"][0] + row["bbox_xywh"][2],
                row["bbox_xywh"][1] + row["bbox_xywh"][3],
            ]
            for row in record["annotations"]
        ],
        dtype=torch.float32,
    ).reshape(-1, 4)


def _containment_keep(
    boxes,
    scores,
    *,
    minimum_containment: float,
    maximum_area_ratio: float,
):
    torch = require_torch()

    if len(boxes) < 2:
        return torch.ones(len(boxes), dtype=torch.bool)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    left_top = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    right_bottom = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(dim=-1)
    containment = intersection / area[:, None].clamp(min=1e-6)
    area_ratio = area[:, None] / area[None, :].clamp(min=1e-6)
    higher_or_equal = scores[None, :] >= scores[:, None]
    diagonal = torch.eye(len(boxes), dtype=torch.bool)
    suppressor = (
        (containment >= minimum_containment)
        & (area_ratio <= maximum_area_ratio)
        & higher_or_equal
        & ~diagonal
    )
    return ~suppressor.any(dim=1)


def _evaluate(records: list[dict], predictions: dict[int, dict], settings: tuple) -> dict:
    torch = require_torch()
    from torchvision.ops import box_iou

    score_threshold, minimum_containment, maximum_area_ratio = settings
    matched = 0
    false_positive = 0
    failures = []
    for record in records:
        row = predictions[int(record["image_id"])]
        boxes = torch.as_tensor(row["boxes_xyxy"], dtype=torch.float32).reshape(-1, 4)
        scores = torch.as_tensor(row["scores"], dtype=torch.float32)
        eligible = scores >= score_threshold
        boxes, scores = boxes[eligible], scores[eligible]
        keep = _containment_keep(
            boxes,
            scores,
            minimum_containment=minimum_containment,
            maximum_area_ratio=maximum_area_ratio,
        )
        boxes, scores = boxes[keep], scores[keep]
        order = scores.argsort(descending=True)
        boxes = boxes[order]
        truth = _truth_boxes(record)
        overlaps = box_iou(boxes, truth) if len(boxes) and len(truth) else None
        unmatched = set(range(len(truth)))
        unmatched_predictions = 0
        for index in range(len(boxes)):
            if not unmatched:
                unmatched_predictions += len(boxes) - index
                break
            best = max(unmatched, key=lambda target: float(overlaps[index, target]))
            if float(overlaps[index, best]) < 0.5:
                unmatched_predictions += 1
                continue
            unmatched.remove(best)
            matched += 1
        false_positive += unmatched_predictions
        if unmatched or unmatched_predictions:
            failures.append(
                {
                    "image_id": int(record["image_id"]),
                    "false_negative_count": len(unmatched),
                    "false_positive_count": unmatched_predictions,
                }
            )
    ground_truth_count = sum(len(row["annotations"]) for row in records)
    return {
        "score_threshold": score_threshold,
        "minimum_containment": minimum_containment,
        "maximum_area_ratio": maximum_area_ratio,
        "matched_count": matched,
        "false_negative_count": ground_truth_count - matched,
        "false_positive_count": false_positive,
        "total_error_count": ground_truth_count - matched + false_positive,
        "failure_image_count": len(failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generic contained-box removal")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = _jsonl(args.manifest)
    predictions = {int(row["image_id"]): row for row in _jsonl(args.predictions)}
    settings = itertools.product(
        (0.20, 0.25, 0.30, 0.32, 0.34, 0.40, 0.50, 0.60, 0.70, 0.735),
        (0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
        (0.35, 0.45, 0.55, 0.65, 0.75),
    )
    results = [_evaluate(records, predictions, setting) for setting in settings]
    results.sort(
        key=lambda row: (
            row["total_error_count"],
            row["false_positive_count"],
            row["false_negative_count"],
        )
    )
    best_by_score = []
    for score_threshold in sorted({row["score_threshold"] for row in results}):
        best_by_score.append(
            next(row for row in results if row["score_threshold"] == score_threshold)
        )
    report = {
        "schema_version": "1.0",
        "best_candidates": results[:50],
        "best_by_score_threshold": best_by_score,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results[:20], indent=2))


if __name__ == "__main__":
    main()
