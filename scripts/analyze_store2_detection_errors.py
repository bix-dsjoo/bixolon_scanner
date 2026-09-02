from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bixolon_scanner.training.models import require_torch


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _boxes(row: dict):
    torch = require_torch()
    return torch.as_tensor(row["boxes_xyxy"], dtype=torch.float32).reshape(-1, 4)


def _best_overlap(query, boxes, scores, classes=None) -> dict | None:
    from torchvision.ops import box_iou

    if not len(boxes):
        return None
    overlaps = box_iou(query.reshape(1, 4), boxes)[0]
    index = int(overlaps.argmax())
    result = {
        "iou": float(overlaps[index]),
        "score": float(scores[index]),
        "box": [float(value) for value in boxes[index]],
    }
    if classes is not None:
        result["class_id"] = int(classes[index])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Store 2 saved detector errors")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-predictions", type=Path, required=True)
    parser.add_argument("--class-aware-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.735)
    parser.add_argument("--nms-threshold", type=float, default=0.4)
    args = parser.parse_args()

    torch = require_torch()
    from torchvision.ops import box_iou, nms

    records = _jsonl(args.manifest)
    primary = {int(row["image_id"]): row for row in _jsonl(args.primary_predictions)}
    class_aware = {int(row["image_id"]): row for row in _jsonl(args.class_aware_predictions)}
    false_positives = []
    false_negatives = []
    matched = 0
    for record in records:
        image_id = int(record["image_id"])
        raw = primary[image_id]
        raw_boxes = _boxes(raw)
        raw_scores = torch.as_tensor(raw["scores"], dtype=torch.float32)
        eligible = raw_scores >= args.score_threshold
        boxes, scores = raw_boxes[eligible], raw_scores[eligible]
        if len(boxes):
            keep = nms(boxes, scores, args.nms_threshold)
            boxes, scores = boxes[keep], scores[keep]
        annotations = record["annotations"]
        gt_boxes = torch.as_tensor(
            [
                [
                    row["bbox_xywh"][0],
                    row["bbox_xywh"][1],
                    row["bbox_xywh"][0] + row["bbox_xywh"][2],
                    row["bbox_xywh"][1] + row["bbox_xywh"][3],
                ]
                for row in annotations
            ],
            dtype=torch.float32,
        ).reshape(-1, 4)
        overlaps = box_iou(boxes, gt_boxes) if len(boxes) and len(gt_boxes) else None
        unmatched = set(range(len(annotations)))
        unmatched_predictions = []
        for prediction_index in range(len(boxes)):
            if not unmatched:
                unmatched_predictions.extend(range(prediction_index, len(boxes)))
                break
            best = max(unmatched, key=lambda index: float(overlaps[prediction_index, index]))
            if float(overlaps[prediction_index, best]) < 0.5:
                unmatched_predictions.append(prediction_index)
                continue
            unmatched.remove(best)
            matched += 1

        secondary = class_aware[image_id]
        secondary_boxes = _boxes(secondary)
        secondary_scores = torch.as_tensor(secondary["scores"], dtype=torch.float32)
        secondary_classes = torch.as_tensor(secondary["class_ids"], dtype=torch.int64)
        for prediction_index in unmatched_predictions:
            query = boxes[prediction_index]
            false_positives.append(
                {
                    "image_id": image_id,
                    "box": [float(value) for value in query],
                    "score": float(scores[prediction_index]),
                    "maximum_gt_iou": (
                        float(box_iou(query.reshape(1, 4), gt_boxes).max())
                        if len(gt_boxes)
                        else 0.0
                    ),
                    "class_aware_best": _best_overlap(
                        query,
                        secondary_boxes,
                        secondary_scores,
                        secondary_classes,
                    ),
                }
            )
        for gt_index in sorted(unmatched):
            query = gt_boxes[gt_index]
            false_negatives.append(
                {
                    "image_id": image_id,
                    "annotation_id": int(annotations[gt_index]["annotation_id"]),
                    "category_id": int(annotations[gt_index]["category_id"]),
                    "box": [float(value) for value in query],
                    "primary_best_all_scores": _best_overlap(query, raw_boxes, raw_scores),
                    "class_aware_best": _best_overlap(
                        query,
                        secondary_boxes,
                        secondary_scores,
                        secondary_classes,
                    ),
                }
            )
    report = {
        "schema_version": "1.0",
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "matched_count": matched,
        "false_negative_count": len(false_negatives),
        "false_positive_count": len(false_positives),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "false_positive_score_summary": {
            "minimum": float(min(row["score"] for row in false_positives)),
            "median": float(np.median([row["score"] for row in false_positives])),
            "maximum": float(max(row["score"] for row in false_positives)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
