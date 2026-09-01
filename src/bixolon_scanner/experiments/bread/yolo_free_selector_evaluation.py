from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ...runtime.proposal_selection import (
    box_iou,
    count_and_class_assisted_select,
)
from .hierarchical_detector import filter_predictions


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _coco_ground_truth(path: Path) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        x, y, width, height = annotation["bbox"]
        by_image[int(annotation["image_id"])].append(
            {
                "box": [x, y, x + width, y + height],
                "class_id": int(annotation["category_id"]) - 1,
            }
        )
    return payload["images"], by_image


def _geometry_metrics(
    images: list[dict[str, Any]],
    ground_truth: dict[int, list[dict[str, Any]]],
    predictions: dict[int, dict[str, Any]],
    match_iou: float,
) -> dict[str, Any]:
    matched_count = 0
    prediction_count = 0
    ground_truth_count = 0
    exact_images = 0
    error_images = []
    for image in images:
        image_id = int(image["id"])
        expected = ground_truth[image_id]
        prediction = predictions[image_id]
        boxes = prediction["boxes_xyxy"]
        prediction_count += len(boxes)
        ground_truth_count += len(expected)
        pairs = sorted(
            (
                (box_iou(box, target["box"]), prediction_index, target_index)
                for prediction_index, box in enumerate(boxes)
                for target_index, target in enumerate(expected)
            ),
            reverse=True,
        )
        used_predictions: set[int] = set()
        used_targets: set[int] = set()
        for overlap, prediction_index, target_index in pairs:
            if overlap < match_iou:
                break
            if prediction_index in used_predictions or target_index in used_targets:
                continue
            used_predictions.add(prediction_index)
            used_targets.add(target_index)
        matched_count += len(used_targets)
        false_positive = len(boxes) - len(used_predictions)
        false_negative = len(expected) - len(used_targets)
        if false_positive == 0 and false_negative == 0:
            exact_images += 1
        else:
            error_images.append(
                {
                    "image_id": image_id,
                    "file_name": image["file_name"],
                    "false_positive_count": false_positive,
                    "false_negative_count": false_negative,
                }
            )
    return {
        "image_count": len(images),
        "ground_truth_count": ground_truth_count,
        "prediction_count": prediction_count,
        "matched_count": matched_count,
        "false_positive_count": prediction_count - matched_count,
        "false_negative_count": ground_truth_count - matched_count,
        "exact_image_count": exact_images,
        "exact_image_rate": exact_images / len(images) if images else 0.0,
        "error_images": error_images,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    images, ground_truth = _coco_ground_truth(args.annotation_path)
    raw_rows = _read_jsonl(args.predictions)
    raw_by_id = {int(row["image_id"]): row for row in raw_rows}
    base_rows = filter_predictions(
        raw_rows,
        score_threshold=args.score_threshold,
        iou_threshold=args.nms_threshold,
        containment_threshold=args.containment_threshold,
        group_minimum=args.group_minimum,
    )
    base_by_id = {int(row["image_id"]): row for row in base_rows}
    final_by_id = dict(base_by_id)
    diagnostics = []

    if args.zero_count_recapture and args.count_predictions:
        zero_count_names = {
            Path(row["image_path"]).name
            for row in _read_jsonl(args.count_predictions)
            if int(row["predicted_count"]) == 0
        }
        for image in images:
            if Path(image["file_name"]).name in zero_count_names:
                image_id = int(image["id"])
                final_by_id[image_id] = {
                    "image_id": image_id,
                    "boxes_xyxy": [],
                    "scores": [],
                    "class_ids": [],
                }
                diagnostics.append(
                    {
                        "image_id": image_id,
                        "mode": "zero_count_recapture",
                        "base_count": len(base_by_id[image_id]["boxes_xyxy"]),
                        "target_count": 0,
                        "selected_count": 0,
                        "localization_refined": False,
                    }
                )

    if args.proposal_classifications and args.count_predictions:
        proposal_by_id = {
            int(row["image_id"]): row for row in _read_jsonl(args.proposal_classifications)
        }
        count_by_id = {
            int(row["image_id"]): int(row["predicted_count"])
            for row in _read_jsonl(args.count_predictions)
        }
        for image_id, proposal_row in proposal_by_id.items():
            target_count = count_by_id[image_id]
            base = base_by_id[image_id]
            raw = raw_by_id[image_id]
            strong_containment = any(
                left_index != right_index and box_iou(left, right) > 0.0
                for left_index, left in enumerate(base["boxes_xyxy"])
                for right_index, right in enumerate(base["boxes_xyxy"])
            )
            if len(base["boxes_xyxy"]) == target_count and not strong_containment:
                continue
            selected, row_diagnostics = count_and_class_assisted_select(
                base,
                raw,
                proposal_row["entries"],
                target_count,
            )
            final_by_id[image_id] = {"image_id": image_id, **selected}
            diagnostics.append({"image_id": image_id, **row_diagnostics})

    metrics = _geometry_metrics(images, ground_truth, final_by_id, args.match_iou_threshold)
    report = {
        "schema_version": "1.0",
        "evaluation": "yolo_free_count_and_class_assisted_selector",
        "selection_is_label_free": True,
        "parameters": {
            "score_threshold": args.score_threshold,
            "nms_threshold": args.nms_threshold,
            "containment_threshold": args.containment_threshold,
            "group_minimum": args.group_minimum,
            "match_iou_threshold": args.match_iou_threshold,
        },
        "metrics": metrics,
        "assisted_image_count": len(diagnostics),
        "assisted_diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_output.write_text(
            "".join(
                json.dumps(final_by_id[int(image["id"])], separators=(",", ":")) + "\n"
                for image in images
            ),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the YOLO-free count/class-assisted detector selector"
    )
    parser.add_argument("--annotation-path", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--proposal-classifications", type=Path)
    parser.add_argument("--count-predictions", type=Path)
    parser.add_argument("--zero-count-recapture", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.4)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--containment-threshold", type=float, default=0.85)
    parser.add_argument("--group-minimum", type=int, default=2)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
