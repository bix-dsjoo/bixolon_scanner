from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...evaluation.detected_roi_dataset import match_detections
from ...pipeline.ports import Detection
from ...runtime.onnx import nms
from ...training.data import read_manifest


def selected_detections(
    prediction: dict[str, Any],
    *,
    score_threshold: float,
    nms_iou_threshold: float,
    nms_containment_threshold: float,
    maximum_aspect_ratio: float,
) -> list[Detection]:
    candidates = []
    for box, score in zip(prediction["boxes_xyxy"], prediction["scores"]):
        x1, y1, x2, y2 = [float(value) for value in box]
        width = x2 - x1
        height = y2 - y1
        if (
            float(score) >= score_threshold
            and width > 0.0
            and height > 0.0
            and max(width / height, height / width) <= maximum_aspect_ratio
        ):
            candidates.append(Detection(x1, y1, x2, y2, float(score), 0))
    return sorted(
        nms(
            candidates,
            nms_iou_threshold,
            nms_containment_threshold,
            False,
        ),
        key=lambda detection: (detection.y1, detection.x1),
    )


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    records = read_manifest(args.manifest)
    records_by_id = {int(row["image_id"]): row for row in records}
    predictions_by_id: dict[int, dict[str, Any]] = {}
    fold_reports = {}
    for fold, path in enumerate(args.predictions):
        fold_predictions = {int(row["image_id"]): row for row in read_manifest(path)}
        expected = {int(row["image_id"]) for row in records if int(row["fold"]) == fold}
        if set(fold_predictions) != expected:
            raise ValueError(f"fold {fold} predictions do not match held-out manifest rows")
        if predictions_by_id.keys() & fold_predictions.keys():
            raise ValueError("OOF detector predictions overlap")
        predictions_by_id.update(fold_predictions)
        fold_reports[str(fold)] = {"image_count": len(expected)}
    if set(predictions_by_id) != set(records_by_id):
        raise ValueError("OOF detector predictions do not cover the full manifest")

    selected_rows = []
    error_rows = []
    recaptured_ids = []
    accepted_gt = accepted_predictions = accepted_matches = 0
    for record in records:
        image_id = int(record["image_id"])
        detections = selected_detections(
            predictions_by_id[image_id],
            score_threshold=args.score_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            nms_containment_threshold=args.nms_containment_threshold,
            maximum_aspect_ratio=args.maximum_aspect_ratio,
        )
        if not detections:
            recaptured_ids.append(image_id)
        else:
            matches = match_detections(
                detections,
                record["annotations"],
                match_iou_threshold=args.match_iou_threshold,
            )
            false_positive_count = len(detections) - len(matches)
            false_negative_count = len(record["annotations"]) - len(matches)
            accepted_gt += len(record["annotations"])
            accepted_predictions += len(detections)
            accepted_matches += len(matches)
            if false_positive_count or false_negative_count:
                error_rows.append(
                    {
                        "image_id": image_id,
                        "fold": int(record["fold"]),
                        "false_positive_count": false_positive_count,
                        "false_negative_count": false_negative_count,
                    }
                )
        selected_rows.append(
            {
                "image_id": image_id,
                "fold": int(record["fold"]),
                "boxes_xyxy": [[item.x1, item.y1, item.x2, item.y2] for item in detections],
                "scores": [item.score for item in detections],
                "class_ids": [0] * len(detections),
                "image_recapture": not detections,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selected-predictions.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected_rows),
        encoding="utf-8",
    )
    false_positive_count = accepted_predictions - accepted_matches
    false_negative_count = accepted_gt - accepted_matches
    report = {
        "schema_version": "1.0",
        "evaluation": "three-fold-product-independent-yolo-objectness-oof",
        "image_count": len(records),
        "ground_truth_count": sum(len(row["annotations"]) for row in records),
        "policy": {
            "score_threshold": args.score_threshold,
            "nms_iou_threshold": args.nms_iou_threshold,
            "nms_containment_threshold": args.nms_containment_threshold,
            "maximum_aspect_ratio": args.maximum_aspect_ratio,
            "match_iou_threshold": args.match_iou_threshold,
            "empty_detection_action": "IMAGE_RECAPTURE",
        },
        "accepted": {
            "image_count": len(records) - len(recaptured_ids),
            "ground_truth_count": accepted_gt,
            "prediction_count": accepted_predictions,
            "matched_count": accepted_matches,
            "false_positive_count": false_positive_count,
            "false_negative_count": false_negative_count,
        },
        "image_recapture": {
            "count": len(recaptured_ids),
            "rate": len(recaptured_ids) / len(records),
            "image_ids": recaptured_ids,
        },
        "error_rows": error_rows,
        "folds": fold_reports,
        "architecture_contract": "one-class-product-independent-objectness",
        "product_labels_used": False,
        "product_pair_rules_used": False,
        "object_count_exception_rules_used": False,
        "difficulty_rules_used": False,
        "image_specific_rules_used": False,
        "independent_test_claimed": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate locked YOLO objectness OOF results")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.65)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.4)
    parser.add_argument("--nms-containment-threshold", type=float, default=0.8)
    parser.add_argument("--maximum-aspect-ratio", type=float, default=20.0)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    aggregate(parser.parse_args(argv))


if __name__ == "__main__":
    main()
