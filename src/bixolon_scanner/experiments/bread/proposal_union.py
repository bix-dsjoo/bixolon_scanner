from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...pipeline.ports import Detection
from ...runtime.onnx import nms
from ...training.data import read_manifest
from .proposal_classifier_verifier import _load_predictions


def filter_prediction(
    prediction: dict[str, Any],
    *,
    score_threshold: float,
    nms_iou_threshold: float,
    source_id: int,
) -> dict[str, Any]:
    candidates = [
        Detection(*box, float(score), int(class_id))
        for box, score, class_id in zip(
            prediction["boxes_xyxy"], prediction["scores"], prediction["class_ids"]
        )
        if score >= score_threshold
    ]
    selected = nms(candidates, nms_iou_threshold)
    return {
        "image_id": prediction["image_id"],
        "boxes_xyxy": [[item.x1, item.y1, item.x2, item.y2] for item in selected],
        "scores": [item.score for item in selected],
        "class_ids": [item.class_id for item in selected],
        "source_ids": [source_id] * len(selected),
    }


def union_predictions(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    left_score_threshold: float,
    left_nms_iou_threshold: float,
    right_score_threshold: float,
    right_nms_iou_threshold: float,
) -> list[dict[str, Any]]:
    if len(left) != len(right):
        raise ValueError("proposal source image counts differ")
    outputs = []
    for left_row, right_row in zip(left, right):
        if left_row["image_id"] != right_row["image_id"]:
            raise ValueError("proposal source image order differs")
        filtered_left = filter_prediction(
            left_row,
            score_threshold=left_score_threshold,
            nms_iou_threshold=left_nms_iou_threshold,
            source_id=0,
        )
        filtered_right = filter_prediction(
            right_row,
            score_threshold=right_score_threshold,
            nms_iou_threshold=right_nms_iou_threshold,
            source_id=1,
        )
        outputs.append(
            {
                "image_id": left_row["image_id"],
                "boxes_xyxy": filtered_left["boxes_xyxy"] + filtered_right["boxes_xyxy"],
                "scores": filtered_left["scores"] + filtered_right["scores"],
                "class_ids": filtered_left["class_ids"] + filtered_right["class_ids"],
                "source_ids": filtered_left["source_ids"] + filtered_right["source_ids"],
            }
        )
    return outputs


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    folds = set(args.folds)
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) in folds
        and row.get("expected_image_status") == "ANNOTATED"
    ]
    left = _load_predictions(args.left, records)
    right = _load_predictions(args.right, records)
    union = union_predictions(
        left,
        right,
        left_score_threshold=args.left_score_threshold,
        left_nms_iou_threshold=args.left_nms_iou_threshold,
        right_score_threshold=args.right_score_threshold,
        right_nms_iou_threshold=args.right_nms_iou_threshold,
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_oof_proposal_union",
        "folds": sorted(folds),
        "left": args.left.name,
        "right": args.right.name,
        "left_candidate_count": sum(
            sum(source == 0 for source in row["source_ids"]) for row in union
        ),
        "right_candidate_count": sum(
            sum(source == 1 for source in row["source_ids"]) for row in union
        ),
        "candidate_count": sum(len(row["scores"]) for row in union),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.write_text(
        "".join(json.dumps(row) + "\n" for row in union), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Union two OOF high-recall proposal sources")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-score-threshold", type=float, required=True)
    parser.add_argument("--left-nms-iou-threshold", type=float, required=True)
    parser.add_argument("--right-score-threshold", type=float, required=True)
    parser.add_argument("--right-nms-iou-threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
