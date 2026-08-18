from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...evaluation.detector import _iou, _xywh_to_xyxy
from ...training.data import read_manifest


def union_coverage(
    records: list[dict[str, Any]],
    prediction_sets: list[dict[str, dict[str, Any]]],
    *,
    score_threshold: float,
    match_iou_threshold: float,
) -> dict[str, Any]:
    misses = []
    ground_truth_count = 0
    for record in records:
        image_id = str(record["image_id"])
        boxes = [
            np.asarray(box, dtype=np.float32)
            for predictions in prediction_sets
            for box, score in zip(
                predictions[image_id]["boxes_xyxy"],
                predictions[image_id]["scores"],
            )
            if score >= score_threshold
        ]
        for annotation in record["annotations"]:
            ground_truth_count += 1
            target = _xywh_to_xyxy(annotation["bbox_xywh"])
            best_iou = max((_iou(box, target) for box in boxes), default=0.0)
            if best_iou < match_iou_threshold:
                misses.append(
                    {
                        "image_id": int(record["image_id"]),
                        "annotation_id": annotation.get("annotation_id"),
                        "category_id": int(annotation["category_id"]),
                        "best_iou": best_iou,
                    }
                )
    return {
        "ground_truth_count": ground_truth_count,
        "covered_count": ground_truth_count - len(misses),
        "miss_count": len(misses),
        "recall_upper_bound": (
            (ground_truth_count - len(misses)) / ground_truth_count if ground_truth_count else 0.0
        ),
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure raw GT coverage available from a detector prediction union"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.01)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
    prediction_sets = [
        {
            str(row["image_id"]): row
            for row in (
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
            )
        }
        for path in args.predictions
    ]
    expected_ids = {str(row["image_id"]) for row in records}
    coverage_matches = all(
        set(predictions) == expected_ids
        if difficulties is None
        else expected_ids.issubset(predictions)
        for predictions in prediction_sets
    )
    if not coverage_matches:
        raise ValueError("union prediction coverage differs from selected manifest rows")
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_detector_raw_union_coverage",
        "folds": sorted(folds),
        "difficulties": sorted(difficulties) if difficulties is not None else None,
        "prediction_files": [path.name for path in args.predictions],
        "score_threshold": args.score_threshold,
        "match_iou_threshold": args.match_iou_threshold,
        **union_coverage(
            records,
            prediction_sets,
            score_threshold=args.score_threshold,
            match_iou_threshold=args.match_iou_threshold,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
