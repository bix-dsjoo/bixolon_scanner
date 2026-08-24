from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ...contracts.catalog import sha256_file
from ...evaluation.detector import _metrics_grid, detection_error_rows


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def search(args: argparse.Namespace) -> dict:
    records = _jsonl(args.manifest)
    predictions_by_id = {int(row["image_id"]): row for row in _jsonl(args.predictions)}
    if len(records) != args.expected_image_count:
        raise ValueError(f"expected {args.expected_image_count} records, observed {len(records)}")
    if set(predictions_by_id) != {int(row["image_id"]) for row in records}:
        raise ValueError("detector predictions do not align with the manifest")
    predictions = [predictions_by_id[int(row["image_id"])] for row in records]
    thresholds = np.linspace(
        args.minimum_score,
        args.maximum_score,
        args.score_steps,
        dtype=np.float64,
    )
    candidates = []
    for iou_threshold in args.nms_iou_thresholds:
        for containment_threshold in args.containment_thresholds:
            for class_aware in (False, True):
                rows = _metrics_grid(
                    records,
                    predictions,
                    score_thresholds=thresholds,
                    nms_iou_threshold=iou_threshold,
                    match_iou_threshold=args.match_iou_threshold,
                    max_queries=args.max_queries,
                    max_object_aspect_ratio=args.max_object_aspect_ratio,
                    nms_containment_threshold=containment_threshold,
                    nms_class_aware_containment=class_aware,
                )
                for row in rows:
                    row.update(
                        {
                            "nms_iou_threshold": iou_threshold,
                            "nms_containment_threshold": containment_threshold,
                            "nms_class_aware_containment": class_aware,
                        }
                    )
                    candidates.append(row)
    candidates.sort(
        key=lambda row: (
            int(row["false_negative_count"]) + int(row["false_positive_count"]),
            int(row["false_negative_count"]),
            -int(row["exact_image_count"]),
            -float(row["score_threshold"]),
        )
    )
    best = candidates[0]
    errors = detection_error_rows(
        records,
        predictions,
        score_threshold=float(best["score_threshold"]),
        nms_iou_threshold=float(best["nms_iou_threshold"]),
        match_iou_threshold=args.match_iou_threshold,
        max_object_aspect_ratio=args.max_object_aspect_ratio,
        nms_containment_threshold=float(best["nms_containment_threshold"]),
        nms_class_aware_containment=bool(best["nms_class_aware_containment"]),
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "generic_detector_postprocess_grid",
        "manifest_sha256": sha256_file(args.manifest),
        "predictions_sha256": sha256_file(args.predictions),
        "image_count": len(records),
        "search_space": {
            "score_thresholds": [float(value) for value in thresholds],
            "nms_iou_thresholds": args.nms_iou_thresholds,
            "containment_thresholds": args.containment_thresholds,
            "class_aware_containment": [False, True],
            "max_object_aspect_ratio": args.max_object_aspect_ratio,
        },
        "best": best,
        "best_error_images": errors,
        "top_candidates": candidates[: args.report_top_k],
        "development_only": True,
        "independent_test_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Search generic detector threshold and NMS postprocessing"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-image-count", type=int, default=415)
    parser.add_argument("--minimum-score", type=float, default=0.30)
    parser.add_argument("--maximum-score", type=float, default=0.70)
    parser.add_argument("--score-steps", type=int, default=161)
    parser.add_argument("--nms-iou-thresholds", type=float, nargs="+", default=[0.4, 0.5, 0.6])
    parser.add_argument(
        "--containment-thresholds",
        type=float,
        nargs="+",
        default=[0.75, 0.8, 0.85, 0.9, 0.95],
    )
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-object-aspect-ratio", type=float, default=5.0)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--report-top-k", type=int, default=20)
    search(parser.parse_args(argv))


if __name__ == "__main__":
    main()
