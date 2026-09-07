from __future__ import annotations

import argparse
import json
from pathlib import Path

from bixolon_scanner.evaluation.detector import _metrics_grid
from bixolon_scanner.evaluation.onnx_detector import (
    _fuse_rotation_predictions,
    load_records,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fuse_sources(
    sources: list[list[dict]], *, support_iou: float, minimum_support: int
) -> list[dict]:
    fused_rows = []
    for grouped in zip(*sources, strict=True):
        image_ids = {int(row["image_id"]) for row in grouped}
        if len(image_ids) != 1:
            raise ValueError("detector prediction sources disagree on image ordering")
        combined = {key: [] for key in ("boxes_xyxy", "scores", "class_ids", "top3_class_ids")}
        view_ids = []
        for source_index, row in enumerate(grouped):
            view_ids.extend([source_index] * len(row["scores"]))
            for key in combined:
                combined[key].extend(row[key])
        fused = _fuse_rotation_predictions(
            combined,
            view_ids,
            view_count=len(sources),
            minimum_support=minimum_support,
            iou_threshold=support_iou,
            score_mode="max",
        )
        fused["image_id"] = image_ids.pop()
        fused_rows.append(fused)
    return fused_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detector prediction consensus")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotation", default="multi_object_instances.json")
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-iou", type=float, nargs="+", required=True)
    parser.add_argument("--minimum-support", type=int)
    parser.add_argument("--nms-threshold", type=float, default=0.35)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-score-threshold", type=float, default=0.4)
    parser.add_argument("--max-score-threshold", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=56)
    args = parser.parse_args()

    records = load_records(args.dataset_root, args.annotation)
    sources = [_read_jsonl(path) for path in args.predictions]
    if any(len(source) != len(records) for source in sources):
        raise ValueError("prediction source count does not match evaluation records")
    minimum_support = args.minimum_support or len(sources)
    candidates = []
    for support_iou in args.support_iou:
        fused = _fuse_sources(
            sources,
            support_iou=support_iou,
            minimum_support=minimum_support,
        )
        if args.threshold_steps == 1:
            thresholds = [args.min_score_threshold]
        else:
            step = (args.max_score_threshold - args.min_score_threshold) / (
                args.threshold_steps - 1
            )
            thresholds = [
                args.min_score_threshold + index * step for index in range(args.threshold_steps)
            ]
        rows = _metrics_grid(
            records,
            fused,
            score_thresholds=thresholds,
            nms_iou_threshold=args.nms_threshold,
            match_iou_threshold=args.match_iou_threshold,
            max_queries=300,
            max_object_aspect_ratio=5.0,
        )
        candidates.extend(dict(row, support_iou=support_iou) for row in rows)
    ranked = sorted(
        candidates,
        key=lambda row: (
            row["false_positive_count"] + row["false_negative_count"],
            row["false_negative_count"],
            row["false_positive_count"],
        ),
    )
    report = {
        "evaluation": "detector_prediction_consensus",
        "prediction_sources": [path.as_posix() for path in args.predictions],
        "minimum_support": minimum_support,
        "best": ranked[0],
        "top_candidates": ranked[:25],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
