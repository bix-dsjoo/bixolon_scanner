from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .detector import _metrics_grid, select_release_threshold_candidate
from .onnx_detector import load_records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.dataset_root.resolve()
    records = load_records(root, "multi_object_instances.json")
    predictions = _read_jsonl(args.multi_predictions)
    candidates = _metrics_grid(
        records,
        predictions,
        score_thresholds=np.linspace(
            args.min_score_threshold, args.max_score_threshold, args.threshold_steps
        ),
        nms_iou_threshold=args.nms_threshold,
        match_iou_threshold=args.match_iou_threshold,
        max_queries=args.max_queries,
        max_object_aspect_ratio=args.max_object_aspect_ratio,
    )
    selected = select_release_threshold_candidate(candidates, args.target_recall)
    report = {
        "evaluation": "combined_annotated_detector_threshold_selection",
        "selection_set": "multi_object_development",
        "threshold_policy": "recall_floor_then_precision_on_development_set",
        "target_recall": args.target_recall,
        "target_recall_satisfied": selected["recall"] >= args.target_recall,
        "selected_score_threshold": selected["score_threshold"],
        "metrics": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a combined annotated detector threshold")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--multi-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nms-threshold", type=float, default=0.7)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--max-object-aspect-ratio", type=float, default=5.0)
    parser.add_argument("--target-recall", type=float, default=0.99)
    parser.add_argument("--min-score-threshold", type=float, default=0.01)
    parser.add_argument("--max-score-threshold", type=float, default=0.99)
    parser.add_argument("--threshold-steps", type=int, default=197)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
