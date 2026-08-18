from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from ...evaluation.detector import _metrics, detection_error_rows
from ...training.data import read_manifest
from .proposal_classifier_verifier import _load_predictions
from .proposal_ranker import select_ranked_predictions


def combine_ranked_predictions(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    left_weight: float,
    mode: str,
) -> list[dict[str, Any]]:
    if not 0.0 <= left_weight <= 1.0:
        raise ValueError("ensemble weight must be in [0, 1]")
    if len(left) != len(right):
        raise ValueError("ranked prediction image counts differ")
    outputs = []
    for left_row, right_row in zip(left, right):
        if left_row["image_id"] != right_row["image_id"]:
            raise ValueError("ranked prediction image order differs")
        left_boxes = np.asarray(left_row["boxes_xyxy"], dtype=np.float32)
        right_boxes = np.asarray(right_row["boxes_xyxy"], dtype=np.float32)
        if left_boxes.shape != right_boxes.shape or not np.allclose(
            left_boxes, right_boxes, atol=1e-4
        ):
            raise ValueError("ranked proposal boxes differ")
        if left_row["class_ids"] != right_row["class_ids"]:
            raise ValueError("ranked proposal classes differ")
        left_scores = np.asarray(left_row["scores"], dtype=np.float64)
        right_scores = np.asarray(right_row["scores"], dtype=np.float64)
        if mode == "arithmetic":
            scores = left_weight * left_scores + (1.0 - left_weight) * right_scores
        elif mode == "geometric":
            scores = np.exp(
                left_weight * np.log(left_scores.clip(1e-9))
                + (1.0 - left_weight) * np.log(right_scores.clip(1e-9))
            )
        elif mode == "minimum":
            scores = np.minimum(left_scores, right_scores)
        else:
            raise ValueError(f"unsupported ensemble mode: {mode}")
        outputs.append({**left_row, "scores": scores.tolist()})
    return outputs


def _read_ranked(path: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _load_predictions(path, records)


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
    left = _read_ranked(args.left, records)
    right = _read_ranked(args.right, records)
    candidates = []
    ranked_cache: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for mode, weight in product(args.modes, args.left_weights):
        cache_key = (mode, weight if mode != "minimum" else 0.0)
        if cache_key not in ranked_cache:
            ranked_cache[cache_key] = combine_ranked_predictions(
                left,
                right,
                left_weight=weight,
                mode=mode,
            )
        ranked = ranked_cache[cache_key]
        for score_threshold, nms_threshold in product(args.score_thresholds, args.nms_thresholds):
            predictions = select_ranked_predictions(
                ranked,
                score_threshold=score_threshold,
                nms_iou_threshold=nms_threshold,
            )
            metrics = _metrics(
                records,
                predictions,
                score_threshold=0.0,
                nms_iou_threshold=1.0,
                match_iou_threshold=0.5,
                max_queries=600,
            )
            candidates.append(
                {
                    "mode": mode,
                    "left_weight": weight,
                    "score_threshold": score_threshold,
                    "nms_iou_threshold": nms_threshold,
                    "metrics": metrics,
                }
            )
    selected = max(
        candidates,
        key=lambda row: (
            -row["metrics"]["false_positive_count"] - row["metrics"]["false_negative_count"],
            row["metrics"]["exact_image_rate"],
            -row["metrics"]["false_negative_count"],
            row["score_threshold"],
        ),
    )
    cache_key = (
        selected["mode"],
        selected["left_weight"] if selected["mode"] != "minimum" else 0.0,
    )
    selected_predictions = select_ranked_predictions(
        ranked_cache[cache_key],
        score_threshold=selected["score_threshold"],
        nms_iou_threshold=selected["nms_iou_threshold"],
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_oof_proposal_score_ensemble",
        "folds": sorted(folds),
        "left": args.left.name,
        "right": args.right.name,
        "candidate_count": len(candidates),
        "zero_error_candidate_count": sum(
            row["metrics"]["false_positive_count"] == 0
            and row["metrics"]["false_negative_count"] == 0
            for row in candidates
        ),
        "selected": selected,
        "error_images": detection_error_rows(
            records,
            selected_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_output.write_text(
            "".join(json.dumps(row) + "\n" for row in selected_predictions),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble two OOF proposal scores")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument(
        "--modes",
        choices=["arithmetic", "geometric", "minimum"],
        nargs="+",
        required=True,
    )
    parser.add_argument("--left-weights", type=float, nargs="+", required=True)
    parser.add_argument("--score-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--nms-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
