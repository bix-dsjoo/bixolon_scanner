from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import softmax
from .rpc_data_scale import LEVELS, evaluate_worker_taxonomy


def _candidate_thresholds(confidence: np.ndarray) -> np.ndarray:
    # A coarse sweep establishes feasibility quickly.  Exact threshold selection
    # remains calibration-only and is performed separately after a recipe wins.
    quantiles = np.linspace(0.0, 1.0, 101)
    candidates = np.quantile(confidence, quantiles)
    candidates = np.concatenate([candidates, np.asarray([0.0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999])])
    return np.unique(np.clip(candidates, 0.0, 1.0))


def _minimum_level_value(report: dict[str, dict[str, Any]], key: str) -> float:
    return min(float(report[level][key]) for level in LEVELS)


def _maximum_level_value(report: dict[str, dict[str, Any]], key: str) -> float:
    return max(float(report[level][key]) for level in LEVELS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--max-misrecognition-rate", type=float, default=0.005)
    parser.add_argument("--target-recognition-rate", type=float, default=0.99)
    parser.add_argument("--write-json", type=Path)
    args = parser.parse_args()

    run_dir = args.output_dir / "runs" / "full" / f"seed{args.seed}"
    predictions_archive = np.load(run_dir / "selection_predictions.npz")
    predictions = {key: predictions_archive[key] for key in predictions_archive.files}
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    detector_report = json.loads(
        (args.output_dir / "prepared" / "worker_gate_report.json").read_text(encoding="utf-8")
    )
    confidence = softmax(predictions["logits"], float(calibration["temperature"])).max(axis=1)
    rows: list[dict[str, Any]] = []
    for threshold in _candidate_thresholds(confidence):
        candidate_calibration = dict(
            calibration,
            approval_threshold=float(threshold),
            risk_control_satisfied=True,
        )
        report = evaluate_worker_taxonomy(
            predictions,
            candidate_calibration,
            detector_report,
            role="selection",
        )
        rows.append(
            {
                "threshold": float(threshold),
                "minimum_recognition_rate": _minimum_level_value(report, "recognition_rate"),
                "maximum_misrecognition_rate": _maximum_level_value(report, "misrecognition_rate"),
                "minimum_end_to_end_success_rate": _minimum_level_value(
                    report, "end_to_end_success_rate"
                ),
                "difficulty": report,
            }
        )
    safe = [
        row
        for row in rows
        if row["maximum_misrecognition_rate"] <= float(args.max_misrecognition_rate)
    ]
    feasible = [
        row
        for row in safe
        if row["minimum_recognition_rate"] >= float(args.target_recognition_rate)
    ]
    best = max(
        safe,
        key=lambda row: (
            row["minimum_recognition_rate"],
            row["minimum_end_to_end_success_rate"],
            -row["maximum_misrecognition_rate"],
        ),
        default=None,
    )
    result = {
        "max_misrecognition_rate": float(args.max_misrecognition_rate),
        "target_recognition_rate": float(args.target_recognition_rate),
        "candidate_count": len(rows),
        "safe_candidate_count": len(safe),
        "feasible_candidate_count": len(feasible),
        "best_safe": best,
        "best_feasible": max(
            feasible,
            key=lambda row: (
                row["minimum_end_to_end_success_rate"],
                row["minimum_recognition_rate"],
                -row["maximum_misrecognition_rate"],
            ),
            default=None,
        ),
    }
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
