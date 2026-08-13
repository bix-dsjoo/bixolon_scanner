from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..training.calibration import (
    binomial_rate_upper_bound,
    fit_temperature,
    select_approval_threshold,
    softmax,
    topk_accuracy,
)


def wilson_interval(
    successes: int, count: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if count == 0:
        return 0.0, 1.0
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    margin = z * np.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count)) / denominator
    return max(0.0, float(center - margin)), min(1.0, float(center + margin))


def evaluate_predictions(
    paths: list[Path], calibration: dict[str, object] | None = None
) -> dict[str, object]:
    archives = [np.load(path) for path in paths]
    logits = np.concatenate([archive["logits"] for archive in archives])
    targets = np.concatenate([archive["targets"] for archive in archives])
    temperature = (
        float(calibration["temperature"]) if calibration else fit_temperature(logits, targets)
    )
    probabilities = softmax(logits, temperature)
    threshold = select_approval_threshold(probabilities, targets) if calibration is None else None
    approval_threshold = (
        float(calibration["approval_threshold"]) if calibration else threshold.threshold
    )
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    approved = confidence >= approval_threshold
    approved_correct = int((predictions[approved] == targets[approved]).sum())
    approved_count = int(approved.sum())
    unknown = ~approved
    unknown_top3 = (
        topk_accuracy(probabilities[unknown], targets[unknown]) if unknown.any() else None
    )
    precision_interval = wilson_interval(approved_correct, approved_count)
    approved_precision = approved_correct / approved_count if approved_count else 1.0
    false_rate_upper = binomial_rate_upper_bound(approved_count - approved_correct, approved_count)
    return {
        "sample_count": int(len(targets)),
        "temperature": temperature,
        "approval_threshold": approval_threshold,
        "approved_count": approved_count,
        "approval_coverage": approved_count / len(targets),
        "approved_precision": approved_precision,
        "approved_precision_95ci": list(precision_interval),
        "approved_false_rate": 1.0 - approved_precision,
        "approved_false_rate_upper_95": false_rate_upper,
        "risk_control_satisfied": (
            bool(calibration["risk_control_satisfied"])
            if calibration
            else threshold.risk_control_satisfied
        ),
        "fixed_calibration": calibration is not None,
        "approved_precision_gate_satisfied": approved_precision >= 0.995,
        "unknown_count": int(unknown.sum()),
        "unknown_top3_accuracy": unknown_top3,
        "overall_top1_accuracy": float((predictions == targets).mean()),
        "overall_top3_accuracy": topk_accuracy(probabilities, targets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate and evaluate classifier prediction archives"
    )
    parser.add_argument("--predictions", type=Path, required=True, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument("--dataset-metadata", type=Path)
    parser.add_argument("--model-version", default="0.1.0")
    args = parser.parse_args()
    calibration = (
        json.loads(args.calibration_report.read_text(encoding="utf-8"))
        if args.calibration_report
        else None
    )
    report = evaluate_predictions(args.predictions, calibration)
    if args.dataset_metadata is not None:
        metadata = json.loads(args.dataset_metadata.read_text(encoding="utf-8"))
        report["dataset_version"] = metadata["dataset_version"]
    report["model_version"] = args.model_version
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
