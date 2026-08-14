from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from .classifier_selective_risk_probe import _softmax, _top3_candidates


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    archive = np.load(args.logits)
    logits = {name: archive[name].astype(np.float32) for name in archive.files if name != "targets"}
    targets = archive["targets"].astype(np.int64)
    matched = targets >= 0
    probabilities = {name: _softmax(values) for name, values in logits.items()}
    candidates = []
    for size in range(1, len(logits) + 1):
        for names in combinations(logits, size):
            for aggregation, values in _top3_candidates(logits, probabilities, names).items():
                top3 = np.argsort(-values, axis=1, kind="stable")[:, :3]
                correct = int(np.count_nonzero(matched & np.any(top3 == targets[:, None], axis=1)))
                candidates.append(
                    {
                        "views": list(names),
                        "aggregation": aggregation,
                        "correct": correct,
                        "matched_top3_accuracy": correct / int(np.count_nonzero(matched)),
                        "overall_recognition_ceiling": correct / args.ground_truth_count,
                    }
                )
    candidates.sort(
        key=lambda row: (row["correct"], -len(row["views"]), row["aggregation"]),
        reverse=True,
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "classifier_top3_aggregation_probe",
        "ground_truth_count": args.ground_truth_count,
        "matched_count": int(np.count_nonzero(matched)),
        "best": candidates[0],
        "top_candidates": candidates[:20],
        "passes_recognition_ceiling": (
            candidates[0]["overall_recognition_ceiling"] >= args.minimum_recognition
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe multi-view Top-3 aggregation ceilings")
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground-truth-count", type=int, required=True)
    parser.add_argument("--minimum-recognition", type=float, default=0.99)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
