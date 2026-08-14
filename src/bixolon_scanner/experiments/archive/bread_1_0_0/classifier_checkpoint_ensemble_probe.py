from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path, prefix: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    values = np.load(path)
    return (
        {
            f"{prefix}:{name}": values[name].astype(np.float32)
            for name in values.files
            if name != "targets"
        },
        values["targets"].astype(np.int64),
    )


def _accuracy(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    return float((logits[mask].argmax(axis=1) == targets[mask]).mean())


def _greedy(
    candidates: dict[str, np.ndarray], targets: np.ndarray, mask: np.ndarray, steps: int
) -> dict[str, Any]:
    selected = []
    total = None
    history = []
    for step in range(steps):
        best = None
        for name, logits in candidates.items():
            combined = logits if total is None else total + logits
            accuracy = _accuracy(combined, targets, mask)
            if best is None or accuracy > best["accuracy"]:
                best = {"name": name, "accuracy": accuracy, "combined": combined}
        selected.append(best["name"])
        total = best["combined"]
        history.append(
            {"step": step + 1, "added": best["name"], "selection_accuracy": best["accuracy"]}
        )
    return {"members": selected, "history": history, "logits": total}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    left, targets = _load(args.left, "soup")
    right, right_targets = _load(args.right, "clutter_v2")
    if not np.array_equal(targets, right_targets):
        raise ValueError("checkpoint logits have different targets")
    candidates = {**left, **right}
    rows = [
        json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line
    ]
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    full = np.ones(len(targets), dtype=bool)
    selected = _greedy(candidates, targets, full, args.steps)
    full_accuracy = _accuracy(selected["logits"], targets, full)
    oof = np.zeros((len(targets), 20), dtype=np.float32)
    fold_reports = []
    for fold in range(3):
        training, held_out = folds != fold, folds == fold
        result = _greedy(candidates, targets, training, args.steps)
        oof[held_out] = result["logits"][held_out]
        fold_reports.append(
            {
                "held_out_fold": fold,
                "members": result["members"],
                "held_out_top1_accuracy": _accuracy(result["logits"], targets, held_out),
            }
        )
    per_candidate = np.stack([values.argmax(axis=1) for values in candidates.values()])
    oracle = np.any(per_candidate == targets[None, :], axis=0)
    report = {
        "schema_version": "1.0",
        "evaluation": "two_checkpoint_geometric_greedy_ensemble_probe",
        "candidate_count": len(candidates),
        "selected": {
            "members": selected["members"],
            "history": selected["history"],
            "top1_accuracy": full_accuracy,
        },
        "grouped_3fold_oof": {
            "top1_accuracy": _accuracy(oof, targets, full),
            "folds": fold_reports,
        },
        "candidate_oracle": {
            "top1_accuracy": float(oracle.mean()),
            "error_count": int(np.count_nonzero(~oracle)),
        },
        "passes_top1_gate": full_accuracy >= 0.99,
        "passes_grouped_oof_top1_gate": _accuracy(oof, targets, full) >= 0.99,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe two checkpoint geometric ensembles")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
