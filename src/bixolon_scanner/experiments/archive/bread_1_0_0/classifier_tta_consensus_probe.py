from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    shifted = values / temperature
    shifted -= shifted.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _margin(values: np.ndarray) -> np.ndarray:
    ordered = np.partition(values, -2, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float((predictions == targets).mean())


def _load_folds(path: Path) -> np.ndarray:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)


def _candidate_predictions(logits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = list(logits)
    values = np.stack([logits[name] for name in names])
    probabilities = np.stack([_softmax(logits[name]) for name in names])
    predictions: dict[str, np.ndarray] = {name: logits[name].argmax(axis=1) for name in names}
    for size in range(2, min(6, len(names)) + 1):
        for selected in combinations(range(len(names)), size):
            label = "+".join(names[index] for index in selected)
            predictions[f"mean_logit:{label}"] = values[list(selected)].mean(axis=0).argmax(axis=1)
            predictions[f"mean_prob:{label}"] = (
                probabilities[list(selected)].mean(axis=0).argmax(axis=1)
            )
    predictions["median_logit:all"] = np.median(values, axis=0).argmax(axis=1)
    predictions["mean_prob:all"] = probabilities.mean(axis=0).argmax(axis=1)
    predictions["mean_logit:all"] = values.mean(axis=0).argmax(axis=1)
    if len(names) >= 5:
        predictions["trimmed_logit:all"] = np.sort(values, axis=0)[1:-1].mean(axis=0).argmax(axis=1)

    per_view_predictions = values.argmax(axis=2)
    per_view_margins = np.stack([_margin(row) for row in probabilities])
    highest_margin_view = per_view_margins.argmax(axis=0)
    sample_indices = np.arange(values.shape[1])
    predictions["highest_probability_margin_view"] = per_view_predictions[
        highest_margin_view, sample_indices
    ]
    vote_scores = np.zeros((values.shape[1], values.shape[2]), dtype=np.float32)
    margin_vote_scores = np.zeros_like(vote_scores)
    for view_index in range(len(names)):
        vote_scores[sample_indices, per_view_predictions[view_index]] += 1
        margin_vote_scores[sample_indices, per_view_predictions[view_index]] += per_view_margins[
            view_index
        ]
    predictions["majority_vote"] = vote_scores.argmax(axis=1)
    predictions["margin_weighted_vote"] = margin_vote_scores.argmax(axis=1)
    return predictions


def _routing_features(logits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = list(logits)
    probabilities = np.stack([_softmax(logits[name]) for name in names])
    per_view_predictions = probabilities.argmax(axis=2)
    base_probability = probabilities[names.index("base")]
    agreement = np.zeros(probabilities.shape[1], dtype=np.float32)
    base_agreement = np.zeros_like(agreement)
    for index in range(probabilities.shape[1]):
        counts = np.bincount(per_view_predictions[:, index], minlength=probabilities.shape[2])
        agreement[index] = counts.max() / len(names)
        base_agreement[index] = counts[per_view_predictions[names.index("base"), index]] / len(
            names
        )
    margins = np.stack([_margin(probabilities[index]) for index in range(len(names))])
    entropy = -(base_probability * np.log(base_probability.clip(1e-12))).sum(axis=1)
    return {
        "base_margin": _margin(base_probability),
        "base_entropy": entropy,
        "base_agreement": base_agreement,
        "majority_agreement": agreement,
        "maximum_view_margin": margins.max(axis=0),
        "view_margin_spread": margins.max(axis=0) - margins.min(axis=0),
    }


def _select_router(
    candidates: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    targets: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    ranked = sorted(
        candidates,
        key=lambda name: _accuracy(candidates[name][mask], targets[mask]),
        reverse=True,
    )[:12]
    best: dict[str, Any] | None = None
    for left in ranked:
        for right in ranked:
            if left == right:
                continue
            for feature_name, feature in features.items():
                thresholds = np.unique(np.quantile(feature[mask], np.linspace(0.03, 0.97, 33)))
                for direction in ("below", "above"):
                    for threshold in thresholds:
                        use_left = (
                            feature < threshold if direction == "below" else feature > threshold
                        )
                        predictions = np.where(use_left, candidates[left], candidates[right])
                        accuracy = _accuracy(predictions[mask], targets[mask])
                        if best is None or accuracy > best["selection_accuracy"]:
                            best = {
                                "left": left,
                                "right": right,
                                "feature": feature_name,
                                "direction": direction,
                                "threshold": float(threshold),
                                "selection_accuracy": accuracy,
                            }
    if best is None:
        raise RuntimeError("router selection produced no candidate")
    return best


def _apply_router(
    router: dict[str, Any], candidates: dict[str, np.ndarray], features: dict[str, np.ndarray]
) -> np.ndarray:
    feature = features[router["feature"]]
    use_left = (
        feature < router["threshold"]
        if router["direction"] == "below"
        else feature > router["threshold"]
    )
    return np.where(use_left, candidates[router["left"]], candidates[router["right"]])


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    loaded = np.load(args.logits)
    targets = loaded["targets"].astype(np.int64)
    logits = {name: loaded[name].astype(np.float32) for name in loaded.files if name != "targets"}
    if "base" not in logits:
        raise ValueError("geometric logits must contain the base view")
    folds = _load_folds(args.records)
    if folds.shape != targets.shape:
        raise ValueError("evaluation records and logits are not aligned")
    candidates = _candidate_predictions(logits)
    features = _routing_features(logits)
    ranked = sorted(
        (
            {"name": name, "top1_accuracy": _accuracy(prediction, targets)}
            for name, prediction in candidates.items()
        ),
        key=lambda row: (row["top1_accuracy"], row["name"]),
        reverse=True,
    )
    full_mask = np.ones(len(targets), dtype=bool)
    selected = _select_router(candidates, features, targets, full_mask)
    selected_predictions = _apply_router(selected, candidates, features)
    selected["top1_accuracy"] = _accuracy(selected_predictions, targets)
    selected["correct"] = int(np.count_nonzero(selected_predictions == targets))
    cross_validation = []
    oof_predictions = np.empty_like(targets)
    for fold in range(3):
        training = folds != fold
        held_out = folds == fold
        router = _select_router(candidates, features, targets, training)
        predictions = _apply_router(router, candidates, features)
        oof_predictions[held_out] = predictions[held_out]
        cross_validation.append(
            {
                "held_out_fold": fold,
                "router": router,
                "held_out_top1_accuracy": _accuracy(predictions[held_out], targets[held_out]),
                "held_out_count": int(np.count_nonzero(held_out)),
            }
        )
    report = {
        "schema_version": "1.0",
        "evaluation": "label_free_geometric_tta_consensus_policy_selection",
        "sample_count": len(targets),
        "view_names": list(logits),
        "candidate_count": len(candidates),
        "best_fixed_consensus": ranked[0],
        "top_fixed_consensus": ranked[:20],
        "selected_router": selected,
        "grouped_3fold_oof": {
            "top1_accuracy": _accuracy(oof_predictions, targets),
            "correct": int(np.count_nonzero(oof_predictions == targets)),
            "folds": cross_validation,
        },
        "passes_top1_gate": selected["top1_accuracy"] >= 0.99,
        "passes_grouped_oof_top1_gate": _accuracy(oof_predictions, targets) >= 0.99,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe label-free geometric TTA consensus rules")
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
