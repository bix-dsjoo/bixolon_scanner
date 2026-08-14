from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def _top3_candidates(
    logits: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
    names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    stacked_probabilities = np.stack([probabilities[name] for name in names])
    rank_scores = np.zeros_like(stacked_probabilities, dtype=np.float32)
    top3_votes = np.zeros_like(stacked_probabilities, dtype=np.float32)
    for view_index, name in enumerate(names):
        order = np.argsort(-logits[name], axis=1, kind="stable")
        ranks = np.empty_like(order)
        ranks[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])[None, :]
        rank_scores[view_index] = 1.0 / (ranks + 1.0)
        top3_votes[view_index] = ranks < 3
    mean_probability = stacked_probabilities.mean(axis=0)
    return {
        "mean_logits": np.mean([logits[name] for name in names], axis=0),
        "mean_probability": mean_probability,
        "maximum_probability": stacked_probabilities.max(axis=0),
        "reciprocal_rank": rank_scores.mean(axis=0),
        "top3_vote": top3_votes.mean(axis=0) + mean_probability * 1e-3,
    }


def _policy_candidates(
    logits: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    soup_best5 = (
        "soup:base",
        "soup:hflip",
        "soup:rot90",
        "soup:rot270",
        "soup:rot-30",
    )
    clutter_best4 = (
        "clutter:hflip",
        "clutter:rot180",
        "clutter:rot270",
        "clutter:rot15",
    )
    clutter_greedy5 = (
        "clutter:hflip",
        "clutter:rot-30",
        "clutter:rot180",
        "clutter:base",
        "clutter:rot270",
    )
    clutter_all10 = tuple(name for name in logits if name.startswith("clutter:"))
    recipes = {
        "soup_base": (("soup:base",), ("soup:base",)),
        "soup_best5": (soup_best5, soup_best5),
        "clutter_base": (("clutter:base",), ("clutter:base",)),
        "clutter_best4": (clutter_best4, clutter_best4),
        "clutter_greedy5": (clutter_greedy5, clutter_greedy5),
        "clutter_greedy5_riskall10": (clutter_greedy5, clutter_all10),
    }
    policies = {}
    all_probabilities = {name: _softmax(values) for name, values in logits.items()}
    for recipe, (classification_names, risk_names) in recipes.items():
        values = np.mean([logits[name] for name in classification_names], axis=0)
        probabilities = _softmax(values)
        predictions = probabilities.argmax(axis=1)
        ordered = np.sort(probabilities, axis=1)
        predicted_probabilities = np.stack(
            [all_probabilities[name][np.arange(len(values)), predictions] for name in risk_names]
        )
        view_predictions = np.stack([all_probabilities[name].argmax(axis=1) for name in risk_names])
        features = {
            "margin": ordered[:, -1] - ordered[:, -2],
            "top1_probability": ordered[:, -1],
            "inverse_entropy": np.sum(probabilities * np.log(probabilities.clip(1e-12)), axis=1),
            "mean_predicted_probability": predicted_probabilities.mean(axis=0),
            "minimum_predicted_probability": predicted_probabilities.min(axis=0),
            "negative_predicted_probability_std": -predicted_probabilities.std(axis=0),
            "prediction_agreement": np.mean(view_predictions == predictions[None, :], axis=0),
        }
        ranked = {name: _rank(feature) for name, feature in features.items()}
        top3_options = _top3_candidates(logits, all_probabilities, risk_names)
        for size in (1, 2, 3):
            for selected in combinations(ranked, size):
                score = np.mean([ranked[name] for name in selected], axis=0)
                for top3_name, top3_values in top3_options.items():
                    policy_name = f"{recipe}:" + "+".join(selected) + f":top3={top3_name}"
                    policies[policy_name] = (values, score, top3_values)
    return policies


def _threshold(score: np.ndarray, mask: np.ndarray, coverage: float) -> float:
    return float(np.quantile(score[mask], 1.0 - coverage, method="lower"))


def _metrics(
    logits: np.ndarray,
    score: np.ndarray,
    top3_values: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = logits.argmax(axis=1)
    top3 = np.argsort(-top3_values, axis=1, kind="stable")[:, :3]
    approved = (score >= threshold) & mask
    unknown = (~approved) & mask
    approved_errors = int(np.count_nonzero(approved & (predictions != targets)))
    recognized = (approved & (predictions == targets)) | (
        unknown & np.any(top3 == targets[:, None], axis=1)
    )
    return {
        "sample_count": int(np.count_nonzero(mask)),
        "approved_count": int(np.count_nonzero(approved)),
        "approval_coverage": float(np.count_nonzero(approved) / np.count_nonzero(mask)),
        "approved_error_count": approved_errors,
        "approved_misrecognition_rate": (
            approved_errors / np.count_nonzero(approved) if np.any(approved) else 1.0
        ),
        "unknown_count": int(np.count_nonzero(unknown)),
        "unknown_top3_accuracy": float(
            np.any(top3[unknown] == targets[unknown, None], axis=1).mean()
        ),
        "recognition_rate": float(np.count_nonzero(recognized) / np.count_nonzero(mask)),
        "top1_accuracy": float((predictions[mask] == targets[mask]).mean()),
    }


def _select(
    policies: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    targets: np.ndarray,
    mask: np.ndarray,
    coverage: float,
) -> dict[str, Any]:
    best = None
    for name, (logits, score, top3_values) in policies.items():
        threshold = _threshold(score, mask, coverage)
        metrics = _metrics(logits, score, top3_values, targets, mask, threshold)
        key = (
            -metrics["approved_error_count"],
            metrics["recognition_rate"],
            metrics["top1_accuracy"],
            name,
        )
        if best is None or key > best["key"]:
            best = {"name": name, "threshold": threshold, "metrics": metrics, "key": key}
    return best


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    left = np.load(args.soup_logits)
    right = np.load(args.clutter_logits)
    targets = left["targets"].astype(np.int64)
    if not np.array_equal(targets, right["targets"]):
        raise ValueError("checkpoint logits targets differ")
    logits = {
        **{
            f"soup:{name}": left[name].astype(np.float32)
            for name in left.files
            if name != "targets"
        },
        **{
            f"clutter:{name}": right[name].astype(np.float32)
            for name in right.files
            if name != "targets"
        },
    }
    rows = [
        json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line
    ]
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    policies = _policy_candidates(logits)
    full = np.ones(len(targets), dtype=bool)
    selected = _select(policies, targets, full, args.coverage)
    selected.pop("key")
    fold_reports = []
    totals = {
        "sample_count": 0,
        "approved_count": 0,
        "approved_error_count": 0,
        "recognized_count": 0,
        "unknown_count": 0,
        "unknown_top3_correct": 0,
    }
    for fold in range(3):
        selection, held_out = folds != fold, folds == fold
        policy = _select(policies, targets, selection, args.coverage)
        logits_value, score, top3_values = policies[policy["name"]]
        metrics = _metrics(logits_value, score, top3_values, targets, held_out, policy["threshold"])
        predictions = logits_value.argmax(axis=1)
        top3 = np.argsort(-top3_values, axis=1, kind="stable")[:, :3]
        approved = (score >= policy["threshold"]) & held_out
        unknown = (~approved) & held_out
        recognized = (approved & (predictions == targets)) | (
            unknown & np.any(top3 == targets[:, None], axis=1)
        )
        totals["sample_count"] += metrics["sample_count"]
        totals["approved_count"] += metrics["approved_count"]
        totals["approved_error_count"] += metrics["approved_error_count"]
        totals["recognized_count"] += int(np.count_nonzero(recognized))
        totals["unknown_count"] += metrics["unknown_count"]
        totals["unknown_top3_correct"] += int(
            np.count_nonzero(np.any(top3[unknown] == targets[unknown, None], axis=1))
        )
        fold_reports.append(
            {
                "held_out_fold": fold,
                "policy": policy["name"],
                "threshold": policy["threshold"],
                "metrics": metrics,
            }
        )
    oof = {
        "sample_count": totals["sample_count"],
        "approved_count": totals["approved_count"],
        "approval_coverage": totals["approved_count"] / totals["sample_count"],
        "approved_error_count": totals["approved_error_count"],
        "approved_misrecognition_rate": (totals["approved_error_count"] / totals["approved_count"]),
        "unknown_count": totals["unknown_count"],
        "unknown_top3_accuracy": (totals["unknown_top3_correct"] / totals["unknown_count"]),
        "recognition_rate": totals["recognized_count"] / totals["sample_count"],
        "folds": fold_reports,
    }
    report = {
        "schema_version": "1.0",
        "evaluation": "selective_approved_unknown_risk_policy_probe",
        "recapture_policy_changed": False,
        "target_approval_coverage": args.coverage,
        "policy_candidate_count": len(policies),
        "selected": selected,
        "grouped_3fold_oof": oof,
        "passes_point_gates": {
            "recognition_at_least_0_99": bool(selected["metrics"]["recognition_rate"] >= 0.99),
            "approved_misrecognition_at_most_0_001": bool(
                selected["metrics"]["approved_misrecognition_rate"] <= 0.001
            ),
            "approval_coverage_at_least_0_85": bool(
                selected["metrics"]["approval_coverage"] >= 0.85
            ),
        },
        "passes_grouped_oof_point_gates": {
            "recognition_at_least_0_99": bool(oof["recognition_rate"] >= 0.99),
            "approved_misrecognition_at_most_0_001": bool(
                oof["approved_misrecognition_rate"] <= 0.001
            ),
            "approval_coverage_at_least_0_85": bool(oof["approval_coverage"] >= 0.85),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe APPROVED/UNKNOWN selective risk")
    parser.add_argument("--soup-logits", type=Path, required=True)
    parser.add_argument("--clutter-logits", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=float, default=0.85)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
