from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def _top3_rankings(view_logits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = tuple(view_logits)
    probabilities = {name: _softmax(view_logits[name]) for name in names}
    stacked_probabilities = np.stack([probabilities[name] for name in names])
    reciprocal_ranks = np.zeros_like(stacked_probabilities, dtype=np.float32)
    top3_votes = np.zeros_like(stacked_probabilities, dtype=np.float32)
    for view_index, name in enumerate(names):
        order = np.argsort(-view_logits[name], axis=1, kind="stable")
        ranks = np.empty_like(order)
        ranks[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])[None, :]
        reciprocal_ranks[view_index] = 1.0 / (ranks + 1.0)
        top3_votes[view_index] = ranks < 3
    mean_probability = stacked_probabilities.mean(axis=0)
    return {
        "mean_logits": np.mean([view_logits[name] for name in names], axis=0),
        "mean_probability": mean_probability,
        "maximum_probability": stacked_probabilities.max(axis=0),
        "reciprocal_rank": reciprocal_ranks.mean(axis=0),
        "top3_vote": top3_votes.mean(axis=0) + mean_probability * 1e-3,
    }


def _view_recipes(names: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    requested = {
        "base": ("base",),
        "best4": ("hflip", "rot180", "rot270", "rot15"),
        "greedy5": ("hflip", "rot-30", "rot180", "base", "rot270"),
        "best5": ("base", "hflip", "rot90", "rot270", "rot-30"),
        "all": names,
    }
    missing = {view for recipe in requested.values() for view in recipe if view not in names}
    if missing:
        raise ValueError(f"geometric logits are missing views: {sorted(missing)}")
    return requested


def _approval_features(
    classification_logits: np.ndarray,
    view_logits: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    probabilities = _softmax(classification_logits)
    predictions = probabilities.argmax(axis=1)
    ordered = np.sort(probabilities, axis=1)
    view_probabilities = np.stack(
        [_softmax(values)[np.arange(len(values)), predictions] for values in view_logits.values()]
    )
    view_predictions = np.stack(
        [_softmax(values).argmax(axis=1) for values in view_logits.values()]
    )
    features = {
        "margin": ordered[:, -1] - ordered[:, -2],
        "top1_probability": ordered[:, -1],
        "inverse_entropy": np.sum(probabilities * np.log(probabilities.clip(1e-12)), axis=1),
        "mean_predicted_probability": view_probabilities.mean(axis=0),
        "minimum_predicted_probability": view_probabilities.min(axis=0),
        "negative_predicted_probability_std": -view_probabilities.std(axis=0),
        "prediction_agreement": np.mean(view_predictions == predictions[None, :], axis=0),
    }
    return predictions, features


def _top3_safety_features(
    ranking: np.ndarray,
    top3: np.ndarray,
    view_logits: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    ordered = np.sort(ranking, axis=1)
    ranking_probabilities = _softmax(ranking)
    ordered_probabilities = np.sort(ranking_probabilities, axis=1)
    view_top3 = [
        np.argsort(-values, axis=1, kind="stable")[:, :3] for values in view_logits.values()
    ]
    set_agreement = np.mean(
        [
            np.mean(
                np.any(candidate[:, :, None] == top3[:, None, :], axis=2),
                axis=1,
            )
            for candidate in view_top3
        ],
        axis=0,
    )
    return {
        "third_score": ordered[:, -3],
        "third_fourth_margin": ordered[:, -3] - ordered[:, -4],
        "third_probability": ordered_probabilities[:, -3],
        "inverse_entropy": np.sum(
            ranking_probabilities * np.log(ranking_probabilities.clip(1e-12)), axis=1
        ),
        "view_top3_set_agreement": set_agreement,
    }


@dataclass(frozen=True)
class Policy:
    name: str
    predictions: np.ndarray
    approval_score: np.ndarray
    top3: np.ndarray
    top3_safety_score: np.ndarray


def policy_candidates(logits: dict[str, np.ndarray]) -> list[Policy]:
    names = tuple(logits)
    candidates: list[Policy] = []
    for recipe_name, recipe_views in _view_recipes(names).items():
        selected = {name: logits[name] for name in recipe_views}
        classification_logits = np.mean(list(selected.values()), axis=0)
        predictions, approval_features = _approval_features(classification_logits, selected)
        for ranking_name, ranking in _top3_rankings(selected).items():
            top3 = np.argsort(-ranking, axis=1, kind="stable")[:, :3]
            safety_features = _top3_safety_features(ranking, top3, selected)
            for approval_name, approval_score in approval_features.items():
                for safety_name, safety_score in safety_features.items():
                    candidates.append(
                        Policy(
                            name=(
                                f"{recipe_name}:approval={approval_name}:"
                                f"top3={ranking_name}:safety={safety_name}"
                            ),
                            predictions=predictions,
                            approval_score=approval_score,
                            top3=top3,
                            top3_safety_score=safety_score,
                        )
                    )
    return candidates


def _guarded_threshold(
    score: np.ndarray,
    failures: np.ndarray,
    calibration: np.ndarray,
    guard_samples: int,
) -> float:
    if not np.any(failures & calibration):
        return float("-inf")
    boundary = float(np.max(score[failures & calibration]))
    safer = np.sort(score[calibration & (score > boundary)])
    if guard_samples <= 0:
        return float(
            np.nextafter(
                np.asarray(boundary, dtype=score.dtype),
                np.asarray(np.inf, dtype=score.dtype),
            )
        )
    if len(safer) < guard_samples:
        return float("inf")
    return float(safer[guard_samples - 1])


def _thresholds(
    policy: Policy,
    targets: np.ndarray,
    calibration: np.ndarray,
    guard_samples: int,
) -> tuple[float, float]:
    prediction_failures = policy.predictions != targets
    approval_threshold = _guarded_threshold(
        policy.approval_score,
        prediction_failures,
        calibration,
        guard_samples,
    )
    approved = policy.approval_score >= approval_threshold
    top3_correct = np.any(policy.top3 == targets[:, None], axis=1)
    top3_failures = (~approved) & (~top3_correct)
    safety_threshold = _guarded_threshold(
        policy.top3_safety_score,
        top3_failures,
        calibration,
        guard_samples,
    )
    return approval_threshold, safety_threshold


def policy_metrics(
    policy: Policy,
    targets: np.ndarray,
    mask: np.ndarray,
    approval_threshold: float,
    safety_threshold: float,
) -> dict[str, Any]:
    approved = mask & (policy.approval_score >= approval_threshold)
    unknown = mask & (~approved) & (policy.top3_safety_score >= safety_threshold)
    recapture = mask & (~approved) & (~unknown)
    approved_errors = approved & (policy.predictions != targets)
    top3_correct = np.any(policy.top3 == targets[:, None], axis=1)
    unknown_misses = unknown & (~top3_correct)
    sample_count = int(np.count_nonzero(mask))
    approved_count = int(np.count_nonzero(approved))
    unknown_count = int(np.count_nonzero(unknown))
    recapture_count = int(np.count_nonzero(recapture))
    return {
        "sample_count": sample_count,
        "approved_count": approved_count,
        "approved_rate": approved_count / sample_count if sample_count else 0.0,
        "approved_error_count": int(np.count_nonzero(approved_errors)),
        "unknown_count": unknown_count,
        "unknown_rate": unknown_count / sample_count if sample_count else 0.0,
        "unknown_top3_miss_count": int(np.count_nonzero(unknown_misses)),
        "segment_recapture_count": recapture_count,
        "segment_recapture_rate": recapture_count / sample_count if sample_count else 0.0,
        "non_recapture_count": approved_count + unknown_count,
        "non_recapture_rate": (
            (approved_count + unknown_count) / sample_count if sample_count else 0.0
        ),
    }


def select_policy(
    candidates: list[Policy],
    targets: np.ndarray,
    calibration: np.ndarray,
    guard_samples: int,
) -> tuple[Policy, float, float, dict[str, Any]]:
    selected = None
    for policy in candidates:
        approval_threshold, safety_threshold = _thresholds(
            policy, targets, calibration, guard_samples
        )
        metrics = policy_metrics(
            policy,
            targets,
            calibration,
            approval_threshold,
            safety_threshold,
        )
        key = (
            -metrics["approved_error_count"] - metrics["unknown_top3_miss_count"],
            metrics["non_recapture_count"],
            metrics["approved_count"],
            policy.name,
        )
        if selected is None or key > selected[0]:
            selected = (
                key,
                policy,
                approval_threshold,
                safety_threshold,
                metrics,
            )
    if selected is None:
        raise ValueError("no classifier policy candidates")
    return selected[1], selected[2], selected[3], selected[4]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    payload = np.load(args.logits)
    targets = payload["targets"].astype(np.int64)
    logits = {name: payload[name].astype(np.float32) for name in payload.files if name != "targets"}
    rows = [
        json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line
    ]
    if len(rows) != len(targets):
        raise ValueError("classifier records and logits have different lengths")
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    candidates = policy_candidates(logits)
    full = np.ones(len(targets), dtype=bool)
    policy, approval_threshold, safety_threshold, selected_metrics = select_policy(
        candidates, targets, full, args.guard_samples
    )
    fold_reports = []
    oof_totals = {
        "sample_count": 0,
        "approved_count": 0,
        "approved_error_count": 0,
        "unknown_count": 0,
        "unknown_top3_miss_count": 0,
        "segment_recapture_count": 0,
        "non_recapture_count": 0,
    }
    for held_out_fold in sorted(set(folds.tolist())):
        calibration = folds != held_out_fold
        held_out = folds == held_out_fold
        fold_policy, fold_approval, fold_safety, _ = select_policy(
            candidates, targets, calibration, args.guard_samples
        )
        metrics = policy_metrics(
            fold_policy,
            targets,
            held_out,
            fold_approval,
            fold_safety,
        )
        for key in oof_totals:
            oof_totals[key] += int(metrics[key])
        fold_reports.append(
            {
                "held_out_fold": held_out_fold,
                "policy": fold_policy.name,
                "approval_threshold": fold_approval,
                "top3_safety_threshold": fold_safety,
                "metrics": metrics,
            }
        )
    oof_count = oof_totals["sample_count"]
    oof = {
        **oof_totals,
        "approved_rate": oof_totals["approved_count"] / oof_count,
        "unknown_rate": oof_totals["unknown_count"] / oof_count,
        "segment_recapture_rate": oof_totals["segment_recapture_count"] / oof_count,
        "non_recapture_rate": oof_totals["non_recapture_count"] / oof_count,
        "folds": fold_reports,
    }
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_classifier_zero_error_selective_policy",
        "source_dataset": args.source_dataset,
        "logits_file": args.logits.name,
        "sample_count": len(targets),
        "policy_candidate_count": len(candidates),
        "guard_samples": args.guard_samples,
        "selection_scope": "all-available-development-no-locked-test",
        "selected": {
            "policy": policy.name,
            "approval_threshold": approval_threshold,
            "top3_safety_threshold": safety_threshold,
            "metrics": selected_metrics,
        },
        "grouped_oof": oof,
        "passes_selected_count_gates": {
            "approved_error_zero": selected_metrics["approved_error_count"] == 0,
            "unknown_top3_miss_zero": selected_metrics["unknown_top3_miss_count"] == 0,
        },
        "passes_oof_count_gates": {
            "approved_error_zero": oof["approved_error_count"] == 0,
            "unknown_top3_miss_zero": oof["unknown_top3_miss_count"] == 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select APPROVED/UNKNOWN/SEGMENT_RECAPTURE count-zero policy"
    )
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-dataset", choices=("single_objects", "single_objects_2"), required=True
    )
    parser.add_argument("--guard-samples", type=int, default=0)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
