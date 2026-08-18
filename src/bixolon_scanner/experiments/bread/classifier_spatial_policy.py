from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .zero_error_classifier import (
    Policy,
    _approval_features,
    _guarded_threshold,
    _softmax,
    _top3_safety_features,
    policy_metrics,
)


@dataclass(frozen=True)
class SpatialRecipe:
    names: tuple[str, ...]
    weights: tuple[float, ...]

    @property
    def name(self) -> str:
        parts = [f"{name}@{weight:.2f}" for name, weight in zip(self.names, self.weights)]
        return "+".join(parts)


def spatial_recipes(
    names: tuple[str, ...], pair_weights: tuple[float, ...]
) -> Iterator[SpatialRecipe]:
    for name in names:
        yield SpatialRecipe((name,), (1.0,))
    for left, right in itertools.combinations(names, 2):
        for right_weight in pair_weights:
            if not 0.0 < right_weight < 1.0:
                raise ValueError("pair weights must be strictly between zero and one")
            yield SpatialRecipe((left, right), (1.0 - right_weight, right_weight))


def _weighted_rankings(
    selected: list[np.ndarray],
    weights: np.ndarray,
    *,
    ranking_tie_break_bias_span: float,
) -> dict[str, np.ndarray]:
    probabilities = np.stack([_softmax(values) for values in selected])
    orders = np.argsort(-np.stack(selected), axis=2, kind="stable")
    ranks = np.empty_like(orders)
    np.put_along_axis(
        ranks,
        orders,
        np.arange(orders.shape[2], dtype=orders.dtype)[None, None, :],
        axis=2,
    )
    weighted = weights[:, None, None]
    weighted_probability = np.sum(probabilities * weighted, axis=0)
    ranking_tie_break = np.linspace(
        0.0,
        -ranking_tie_break_bias_span,
        selected[0].shape[1],
        dtype=np.float32,
    )[None, :]
    return {
        "weighted_logits": np.sum(np.stack(selected) * weighted, axis=0),
        "weighted_probability": weighted_probability,
        "weighted_reciprocal_rank": np.sum((1.0 / (ranks + 1.0)) * weighted, axis=0)
        + weighted_probability * 1e-3
        + ranking_tie_break,
        "weighted_top3_vote": np.sum((ranks < 3) * weighted, axis=0) + weighted_probability * 1e-3,
    }


def _spatial_safety_features(
    ranking: np.ndarray,
    top3: np.ndarray,
    selected: list[np.ndarray],
) -> dict[str, np.ndarray]:
    views = {str(index): values for index, values in enumerate(selected)}
    features = _top3_safety_features(ranking, top3, views)
    view_orders = np.argsort(-np.stack(selected), axis=2, kind="stable")[:, :, :3]
    support = np.stack(
        [
            np.mean(np.any(view_orders == candidate[None, :, None], axis=2), axis=0)
            for candidate in top3.T
        ],
        axis=1,
    )
    union_size = np.asarray(
        [len(set(values.reshape(-1).tolist())) for values in np.transpose(view_orders, (1, 0, 2))],
        dtype=np.float32,
    )
    features.update(
        {
            "mean_candidate_view_support": support.mean(axis=1),
            "minimum_candidate_view_support": support.min(axis=1),
            "negative_top3_union_size": -union_size,
            "support_minus_union": support.mean(axis=1) - union_size / (3.0 * len(selected)),
        }
    )
    return features


def _thresholds(
    policy: Policy,
    targets: np.ndarray,
    calibration: np.ndarray,
    *,
    approval_guard_samples: int,
    safety_guard_samples: int,
) -> tuple[float, float]:
    approval_threshold = _guarded_threshold(
        policy.approval_score,
        policy.predictions != targets,
        calibration,
        approval_guard_samples,
    )
    approved = policy.approval_score >= approval_threshold
    top3_correct = np.any(policy.top3 == targets[:, None], axis=1)
    safety_threshold = _guarded_threshold(
        policy.top3_safety_score,
        (~approved) & (~top3_correct),
        calibration,
        safety_guard_samples,
    )
    return approval_threshold, safety_threshold


def select_spatial_policy(
    logits: dict[str, np.ndarray],
    targets: np.ndarray,
    calibration: np.ndarray,
    *,
    pair_weights: tuple[float, ...],
    approval_guard_samples: int,
    safety_guard_samples: int,
    ranking_tie_break_bias_span: float,
    approval_feature_names: tuple[str, ...] = ("margin",),
) -> tuple[Policy, float, float, dict[str, Any]]:
    selected_result = None
    for recipe in spatial_recipes(tuple(logits), pair_weights):
        selected_views = [logits[name] for name in recipe.names]
        weights = np.asarray(recipe.weights, dtype=np.float32)
        classification_logits = np.sum(np.stack(selected_views) * weights[:, None, None], axis=0)
        predictions, approval_features = _approval_features(
            classification_logits,
            {str(index): values for index, values in enumerate(selected_views)},
        )
        for ranking_name, ranking in _weighted_rankings(
            selected_views,
            weights,
            ranking_tie_break_bias_span=ranking_tie_break_bias_span,
        ).items():
            top3 = np.argsort(-ranking, axis=1, kind="stable")[:, :3]
            safety_ranking = ranking
            if ranking_name == "weighted_reciprocal_rank":
                safety_ranking = (
                    ranking
                    - np.linspace(
                        0.0,
                        -ranking_tie_break_bias_span,
                        ranking.shape[1],
                        dtype=np.float32,
                    )[None, :]
                )
            safety_features = _spatial_safety_features(safety_ranking, top3, selected_views)
            for approval_name, approval_score in approval_features.items():
                if approval_name not in approval_feature_names:
                    continue
                for safety_name, safety_score in safety_features.items():
                    policy = Policy(
                        name=(
                            f"{recipe.name}:approval={approval_name}:"
                            f"top3={ranking_name}:safety={safety_name}"
                        ),
                        predictions=predictions,
                        approval_score=approval_score,
                        top3=top3,
                        top3_safety_score=safety_score,
                    )
                    approval_threshold, safety_threshold = _thresholds(
                        policy,
                        targets,
                        calibration,
                        approval_guard_samples=approval_guard_samples,
                        safety_guard_samples=safety_guard_samples,
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
                    if selected_result is None or key > selected_result[0]:
                        selected_result = (
                            key,
                            policy,
                            approval_threshold,
                            safety_threshold,
                            metrics,
                        )
    if selected_result is None:
        raise ValueError("no spatial policy candidates")
    return selected_result[1], selected_result[2], selected_result[3], selected_result[4]


def _metric_counts(
    *,
    targets: np.ndarray,
    mask: np.ndarray,
    predictions: np.ndarray,
    approved: np.ndarray,
    top3: np.ndarray,
    unknown: np.ndarray,
) -> dict[str, Any]:
    recapture = mask & (~approved) & (~unknown)
    count = int(np.count_nonzero(mask))
    return {
        "sample_count": count,
        "approved_count": int(np.count_nonzero(mask & approved)),
        "approved_error_count": int(np.count_nonzero(mask & approved & (predictions != targets))),
        "unknown_count": int(np.count_nonzero(mask & unknown)),
        "unknown_top3_miss_count": int(
            np.count_nonzero(mask & unknown & (~np.any(top3 == targets[:, None], axis=1)))
        ),
        "segment_recapture_count": int(np.count_nonzero(recapture)),
        "segment_recapture_rate": float(np.count_nonzero(recapture) / count) if count else 0.0,
    }


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
    full = np.ones(len(targets), dtype=bool)
    pair_weights = tuple(float(value) for value in args.pair_weights)
    policy, approval_threshold, safety_threshold, selected_metrics = select_spatial_policy(
        logits,
        targets,
        full,
        pair_weights=pair_weights,
        approval_guard_samples=args.approval_guard_samples,
        safety_guard_samples=args.safety_guard_samples,
        ranking_tie_break_bias_span=args.ranking_tie_break_bias_span,
        approval_feature_names=tuple(args.approval_features),
    )
    evaluation_mask = np.asarray(
        [int(row["image_id"]) not in set(args.evaluation_exclude_image_ids) for row in rows],
        dtype=bool,
    )
    evaluation_metrics = policy_metrics(
        policy,
        targets,
        evaluation_mask,
        approval_threshold,
        safety_threshold,
    )

    oof_predictions = np.zeros(len(targets), dtype=np.int64)
    oof_approval_scores = np.zeros(len(targets), dtype=np.float32)
    oof_top3 = np.zeros((len(targets), 3), dtype=np.int64)
    oof_safety_scores = np.zeros(len(targets), dtype=np.float32)
    fold_reports = []
    for held_out_fold in sorted(set(folds.tolist())):
        calibration = folds != held_out_fold
        held_out = folds == held_out_fold
        fold_policy, fold_approval, fold_safety, _ = select_spatial_policy(
            logits,
            targets,
            calibration,
            pair_weights=pair_weights,
            approval_guard_samples=args.approval_guard_samples,
            safety_guard_samples=args.safety_guard_samples,
            ranking_tie_break_bias_span=args.ranking_tie_break_bias_span,
            approval_feature_names=tuple(args.approval_features),
        )
        fold_metrics = policy_metrics(
            fold_policy,
            targets,
            held_out,
            fold_approval,
            fold_safety,
        )
        oof_predictions[held_out] = fold_policy.predictions[held_out]
        oof_approval_scores[held_out] = fold_policy.approval_score[held_out]
        oof_top3[held_out] = fold_policy.top3[held_out]
        oof_safety_scores[held_out] = fold_policy.top3_safety_score[held_out]
        fold_reports.append(
            {
                "held_out_fold": held_out_fold,
                "policy": fold_policy.name,
                "approval_threshold": fold_approval,
                "top3_safety_threshold": fold_safety,
                "metrics": fold_metrics,
            }
        )
    pooled_approval_threshold = _guarded_threshold(
        oof_approval_scores,
        oof_predictions != targets,
        full,
        args.approval_guard_samples,
    )
    oof_approved = oof_approval_scores >= pooled_approval_threshold
    oof_top3_correct = np.any(oof_top3 == targets[:, None], axis=1)
    pooled_safety_threshold = _guarded_threshold(
        oof_safety_scores,
        (~oof_approved) & (~oof_top3_correct),
        full,
        args.safety_guard_samples,
    )
    oof_unknown = (~oof_approved) & (oof_safety_scores >= pooled_safety_threshold)
    pooled_oof_metrics = _metric_counts(
        targets=targets,
        mask=evaluation_mask,
        predictions=oof_predictions,
        approved=oof_approved,
        top3=oof_top3,
        unknown=oof_unknown,
    )
    selected_approved = policy.approval_score >= approval_threshold
    selected_unknown = (~selected_approved) & (policy.top3_safety_score >= safety_threshold)
    difficulty_reports = None
    if args.detector_manifest is not None:
        manifest_rows = [
            json.loads(line)
            for line in args.detector_manifest.read_text(encoding="utf-8").splitlines()
            if line
        ]
        difficulty_by_image = {
            int(row["image_id"]): str(row["difficulty"]).upper() for row in manifest_rows
        }
        difficulty_reports = {}
        for difficulty in ("EASY", "MEDIUM", "HARD", "SCAN_LOG"):
            bucket = evaluation_mask & np.asarray(
                [difficulty_by_image[int(row["image_id"])] == difficulty for row in rows],
                dtype=bool,
            )
            difficulty_reports[difficulty] = {
                "selected": policy_metrics(
                    policy,
                    targets,
                    bucket,
                    approval_threshold,
                    safety_threshold,
                ),
                "pooled_oof": _metric_counts(
                    targets=targets,
                    mask=bucket,
                    predictions=oof_predictions,
                    approved=oof_approved,
                    top3=oof_top3,
                    unknown=oof_unknown,
                ),
            }
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_classifier_spatial_zero_error_policy",
        "sample_count": len(targets),
        "view_count": len(logits),
        "maximum_runtime_views": 2,
        "pair_weights": pair_weights,
        "approval_guard_samples": args.approval_guard_samples,
        "approval_features": args.approval_features,
        "safety_guard_samples": args.safety_guard_samples,
        "ranking_tie_break_bias_span": args.ranking_tie_break_bias_span,
        "selection_scope": "all-available-grouped-development-no-locked-test",
        "selected": {
            "policy": policy.name,
            "approval_threshold": approval_threshold,
            "top3_safety_threshold": safety_threshold,
            "calibration_metrics": selected_metrics,
            "evaluation_after_image_gates": evaluation_metrics,
        },
        "grouped_oof": {
            "folds": fold_reports,
            "pooled_approval_threshold": pooled_approval_threshold,
            "pooled_safety_threshold": pooled_safety_threshold,
            "evaluation_after_image_gates": pooled_oof_metrics,
        },
        "by_difficulty": difficulty_reports,
        "evaluation_excluded_image_ids": args.evaluation_exclude_image_ids,
        "passes_selected_count_gates": {
            "approved_error_zero": evaluation_metrics["approved_error_count"] == 0,
            "unknown_top3_miss_zero": evaluation_metrics["unknown_top3_miss_count"] == 0,
            "segment_recapture_at_most_five_percent": evaluation_metrics["segment_recapture_rate"]
            <= 0.05,
        },
        "passes_pooled_oof_count_gates": {
            "approved_error_zero": pooled_oof_metrics["approved_error_count"] == 0,
            "unknown_top3_miss_zero": pooled_oof_metrics["unknown_top3_miss_count"] == 0,
            "segment_recapture_at_most_five_percent": pooled_oof_metrics["segment_recapture_rate"]
            <= 0.05,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.decisions_output is not None:
        args.decisions_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.decisions_output,
            targets=targets,
            evaluation_mask=evaluation_mask,
            selected_predictions=policy.predictions,
            selected_top3=policy.top3,
            selected_approved=selected_approved,
            selected_unknown=selected_unknown,
            oof_predictions=oof_predictions,
            oof_top3=oof_top3,
            oof_approved=oof_approved,
            oof_unknown=oof_unknown,
        )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a two-view spatial zero-error policy")
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions-output", type=Path)
    parser.add_argument("--detector-manifest", type=Path)
    parser.add_argument("--pair-weights", type=float, nargs="+", default=(0.25, 0.5, 0.75))
    parser.add_argument("--approval-guard-samples", type=int, default=0)
    parser.add_argument("--approval-features", nargs="+", default=("margin",))
    parser.add_argument("--safety-guard-samples", type=int, default=0)
    parser.add_argument("--ranking-tie-break-bias-span", type=float, default=0.0)
    parser.add_argument("--evaluation-exclude-image-ids", type=int, nargs="*", default=())
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
