from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .classifier_spatial_policy import _spatial_safety_features, _weighted_rankings
from .zero_error_classifier import Policy, _approval_features, _guarded_threshold


@dataclass(frozen=True)
class RatePolicyResult:
    first_view_weight: float
    policy: Policy
    approved: np.ndarray
    unknown: np.ndarray
    confirmation: np.ndarray
    fold_reports: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    gates: dict[str, bool]


def validate_group_folds(rows: list[dict[str, Any]]) -> int:
    folds_by_group: dict[str, set[int]] = {}
    for row in rows:
        group = str(row["group_id"])
        folds_by_group.setdefault(group, set()).add(int(row["fold"]))
    overlap = sum(len(folds) > 1 for folds in folds_by_group.values())
    if overlap:
        raise ValueError(f"classifier group-aware folds overlap for {overlap} groups")
    return overlap


def _class_conditional_thresholds(
    policy: Policy,
    targets: np.ndarray,
    calibration: np.ndarray,
    *,
    class_count: int,
    guard_samples: int,
    no_failure_threshold: float,
) -> np.ndarray:
    if not 0.0 <= no_failure_threshold <= 1.0:
        raise ValueError("no-failure approval threshold must be in [0, 1]")
    failures = policy.predictions != targets
    thresholds = np.empty(class_count, dtype=np.float32)
    for class_index in range(class_count):
        predicted_class = policy.predictions == class_index
        class_calibration = calibration & predicted_class
        if not np.any(class_calibration & failures):
            thresholds[class_index] = no_failure_threshold
            continue
        thresholds[class_index] = _guarded_threshold(
            policy.approval_score,
            failures,
            class_calibration,
            guard_samples,
        )
    return thresholds


def _approved(policy: Policy, thresholds: np.ndarray) -> np.ndarray:
    return policy.approval_score >= thresholds[policy.predictions]


def unanimous_top1_confirmation(
    predictions: np.ndarray,
    confirmation_predictions: np.ndarray,
) -> np.ndarray:
    values = np.asarray(confirmation_predictions)
    if values.ndim != 2 or values.shape[0] != len(predictions):
        raise ValueError("confirmation predictions must have shape [samples, views]")
    if values.shape[1] < 1:
        raise ValueError("at least one confirmation view is required")
    return np.all(values == np.asarray(predictions)[:, None], axis=1)


def _metrics(
    policy: Policy,
    targets: np.ndarray,
    mask: np.ndarray,
    approved: np.ndarray,
    unknown: np.ndarray,
) -> dict[str, Any]:
    top3_correct = np.any(policy.top3 == targets[:, None], axis=1)
    recapture = (~approved) & (~unknown)
    sample_count = int(np.count_nonzero(mask))
    approved_count = int(np.count_nonzero(mask & approved))
    approved_errors = int(np.count_nonzero(mask & approved & (policy.predictions != targets)))
    unknown_count = int(np.count_nonzero(mask & unknown))
    candidate_out = int(np.count_nonzero(mask & unknown & (~top3_correct)))
    recapture_count = int(np.count_nonzero(mask & recapture))
    denominator = max(1, sample_count)
    return {
        "sample_count": sample_count,
        "approved_count": approved_count,
        "approved_rate": approved_count / denominator,
        "approved_misrecognition_count": approved_errors,
        "approved_misrecognition_rate": approved_errors / denominator,
        "unknown_count": unknown_count,
        "unknown_rate": unknown_count / denominator,
        "unknown_top3_candidate_out_count": candidate_out,
        "unknown_top3_candidate_out_rate": candidate_out / denominator,
        "segment_recapture_count": recapture_count,
        "segment_recapture_rate": recapture_count / denominator,
    }


def _official_classifier_gates(
    metrics: dict[str, Any],
    *,
    minimum_approved_rate: float,
    maximum_approved_misrecognition_rate: float,
    maximum_unknown_top3_candidate_out_rate: float,
) -> dict[str, bool]:
    gates = {
        "approved_rate": metrics["approved_rate"] >= minimum_approved_rate,
        "approved_misrecognition_rate": metrics["approved_misrecognition_rate"]
        <= maximum_approved_misrecognition_rate,
        "unknown_top3_candidate_out_rate": metrics["unknown_top3_candidate_out_rate"]
        <= maximum_unknown_top3_candidate_out_rate,
    }
    return {**gates, "all_met": all(gates.values())}


def build_rate_policy(
    left_logits: np.ndarray,
    right_logits: np.ndarray,
    *,
    left_name: str,
    right_name: str,
    first_view_weight: float,
    ranking_tie_break_bias_span: float,
) -> Policy:
    if not 0.0 < first_view_weight < 1.0:
        raise ValueError("first-view weight must be strictly between zero and one")
    views = [left_logits, right_logits]
    weights = np.asarray([first_view_weight, 1.0 - first_view_weight], dtype=np.float32)
    classification_logits = np.sum(np.stack(views) * weights[:, None, None], axis=0)
    predictions, approval_features = _approval_features(
        classification_logits,
        {left_name: left_logits, right_name: right_logits},
    )
    ranking = _weighted_rankings(
        views,
        weights,
        ranking_tie_break_bias_span=ranking_tie_break_bias_span,
    )["weighted_reciprocal_rank"]
    top3 = np.argsort(-ranking, axis=1, kind="stable")[:, :3]
    unbiased_ranking = (
        ranking
        - np.linspace(
            0.0,
            -ranking_tie_break_bias_span,
            ranking.shape[1],
            dtype=np.float32,
        )[None, :]
    )
    safety_score = _spatial_safety_features(unbiased_ranking, top3, views)["inverse_entropy"]
    return Policy(
        name=(
            f"{left_name}@{first_view_weight:.2f}+"
            f"{right_name}@{1.0 - first_view_weight:.2f}:"
            "approval=class_conditional_margin:"
            "top3=weighted_reciprocal_rank:safety=inverse_entropy"
        ),
        predictions=predictions,
        approval_score=approval_features["margin"],
        top3=top3,
        top3_safety_score=safety_score,
    )


def evaluate_rate_candidate(
    policy: Policy,
    targets: np.ndarray,
    folds: np.ndarray,
    evaluation_mask: np.ndarray,
    *,
    first_view_weight: float,
    class_count: int,
    approval_guard_samples: int,
    safety_guard_samples: int,
    no_failure_approval_threshold: float,
    minimum_approved_rate: float,
    maximum_approved_misrecognition_rate: float,
    maximum_unknown_top3_candidate_out_rate: float,
    confirmation_predictions: np.ndarray | None = None,
) -> RatePolicyResult:
    approved = np.zeros(len(targets), dtype=bool)
    unknown = np.zeros(len(targets), dtype=bool)
    top3_correct = np.any(policy.top3 == targets[:, None], axis=1)
    fold_reports: list[dict[str, Any]] = []
    confirmation = (
        np.ones(len(targets), dtype=bool)
        if confirmation_predictions is None
        else unanimous_top1_confirmation(policy.predictions, confirmation_predictions)
    )
    for held_out_fold in sorted(set(folds.tolist())):
        calibration = folds != held_out_fold
        held_out = folds == held_out_fold
        approval_thresholds = _class_conditional_thresholds(
            policy,
            targets,
            calibration,
            class_count=class_count,
            guard_samples=approval_guard_samples,
            no_failure_threshold=no_failure_approval_threshold,
        )
        calibrated_approved = _approved(policy, approval_thresholds)
        safety_threshold = _guarded_threshold(
            policy.top3_safety_score,
            (~calibrated_approved) & (~top3_correct),
            calibration,
            safety_guard_samples,
        )
        approved[held_out] = calibrated_approved[held_out]
        unknown[held_out] = (~approved[held_out]) & (
            policy.top3_safety_score[held_out] >= safety_threshold
        )
        withheld = held_out & approved & (~confirmation)
        approved[withheld] = False
        unknown[withheld] = True
        fold_mask = evaluation_mask & held_out
        fold_reports.append(
            {
                "held_out_fold": int(held_out_fold),
                "approval_thresholds": approval_thresholds.tolist(),
                "top3_safety_threshold": safety_threshold,
                "confirmation_withheld_approved_count": int(np.count_nonzero(fold_mask & withheld)),
                "metrics": _metrics(policy, targets, fold_mask, approved, unknown),
            }
        )
    metrics = _metrics(policy, targets, evaluation_mask, approved, unknown)
    gates = _official_classifier_gates(
        metrics,
        minimum_approved_rate=minimum_approved_rate,
        maximum_approved_misrecognition_rate=maximum_approved_misrecognition_rate,
        maximum_unknown_top3_candidate_out_rate=maximum_unknown_top3_candidate_out_rate,
    )
    return RatePolicyResult(
        first_view_weight=first_view_weight,
        policy=policy,
        approved=approved,
        unknown=unknown,
        confirmation=confirmation,
        fold_reports=tuple(fold_reports),
        metrics=metrics,
        gates=gates,
    )


def _selection_key(result: RatePolicyResult) -> tuple[Any, ...]:
    gates_met = sum(value for name, value in result.gates.items() if name != "all_met")
    safety_errors = (
        result.metrics["approved_misrecognition_count"]
        + result.metrics["unknown_top3_candidate_out_count"]
    )
    return (
        result.gates["all_met"],
        gates_met,
        -safety_errors,
        result.metrics["approved_count"],
        -result.metrics["segment_recapture_count"],
        result.first_view_weight,
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    payload = np.load(args.logits)
    targets = payload["targets"].astype(np.int64)
    view_names = [name for name in payload.files if name != "targets"]
    if len(view_names) != 2:
        raise ValueError("rate policy requires exactly two classifier logit views")
    rows = [
        json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line
    ]
    if len(rows) != len(targets):
        raise ValueError("classifier records and logits have different lengths")
    group_overlap_count = validate_group_folds(rows)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    excluded_ids = set(args.evaluation_exclude_image_ids)
    evaluation_mask = np.asarray(
        [int(row["image_id"]) not in excluded_ids for row in rows],
        dtype=bool,
    )
    left = payload[view_names[0]].astype(np.float32)
    right = payload[view_names[1]].astype(np.float32)
    confirmation_views: list[np.ndarray] = []
    for path in args.confirmation_logits:
        confirmation_payload = np.load(path)
        if not np.array_equal(confirmation_payload["targets"], targets):
            raise ValueError(f"confirmation logits targets differ: {path}")
        if args.confirmation_logit_key not in confirmation_payload.files:
            raise ValueError(
                f"confirmation logit key {args.confirmation_logit_key!r} missing: {path}"
            )
        values = confirmation_payload[args.confirmation_logit_key]
        if values.shape != left.shape:
            raise ValueError(f"confirmation logits shape differs: {path}")
        confirmation_views.append(np.argmax(values, axis=1))
    confirmation_predictions = np.stack(confirmation_views, axis=1) if confirmation_views else None
    class_count = left.shape[1]
    candidates: list[RatePolicyResult] = []
    for first_view_weight in args.first_view_weights:
        policy = build_rate_policy(
            left,
            right,
            left_name=view_names[0],
            right_name=view_names[1],
            first_view_weight=float(first_view_weight),
            ranking_tie_break_bias_span=args.ranking_tie_break_bias_span,
        )
        candidates.append(
            evaluate_rate_candidate(
                policy,
                targets,
                folds,
                evaluation_mask,
                first_view_weight=float(first_view_weight),
                class_count=class_count,
                approval_guard_samples=args.approval_guard_samples,
                safety_guard_samples=args.safety_guard_samples,
                no_failure_approval_threshold=args.no_failure_approval_threshold,
                minimum_approved_rate=args.minimum_approved_rate,
                maximum_approved_misrecognition_rate=(args.maximum_approved_misrecognition_rate),
                maximum_unknown_top3_candidate_out_rate=(
                    args.maximum_unknown_top3_candidate_out_rate
                ),
                confirmation_predictions=confirmation_predictions,
            )
        )
    selected = max(candidates, key=_selection_key)
    full = np.ones(len(targets), dtype=bool)
    final_approval_thresholds = _class_conditional_thresholds(
        selected.policy,
        targets,
        full,
        class_count=class_count,
        guard_samples=args.approval_guard_samples,
        no_failure_threshold=args.no_failure_approval_threshold,
    )
    final_approved = _approved(selected.policy, final_approval_thresholds)
    top3_correct = np.any(selected.policy.top3 == targets[:, None], axis=1)
    final_safety_threshold = _guarded_threshold(
        selected.policy.top3_safety_score,
        (~final_approved) & (~top3_correct),
        full,
        args.safety_guard_samples,
    )
    final_unknown = (~final_approved) & (
        selected.policy.top3_safety_score >= final_safety_threshold
    )
    final_confirmation = (
        np.ones(len(targets), dtype=bool)
        if confirmation_predictions is None
        else unanimous_top1_confirmation(
            selected.policy.predictions,
            confirmation_predictions,
        )
    )
    final_withheld = final_approved & (~final_confirmation)
    final_approved[final_withheld] = False
    final_unknown[final_withheld] = True
    final_metrics = _metrics(
        selected.policy,
        targets,
        evaluation_mask,
        final_approved,
        final_unknown,
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_classifier_grouped_rate_policy",
        "source_dataset": args.source_dataset,
        "mixed_support_sources": False,
        "selection_scope": "grouped-development-oof-no-locked-test",
        "sample_count": len(targets),
        "evaluation_sample_count": int(np.count_nonzero(evaluation_mask)),
        "group_fold_overlap_count": group_overlap_count,
        "official_classifier_gates": {
            "minimum_approved_rate": args.minimum_approved_rate,
            "maximum_approved_misrecognition_rate": (args.maximum_approved_misrecognition_rate),
            "maximum_unknown_top3_candidate_out_rate": (
                args.maximum_unknown_top3_candidate_out_rate
            ),
        },
        "diagnostic_only_metrics": ["unknown_rate", "segment_recapture_rate"],
        "selection": {
            "candidate_first_view_weights": args.first_view_weights,
            "approval_guard_samples": args.approval_guard_samples,
            "safety_guard_samples": args.safety_guard_samples,
            "no_failure_approval_threshold": args.no_failure_approval_threshold,
            "ranking_tie_break_bias_span": args.ranking_tie_break_bias_span,
            "confirmation_policy": ("none" if not confirmation_views else "unanimous_top1"),
            "confirmation_logit_key": args.confirmation_logit_key,
            "confirmation_view_count": len(confirmation_views),
            "confirmation_logits": [str(path) for path in args.confirmation_logits],
        },
        "selected": {
            "policy": selected.policy.name,
            "first_view_weight": selected.first_view_weight,
            "grouped_oof": {
                "metrics": selected.metrics,
                "gates": selected.gates,
                "folds": selected.fold_reports,
            },
            "full_development_calibration_for_package": {
                "approval_thresholds": final_approval_thresholds.tolist(),
                "top3_safety_threshold": final_safety_threshold,
                "confirmation_withheld_approved_count": int(
                    np.count_nonzero(evaluation_mask & final_withheld)
                ),
                "metrics": final_metrics,
            },
        },
        "candidates": [
            {
                "first_view_weight": candidate.first_view_weight,
                "metrics": candidate.metrics,
                "gates": candidate.gates,
            }
            for candidate in candidates
        ],
        "evaluation_excluded_image_ids": sorted(excluded_ids),
        "promotion_ready": False,
        "promotion_blocker": "independent locked test and deployable package validation pending",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.decisions_output is not None:
        args.decisions_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.decisions_output,
            targets=targets,
            evaluation_mask=evaluation_mask,
            predictions=selected.policy.predictions,
            top3=selected.policy.top3,
            grouped_oof_approved=selected.approved,
            grouped_oof_unknown=selected.unknown,
            final_approved=final_approved,
            final_unknown=final_unknown,
            confirmation=selected.confirmation,
        )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the Bread 1.1 grouped classifier rate policy"
    )
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions-output", type=Path)
    parser.add_argument(
        "--source-dataset", choices=("single_objects", "single_objects_2"), required=True
    )
    parser.add_argument(
        "--first-view-weights",
        type=float,
        nargs="+",
        default=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95),
    )
    parser.add_argument("--approval-guard-samples", type=int, default=2)
    parser.add_argument("--safety-guard-samples", type=int, default=8)
    parser.add_argument("--no-failure-approval-threshold", type=float, default=0.98)
    parser.add_argument("--ranking-tie-break-bias-span", type=float, default=0.0002)
    parser.add_argument("--confirmation-logits", type=Path, nargs="*", default=())
    parser.add_argument("--confirmation-logit-key", default="normalized_bias0.000")
    parser.add_argument("--minimum-approved-rate", type=float, default=0.90)
    parser.add_argument("--maximum-approved-misrecognition-rate", type=float, default=0.001)
    parser.add_argument("--maximum-unknown-top3-candidate-out-rate", type=float, default=0.001)
    parser.add_argument("--evaluation-exclude-image-ids", type=int, nargs="*", default=())
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
