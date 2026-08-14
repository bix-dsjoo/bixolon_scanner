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


def _approval_threshold(confidence: np.ndarray, coverage: float) -> float:
    return float(np.quantile(confidence, 1.0 - coverage, method="lower"))


def _ranking_candidates(
    logits: dict[str, np.ndarray], names: tuple[str, ...]
) -> dict[str, np.ndarray]:
    probabilities = np.stack([_softmax(logits[name]) for name in names])
    rank_scores = np.zeros_like(probabilities, dtype=np.float32)
    top3_votes = np.zeros_like(probabilities, dtype=np.float32)
    for view_index, name in enumerate(names):
        order = np.argsort(-logits[name], axis=1, kind="stable")
        ranks = np.empty_like(order)
        ranks[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])[None, :]
        rank_scores[view_index] = 1.0 / (ranks + 1.0)
        top3_votes[view_index] = ranks < 3
    mean_probability = probabilities.mean(axis=0)
    return {
        "mean_logits": np.mean([logits[name] for name in names], axis=0),
        "mean_probability": mean_probability,
        "maximum_probability": probabilities.max(axis=0),
        "reciprocal_rank": rank_scores.mean(axis=0),
        "top3_vote": top3_votes.mean(axis=0) + mean_probability * 1e-3,
    }


def _outcome(
    *,
    predictions: np.ndarray,
    approved: np.ndarray,
    top3: np.ndarray,
    targets: np.ndarray,
    ground_truth_count: int,
) -> dict[str, Any]:
    unknown = ~approved
    matched = targets >= 0
    approved_errors = int(np.count_nonzero(approved & (~matched | (predictions != targets))))
    recognized = matched & (
        (approved & (predictions == targets)) | (unknown & np.any(top3 == targets[:, None], axis=1))
    )
    approved_count = int(np.count_nonzero(approved))
    recognized_count = int(np.count_nonzero(recognized))
    unknown_matched = unknown & matched
    return {
        "matched_sample_count": int(np.count_nonzero(matched)),
        "ground_truth_count": ground_truth_count,
        "detector_missed_count": ground_truth_count - int(np.count_nonzero(matched)),
        "false_segmentation_count": int(np.count_nonzero(~matched)),
        "approved_count": approved_count,
        "approval_coverage": approved_count / len(targets),
        "approved_error_count": approved_errors,
        "approved_misrecognition_rate": approved_errors / approved_count,
        "unknown_count": int(np.count_nonzero(unknown)),
        "unknown_top3_accuracy": (
            float(np.any(top3[unknown_matched] == targets[unknown_matched, None], axis=1).mean())
            if np.any(unknown_matched)
            else None
        ),
        "matched_recognized_count": recognized_count,
        "matched_recognition_rate": recognized_count / int(np.count_nonzero(matched)),
        "overall_recognition_rate": recognized_count / ground_truth_count,
    }


def _passes(metrics: dict[str, Any], *, minimum_coverage: float) -> bool:
    return bool(
        metrics["overall_recognition_rate"] >= 0.99
        and metrics["approved_misrecognition_rate"] <= 0.001
        and metrics["approval_coverage"] >= minimum_coverage
    )


def select_final_policy(
    logits: dict[str, np.ndarray],
    targets: np.ndarray,
    *,
    ground_truth_count: int,
    coverage: float,
    maximum_views: int,
) -> dict[str, Any]:
    names = tuple(logits)
    candidates = []
    for size in range(1, maximum_views + 1):
        for selected in combinations(names, size):
            values = np.mean([logits[name] for name in selected], axis=0)
            probabilities = _softmax(values)
            confidence = probabilities.max(axis=1)
            threshold = _approval_threshold(confidence, coverage)
            approved = confidence >= threshold
            ranking_candidates = []
            for top3_size in range(len(selected), len(names) + 1):
                for top3_views in combinations(names, top3_size):
                    if not set(selected).issubset(top3_views):
                        continue
                    for aggregation, ranking in _ranking_candidates(logits, top3_views).items():
                        top3 = np.argsort(-ranking, axis=1, kind="stable")[:, :3]
                        metrics = _outcome(
                            predictions=values.argmax(axis=1),
                            approved=approved,
                            top3=top3,
                            targets=targets,
                            ground_truth_count=ground_truth_count,
                        )
                        ranking_candidates.append(
                            {
                                "top3_views": top3_views,
                                "top3_aggregation": aggregation,
                                "metrics": metrics,
                                "passes": _passes(metrics, minimum_coverage=coverage),
                            }
                        )
            best_ranking = max(
                ranking_candidates,
                key=lambda candidate: (
                    candidate["passes"],
                    candidate["metrics"]["overall_recognition_rate"],
                    -candidate["metrics"]["approved_error_count"],
                    -len(candidate["top3_views"]),
                ),
            )
            candidates.append(
                {
                    "views": selected,
                    "threshold": threshold,
                    **best_ranking,
                }
            )
    passing = [candidate for candidate in candidates if candidate["passes"]]
    pool = passing or candidates
    return max(
        pool,
        key=lambda candidate: (
            candidate["passes"],
            candidate["metrics"]["overall_recognition_rate"],
            -candidate["metrics"]["approved_error_count"],
            -len(candidate["views"]),
        ),
    )


def select_top3_view_budget(
    logits: dict[str, np.ndarray],
    targets: np.ndarray,
    *,
    final_policy: dict[str, Any],
    ground_truth_count: int,
    minimum_coverage: float,
) -> dict[str, Any]:
    names = tuple(logits)
    final_views = tuple(final_policy["views"])
    final_values = np.mean([logits[name] for name in final_views], axis=0)
    final_probabilities = _softmax(final_values)
    approved = final_probabilities.max(axis=1) >= float(final_policy["threshold"])
    predictions = final_values.argmax(axis=1)
    candidates = []
    for size in range(len(final_views), len(names) + 1):
        for top3_views in combinations(names, size):
            if not set(final_views).issubset(top3_views):
                continue
            top3 = np.argsort(
                -np.mean([logits[name] for name in top3_views], axis=0),
                axis=1,
                kind="stable",
            )[:, :3]
            metrics = _outcome(
                predictions=predictions,
                approved=approved,
                top3=top3,
                targets=targets,
                ground_truth_count=ground_truth_count,
            )
            if _passes(metrics, minimum_coverage=minimum_coverage):
                candidates.append({"top3_views": top3_views, "metrics": metrics, "passes": True})
        if candidates:
            break
    if not candidates:
        raise ValueError("no Top-3 view subset preserves the point gates")
    return max(
        candidates,
        key=lambda candidate: (
            candidate["metrics"]["overall_recognition_rate"],
            -candidate["metrics"]["approved_error_count"],
        ),
    )


def select_staged_policy(
    logits: dict[str, np.ndarray],
    targets: np.ndarray,
    image_ids: np.ndarray,
    *,
    final_policy: dict[str, Any],
    ground_truth_count: int,
    minimum_coverage: float,
    allowed_first_views: set[str] | None = None,
) -> dict[str, Any]:
    names = tuple(final_policy.get("top3_views", tuple(logits)))
    final_views = tuple(final_policy["views"])
    final_values = np.mean([logits[name] for name in final_views], axis=0)
    final_probabilities = _softmax(final_values)
    final_confidence = final_probabilities.max(axis=1)
    final_predictions = final_values.argmax(axis=1)
    final_approved = final_confidence >= float(final_policy["threshold"])
    ranking = _ranking_candidates(logits, names)[
        str(final_policy.get("top3_aggregation", "mean_logits"))
    ]
    all_view_top3 = np.argsort(-ranking, axis=1, kind="stable")[:, :3]
    candidates = []
    first_views = [
        name for name in final_views if allowed_first_views is None or name in allowed_first_views
    ]
    if not first_views:
        raise ValueError("final policy has no allowed first-stage view")
    for first_view in first_views:
        first_values = logits[first_view]
        first_probabilities = _softmax(first_values)
        first_confidence = first_probabilities.max(axis=1)
        first_predictions = first_values.argmax(axis=1)
        thresholds = {float(value) for value in first_confidence}
        thresholds.add(1.0)
        for threshold in sorted(thresholds):
            early = (
                np.zeros_like(first_confidence, dtype=bool)
                if threshold >= 1.0
                else first_confidence >= threshold
            )
            predictions = np.where(early, first_predictions, final_predictions)
            approved = early | (~early & final_approved)
            metrics = _outcome(
                predictions=predictions,
                approved=approved,
                top3=all_view_top3,
                targets=targets,
                ground_truth_count=ground_truth_count,
            )
            if not _passes(metrics, minimum_coverage=minimum_coverage):
                continue
            final_unknown = (~early) & (~final_approved)
            classifier_invocations = (
                len(targets)
                + (len(final_views) - 1) * int(np.count_nonzero(~early))
                + (len(names) - len(final_views)) * int(np.count_nonzero(final_unknown))
            )
            per_image = []
            for image_id in sorted(set(int(value) for value in image_ids)):
                mask = image_ids == image_id
                count = int(np.count_nonzero(mask))
                per_image.append(
                    count
                    + (len(final_views) - 1) * int(np.count_nonzero(mask & ~early))
                    + (len(names) - len(final_views)) * int(np.count_nonzero(mask & final_unknown))
                )
            candidates.append(
                {
                    "first_view": first_view,
                    "early_approval_threshold": threshold,
                    "early_approved_count": int(np.count_nonzero(early)),
                    "ambiguous_count": int(np.count_nonzero(~early)),
                    "final_unknown_count": int(np.count_nonzero(final_unknown)),
                    "mean_view_inferences_per_roi": classifier_invocations / len(targets),
                    "image_view_inferences": {
                        "mean": float(np.mean(per_image)),
                        "p50": float(np.percentile(per_image, 50)),
                        "p95": float(np.percentile(per_image, 95)),
                        "p99": float(np.percentile(per_image, 99)),
                        "maximum": int(max(per_image)),
                    },
                    "metrics": metrics,
                }
            )
    if not candidates:
        raise ValueError("no staged policy satisfies the point gates")
    return min(
        candidates,
        key=lambda candidate: (
            candidate["image_view_inferences"]["p95"],
            candidate["mean_view_inferences_per_roi"],
            candidate["metrics"]["approved_error_count"],
            -candidate["metrics"]["overall_recognition_rate"],
        ),
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    archive = np.load(args.logits)
    selected_names = set(args.views) if args.views else None
    logits = {
        name: archive[name].astype(np.float32)
        for name in archive.files
        if name != "targets" and (selected_names is None or name in selected_names)
    }
    targets = archive["targets"].astype(np.int64)
    rows = [
        json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line
    ]
    image_ids = np.asarray([int(row["image_id"]) for row in rows], dtype=np.int64)
    final_policy = select_final_policy(
        logits,
        targets,
        ground_truth_count=args.ground_truth_count,
        coverage=args.coverage,
        maximum_views=args.maximum_final_views,
    )
    staged_policy = select_staged_policy(
        logits,
        targets,
        image_ids,
        final_policy=final_policy,
        ground_truth_count=args.ground_truth_count,
        minimum_coverage=args.coverage,
        allowed_first_views=set(args.first_views) if args.first_views else None,
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "runtime_deployable_staged_classifier_policy",
        "selection_set": "multi_object_scenes",
        "recapture_policy_changed": False,
        "all_view_names": list(logits),
        "final_policy": final_policy,
        "staged_policy": staged_policy,
        "passes_point_gates": {
            "recognition_at_least_0_99": bool(
                staged_policy["metrics"]["overall_recognition_rate"] >= 0.99
            ),
            "approved_misrecognition_at_most_0_001": bool(
                staged_policy["metrics"]["approved_misrecognition_rate"] <= 0.001
            ),
            "approval_coverage_at_least_target": bool(
                staged_policy["metrics"]["approval_coverage"] >= args.coverage
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a deployable staged classifier policy")
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground-truth-count", type=int, required=True)
    parser.add_argument("--coverage", type=float, default=0.85)
    parser.add_argument("--maximum-final-views", type=int, default=5)
    parser.add_argument("--views", nargs="+")
    parser.add_argument("--first-views", nargs="+")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
