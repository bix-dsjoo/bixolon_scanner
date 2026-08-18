from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


def l2_normalized_logit_margin(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-values, axis=1, kind="stable")
    ranked = np.take_along_axis(values, order, axis=1)
    margins = ranked[:, 0] - ranked[:, 1]
    return margins / np.linalg.norm(values, axis=1).clip(min=1e-12)


def apply_normalized_margin_thresholds(
    scores: np.ndarray, thresholds: list[float | None]
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    predictions = np.argmax(values, axis=1)
    confidence = l2_normalized_logit_margin(values)
    rejected = np.zeros(len(values), dtype=bool)
    for class_id, threshold in enumerate(thresholds):
        if threshold is not None:
            rejected |= (predictions == class_id) & (confidence < threshold)
    return rejected


def select_normalized_margin_policy(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    maximum_approved_errors: int,
    maximum_unknown_top3_misses: int,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    predictions = np.argmax(values, axis=1)
    order = np.argsort(-values, axis=1, kind="stable")
    confidence = l2_normalized_logit_margin(values)
    errors = np.flatnonzero(predictions != labels)
    top3_misses = ~np.any(order[:, :3] == labels[:, None], axis=1)
    candidates = []
    for leave_count in range(min(maximum_approved_errors, len(errors)) + 1):
        for approved_error_indices in itertools.combinations(errors.tolist(), leave_count):
            approved_errors = set(approved_error_indices)
            thresholds: list[float | None] = [None] * values.shape[1]
            for index in errors:
                if int(index) in approved_errors:
                    continue
                class_id = int(predictions[index])
                current = thresholds[class_id]
                threshold = float(np.nextafter(confidence[index], np.inf))
                thresholds[class_id] = threshold if current is None else max(current, threshold)
            rejected = apply_normalized_margin_thresholds(values, thresholds)
            approved_errors_count = int(np.count_nonzero((predictions != labels) & ~rejected))
            candidate_out_count = int(np.count_nonzero(top3_misses & rejected))
            if (
                approved_errors_count <= maximum_approved_errors
                and candidate_out_count <= maximum_unknown_top3_misses
            ):
                candidates.append(
                    {
                        "thresholds": thresholds,
                        "rejected": rejected,
                        "approved_count": int(np.count_nonzero(~rejected)),
                        "approved_error_count": approved_errors_count,
                        "unknown_count": int(np.count_nonzero(rejected)),
                        "unknown_top3_miss_count": candidate_out_count,
                    }
                )
    if not candidates:
        raise ValueError("no normalized margin policy satisfies the supplied error limits")
    return max(
        candidates,
        key=lambda row: (
            row["approved_count"],
            -row["approved_error_count"],
            -row["unknown_top3_miss_count"],
        ),
    )


def policy_metrics(scores: np.ndarray, targets: np.ndarray, rejected: np.ndarray) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    predictions = np.argmax(values, axis=1)
    order = np.argsort(-values, axis=1, kind="stable")
    top3_misses = ~np.any(order[:, :3] == labels[:, None], axis=1)
    sample_count = len(labels)
    return {
        "sample_count": sample_count,
        "approved_count": int(np.count_nonzero(~rejected)),
        "approved_rate": float(np.count_nonzero(~rejected) / sample_count),
        "approved_error_count": int(np.count_nonzero((predictions != labels) & ~rejected)),
        "approved_error_rate_all_gt": float(
            np.count_nonzero((predictions != labels) & ~rejected) / sample_count
        ),
        "unknown_count": int(np.count_nonzero(rejected)),
        "unknown_rate": float(np.count_nonzero(rejected) / sample_count),
        "unknown_top3_miss_count": int(np.count_nonzero(top3_misses & rejected)),
        "unknown_top3_miss_rate_all_gt": float(
            np.count_nonzero(top3_misses & rejected) / sample_count
        ),
    }


def _operational_matched_scores(
    cache_path: Path, decisions_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    cache = np.load(cache_path)
    by_key = {
        (int(image_id), int(detection_index)): score
        for image_id, detection_index, score in zip(
            cache["image_ids"], cache["detection_indices"], cache["scores"]
        )
    }
    decisions = [
        json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines() if line
    ]
    scores = np.asarray(
        [by_key[(int(row["image_id"]), int(row["detection_index"]))] for row in decisions],
        dtype=np.float64,
    )
    targets = np.asarray([int(row["target"]) for row in decisions], dtype=np.int64)
    return scores, targets


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    oof = np.load(args.oof_logits)
    oof_scores = np.asarray(oof["scores"], dtype=np.float64)
    oof_targets = np.asarray(oof["targets"], dtype=np.int64)
    operational_scores, operational_targets = _operational_matched_scores(
        args.operational_scores, args.operational_decisions
    )
    scores = np.concatenate((oof_scores, operational_scores))
    targets = np.concatenate((oof_targets, operational_targets))
    policy = select_normalized_margin_policy(
        scores,
        targets,
        maximum_approved_errors=args.maximum_approved_errors,
        maximum_unknown_top3_misses=args.maximum_unknown_top3_misses,
    )
    rejected = np.asarray(policy.pop("rejected"), dtype=bool)
    split = len(oof_scores)
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_classifier_l2_normalized_margin_policy",
        "selection_scope": (
            "grouped multi-object OOF plus the rejected v2 operational set as development"
        ),
        "confidence_metric": "(top1_logit - top2_logit) / L2(logits)",
        "threshold_boundary": "APPROVED when confidence >= threshold",
        "policy": policy,
        "combined_development": policy_metrics(scores, targets, rejected),
        "multi_object_grouped_oof": policy_metrics(oof_scores, oof_targets, rejected[:split]),
        "rejected_v2_operational_development": policy_metrics(
            operational_scores, operational_targets, rejected[split:]
        ),
        "independent_evidence": False,
        "new_independent_test_required": True,
        "promotion_ready": False,
    }
    args.decisions_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.decisions_output,
        scores=scores.astype(np.float32),
        targets=targets,
        rejected=rejected,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a scale-stable LDA margin policy")
    parser.add_argument("--oof-logits", type=Path, required=True)
    parser.add_argument("--operational-scores", type=Path, required=True)
    parser.add_argument("--operational-decisions", type=Path, required=True)
    parser.add_argument("--maximum-approved-errors", type=int, default=1)
    parser.add_argument("--maximum-unknown-top3-misses", type=int, default=1)
    parser.add_argument("--decisions-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
