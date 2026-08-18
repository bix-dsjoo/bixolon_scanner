from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .zero_error_classifier import (
    _guarded_threshold,
    policy_candidates,
    policy_metrics,
    select_policy,
)


def exclude_image_records(
    targets: np.ndarray,
    logits: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    excluded_image_ids: set[int],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    observed_image_ids = {int(row["image_id"]) for row in rows}
    missing = excluded_image_ids - observed_image_ids
    if missing:
        raise ValueError(f"excluded image ids are absent from classifier records: {missing}")
    retained = np.asarray(
        [int(row["image_id"]) not in excluded_image_ids for row in rows],
        dtype=bool,
    )
    return (
        targets[retained],
        {name: values[retained] for name, values in logits.items()},
        [row for row, keep in zip(rows, retained) if keep],
    )


def evaluate_guard(
    candidates,
    targets: np.ndarray,
    folds: np.ndarray,
    guard_samples: int,
    records: list[dict[str, Any]] | None = None,
    evaluation_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    full = np.ones(len(targets), dtype=bool)
    policy, approval_threshold, safety_threshold, selected = select_policy(
        candidates, targets, full, guard_samples
    )
    totals = {
        "sample_count": 0,
        "approved_count": 0,
        "approved_error_count": 0,
        "unknown_count": 0,
        "unknown_top3_miss_count": 0,
        "segment_recapture_count": 0,
        "non_recapture_count": 0,
    }
    oof_approved = np.zeros(len(targets), dtype=bool)
    oof_predictions = np.zeros(len(targets), dtype=np.int64)
    oof_top3 = np.zeros((len(targets), 3), dtype=np.int64)
    oof_safety_scores = np.zeros(len(targets), dtype=np.float32)
    fold_rows = []
    for held_out_fold in sorted(set(folds.tolist())):
        calibration = folds != held_out_fold
        held_out = folds == held_out_fold
        fold_policy, fold_approval, fold_safety, _ = select_policy(
            candidates, targets, calibration, guard_samples
        )
        metrics = policy_metrics(
            fold_policy,
            targets,
            held_out,
            fold_approval,
            fold_safety,
        )
        approved = held_out & (fold_policy.approval_score >= fold_approval)
        unknown = held_out & ~approved & (fold_policy.top3_safety_score >= fold_safety)
        approved_errors = approved & (fold_policy.predictions != targets)
        top3_correct = np.any(fold_policy.top3 == targets[:, None], axis=1)
        unknown_misses = unknown & ~top3_correct
        oof_approved[held_out] = approved[held_out]
        oof_predictions[held_out] = fold_policy.predictions[held_out]
        oof_top3[held_out] = fold_policy.top3[held_out]
        oof_safety_scores[held_out] = fold_policy.top3_safety_score[held_out]
        for key in totals:
            totals[key] += int(metrics[key])
        fold_rows.append(
            {
                "held_out_fold": held_out_fold,
                "policy": fold_policy.name,
                "approval_threshold": fold_approval,
                "top3_safety_threshold": fold_safety,
                "metrics": metrics,
                "approved_error_records": [
                    records[index] if records is not None else {"index": index}
                    for index in np.flatnonzero(approved_errors)
                ],
                "unknown_top3_miss_records": [
                    records[index] if records is not None else {"index": index}
                    for index in np.flatnonzero(unknown_misses)
                ],
            }
        )
    sample_count = totals["sample_count"]
    grouped_oof = {
        **totals,
        "approved_rate": totals["approved_count"] / sample_count,
        "unknown_rate": totals["unknown_count"] / sample_count,
        "segment_recapture_rate": totals["segment_recapture_count"] / sample_count,
        "non_recapture_rate": totals["non_recapture_count"] / sample_count,
        "folds": fold_rows,
    }
    pooled_top3_correct = np.any(oof_top3 == targets[:, None], axis=1)
    pooled_top3_failures = ~oof_approved & ~pooled_top3_correct
    pooled_safety_threshold = _guarded_threshold(
        oof_safety_scores,
        pooled_top3_failures,
        np.ones(len(targets), dtype=bool),
        0,
    )
    pooled_unknown = ~oof_approved & (oof_safety_scores >= pooled_safety_threshold)
    pooled_recapture = ~oof_approved & ~pooled_unknown
    pooled_approved_errors = oof_approved & (oof_predictions != targets)
    pooled_unknown_misses = pooled_unknown & ~pooled_top3_correct
    pooled_oof_calibration = {
        "selection_scope": "all-available-oof-safety-scores-no-locked-test",
        "top3_safety_threshold": pooled_safety_threshold,
        "sample_count": len(targets),
        "approved_count": int(oof_approved.sum()),
        "approved_error_count": int(pooled_approved_errors.sum()),
        "unknown_count": int(pooled_unknown.sum()),
        "unknown_top3_miss_count": int(pooled_unknown_misses.sum()),
        "segment_recapture_count": int(pooled_recapture.sum()),
        "segment_recapture_rate": float(pooled_recapture.mean()),
        "non_recapture_count": int((oof_approved | pooled_unknown).sum()),
        "non_recapture_rate": float((oof_approved | pooled_unknown).mean()),
    }
    if evaluation_mask is None:
        evaluation_mask = np.ones(len(targets), dtype=bool)
    if evaluation_mask.shape != targets.shape:
        raise ValueError("classifier evaluation mask has the wrong shape")
    evaluation_count = int(evaluation_mask.sum())
    subset_approved = evaluation_mask & oof_approved
    subset_unknown = evaluation_mask & pooled_unknown
    subset_recapture = evaluation_mask & pooled_recapture
    evaluation_subset = {
        "selection_scope": "locked full-data OOF policy evaluated after image gate",
        "sample_count": evaluation_count,
        "approved_count": int(subset_approved.sum()),
        "approved_error_count": int(
            np.count_nonzero(subset_approved & (oof_predictions != targets))
        ),
        "unknown_count": int(subset_unknown.sum()),
        "unknown_top3_miss_count": int(np.count_nonzero(subset_unknown & ~pooled_top3_correct)),
        "segment_recapture_count": int(subset_recapture.sum()),
        "segment_recapture_rate": (
            float(subset_recapture.sum() / evaluation_count) if evaluation_count else None
        ),
        "non_recapture_count": int((subset_approved | subset_unknown).sum()),
        "non_recapture_rate": (
            float((subset_approved | subset_unknown).sum() / evaluation_count)
            if evaluation_count
            else None
        ),
    }
    return {
        "guard_samples": guard_samples,
        "selected": {
            "policy": policy.name,
            "approval_threshold": approval_threshold,
            "top3_safety_threshold": safety_threshold,
            "metrics": selected,
        },
        "grouped_oof": grouped_oof,
        "pooled_oof_calibration": pooled_oof_calibration,
        "evaluation_subset": evaluation_subset,
        "passes_zero_error_gates": (
            selected["approved_error_count"] == 0
            and selected["unknown_top3_miss_count"] == 0
            and grouped_oof["approved_error_count"] == 0
            and grouped_oof["unknown_top3_miss_count"] == 0
        ),
        "passes_pooled_zero_error_gates": (
            selected["approved_error_count"] == 0
            and selected["unknown_top3_miss_count"] == 0
            and pooled_oof_calibration["approved_error_count"] == 0
            and pooled_oof_calibration["unknown_top3_miss_count"] == 0
        ),
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
    excluded_image_ids = set(args.exclude_image_ids or [])
    if excluded_image_ids:
        targets, logits, rows = exclude_image_records(targets, logits, rows, excluded_image_ids)
    evaluation_excluded_image_ids = set(args.evaluation_exclude_image_ids or [])
    observed_image_ids = {int(row["image_id"]) for row in rows}
    missing_evaluation_ids = evaluation_excluded_image_ids - observed_image_ids
    if missing_evaluation_ids:
        raise ValueError(
            "evaluation-excluded image ids are absent from classifier records: "
            f"{missing_evaluation_ids}"
        )
    evaluation_mask = np.asarray(
        [int(row["image_id"]) not in evaluation_excluded_image_ids for row in rows],
        dtype=bool,
    )
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    candidates = policy_candidates(logits)
    evaluations = [
        evaluate_guard(
            candidates,
            targets,
            folds,
            guard,
            rows,
            evaluation_mask=evaluation_mask,
        )
        for guard in args.guard_samples
    ]
    passing = [row for row in evaluations if row["passes_zero_error_gates"]]
    pooled_passing = [row for row in evaluations if row["passes_pooled_zero_error_gates"]]
    selected = (
        min(
            passing,
            key=lambda row: (
                row["grouped_oof"]["segment_recapture_count"],
                row["selected"]["metrics"]["segment_recapture_count"],
                row["guard_samples"],
            ),
        )
        if passing
        else None
    )
    selected_pooled = (
        min(
            pooled_passing,
            key=lambda row: (
                row["pooled_oof_calibration"]["segment_recapture_count"],
                row["selected"]["metrics"]["segment_recapture_count"],
                row["guard_samples"],
            ),
        )
        if pooled_passing
        else None
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_classifier_zero_error_guard_search",
        "source_dataset": args.source_dataset,
        "excluded_image_ids": sorted(excluded_image_ids),
        "evaluation_excluded_image_ids": sorted(evaluation_excluded_image_ids),
        "sample_count": len(targets),
        "policy_candidate_count": len(candidates),
        "guard_candidate_count": len(evaluations),
        "zero_error_guard_count": len(passing),
        "selected": selected,
        "pooled_zero_error_guard_count": len(pooled_passing),
        "selected_pooled": selected_pooled,
        "candidates": evaluations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search classifier safety guards without reloading view logits"
    )
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-dataset",
        choices=("single_objects", "single_objects_2"),
        required=True,
    )
    parser.add_argument("--guard-samples", type=int, nargs="+", required=True)
    parser.add_argument("--exclude-image-ids", type=int, nargs="+")
    parser.add_argument("--evaluation-exclude-image-ids", type=int, nargs="+")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
