"""Reject accuracy gains achieved by losing correct objects or changing denominators."""

from __future__ import annotations

import math


def assess_model_change(
    baseline: dict, candidate: dict, baseline_source: dict, candidate_source: dict, policy: dict
) -> dict:
    old_repeats, new_repeats = baseline["repetitions"], candidate["repetitions"]
    if not old_repeats or len(old_repeats) != len(new_repeats):
        raise ValueError("model comparison must have the same nonzero repetition count")
    reasons = []
    gains = []
    for index, (old, new) in enumerate(zip(old_repeats, new_repeats, strict=True)):
        if any(old[k] != new[k] for k in ("image_count", "ground_truth_count")):
            raise ValueError("model comparison denominators changed")
        gain = new["correct_approved_count"] - old["correct_approved_count"]
        reduction = old["wrong_approved_count"] - new["wrong_approved_count"]
        gains.append({"repeat": index + 1, "correct_gain": gain, "wrong_reduction": reduction})
        if gain < 0:
            reasons.append(f"repeat_{index + 1}:correct_approvals_decreased")
        if reduction < 0:
            reasons.append(f"repeat_{index + 1}:wrong_approvals_increased")
        if (
            gain < policy["minimum_correct_approval_gain"]
            and reduction < policy["alternative_minimum_wrong_approval_reduction"]
        ):
            reasons.append(f"repeat_{index + 1}:insufficient_improvement")
        if new["image_status_counts"].get("ERROR", 0):
            reasons.append(f"repeat_{index + 1}:worker_errors")
        for group in ("all", "full_path"):
            before, after = old["latency"][group]["p95_ms"], new["latency"][group]["p95_ms"]
            if (
                before is None
                or after is None
                or not math.isfinite(before)
                or not math.isfinite(after)
            ):
                reasons.append(f"repeat_{index + 1}:{group}_latency_missing")
            elif after > before * policy["maximum_cpu_p95_ratio"]:
                reasons.append(f"repeat_{index + 1}:{group}_latency_regression")
            elif before <= policy["maximum_cpu_p95_ms_if_baseline_passes"] < after:
                reasons.append(f"repeat_{index + 1}:{group}_300ms_regression")
    source_old, source_new = baseline_source["summary"], candidate_source["summary"]
    if any(source_old[k] != source_new[k] for k in ("image_count", "ground_truth_count")):
        raise ValueError("source regression denominators changed")
    if source_new["correct_approved_count"] < source_old["correct_approved_count"]:
        reasons.append("source_correct_approvals_decreased")
    if source_new["wrong_approved_count"] > source_old["wrong_approved_count"]:
        reasons.append("source_wrong_approvals_increased")
    if source_new["image_status_counts"].get("ERROR", 0):
        reasons.append("source_worker_errors")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "repeat_gains": gains,
        "source_correct_gain": source_new["correct_approved_count"]
        - source_old["correct_approved_count"],
    }
