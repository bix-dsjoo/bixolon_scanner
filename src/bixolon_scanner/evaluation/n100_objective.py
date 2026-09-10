"""Object-level log132 objective, without hiding recaptures or hardware limits."""

from __future__ import annotations


def assess(summary: dict, targets: dict) -> dict:
    if (
        summary["image_count"] != targets["image_count"]
        or summary["ground_truth_count"] != targets["ground_truth_count"]
    ):
        raise ValueError("N100 evaluation denominator differs from the frozen inputs")
    accuracy = summary["correct_approved_count"] >= targets["minimum_correct_approved_count"]
    wrong = summary["wrong_approved_count"] <= targets["maximum_wrong_approved_count"]
    missed = summary["missed_count"] <= targets["maximum_missed_count"]
    errors = summary["image_status_counts"].get("ERROR", 0) == 0
    speed = all(
        summary["latency"][name]["p95_ms"] is not None
        and summary["latency"][name]["p95_ms"] <= targets["maximum_http_p95_ms"]
        for name in ("all", "full_path")
    )
    return {
        "recognition_met": accuracy,
        "zero_wrong_approval_met": wrong,
        "zero_missed_met": missed,
        "zero_errors_met": errors,
        "http_p95_met": speed,
        "measurement_targets_met": accuracy and wrong and missed and errors and speed,
        "hardware_scope": "This verdict applies only to the hardware recorded by the measurement; it does not establish N100 performance on a different host.",
    }
