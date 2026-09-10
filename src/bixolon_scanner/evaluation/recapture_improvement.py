"""Compare measured recapture candidates without excluding missed or rejected GT."""

from __future__ import annotations

from statistics import median


def retained_prediction_parity(before: list[dict], after: list[dict]) -> dict:
    """Require retained boxes to keep their measured status, reasons and class ranks."""
    old = {r["image_id"]: r for r in before}
    new = {r["image_id"]: r for r in after}
    if len(old) != len(before) or len(new) != len(after) or old.keys() != new.keys():
        raise ValueError("retained parity requires unique matching image IDs")
    mismatches = []
    maximum_confidence_delta = 0.0
    for image_id, row in old.items():

        def key(seg):
            return tuple(seg["bbox"][k] for k in ("x", "y", "width", "height"))

        previous = {key(seg): seg for seg in row["response"]["segmentations"]}
        for seg in new[image_id]["response"]["segmentations"]:
            original = previous.get(key(seg))
            if (
                original is None
                or any(original[k] != seg[k] for k in ("status", "reason_codes", "prediction"))
                or [r["class_id"] for r in original["top3"]] != [r["class_id"] for r in seg["top3"]]
            ):
                mismatches.append({"image_id": image_id, "bbox": seg["bbox"]})
            if original is not None:
                maximum_confidence_delta = max(
                    maximum_confidence_delta, abs(original["confidence"] - seg["confidence"])
                )
    return {
        "unchanged_retained_decisions": not mismatches,
        "mismatches": mismatches,
        "maximum_confidence_delta": maximum_confidence_delta,
    }


def compare_repetitions(baseline: list[dict], candidate: list[dict]) -> dict:
    """Require complete, zero-error log repetitions and a non-regressing HTTP p95."""
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("equal nonempty repetition lists are required")
    checks = []
    for before, after in zip(baseline, candidate, strict=True):
        complete = all(
            row["image_count"] == 132
            and row["ground_truth_count"] == 1096
            and row["latency"]["all"]["count"] == 132
            for row in (before, after)
        )
        checks.append(
            {
                "complete_fixed_set": complete,
                "zero_wrong_approvals": after["wrong_approved_count"] == 0,
                "zero_missed": after["missed_count"] == 0 and after["matched_count"] == 1096,
                "approval_floor": after["correct_approved_count"]
                >= max(1078, before["correct_approved_count"]),
                "zero_errors": after["image_status_counts"].get("ERROR", 0) == 0,
                "fewer_recaptures": after["item_status_counts"].get("SEGMENT_RECAPTURE", 0)
                < before["item_status_counts"].get("SEGMENT_RECAPTURE", 0),
                "no_extra_regression": after["extra_count"] <= before["extra_count"],
            }
        )
    old = [r["latency"]["all"]["p95_ms"] for r in baseline]
    new = [r["latency"]["all"]["p95_ms"] for r in candidate]
    if any(value is None or value <= 0 for value in old + new):
        raise ValueError("HTTP p95 measurements are required")
    no_slower = median(new) <= median(old)
    return {
        "checks": checks,
        "baseline_p95_ms": old,
        "candidate_p95_ms": new,
        "median_p95_ratio": median(new) / median(old),
        "every_pair_no_slower": all(b <= a for a, b in zip(old, new, strict=True)),
        "median_p95_no_slower": no_slower,
        "accepted": all(all(row.values()) for row in checks)
        and no_slower
        and all(b <= a for a, b in zip(old, new, strict=True)),
        "scope": "Fixed exposed log132 only; caller must verify identical hardware, provider, threads, input hashes and interpreter. Repetition median limits ordering noise; every pair remains reported.",
    }
