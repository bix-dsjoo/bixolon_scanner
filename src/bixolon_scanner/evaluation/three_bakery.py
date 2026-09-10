"""Score public Worker responses with image-level, class-blind matching."""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..contracts import ItemStatus, ScanResponse, Status
from ..runtime.geometry import box_iou_matrix


def match_boxes(
    predictions: list[list[float]], targets: list[list[float]], threshold: float
) -> dict[int, int]:
    """Maximize match cardinality, then summed IoU; class labels never enter."""
    if not 0 < threshold <= 1:
        raise ValueError("matching IoU must be in (0, 1]")
    if not predictions or not targets:
        return {}
    overlaps = box_iou_matrix(
        np.asarray(predictions, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    )
    n, m = overlaps.shape
    # One cardinality reward exceeds every possible difference in total IoU.
    costs = np.zeros((n, m + n), dtype=np.float64)
    costs[:, :m] = np.where(overlaps >= threshold, -(min(n, m) + 1 + overlaps), 1.0)
    rows, columns = linear_sum_assignment(costs)
    return {
        int(i): int(j)
        for i, j in zip(rows, columns, strict=True)
        if j < m and overlaps[i, j] >= threshold
    }


def score_response(
    response: ScanResponse, annotations: list[dict], *, threshold: float = 0.5
) -> dict:
    boxes = [
        [s.bbox.x, s.bbox.y, s.bbox.x + s.bbox.width, s.bbox.y + s.bbox.height]
        for s in response.segmentations
    ]
    gt_boxes = []
    for annotation in annotations:
        x, y, w, h = annotation.get("bbox_xywh", annotation.get("bbox"))
        gt_boxes.append([x, y, x + w, y + h])
    matches = match_boxes(boxes, gt_boxes, threshold)
    overlaps = (
        box_iou_matrix(np.asarray(boxes), np.asarray(gt_boxes))
        if boxes and gt_boxes
        else np.zeros((len(boxes), len(gt_boxes)))
    )
    duplicates = {
        index
        for index in range(len(boxes))
        if index not in matches and np.any(overlaps[index] >= threshold)
    }
    correct = wrong_class = unmatched_approved = top3_miss = 0
    duplicate_approved = 0
    details = []
    counts = Counter(s.status.value for s in response.segmentations)
    for i, segmentation in enumerate(response.segmentations):
        target_index = matches.get(i)
        target = (
            None
            if target_index is None
            else f"bread_{int(annotations[target_index]['category_id']):02d}"
        )
        predicted = None if segmentation.prediction is None else segmentation.prediction.class_id
        if segmentation.status is ItemStatus.APPROVED:
            if target is None:
                unmatched_approved += 1
                duplicate_approved += i in duplicates
            elif predicted != target:
                wrong_class += 1
            else:
                correct += 1
        elif segmentation.status is ItemStatus.UNKNOWN and target is not None:
            top3_miss += target not in {c.class_id for c in segmentation.top3}
        details.append(
            {
                "segmentation_id": segmentation.segmentation_id,
                "target_index": target_index,
                "target_class_id": target,
                "predicted_class_id": predicted,
                "status": segmentation.status.value,
            }
        )
    complete = (
        response.status is Status.SEGMENTATION
        and bool(annotations)
        and correct == len(annotations) == len(response.segmentations)
    )
    return {
        "complete_image": complete,
        "status": response.status.value,
        "ground_truth_count": len(annotations),
        "prediction_count": len(boxes),
        "matched_count": len(matches),
        "correct_approved_count": correct,
        "wrong_class_approved_count": wrong_class,
        "unmatched_approved_count": unmatched_approved,
        "duplicate_approved_count": duplicate_approved,
        "background_approved_count": unmatched_approved - duplicate_approved,
        "wrong_approved_count": wrong_class + unmatched_approved,
        "missed_count": len(annotations) - len(matches),
        "extra_count": len(boxes) - len(matches),
        "duplicate_count": len(duplicates),
        "background_prediction_count": len(boxes) - len(matches) - len(duplicates),
        "unknown_top3_miss_count": int(top3_miss),
        "status_counts": dict(counts),
        "items": details,
    }


def summarize(
    rows: list[dict],
    *,
    minimum_complete_images: int | None = None,
    maximum_p95_ms: float | None = None,
) -> dict:
    if not rows:
        raise ValueError("evaluation requires at least one image")
    if any(not np.isfinite(r["elapsed_ms"]) or r["elapsed_ms"] < 0 for r in rows):
        raise ValueError("latencies must be finite and nonnegative")
    keys = (
        "ground_truth_count",
        "prediction_count",
        "matched_count",
        "correct_approved_count",
        "wrong_class_approved_count",
        "unmatched_approved_count",
        "duplicate_approved_count",
        "background_approved_count",
        "wrong_approved_count",
        "missed_count",
        "extra_count",
        "duplicate_count",
        "background_prediction_count",
        "unknown_top3_miss_count",
    )
    counts = {key: sum(r["metrics"][key] for r in rows) for key in keys}
    complete = sum(r["metrics"]["complete_image"] for r in rows)
    image_statuses = Counter(r["metrics"]["status"] for r in rows)
    item_statuses = Counter()
    for row in rows:
        item_statuses.update(row["metrics"]["status_counts"])
    groups = {}
    for name, predicate in (
        ("full_path", lambda r: r["metrics"]["status"] == "SEGMENTATION"),
        ("image_recapture", lambda r: r["metrics"]["status"] == "IMAGE_RECAPTURE"),
        ("error", lambda r: r["metrics"]["status"] == "ERROR"),
        ("all", lambda r: True),
    ):
        values = [r["elapsed_ms"] for r in rows if predicate(r)]
        groups[name] = {
            "count": len(values),
            "max_ms": max(values) if values else None,
            **{
                f"p{q}_ms": float(np.percentile(values, q)) if values else None
                for q in (50, 95, 99)
            },
        }
    by_difficulty = {}
    for difficulty in sorted({r.get("difficulty", "unspecified") for r in rows}):
        subset = [r for r in rows if r.get("difficulty", "unspecified") == difficulty]
        by_difficulty[difficulty] = {
            "image_count": len(subset),
            "complete_images": sum(r["metrics"]["complete_image"] for r in subset),
            "wrong_approved_count": sum(r["metrics"]["wrong_approved_count"] for r in subset),
        }
    accuracy_met = (
        None
        if minimum_complete_images is None
        else complete >= minimum_complete_images and counts["wrong_approved_count"] == 0
    )
    speed_met = (
        None
        if maximum_p95_ms is None
        else all(
            groups[name]["p95_ms"] is not None and groups[name]["p95_ms"] <= maximum_p95_ms
            for name in ("all", "full_path")
        )
    )
    return {
        **counts,
        "image_count": len(rows),
        "complete_images": complete,
        "complete_image_rate": complete / len(rows),
        "correct_approved_rate": counts["correct_approved_count"] / counts["ground_truth_count"]
        if counts["ground_truth_count"]
        else None,
        "image_status_counts": dict(image_statuses),
        "item_status_counts": dict(item_statuses),
        "latency": groups,
        "by_difficulty": by_difficulty,
        "accuracy_target_met": accuracy_met,
        "speed_target_met": speed_met,
        "maximum_p95_ms": maximum_p95_ms,
        "target_met": None if accuracy_met is None else accuracy_met and speed_met is not False,
    }


def candidate_rank(real: dict, stress: dict) -> tuple:
    """The prespecified source diagnostic ordering; no final test input."""
    errors = real["image_status_counts"].get("ERROR", 0) + stress["image_status_counts"].get(
        "ERROR", 0
    )
    if errors:
        raise ValueError("a candidate with execution errors cannot be ranked")
    return (
        real["wrong_approved_count"] + stress["wrong_approved_count"],
        -real["by_difficulty"]["multi"]["complete_images"],
        -stress["complete_images"],
        real["missed_count"] + stress["missed_count"],
        real["unknown_top3_miss_count"] + stress["unknown_top3_miss_count"],
        real["latency"]["full_path"]["p95_ms"] or 1e308,
    )


def robust_rank(ranks: list[tuple], candidate_id: str) -> tuple:
    if len(ranks) != 3:
        raise ValueError("robust selection requires all three prespecified seeds")
    ordered = sorted(ranks)
    # Quality across seeds takes precedence over latency; latency is the next tie-break.
    return ordered[-1][:-1], ordered[1][:-1], sorted(rank[-1] for rank in ranks)[1], candidate_id
