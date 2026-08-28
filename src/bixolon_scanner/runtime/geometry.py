from __future__ import annotations

from ..pipeline.ports import Detection


def box_iou(left: Detection, right: Detection) -> float:
    ix1 = max(left.x1, right.x1)
    iy1 = max(left.y1, right.y1)
    ix2 = min(left.x2, right.x2)
    iy2 = min(left.y2, right.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def box_containment(outer: Detection, inner: Detection) -> float:
    ix1 = max(outer.x1, inner.x1)
    iy1 = max(outer.y1, inner.y1)
    ix2 = min(outer.x2, inner.x2)
    iy2 = min(outer.y2, inner.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    inner_area = max(0.0, inner.x2 - inner.x1) * max(0.0, inner.y2 - inner.y1)
    return intersection / inner_area if inner_area > 0.0 else 0.0


def nms(
    detections: list[Detection],
    threshold: float,
    containment_threshold: float | None = None,
    class_aware_containment: bool = False,
) -> list[Detection]:
    """Apply stable score-ordered NMS with optional containment suppression."""

    ordered = sorted(detections, key=lambda detection: detection.score, reverse=True)
    kept: list[Detection] = []
    while ordered:
        current = ordered.pop(0)
        kept.append(current)
        remaining: list[Detection] = []
        current_area = max(0.0, current.x2 - current.x1) * max(0.0, current.y2 - current.y1)
        for candidate in ordered:
            intersection = max(
                0.0, min(current.x2, candidate.x2) - max(current.x1, candidate.x1)
            ) * max(
                0.0,
                min(current.y2, candidate.y2) - max(current.y1, candidate.y1),
            )
            candidate_area = max(0.0, candidate.x2 - candidate.x1) * max(
                0.0,
                candidate.y2 - candidate.y1,
            )
            union = current_area + candidate_area - intersection
            smaller_area = min(current_area, candidate_area)
            contained = (
                containment_threshold is not None
                and smaller_area > 0.0
                and intersection / smaller_area >= containment_threshold
                and (
                    not class_aware_containment
                    or current.class_id is not None
                    and current.class_id == candidate.class_id
                )
            )
            if (union <= 0.0 or intersection / union <= threshold) and not contained:
                remaining.append(candidate)
        ordered = remaining
    return kept


__all__ = ["box_containment", "box_iou", "nms"]
