from __future__ import annotations

import numpy as np
from PIL import Image

from ..contracts.image import image_original_size
from ..contracts.model_package import QualityMetadata
from .ports import Detection, DetectionResult


def _as_array(image: np.ndarray | Image.Image) -> np.ndarray:
    return image if isinstance(image, np.ndarray) else np.asarray(image, dtype=np.uint8)


def _sharpness(image: np.ndarray | Image.Image) -> float:
    gray = _as_array(image).astype(np.float32).mean(axis=2)
    if min(gray.shape) < 3:
        return 0.0
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(laplacian.var())


def quality_reasons(
    image: np.ndarray | Image.Image,
    result: DetectionResult,
    metadata: QualityMetadata,
) -> list[str]:
    """Return detector hard-gate reasons in the canonical evaluation order."""

    if result.capacity_saturated:
        return ["DETECTOR_CAPACITY_EXCEEDED"]
    if not result.detections:
        return ["DETECTOR_NO_OBJECT"]
    if isinstance(image, Image.Image):
        width, height = image_original_size(image)
    else:
        height, width = image.shape[:2]
    image_area = float(height * width)
    reasons: list[str] = []
    for detection in result.detections:
        area_ratio = (detection.x2 - detection.x1) * (detection.y2 - detection.y1) / image_area
        if area_ratio < metadata.min_object_area_ratio:
            reasons.append("DETECTOR_OBJECT_TOO_SMALL")
            break
        if metadata.border_policy == "always_recapture" and touches_border(
            detection,
            width,
            height,
            metadata.border_margin_ratio,
        ):
            reasons.append("DETECTOR_BORDER_CLIPPED")
            break
    if metadata.min_mean_luminance is not None or metadata.max_mean_luminance is not None:
        mean_luminance = float(_as_array(image).astype(np.float32).mean())
        if metadata.min_mean_luminance is not None and mean_luminance < metadata.min_mean_luminance:
            reasons.append("DETECTOR_UNDEREXPOSED")
        if metadata.max_mean_luminance is not None and mean_luminance > metadata.max_mean_luminance:
            reasons.append("DETECTOR_OVEREXPOSED")
    if metadata.min_sharpness is not None and _sharpness(image) < metadata.min_sharpness:
        reasons.append("DETECTOR_BLUR")
    return list(dict.fromkeys(reasons))


def touches_border(
    detection: Detection,
    width: int,
    height: int,
    margin_ratio: float,
) -> bool:
    border_x = width * margin_ratio
    border_y = height * margin_ratio
    return (
        detection.x1 <= border_x
        or detection.y1 <= border_y
        or detection.x2 >= width - border_x
        or detection.y2 >= height - border_y
    )


def border_detection_indices(
    image: np.ndarray | Image.Image,
    detections: list[Detection],
    metadata: QualityMetadata,
) -> set[int]:
    if isinstance(image, Image.Image):
        width, height = image_original_size(image)
    else:
        height, width = image.shape[:2]
    return {
        index
        for index, detection in enumerate(detections)
        if touches_border(detection, width, height, metadata.border_margin_ratio)
    }


def contained_detection_pairs(
    detections: list[Detection],
    threshold: float | None,
) -> list[tuple[int, int]]:
    """Return (lower-score, higher-score) pairs with near-complete containment."""

    if threshold is None:
        return []
    pairs: list[tuple[int, int]] = []
    for left_index, left in enumerate(detections):
        left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
        for right_index in range(left_index + 1, len(detections)):
            right = detections[right_index]
            right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
            smaller_area = min(left_area, right_area)
            if smaller_area <= 0.0:
                continue
            intersection = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1)) * max(
                0.0,
                min(left.y2, right.y2) - max(left.y1, right.y1),
            )
            if intersection / smaller_area < threshold:
                continue
            if left.score < right.score:
                pairs.append((left_index, right_index))
            elif right.score < left.score:
                pairs.append((right_index, left_index))
    return pairs


__all__ = [
    "border_detection_indices",
    "contained_detection_pairs",
    "quality_reasons",
    "touches_border",
]
