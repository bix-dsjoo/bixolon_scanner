from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def _intersection_over_area(
    box: Sequence[float],
    other: Sequence[float],
) -> float:
    x, y, width, height = (float(value) for value in box)
    other_x, other_y, other_width, other_height = (float(value) for value in other)
    intersection_width = max(
        0.0,
        min(x + width, other_x + other_width) - max(x, other_x),
    )
    intersection_height = max(
        0.0,
        min(y + height, other_y + other_height) - max(y, other_y),
    )
    return intersection_width * intersection_height / max(width * height, 1e-6)


def extract_scene_geometry_features(
    record: Mapping[str, object],
    annotation_index: int,
) -> np.ndarray:
    """Describe ROI completeness using class-agnostic scene geometry.

    All dimensions are normalized so the estimator can be calibrated on BIX-only
    composites and applied to camera frames with a different resolution.
    """
    annotations = list(record.get("annotations", []))
    if not 0 <= annotation_index < len(annotations):
        raise IndexError("annotation index is outside the scene")
    annotation = annotations[annotation_index]
    if not isinstance(annotation, Mapping):
        raise TypeError("annotation must be a mapping")
    box = annotation.get("bbox_xywh")
    if not isinstance(box, Sequence) or len(box) != 4:
        raise ValueError("annotation bbox_xywh must contain four values")
    image_width = max(float(record["width"]), 1.0)
    image_height = max(float(record["height"]), 1.0)
    x, y, width, height = (float(value) for value in box)
    width = max(width, 1e-6)
    height = max(height, 1e-6)
    area = width * height
    scene_areas = []
    covers = []
    for other_index, other in enumerate(annotations):
        if not isinstance(other, Mapping):
            continue
        other_box = other.get("bbox_xywh")
        if not isinstance(other_box, Sequence) or len(other_box) != 4:
            continue
        other_width = max(float(other_box[2]), 0.0)
        other_height = max(float(other_box[3]), 0.0)
        scene_areas.append(other_width * other_height)
        if other_index != annotation_index:
            covers.append(_intersection_over_area(box, other_box))
    median_area = float(np.median(scene_areas)) if scene_areas else area
    maximum_cover = max(covers, default=0.0)
    total_cover = min(sum(covers), 1.0)
    horizontal_border = min(x, image_width - (x + width)) / image_width
    vertical_border = min(y, image_height - (y + height)) / image_height
    border_touch_count = sum(
        (
            x <= 1.0,
            y <= 1.0,
            x + width >= image_width - 1.0,
            y + height >= image_height - 1.0,
        )
    )
    return np.asarray(
        [
            width / image_width,
            height / image_height,
            area / (image_width * image_height),
            math.log(width / height),
            math.log(max(area / max(median_area, 1e-6), 1e-6)),
            maximum_cover,
            total_cover,
            float(np.mean(covers)) if covers else 0.0,
            max(horizontal_border, -1.0),
            max(vertical_border, -1.0),
            float(border_touch_count) / 4.0,
            min(len(annotations), 20) / 20.0,
            (x + width * 0.5) / image_width,
            (y + height * 0.5) / image_height,
        ],
        dtype=np.float32,
    )


__all__ = ["extract_scene_geometry_features"]
