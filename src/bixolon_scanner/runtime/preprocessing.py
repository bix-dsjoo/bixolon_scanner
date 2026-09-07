from __future__ import annotations

import numpy as np
from PIL import Image

from ..pipeline.ports import Detection


def prepare_rgb(
    image: np.ndarray | Image.Image,
    size: tuple[int, int],
    mean: tuple[float, ...],
    std: tuple[float, ...],
    *,
    reducing_gap: float | None = None,
) -> np.ndarray:
    source = image if isinstance(image, Image.Image) else Image.fromarray(image, mode="RGB")
    pil = source.resize(
        (size[1], size[0]),
        Image.Resampling.BILINEAR,
        reducing_gap=reducing_gap,
    )
    tensor = np.ascontiguousarray(
        np.transpose(np.asarray(pil, dtype=np.uint8), (2, 0, 1)),
        dtype=np.float32,
    )
    tensor /= np.float32(255.0)
    if any(value != 0.0 for value in mean):
        tensor -= np.asarray(mean, dtype=np.float32)[:, None, None]
    if any(value != 1.0 for value in std):
        tensor /= np.asarray(std, dtype=np.float32)[:, None, None]
    return tensor


def classifier_neighbor_ownership_mask(
    detections: list[Detection],
    target_index: int,
    *,
    image_width: int,
    image_height: int,
    output_size: int,
    margin_ratio: float,
    distance_bias: float,
    shared_scale: bool,
    crop_mode: str = "box_resize",
) -> np.ndarray:
    if not 0 <= target_index < len(detections):
        raise ValueError("target detection index is outside the detection list")
    if output_size < 1 or margin_ratio < 0.0 or distance_bias < -1.0:
        raise ValueError("mask size, margin, and distance bias are invalid")
    target = detections[target_index]
    target_width = target.x2 - target.x1
    target_height = target.y2 - target.y1
    if target_width <= 0.0 or target_height <= 0.0:
        raise ValueError("target detection box is empty")
    # The image crop uses integer pixel boundaries. Use the same grid for masks;
    # a separate fractional grid can amplify tiny provider differences into a
    # different masked pixel and a materially different classifier score.
    crop_x1, crop_y1, crop_x2, crop_y2 = classifier_crop_box(
        target,
        image_width,
        image_height,
        margin_ratio=margin_ratio,
        crop_mode=crop_mode,
    )
    x = crop_x1 + (np.arange(output_size) + 0.5) * (crop_x2 - crop_x1) / output_size
    y = crop_y1 + (np.arange(output_size) + 0.5) * (crop_y2 - crop_y1) / output_size
    target_center_x = (target.x1 + target.x2) / 2.0
    target_center_y = (target.y1 + target.y2) / 2.0
    target_distance = (((x - target_center_x) / max(target_width / 2.0, 1e-12)) ** 2)[None, :] + (
        ((y - target_center_y) / max(target_height / 2.0, 1e-12)) ** 2
    )[:, None]
    mask = np.zeros((output_size, output_size), dtype=bool)
    for index, other in enumerate(detections):
        if index == target_index:
            continue
        other_width = other.x2 - other.x1
        other_height = other.y2 - other.y1
        if other_width <= 0.0 or other_height <= 0.0:
            continue
        inside = ((x >= other.x1) & (x <= other.x2))[None, :] & ((y >= other.y1) & (y <= other.y2))[
            :, None
        ]
        width_scale = target_width if shared_scale else other_width
        height_scale = target_height if shared_scale else other_height
        other_distance = (((x - (other.x1 + other.x2) / 2.0) / max(width_scale / 2.0, 1e-12)) ** 2)[
            None, :
        ] + (((y - (other.y1 + other.y2) / 2.0) / max(height_scale / 2.0, 1e-12)) ** 2)[:, None]
        mask |= inside & (other_distance + distance_bias < target_distance)
    return mask


def apply_classifier_background_masks(batch: np.ndarray, masks: np.ndarray) -> np.ndarray:
    if batch.ndim != 4 or masks.shape != (len(batch), batch.shape[2], batch.shape[3]):
        raise ValueError("classifier batch and neighbor masks are not aligned")
    output = batch.copy()
    borders = np.concatenate(
        (
            batch[:, :, 0, :],
            batch[:, :, -1, :],
            batch[:, :, 1:-1, 0],
            batch[:, :, 1:-1, -1],
        ),
        axis=2,
    )
    background = np.median(borders, axis=2)
    for channel in range(batch.shape[1]):
        output[:, channel] = np.where(
            masks,
            background[:, channel, None, None],
            batch[:, channel],
        )
    return output


def classifier_crop_box(
    detection: Detection,
    image_width: int,
    image_height: int,
    *,
    margin_ratio: float,
    crop_mode: str,
) -> tuple[int, int, int, int]:
    margin_x = (detection.x2 - detection.x1) * margin_ratio
    margin_y = (detection.y2 - detection.y1) * margin_ratio
    x1 = max(0.0, detection.x1 - margin_x)
    y1 = max(0.0, detection.y1 - margin_y)
    x2 = min(float(image_width), detection.x2 + margin_x)
    y2 = min(float(image_height), detection.y2 + margin_y)
    if crop_mode == "square_context":
        side = min(max(x2 - x1, y2 - y1), float(image_width), float(image_height))
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        x1 = min(max(0.0, center_x - side * 0.5), image_width - side)
        y1 = min(max(0.0, center_y - side * 0.5), image_height - side)
        x2 = x1 + side
        y2 = y1 + side
    elif crop_mode != "box_resize":
        raise ValueError(f"unsupported classifier crop mode: {crop_mode}")
    return (
        max(0, int(np.floor(x1))),
        max(0, int(np.floor(y1))),
        min(image_width, int(np.ceil(x2))),
        min(image_height, int(np.ceil(y2))),
    )


__all__ = [
    "apply_classifier_background_masks",
    "classifier_crop_box",
    "classifier_neighbor_ownership_mask",
    "prepare_rgb",
]
