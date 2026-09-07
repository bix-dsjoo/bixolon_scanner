from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image


def extract_shape_quality_features(image: Image.Image) -> np.ndarray:
    """Return color-invariant contour features for a tightly cropped ROI."""
    rgb = np.asarray(image.convert("RGB").resize((192, 192), Image.Resampling.BICUBIC))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    border = np.concatenate((lab[0], lab[-1], lab[1:-1, 0], lab[1:-1, -1]), axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab.astype(np.float32) - background, axis=2)
    threshold = max(10.0, float(np.quantile(distance, 0.35)))
    mask = (distance > threshold).astype(np.uint8)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if component_count > 1:
        selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == selected).astype(np.uint8)
    if int(mask.sum()) < 64:
        mask[:] = 1

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    height, width = mask.shape
    crop_aspect = math.log(width / max(height, 1))
    edge_density = float(cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 80, 160).mean() / 255.0)
    touch = (
        float(mask[0].mean()),
        float(mask[-1].mean()),
        float(mask[:, 0].mean()),
        float(mask[:, -1].mean()),
    )
    hole_count = 0
    if hierarchy is not None:
        hole_count = int(np.count_nonzero(hierarchy[0, :, 3] >= 0))
    if contour is None:
        base = [1.0, crop_aspect, 1.0, 1.0, 1.0, 0.0, 0.5, 0.5, *touch]
        hu = [0.0] * 7
    else:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        contour_area = max(float(cv2.contourArea(contour)), 1.0)
        perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
        hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
        moments = cv2.moments(contour)
        centroid_x = moments["m10"] / max(moments["m00"], 1e-6) / width
        centroid_y = moments["m01"] / max(moments["m00"], 1e-6) / height
        base = [
            float(mask.mean()),
            crop_aspect,
            math.log(box_width / max(box_height, 1)),
            float(mask.sum()) / max(box_width * box_height, 1),
            contour_area / hull_area,
            4.0 * math.pi * contour_area / (perimeter * perimeter),
            centroid_x,
            centroid_y,
            *touch,
        ]
        raw_hu = cv2.HuMoments(moments).ravel()
        hu = (-np.sign(raw_hu) * np.log10(np.abs(raw_hu).clip(min=1e-12))).tolist()
    return np.asarray([*base, float(hole_count), edge_density, *hu], dtype=np.float32)


__all__ = ["extract_shape_quality_features"]
