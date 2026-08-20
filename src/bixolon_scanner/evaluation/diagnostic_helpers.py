"""Shared recording and matching helpers for optional diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..pipeline.ports import Detection, DetectionResult
from .detector import _iou, _xywh_to_xyxy


@dataclass
class RecordingDetector:
    detector: Any
    last_result: DetectionResult | None = None
    last_ms: float = 0.0

    @property
    def version(self) -> str:
        return self.detector.version

    def detect(self, image: Any) -> DetectionResult:
        started = time.perf_counter()
        self.last_result = self.detector.detect(image)
        self.last_ms = (time.perf_counter() - started) * 1000.0
        return self.last_result


@dataclass
class RecordingClassifier:
    classifier: Any
    last_ms: float = 0.0

    @property
    def version(self) -> str:
        return self.classifier.version

    def classify(self, image: Any, detections: list[Detection]) -> Any:
        started = time.perf_counter()
        result = self.classifier.classify(image, detections)
        self.last_ms = (time.perf_counter() - started) * 1000.0
        return result


def match_detections(
    detections: list[Detection], annotations: list[dict[str, Any]], threshold: float
) -> tuple[dict[int, int], set[int]]:
    gt_boxes = [_xywh_to_xyxy(row["bbox"]) for row in annotations]
    remaining = set(range(len(gt_boxes)))
    matches: dict[int, int] = {}
    for detection_index, detection in sorted(
        enumerate(detections), key=lambda item: item[1].score, reverse=True
    ):
        box = np.asarray([detection.x1, detection.y1, detection.x2, detection.y2], dtype=np.float32)
        candidates = [(index, _iou(box, gt_boxes[index])) for index in remaining]
        if candidates:
            gt_index, overlap = max(candidates, key=lambda item: item[1])
            if overlap >= threshold:
                matches[detection_index] = gt_index
                remaining.remove(gt_index)
    return matches, remaining
