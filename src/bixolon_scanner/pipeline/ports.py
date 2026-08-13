from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


@dataclass(frozen=True)
class DetectionResult:
    detections: list[Detection]
    capacity_saturated: bool = False
    verified_count: int | None = None
    count_confidence: float | None = None
    uncertain_candidate_count: int = 0
    uncertain_candidate_scores: tuple[float, ...] = ()


class Detector(Protocol):
    version: str

    def detect(self, image: np.ndarray | Image.Image) -> DetectionResult: ...


class Classifier(Protocol):
    version: str

    def classify(
        self, image: np.ndarray | Image.Image, detections: list[Detection]
    ) -> np.ndarray: ...
