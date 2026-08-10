from __future__ import annotations

import logging
import time
from typing import Protocol

import numpy as np
from PIL import Image

from .contracts import (
    BoundingBox,
    Candidate,
    ItemStatus,
    ModelVersions,
    Prediction,
    ScanItem,
    ScanResponse,
    Status,
)
from .inference import Detection, DetectionResult
from .imaging import image_original_size
from .package import ClassifierMetadata, CountVerifierMetadata, QualityMetadata


LOGGER = logging.getLogger(__name__)


class Detector(Protocol):
    version: str

    def detect(self, image: np.ndarray | Image.Image) -> DetectionResult: ...


class Classifier(Protocol):
    version: str

    def classify(
        self, image: np.ndarray | Image.Image, detections: list[Detection]
    ) -> np.ndarray: ...


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(scaled)
    return (exponential / exponential.sum(axis=1, keepdims=True)).astype(np.float32)


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
    image: np.ndarray | Image.Image, result: DetectionResult, metadata: QualityMetadata
) -> list[str]:
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
        if metadata.border_policy == "always_recapture" and _touches_border(
            detection, width, height, metadata.border_margin_ratio
        ):
            reasons.append("DETECTOR_BORDER_CLIPPED")
            break
    if metadata.min_mean_luminance is not None or metadata.max_mean_luminance is not None:
        mean_luminance = float(_as_array(image).astype(np.float32).mean())
        if (
            metadata.min_mean_luminance is not None
            and mean_luminance < metadata.min_mean_luminance
        ):
            reasons.append("DETECTOR_UNDEREXPOSED")
        if (
            metadata.max_mean_luminance is not None
            and mean_luminance > metadata.max_mean_luminance
        ):
            reasons.append("DETECTOR_OVEREXPOSED")
    if metadata.min_sharpness is not None and _sharpness(image) < metadata.min_sharpness:
        reasons.append("DETECTOR_BLUR")
    return list(dict.fromkeys(reasons))


def _touches_border(
    detection: Detection, width: int, height: int, margin_ratio: float
) -> bool:
    border_x = width * margin_ratio
    border_y = height * margin_ratio
    return (
        detection.x1 <= border_x
        or detection.y1 <= border_y
        or detection.x2 >= width - border_x
        or detection.y2 >= height - border_y
    )


def _border_detection_indices(
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
        if _touches_border(detection, width, height, metadata.border_margin_ratio)
    }


class DecisionPipeline:
    def __init__(
        self,
        detector: Detector,
        classifier: Classifier,
        classifier_metadata: ClassifierMetadata,
        quality_metadata: QualityMetadata,
        count_verifier_metadata: CountVerifierMetadata | None = None,
    ):
        self.detector = detector
        self.classifier = classifier
        self.classifier_metadata = classifier_metadata
        self.quality_metadata = quality_metadata
        self.count_verifier_metadata = count_verifier_metadata

    @property
    def versions(self) -> ModelVersions:
        return ModelVersions(detector=self.detector.version, classifier=self.classifier.version)

    def scan(self, image: np.ndarray | Image.Image, request_id: str) -> ScanResponse:
        started = time.perf_counter()
        detector_started = time.perf_counter()
        detection_result = self.detector.detect(image)
        detector_ms = (time.perf_counter() - detector_started) * 1000.0
        reasons = quality_reasons(image, detection_result, self.quality_metadata)
        if detection_result.uncertain_candidate_count:
            reasons.append("DETECTOR_UNCERTAIN_OBJECT")
        if self.count_verifier_metadata is not None:
            if (
                detection_result.verified_count is None
                or detection_result.count_confidence is None
            ):
                raise ValueError("count verifier result is missing")
            if (
                detection_result.count_confidence
                < self.count_verifier_metadata.confidence_threshold
            ):
                reasons.append("DETECTOR_COUNT_UNCERTAIN")
            elif detection_result.verified_count != len(detection_result.detections):
                reasons.append("DETECTOR_COUNT_MISMATCH")
            reasons = list(dict.fromkeys(reasons))
        if reasons:
            response = ScanResponse(
                request_id=request_id,
                status=Status.RECAPTURE,
                reason_codes=reasons,
                items=[],
                processing_time_ms=(time.perf_counter() - started) * 1000.0,
                model_versions=ModelVersions(detector=self.detector.version, classifier=None),
            )
            self._log(response, detector_ms=detector_ms, classifier_ms=0.0)
            return response

        ordered = sorted(detection_result.detections, key=lambda detection: (detection.y1, detection.x1))
        classifier_started = time.perf_counter()
        logits = self.classifier.classify(image, ordered)
        classifier_ms = (time.perf_counter() - classifier_started) * 1000.0
        if logits.shape != (len(ordered), len(self.classifier_metadata.labels)):
            raise ValueError("classifier output shape does not match package labels")
        probabilities = _softmax(logits, self.classifier_metadata.temperature)
        top_indices = np.argsort(-probabilities, axis=1)

        recapture_labels = {
            index for index, label in enumerate(self.classifier_metadata.labels) if label.recapture
        }
        if any(int(indices[0]) in recapture_labels for indices in top_indices):
            response = ScanResponse(
                request_id=request_id,
                status=Status.RECAPTURE,
                reason_codes=["CLASSIFIER_QUALITY_CLASS"],
                items=[],
                processing_time_ms=(time.perf_counter() - started) * 1000.0,
                model_versions=self.versions,
            )
            self._log(response, detector_ms=detector_ms, classifier_ms=classifier_ms)
            return response

        if self.quality_metadata.border_policy == "classifier_confidence":
            border_indices = _border_detection_indices(
                image, ordered, self.quality_metadata
            )
            low_confidence_border = any(
                float(probabilities[index, int(top_indices[index][0])])
                < self.classifier_metadata.approval_threshold
                for index in border_indices
            )
            if low_confidence_border:
                response = ScanResponse(
                    request_id=request_id,
                    status=Status.RECAPTURE,
                    reason_codes=["DETECTOR_BORDER_CLIPPED"],
                    items=[],
                    processing_time_ms=(time.perf_counter() - started) * 1000.0,
                    model_versions=self.versions,
                )
                self._log(response, detector_ms=detector_ms, classifier_ms=classifier_ms)
                return response

        items: list[ScanItem] = []
        for ordinal, (detection, indices, scores) in enumerate(zip(ordered, top_indices, probabilities), start=1):
            top1_index = int(indices[0])
            top1_score = float(scores[top1_index])
            label = self.classifier_metadata.labels[top1_index]
            bbox = BoundingBox(
                x=max(0, int(round(detection.x1))),
                y=max(0, int(round(detection.y1))),
                width=max(1, int(round(detection.x2 - detection.x1))),
                height=max(1, int(round(detection.y2 - detection.y1))),
            )
            if top1_score >= self.classifier_metadata.approval_threshold:
                item = ScanItem(
                    item_id=f"item_{ordinal:03d}",
                    bbox=bbox,
                    status=ItemStatus.APPROVED,
                    reason_codes=[],
                    prediction=Prediction(class_id=label.class_id, class_name=label.class_name),
                    top3=[],
                    confidence=top1_score,
                )
            else:
                candidates = [
                    Candidate(
                        class_id=self.classifier_metadata.labels[int(index)].class_id,
                        class_name=self.classifier_metadata.labels[int(index)].class_name,
                        confidence=float(scores[int(index)]),
                    )
                    for index in indices[:3]
                ]
                item = ScanItem(
                    item_id=f"item_{ordinal:03d}",
                    bbox=bbox,
                    status=ItemStatus.UNKNOWN,
                    reason_codes=["BELOW_APPROVAL_THRESHOLD"],
                    prediction=None,
                    top3=candidates,
                    confidence=top1_score,
                )
            items.append(item)

        has_unknown = any(item.status is ItemStatus.UNKNOWN for item in items)
        response = ScanResponse(
            request_id=request_id,
            status=Status.UNKNOWN if has_unknown else Status.APPROVED,
            reason_codes=["ITEM_BELOW_APPROVAL_THRESHOLD"] if has_unknown else [],
            items=items,
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
            model_versions=self.versions,
        )
        self._log(response, detector_ms=detector_ms, classifier_ms=classifier_ms)
        return response

    @staticmethod
    def _log(response: ScanResponse, *, detector_ms: float, classifier_ms: float) -> None:
        LOGGER.info(
            "scan_complete",
            extra={
                "request_id": response.request_id,
                "status": response.status.value,
                "reason_codes": response.reason_codes,
                "item_count": len(response.items),
                "detector_ms": round(detector_ms, 3),
                "classifier_ms": round(classifier_ms, 3),
                "processing_time_ms": round(response.processing_time_ms, 3),
                "detector_version": response.model_versions.detector,
                "classifier_version": response.model_versions.classifier,
            },
        )
