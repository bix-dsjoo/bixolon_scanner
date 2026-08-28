from __future__ import annotations

import logging
import time

import numpy as np
from PIL import Image

from ..contracts import (
    ModelVersions,
    ScanResponse,
    Status,
)
from ..contracts.model_package import ClassifierMetadata, CountVerifierMetadata, QualityMetadata
from .classification import normalize_classification, softmax
from .ports import Classifier, Detector
from .quality import border_detection_indices, contained_detection_pairs, quality_reasons
from .segmentation import build_scan_items, summarize_reason_codes

LOGGER = logging.getLogger(__name__)


_softmax = softmax


class DecisionPipeline:
    def __init__(
        self,
        detector: Detector,
        classifier: Classifier,
        classifier_metadata: ClassifierMetadata,
        quality_metadata: QualityMetadata,
        count_verifier_metadata: CountVerifierMetadata | None = None,
        *,
        worker_version: str = "1.0.0",
        embedder_version: str | None = None,
        detector_policy_version: str | None = None,
        classifier_policy_version: str | None = None,
        catalog_version: str | None = None,
    ):
        self.detector = detector
        self.classifier = classifier
        self.classifier_metadata = classifier_metadata
        self.quality_metadata = quality_metadata
        self.count_verifier_metadata = count_verifier_metadata
        self.worker_version = worker_version
        self.embedder_version = embedder_version
        self.detector_policy_version = detector_policy_version
        self.classifier_policy_version = classifier_policy_version
        self.catalog_version = catalog_version

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
            if detection_result.verified_count is None or detection_result.count_confidence is None:
                raise ValueError("count verifier result is missing")
            if (
                detection_result.count_confidence
                < self.count_verifier_metadata.confidence_threshold
            ):
                reasons.append("DETECTOR_COUNT_UNCERTAIN")
            elif detection_result.verified_count != (
                int(bool(detection_result.detections))
                if self.count_verifier_metadata.comparison_mode == "object_presence"
                else len(detection_result.detections)
            ):
                reasons.append("DETECTOR_COUNT_MISMATCH")
            reasons = list(dict.fromkeys(reasons))
        if reasons:
            response = ScanResponse(
                request_id=request_id,
                status=Status.IMAGE_RECAPTURE,
                reason_codes=["IMAGE_RECAPTURE_REQUIRED"],
                segmentations=[],
                processing_time_ms=(time.perf_counter() - started) * 1000.0,
                worker_version=self.worker_version,
                detector_version=self.detector.version,
                classifier_version=None,
                embedder_version=None,
                detector_policy_version=self.detector_policy_version,
                classifier_policy_version=None,
                catalog_version=None,
            )
            self._log(
                response,
                detector_ms=detector_ms,
                classifier_ms=0.0,
                diagnostic_reason_codes=reasons,
            )
            return response

        ordered = sorted(
            detection_result.detections, key=lambda detection: (detection.y1, detection.x1)
        )
        classifier_started = time.perf_counter()
        classification = self.classifier.classify(image, ordered)
        classifier_ms = (time.perf_counter() - classifier_started) * 1000.0
        batch = normalize_classification(
            classification,
            detection_count=len(ordered),
            metadata=self.classifier_metadata,
        )
        decision_indices = batch.decision_indices
        duplicate_review_indices = {
            lower_index
            for lower_index, higher_index in contained_detection_pairs(
                ordered, self.quality_metadata.duplicate_review_containment_threshold
            )
            if decision_indices[lower_index, 0] == decision_indices[higher_index, 0]
        }

        border_indices: set[int] = set()
        if self.quality_metadata.border_policy == "classifier_confidence":
            border_indices = border_detection_indices(image, ordered, self.quality_metadata)

        items = build_scan_items(
            ordered,
            batch,
            self.classifier_metadata,
            border_indices=border_indices,
            duplicate_review_indices=duplicate_review_indices,
        )
        response = ScanResponse(
            request_id=request_id,
            status=Status.SEGMENTATION,
            reason_codes=summarize_reason_codes(items),
            segmentations=items,
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
            worker_version=self.worker_version,
            detector_version=self.detector.version,
            classifier_version=self.classifier.version,
            embedder_version=self.embedder_version,
            detector_policy_version=self.detector_policy_version,
            classifier_policy_version=self.classifier_policy_version,
            catalog_version=self.catalog_version,
        )
        self._log(response, detector_ms=detector_ms, classifier_ms=classifier_ms)
        return response

    def close(self) -> None:
        """Release optional runtime resources owned by pipeline adapters."""

        closed: set[int] = set()
        for adapter in (self.classifier, self.detector):
            if id(adapter) in closed:
                continue
            closed.add(id(adapter))
            close = getattr(adapter, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _log(
        response: ScanResponse,
        *,
        detector_ms: float,
        classifier_ms: float,
        diagnostic_reason_codes: list[str] | None = None,
    ) -> None:
        LOGGER.info(
            "scan_complete",
            extra={
                "request_id": response.request_id,
                "status": response.status.value,
                "reason_codes": response.reason_codes,
                "diagnostic_reason_codes": diagnostic_reason_codes or [],
                "segmentation_count": len(response.segmentations),
                "detector_ms": round(detector_ms, 3),
                "classifier_ms": round(classifier_ms, 3),
                "processing_time_ms": round(response.processing_time_ms, 3),
                "worker_version": response.worker_version,
                "detector_version": response.detector_version,
                "classifier_version": response.classifier_version,
            },
        )
