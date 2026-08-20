from __future__ import annotations

import logging
import time

import numpy as np
from PIL import Image

from ..contracts import (
    BoundingBox,
    Candidate,
    ItemStatus,
    ModelVersions,
    Prediction,
    ScanItem,
    ScanResponse,
    Status,
)
from ..contracts.image import image_original_size
from ..contracts.model_package import ClassifierMetadata, CountVerifierMetadata, QualityMetadata
from .ports import ClassificationResult, Classifier, Detection, DetectionResult, Detector

LOGGER = logging.getLogger(__name__)


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(scaled)
    return (exponential / exponential.sum(axis=1, keepdims=True)).astype(np.float32)


softmax = _softmax


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
        if metadata.min_mean_luminance is not None and mean_luminance < metadata.min_mean_luminance:
            reasons.append("DETECTOR_UNDEREXPOSED")
        if metadata.max_mean_luminance is not None and mean_luminance > metadata.max_mean_luminance:
            reasons.append("DETECTOR_OVEREXPOSED")
    if metadata.min_sharpness is not None and _sharpness(image) < metadata.min_sharpness:
        reasons.append("DETECTOR_BLUR")
    return list(dict.fromkeys(reasons))


def _touches_border(detection: Detection, width: int, height: int, margin_ratio: float) -> bool:
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


def _contained_detection_pairs(
    detections: list[Detection], threshold: float | None
) -> list[tuple[int, int]]:
    """Return (lower-score, higher-score) pairs with near-complete box containment."""
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
                0.0, min(left.y2, right.y2) - max(left.y1, right.y1)
            )
            if intersection / smaller_area < threshold:
                continue
            if left.score < right.score:
                pairs.append((left_index, right_index))
            elif right.score < left.score:
                pairs.append((right_index, left_index))
    return pairs


def _top3_candidates(
    metadata: ClassifierMetadata,
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
) -> list[Candidate]:
    return [
        Candidate(
            class_id=metadata.labels[int(candidate_index)].class_id,
            class_name=metadata.labels[int(candidate_index)].class_name,
            confidence=float(candidate_scores[int(candidate_index)]),
        )
        for candidate_index in candidate_indices[:3]
    ]


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
            elif detection_result.verified_count != len(detection_result.detections):
                reasons.append("DETECTOR_COUNT_MISMATCH")
            reasons = list(dict.fromkeys(reasons))
        if reasons:
            response = ScanResponse(
                request_id=request_id,
                status=Status.IMAGE_RECAPTURE,
                reason_codes=reasons,
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
            self._log(response, detector_ms=detector_ms, classifier_ms=0.0)
            return response

        ordered = sorted(
            detection_result.detections, key=lambda detection: (detection.y1, detection.x1)
        )
        classifier_started = time.perf_counter()
        classification = self.classifier.classify(image, ordered)
        classifier_ms = (time.perf_counter() - classifier_started) * 1000.0
        if isinstance(classification, ClassificationResult):
            logits = classification.logits
            ranking_logits = classification.ranking_logits
            approval_scores = classification.approval_scores
            top3_safety_scores = classification.top3_safety_scores
            ranking_scores = classification.ranking_scores
            segment_recapture_reasons = classification.segment_recapture_reasons
            unknown_reasons = classification.unknown_reasons
            approval_blocked = classification.approval_blocked
        else:
            logits = classification
            ranking_logits = classification
            approval_scores = None
            top3_safety_scores = None
            ranking_scores = None
            segment_recapture_reasons = None
            unknown_reasons = None
            approval_blocked = None
        if logits.shape != (len(ordered), len(self.classifier_metadata.labels)):
            raise ValueError("classifier output shape does not match package labels")
        if ranking_logits.shape != logits.shape:
            raise ValueError("classifier ranking output shape does not match logits")
        probabilities = _softmax(logits, self.classifier_metadata.temperature)
        ranking_probabilities = (
            _softmax(ranking_logits, self.classifier_metadata.temperature)
            if ranking_scores is None
            else np.asarray(ranking_scores, dtype=np.float32)
        )
        if ranking_probabilities.shape != logits.shape or np.any(
            (ranking_probabilities < 0.0) | (ranking_probabilities > 1.0)
        ):
            raise ValueError("classifier ranking scores must match labels and be in [0, 1]")
        decision_indices = np.argsort(-ranking_probabilities, axis=1, kind="stable")
        ranking_indices = np.argsort(-ranking_probabilities, axis=1, kind="stable")
        if approval_scores is None:
            approval_scores = probabilities.max(axis=1)
        if approval_scores.shape != (len(ordered),):
            raise ValueError("classifier approval scores do not match detections")
        if top3_safety_scores is not None and top3_safety_scores.shape != (len(ordered),):
            raise ValueError("classifier Top-3 safety scores do not match detections")
        if segment_recapture_reasons is not None and len(segment_recapture_reasons) != len(ordered):
            raise ValueError("classifier recapture reasons do not match detections")
        if unknown_reasons is not None and len(unknown_reasons) != len(ordered):
            raise ValueError("classifier unknown reasons do not match detections")
        if approval_blocked is not None:
            approval_blocked = np.asarray(approval_blocked, dtype=bool)
            if approval_blocked.shape != (len(ordered),):
                raise ValueError("classifier approval blocks do not match detections")
        configured_thresholds = self.classifier_metadata.approval_thresholds
        if configured_thresholds is None:
            staged_policy = self.classifier_metadata.staged_inference
            default_threshold = (
                self.classifier_metadata.approval_threshold
                if staged_policy is None or staged_policy.approval_threshold is None
                else staged_policy.approval_threshold
            )
            approval_thresholds = np.full(len(ordered), default_threshold, dtype=np.float32)
        else:
            approval_thresholds = np.asarray(
                [
                    self.classifier_metadata.approval_threshold
                    if configured_thresholds[int(indices[0])] is None
                    else configured_thresholds[int(indices[0])]
                    for indices in decision_indices
                ],
                dtype=np.float32,
            )
        approved = approval_scores >= approval_thresholds
        if approval_blocked is not None:
            approved &= ~approval_blocked
        mask_policy = self.classifier_metadata.neighbor_mask_inference
        staged_policy = self.classifier_metadata.staged_inference
        top3_safety_threshold = (
            mask_policy.top3_safety_threshold
            if mask_policy is not None
            else (staged_policy.top3_safety_threshold if staged_policy is not None else None)
        )
        top3_unsafe = (
            np.zeros(len(ordered), dtype=bool)
            if top3_safety_scores is None or top3_safety_threshold is None
            else top3_safety_scores < top3_safety_threshold
        )
        duplicate_review_indices = {
            lower_index
            for lower_index, higher_index in _contained_detection_pairs(
                ordered, self.quality_metadata.duplicate_review_containment_threshold
            )
            if decision_indices[lower_index, 0] == decision_indices[higher_index, 0]
        }

        recapture_labels = {
            index for index, label in enumerate(self.classifier_metadata.labels) if label.recapture
        }
        border_indices: set[int] = set()
        if self.quality_metadata.border_policy == "classifier_confidence":
            border_indices = _border_detection_indices(image, ordered, self.quality_metadata)

        items: list[ScanItem] = []
        for index, (detection, indices, scores, candidate_indices, candidate_scores) in enumerate(
            zip(
                ordered,
                decision_indices,
                probabilities,
                ranking_indices,
                ranking_probabilities,
            )
        ):
            ordinal = index + 1
            top1_index = int(indices[0])
            top1_score = float(scores[top1_index])
            label = self.classifier_metadata.labels[top1_index]
            bbox = BoundingBox(
                x=max(0, int(round(detection.x1))),
                y=max(0, int(round(detection.y1))),
                width=max(1, int(round(detection.x2 - detection.x1))),
                height=max(1, int(round(detection.y2 - detection.y1))),
            )
            if top1_index in recapture_labels:
                item = ScanItem(
                    segmentation_id=f"segmentation_{ordinal:03d}",
                    bbox=bbox,
                    status=ItemStatus.SEGMENT_RECAPTURE,
                    reason_codes=["CLASSIFIER_QUALITY_CLASS"],
                    prediction=None,
                    top3=[],
                    confidence=top1_score,
                )
            elif index in border_indices and not approved[index]:
                item = ScanItem(
                    segmentation_id=f"segmentation_{ordinal:03d}",
                    bbox=bbox,
                    status=ItemStatus.SEGMENT_RECAPTURE,
                    reason_codes=["DETECTOR_BORDER_CLIPPED"],
                    prediction=None,
                    top3=[],
                    confidence=top1_score,
                )
            elif segment_recapture_reasons is not None and segment_recapture_reasons[index]:
                item = ScanItem(
                    segmentation_id=f"segmentation_{ordinal:03d}",
                    bbox=bbox,
                    status=ItemStatus.SEGMENT_RECAPTURE,
                    reason_codes=[segment_recapture_reasons[index]],
                    prediction=None,
                    top3=[],
                    confidence=0.0,
                )
            elif index in duplicate_review_indices and approved[index]:
                candidates = _top3_candidates(
                    self.classifier_metadata, candidate_indices, candidate_scores
                )
                item = ScanItem(
                    segmentation_id=f"segmentation_{ordinal:03d}",
                    bbox=bbox,
                    status=ItemStatus.UNKNOWN,
                    reason_codes=["DETECTOR_CONTAINED_DUPLICATE"],
                    prediction=None,
                    top3=candidates,
                    confidence=float(approval_scores[index]),
                )
            elif approved[index]:
                item = ScanItem(
                    segmentation_id=f"segmentation_{ordinal:03d}",
                    bbox=bbox,
                    status=ItemStatus.APPROVED,
                    reason_codes=[],
                    prediction=Prediction(class_id=label.class_id, class_name=label.class_name),
                    top3=[],
                    confidence=float(approval_scores[index]),
                )
            elif top3_unsafe[index]:
                item = ScanItem(
                    segmentation_id=f"segmentation_{ordinal:03d}",
                    bbox=bbox,
                    status=ItemStatus.SEGMENT_RECAPTURE,
                    reason_codes=["CLASSIFIER_TOP3_UNSAFE"],
                    prediction=None,
                    top3=[],
                    confidence=(
                        0.0
                        if ranking_scores is not None
                        else float(candidate_scores[int(candidate_indices[0])])
                    ),
                )
            else:
                candidates = _top3_candidates(
                    self.classifier_metadata, candidate_indices, candidate_scores
                )
                item = ScanItem(
                    segmentation_id=f"segmentation_{ordinal:03d}",
                    bbox=bbox,
                    status=ItemStatus.UNKNOWN,
                    reason_codes=[
                        "BELOW_APPROVAL_THRESHOLD"
                        if unknown_reasons is None or unknown_reasons[index] is None
                        else unknown_reasons[index]
                    ],
                    prediction=None,
                    top3=candidates,
                    confidence=float(approval_scores[index]),
                )
            items.append(item)

        reason_codes: list[str] = []
        if any(
            item.status is ItemStatus.UNKNOWN
            and item.reason_codes != ["DETECTOR_CONTAINED_DUPLICATE"]
            for item in items
        ):
            reason_codes.append("SEGMENT_BELOW_APPROVAL_THRESHOLD")
        if any("DETECTOR_CONTAINED_DUPLICATE" in item.reason_codes for item in items):
            reason_codes.append("SEGMENT_DUPLICATE_REVIEW_REQUIRED")
        if any(item.status is ItemStatus.SEGMENT_RECAPTURE for item in items):
            reason_codes.append("SEGMENT_RECAPTURE_REQUIRED")
        response = ScanResponse(
            request_id=request_id,
            status=Status.SEGMENTATION,
            reason_codes=reason_codes,
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

    @staticmethod
    def _log(response: ScanResponse, *, detector_ms: float, classifier_ms: float) -> None:
        LOGGER.info(
            "scan_complete",
            extra={
                "request_id": response.request_id,
                "status": response.status.value,
                "reason_codes": response.reason_codes,
                "segmentation_count": len(response.segmentations),
                "detector_ms": round(detector_ms, 3),
                "classifier_ms": round(classifier_ms, 3),
                "processing_time_ms": round(response.processing_time_ms, 3),
                "worker_version": response.worker_version,
                "detector_version": response.detector_version,
                "classifier_version": response.classifier_version,
            },
        )
