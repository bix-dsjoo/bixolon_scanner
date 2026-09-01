from __future__ import annotations

from threading import Event
from types import SimpleNamespace

import numpy as np

from bixolon_scanner.pipeline.ports import Detection, DetectionResult
from bixolon_scanner.runtime.assisted_detector import ClassifierAssistedEnsembleDetector


class _Detector:
    version = "0.1.7"

    def __init__(self, verification_started: Event) -> None:
        self.verification_started = verification_started
        self.closed = False
        self.ensemble = SimpleNamespace(
            class_verified_selector=SimpleNamespace(
                zero_count_minimum_confidence=0.6,
                count_assistance_minimum_dimension=1000,
                count_assistance_on_count_mismatch=True,
            )
        )

    def predict_candidates(self, image):
        del image
        assert self.verification_started.wait(timeout=1.0)
        empty = {
            "boxes_xyxy": [],
            "scores": [],
            "class_ids": [],
            "detector_class_ids": [],
            "class_support_counts": [],
        }
        return empty, empty, 0, False, False

    def close(self) -> None:
        self.closed = True


class _Verifier:
    def __init__(self, started: Event, result: tuple[int, float] = (0, 0.9)) -> None:
        self.started = started
        self.result = result
        self.closed = False

    def verify(self, image):
        del image
        self.started.set()
        return self.result

    def close(self) -> None:
        self.closed = True


class _LowResolutionDetector(_Detector):
    def __init__(self, verification_started: Event) -> None:
        super().__init__(verification_started)
        self.base_detected = False
        self.low_resolution_override = None
        self.ensemble.class_verified_selector.low_resolution_maximum_dimension = 1000

    def detect(self, image):
        del image
        self.base_detected = True
        return DetectionResult(
            detections=[Detection(1.0, 2.0, 10.0, 12.0, 0.9)],
            detector_class_ids=(3,),
            detector_class_support_counts=(4,),
        )

    def predict_candidates(self, image, *, apply_low_resolution_override=True):
        del image
        self.low_resolution_override = apply_low_resolution_override
        selected = {
            "boxes_xyxy": [[1.0, 2.0, 10.0, 12.0]],
            "scores": [0.9],
            "class_ids": [0],
        }
        raw = {
            **selected,
            "detector_class_ids": [3],
            "class_support_counts": [4],
        }
        return selected, raw, 4, False, False


class _LowResolutionFallbackDetector(_LowResolutionDetector):
    def __init__(self, verification_started: Event) -> None:
        super().__init__(verification_started)
        self.ensemble.class_verified_selector.low_resolution_positive_count_minimum_confidence = (
            0.55
        )
        self.ensemble.class_verified_selector.low_resolution_recovery_maximum_count = 1
        self.ensemble.class_verified_selector.low_resolution_recovery_maximum_aspect_ratio = 2.0
        self.ensemble.class_verified_selector.low_resolution_recovery_minimum_approval_score = 0.1
        self.ensemble.class_verified_selector.candidate_minimum_score = 0.03
        self.ensemble.class_verified_selector.candidate_minimum_support = 3

    def predict_candidates(self, image, *, apply_low_resolution_override=True):
        if not apply_low_resolution_override:
            return super().predict_candidates(
                image,
                apply_low_resolution_override=apply_low_resolution_override,
            )
        del image
        self.low_resolution_override = apply_low_resolution_override
        selected = {
            "boxes_xyxy": [],
            "scores": [],
            "class_ids": [],
        }
        raw = {
            "boxes_xyxy": [[1.0, 2.0, 10.0, 12.0]],
            "scores": [0.04],
            "class_ids": [0],
            "support_counts": [3],
            "detector_class_ids": [3],
            "class_support_counts": [4],
        }
        return selected, raw, 0, False, False


class _LowResolutionEnsembleFallbackDetector(_LowResolutionDetector):
    def __init__(self, verification_started: Event) -> None:
        super().__init__(verification_started)
        self.calls: list[bool] = []
        policy = self.ensemble.class_verified_selector
        policy.low_resolution_maximum_dimension = 5000
        policy.low_resolution_primary_member_filename = "detector-production.onnx"
        policy.low_resolution_ensemble_fallback_minimum_dimension = 1000
        policy.low_resolution_ensemble_fallback_minimum_count_confidence = 0.7

    def predict_candidates(self, image, *, apply_low_resolution_override=True):
        del image
        self.calls.append(apply_low_resolution_override)
        box_count = 1 if apply_low_resolution_override else 2
        boxes = [
            [float(index * 20), 1.0, float(index * 20 + 10), 11.0] for index in range(box_count)
        ]
        selected = {
            "boxes_xyxy": boxes,
            "scores": [0.9] * box_count,
            "class_ids": [0] * box_count,
        }
        raw = {
            **selected,
            "detector_class_ids": list(range(box_count)),
            "class_support_counts": [4] * box_count,
        }
        return selected, raw, 4, False, False


def test_assisted_detector_preserves_parallel_count_verification() -> None:
    started = Event()
    detector = _Detector(started)
    verifier = _Verifier(started)
    assisted = ClassifierAssistedEnsembleDetector(
        detector,
        verifier,
        classifier=object(),
        parallel_verification=True,
    )
    try:
        result = assisted.detect(np.zeros((1200, 1200, 3), dtype=np.uint8))

        assert result.detections == []
        assert result.verified_count == 0
    finally:
        assisted.close()
    assert detector.closed
    assert verifier.closed


def test_assisted_detector_retains_base_low_resolution_selection() -> None:
    started = Event()
    detector = _LowResolutionDetector(started)
    verifier = _Verifier(started, result=(1, 0.9))
    verifier.metadata = SimpleNamespace(comparison_mode="exact_count")
    assisted = ClassifierAssistedEnsembleDetector(detector, verifier, classifier=object())

    try:
        result = assisted.detect(np.zeros((720, 743, 3), dtype=np.uint8))
    finally:
        assisted.close()

    assert detector.base_detected is False
    assert detector.low_resolution_override is True
    assert len(result.detections) == 1
    assert result.verified_count == 1
    assert result.uncertain_candidate_count == 0


def test_assisted_detector_uses_low_resolution_positive_count_fallback() -> None:
    started = Event()
    detector = _LowResolutionFallbackDetector(started)
    verifier = _Verifier(started, result=(1, 0.58))
    assisted = ClassifierAssistedEnsembleDetector(detector, verifier, classifier=object())
    assisted._classify_proposals = lambda image, selected, raw: [
        {
            "proposal_index": 0,
            "box": raw["boxes_xyxy"][0],
            "detector_score": raw["scores"][0],
            "support_count": raw["support_counts"][0],
            "predicted_class": 3,
            "approval_score": 0.9,
        }
    ]

    try:
        result = assisted.detect(np.zeros((720, 743, 3), dtype=np.uint8))
    finally:
        assisted.close()

    assert detector.low_resolution_override is True
    assert len(result.detections) == 1
    assert result.verified_count == 1


def test_assisted_detector_falls_back_to_ensemble_on_confident_large_image_count_mismatch() -> None:
    started = Event()
    detector = _LowResolutionEnsembleFallbackDetector(started)
    verifier = _Verifier(started, result=(2, 0.9))
    assisted = ClassifierAssistedEnsembleDetector(detector, verifier, classifier=object())

    try:
        result = assisted.detect(np.zeros((1200, 1600, 3), dtype=np.uint8))
    finally:
        assisted.close()

    assert detector.calls == [True, False]
    assert len(result.detections) == 2
    assert result.verified_count == 2
