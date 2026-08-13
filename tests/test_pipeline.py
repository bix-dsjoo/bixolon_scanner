from __future__ import annotations

import numpy as np

from bixolon_scanner.contracts import ItemStatus, Status
from bixolon_scanner.inference import Detection, DetectionResult
from bixolon_scanner.package import CountVerifierMetadata
from bixolon_scanner.pipeline import DecisionPipeline


class FakeDetector:
    version = "1.0.0"

    def __init__(self, result: DetectionResult):
        self.result = result

    def detect(self, image):
        return self.result


class FakeClassifier:
    version = "1.0.0"

    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float32)
        self.calls = 0

    def classify(self, image, detections):
        self.calls += 1
        return self.logits


def test_detector_recapture_skips_classifier(classifier_metadata, quality_metadata):
    classifier = FakeClassifier([])
    pipeline = DecisionPipeline(
        FakeDetector(DetectionResult([])), classifier, classifier_metadata, quality_metadata
    )
    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request01")
    assert response.status is Status.RECAPTURE
    assert response.reason_codes == ["DETECTOR_NO_OBJECT"]
    assert response.model_versions.classifier is None
    assert classifier.calls == 0


def test_025_bundle_reports_one_version_and_preserves_early_exit_null(
    classifier_metadata, quality_metadata
):
    detector = FakeDetector(DetectionResult([Detection(10, 10, 40, 40, 0.95)]))
    classifier = FakeClassifier([[10.0, 0.0, 0.0]])
    detector.version = "0.2.5"
    classifier.version = "0.2.5"
    pipeline = DecisionPipeline(detector, classifier, classifier_metadata, quality_metadata)

    approved = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request-025-approved")
    assert approved.model_versions.detector == "0.2.5"
    assert approved.model_versions.classifier == "0.2.5"

    detector.result = DetectionResult([])
    recapture = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request-025-recapture")
    assert recapture.model_versions.detector == "0.2.5"
    assert recapture.model_versions.classifier is None


def test_multiple_items_are_sorted_and_aggregated_unknown(classifier_metadata, quality_metadata):
    detections = [
        Detection(50, 50, 80, 80, 0.9),
        Detection(10, 10, 40, 40, 0.95),
    ]
    classifier = FakeClassifier([[7.0, 0.0, -1.0], [0.4, 0.3, 0.2]])
    pipeline = DecisionPipeline(
        FakeDetector(DetectionResult(detections)), classifier, classifier_metadata, quality_metadata
    )
    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request02")
    assert response.status is Status.UNKNOWN
    assert response.reason_codes == ["ITEM_BELOW_APPROVAL_THRESHOLD"]
    assert [item.item_id for item in response.items] == ["item_001", "item_002"]
    assert [item.bbox.x for item in response.items] == [10, 50]
    assert response.items[0].status is ItemStatus.APPROVED
    assert response.items[1].status is ItemStatus.UNKNOWN
    assert len(response.items[1].top3) == 3
    assert classifier.calls == 1


def test_all_items_approved(classifier_metadata, quality_metadata):
    detections = [Detection(10, 10, 40, 40, 0.95)]
    classifier = FakeClassifier([[10.0, 0.0, 0.0]])
    pipeline = DecisionPipeline(
        FakeDetector(DetectionResult(detections)), classifier, classifier_metadata, quality_metadata
    )
    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request03")
    assert response.status is Status.APPROVED
    assert response.reason_codes == []
    assert response.items[0].prediction.class_id == "bread_01"


def test_capacity_saturation_recaptures(classifier_metadata, quality_metadata):
    classifier = FakeClassifier([])
    result = DetectionResult([Detection(10, 10, 40, 40, 0.9)], capacity_saturated=True)
    pipeline = DecisionPipeline(
        FakeDetector(result), classifier, classifier_metadata, quality_metadata
    )
    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request04")
    assert response.status is Status.RECAPTURE
    assert response.reason_codes == ["DETECTOR_CAPACITY_EXCEEDED"]
    assert classifier.calls == 0


def test_legacy_border_policy_recaptures_before_classifier(classifier_metadata, quality_metadata):
    classifier = FakeClassifier([[10.0, 0.0, 0.0]])
    result = DetectionResult([Detection(0, 10, 40, 40, 0.95)])
    pipeline = DecisionPipeline(
        FakeDetector(result), classifier, classifier_metadata, quality_metadata
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request05")

    assert response.status is Status.RECAPTURE
    assert response.reason_codes == ["DETECTOR_BORDER_CLIPPED"]
    assert response.model_versions.classifier is None
    assert classifier.calls == 0


def test_confident_border_item_is_approved(classifier_metadata, quality_metadata):
    quality_metadata.border_policy = "classifier_confidence"
    classifier = FakeClassifier([[10.0, 0.0, 0.0]])
    result = DetectionResult([Detection(0, 10, 40, 40, 0.95)])
    pipeline = DecisionPipeline(
        FakeDetector(result), classifier, classifier_metadata, quality_metadata
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request06")

    assert response.status is Status.APPROVED
    assert response.reason_codes == []
    assert classifier.calls == 1


def test_uncertain_border_item_recaptures_after_classifier(classifier_metadata, quality_metadata):
    quality_metadata.border_policy = "classifier_confidence"
    classifier = FakeClassifier([[0.4, 0.3, 0.2]])
    result = DetectionResult([Detection(0, 10, 40, 40, 0.95)])
    pipeline = DecisionPipeline(
        FakeDetector(result), classifier, classifier_metadata, quality_metadata
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request07")

    assert response.status is Status.RECAPTURE
    assert response.reason_codes == ["DETECTOR_BORDER_CLIPPED"]
    assert response.model_versions.classifier == "1.0.0"
    assert response.items == []
    assert classifier.calls == 1


def _count_metadata(confidence_threshold=0.9):
    return CountVerifierMetadata(
        filename="count_verifier.onnx",
        version="1.0.0",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        count_labels=[1, 2, 3],
        confidence_threshold=confidence_threshold,
    )


def test_count_mismatch_recaptures_before_classifier(classifier_metadata, quality_metadata):
    classifier = FakeClassifier([])
    result = DetectionResult(
        [Detection(10, 10, 40, 40, 0.95)],
        verified_count=2,
        count_confidence=0.99,
    )
    pipeline = DecisionPipeline(
        FakeDetector(result),
        classifier,
        classifier_metadata,
        quality_metadata,
        _count_metadata(),
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request08")

    assert response.status is Status.RECAPTURE
    assert response.reason_codes == ["DETECTOR_COUNT_MISMATCH"]
    assert response.model_versions.classifier is None
    assert classifier.calls == 0


def test_uncertain_count_recaptures_before_classifier(classifier_metadata, quality_metadata):
    classifier = FakeClassifier([])
    result = DetectionResult(
        [Detection(10, 10, 40, 40, 0.95)],
        verified_count=1,
        count_confidence=0.75,
    )
    pipeline = DecisionPipeline(
        FakeDetector(result),
        classifier,
        classifier_metadata,
        quality_metadata,
        _count_metadata(),
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request09")

    assert response.status is Status.RECAPTURE
    assert response.reason_codes == ["DETECTOR_COUNT_UNCERTAIN"]
    assert classifier.calls == 0


def test_verified_count_allows_classifier(classifier_metadata, quality_metadata):
    classifier = FakeClassifier([[10.0, 0.0, 0.0]])
    result = DetectionResult(
        [Detection(10, 10, 40, 40, 0.95)],
        verified_count=1,
        count_confidence=0.99,
    )
    pipeline = DecisionPipeline(
        FakeDetector(result),
        classifier,
        classifier_metadata,
        quality_metadata,
        _count_metadata(),
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request10")

    assert response.status is Status.APPROVED
    assert classifier.calls == 1


def test_uncertain_detector_candidate_recaptures_before_classifier(
    classifier_metadata, quality_metadata
):
    classifier = FakeClassifier([])
    result = DetectionResult([Detection(10, 10, 40, 40, 0.95)], uncertain_candidate_count=1)
    pipeline = DecisionPipeline(
        FakeDetector(result), classifier, classifier_metadata, quality_metadata
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request11")

    assert response.status is Status.RECAPTURE
    assert response.reason_codes == ["DETECTOR_UNCERTAIN_OBJECT"]
    assert classifier.calls == 0
