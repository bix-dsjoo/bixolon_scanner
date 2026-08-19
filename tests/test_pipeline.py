from __future__ import annotations

import numpy as np

from bixolon_scanner.contracts import ItemStatus, Status
from bixolon_scanner.inference import Detection, DetectionResult
from bixolon_scanner.package import (
    CountVerifierMetadata,
    NeighborMaskClassifierMetadata,
    NeighborMaskClassifierView,
)
from bixolon_scanner.pipeline import DecisionPipeline
from bixolon_scanner.pipeline.ports import ClassificationResult


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
    assert response.reason_codes == ["SEGMENT_BELOW_APPROVAL_THRESHOLD"]
    assert [item.segmentation_id for item in response.items] == [
        "segmentation_001",
        "segmentation_002",
    ]
    assert [item.bbox.x for item in response.items] == [10, 50]
    assert response.items[0].status is ItemStatus.APPROVED
    assert response.items[1].status is ItemStatus.UNKNOWN
    assert len(response.items[1].top3) == 3
    assert classifier.calls == 1


def test_unknown_top3_uses_separate_ranking_logits(classifier_metadata, quality_metadata):
    classifier = FakeClassifier([[0.4, 0.3, 0.2]])
    classifier.logits = ClassificationResult(
        logits=np.asarray([[0.4, 0.3, 0.2]], dtype=np.float32),
        ranking_logits=np.asarray([[0.1, 0.2, 0.9]], dtype=np.float32),
    )
    pipeline = DecisionPipeline(
        FakeDetector(DetectionResult([Detection(10, 10, 40, 40, 0.95)])),
        classifier,
        classifier_metadata,
        quality_metadata,
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request-ranked")

    assert response.items[0].status is ItemStatus.UNKNOWN
    assert response.items[0].top3[0].class_id == "bread_03"


def test_unsafe_classifier_top3_becomes_segment_recapture(classifier_metadata, quality_metadata):
    classifier_metadata.approval_threshold = 0.9
    classifier_metadata.neighbor_mask_inference = NeighborMaskClassifierMetadata(
        views=[NeighborMaskClassifierView(name="mask", distance_bias=0.0, weight=1.0)],
        top3_safety_threshold=-2.96,
    )
    classifier = FakeClassifier([[0.4, 0.3, 0.2]])
    classifier.logits = ClassificationResult(
        logits=np.asarray([[0.4, 0.3, 0.2]], dtype=np.float32),
        ranking_logits=np.asarray([[0.5, 0.4, 0.3]], dtype=np.float32),
        approval_scores=np.asarray([0.1], dtype=np.float32),
        top3_safety_scores=np.asarray([-3.0], dtype=np.float32),
    )
    pipeline = DecisionPipeline(
        FakeDetector(DetectionResult([Detection(10, 10, 40, 40, 0.95)])),
        classifier,
        classifier_metadata,
        quality_metadata,
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "unsafe-top3")

    assert response.items[0].status is ItemStatus.SEGMENT_RECAPTURE
    assert response.items[0].reason_codes == ["CLASSIFIER_TOP3_UNSAFE"]
    assert response.reason_codes == ["SEGMENT_RECAPTURE_REQUIRED"]


def test_safe_classifier_top3_remains_unknown(classifier_metadata, quality_metadata):
    classifier_metadata.approval_threshold = 0.9
    classifier_metadata.neighbor_mask_inference = NeighborMaskClassifierMetadata(
        views=[NeighborMaskClassifierView(name="mask", distance_bias=0.0, weight=1.0)],
        top3_safety_threshold=-2.96,
    )
    classifier = FakeClassifier([[0.4, 0.3, 0.2]])
    classifier.logits = ClassificationResult(
        logits=np.asarray([[0.4, 0.3, 0.2]], dtype=np.float32),
        ranking_logits=np.asarray([[0.5, 0.4, 0.3]], dtype=np.float32),
        approval_scores=np.asarray([0.1], dtype=np.float32),
        top3_safety_scores=np.asarray([-2.9], dtype=np.float32),
    )
    pipeline = DecisionPipeline(
        FakeDetector(DetectionResult([Detection(10, 10, 40, 40, 0.95)])),
        classifier,
        classifier_metadata,
        quality_metadata,
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "safe-top3")

    assert response.items[0].status is ItemStatus.UNKNOWN
    assert response.items[0].reason_codes == ["BELOW_APPROVAL_THRESHOLD"]


def test_unsafe_staged_classifier_top3_becomes_segment_recapture(
    classifier_metadata, quality_metadata
):
    from bixolon_scanner.package import ClassifierView, StagedClassifierMetadata

    classifier_metadata.approval_threshold = 0.0
    classifier_metadata.staged_inference = StagedClassifierMetadata(
        center_crop_scale=0.855,
        views=[ClassifierView(name="base", affine=((1, 0, 0), (0, 1, 0)))],
        first_view="base",
        early_approval_threshold=1.0,
        final_views=["base"],
        top3_views=["base"],
        approval_metric="inverse_entropy",
        approval_threshold=-0.1,
        top3_safety_metric="inverse_entropy",
        top3_safety_threshold=-2.96,
    )
    classifier = FakeClassifier([[0.4, 0.3, 0.2]])
    classifier.logits = ClassificationResult(
        logits=np.asarray([[0.4, 0.3, 0.2]], dtype=np.float32),
        ranking_logits=np.asarray([[0.5, 0.4, 0.3]], dtype=np.float32),
        approval_scores=np.asarray([-0.2], dtype=np.float32),
        top3_safety_scores=np.asarray([-3.0], dtype=np.float32),
    )
    pipeline = DecisionPipeline(
        FakeDetector(DetectionResult([Detection(10, 10, 40, 40, 0.95)])),
        classifier,
        classifier_metadata,
        quality_metadata,
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "unsafe-staged")

    assert response.items[0].status is ItemStatus.SEGMENT_RECAPTURE
    assert response.items[0].reason_codes == ["CLASSIFIER_TOP3_UNSAFE"]


def test_per_class_approval_threshold_uses_predicted_class(classifier_metadata, quality_metadata):
    classifier_metadata.approval_threshold = 0.1
    classifier_metadata.approval_thresholds = [0.3, None, None]
    classifier = FakeClassifier([])
    classifier.logits = ClassificationResult(
        logits=np.asarray([[3.0, 1.0, 0.0], [1.0, 3.0, 0.0]], dtype=np.float32),
        ranking_logits=np.asarray([[3.0, 1.0, 0.0], [1.0, 3.0, 0.0]], dtype=np.float32),
        approval_scores=np.asarray([0.2, 0.2], dtype=np.float32),
    )
    pipeline = DecisionPipeline(
        FakeDetector(
            DetectionResult([Detection(10, 10, 30, 30, 0.9), Detection(40, 40, 60, 60, 0.8)])
        ),
        classifier,
        classifier_metadata,
        quality_metadata,
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "per-class")

    assert response.items[0].status is ItemStatus.UNKNOWN
    assert response.items[1].status is ItemStatus.APPROVED


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


def test_confident_same_class_contained_duplicate_is_unknown_without_recapture(
    classifier_metadata, quality_metadata
):
    quality_metadata.duplicate_review_containment_threshold = 0.999
    detections = [
        Detection(10, 10, 90, 90, 0.90),
        Detection(20, 20, 80, 80, 0.95),
    ]
    classifier = FakeClassifier([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    pipeline = DecisionPipeline(
        FakeDetector(DetectionResult(detections)), classifier, classifier_metadata, quality_metadata
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "duplicate-review")

    assert response.status is Status.SEGMENTATION
    assert response.reason_codes == ["SEGMENT_DUPLICATE_REVIEW_REQUIRED"]
    assert len(response.items) == 2
    assert response.items[0].status is ItemStatus.UNKNOWN
    assert response.items[0].reason_codes == ["DETECTOR_CONTAINED_DUPLICATE"]
    assert len(response.items[0].top3) == 3
    assert response.items[1].status is ItemStatus.APPROVED
    assert classifier.calls == 1


def test_low_confidence_contained_detection_keeps_threshold_unknown_reason(
    classifier_metadata, quality_metadata
):
    quality_metadata.duplicate_review_containment_threshold = 0.999
    detections = [
        Detection(10, 10, 90, 90, 0.90),
        Detection(20, 20, 80, 80, 0.95),
    ]
    classifier = FakeClassifier([[0.4, 0.3, 0.2], [10.0, 0.0, 0.0]])
    pipeline = DecisionPipeline(
        FakeDetector(DetectionResult(detections)), classifier, classifier_metadata, quality_metadata
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "contained-uncertain")

    assert response.reason_codes == ["SEGMENT_BELOW_APPROVAL_THRESHOLD"]
    assert response.items[0].status is ItemStatus.UNKNOWN
    assert response.items[0].reason_codes == ["BELOW_APPROVAL_THRESHOLD"]
    assert response.items[1].status is ItemStatus.APPROVED


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


def test_uncertain_border_item_is_segment_recapture_after_classifier(
    classifier_metadata, quality_metadata
):
    quality_metadata.border_policy = "classifier_confidence"
    classifier = FakeClassifier([[0.4, 0.3, 0.2]])
    result = DetectionResult([Detection(0, 10, 40, 40, 0.95)])
    pipeline = DecisionPipeline(
        FakeDetector(result), classifier, classifier_metadata, quality_metadata
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request07")

    assert response.status is Status.SEGMENTATION
    assert response.reason_codes == ["SEGMENT_RECAPTURE_REQUIRED"]
    assert response.model_versions.classifier == "1.0.0"
    assert response.items[0].status is ItemStatus.SEGMENT_RECAPTURE
    assert response.items[0].reason_codes == ["DETECTOR_BORDER_CLIPPED"]
    assert classifier.calls == 1


def test_classifier_quality_class_is_segment_recapture(classifier_metadata, quality_metadata):
    classifier_metadata.labels[1].recapture = True
    classifier = FakeClassifier([[0.0, 10.0, 0.0]])
    pipeline = DecisionPipeline(
        FakeDetector(DetectionResult([Detection(10, 10, 40, 40, 0.95)])),
        classifier,
        classifier_metadata,
        quality_metadata,
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request-quality")

    assert response.status is Status.SEGMENTATION
    assert response.items[0].status is ItemStatus.SEGMENT_RECAPTURE
    assert response.items[0].reason_codes == ["CLASSIFIER_QUALITY_CLASS"]


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
