from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bixolon_scanner.contracts import (
    DetectorPrimaryClassifierRoutingMetadata,
    ItemStatus,
    Status,
)
from bixolon_scanner.inference import Detection, DetectionResult
from bixolon_scanner.package import (
    CountVerifierMetadata,
    DetectorClassifierConsensusMetadata,
    NeighborMaskClassifierMetadata,
    NeighborMaskClassifierView,
)
from bixolon_scanner.pipeline import DecisionPipeline
from bixolon_scanner.pipeline.classification import (
    ClassifierBatch,
    merge_selected_classifier_batch,
)
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
        self.logits = (
            logits
            if isinstance(logits, ClassificationResult)
            else np.asarray(logits, dtype=np.float32)
        )
        self.calls = 0

    def classify(self, image, detections):
        self.calls += 1
        return self.logits


class SelectiveFakeClassifier(FakeClassifier):
    def __init__(self, selected_logits):
        super().__init__(selected_logits)
        self.selected_calls: list[np.ndarray] = []

    def classify(self, image, detections):
        raise AssertionError("full classifier path must not be used")

    def classify_selected(self, image, detections, detection_indices):
        del image, detections
        indices = np.asarray(detection_indices, dtype=np.int64)
        self.selected_calls.append(indices)
        return self.logits[: len(indices)]


class SequencedClassifier(FakeClassifier):
    def __init__(self, outputs):
        self.outputs = [np.asarray(output, dtype=np.float32) for output in outputs]
        self.calls = 0

    def classify(self, image, detections):
        del image, detections
        output = self.outputs[self.calls]
        self.calls += 1
        return output


class RefiningFakeDetector(FakeDetector):
    def __init__(self, result: DetectionResult, refined: DetectionResult):
        super().__init__(result)
        self.refined = refined
        self.refinement_calls = 0

    def refine_unknown_detections(self, image, current):
        del image
        assert current is self.result
        self.refinement_calls += 1
        return self.refined


class AssistedFakeClassifier(FakeClassifier):
    def __init__(self, contextual, single_view):
        self.logits = contextual
        self.calls = 0
        self.single_view = single_view

    def classify_single_views(self, image, detections):
        self.single_view_detection_counts = getattr(self, "single_view_detection_counts", [])
        self.single_view_detection_counts.append(len(detections))
        return self.single_view


class ResolutionFallbackFakeClassifier(FakeClassifier):
    def __init__(
        self,
        primary,
        fallback,
        *,
        fallback_on_unknown: bool,
        fallback_on_unsafe: bool = False,
        rules=(),
        review_rules=(),
        shape_review_rules=(),
        minimum_detector_support: int = 3,
        selective_roi_only: bool = False,
        fuse_unapproved_top3: bool = False,
        minimum_fallback_approval_score: float | None = None,
    ):
        self.logits = (
            primary
            if isinstance(primary, ClassificationResult)
            else np.asarray(primary, dtype=np.float32)
        )
        self.calls = 0
        self.fallback = (
            fallback
            if isinstance(fallback, ClassificationResult)
            else np.asarray(fallback, dtype=np.float32)
        )
        self.fallback_calls = 0
        self.selective_fallback_calls = 0
        self.selected_indices = None
        self.resolution_fallback_metadata = SimpleNamespace(
            fallback_on_unknown=fallback_on_unknown,
            fallback_on_unsafe=fallback_on_unsafe,
            selective_roi_only=selective_roi_only,
            fuse_unapproved_top3=fuse_unapproved_top3,
            minimum_fallback_approval_score=minimum_fallback_approval_score,
            minimum_detector_support=minimum_detector_support,
            approval_disagreement_rules=[
                SimpleNamespace(
                    minimum_detection_count=count,
                    maximum_detection_count=None,
                    maximum_approval_score=score,
                    require_detector_disagreement=True,
                    minimum_box_aspect_ratio=None,
                    maximum_approval_score_decrease=None,
                )
                for count, score in rules
            ]
            + [
                SimpleNamespace(
                    minimum_detection_count=count,
                    maximum_detection_count=count,
                    maximum_approval_score=score,
                    require_detector_disagreement=False,
                    minimum_box_aspect_ratio=None,
                    maximum_approval_score_decrease=None,
                )
                for count, score in review_rules
            ]
            + [
                SimpleNamespace(
                    minimum_detection_count=1,
                    maximum_detection_count=None,
                    maximum_approval_score=score,
                    require_detector_disagreement=False,
                    minimum_box_aspect_ratio=aspect_ratio,
                    maximum_approval_score_decrease=maximum_decrease,
                )
                for aspect_ratio, score, maximum_decrease in shape_review_rules
            ],
        )

    def classify_fallback(self, image, detections):
        del image, detections
        self.fallback_calls += 1
        return self.fallback

    def classify_fallback_selected(self, image, detections, detection_indices):
        del image, detections
        self.selective_fallback_calls += 1
        self.selected_indices = np.asarray(detection_indices, dtype=np.int64)
        if isinstance(self.fallback, ClassificationResult):
            raise AssertionError("this fake only slices array fallback results")
        return self.fallback[self.selected_indices]


def _consensus_quality(quality_metadata, *, disagreement_maximum: float = 0.9):
    return quality_metadata.model_copy(
        update={
            "detector_classifier_consensus": DetectorClassifierConsensusMetadata(
                minimum_detector_score=0.7,
                approved_disagreement_maximum_classifier_score=disagreement_maximum,
            )
        }
    )


def _detector_primary_routing(*direct_classes: int):
    return DetectorPrimaryClassifierRoutingMetadata(
        direct_approval_class_indices=list(direct_classes),
        minimum_detector_score=0.98,
        require_unique_class_per_image=True,
    )


def test_detector_primary_routing_classifies_only_risky_rois(
    classifier_metadata,
    quality_metadata,
):
    detector = FakeDetector(
        DetectionResult(
            [
                Detection(10, 10, 40, 40, 0.995, class_id=2),
                Detection(50, 10, 80, 40, 0.999, class_id=1),
            ]
        )
    )
    classifier = SelectiveFakeClassifier([[0.0, 5.0, 0.0]])
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier_metadata,
        quality_metadata,
        detector_primary_classifier_routing=_detector_primary_routing(2),
    )

    response = pipeline.scan(
        np.full((100, 100, 3), 128, dtype=np.uint8),
        "detector-primary-selected",
    )

    assert len(classifier.selected_calls) == 1
    assert classifier.selected_calls[0].tolist() == [1]
    assert [item.status for item in response.items] == [
        ItemStatus.APPROVED,
        ItemStatus.APPROVED,
    ]
    assert [item.prediction.class_id for item in response.items] == [
        "bread_03",
        "bread_02",
    ]


def test_detector_primary_routing_skips_classifier_when_all_rois_are_safe(
    classifier_metadata,
    quality_metadata,
):
    detector = FakeDetector(DetectionResult([Detection(10, 10, 40, 40, 0.995, class_id=2)]))
    classifier = SelectiveFakeClassifier([[0.0, 0.0, 5.0]])
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier_metadata,
        quality_metadata,
        detector_primary_classifier_routing=_detector_primary_routing(2),
    )

    response = pipeline.scan(
        np.full((100, 100, 3), 128, dtype=np.uint8),
        "detector-primary-direct",
    )

    assert classifier.selected_calls == []
    assert response.items[0].status is ItemStatus.APPROVED
    assert response.items[0].prediction.class_id == "bread_03"
    assert response.items[0].confidence == pytest.approx(0.995)


@pytest.mark.parametrize(
    ("detections", "expected_indices"),
    [
        ([Detection(10, 10, 40, 40, 0.97, class_id=2)], [0]),
        (
            [
                Detection(10, 10, 40, 40, 0.995, class_id=2),
                Detection(50, 10, 80, 40, 0.996, class_id=2),
            ],
            [0, 1],
        ),
    ],
)
def test_detector_primary_routing_sends_low_score_and_duplicate_classes_to_classifier(
    classifier_metadata,
    quality_metadata,
    detections,
    expected_indices,
):
    detector = FakeDetector(DetectionResult(detections))
    classifier = SelectiveFakeClassifier(
        np.tile(np.asarray([[0.0, 0.0, 5.0]], dtype=np.float32), (len(detections), 1))
    )
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier_metadata,
        quality_metadata,
        detector_primary_classifier_routing=_detector_primary_routing(2),
    )

    pipeline.scan(
        np.full((100, 100, 3), 128, dtype=np.uint8),
        "detector-primary-guard",
    )

    assert classifier.selected_calls[0].tolist() == expected_indices


def test_detector_consensus_demotes_risky_approved_disagreement(
    classifier_metadata, quality_metadata
):
    detector = FakeDetector(DetectionResult([Detection(10, 10, 40, 40, 0.95, class_id=1)]))
    classifier = FakeClassifier(
        ClassificationResult(
            logits=np.asarray([[5.0, 1.0, 0.0]], dtype=np.float32),
            ranking_logits=np.asarray([[5.0, 1.0, 0.0]], dtype=np.float32),
            approval_scores=np.asarray([0.85], dtype=np.float32),
        )
    )
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier_metadata,
        _consensus_quality(quality_metadata),
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "consensus-demote")

    assert response.items[0].status is ItemStatus.UNKNOWN
    assert response.items[0].top3[0].class_id == "bread_02"


def test_detector_consensus_promotes_safe_unknown_top3_agreement(
    classifier_metadata, quality_metadata
):
    detector = FakeDetector(DetectionResult([Detection(10, 10, 40, 40, 0.95, class_id=1)]))
    classifier = FakeClassifier(
        ClassificationResult(
            logits=np.asarray([[0.4, 0.3, 0.2]], dtype=np.float32),
            ranking_logits=np.asarray([[0.4, 0.3, 0.2]], dtype=np.float32),
            approval_scores=np.asarray([0.2], dtype=np.float32),
        )
    )
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier_metadata,
        _consensus_quality(quality_metadata),
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "consensus-promote")

    assert response.items[0].status is ItemStatus.APPROVED
    assert response.items[0].prediction.class_id == "bread_02"


def test_detector_consensus_promotes_recapture_only_on_top1_agreement(
    classifier_metadata, quality_metadata
):
    detector = FakeDetector(DetectionResult([Detection(10, 10, 40, 40, 0.95, class_id=1)]))
    classifier = FakeClassifier(
        ClassificationResult(
            logits=np.asarray([[0.2, 0.4, 0.1]], dtype=np.float32),
            ranking_logits=np.asarray([[0.2, 0.4, 0.1]], dtype=np.float32),
            approval_scores=np.asarray([0.1], dtype=np.float32),
            segment_recapture_reasons=("RETRIEVAL_OOD",),
        )
    )
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier_metadata,
        _consensus_quality(quality_metadata),
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "consensus-ood")

    assert response.items[0].status is ItemStatus.APPROVED
    assert response.items[0].prediction.class_id == "bread_02"


def _assisted_policy(
    minimum_single_view_approval: float,
    *,
    risky_approval_fallback: bool = False,
    risky_approval_maximum_score: float | None = 1.0,
    risky_approval_minimum_aspect_ratio: float | None = None,
):
    return SimpleNamespace(
        wide_pair_detector_class_index=None,
        wide_pair_classifier_class_index=None,
        wide_pair_target_aspect_ratio=None,
        unknown_consensus_promotion=True,
        unknown_promotion_minimum_single_view_approval=(minimum_single_view_approval),
        single_view_top3_fusion=True,
        candidate_minimum_support=3,
        low_resolution_risky_approval_ensemble_fallback=risky_approval_fallback,
        low_resolution_risky_approval_maximum_score=risky_approval_maximum_score,
        low_resolution_risky_approval_minimum_aspect_ratio=(risky_approval_minimum_aspect_ratio),
    )


def test_detector_recapture_skips_classifier(classifier_metadata, quality_metadata):
    classifier = FakeClassifier([])
    pipeline = DecisionPipeline(
        FakeDetector(DetectionResult([])), classifier, classifier_metadata, quality_metadata
    )
    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request01")
    assert response.status is Status.RECAPTURE
    assert response.reason_codes == ["IMAGE_RECAPTURE_REQUIRED"]
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


def test_safe_unknown_uses_classifier_resolution_fallback(classifier_metadata, quality_metadata):
    detector = FakeDetector(
        DetectionResult(
            [Detection(10, 10, 40, 40, 0.95)],
            detector_class_ids=(1,),
            detector_class_support_counts=(3,),
        )
    )
    classifier = ResolutionFallbackFakeClassifier(
        [[0.4, 0.3, 0.2]],
        [[0.0, 10.0, 0.0]],
        fallback_on_unknown=True,
    )
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier_metadata,
        quality_metadata,
    )

    response = pipeline.scan(
        np.full((100, 100, 3), 128, dtype=np.uint8),
        "classifier-resolution-unknown",
    )

    assert classifier.calls == 1
    assert classifier.fallback_calls == 1
    assert response.items[0].status is ItemStatus.APPROVED
    assert response.items[0].prediction.class_id == "bread_02"


def test_selective_resolution_fallback_replaces_only_triggered_roi(
    classifier_metadata,
    quality_metadata,
):
    detector = FakeDetector(
        DetectionResult(
            [
                Detection(10, 10, 40, 40, 0.95),
                Detection(50, 10, 80, 40, 0.95),
            ]
        )
    )
    classifier = ResolutionFallbackFakeClassifier(
        [[10.0, 0.0, 0.0], [0.4, 0.3, 0.2]],
        [[0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        fallback_on_unknown=True,
        selective_roi_only=True,
    )
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier_metadata,
        quality_metadata,
    )

    response = pipeline.scan(
        np.full((100, 100, 3), 128, dtype=np.uint8),
        "classifier-resolution-selective",
    )

    assert classifier.fallback_calls == 0
    assert classifier.selective_fallback_calls == 1
    assert classifier.selected_indices.tolist() == [1]
    assert [item.prediction.class_id for item in response.items] == ["bread_01", "bread_03"]


def test_unsafe_fast_result_uses_classifier_resolution_fallback(
    classifier_metadata, quality_metadata
):
    detector = FakeDetector(
        DetectionResult(
            [Detection(10, 10, 40, 40, 0.95)],
            detector_class_ids=(1,),
            detector_class_support_counts=(3,),
        )
    )
    classifier = ResolutionFallbackFakeClassifier(
        ClassificationResult(
            logits=np.asarray([[0.4, 0.3, 0.2]], dtype=np.float32),
            ranking_logits=np.asarray([[0.4, 0.3, 0.2]], dtype=np.float32),
            approval_scores=np.asarray([0.05], dtype=np.float32),
            segment_recapture_reasons=("CLASSIFIER_TOP3_UNSAFE",),
        ),
        [[0.0, 10.0, 0.0]],
        fallback_on_unknown=True,
        fallback_on_unsafe=True,
    )
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier_metadata,
        quality_metadata,
    )

    response = pipeline.scan(
        np.full((100, 100, 3), 128, dtype=np.uint8),
        "classifier-resolution-unsafe",
    )

    assert classifier.fallback_calls == 1
    assert response.items[0].status is ItemStatus.APPROVED


@pytest.mark.parametrize(
    ("minimum_count", "expected_fallback_calls"),
    [(1, 1), (2, 0)],
)
def test_supported_approved_disagreement_uses_count_aware_classifier_fallback(
    classifier_metadata,
    quality_metadata,
    minimum_count,
    expected_fallback_calls,
):
    detector = FakeDetector(
        DetectionResult(
            [Detection(10, 10, 40, 40, 0.95)],
            detector_class_ids=(1,),
            detector_class_support_counts=(3,),
        )
    )
    classifier = ResolutionFallbackFakeClassifier(
        [[10.0, 0.0, 0.0]],
        [[0.0, 10.0, 0.0]],
        fallback_on_unknown=False,
        rules=[(minimum_count, 1.0)],
    )
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier_metadata,
        quality_metadata,
    )

    response = pipeline.scan(
        np.full((100, 100, 3), 128, dtype=np.uint8),
        f"classifier-resolution-count-{minimum_count}",
    )

    assert classifier.fallback_calls == expected_fallback_calls
    expected_class = "bread_02" if expected_fallback_calls else "bread_01"
    assert response.items[0].prediction.class_id == expected_class


def test_low_score_approved_dense_scene_can_use_resolution_fallback_without_disagreement(
    classifier_metadata,
    quality_metadata,
):
    detections = [Detection(index * 10 + 5, 10, index * 10 + 13, 40, 0.95) for index in range(5)]
    detector = FakeDetector(
        DetectionResult(
            detections,
            detector_class_ids=(0, 0, 0, 0, 0),
            detector_class_support_counts=(1, 1, 1, 1, 1),
        )
    )
    classifier = ResolutionFallbackFakeClassifier(
        np.tile([[10.0, 0.0, 0.0]], (5, 1)),
        np.tile([[0.0, 10.0, 0.0]], (5, 1)),
        fallback_on_unknown=False,
        review_rules=[(5, 1.0)],
    )
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier_metadata,
        quality_metadata,
    )

    response = pipeline.scan(
        np.full((100, 100, 3), 128, dtype=np.uint8),
        "classifier-resolution-dense-review",
    )

    assert classifier.fallback_calls == 1
    assert all(item.prediction.class_id == "bread_02" for item in response.items)


def test_extreme_aspect_fallback_demotes_resolution_unstable_approval(
    classifier_metadata,
    quality_metadata,
):
    detector = FakeDetector(
        DetectionResult(
            [
                Detection(10, 10, 90, 30, 0.95),
                Detection(10, 50, 50, 90, 0.95),
            ]
        )
    )
    primary = ClassificationResult(
        logits=np.asarray([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32),
        ranking_logits=np.asarray(
            [[5.0, 4.0, 0.0], [5.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        approval_scores=np.asarray([0.9, 0.9], dtype=np.float32),
    )
    fallback = ClassificationResult(
        logits=np.asarray([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32),
        ranking_logits=np.asarray(
            [[5.0, 0.0, 4.0], [5.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        approval_scores=np.asarray([0.6, 0.9], dtype=np.float32),
    )
    classifier = ResolutionFallbackFakeClassifier(
        primary,
        fallback,
        fallback_on_unknown=False,
        shape_review_rules=[(3.0, 0.95, 0.2)],
        fuse_unapproved_top3=True,
    )
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier_metadata.model_copy(update={"approval_threshold": 0.5}),
        quality_metadata,
    )

    response = pipeline.scan(
        np.full((100, 100, 3), 128, dtype=np.uint8),
        "classifier-resolution-shape-stability",
    )

    assert classifier.fallback_calls == 1
    assert response.items[0].status is ItemStatus.UNKNOWN
    assert [candidate.class_id for candidate in response.items[0].top3] == [
        "bread_01",
        "bread_02",
        "bread_03",
    ]
    assert response.items[1].status is ItemStatus.APPROVED


def test_fallback_minimum_approval_score_stabilizes_provider_boundary(
    classifier_metadata,
    quality_metadata,
):
    detector = FakeDetector(DetectionResult([Detection(10, 10, 40, 40, 0.95)]))
    primary = ClassificationResult(
        logits=np.asarray([[5.0, 0.0, 0.0]], dtype=np.float32),
        ranking_logits=np.asarray([[5.0, 0.0, 0.0]], dtype=np.float32),
        approval_scores=np.asarray([0.01], dtype=np.float32),
    )
    fallback = ClassificationResult(
        logits=np.asarray([[5.0, 0.0, 0.0]], dtype=np.float32),
        ranking_logits=np.asarray([[5.0, 0.0, 0.0]], dtype=np.float32),
        approval_scores=np.asarray([0.07], dtype=np.float32),
    )
    classifier = ResolutionFallbackFakeClassifier(
        primary,
        fallback,
        fallback_on_unknown=True,
        minimum_fallback_approval_score=0.08,
    )
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier_metadata.model_copy(update={"approval_threshold": 0.05}),
        quality_metadata,
    )

    response = pipeline.scan(
        np.full((100, 100, 3), 128, dtype=np.uint8),
        "classifier-resolution-provider-boundary",
    )

    assert classifier.fallback_calls == 1
    assert response.items[0].status is ItemStatus.UNKNOWN
    assert response.items[0].reason_codes == ["BELOW_APPROVAL_THRESHOLD"]


def test_merge_selected_classifier_batch_fuses_unapproved_ranking_candidates():
    def batch(ranking, *, approved):
        ranking_array = np.asarray([ranking], dtype=np.float32)
        return ClassifierBatch(
            probabilities=ranking_array,
            ranking_probabilities=ranking_array,
            decision_indices=np.argsort(-ranking_array, axis=1, kind="stable"),
            approval_scores=np.asarray([0.1], dtype=np.float32),
            approved=np.asarray([approved], dtype=bool),
            top3_unsafe=np.asarray([False], dtype=bool),
            segment_recapture_reasons=None,
            unknown_reasons=("BELOW_APPROVAL_THRESHOLD",),
            uses_explicit_ranking_scores=True,
        )

    merged = merge_selected_classifier_batch(
        batch([0.9, 0.8, 0.7, 0.1], approved=False),
        batch([0.1, 0.6, 0.2, 0.95], approved=False),
        np.asarray([0], dtype=np.int64),
        fuse_unapproved_top3=True,
    )

    np.testing.assert_allclose(
        merged.ranking_probabilities,
        [[0.9, 0.8, 0.7, 0.95]],
    )
    assert merged.decision_indices.tolist() == [[3, 0, 1, 2]]


def test_unknown_primary_result_can_trigger_detector_refinement(
    classifier_metadata, quality_metadata
):
    initial = DetectionResult([Detection(10, 10, 40, 40, 0.95)])
    refined = DetectionResult(
        [Detection(9, 9, 41, 41, 0.96)],
        detector_class_ids=(0,),
        detector_class_support_counts=(4,),
    )
    detector = RefiningFakeDetector(initial, refined)
    classifier = SequencedClassifier(
        [
            [[0.4, 0.3, 0.2]],
            [[10.0, 0.0, 0.0]],
        ]
    )
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier_metadata,
        quality_metadata,
    )

    response = pipeline.scan(
        np.full((100, 100, 3), 128, dtype=np.uint8),
        "unknown-refinement",
    )

    assert detector.refinement_calls == 1
    assert classifier.calls == 2
    assert response.items[0].status is ItemStatus.APPROVED
    assert response.items[0].bbox.x == 9


def test_supported_detector_class_disagreement_can_trigger_fallback_refinement(
    classifier_metadata, quality_metadata
):
    initial = DetectionResult(
        [Detection(10, 10, 40, 40, 0.95)],
        detector_class_ids=(1,),
        detector_class_support_counts=(3,),
    )
    refined = DetectionResult(
        [Detection(9, 9, 41, 41, 0.96)],
        detector_class_ids=(0,),
        detector_class_support_counts=(4,),
    )
    detector = RefiningFakeDetector(initial, refined)
    classifier = AssistedFakeClassifier(
        np.asarray([[10.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[10.0, 0.0, 0.0]], dtype=np.float32),
    )
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier_metadata,
        quality_metadata,
        assisted_policy=_assisted_policy(0.1, risky_approval_fallback=True),
    )

    response = pipeline.scan(
        np.full((100, 100, 3), 128, dtype=np.uint8),
        "risky-approval-refinement",
    )

    assert detector.refinement_calls == 1
    assert classifier.calls == 2
    assert response.items[0].bbox.x == 9


def test_confident_compact_detector_disagreement_stays_on_primary_path(
    classifier_metadata, quality_metadata
):
    initial = DetectionResult(
        [Detection(10, 10, 40, 40, 0.95)],
        detector_class_ids=(1,),
        detector_class_support_counts=(3,),
    )
    refined = DetectionResult([Detection(9, 9, 41, 41, 0.96)])
    detector = RefiningFakeDetector(initial, refined)
    classifier = AssistedFakeClassifier(
        np.asarray([[10.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[10.0, 0.0, 0.0]], dtype=np.float32),
    )
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier_metadata,
        quality_metadata,
        assisted_policy=_assisted_policy(
            0.1,
            risky_approval_fallback=True,
            risky_approval_maximum_score=0.5,
            risky_approval_minimum_aspect_ratio=2.0,
        ),
    )

    response = pipeline.scan(
        np.full((100, 100, 3), 128, dtype=np.uint8),
        "safe-disagreement-primary",
    )

    assert detector.refinement_calls == 0
    assert classifier.calls == 1
    assert response.items[0].bbox.x == 10


def test_single_view_promotion_can_trigger_fallback_refinement(
    classifier_metadata, quality_metadata
):
    contextual = ClassificationResult(
        logits=np.asarray([[1.0, 0.5, 0.0]], dtype=np.float32),
        ranking_logits=np.asarray([[1.0, 0.5, 0.0]], dtype=np.float32),
        approval_scores=np.asarray([0.05], dtype=np.float32),
    )
    single_view = ClassificationResult(
        logits=np.asarray([[0.5, 1.0, 0.0]], dtype=np.float32),
        ranking_logits=np.asarray([[0.5, 1.0, 0.0]], dtype=np.float32),
        approval_scores=np.asarray([0.2], dtype=np.float32),
    )
    initial = DetectionResult(
        [Detection(10, 10, 40, 40, 0.95)],
        detector_class_ids=(0,),
        detector_class_support_counts=(3,),
    )
    refined = DetectionResult(
        [Detection(9, 9, 41, 41, 0.96)],
        detector_class_ids=(0,),
        detector_class_support_counts=(4,),
    )
    detector = RefiningFakeDetector(initial, refined)
    pipeline = DecisionPipeline(
        detector,
        AssistedFakeClassifier(contextual, single_view),
        classifier_metadata,
        quality_metadata,
        assisted_policy=_assisted_policy(0.1, risky_approval_fallback=True),
    )

    response = pipeline.scan(
        np.full((100, 100, 3), 128, dtype=np.uint8),
        "promotion-refinement",
    )

    assert detector.refinement_calls == 1
    assert response.items[0].bbox.x == 9


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


@pytest.mark.parametrize(
    ("single_view_approval", "expected_status"),
    [(0.09, ItemStatus.UNKNOWN), (0.11, ItemStatus.APPROVED)],
)
def test_detector_consensus_promotion_requires_single_view_safety_floor(
    classifier_metadata,
    quality_metadata,
    single_view_approval,
    expected_status,
):
    contextual = ClassificationResult(
        logits=np.asarray([[1.0, 0.5, 0.0]], dtype=np.float32),
        ranking_logits=np.asarray([[1.0, 0.5, 0.0]], dtype=np.float32),
        approval_scores=np.asarray([0.05], dtype=np.float32),
    )
    single_view = ClassificationResult(
        logits=np.asarray([[0.5, 1.0, 0.0]], dtype=np.float32),
        ranking_logits=np.asarray([[0.5, 1.0, 0.0]], dtype=np.float32),
        approval_scores=np.asarray([single_view_approval], dtype=np.float32),
    )
    detector_result = DetectionResult(
        [Detection(10, 10, 40, 40, 0.95)],
        detector_class_ids=(0,),
        detector_class_support_counts=(3,),
    )
    pipeline = DecisionPipeline(
        FakeDetector(detector_result),
        AssistedFakeClassifier(contextual, single_view),
        classifier_metadata,
        quality_metadata,
        assisted_policy=_assisted_policy(0.1),
    )

    response = pipeline.scan(
        np.full((100, 100, 3), 128, dtype=np.uint8),
        f"promotion-{single_view_approval}",
    )

    assert response.items[0].status is expected_status
    if expected_status is ItemStatus.APPROVED:
        assert response.items[0].prediction.class_id == "bread_01"
    else:
        assert {candidate.class_id for candidate in response.items[0].top3} == {
            "bread_01",
            "bread_02",
            "bread_03",
        }


def test_single_view_verification_only_embeds_unapproved_detections(
    classifier_metadata, quality_metadata
):
    contextual = ClassificationResult(
        logits=np.asarray([[3.0, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=np.float32),
        ranking_logits=np.asarray([[3.0, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=np.float32),
        approval_scores=np.asarray([0.9, 0.05], dtype=np.float32),
    )
    single_view = ClassificationResult(
        logits=np.asarray([[0.4, 1.0, 0.0]], dtype=np.float32),
        ranking_logits=np.asarray([[0.4, 1.0, 0.0]], dtype=np.float32),
        approval_scores=np.asarray([0.05], dtype=np.float32),
    )
    classifier = AssistedFakeClassifier(contextual, single_view)
    detector_result = DetectionResult(
        [Detection(10, 10, 30, 30, 0.95), Detection(40, 40, 60, 60, 0.9)],
        detector_class_ids=(0, 1),
        detector_class_support_counts=(3, 3),
    )
    pipeline = DecisionPipeline(
        FakeDetector(detector_result),
        classifier,
        classifier_metadata,
        quality_metadata,
        assisted_policy=_assisted_policy(0.1),
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "selective-single-view")

    assert classifier.single_view_detection_counts == [1]
    assert response.items[0].status is ItemStatus.APPROVED
    assert response.items[1].status is ItemStatus.UNKNOWN


def test_2_0_direct_ranking_scores_and_catalog_reason(classifier_metadata, quality_metadata):
    classifier = FakeClassifier([[0.0, 0.0, 0.0]])
    classifier.logits = ClassificationResult(
        logits=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        ranking_logits=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        approval_scores=np.asarray([0.42], dtype=np.float32),
        ranking_scores=np.asarray([[0.91, 0.88, 0.1]], dtype=np.float32),
        unknown_reasons=("CLASSIFIER_CATALOG_CONFLICT",),
        approval_blocked=np.asarray([True]),
    )
    pipeline = DecisionPipeline(
        FakeDetector(DetectionResult([Detection(10, 10, 40, 40, 0.95)])),
        classifier,
        classifier_metadata,
        quality_metadata,
        worker_version="2.0.0-rc.2",
        embedder_version="2.0.0-rc.2",
        classifier_policy_version="2.0.0-rc.2",
        catalog_version="2.0.0",
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request-v2")

    assert response.items[0].status is ItemStatus.UNKNOWN
    assert response.items[0].reason_codes == ["CLASSIFIER_CATALOG_CONFLICT"]
    assert response.items[0].confidence == pytest.approx(0.42)
    assert response.items[0].top3[0].confidence == pytest.approx(0.91)
    assert response.catalog_version == "2.0.0"


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
    assert response.items[0].reason_codes == ["SEGMENT_RECAPTURE_REQUIRED"]
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


def test_approved_classifier_is_not_rejected_by_unknown_top3_safety(
    classifier_metadata, quality_metadata
):
    classifier_metadata.approval_threshold = 0.9
    classifier_metadata.neighbor_mask_inference = NeighborMaskClassifierMetadata(
        views=[NeighborMaskClassifierView(name="mask", distance_bias=0.0, weight=1.0)],
        top3_safety_threshold=-2.96,
    )
    classifier = FakeClassifier([])
    classifier.logits = ClassificationResult(
        logits=np.asarray([[4.0, 0.3, 0.2]], dtype=np.float32),
        ranking_logits=np.asarray([[1.0, 0.5, 0.25]], dtype=np.float32),
        approval_scores=np.asarray([1.0], dtype=np.float32),
        top3_safety_scores=np.asarray([-3.0], dtype=np.float32),
    )
    pipeline = DecisionPipeline(
        FakeDetector(DetectionResult([Detection(10, 10, 40, 40, 0.95)])),
        classifier,
        classifier_metadata,
        quality_metadata,
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "approved-top3")

    assert response.items[0].status is ItemStatus.APPROVED


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
    assert response.items[0].reason_codes == ["SEGMENT_RECAPTURE_REQUIRED"]


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


def test_capacity_saturation_recaptures(classifier_metadata, quality_metadata, caplog):
    caplog.set_level("INFO", logger="bixolon_scanner.pipeline.decision")
    classifier = FakeClassifier([])
    result = DetectionResult([Detection(10, 10, 40, 40, 0.9)], capacity_saturated=True)
    pipeline = DecisionPipeline(
        FakeDetector(result), classifier, classifier_metadata, quality_metadata
    )
    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request04")
    assert response.status is Status.RECAPTURE
    assert response.reason_codes == ["IMAGE_RECAPTURE_REQUIRED"]
    assert caplog.records[-1].diagnostic_reason_codes == ["DETECTOR_CAPACITY_EXCEEDED"]
    assert classifier.calls == 0


def test_legacy_border_policy_recaptures_before_classifier(classifier_metadata, quality_metadata):
    classifier = FakeClassifier([[10.0, 0.0, 0.0]])
    result = DetectionResult([Detection(0, 10, 40, 40, 0.95)])
    pipeline = DecisionPipeline(
        FakeDetector(result), classifier, classifier_metadata, quality_metadata
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "request05")

    assert response.status is Status.RECAPTURE
    assert response.reason_codes == ["IMAGE_RECAPTURE_REQUIRED"]
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
    assert response.items[0].reason_codes == ["SEGMENT_RECAPTURE_REQUIRED"]
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
    assert response.items[0].reason_codes == ["SEGMENT_RECAPTURE_REQUIRED"]


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
    assert response.reason_codes == ["IMAGE_RECAPTURE_REQUIRED"]
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
    assert response.reason_codes == ["IMAGE_RECAPTURE_REQUIRED"]
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


def test_object_presence_verifier_accepts_multiple_detections(
    classifier_metadata, quality_metadata
):
    classifier = FakeClassifier([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    result = DetectionResult(
        [
            Detection(10, 10, 40, 40, 0.95),
            Detection(50, 50, 80, 80, 0.94),
        ],
        verified_count=1,
        count_confidence=0.99,
    )
    metadata = CountVerifierMetadata(
        filename="presence_verifier.onnx",
        version="1.0.0",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        count_labels=[0, 1],
        comparison_mode="object_presence",
        confidence_threshold=0.5,
    )
    pipeline = DecisionPipeline(
        FakeDetector(result),
        classifier,
        classifier_metadata,
        quality_metadata,
        metadata,
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "presence")

    assert response.status is Status.APPROVED
    assert classifier.calls == 1


def test_object_presence_verifier_recaptures_false_detection(classifier_metadata, quality_metadata):
    classifier = FakeClassifier([])
    result = DetectionResult(
        [Detection(10, 10, 40, 40, 0.95)],
        verified_count=0,
        count_confidence=0.99,
    )
    metadata = CountVerifierMetadata(
        filename="presence_verifier.onnx",
        version="1.0.0",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        count_labels=[0, 1],
        comparison_mode="object_presence",
        confidence_threshold=0.5,
    )
    pipeline = DecisionPipeline(
        FakeDetector(result),
        classifier,
        classifier_metadata,
        quality_metadata,
        metadata,
    )

    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "presence-empty")

    assert response.status is Status.RECAPTURE
    assert response.reason_codes == ["IMAGE_RECAPTURE_REQUIRED"]
    assert classifier.calls == 0


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
    assert response.reason_codes == ["IMAGE_RECAPTURE_REQUIRED"]
    assert classifier.calls == 0
