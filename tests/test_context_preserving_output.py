import numpy as np
import pytest

from bixolon_scanner.contracts import ItemStatus, Status
from bixolon_scanner.contracts.model_package import QualityMetadata
from bixolon_scanner.pipeline import DecisionPipeline
from bixolon_scanner.pipeline.ports import Detection, DetectionResult
from bixolon_scanner.runtime.preprocessing import classifier_neighbor_ownership_mask


class Detector:
    version = "1.0.0"

    def __init__(self, detections):
        self.detections = detections

    def detect(self, image):
        return DetectionResult(self.detections)


class ContextClassifier:
    version = "1.0.0"

    def __init__(self):
        self.calls = []

    def classify_selected(self, image, detections, indices):
        self.calls.append((list(detections), indices.tolist()))
        return np.tile(np.array([10.0, 0.0, 0.0], dtype=np.float32), (len(indices), 1))


def quality(**kwargs):
    return QualityMetadata(
        min_object_area_ratio=0.001,
        border_margin_ratio=0,
        detector_segment_recapture_score_threshold=0.25,
        detector_output_score_threshold=0.04,
        **kwargs,
    )


def test_keeps_weak_neighbors_as_context_and_retains_real_low_score_roi(classifier_metadata):
    boxes = [
        Detection(10, 10, 80, 80, 0.99),
        Detection(20, 20, 55, 50, 0.03),
        Detection(70, 70, 95, 95, 0.05),
    ]
    classifier = ContextClassifier()
    pipeline = DecisionPipeline(Detector(boxes), classifier, classifier_metadata, quality())
    response = pipeline.scan(np.full((100, 100, 3), 128, dtype=np.uint8), "context-test")
    context, indices = classifier.calls[0]
    assert context == boxes
    assert indices == [0, 2]
    mask_args = dict(
        image_width=100,
        image_height=100,
        output_size=32,
        margin_ratio=0,
        distance_bias=-0.1,
        shared_scale=False,
    )
    expected = classifier_neighbor_ownership_mask(boxes, 0, **mask_args)
    np.testing.assert_array_equal(
        classifier_neighbor_ownership_mask(context, 0, **mask_args), expected
    )
    assert not np.array_equal(
        classifier_neighbor_ownership_mask([boxes[0], boxes[2]], 0, **mask_args), expected
    )
    assert [s.segmentation_id for s in response.segmentations] == [
        "segmentation_001",
        "segmentation_002",
    ]
    assert [s.status for s in response.segmentations] == [
        ItemStatus.APPROVED,
        ItemStatus.SEGMENT_RECAPTURE,
    ]
    assert response.segmentations[1].bbox.x == 70


def test_all_weak_output_proposals_exit_before_classifier(classifier_metadata):
    classifier = ContextClassifier()
    response = DecisionPipeline(
        Detector([Detection(10, 10, 40, 40, 0.03)]), classifier, classifier_metadata, quality()
    ).scan(np.zeros((100, 100, 3), dtype=np.uint8), "empty-output")
    assert response.status is Status.IMAGE_RECAPTURE
    assert response.segmentations == []
    assert response.classifier_version is None
    assert classifier.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"detector_output_score_threshold": 0.04},
        {
            "detector_output_score_threshold": 0.3,
            "detector_segment_recapture_score_threshold": 0.25,
        },
        {
            "detector_output_score_threshold": 0.04,
            "detector_segment_recapture_score_threshold": 0.25,
            "skip_low_score_classification": True,
        },
    ],
)
def test_rejects_output_filters_that_can_hide_approved_score_ranges(payload):
    with pytest.raises(ValueError):
        QualityMetadata(**payload)


def test_legacy_metadata_defaults_to_no_output_filter():
    assert QualityMetadata().detector_output_score_threshold is None
