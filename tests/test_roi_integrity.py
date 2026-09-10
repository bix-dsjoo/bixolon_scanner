from types import SimpleNamespace

import numpy as np
import pytest
from test_pipeline import FakeClassifier, FakeDetector, ResolutionFallbackFakeClassifier

from bixolon_scanner.contracts import ItemStatus, Status
from bixolon_scanner.contracts.errors import ModelExecutionError
from bixolon_scanner.pipeline import DecisionPipeline
from bixolon_scanner.pipeline.ports import ClassificationResult, Detection, DetectionResult
from bixolon_scanner.runtime.catalog import OnnxEmbedder
from bixolon_scanner.training.roi_integrity import grouped_detections


@pytest.mark.parametrize(
    "score,status",
    [
        (0.7999, ItemStatus.APPROVED),
        (0.8, ItemStatus.SEGMENT_RECAPTURE),
        (1.0, ItemStatus.SEGMENT_RECAPTURE),
    ],
)
def test_high_confidence_merged_roi_is_recaptured_without_rejecting_neighbor(
    classifier_metadata, quality_metadata, score, status
):
    logits = np.array([[30.0, 0.0, 0.0], [0.0, 30.0, 0.0]], dtype=np.float32)
    classifier = FakeClassifier(
        ClassificationResult(logits, logits, multi_object_probabilities=np.array([score, 0.0]))
    )
    pipeline = DecisionPipeline(
        FakeDetector(
            DetectionResult([Detection(10, 10, 40, 40, 1.0), Detection(50, 10, 80, 40, 1.0)])
        ),
        classifier,
        classifier_metadata,
        quality_metadata.model_copy(update={"multi_object_recapture_threshold": 0.8}),
    )
    response = pipeline.scan(np.zeros((100, 100, 3), dtype=np.uint8), "roi-integrity")
    assert response.status is Status.SEGMENTATION
    assert [s.status for s in response.segmentations] == [status, ItemStatus.APPROVED]
    if status is ItemStatus.SEGMENT_RECAPTURE:
        assert response.segmentations[0].prediction is None
        assert response.segmentations[0].top3 == []
        assert response.segmentations[0].reason_codes == ["SEGMENT_RECAPTURE_REQUIRED"]
    assert classifier.calls == 1


def test_integrity_recapture_cannot_be_promoted_by_detail(classifier_metadata, quality_metadata):
    logits = np.array([[30.0, 0.0, 0.0]], dtype=np.float32)
    classifier = ResolutionFallbackFakeClassifier(
        ClassificationResult(logits, logits, multi_object_probabilities=np.array([0.99])),
        [[0.0, 30.0, 0.0]],
        fallback_on_unknown=True,
        fallback_on_unsafe=True,
    )
    pipeline = DecisionPipeline(
        FakeDetector(DetectionResult([Detection(10, 10, 90, 90, 1.0)])),
        classifier,
        classifier_metadata,
        quality_metadata.model_copy(update={"multi_object_recapture_threshold": 0.8}),
    )
    result = pipeline.scan(np.zeros((100, 100, 3), dtype=np.uint8), "integrity-detail")
    assert result.segmentations[0].status is ItemStatus.SEGMENT_RECAPTURE
    assert classifier.fallback_calls == 0


@pytest.mark.parametrize("scores", [None, np.array([np.nan]), np.array([1.01]), np.array([])])
def test_missing_or_corrupt_integrity_output_is_model_error(
    classifier_metadata, quality_metadata, scores
):
    logits = np.array([[30.0, 0.0, 0.0]], dtype=np.float32)
    pipeline = DecisionPipeline(
        FakeDetector(DetectionResult([Detection(10, 10, 90, 90, 1.0)])),
        FakeClassifier(ClassificationResult(logits, logits, multi_object_probabilities=scores)),
        classifier_metadata,
        quality_metadata.model_copy(update={"multi_object_recapture_threshold": 0.8}),
    )
    with pytest.raises(ModelExecutionError):
        pipeline.scan(np.zeros((100, 100, 3), dtype=np.uint8), "integrity-corrupt")


def test_integrity_shares_one_inference_and_discards_fixed_batch_padding():
    calls = []

    class Runner:
        def run(self, outputs, name, inputs):
            calls.append((outputs, inputs.shape))
            return [np.array([[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]]), np.array([0.9, 0.2, 1.0])]

    embedder = object.__new__(OnnxEmbedder)
    embedder.metadata = SimpleNamespace(
        multi_object_output_name="multiplicity",
        output_name="embeddings",
        input_name="images",
        input_size=(2, 2),
        embedding_dimension=2,
        fixed_batch_size=3,
        horizontal_flip_tta=False,
        rotation_180_tta=False,
    )
    embedder.runner = Runner()
    embeddings, probabilities = embedder.embed_prepared_tensors_with_integrity(
        np.zeros((2, 3, 2, 2), dtype=np.float32)
    )
    assert len(calls) == 1
    assert calls[0][0] == ["embeddings", "multiplicity"]
    np.testing.assert_array_equal(embeddings, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(probabilities, [0.9, 0.2])


def test_merged_training_group_retains_only_other_object_mask_owners():
    boxes = [[10, 10, 40, 40], [30, 20, 60, 50], [70, 70, 90, 90]]
    detections = grouped_detections(boxes, [0, 1])
    assert detections == [Detection(10, 10, 60, 50, 1.0), Detection(70, 70, 90, 90, 1.0)]
    with pytest.raises(ValueError):
        grouped_detections(boxes, [0, 0])
