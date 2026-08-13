from types import SimpleNamespace

import numpy as np
import pytest

from bixolon_scanner import inference
from bixolon_scanner.errors import ProviderInitializationError


class _Adapter:
    def __init__(self, path, metadata, provider, cuda_dll_dir=None):
        del path, metadata, cuda_dll_dir
        if provider == "cuda":
            raise ProviderInitializationError
        self.provider = provider

    def warmup(self):
        return None


def _package():
    return SimpleNamespace(
        detector_path="detector.onnx",
        classifier_path="classifier.onnx",
        metadata=SimpleNamespace(detector=object(), classifier=object()),
    )


def test_auto_provider_falls_back_both_sessions_to_cpu(monkeypatch):
    monkeypatch.setattr(inference, "select_provider", lambda mode: "cuda")
    monkeypatch.setattr(inference, "OnnxDetector", _Adapter)
    monkeypatch.setattr(inference, "OnnxClassifier", _Adapter)
    detector, classifier, provider = inference.build_onnx_adapters(_package(), "auto")
    assert provider == "cpu"
    assert detector.provider == classifier.provider == "cpu"


def test_forced_cuda_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(inference, "select_provider", lambda mode: "cuda")
    monkeypatch.setattr(inference, "OnnxDetector", _Adapter)
    monkeypatch.setattr(inference, "OnnxClassifier", _Adapter)
    with pytest.raises(ProviderInitializationError):
        inference.build_onnx_adapters(_package(), "cuda")


def test_classifier_warms_configured_dynamic_batch_sizes(classifier_metadata):
    shapes = []

    class Runner:
        def run(self, output_names, input_name, tensor):
            del output_names, input_name
            shapes.append(tensor.shape)
            return [None]

    classifier_metadata.warmup_batch_sizes = [1, 3, 7]
    adapter = inference.OnnxClassifier.__new__(inference.OnnxClassifier)
    adapter.metadata = classifier_metadata
    adapter.runner = Runner()
    adapter.warmup()
    assert [shape[0] for shape in shapes] == [1, 3, 7]


def test_count_verifier_returns_label_and_confidence():
    metadata = SimpleNamespace(
        input_size=(32, 32),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        resize_reducing_gap=None,
        logits_output="logits",
        input_name="pixel_values",
        temperature=1.0,
        count_labels=[3, 4, 5],
    )

    class Runner:
        def run(self, output_names, input_name, tensor):
            del output_names, input_name
            assert tensor.shape == (1, 3, 32, 32)
            return [np.asarray([[0.0, 4.0, 1.0]], dtype=np.float32)]

    adapter = inference.OnnxCountVerifier.__new__(inference.OnnxCountVerifier)
    adapter.metadata = metadata
    adapter.runner = Runner()

    count, confidence = adapter.verify(np.zeros((64, 64, 3), dtype=np.uint8))

    assert count == 4
    assert confidence > 0.9


def test_detector_reports_separate_low_score_candidate():
    metadata = SimpleNamespace(
        input_size=(32, 32),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        resize_reducing_gap=None,
        logits_output="logits",
        boxes_output="boxes",
        input_name="pixel_values",
        score_threshold=0.7,
        uncertainty_score_threshold=0.4,
        uncertainty_min_area_ratio=0.03,
        uncertainty_match_iou_threshold=0.5,
        nms_iou_threshold=0.7,
        max_queries=300,
    )

    class Runner:
        def run(self, output_names, input_name, tensor):
            del output_names, input_name, tensor
            logits = np.asarray([[[3.0], [0.0]]], dtype=np.float32)
            boxes = np.asarray(
                [[[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]]],
                dtype=np.float32,
            )
            return [logits, boxes]

    adapter = inference.OnnxDetector.__new__(inference.OnnxDetector)
    adapter.metadata = metadata
    adapter.runner = Runner()

    result = adapter.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(result.detections) == 1
    assert result.uncertain_candidate_count == 1
    assert result.uncertain_candidate_scores == pytest.approx((0.5,))


@pytest.mark.parametrize(
    ("candidate_logit", "candidate_box"),
    [
        (-1.0, [0.75, 0.75, 0.2, 0.2]),
        (0.0, [0.75, 0.75, 0.1, 0.1]),
        (0.0, [0.29, 0.25, 0.2, 0.2]),
    ],
    ids=("below-score", "below-area", "overlaps-accepted"),
)
def test_detector_ignores_unqualified_uncertainty_candidates(candidate_logit, candidate_box):
    metadata = SimpleNamespace(
        input_size=(32, 32),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        resize_reducing_gap=None,
        logits_output="logits",
        boxes_output="boxes",
        input_name="pixel_values",
        score_threshold=0.7,
        uncertainty_score_threshold=0.4,
        uncertainty_min_area_ratio=0.03,
        uncertainty_match_iou_threshold=0.5,
        nms_iou_threshold=0.7,
        max_queries=300,
    )

    class Runner:
        def run(self, output_names, input_name, tensor):
            del output_names, input_name, tensor
            logits = np.asarray([[[3.0], [candidate_logit]]], dtype=np.float32)
            boxes = np.asarray([[[0.25, 0.25, 0.2, 0.2], candidate_box]], dtype=np.float32)
            return [logits, boxes]

    adapter = inference.OnnxDetector.__new__(inference.OnnxDetector)
    adapter.metadata = metadata
    adapter.runner = Runner()

    result = adapter.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(result.detections) == 1
    assert result.uncertain_candidate_count == 0
    assert result.uncertain_candidate_scores == ()
