from types import SimpleNamespace

import numpy as np
import pytest

from bixolon_scanner import inference
from bixolon_scanner.errors import ProviderInitializationError
from bixolon_scanner.package import (
    ClassifierView,
    NeighborMaskClassifierMetadata,
    NeighborMaskClassifierView,
    StagedClassifierMetadata,
)
from bixolon_scanner.pipeline.ports import Detection


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


def test_neighbor_mask_classifier_warmup_includes_every_view(classifier_metadata):
    shapes = []

    class Runner:
        def run(self, output_names, input_name, tensor):
            del output_names, input_name
            shapes.append(tensor.shape)
            return [None]

    classifier_metadata.warmup_batch_sizes = [1, 3]
    classifier_metadata.neighbor_mask_inference = NeighborMaskClassifierMetadata(
        views=[
            NeighborMaskClassifierView(name="strict", distance_bias=0.0, weight=0.75),
            NeighborMaskClassifierView(name="guarded", distance_bias=0.25, weight=0.25),
        ],
        top3_safety_threshold=-2.96,
    )
    adapter = inference.OnnxClassifier.__new__(inference.OnnxClassifier)
    adapter.metadata = classifier_metadata
    adapter.runner = Runner()

    adapter.warmup()

    assert [shape[0] for shape in shapes] == [2, 6]


def test_neighbor_mask_classifier_returns_weighted_decision_and_safety_scores(
    classifier_metadata,
):
    classifier_metadata.neighbor_mask_inference = NeighborMaskClassifierMetadata(
        views=[
            NeighborMaskClassifierView(name="strict", distance_bias=0.0, weight=0.75),
            NeighborMaskClassifierView(name="guarded", distance_bias=0.25, weight=0.25),
        ],
        top3_safety_threshold=-2.96,
    )
    calls = []

    class Runner:
        def run(self, output_names, input_name, tensor):
            del output_names, input_name
            calls.append(tensor.shape)
            return [
                np.asarray(
                    [
                        [4.0, 1.0, 0.0],
                        [1.0, 3.0, 0.0],
                        [2.0, 0.0, 1.0],
                        [0.0, 2.0, 1.0],
                    ],
                    dtype=np.float32,
                )
            ]

    adapter = inference.OnnxClassifier.__new__(inference.OnnxClassifier)
    adapter.metadata = classifier_metadata
    adapter.runner = Runner()
    detections = [
        Detection(0.0, 0.0, 40.0, 40.0, 0.9),
        Detection(30.0, 0.0, 70.0, 40.0, 0.8),
    ]

    result = adapter._neighbor_mask_classify(
        np.zeros((2, 3, 224, 224), dtype=np.float32),
        detections,
        image_width=100,
        image_height=100,
    )

    assert calls == [(4, 3, 224, 224)]
    np.testing.assert_allclose(result.logits[0], [3.5, 0.75, 0.25])
    assert result.approval_scores.shape == (2,)
    assert result.top3_safety_scores.shape == (2,)
    assert np.all(result.top3_safety_scores <= 0.0)


def test_neighbor_mask_classifier_returns_l2_normalized_logit_margin(
    classifier_metadata,
):
    classifier_metadata.neighbor_mask_inference = NeighborMaskClassifierMetadata(
        views=[NeighborMaskClassifierView(name="mask", distance_bias=0.0, weight=1.0)],
        approval_metric="l2_normalized_logit_margin",
        top3_safety_threshold=-2.96,
    )

    class Runner:
        def run(self, output_names, input_name, tensor):
            del output_names, input_name, tensor
            return [np.asarray([[3.0, 2.0, -1.0]], dtype=np.float32)]

    adapter = inference.OnnxClassifier.__new__(inference.OnnxClassifier)
    adapter.metadata = classifier_metadata
    adapter.runner = Runner()
    result = adapter._neighbor_mask_classify(
        np.zeros((1, 3, 224, 224), dtype=np.float32),
        [Detection(0.0, 0.0, 40.0, 40.0, 0.9)],
        image_width=100,
        image_height=100,
    )

    assert result.approval_scores[0] == pytest.approx(1.0 / np.sqrt(14.0))


def test_neighbor_mask_classifier_uses_global_rank_tie_break(classifier_metadata):
    classifier_metadata.neighbor_mask_inference = NeighborMaskClassifierMetadata(
        views=[
            NeighborMaskClassifierView(name="left", distance_bias=0.0, weight=0.5),
            NeighborMaskClassifierView(name="right", distance_bias=0.25, weight=0.5),
        ],
        top3_safety_threshold=-2.96,
        ranking_tie_break_bias_span=0.002,
    )

    class Runner:
        def run(self, output_names, input_name, tensor):
            del output_names, input_name, tensor
            return [
                np.asarray(
                    [[4.0, 3.0, 2.0], [4.0, 2.0, 3.0]],
                    dtype=np.float32,
                )
            ]

    adapter = inference.OnnxClassifier.__new__(inference.OnnxClassifier)
    adapter.metadata = classifier_metadata
    adapter.runner = Runner()

    result = adapter._neighbor_mask_classify(
        np.zeros((1, 3, 224, 224), dtype=np.float32),
        [Detection(0.0, 0.0, 40.0, 40.0, 0.9)],
        image_width=100,
        image_height=100,
    )

    assert result.ranking_logits[0, 1] - result.ranking_logits[0, 2] == pytest.approx(0.001)


def test_staged_classifier_batches_only_ambiguous_and_unknown_views(classifier_metadata):
    classifier_metadata.approval_threshold = 0.8
    classifier_metadata.staged_inference = StagedClassifierMetadata(
        center_crop_scale=0.855,
        views=[
            ClassifierView(name="vflip", affine=((1, 0, 0), (0, -1, 0))),
            ClassifierView(name="rot15", affine=((0.96, 0.26, 0), (-0.26, 0.96, 0))),
            ClassifierView(name="rot30", affine=((0.87, 0.5, 0), (-0.5, 0.87, 0))),
        ],
        first_view="vflip",
        early_approval_threshold=0.99,
        final_views=["vflip", "rot15"],
        top3_views=["vflip", "rot15", "rot30"],
    )
    calls = []

    class Runner:
        def run_inputs(self, output_names, inputs):
            del output_names
            batch_size = len(inputs["pixel_values"])
            calls.append(batch_size)
            if len(calls) == 1:
                return [np.asarray([[20.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)]
            if len(calls) == 2:
                return [np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)]
            return [np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32)]

    adapter = inference.OnnxClassifier.__new__(inference.OnnxClassifier)
    adapter.metadata = classifier_metadata
    adapter.runner = Runner()

    result = adapter._staged_classify(np.zeros((2, 3, 224, 224), dtype=np.float32))

    assert calls == [2, 1, 1]
    np.testing.assert_array_equal(result.logits[0], [20.0, 0.0, 0.0])
    np.testing.assert_allclose(result.logits[1], [0.5, 0.5, 0.0])
    assert result.ranking_logits[1].argmax() == 2


def test_staged_classifier_supports_top3_vote_ranking(classifier_metadata):
    classifier_metadata.approval_threshold = 0.9
    classifier_metadata.staged_inference = StagedClassifierMetadata(
        center_crop_scale=0.855,
        views=[
            ClassifierView(name="base", affine=((1, 0, 0), (0, 1, 0))),
            ClassifierView(name="rot15", affine=((0.96, 0.26, 0), (-0.26, 0.96, 0))),
        ],
        first_view="base",
        early_approval_threshold=0.99,
        final_views=["base"],
        top3_views=["base", "rot15"],
        ranking_aggregation="top3_vote",
    )
    calls = []

    class Runner:
        def run_inputs(self, output_names, inputs):
            del output_names
            calls.append(len(inputs["pixel_values"]))
            if len(calls) == 1:
                return [np.asarray([[0.2, 0.1, 0.0, -0.1]], dtype=np.float32)]
            return [
                np.asarray(
                    [[-0.1, 0.0, 0.1, 0.2]],
                    dtype=np.float32,
                )
            ]

    adapter = inference.OnnxClassifier.__new__(inference.OnnxClassifier)
    adapter.metadata = classifier_metadata
    adapter.runner = Runner()

    result = adapter._staged_classify(np.zeros((1, 3, 224, 224), dtype=np.float32))

    assert calls == [1, 1]
    assert set(np.argsort(-result.ranking_logits[0])[:3]) == {0, 1, 2}


def test_staged_classifier_batches_all_final_views_when_early_exit_is_disabled(
    classifier_metadata,
):
    classifier_metadata.approval_threshold = 0.9
    classifier_metadata.staged_inference = StagedClassifierMetadata(
        center_crop_scale=0.855,
        views=[
            ClassifierView(name="base", affine=((1, 0, 0), (0, 1, 0))),
            ClassifierView(name="rot15", affine=((0.96, 0.26, 0), (-0.26, 0.96, 0))),
            ClassifierView(name="rot30", affine=((0.87, 0.5, 0), (-0.5, 0.87, 0))),
        ],
        first_view="base",
        early_approval_threshold=1.0,
        final_views=["base", "rot15", "rot30"],
        top3_views=["base", "rot15", "rot30"],
    )
    calls = []

    class Runner:
        def run_inputs(self, output_names, inputs):
            del output_names
            calls.append(len(inputs["pixel_values"]))
            return [
                np.asarray(
                    [
                        [3.0, 0.0, 0.0],
                        [0.0, 3.0, 0.0],
                        [0.0, 0.0, 3.0],
                    ],
                    dtype=np.float32,
                )
            ]

    adapter = inference.OnnxClassifier.__new__(inference.OnnxClassifier)
    adapter.metadata = classifier_metadata
    adapter.runner = Runner()

    result = adapter._staged_classify(np.zeros((1, 3, 224, 224), dtype=np.float32))

    assert calls == [3]
    np.testing.assert_allclose(result.logits, [[1.0, 1.0, 1.0]])


def test_square_classifier_crop_preserves_object_shape_and_shifts_inside_frame():
    box = inference.classifier_crop_box(
        Detection(80.0, 10.0, 100.0, 50.0, 0.9),
        100,
        80,
        margin_ratio=0.0,
        crop_mode="square_context",
    )

    assert box == (60, 10, 100, 50)
    assert box[2] - box[0] == box[3] - box[1]


def test_prepare_rgb_returns_contiguous_values_with_original_arithmetic():
    image = np.arange(17 * 23 * 3, dtype=np.uint8).reshape(17, 23, 3)
    resized = inference.Image.fromarray(image, mode="RGB").resize(
        (19, 13), inference.Image.Resampling.BILINEAR
    )
    expected = np.asarray(resized, dtype=np.float32) / 255.0
    expected = (expected - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)) / np.asarray(
        (0.229, 0.224, 0.225), dtype=np.float32
    )
    expected = np.transpose(expected, (2, 0, 1))

    actual = inference.prepare_rgb(
        image,
        (13, 19),
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous


def test_legacy_classifier_crop_remains_rectangular():
    assert inference.classifier_crop_box(
        Detection(10.0, 20.0, 30.0, 60.0, 0.9),
        100,
        100,
        margin_ratio=0.0,
        crop_mode="box_resize",
    ) == (10, 20, 30, 60)


def test_containment_nms_suppresses_nested_duplicate_with_low_iou():
    outer = Detection(0.0, 0.0, 20.0, 20.0, 0.9)
    inner = Detection(5.0, 5.0, 15.0, 15.0, 0.8)

    assert inference.nms([outer, inner], 0.7) == [outer, inner]
    assert inference.nms([outer, inner], 0.7, 0.7) == [outer]


def test_containment_nms_can_require_same_detector_class():
    outer = Detection(0.0, 0.0, 20.0, 20.0, 0.9, class_id=1)
    same_class_inner = Detection(5.0, 5.0, 15.0, 15.0, 0.8, class_id=1)
    different_class_inner = Detection(5.0, 5.0, 15.0, 15.0, 0.8, class_id=2)

    assert inference.nms([outer, same_class_inner], 0.7, 0.7, True) == [outer]
    assert inference.nms([outer, different_class_inner], 0.7, 0.7, True) == [
        outer,
        different_class_inner,
    ]


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


def test_detector_filters_configured_extreme_aspect_ratio_candidates():
    metadata = SimpleNamespace(
        input_size=(32, 32),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        resize_reducing_gap=None,
        logits_output="logits",
        boxes_output="boxes",
        input_name="pixel_values",
        score_threshold=0.7,
        uncertainty_score_threshold=None,
        nms_iou_threshold=0.7,
        max_object_aspect_ratio=5.0,
        max_queries=300,
    )

    class Runner:
        def run(self, output_names, input_name, tensor):
            del output_names, input_name, tensor
            logits = np.asarray([[[3.0], [3.0]]], dtype=np.float32)
            boxes = np.asarray([[[0.25, 0.25, 0.2, 0.2], [0.75, 0.5, 0.02, 0.4]]], dtype=np.float32)
            return [logits, boxes]

    adapter = inference.OnnxDetector.__new__(inference.OnnxDetector)
    adapter.metadata = metadata
    adapter.runner = Runner()

    result = adapter.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(result.detections) == 1
    assert result.detections[0].x1 == pytest.approx(15.0)


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
