import numpy as np
import pytest

from bixolon_scanner.evaluation.structural_benchmark import percentiles
from bixolon_scanner.training.structural_student import evidence_loss


def settings():
    return dict(
        integrity_weight=1.0,
        cosine_scale=16.0,
        margin_target=0.9,
        margin_weight=4.0,
        distillation_scale=8.0,
        distillation_weight=1.0,
        relation_weight=1.0,
    )


@pytest.mark.parametrize("objective", ["supervised", "distilled", "relational"])
def test_ground_truth_can_correct_wrong_teacher(objective):
    import torch

    # Teacher says class 0; authoritative GT is class 1. Its gradient must survive KD.
    embeddings = torch.tensor([[0.9, 0.1]], requires_grad=True)
    loss = evidence_loss(
        embeddings,
        torch.tensor([[2.0, 0.0]], requires_grad=True),
        torch.tensor([1]),
        torch.tensor([0]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([0.0]),
        settings(),
        objective,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert embeddings.grad[0, 1] < 0


def test_multi_object_only_batch_does_not_invent_sku_labels():
    import torch

    scores = torch.tensor([[0.9, 0.1], [0.1, 0.9]], requires_grad=True)
    integrity = torch.zeros(2, 2, requires_grad=True)
    loss = evidence_loss(
        scores,
        integrity,
        torch.tensor([-1, -1]),
        torch.tensor([1, 1]),
        scores.detach(),
        torch.ones(2),
        settings(),
        "supervised",
    )
    loss.backward()
    assert scores.grad is None
    assert (integrity.grad[:, 1] < 0).all()


def test_invalid_objective_is_not_silently_supervised():
    with pytest.raises(ValueError, match="objective"):
        evidence_loss(None, None, None, None, None, None, settings(), "typo")


def test_timing_has_sample_count_and_worst_case():
    result = percentiles([1.0, 2.0, 10.0])
    assert result["count"] == 3
    assert result["maximum_ms"] == 10
    assert result["p95_ms"] == np.percentile([1, 2, 10], 95)


def test_detector_teacher_never_invents_extra_gt():
    from bixolon_scanner.training.structural_detector import match_teacher

    gt = np.array([[0.2, 0.2, 0.1, 0.1]])
    predicted = np.array([[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]])
    result = match_teacher(gt, predicted, np.array([0.9, 0.99]), 0.5)
    assert len(result["boxes"]) == 1
    assert result["scores"] == pytest.approx([0.9])


def test_detector_box_rotation_is_reversible_including_empty_frames():
    from bixolon_scanner.training.structural_detector import rotate_boxes

    boxes = np.array([[0.2, 0.3, 0.1, 0.15]])
    np.testing.assert_allclose(rotate_boxes(rotate_boxes(boxes, 1), 3), boxes)
    assert rotate_boxes(np.empty((0, 4)), 2).shape == (0, 4)


def test_shared_roi_pooling_preserves_every_region_and_channel():
    from bixolon_scanner.training.structural_shared_roi import pool_rois

    features = np.stack([np.ones((8, 8)), np.ones((8, 8)) * 2])[None].astype(np.float32)
    result = pool_rois(features, [[0, 0, 80, 80], [20, 20, 60, 60]], (80, 80), 3)
    assert result.shape == (2, 2, 3, 3)
    np.testing.assert_allclose(result[:, 0], 1)
    np.testing.assert_allclose(result[:, 1], 2)


def test_quantization_signature_cannot_trade_quality_for_identity():
    from bixolon_scanner.training.structural_quantization import evidence_signature

    embeddings = np.array([[0.99, 0.01], [0.99, 0.01]])
    signature = evidence_signature(embeddings, np.array([0.1, 0.9]), 0.8, 0.85, 0.8)
    assert signature[0, 0] == signature[1, 0]
    assert signature[0, -1] != signature[1, -1]


def test_detector_quantization_signature_preserves_target_identity_and_counts_extras():
    from bixolon_scanner.experiments.bread.structural_detector_quantization import (
        detection_signature,
    )

    row = {
        "width": 100,
        "height": 100,
        "annotations": [{"bbox_xywh": [10, 10, 20, 20]}, {"bbox_xywh": [60, 60, 20, 20]}],
    }
    metadata = {"score_threshold": 0.5, "max_object_aspect_ratio": 5, "nms_iou_threshold": 0.5}
    boxes = np.array([[[0.2, 0.2, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2], [0.5, 0.2, 0.1, 0.1]]])
    logits = np.array([[[10], [-10], [10]]])
    matched, extra = detection_signature((logits, boxes), row, metadata)
    assert matched == {0}
    assert extra == 1


def test_quantization_constant_aliases_keep_payload_and_dynamic_identities():
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from bixolon_scanner.experiments.bread.structural_detector_quantization import (
        materialize_constant_aliases,
    )

    weight = numpy_helper.from_array(np.array([1.5, -2], np.float32), "weight")
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["weight"], ["alias"], name="constant_alias"),
            helper.make_node("Constant", [], ["literal"], value=weight),
            helper.make_node("Identity", ["literal"], ["literal_alias"], name="literal_copy"),
            helper.make_node("Identity", ["input"], ["dynamic"], name="dynamic_alias"),
            helper.make_node("Add", ["dynamic", "alias"], ["output"]),
        ],
        "aliases",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
        [weight],
    )
    original = helper.make_model(graph)
    prepared, aliases = materialize_constant_aliases(original)
    onnx.checker.check_model(prepared)
    assert aliases == ["constant_alias", "literal_copy"]
    assert len(original.graph.node) == 5
    assert any(node.name == "dynamic_alias" for node in prepared.graph.node)
    values = {tensor.name: numpy_helper.to_array(tensor) for tensor in prepared.graph.initializer}
    np.testing.assert_array_equal(values["alias"], values["weight"])
    np.testing.assert_array_equal(values["literal_alias"], values["weight"])


def test_qat_scalar_parameters_preserve_qdq_math_with_identity_alias():
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    from bixolon_scanner.training.structural_qat import normalize_per_tensor_qparams

    nodes = [helper.make_node("Identity", ["zero"], ["zero_alias"])]
    for name in ["x", "y"]:
        nodes.extend(
            [
                helper.make_node("QuantizeLinear", [name, "scale", "zero_alias"], [name + "q"]),
                helper.make_node(
                    "DequantizeLinear", [name + "q", "scale", "zero_alias"], [name + "d"]
                ),
            ]
        )
    nodes.extend(
        [
            helper.make_node("Add", ["xd", "yd"], ["sum"]),
            helper.make_node("QuantizeLinear", ["sum", "scale", "zero_alias"], ["zq"]),
            helper.make_node("DequantizeLinear", ["zq", "scale", "zero_alias"], ["z"]),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "scalar_qparams",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [2]) for name in ["x", "y"]],
        [helper.make_tensor_value_info("z", TensorProto.FLOAT, [2])],
        [
            numpy_helper.from_array(np.array([0.1], np.float32), "scale"),
            numpy_helper.from_array(np.array([0], np.uint8), "zero"),
        ],
    )
    original = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    prepared, aliases = normalize_per_tensor_qparams(original)
    assert "zero_alias" in aliases
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    left = ort.InferenceSession(
        original.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    right = ort.InferenceSession(prepared.SerializeToString(), providers=["CPUExecutionProvider"])
    values = {"x": np.array([0.3, 1.1], np.float32), "y": np.array([0.7, 0.2], np.float32)}
    np.testing.assert_allclose(left.run(None, values)[0], right.run(None, values)[0], atol=1e-6)
