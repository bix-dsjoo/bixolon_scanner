import json

import pytest

from bixolon_scanner.operations.static_batch_candidate import specialize_runtime


def test_specialization_preserves_source_weights_and_operators(tmp_path):
    onnx = pytest.importorskip("onnx")
    source, target = tmp_path / "source", tmp_path / "candidate"
    source.mkdir()
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["input", "bias"], ["output"])],
        "example",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, ["batch", 3])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, ["batch", 3])],
        [onnx.helper.make_tensor("bias", onnx.TensorProto.FLOAT, [3], [1, 2, 3])],
    )
    onnx.save(onnx.helper.make_model(graph), source / "model.onnx")
    original = (source / "model.onnx").read_bytes()
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "embedder": {"filename": "model.onnx", "input_name": "input"},
                "checksums": {},
            }
        )
    )
    report = specialize_runtime(source, target, primary_batch_size=4, primary_batch_variants=(1, 2))
    assert (source / "model.onnx").read_bytes() == original
    model = onnx.load(target / "model.onnx")
    assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 4
    assert model.graph.output[0].type.tensor_type.shape.dim[0].dim_value == 4
    assert report["models"][0]["weights_identical"]
    assert report["models"][0]["operators_identical"]
    for size in (1, 2):
        variant = onnx.load(target / f"model.batch{size}.onnx")
        assert variant.graph.input[0].type.tensor_type.shape.dim[0].dim_value == size
        assert variant.graph.initializer == model.graph.initializer
        assert variant.graph.node == model.graph.node
    assert len(report["models"]) == 3
    with pytest.raises(ValueError, match="must be new"):
        specialize_runtime(source, target)
