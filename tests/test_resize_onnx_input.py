from __future__ import annotations

import pytest

from bixolon_scanner.training.resize_onnx_input import resize_static_image_input


def test_resize_static_image_input_preserves_graph_weights(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    source = tmp_path / "source.onnx"
    output = tmp_path / "resized.onnx"
    input_info = onnx.helper.make_tensor_value_info(
        "pixel_values", onnx.TensorProto.FLOAT, ["batch", 3, 224, 224]
    )
    output_info = onnx.helper.make_tensor_value_info(
        "embeddings", onnx.TensorProto.FLOAT, ["batch", 3]
    )
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node(
                    "ReduceMean",
                    ["pixel_values"],
                    ["embeddings"],
                    axes=[2, 3],
                    keepdims=0,
                )
            ],
            "resize-input-test",
            [input_info],
            [output_info],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )
    onnx.save(model, source)

    report = resize_static_image_input(source, output, image_size=192)
    converted = onnx.load(output)
    dimensions = converted.graph.input[0].type.tensor_type.shape.dim

    assert report["source_image_size"] == 224
    assert report["image_size"] == 192
    assert report["weights_modified"] is False
    assert [dimension.dim_value for dimension in dimensions[1:]] == [3, 192, 192]
    assert [node.op_type for node in converted.graph.node] == ["ReduceMean"]


@pytest.mark.parametrize("image_size", [0, 31, 48, 225])
def test_resize_static_image_input_rejects_unsupported_size(tmp_path, image_size) -> None:
    source = tmp_path / "missing.onnx"
    with pytest.raises(ValueError):
        resize_static_image_input(source, tmp_path / "output.onnx", image_size=image_size)
