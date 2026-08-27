from __future__ import annotations

import pytest

from bixolon_scanner.training.fixed_batch_onnx import convert_to_fixed_batch


def test_convert_to_fixed_batch_preserves_graph_and_fixes_public_shapes(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    source = tmp_path / "dynamic.onnx"
    output = tmp_path / "fixed.onnx"
    input_info = onnx.helper.make_tensor_value_info(
        "pixel_values", onnx.TensorProto.FLOAT, ["batch", 3, 8, 8]
    )
    output_info = onnx.helper.make_tensor_value_info(
        "logits", onnx.TensorProto.FLOAT, ["batch", 3, 8, 8]
    )
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["pixel_values"], ["logits"])],
            "fixed-batch-test",
            [input_info],
            [output_info],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 18)],
    )
    onnx.save(model, source)

    report = convert_to_fixed_batch(source, output, fixed_batch_size=1)
    converted = onnx.load(output)

    assert report["fixed_batch_size"] == 1
    assert report["weights_modified"] is False
    assert report["source_onnx_sha256"] != report["onnx_sha256"]
    assert [value.type.tensor_type.shape.dim[0].dim_value for value in converted.graph.input] == [1]
    assert [value.type.tensor_type.shape.dim[0].dim_value for value in converted.graph.output] == [
        1
    ]


def test_convert_to_fixed_batch_refuses_overwrite(tmp_path) -> None:
    source = tmp_path / "source.onnx"
    output = tmp_path / "existing.onnx"
    source.write_bytes(b"source")
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        convert_to_fixed_batch(source, output)
