import json

import numpy as np
import pytest

from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.operations.verifier_candidate import optimize_verifier


@pytest.mark.parametrize("quantize", [False, True])
def test_verifier_conversion_preserves_source_policy_and_checksums(tmp_path, quantize):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    source, destination = tmp_path / "source", tmp_path / "candidate"
    source.mkdir()
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "example",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])],
        [onnx.helper.make_tensor("weight", onnx.TensorProto.FLOAT, [3, 2], [1, 2, 3, 4, 5, 6])],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, source / "verifier.onnx")
    metadata = {
        "classifier_verification": {"independent_embedder": {"filename": "verifier.onnx"}},
        "policy": {"approval_threshold": 0.7},
        "checksums": {"verifier.onnx": sha256_file(source / "verifier.onnx")},
    }
    (source / "metadata.json").write_text(json.dumps(metadata))
    original = (source / "verifier.onnx").read_bytes()
    report = optimize_verifier(source, destination, quantize=quantize)
    assert (source / "verifier.onnx").read_bytes() == original
    result = json.loads((destination / "metadata.json").read_text())
    assert result["policy"] == metadata["policy"]
    assert result["checksums"]["verifier.onnx"] == report["candidate_sha256"]
    tensor = np.ones((1, 3), dtype=np.float32)
    session = ort.InferenceSession(
        str(destination / "verifier.onnx"), providers=["CPUExecutionProvider"]
    )
    np.testing.assert_allclose(session.run(None, {"input": tensor})[0], [[9, 12]], atol=0.1)
    with pytest.raises(ValueError, match="must be new"):
        optimize_verifier(source, destination)
