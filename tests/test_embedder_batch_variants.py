from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from bixolon_scanner.contracts.runtime_package_v2 import EmbedderMetadata
from bixolon_scanner.runtime.catalog import OnnxEmbedder


def metadata(**changes):
    return EmbedderMetadata(
        embedder_id="test",
        version="0.2.1",
        embedding_dimension=2,
        input_size=(2, 2),
        fixed_batch_size=4,
        batch_variants=[
            {"filename": "batch1.onnx", "batch_size": 1},
            {"filename": "batch2.onnx", "batch_size": 2},
        ],
    ).model_copy(update=changes)


def embedder_with_runners(**changes):
    calls = []

    class Runner:
        def __init__(self, size):
            self.size = size
            self.closed = False

        def run(self, outputs, name, values):
            assert len(values) == self.size
            calls.append(self.size)
            result = np.repeat(values[:, 0, 0, 0, None], 2, axis=1)
            return [result, values[:, 0, 0, 0] / 20] if len(outputs) == 2 else [result]

        def close(self):
            self.closed = True

    instance = object.__new__(OnnxEmbedder)
    instance.metadata = metadata(**changes)
    instance.runner = Runner(4)
    instance.batch_runners = {1: Runner(1), 2: Runner(2)}
    return instance, calls


@pytest.mark.parametrize("integrity", [False, True])
def test_remaining_rows_select_exact_graph_and_preserve_every_result(integrity):
    embedder, calls = embedder_with_runners(
        multi_object_output_name="integrity" if integrity else None
    )
    values = np.broadcast_to(
        np.arange(11, dtype=np.float32)[:, None, None, None], (11, 3, 2, 2)
    ).copy()
    if integrity:
        result, scores = embedder.embed_prepared_tensors_with_integrity(values)
        np.testing.assert_allclose(scores, np.arange(11) / 20)
    else:
        result = embedder.embed_prepared_tensors_raw(values)
    assert calls == [4, 4, 2, 1]
    np.testing.assert_array_equal(result, np.repeat(np.arange(11)[:, None], 2, axis=1))
    # A single rotated verification ROI uses batch one without dummy forward rows.
    embedder.embed_prepared_tensors_raw(values[:1, :, ::-1, ::-1])
    assert calls[-1] == 1
    assert embedder.embed_prepared_tensors_raw(values[:0]).shape == (0, 2)


def test_variants_warmup_static_shapes_with_tta_and_close_all():
    embedder, calls = embedder_with_runners(horizontal_flip_tta=True)
    embedder.warmup()
    assert calls == [4, 1, 2]
    embedder.close()
    assert all(runner.closed for runner in [embedder.runner, *embedder.batch_runners.values()])


@pytest.mark.parametrize(
    "variants",
    [
        [{"filename": "same.onnx", "batch_size": 1}, {"filename": "same.onnx", "batch_size": 2}],
        [{"filename": "a.onnx", "batch_size": 1}, {"filename": "b.onnx", "batch_size": 1}],
        [{"filename": "a.onnx", "batch_size": 4}],
        [{"filename": "../escape.onnx", "batch_size": 1}],
        [{"filename": "a.onnx", "batch_size": 2}],
    ],
)
def test_invalid_variant_configuration_is_rejected(variants):
    payload = metadata().model_dump()
    payload["batch_variants"] = variants
    with pytest.raises(ValidationError):
        EmbedderMetadata.model_validate(payload)


def test_variants_require_static_primary():
    payload = metadata().model_dump()
    payload["fixed_batch_size"] = None
    with pytest.raises(ValidationError):
        EmbedderMetadata.model_validate(payload)


def test_failed_variant_initialization_closes_previously_created_sessions(monkeypatch, tmp_path):
    import bixolon_scanner.runtime.catalog as module

    runners = []

    def create(path, *args, **kwargs):
        if len(runners) == 2:
            raise RuntimeError("session failure")
        runner = SimpleNamespace(close=lambda: closed.append(path))
        runners.append(runner)
        return runner

    closed = []
    for name in ("batch1.onnx", "batch2.onnx"):
        (tmp_path / name).write_bytes(b"model")
    monkeypatch.setattr(module, "OrtRunner", create)
    package = SimpleNamespace(
        root=tmp_path,
        embedder_path=tmp_path / "embedder.onnx",
        metadata=SimpleNamespace(embedder=metadata()),
    )
    with pytest.raises(RuntimeError, match="session failure"):
        OnnxEmbedder(package, "cpu")
    assert len(closed) == 2
