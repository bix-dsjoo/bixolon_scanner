from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from bixolon_scanner.runtime import detector_v2, onnx_session
from bixolon_scanner.runtime.catalog import OnnxEmbedder
from bixolon_scanner.runtime.detector_v2 import FixedEnsembleOnnxDetector
from bixolon_scanner.worker.settings import WorkerSettings


class _FakeSessionOptions:
    pass


class _FakeSession:
    def __init__(self, _path, *, sess_options, providers):
        self.options = sess_options
        self.providers = providers

    def get_providers(self):
        first = self.providers[0]
        return [first[0] if isinstance(first, tuple) else first]


def _fake_ort():
    captured: dict[str, object] = {}

    def create_session(path, *, sess_options, providers):
        session = _FakeSession(path, sess_options=sess_options, providers=providers)
        captured["session"] = session
        return session

    module = SimpleNamespace(
        SessionOptions=_FakeSessionOptions,
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        InferenceSession=create_session,
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    return module, captured


def test_cpu_runner_applies_explicit_thread_contract(monkeypatch, tmp_path: Path) -> None:
    fake_ort, captured = _fake_ort()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    runner = onnx_session.OrtRunner(
        tmp_path / "model.onnx",
        "cpu",
        cpu_intra_op_threads=4,
    )

    options = captured["session"].options
    assert options.graph_optimization_level == "all"
    assert options.execution_mode == "sequential"
    assert options.inter_op_num_threads == 1
    assert options.intra_op_num_threads == 4
    assert runner.cuda is False


def test_cuda_runner_does_not_apply_cpu_thread_contract(monkeypatch, tmp_path: Path) -> None:
    fake_ort, captured = _fake_ort()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    runner = onnx_session.OrtRunner(
        tmp_path / "model.onnx",
        "cuda",
        cpu_intra_op_threads=4,
    )

    options = captured["session"].options
    assert not hasattr(options, "intra_op_num_threads")
    assert runner.cuda is True


def test_worker_settings_validate_cpu_execution_limits() -> None:
    settings = WorkerSettings(
        cpu_detector_workers=4,
        cpu_detector_intra_op_threads=1,
        cpu_embedder_intra_op_threads=4,
    )
    assert settings.cpu_detector_workers == 4
    assert settings.cpu_detector_intra_op_threads == 1
    assert settings.cpu_embedder_intra_op_threads == 4

    with pytest.raises(ValidationError):
        WorkerSettings(cpu_detector_workers=5)
    with pytest.raises(ValidationError):
        WorkerSettings(cpu_detector_intra_op_threads=-1)


class _WarmupRunner:
    def __init__(self, *, cuda: bool):
        self.cuda = cuda
        self.batch_sizes: list[int] = []

    def run(self, _outputs, _input_name, tensor):
        self.batch_sizes.append(len(tensor))
        return [np.zeros((len(tensor), 1), dtype=np.float32)]


def test_embedder_cpu_warmup_uses_only_batch_one() -> None:
    embedder = object.__new__(OnnxEmbedder)
    embedder.metadata = SimpleNamespace(
        input_size=(16, 16),
        warmup_batch_sizes=[1, 2, 4, 8],
        output_name="output",
        input_name="input",
    )
    embedder.runner = _WarmupRunner(cuda=False)

    embedder.warmup()

    assert embedder.runner.batch_sizes == [1]


def test_embedder_cuda_warmup_keeps_all_metadata_batches() -> None:
    embedder = object.__new__(OnnxEmbedder)
    embedder.metadata = SimpleNamespace(
        input_size=(16, 16),
        warmup_batch_sizes=[1, 2, 4, 8],
        output_name="output",
        input_name="input",
    )
    embedder.runner = _WarmupRunner(cuda=True)

    embedder.warmup()

    assert embedder.runner.batch_sizes == [1, 2, 4, 8]


def _ensemble_package(tmp_path: Path):
    members = [SimpleNamespace(filename=f"detector-{index}.onnx") for index in range(4)]
    ensemble = SimpleNamespace(
        members=members,
        parallel_execution=False,
        cuda_graph_execution=True,
    )
    detector = SimpleNamespace(
        ensemble=ensemble,
        logits_output="logits",
        boxes_output="boxes",
        max_queries=10,
        input_size=(16, 16),
        input_name="input",
        version="0.0.2",
    )
    return SimpleNamespace(
        root=tmp_path,
        metadata=SimpleNamespace(detector=detector, detector_class_count=3),
    )


def test_cpu_detector_uses_requested_workers_and_threads(monkeypatch, tmp_path: Path) -> None:
    received_threads: list[int] = []

    class FakeRunner(_WarmupRunner):
        def __init__(self, _path, _provider, _cuda_dir, **kwargs):
            super().__init__(cuda=False)
            received_threads.append(kwargs["cpu_intra_op_threads"])

        def run(self, _outputs, _input_name, tensor):
            self.batch_sizes.append(len(tensor))
            return [
                np.zeros((1, 10, 3), dtype=np.float32),
                np.zeros((1, 10, 4), dtype=np.float32),
            ]

    monkeypatch.setattr(detector_v2, "OrtRunner", FakeRunner)
    detector = FixedEnsembleOnnxDetector(
        _ensemble_package(tmp_path),
        "cpu",
        cpu_detector_workers=4,
        cpu_intra_op_threads=1,
    )
    try:
        assert detector.executor is not None
        assert detector.executor._max_workers == 4
        assert received_threads == [1, 1, 1, 1]
        detector.warmup()
        assert [runner.batch_sizes for runner in detector.runners] == [[1], [1], [1], [1]]
    finally:
        detector.close()


def test_cuda_detector_keeps_metadata_parallel_setting(monkeypatch, tmp_path: Path) -> None:
    class FakeRunner(_WarmupRunner):
        def __init__(self, _path, _provider, _cuda_dir, **_kwargs):
            super().__init__(cuda=True)

    monkeypatch.setattr(detector_v2, "OrtRunner", FakeRunner)
    detector = FixedEnsembleOnnxDetector(
        _ensemble_package(tmp_path),
        "cuda",
        cpu_detector_workers=4,
        cpu_intra_op_threads=1,
    )
    assert detector.executor is None
