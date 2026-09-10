from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from bixolon_scanner.pipeline.ports import Detection
from bixolon_scanner.runtime import catalog, detector_v2, onnx_session
from bixolon_scanner.runtime.catalog import OnnxEmbedder
from bixolon_scanner.runtime.detector_v2 import FixedEnsembleOnnxDetector
from bixolon_scanner.worker.settings import WorkerSettings


class _FakeSessionOptions:
    def __init__(self) -> None:
        self.config_entries: dict[str, str] = {}

    def add_session_config_entry(self, name: str, value: str) -> None:
        self.config_entries[name] = value


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
        GraphOptimizationLevel=SimpleNamespace(
            ORT_ENABLE_ALL="all",
            ORT_DISABLE_ALL="disabled",
        ),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        InferenceSession=create_session,
        get_available_providers=lambda: [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ],
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
    assert options.config_entries["session.intra_op.allow_spinning"] == "0"
    assert options.config_entries["session.inter_op.allow_spinning"] == "0"
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
    assert "session.intra_op.allow_spinning" not in options.config_entries
    assert "session.inter_op.allow_spinning" not in options.config_entries
    assert runner.cuda is True
    assert runner.accelerated is True


def test_directml_runner_applies_required_session_contract(monkeypatch, tmp_path: Path) -> None:
    fake_ort, captured = _fake_ort()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    runner = onnx_session.OrtRunner(tmp_path / "model.onnx", "directml")

    session = captured["session"]
    options = session.options
    assert options.execution_mode == "sequential"
    assert options.enable_mem_pattern is False
    assert session.providers == [("DmlExecutionProvider", {"device_id": "0"})]
    assert runner.cuda is False
    assert runner.accelerated is True


def test_explicit_directml_never_falls_back(monkeypatch) -> None:
    fake_ort, _ = _fake_ort()
    fake_ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    with pytest.raises(onnx_session.ProviderInitializationError):
        onnx_session.select_provider("directml")


def test_openvino_runner_applies_cpu_device_and_thread_contract(
    monkeypatch, tmp_path: Path
) -> None:
    fake_ort, captured = _fake_ort()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    runner = onnx_session.OrtRunner(
        tmp_path / "model.onnx",
        "openvino",
        cpu_intra_op_threads=4,
    )

    session = captured["session"]
    assert session.providers == [
        (
            "OpenVINOExecutionProvider",
            {
                "device_type": "CPU",
                "load_config": (
                    '{"CPU":{"PERFORMANCE_HINT":"LATENCY","NUM_STREAMS":"1",'
                    '"INFERENCE_PRECISION_HINT":"f32","INFERENCE_NUM_THREADS":"4"}}'
                ),
            },
        )
    ]
    assert session.options.graph_optimization_level == "disabled"
    assert runner.accelerated is True


def test_explicit_openvino_never_falls_back(monkeypatch) -> None:
    fake_ort, _ = _fake_ort()
    fake_ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    with pytest.raises(onnx_session.ProviderInitializationError):
        onnx_session.select_provider("openvino")


def test_openvino_gpu_runner_selects_gpu_and_disables_cpu_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    fake_ort, captured = _fake_ort()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    runner = onnx_session.OrtRunner(
        tmp_path / "model.onnx",
        "openvino_gpu",
    )

    session = captured["session"]
    assert session.providers == [
        (
            "OpenVINOExecutionProvider",
            {
                "device_type": "GPU",
                "load_config": '{"GPU":{"PERFORMANCE_HINT":"LATENCY","NUM_STREAMS":"1","INFERENCE_PRECISION_HINT":"f32"}}',
            },
        )
    ]
    assert session.options.config_entries == {"session.disable_cpu_ep_fallback": "1"}
    assert runner.accelerated is True


@pytest.mark.parametrize(
    ("provider", "device"),
    [("openvino", "CPU"), ("openvino_gpu", "GPU")],
)
def test_openvino_runner_enables_persistent_speed_cache(
    monkeypatch, tmp_path: Path, provider: str, device: str
) -> None:
    fake_ort, captured = _fake_ort()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    cache_root = tmp_path / "compiled-model-cache"
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model-one")

    onnx_session.OrtRunner(
        model_path,
        provider,
        openvino_cache_dir=cache_root,
    )

    session = captured["session"]
    provider_options = session.providers[0][1]
    load_config = json.loads(provider_options["load_config"])
    model_cache = Path(load_config[device]["CACHE_DIR"])
    assert model_cache.parent == (cache_root / device.lower()).resolve()
    assert model_cache.name.startswith("model-")
    assert load_config[device]["CACHE_MODE"] == "OPTIMIZE_SPEED"
    assert model_cache.is_dir()

    other_model_path = tmp_path / "other-model.onnx"
    other_model_path.write_bytes(b"model-two")
    onnx_session.OrtRunner(
        other_model_path,
        provider,
        openvino_cache_dir=cache_root,
    )
    other_config = json.loads(captured["session"].providers[0][1]["load_config"])
    assert Path(other_config[device]["CACHE_DIR"]) != model_cache


def test_explicit_openvino_gpu_never_falls_back(monkeypatch) -> None:
    fake_ort, _ = _fake_ort()
    fake_ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    with pytest.raises(onnx_session.ProviderInitializationError):
        onnx_session.select_provider("openvino_gpu")


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


def test_worker_settings_accept_explicit_directml_embedder() -> None:
    settings = WorkerSettings(provider="cpu", embedder_provider="directml")

    assert settings.provider == "cpu"
    assert settings.embedder_provider == "directml"


def test_worker_settings_accept_explicit_openvino_provider() -> None:
    settings = WorkerSettings(provider="openvino")

    assert settings.provider == "openvino"


def test_worker_settings_accept_openvino_cpu_gpu_split() -> None:
    settings = WorkerSettings(
        provider="openvino",
        embedder_provider="openvino_gpu",
        embedder_fallback_provider="same",
    )

    assert settings.provider == "openvino"
    assert settings.embedder_provider == "openvino_gpu"
    assert settings.embedder_fallback_provider == "same"


def test_worker_settings_accept_openvino_cache_directory(tmp_path: Path) -> None:
    settings = WorkerSettings(openvino_cache_dir=tmp_path / "openvino-cache")

    assert settings.openvino_cache_dir == tmp_path / "openvino-cache"


class _WarmupRunner:
    def __init__(self, *, cuda: bool):
        self.cuda = cuda
        self.accelerated = cuda
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


def test_embedder_directml_warmup_keeps_all_metadata_batches() -> None:
    embedder = object.__new__(OnnxEmbedder)
    embedder.metadata = SimpleNamespace(
        input_size=(16, 16),
        warmup_batch_sizes=[1, 2, 4, 8],
        output_name="output",
        input_name="input",
    )
    embedder.runner = _WarmupRunner(cuda=False)
    embedder.runner.accelerated = True

    embedder.warmup()

    assert embedder.runner.batch_sizes == [1, 2, 4, 8]


def test_embedder_reuses_float32_input_buffer() -> None:
    class CaptureRunner:
        def __init__(self) -> None:
            self.tensor = None

        def run(self, _outputs, _input_name, tensor):
            self.tensor = tensor
            return [np.zeros((len(tensor), 3), dtype=np.float32)]

    embedder = object.__new__(OnnxEmbedder)
    embedder.metadata = SimpleNamespace(
        output_name="output",
        input_name="input",
        embedding_dimension=3,
    )
    embedder.runner = CaptureRunner()
    batch = np.zeros((2, 3, 4, 4), dtype=np.float32)

    result = embedder._run_raw_tensors(batch)

    assert embedder.runner.tensor is batch
    assert result.shape == (2, 3)


def test_embedder_fixed_batch_chunks_and_discards_padding() -> None:
    class CaptureRunner:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def run(self, _outputs, _input_name, tensor):
            self.batch_sizes.append(len(tensor))
            values = tensor[:, 0, 0, 0][:, None]
            return [np.repeat(values, 2, axis=1)]

    embedder = object.__new__(OnnxEmbedder)
    embedder.metadata = SimpleNamespace(
        output_name="output",
        input_name="input",
        embedding_dimension=2,
        fixed_batch_size=2,
    )
    embedder.runner = CaptureRunner()
    batch = np.arange(3, dtype=np.float32).reshape(3, 1, 1, 1)

    result = embedder._run_raw_tensors(batch)

    assert embedder.runner.batch_sizes == [2, 2]
    assert result.tolist() == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]


def test_embedder_fixed_batch_warmup_uses_only_declared_shape() -> None:
    embedder = object.__new__(OnnxEmbedder)
    embedder.metadata = SimpleNamespace(
        input_size=(16, 16),
        warmup_batch_sizes=[1, 2, 4, 8],
        fixed_batch_size=3,
        output_name="output",
        input_name="input",
    )
    embedder.runner = _WarmupRunner(cuda=True)

    embedder.warmup()

    assert embedder.runner.batch_sizes == [3]


def test_embedder_horizontal_flip_tta_averages_two_generic_views() -> None:
    class CaptureRunner:
        def __init__(self) -> None:
            self.tensor = None

        def run(self, _outputs, _input_name, tensor):
            self.tensor = tensor
            features = tensor[:, 0, 0, :2]
            return [np.asarray(features, dtype=np.float32)]

    embedder = object.__new__(OnnxEmbedder)
    embedder.metadata = SimpleNamespace(
        output_name="output",
        input_name="input",
        embedding_dimension=2,
        horizontal_flip_tta=True,
    )
    embedder.runner = CaptureRunner()
    batch = np.asarray([[[[1.0, 2.0, 3.0]]]], dtype=np.float32)

    result = embedder._run_view_averaged_tensors(batch)

    assert embedder.runner.tensor.shape == (2, 1, 1, 3)
    assert result.tolist() == [[2.0, 2.0]]


def test_embedder_rotation_180_tta_averages_two_generic_views() -> None:
    class CaptureRunner:
        def __init__(self) -> None:
            self.tensor = None

        def run(self, _outputs, _input_name, tensor):
            self.tensor = tensor
            features = tensor[:, 0, 0, :2]
            return [np.asarray(features, dtype=np.float32)]

    embedder = object.__new__(OnnxEmbedder)
    embedder.metadata = SimpleNamespace(
        output_name="output",
        input_name="input",
        embedding_dimension=2,
        rotation_180_tta=True,
    )
    embedder.runner = CaptureRunner()
    batch = np.asarray([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)

    result = embedder._run_view_averaged_tensors(batch)

    assert embedder.runner.tensor.shape == (2, 1, 2, 2)
    assert result.tolist() == [[2.5, 2.5]]


def test_embedder_prepared_tensor_contract_validates_shape() -> None:
    embedder = object.__new__(OnnxEmbedder)
    embedder.metadata = SimpleNamespace(input_size=(8, 8))

    with pytest.raises(ValueError, match="prepared embedder tensors"):
        embedder.embed_prepared_tensors_raw(np.zeros((1, 3, 7, 8), dtype=np.float32))


def test_embedder_prepared_tensor_contract_runs_contiguous_float32() -> None:
    embedder = object.__new__(OnnxEmbedder)
    embedder.metadata = SimpleNamespace(input_size=(2, 3))
    captured = None

    def run(batch):
        nonlocal captured
        captured = batch
        return np.ones((len(batch), 4), dtype=np.float32)

    embedder._run_view_averaged_tensors = run
    source = np.zeros((2, 3, 2, 3), dtype=np.float64)[:, :, :, ::-1]

    result = embedder.embed_prepared_tensors_raw(source)

    assert captured.dtype == np.float32
    assert captured.flags.c_contiguous
    assert result.shape == (2, 4)


def test_embedder_prepares_only_selected_rois_with_full_neighbor_context(monkeypatch) -> None:
    embedder = object.__new__(OnnxEmbedder)
    embedder.metadata = SimpleNamespace(
        input_size=(8, 8),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        crop_margin_ratio=0.0,
        crop_mode="box_resize",
        resize_reducing_gap=None,
        neighbor_mask=True,
        neighbor_distance_bias=-0.1,
        neighbor_shared_scale=False,
    )
    image = np.arange(24 * 24 * 3, dtype=np.uint8).reshape(24, 24, 3)
    detections = [
        Detection(0, 0, 12, 12, 0.9),
        Detection(6, 0, 18, 12, 0.8),
        Detection(12, 12, 24, 24, 0.7),
    ]
    full = embedder.prepare_detection_tensors(image, detections)
    prepare_calls = 0
    original_prepare = catalog.prepare_rgb

    def counting_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(catalog, "prepare_rgb", counting_prepare)

    selected = embedder.prepare_selected_detection_tensors(
        image,
        detections,
        np.asarray([2, 0], dtype=np.int64),
    )

    assert prepare_calls == 2
    np.testing.assert_array_equal(selected, full[[2, 0]])


def _ensemble_package(tmp_path: Path):
    members = [SimpleNamespace(filename=f"detector-{index}.onnx") for index in range(4)]
    ensemble = SimpleNamespace(
        members=members,
        parallel_execution=False,
        cuda_graph_execution=True,
        selective_cascade=None,
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


def test_detector_prepares_each_member_at_its_declared_input_size(monkeypatch) -> None:
    class ShapeRunner:
        def __init__(self) -> None:
            self.shapes: list[tuple[int, ...]] = []

        def run(self, _outputs, _input_name, tensor):
            self.shapes.append(tensor.shape)
            return [
                np.zeros((1, 1, 1), dtype=np.float32),
                np.zeros((1, 1, 4), dtype=np.float32),
            ]

    detector = object.__new__(FixedEnsembleOnnxDetector)
    detector.metadata = SimpleNamespace(
        input_size=(640, 640),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        resize_reducing_gap=1.0,
        logits_output="logits",
        boxes_output="boxes",
        input_name="input",
    )
    detector.ensemble = SimpleNamespace(
        members=[
            SimpleNamespace(filename="fast.onnx", input_size=(480, 480)),
            SimpleNamespace(filename="fallback.onnx", input_size=None),
        ],
        selective_cascade=None,
    )
    detector.runners = [ShapeRunner(), ShapeRunner()]
    detector.executor = None
    detector._select_outputs = lambda outputs, **_kwargs: outputs
    monkeypatch.setattr(
        detector_v2,
        "prepare_rgb",
        lambda _image, size, *_args, **_kwargs: np.zeros((3, *size), dtype=np.float32),
    )

    detector._predict(np.zeros((8, 8, 3), dtype=np.uint8), width=8, height=8)

    assert detector.runners[0].shapes == [(1, 3, 480, 480)]
    assert detector.runners[1].shapes == [(1, 3, 640, 640)]


def test_detector_keeps_fast_primary_out_of_full_fallback(monkeypatch) -> None:
    class MarkerRunner:
        def __init__(self, marker: float) -> None:
            self.marker = marker
            self.calls = 0

        def run(self, _outputs, _input_name, _tensor):
            self.calls += 1
            return [
                np.full((1, 1, 1), self.marker, dtype=np.float32),
                np.zeros((1, 1, 4), dtype=np.float32),
            ]

    detector = object.__new__(FixedEnsembleOnnxDetector)
    detector.metadata = SimpleNamespace(
        input_size=(640, 640),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        resize_reducing_gap=1.0,
        logits_output="logits",
        boxes_output="boxes",
        input_name="input",
    )
    detector.ensemble = SimpleNamespace(
        members=[
            SimpleNamespace(
                filename="fast.onnx",
                input_size=(480, 480),
                ensemble_fallback=False,
            ),
            SimpleNamespace(
                filename="fold.onnx",
                input_size=None,
                ensemble_fallback=True,
            ),
            SimpleNamespace(
                filename="production-640.onnx",
                input_size=None,
                ensemble_fallback=True,
            ),
        ],
        selective_cascade=None,
    )
    detector.runners = [MarkerRunner(1.0), MarkerRunner(2.0), MarkerRunner(3.0)]
    detector.executor = None
    captured: list[tuple[list[float], list[int]]] = []

    def capture(outputs, **kwargs):
        markers = [float(output[0][0, 0]) for output in outputs]
        captured.append((markers, kwargs["member_indices"]))
        return outputs

    detector._select_outputs = capture
    monkeypatch.setattr(
        detector_v2,
        "prepare_rgb",
        lambda _image, size, *_args, **_kwargs: np.zeros((3, *size), dtype=np.float32),
    )

    detector._predict(
        np.zeros((8, 8, 3), dtype=np.uint8),
        width=8,
        height=8,
        replicated_member_filename="fast.onnx",
    )
    detector._predict(np.zeros((8, 8, 3), dtype=np.uint8), width=8, height=8)

    assert captured == [([1.0, 1.0], [1, 2]), ([2.0, 3.0], [1, 2])]
    assert [runner.calls for runner in detector.runners] == [1, 1, 1]


def test_detector_fast_primary_respects_source_dimension_floor() -> None:
    detector = object.__new__(FixedEnsembleOnnxDetector)
    detector.ensemble = SimpleNamespace(
        class_verified_selector=SimpleNamespace(
            low_resolution_maximum_dimension=5000,
            low_resolution_primary_minimum_dimension=1000,
            low_resolution_small_image_primary_member_filename="fallback.onnx",
            low_resolution_small_image_score_threshold=0.58,
            low_resolution_score_threshold=0.45,
            low_resolution_primary_member_filename="fast.onnx",
            low_resolution_maximum_box_area_ratio=0.35,
        )
    )
    calls = []

    def capture(_image, **kwargs):
        calls.append(kwargs)
        return kwargs

    detector._predict = capture

    detector.predict_candidates(np.zeros((640, 640, 3), dtype=np.uint8))
    detector.predict_candidates(np.zeros((1200, 1200, 3), dtype=np.uint8))

    assert calls[0]["replicated_member_filename"] == "fallback.onnx"
    assert calls[0]["score_threshold"] == 0.58
    assert calls[1]["replicated_member_filename"] == "fast.onnx"
    assert calls[1]["score_threshold"] == 0.45


def test_openvino_cpu_detector_uses_requested_worker_limit(monkeypatch, tmp_path: Path) -> None:
    class FakeRunner(_WarmupRunner):
        def __init__(self, _path, _provider, _cuda_dir, **_kwargs):
            super().__init__(cuda=False)

    monkeypatch.setattr(detector_v2, "OrtRunner", FakeRunner)
    package = _ensemble_package(tmp_path)
    package.metadata.detector.ensemble.parallel_execution = True
    detector = FixedEnsembleOnnxDetector(
        package,
        "openvino",
        cpu_detector_workers=2,
        cpu_intra_op_threads=1,
    )
    try:
        assert detector.executor is not None
        assert detector.executor._max_workers == 2
    finally:
        detector.close()


@pytest.mark.parametrize(
    ("selected_count", "expected_secondary_calls"),
    ((5, 0), (6, 1)),
)
def test_detector_cascade_runs_secondary_only_for_triggered_count(
    monkeypatch,
    selected_count: int,
    expected_secondary_calls: int,
) -> None:
    class CountingRunner:
        def __init__(self, marker: float) -> None:
            self.marker = marker
            self.calls = 0

        def run(self, _outputs, _input_name, _tensor):
            self.calls += 1
            return [
                np.asarray([[[self.marker]]], dtype=np.float32),
                np.zeros((1, 1, 4), dtype=np.float32),
            ]

    detector = object.__new__(FixedEnsembleOnnxDetector)
    detector.metadata = SimpleNamespace(
        input_size=(8, 8),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        resize_reducing_gap=1.0,
        logits_output="logits",
        boxes_output="boxes",
        input_name="input",
    )
    detector.ensemble = SimpleNamespace(
        members=[
            SimpleNamespace(filename="detector-fold1.onnx"),
            SimpleNamespace(filename="detector-production.onnx"),
        ],
        selective_cascade=SimpleNamespace(
            primary_member_filename="detector-production.onnx",
            secondary_trigger_selected_counts=[6],
            secondary_trigger_minimum_score_maximum=None,
            secondary_trigger_minimum_score_minimum=None,
            secondary_trigger_on_uncertain=False,
        ),
    )
    secondary = CountingRunner(1.0)
    primary = CountingRunner(2.0)
    detector.runners = [secondary, primary]

    def select_outputs(outputs, **_kwargs):
        repeated_primary = float(outputs[0][0][0, 0]) == float(outputs[1][0][0, 0]) == 2.0
        count = selected_count if repeated_primary else 3
        return ({"scores": [0.9] * count}, 2, False, False)

    detector._select_outputs = select_outputs
    monkeypatch.setattr(
        detector_v2,
        "prepare_rgb",
        lambda *_args, **_kwargs: np.zeros((3, 8, 8), dtype=np.float32),
    )

    detector._predict(np.zeros((8, 8, 3), dtype=np.uint8), width=8, height=8)

    assert primary.calls == 1
    assert secondary.calls == expected_secondary_calls
