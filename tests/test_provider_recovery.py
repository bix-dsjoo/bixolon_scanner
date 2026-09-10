from types import SimpleNamespace

import pytest

from bixolon_scanner.contracts.errors import ModelExecutionError, ProviderExecutionError
from bixolon_scanner.worker import runtime_factory
from bixolon_scanner.worker.runtime_factory import WorkerRuntime
from bixolon_scanner.worker.settings import WorkerSettings


class Pipeline:
    def __init__(self, error=None):
        self.error = error
        self.closed = 0
        self.calls = 0

    def scan(self, image, request_id):
        self.calls += 1
        if self.error:
            raise self.error
        return (image, request_id)

    def close(self):
        self.closed += 1


@pytest.mark.parametrize("recovery_fails", [False, True])
def test_gpu_failure_is_error_and_only_next_request_uses_cpu(monkeypatch, recovery_fails):
    gpu, cpu = Pipeline(ProviderExecutionError()), Pipeline()
    runtime = WorkerRuntime(
        gpu,
        "cpu+openvino_gpu",
        1200,
        settings=WorkerSettings(provider_execution_cpu_fallback=True),
    )
    attempts = []

    def recover(settings):
        assert runtime.recovering
        assert settings.provider == "cpu"
        assert settings.embedder_provider == "same"
        assert not settings.provider_execution_cpu_fallback
        attempts.append(settings)
        if recovery_fails:
            raise ModelExecutionError
        return WorkerRuntime(cpu, "cpu", 1200, settings=settings)

    monkeypatch.setattr(runtime_factory, "_build_v2_runtime", recover)
    with pytest.raises(ProviderExecutionError):
        runtime.scan("image", "failed-request")
    assert len(attempts) == 1
    assert gpu.closed == 1
    assert cpu.calls == 0
    assert not runtime.recovering
    if recovery_fails:
        assert runtime.failed
        with pytest.raises(ModelExecutionError):
            runtime.scan("image", "next-request")
    else:
        assert not runtime.failed
        assert runtime.provider == "cpu"
        assert runtime.scan("image", "next-request") == ("image", "next-request")
    runtime.close()
    runtime.close()
    assert gpu.closed == 1
    assert cpu.closed == int(not recovery_fails)


@pytest.mark.parametrize(
    "provider,error,enabled",
    [
        ("cpu+openvino_gpu", ModelExecutionError(), True),
        ("cpu+openvino_gpu", ProviderExecutionError(), False),
        ("cpu", ProviderExecutionError(), True),
    ],
)
def test_cpu_generic_errors_and_optout_do_not_rebuild(monkeypatch, provider, error, enabled):
    def unexpected(_):
        pytest.fail("unexpected CPU rebuild")

    monkeypatch.setattr(runtime_factory, "_build_v2_runtime", unexpected)
    pipeline = Pipeline(error)
    runtime = WorkerRuntime(
        pipeline,
        provider,
        1200,
        settings=WorkerSettings(provider_execution_cpu_fallback=enabled),
    )
    with pytest.raises(ModelExecutionError):
        runtime.scan(None, "error")
    assert pipeline.closed == 0
    runtime.close()
    assert pipeline.closed == 1


def test_gpu_session_execution_failure_has_specific_internal_error():
    from bixolon_scanner.runtime.onnx_session import OrtRunner

    runner = object.__new__(OrtRunner)
    runner.provider = "openvino_gpu"
    runner.cuda_graph_enabled = False
    runner.cuda = False

    def failure(*args, **kwargs):
        raise RuntimeError("device removed")

    runner.session = SimpleNamespace(run=failure)
    with pytest.raises(ProviderExecutionError):
        runner.run_inputs(["out"], {"in": None})
