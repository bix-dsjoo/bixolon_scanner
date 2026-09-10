import json
from pathlib import Path

import pytest

from bixolon_scanner.operations.n100_test_kit import (
    child,
    digest,
    overlay_runtime,
    summarize,
    timing,
    verify_kit,
    worker_environment,
)


def test_manifest_rejects_escape_and_changed_file(tmp_path):
    with pytest.raises(ValueError):
        child(tmp_path, "../outside")
    image = tmp_path / "image.jpg"
    image.write_bytes(b"original")
    payload = {
        "files": [{"path": image.name, "sha256": digest(image)}],
        "inputs": [{"path": image.name, "image_sha256": digest(image)}] * 132,
    }
    (tmp_path / "KIT-MANIFEST.json").write_text(json.dumps(payload), encoding="utf-8")
    assert verify_kit(tmp_path) == payload
    image.write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify_kit(tmp_path)


def test_percentiles_and_errors_are_not_excluded():
    rows = [
        {
            "elapsed_ms": 10,
            "response": {"status": "SEGMENTATION", "segmentations": [{"status": "UNKNOWN"}]},
        },
        {"elapsed_ms": 100, "response": {"status": "ERROR", "segmentations": []}},
    ]
    result = summarize(rows)
    assert result["latency"]["count"] == 2
    assert result["latency"]["p95_ms"] == pytest.approx(95.5)
    assert result["error"]["count"] == 1
    assert result["item_status_counts"] == {"UNKNOWN": 1}
    assert timing([])["p95_ms"] is None


def test_profile_does_not_inherit_host_override(monkeypatch):
    monkeypatch.setenv("BIXOLON_CUDA_DLL_DIR", "unrelated")
    profile = {
        "provider": "cpu",
        "embedder_provider": "openvino_gpu",
        "embedder_fallback_provider": "same",
        "provider_execution_cpu_fallback": True,
        "verifier_provider": "cpu",
        "cpu_detector_intra_op_threads": 2,
        "cpu_embedder_intra_op_threads": 4,
    }
    env = worker_environment(Path("worker"), profile, 8123)
    assert "BIXOLON_CUDA_DLL_DIR" not in env
    assert env["BIXOLON_EMBEDDER_PROVIDER"] == "openvino_gpu"
    assert env["BIXOLON_PROVIDER_EXECUTION_CPU_FALLBACK"] == "true"
    assert env["BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS"] == "2"


def test_matrix_trial_passes_its_own_catalog_to_worker(monkeypatch, tmp_path):
    from bixolon_scanner.operations import n100_test_kit as kit

    package = tmp_path / "Trial"
    (package / "worker").mkdir(parents=True)
    profile = {
        "provider": "cpu",
        "embedder_provider": "same",
        "embedder_fallback_provider": "same",
        "provider_execution_cpu_fallback": True,
        "verifier_provider": "cpu",
        "cpu_detector_intra_op_threads": 4,
        "cpu_embedder_intra_op_threads": 4,
    }
    (package / "provenance.json").write_text(json.dumps({"execution_profile": profile}))
    catalog = tmp_path / "student-catalog"
    catalog.mkdir()
    captured = {}
    shared_runtime = tmp_path / "shared-runtime"
    shared_runtime.mkdir()
    (shared_runtime / "model.onnx").write_bytes(b"shared model")
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "metadata.json").write_bytes(b"candidate metadata")

    def capture(*args, **kwargs):
        captured.update(kwargs["env"])
        raise RuntimeError("captured before process start")

    monkeypatch.setattr(kit.subprocess, "Popen", capture)
    with pytest.raises(RuntimeError, match="captured"):
        kit.run_worker(
            tmp_path,
            "0.2.1",
            "Trial",
            [],
            tmp_path / "result",
            smoke=True,
            experiment={
                "id": "student",
                "catalog": "student-catalog",
                "runtime": "shared-runtime",
                "runtime_overlay": "overlay",
            },
        )
    assert Path(captured["BIXOLON_CATALOG_DIR"]) == catalog
    materialized = Path(captured["BIXOLON_PACKAGE_DIR"])
    assert (materialized / "model.onnx").read_bytes() == b"shared model"
    assert (materialized / "metadata.json").read_bytes() == b"candidate metadata"


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows local cache")
def test_shared_benchmark_dependencies_are_copied_and_cannot_escape(monkeypatch, tmp_path):
    from bixolon_scanner.operations.n100_test_kit import stage_benchmark

    root = tmp_path / "usb"
    root.mkdir()
    (root / "KIT-MANIFEST.json").write_bytes(b"test identity")
    rows = []
    for name in ["R4/worker.exe", "R3/models/detector.onnx", "unrelated/result.txt"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        rows.append({"path": name, "sha256": digest(path)})
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    manifest = {"benchmark": "R4", "benchmark_dependencies": ["R3/models"], "files": rows}
    cache = stage_benchmark(root, manifest)
    assert (cache / "R3/models/detector.onnx").read_bytes() == b"R3/models/detector.onnx"
    assert not (cache / "unrelated/result.txt").exists()
    manifest["benchmark_dependencies"] = ["../escape"]
    with pytest.raises(ValueError, match="manifest path"):
        stage_benchmark(root, manifest)


def test_overlay_preserves_source_and_replaces_links_without_mutating_old_package(tmp_path):
    base, overlay, output = (tmp_path / name for name in ("base", "overlay", "out"))
    base.mkdir()
    overlay.mkdir()
    (base / "model.onnx").write_bytes(b"weights")
    (base / "metadata.json").write_bytes(b"base")
    (overlay / "metadata.json").write_bytes(b"candidate")
    overlay_runtime(base, overlay, output)
    assert (output / "model.onnx").read_bytes() == b"weights"
    assert (output / "metadata.json").read_bytes() == b"candidate"
    (overlay / "metadata.json").unlink()
    (overlay / "metadata.json").write_bytes(b"next candidate")
    overlay_runtime(base, overlay, output)
    assert (output / "metadata.json").read_bytes() == b"next candidate"
    assert (base / "metadata.json").read_bytes() == b"base"


def test_matrix_preserves_failed_trial_and_continues_with_fresh_worker(monkeypatch, tmp_path):
    from bixolon_scanner.operations import n100_test_kit as kit

    (tmp_path / "image.jpg").write_bytes(b"input")
    experiments = [
        {"id": name, "description": name, "profile": {"parallel_verification": True}}
        for name in ("first", "second")
    ]
    manifest = {
        "version": "0.2.1",
        "benchmark": "Trial",
        "inputs": [{"path": "image.jpg", "image_id": 1}],
        "experiments": experiments,
    }
    monkeypatch.setattr(kit, "verify_kit", lambda _: manifest)
    monkeypatch.setattr(kit, "stage_benchmark", lambda root, _: root)
    monkeypatch.setattr(kit, "ensure_idle", lambda: None)
    monkeypatch.setattr(kit, "acquire_measurement_lock", lambda: None)
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("provider failed")
        return {"passed": True}

    monkeypatch.setattr(kit, "run_worker", run)
    assert kit.main(["--kit-root", str(tmp_path), "--matrix"]) == 1
    assert [call["experiment"]["id"] for call in calls] == ["first", "second"]
    assert all(call["repetitions"] == 1 for call in calls)
    result = next((tmp_path / "results").glob("*/measurement-summary.json"))
    summary = json.loads(result.read_text())["summary"]
    assert summary[0]["failed"] == "provider failed"
    assert summary[1]["summary"]["passed"]
    assert list((tmp_path / "results").glob("*.zip"))


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows named mutex")
def test_duplicate_measurement_rejected_and_lock_released():
    from bixolon_scanner.operations.n100_test_kit import (
        acquire_measurement_lock,
        release_measurement_lock,
    )

    first = acquire_measurement_lock()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            acquire_measurement_lock()
    finally:
        release_measurement_lock(first)
    second = acquire_measurement_lock()
    release_measurement_lock(second)
