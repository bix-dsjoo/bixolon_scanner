from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_recorded_n100_014_result_under_current_operational_target() -> None:
    report = json.loads(
        (ROOT / "docs/diagnostics/n100-0.1.4-openvino-device-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    hybrid = report["profiles"]["openvino_cpu_detector_intel_gpu_embedder"]
    full_path = hybrid["client_total_ms"]["full_path"]

    assert full_path["mean"] <= 500
    assert full_path["p95"] > 500
    assert report["comparison"]["recommended_provider"] == "openvino"
    assert report["passes"] is False


def test_n100_gpu_cmd_runs_versioned_openvino_device_benchmark() -> None:
    command = (ROOT / "scripts/handoff/RUN-N100-GPU-TEST.cmd").read_text(encoding="utf-8")
    benchmark = (ROOT / "scripts/handoff/N100-GPU-BENCHMARK.ps1").read_text(encoding="utf-8")

    assert "N100-GPU-BENCHMARK.ps1" in command
    assert "n100-0.1.4-openvino-device-matrix.json" in command
    assert 'ExpectedVersion = "0.1.4"' in benchmark
    assert "BIXOLON_PROVIDER = [string]$Profile.DetectorProvider" in benchmark
    assert 'DetectorProvider = "openvino"' in benchmark
    assert 'EmbedderProvider = "same"' in benchmark
    assert 'EmbedderProvider = "openvino_gpu"' in benchmark
    assert 'ExpectedProvider = "openvino+openvino_gpu"' in benchmark
    assert 'Name = "openvino-cpu-only"' in benchmark
    assert 'Name = "openvino-cpu-detector-intel-gpu-embedder"' in benchmark
    assert 'BIXOLON_CPU_DETECTOR_WORKERS = "1"' in benchmark
    assert 'BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = "4"' in benchmark
    assert "semantic_mismatch_count" in benchmark
    assert "maximum_confidence_delta" in benchmark
    assert "minimum_mean_speedup_ratio" in benchmark
    assert "maximum_peak_working_set_bytes" in benchmark
    assert "MaximumWorkingSetBytes = 2147483648" in benchmark
    assert "MaximumFullPathLatencyMs = 500.0" in benchmark
    assert "mean_within_target" in benchmark
    assert "p95_within_target" in benchmark
    assert "gpu_embedder_target_met" in benchmark
    assert "hybrid_resource_safe" in benchmark
    assert "same_runtime_catalog_and_policy = $true" in benchmark
    assert 'independent_verifier_embedder = "OpenVINOExecutionProvider:GPU"' in benchmark
    assert "silent_cpu_fallback_allowed = $false" in benchmark
    assert "image_paths_recorded = $false" in benchmark
    assert "image_bytes_recorded = $false" in benchmark


def test_n100_gpu_builder_preserves_models_and_packages_openvino_gpu() -> None:
    build_script = (ROOT / "scripts/build_n100_gpu_test.ps1").read_text(encoding="utf-8")
    dependency_lock = (ROOT / "configs/runtime/requirements-windows-openvino.lock").read_text(
        encoding="utf-8"
    )

    assert "onnxruntime-openvino==1.24.1" in dependency_lock
    assert "DmlExecutionProvider" in build_script
    assert "CPUExecutionProvider" in build_script
    assert "OpenVINOExecutionProvider" in build_script
    assert "Assert-DirectoryCopyMatches" in build_script
    assert "runtime_or_catalog_payload_changed = $false" in build_script
    assert "model_graph_or_weight_changed = $false" in build_script
    assert "decision_policy_changed = $false" in build_script
    assert "openvino_intel_cpu_plugin.dll" in build_script
    assert "openvino_intel_gpu_plugin.dll" in build_script
    assert "IncludeOpenVinoGpu" in build_script
    assert "candidate-manifest.json" in build_script
    assert "package-manifest.json" in build_script
    assert "target_full_path_latency_ms = 500" in build_script
    assert '[string]$Version = "0.1.4"' in build_script
    assert 'candidate_primary_embedder = "OpenVINOExecutionProvider:GPU"' in build_script
    assert 'candidate_rotation_180_embedder = "OpenVINOExecutionProvider:GPU"' in build_script
    assert (
        'candidate_independent_verifier_embedder = "OpenVINOExecutionProvider:GPU"' in build_script
    )
    assert "Get-RelativePackagePath" in build_script
    assert "[System.IO.Path]::GetRelativePath" not in build_script
