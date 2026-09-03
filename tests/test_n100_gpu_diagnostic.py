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
    assert "n100-0.1.14-openvino-device-matrix.json" in command
    assert 'ExpectedVersion = "0.1.14"' in benchmark
    assert "BIXOLON_PROVIDER = [string]$Profile.DetectorProvider" in benchmark
    assert 'DetectorProvider = "openvino"' in benchmark
    assert 'EmbedderProvider = "same"' in benchmark
    assert 'EmbedderProvider = "openvino_gpu"' in benchmark
    assert 'ExpectedProvider = "openvino+openvino_gpu"' in benchmark
    assert 'Name = "openvino-cpu-only"' in benchmark
    assert 'Name = "openvino-cpu-detector-intel-gpu-embedder"' in benchmark
    assert "object_presence_verifier = if ($hasCountVerifier)" in benchmark
    assert '"OpenVINOExecutionProvider:CPU"' in benchmark
    assert '"OpenVINOExecutionProvider:GPU"' in benchmark
    assert "object_presence_execution = if ($hasCountVerifier)" in benchmark
    assert "object_presence_verifier_sha256" in benchmark
    assert "object_presence_confidence_threshold" in benchmark
    assert 'BIXOLON_CPU_DETECTOR_WORKERS = "1"' in benchmark
    assert 'BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = "4"' in benchmark
    assert "semantic_mismatch_count" in benchmark
    assert "maximum_confidence_delta" in benchmark
    assert "ConfidenceTolerance = 0.02" in benchmark
    assert "count_verifier_configured = $hasCountVerifier" in benchmark
    assert "minimum_mean_speedup_ratio" in benchmark
    assert "maximum_peak_working_set_bytes" in benchmark
    assert "MaximumWorkingSetBytes = 2415919104" in benchmark
    assert "MaximumFullPathLatencyMs = 1000.0" in benchmark
    assert "mean_within_target" in benchmark
    assert "p95_within_target" in benchmark
    assert "gpu_embedder_target_met" in benchmark
    assert "hybrid_resource_safe" in benchmark
    assert "same_runtime_catalog_and_policy = $true" in benchmark
    assert "model_graph_or_weight_changed = [bool]$ModelGraphOrWeightChanged" in benchmark
    assert "decision_policy_changed = [bool]$DecisionPolicyChanged" in benchmark
    assert 'independent_verifier_embedder = "OpenVINOExecutionProvider:GPU"' in benchmark
    assert "silent_cpu_fallback_allowed = $false" in benchmark
    assert "image_paths_recorded = $false" in benchmark
    assert "image_bytes_recorded = $false" in benchmark
    assert "image_sha256_recorded = $true" in benchmark
    assert "image_sha256 = $imageSha256" in benchmark


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
    assert "target_full_path_latency_ms = 1000" in build_script
    assert 'baseline_object_presence_verifier = "not_configured"' in build_script
    assert 'candidate_object_presence_verifier = "not_configured"' in build_script
    assert 'candidate_object_presence_execution = "not_configured"' in build_script
    assert "$null -ne $runtimeMetadata.count_verifier" in build_script
    assert "classifier_resolution_fallback.selective_roi_only" in build_script
    assert "detector_primary_classifier_routing" in build_script
    assert '[string]$Version = "0.1.14"' in build_script
    assert 'candidate_primary_embedder = "OpenVINOExecutionProvider:GPU"' in build_script
    assert 'candidate_rotation_180_embedder = "OpenVINOExecutionProvider:GPU"' in build_script
    assert (
        'candidate_independent_verifier_embedder = "OpenVINOExecutionProvider:GPU"' in build_script
    )
    assert "Get-RelativePackagePath" in build_script
    assert "[System.IO.Path]::GetRelativePath" not in build_script


def test_yolo_free_n100_diagnostic_uses_exact_count_and_1000ms_target() -> None:
    build_script = (ROOT / "scripts/build_yolo_free_n100_gpu_test.ps1").read_text(encoding="utf-8")
    command = (ROOT / "scripts/handoff/RUN-YOLO-FREE-N100-GPU-TEST.cmd").read_text(encoding="utf-8")
    benchmark = (ROOT / "scripts/handoff/N100-GPU-BENCHMARK.ps1").read_text(encoding="utf-8")

    assert "runtime-candidate-v16-primary192-verifier160" in build_script
    assert "detector-production.onnx" in build_script
    assert 'comparison_mode -ne "exact_count"' in build_script
    assert "checkpoint_weights_changed = $false" in build_script
    assert "model_graph_changed = -not $DetectorPrimary" in build_script
    assert "runtime_policy_changed = $true" in build_script
    assert "catalog_payload_changed = $false" in build_script
    assert "openvino_intel_cpu_plugin.dll" in build_script
    assert "openvino_intel_gpu_plugin.dll" in build_script
    assert "IncludeOpenVinoGpu" in build_script
    assert "target_full_path_p95_ms = 1000" in build_script
    assert 'ExpectedVersion "0.1.7"' in command
    assert 'ExpectedCountComparisonMode "exact_count"' in command
    assert "MaximumFullPathLatencyMs 1000.0" in command
    assert 'ExpectedCountComparisonMode = "object_presence"' in benchmark
    assert "count_verifier_comparison_mode" in benchmark
    assert "count_verifier_sha256" in benchmark


def test_ssdlite_n100_diagnostic_uses_selective_fallback_and_measured_tolerances() -> None:
    build_script = (ROOT / "scripts/build_yolo_free_n100_gpu_test.ps1").read_text(encoding="utf-8")
    command = (ROOT / "scripts/handoff/RUN-SSDLITE-N100-GPU-TEST.cmd").read_text(encoding="utf-8")
    readme = (ROOT / "scripts/handoff/README-SSDLITE-N100-KO.txt").read_text(encoding="utf-8")

    assert "[switch]$Ssdlite" in build_script
    assert "runtime-candidate-v42-ssdlite320-selective-fallback" in build_script
    assert "selective_roi_only" in build_script
    assert "SSDLite320" in build_script
    assert "$null -ne $runtimeMetadata.count_verifier" in build_script
    assert "RUN-SSDLITE-N100-GPU-TEST.cmd" in build_script
    assert "-AllowNoCountVerifier" in command
    assert "-ConfidenceTolerance 0.02" in command
    assert "-MaximumWorkingSetBytes 2415919104" in command
    assert "MaximumFullPathLatencyMs 1000.0" in command
    assert "n100-ssdlite-0.1.7-openvino-device-matrix.json" in command
    assert "class rank" in readme
    assert "confidence parity" in readme


def test_detector_primary_n100_diagnostic_preserves_models_and_routes_selected_rois() -> None:
    build_script = (ROOT / "scripts/build_yolo_free_n100_gpu_test.ps1").read_text(encoding="utf-8")
    command = (ROOT / "scripts/handoff/RUN-DETECTOR-PRIMARY-N100-GPU-TEST.cmd").read_text(
        encoding="utf-8"
    )
    readme = (ROOT / "scripts/handoff/README-DETECTOR-PRIMARY-N100-KO.txt").read_text(
        encoding="utf-8"
    )

    assert "[switch]$DetectorPrimary" in build_script
    assert "runtime-candidate-v2-0.1.11-selective-dino" in build_script
    assert "detector_primary_classifier_routing" in build_script
    assert "minimum_detector_score -ne 0.98" in build_script
    assert "model_graph_changed = -not $DetectorPrimary" in build_script
    assert 'ExpectedVersion "0.1.11"' in command
    assert "-DecisionPolicyChanged" in command
    assert "n100-detector-primary-0.1.11-openvino-device-matrix.json" in command
    assert "DINOv3" in readme
    assert "detector score" in readme
    assert "integrity.decision_policy_changed" in readme
