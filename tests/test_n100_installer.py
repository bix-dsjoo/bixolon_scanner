from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_n100_installer_uses_cpu_detector_gpu_embedder_payload_and_profile() -> None:
    build_script = (ROOT / "scripts" / "build_n100_installer.ps1").read_text(encoding="utf-8")
    inno_script = (ROOT / "installer" / "n100" / "BixolonScanner-N100.iss").read_text(
        encoding="utf-8"
    )
    launcher = (ROOT / "installer" / "n100" / "start-bixolon-scanner.ps1").read_text(
        encoding="utf-8"
    )
    worker_launcher = (ROOT / "installer" / "n100" / "start-bixolon-worker.ps1").read_text(
        encoding="utf-8"
    )

    assert "openvino-gpu-worker-build/bixolon-worker" in build_script
    assert "requirements-windows-openvino.lock" in build_script
    assert "onnxruntime_providers_openvino.dll" in build_script
    assert "openvino_intel_cpu_plugin.dll" in build_script
    assert "openvino_intel_gpu_plugin.dll" in build_script
    assert '"worker/model-package"' in build_script
    assert '"worker/store-catalog"' in build_script
    assert "runtime_or_catalog_payload_changed = $false" in build_script
    assert "cuda_runtime_included = $false" in build_script
    assert "onnxruntime_providers_(cuda|tensorrt|dml)" in build_script
    assert "installer-payload-manifest.json" in build_script
    assert "n100-reference-result.json" in build_script
    assert '"REFERENCE_MEASURED_DIAGNOSTIC"' in build_script
    assert "reference_product_version = $diagnosticVersion" in build_script
    assert "runtime.manifest_sha256" in build_script
    assert "catalog.manifest_sha256" in build_script
    assert "n100_latency_target_applied = $false" in build_script
    assert "hardware.target_cpu_detected" in build_script
    assert "hardware.target_intel_gpu_detected" in build_script
    assert "parity.semantic_mismatch_count" in build_script
    assert "operational_diagnostic_target_ms = 500" in build_script
    assert "BixolonScanner-N100-$Version-Worker" in build_script
    assert "worker-manifest.json" in build_script
    assert "$workerZipHashPath" in build_script
    assert "$recommendedDetectorWorkers" in build_script
    assert "$renderedLauncher" in build_script
    assert "Get-FileHash -Algorithm SHA256" in build_script

    assert 'BIXOLON_PROVIDER = "openvino"' in launcher
    assert 'BIXOLON_EMBEDDER_PROVIDER = "openvino_gpu"' in launcher
    assert 'BIXOLON_EMBEDDER_FALLBACK_PROVIDER = "same"' in launcher
    assert 'BIXOLON_CPU_DETECTOR_WORKERS = "1"' in launcher
    assert 'BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = "4"' in launcher
    assert 'BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS = "0"' in launcher
    assert "BIXOLON_OPENVINO_CACHE_DIR" in launcher
    assert 'BIXOLON_PROVIDER = "openvino"' in worker_launcher
    assert 'BIXOLON_EMBEDDER_PROVIDER = "openvino_gpu"' in worker_launcher
    assert 'BIXOLON_EMBEDDER_FALLBACK_PROVIDER = "same"' in worker_launcher
    assert "BIXOLON_OPENVINO_CACHE_DIR" in worker_launcher
    assert "start-bixolon-scanner.ps1" in inno_script
    assert "ArchitecturesAllowed=x64compatible" in inno_script
    assert "MinVersion=10.0.17763" in inno_script
    assert "vc_redist.x64.exe" in inno_script
    assert "PrivilegesRequired=admin" in inno_script


def test_n100_installer_documents_target_requirements_and_limits() -> None:
    guide = (ROOT / "installer" / "n100" / "INSTALL-N100-KO.txt").read_text(encoding="utf-8")

    assert "Windows 10 1809" in guide
    assert "Python, Flutter, CUDA 별도 설치 불필요" in guide
    assert "OpenVINOExecutionProvider" in guide
    assert "CPU, 1 worker x 4 threads" in guide
    assert "OpenVINOExecutionProvider Intel GPU" in guide
    assert "CPU Embedder로 명시적으로 fallback" in guide
    assert "평균과 p95 500ms 이하" in guide
    assert "지연시간 또는 SLA를 보장하지 않습니다" in guide
    assert "Authenticode 서명은 없습니다" in guide


def test_windows_packaging_scripts_support_windows_powershell_51() -> None:
    for relative_path in (
        "scripts/build_n100_test_candidate.ps1",
        "scripts/build_n100_installer.ps1",
        "scripts/build_worker_handoff.ps1",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "[System.IO.Path]::GetRelativePath" not in source
        assert "Get-RelativePackagePath" in source
