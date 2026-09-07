from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_windows_installer_uses_portable_cpu_worker_payload() -> None:
    build_script = (ROOT / "scripts" / "build_windows_installer.ps1").read_text(encoding="utf-8")
    version_config = (ROOT / "configs" / "versions" / "0.1.16.json").read_text(encoding="utf-8")
    inno_script = (ROOT / "installer" / "windows" / "BixolonBakeryAIScanner.iss").read_text(
        encoding="utf-8"
    )
    launcher = (ROOT / "installer" / "windows" / "start-bixolon-scanner.ps1").read_text(
        encoding="utf-8"
    )
    worker_launcher = (ROOT / "installer" / "windows" / "start-bixolon-worker.ps1").read_text(
        encoding="utf-8"
    )

    assert "cpu-worker-build/bixolon-worker" in build_script
    assert "requirements-windows-cpu.lock" in build_script
    assert "requirements-windows-openvino.lock" not in build_script
    assert "onnxruntime_providers_openvino.dll" not in build_script
    assert '"worker/model-package"' in build_script
    assert '"worker/store-catalog"' in build_script
    assert "runtime_or_catalog_payload_changed = $false" in build_script
    assert "cuda_runtime_included = $false" in build_script
    assert "openvino_runtime_included = $false" in build_script
    assert "onnxruntime_providers_(cuda|tensorrt|dml|openvino)" in build_script
    assert "installer-payload-manifest.json" in build_script
    assert "deployment-provenance.json" in build_script
    assert "$recommendedDetectorWorkers = 1" in build_script
    assert "$recommendedDetectorThreads = 4" in build_script
    assert "$recommendedEmbedderThreads = 4" in build_script
    assert 'processor_profile = "ONNX Runtime CPU"' in build_script
    assert 'target = "windows-x64-cpu"' in build_script
    assert "worker-manifest.json" in build_script
    assert "$workerZipHashPath" in build_script
    assert "$renderedLauncher" in build_script
    assert "Get-FileHash -Algorithm SHA256" in build_script
    assert "n100" not in version_config.lower()

    for source in (launcher, worker_launcher):
        assert 'BIXOLON_PROVIDER = "cpu"' in source
        assert 'BIXOLON_EMBEDDER_PROVIDER = "same"' in source
        assert 'BIXOLON_EMBEDDER_FALLBACK_PROVIDER = "none"' in source
        assert "BIXOLON_OPENVINO_CACHE_DIR" not in source
    assert 'BIXOLON_CPU_DETECTOR_WORKERS = "1"' in launcher
    assert 'BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = "4"' in launcher
    assert 'BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS = "4"' in launcher
    assert "start-bixolon-scanner.ps1" in inno_script
    assert "ArchitecturesAllowed=x64compatible" in inno_script
    assert "MinVersion=10.0.17763" in inno_script
    assert "vc_redist.x64.exe" in inno_script
    assert "PrivilegesRequired=admin" in inno_script
    assert "BixolonBakeryAIScanner-{#AppVersion}-Setup" in inno_script


def test_windows_installer_documents_target_requirements_and_limits() -> None:
    guide = (ROOT / "installer" / "windows" / "INSTALL-KO.txt").read_text(encoding="utf-8")

    assert "Windows 10 1809" in guide
    assert "Python, Flutter, CUDA, OpenVINO 별도 설치 불필요" in guide
    assert "ONNX Runtime CPU" in guide
    assert "CPU, 1 worker x 4 threads" in guide
    assert "SLA가 아닙니다" in guide
    assert "Authenticode 서명은 없습니다" in guide


def test_windows_packaging_scripts_support_windows_powershell_51() -> None:
    for relative_path in (
        "scripts/build_openvino_worker.ps1",
        "scripts/build_n100_test_candidate.ps1",
        "scripts/build_windows_installer.ps1",
        "scripts/build_worker_handoff.ps1",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "[System.IO.Path]::GetRelativePath" not in source
        if relative_path != "scripts/build_openvino_worker.ps1":
            assert "Get-RelativePackagePath" in source
