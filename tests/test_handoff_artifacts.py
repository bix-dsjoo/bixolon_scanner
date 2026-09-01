from __future__ import annotations

import json
from pathlib import Path

import pytest

from bixolon_scanner.contracts.api import ItemStatus, ScanResponse, Status

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs" / "contracts" / "examples" / "0.1.12"


@pytest.mark.parametrize(
    ("name", "status", "item_status"),
    [
        ("approved.json", Status.SEGMENTATION, ItemStatus.APPROVED),
        ("unknown.json", Status.SEGMENTATION, ItemStatus.UNKNOWN),
        ("segment-recapture.json", Status.SEGMENTATION, ItemStatus.SEGMENT_RECAPTURE),
        ("image-recapture.json", Status.IMAGE_RECAPTURE, None),
        ("error.json", Status.ERROR, None),
    ],
)
def test_handoff_examples_follow_python_contract(
    name: str,
    status: Status,
    item_status: ItemStatus | None,
) -> None:
    response = ScanResponse.model_validate_json((EXAMPLES / name).read_text(encoding="utf-8"))

    assert response.status is status
    assert response.worker_version == "0.1.12"
    assert all(
        value == "0.1.12"
        for value in (
            response.detector_version,
            response.classifier_version,
            response.embedder_version,
            response.detector_policy_version,
            response.classifier_policy_version,
            response.catalog_version,
        )
        if value is not None
    )
    if item_status is None:
        assert response.segmentations == []
    else:
        assert response.segmentations[0].status is item_status


def test_handoff_schema_has_the_complete_public_response_shape() -> None:
    schema = json.loads(
        (ROOT / "schemas" / "scan-response.schema.json").read_text(encoding="utf-8")
    )

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "request_id",
        "status",
        "reason_codes",
        "segmentations",
        "processing_time_ms",
        "worker_version",
        "detector_version",
        "classifier_version",
        "embedder_version",
        "detector_policy_version",
        "classifier_policy_version",
        "catalog_version",
    }
    assert schema["properties"]["status"]["enum"] == [
        "SEGMENTATION",
        "IMAGE_RECAPTURE",
        "ERROR",
    ]
    assert schema["$defs"]["Segmentation"]["properties"]["status"]["enum"] == [
        "APPROVED",
        "UNKNOWN",
        "SEGMENT_RECAPTURE",
    ]


def test_packaged_cpu_handoff_scripts_target_n100_cpu() -> None:
    lock = (ROOT / "configs" / "runtime" / "requirements-windows-openvino.lock").read_text(
        encoding="utf-8"
    )
    start_script = (ROOT / "scripts" / "handoff" / "start-worker.ps1").read_text(encoding="utf-8")
    benchmark = (ROOT / "scripts" / "handoff" / "benchmark-n100.ps1").read_text(encoding="utf-8")
    commands = (ROOT / "scripts" / "handoff" / "RUN-COMMANDS.txt").read_text(encoding="utf-8")
    build_script = (ROOT / "scripts" / "build_worker_handoff.ps1").read_text(encoding="utf-8")

    assert "onnxruntime-openvino==1.24.1" in lock
    assert "openvino==2025.4.1" in lock
    assert "onnxruntime-gpu" not in lock.lower()
    assert 'BIXOLON_PROVIDER = "openvino"' in start_script
    assert 'BIXOLON_REQUEST_TIMEOUT_SECONDS = "60"' in start_script
    assert "BIXOLON_CPU_DETECTOR_WORKERS" in start_script
    assert "[int]$DetectorWorkers = 1" in start_script
    assert "[int]$DetectorThreads = 0" in start_script
    assert 'Name = "candidate-openvino-1xauto"' in benchmark
    assert "DetectorWorkers = 1" in benchmark
    assert "DetectorThreads = 0" in benchmark
    assert 'Provider = "openvino"' in benchmark
    assert "segmentation_status_counts" in benchmark
    assert "response_contract_safe" in benchmark
    assert "cross_provider_parity_checked = $false" in benchmark
    assert "MaximumFullPathLatencyMs = 1000.0" in benchmark
    assert "candidate.P95Ms -le $MaximumFullPathLatencyMs" in benchmark
    assert "$targetCpuDetected" in benchmark
    assert "mean_within_target" in benchmark
    assert "p95_within_target" in benchmark
    assert "image_paths_recorded = $false" in benchmark
    assert '[string]$OutputPath = ""' in benchmark
    assert 'Join-Path $PSScriptRoot "n100-benchmark-result.json"' in benchmark
    assert "$index = [int][Math]::Ceiling" in benchmark
    assert (
        'powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\\start-worker.ps1"' in commands
    )
    assert (
        'powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\\benchmark-n100.ps1"'
        in commands
    )
    assert "onnxruntime_providers_(cuda|tensorrt|dml)" in build_script
    assert "requirements-windows-openvino.lock" in build_script
    assert 'provider = "OpenVINOExecutionProvider:CPU"' in build_script
    assert "worker-manifest.json" in build_script
    assert '"PENDING_FIELD_MEASUREMENT"' in build_script
    assert '"MEASURED_DIAGNOSTIC"' in build_script
    assert "$n100Benchmark" in build_script
    assert ".zip.sha256" not in build_script
    assert '"$zipPath.sha256"' in build_script


def test_n100_candidate_exposes_double_click_and_requested_powershell_entrypoint() -> None:
    build_script = (ROOT / "scripts" / "build_n100_test_candidate.ps1").read_text(encoding="utf-8")
    command_script = (ROOT / "scripts" / "handoff" / "RUN-N100-TEST.cmd").read_text(
        encoding="utf-8"
    )
    readme = (ROOT / "scripts" / "handoff" / "README-N100-KO.txt").read_text(encoding="utf-8")

    assert '"N100-STAGE-TEST.ps1"' in build_script
    assert '"scripts/handoff/README-N100-KO.txt"' in build_script
    assert "config.evaluation_evidence" in build_script
    assert "reference_evidence_sha256" in build_script
    assert "v2-scanner415-openvino-cpu" in build_script
    assert "$null -ne $metadata.count_verifier" in build_script
    assert "classifier_resolution_fallback.selective_roi_only" in build_script
    assert "detector_primary_classifier_routing" in build_script
    assert 'provider = "OpenVINOExecutionProvider:CPU"' in build_script
    assert 'detector_family = "SSDLite320"' in build_script
    assert "object_presence_verifier = $null" in build_script
    assert "target_full_path_latency_ms = 1000" in build_script
    assert "scanner-$Version/full-valid-openvino.json" not in build_script
    assert '-File ".\\N100-STAGE-TEST.ps1"' in readme
    assert "N100-STAGE-TEST.ps1" in command_script
    assert "n100-0.1.12-result.json" in command_script
    assert "image_sha256_recorded = $true" in (
        ROOT / "scripts" / "handoff" / "benchmark-n100.ps1"
    ).read_text(encoding="utf-8")
