from __future__ import annotations

import json
from pathlib import Path

import pytest

from bixolon_scanner.contracts.api import ItemStatus, ScanResponse, Status

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs" / "contracts" / "examples" / "0.0.2"


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
    assert response.worker_version == "0.0.2"
    assert all(
        value == "0.0.2"
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


def test_cpu_dependency_lock_and_handoff_scripts_are_cpu_only() -> None:
    lock = (ROOT / "configs" / "runtime" / "requirements-windows-cpu.lock").read_text(
        encoding="utf-8"
    )
    start_script = (ROOT / "scripts" / "handoff" / "start-worker.ps1").read_text(encoding="utf-8")
    benchmark = (ROOT / "scripts" / "handoff" / "benchmark-n100.ps1").read_text(encoding="utf-8")
    commands = (ROOT / "scripts" / "handoff" / "RUN-COMMANDS.txt").read_text(encoding="utf-8")
    build_script = (ROOT / "scripts" / "build_worker_handoff.ps1").read_text(encoding="utf-8")

    assert "onnxruntime==1.28.0" in lock
    assert "onnxruntime-gpu" not in lock.lower()
    assert 'BIXOLON_PROVIDER = "cpu"' in start_script
    assert 'BIXOLON_REQUEST_TIMEOUT_SECONDS = "60"' in start_script
    assert "BIXOLON_CPU_DETECTOR_WORKERS" in start_script
    assert "DetectorWorkers = 1; DetectorThreads = 4" in benchmark
    assert "DetectorWorkers = 2; DetectorThreads = 2" in benchmark
    assert "DetectorWorkers = 4; DetectorThreads = 1" in benchmark
    assert "image_paths_recorded = $false" in benchmark
    assert '[string]$OutputPath = ""' in benchmark
    assert 'Join-Path $PSScriptRoot "n100-benchmark-result.json"' in benchmark
    assert "$index = [int][Math]::Ceiling" in benchmark
    assert "return $values.ToArray()" in benchmark
    assert "return ,$values.ToArray()" not in benchmark
    assert (
        'powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\\start-worker.ps1"' in commands
    )
    assert (
        'powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\\benchmark-n100.ps1"'
        in commands
    )
    assert "onnxruntime_providers_(cuda|tensorrt)" in build_script
    assert "worker-manifest.json" in build_script
    assert ".zip.sha256" not in build_script
    assert '"$zipPath.sha256"' in build_script
