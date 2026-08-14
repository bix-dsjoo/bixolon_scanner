from __future__ import annotations

import json
from pathlib import Path

import pytest

from bixolon_scanner.cli import COMMANDS
from bixolon_scanner.operations.worker_metrics import aggregate_worker_logs


def test_worker_metrics_aggregate_status_latency_and_versions(tmp_path: Path) -> None:
    path = tmp_path / "worker.jsonl"
    rows = [
        {
            "level": "INFO",
            "message": "scan_request_complete",
            "status": "SEGMENTATION",
            "approved_count": 2,
            "unknown_count": 1,
            "segment_recapture_count": 0,
            "processing_time_ms": 80.0,
            "worker_version": "1.0.0",
            "detector_version": "1.0.0",
            "classifier_version": "1.0.0",
        },
        {
            "level": "INFO",
            "message": "scan_request_complete",
            "status": "IMAGE_RECAPTURE",
            "approved_count": 0,
            "unknown_count": 0,
            "segment_recapture_count": 0,
            "processing_time_ms": 20.0,
            "worker_version": "1.0.0",
            "detector_version": "1.0.0",
            "classifier_version": None,
        },
        {"level": "ERROR", "message": "worker_model_error"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = aggregate_worker_logs([path])

    assert report["request_count"] == 2
    assert report["error_log_count"] == 1
    assert report["image_status_counts"]["IMAGE_RECAPTURE"] == 1
    assert report["segment_status_counts"]["UNKNOWN"] == 1
    assert report["recognition_proxy"]["approved_rate"] == pytest.approx(2 / 3)
    assert report["latency_ms"]["mean"] == 50.0
    assert len(report["version_compositions"]) == 2
    assert ("operations", "metrics") in COMMANDS
