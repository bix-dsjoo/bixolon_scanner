from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from bixolon_scanner.training.data import DetectionDataset
from bixolon_scanner.training.operational_logs import _load_scan_logs, _validated_contract


def _contract(tmp_path: Path, scan_ids: list[str]) -> Path:
    path = tmp_path / "decisions.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "expected_log_count": len(scan_ids),
                "capture_session_id": "session",
                "physical_target_group_id": "targets",
                "camera": "camera",
                "ordered_scan_ids": scan_ids,
                "empty_recapture_scan_ids": scan_ids[:1],
                "blur_recapture_scan_ids": scan_ids[1:2],
            }
        ),
        encoding="utf-8",
    )
    return path


def _scan_log(root: Path, scan_id: str, recorded_at: str) -> None:
    Image.new("RGB", (32, 24), "white").save(root / f"{scan_id}.jpg")
    (root / f"{scan_id}.json").write_text(
        json.dumps(
            {
                "log_schema_version": 2,
                "scan_id": scan_id,
                "worker_status": "RECAPTURE",
                "reason_codes": ["DETECTOR_UNCERTAIN_OBJECT"],
                "recorded_at": recorded_at,
                "original_image": f"{scan_id}.jpg",
                "model_versions": {"detector": "0.1.0", "classifier": None},
            }
        ),
        encoding="utf-8",
    )


def test_operational_contract_requires_exact_unique_ordered_ids(tmp_path):
    scan_id = "a" * 32
    contract = _contract(tmp_path, [scan_id, scan_id])
    with pytest.raises(ValueError, match="duplicates"):
        _validated_contract(contract)


def test_operational_logs_reject_duplicate_image_bytes(tmp_path):
    first, second = "a" * 32, "b" * 32
    contract = _validated_contract(_contract(tmp_path, [first, second]))
    _scan_log(tmp_path, first, "2026-08-11T01:00:00Z")
    _scan_log(tmp_path, second, "2026-08-11T01:01:00Z")
    with pytest.raises(ValueError, match="duplicate operational images"):
        _load_scan_logs([tmp_path], contract)


def test_detection_dataset_keeps_empty_negative_and_excludes_quality_regression(tmp_path):
    image = tmp_path / "image.jpg"
    Image.new("RGB", (32, 24), "white").save(image)
    records = [
        {
            "record_type": "detection",
            "image_id": 1,
            "image_path": image.name,
            "width": 32,
            "height": 24,
            "split": "development",
            "fold": 1,
            "annotations": [],
        },
        {
            "record_type": "detection",
            "image_id": 2,
            "image_path": image.name,
            "width": 32,
            "height": 24,
            "split": "development",
            "fold": 1,
            "exclude_from_detector_training": True,
            "annotations": [],
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    dataset = DetectionDataset(manifest, tmp_path, mode="final_train")

    assert len(dataset) == 1
    _, target = dataset[0]
    assert target["annotations"] == []
