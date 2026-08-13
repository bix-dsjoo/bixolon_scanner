from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from bixolon_scanner.training.scan_log_review_export import (
    _corrections,
    _logged_annotations,
    load_latest_logs,
)


def _write_log(root: Path, scan_id: str, recorded_at: str) -> None:
    Image.new("RGB", (20, 10), "white").save(root / f"{scan_id}.jpg")
    (root / f"{scan_id}.json").write_text(
        json.dumps(
            {
                "scan_id": scan_id,
                "recorded_at": recorded_at,
                "original_image": f"{scan_id}.jpg",
            }
        ),
        encoding="utf-8",
    )


def test_load_latest_logs_selects_latest_then_returns_chronological(tmp_path):
    _write_log(tmp_path, "a" * 32, "2026-08-11T01:00:00Z")
    _write_log(tmp_path, "b" * 32, "2026-08-11T02:00:00Z")
    _write_log(tmp_path, "c" * 32, "2026-08-11T03:00:00Z")

    selected = load_latest_logs(tmp_path, 2)

    assert [row.scan_id for row in selected] == ["b" * 32, "c" * 32]


def test_logged_annotations_prefer_final_product_and_keep_bbox():
    payload = {
        "detections": [
            {
                "bbox": {"x": 10, "y": 20, "width": 30, "height": 40},
                "initial_ai_prediction": {"class_id": "bread_01"},
                "final_product": {"class_id": "bread_02"},
                "initial_confidence": 0.75,
                "user_modified": True,
            }
        ]
    }

    annotations = _logged_annotations(payload, {"bread_01": 1, "bread_02": 2})

    assert annotations == [
        {
            "category_id": 2,
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "area": 1200.0,
            "iscrowd": 0,
            "source": "scan_log_final",
            "review_status": "pending_user_review",
            "detector_score": None,
            "classifier_confidence": 0.75,
            "user_modified": True,
        }
    ]


def test_corrections_require_supported_schema(tmp_path):
    path = tmp_path / "corrections.json"
    path.write_text(json.dumps({"schema_version": "2.0"}), encoding="utf-8")

    try:
        _corrections(path)
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("unsupported correction schema was accepted")


def test_default_corrections_include_confirmed_recapture():
    assert _corrections(None)["confirmed_recapture"] == {}
