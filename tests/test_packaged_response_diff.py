from __future__ import annotations

import json

from bixolon_scanner.evaluation.packaged_response_diff import compare_packaged_responses


def _row(image_id: int, *, class_id: str = "bread_01", latency: float = 10.0) -> dict:
    return {
        "image_id": image_id,
        "response": {
            "status": "SEGMENTATION",
            "reason_codes": [],
            "segmentations": [
                {
                    "bbox": {"x": 1, "y": 2, "width": 3, "height": 4},
                    "status": "APPROVED",
                    "reason_codes": [],
                    "prediction": {"class_id": class_id, "class_name": "Bread"},
                    "top3": [],
                    "confidence": 0.9,
                }
            ],
            "processing_time_ms": latency,
            "worker_version": "0.1.7",
            "classifier_version": "0.1.7",
        },
    }


def test_packaged_response_diff_reports_exact_semantic_parity(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.jsonl"
    baseline.write_text(json.dumps({"rows": [_row(1)]}), encoding="utf-8")
    candidate.write_text(json.dumps(_row(1, latency=12.0)) + "\n", encoding="utf-8")

    report = compare_packaged_responses(baseline, candidate)

    assert report["image_count"] == 1
    assert report["semantic_diff_count"] == 0
    assert not any(report["component_diff_counts"].values())
    assert report["full_path_processing_time_ms"]["delta"]["mean_ms"] == 2.0


def test_packaged_response_diff_reports_class_change(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.jsonl"
    baseline.write_text(json.dumps({"rows": [_row(1)]}), encoding="utf-8")
    candidate.write_text(json.dumps(_row(1, class_id="bread_02")) + "\n", encoding="utf-8")

    report = compare_packaged_responses(baseline, candidate)

    assert report["semantic_diff_count"] == 1
    assert report["component_diff_counts"]["prediction"] == 1
