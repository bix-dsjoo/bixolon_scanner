from __future__ import annotations

import json
from pathlib import Path

from bixolon_scanner.training.compare_difficulty import compare, flatten_report


def _counts(images: int) -> dict:
    return {
        "images": images,
        "response_status": {"APPROVED": images - 2, "UNKNOWN": 1, "RECAPTURE": 1},
        "ground_truth_boxes": images * 2,
        "matched_boxes": images * 2 - 1,
        "missed_boxes": 1,
        "false_positive_boxes": 0,
        "classified_matched_boxes": images * 2 - 2,
        "approved_correct": images * 2 - 4,
        "approved_wrong": 1,
        "unknown_top3_correct": 1,
        "unknown_top3_missing": 0,
        "rates": {
            "top1_recognition_accuracy": 0.9,
            "unknown_top3_accuracy": 1.0,
        },
        "end_to_end_latency_ms": {"sample_count": images, "mean": 12.5},
    }


def _report() -> dict:
    return {
        "provider": "cuda",
        "package_version": "0.1.0-test",
        "dataset_root": "dataset",
        "match_iou_threshold": 0.5,
        "by_difficulty": {key: _counts(2) for key in ("E", "M", "H")},
        "overall": _counts(6),
    }


def test_flatten_report_preserves_emh_and_all_order():
    rows = flatten_report("n5", _report())
    assert [row["difficulty"] for row in rows] == ["E", "M", "H", "ALL"]
    assert rows[0]["condition"] == "n5"
    assert rows[0]["recapture_ground_truth_boxes"] == 2


def test_compare_writes_json_csv_and_korean_markdown(tmp_path: Path):
    source = tmp_path / "report.json"
    source.write_text(json.dumps(_report()), encoding="utf-8")
    output_json = tmp_path / "comparison.json"
    output_csv = tmp_path / "comparison.csv"
    output_markdown = tmp_path / "comparison.md"
    result = compare(
        [("current", source), ("n5", source)],
        output_json=output_json,
        output_csv=output_csv,
        output_markdown=output_markdown,
        dataset_note="독립 test 아님",
    )
    assert len(result["rows"]) == 8
    assert json.loads(output_json.read_text(encoding="utf-8"))["selected_n"] is None
    assert output_csv.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "자동 N 선택" in output_markdown.read_text(encoding="utf-8")
