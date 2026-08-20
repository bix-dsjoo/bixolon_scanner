from __future__ import annotations

import json

import pytest

from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.evaluation.scanner_v2_breakdown import build_difficulty_breakdown


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_breakdown_uses_segmentation_outputs_for_item_status_rates(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    trace = tmp_path / "trace.jsonl"
    development = tmp_path / "development.json"
    output = tmp_path / "breakdown.json"
    _write_jsonl(
        manifest,
        [
            {"image_id": 1, "difficulty": "EASY"},
            {"image_id": 2, "difficulty": "HARD"},
        ],
    )
    _write_jsonl(
        trace,
        [
            {
                "image_id": 1,
                "status": "SEGMENTATION",
                "latency_ms": 10.0,
                "ground_truth_count": 3,
                "decision": {
                    "segmentations": [
                        {"status": "APPROVED"},
                        {"status": "UNKNOWN"},
                    ]
                },
                "prediction_count": 2,
                "matched_count": 2,
                "false_negative_count": 1,
                "false_positive_count": 0,
                "matched_classifier_diagnostics": [
                    {
                        "final_status": "APPROVED",
                        "classifier_top1_correct": True,
                        "classifier_top3_hit": True,
                    },
                    {
                        "final_status": "UNKNOWN",
                        "classifier_top1_correct": False,
                        "classifier_top3_hit": False,
                    },
                ],
            },
            {
                "image_id": 2,
                "status": "IMAGE_RECAPTURE",
                "latency_ms": 20.0,
                "ground_truth_count": 2,
                "decision": {"segmentations": []},
            },
        ],
    )
    development.write_text(
        json.dumps(
            {
                "evaluation": "scanner_2_0_development_300",
                "promotion_evidence": False,
                "dataset": {"manifest_sha256": sha256_file(manifest)},
                "trace": {"sha256": sha256_file(trace)},
                "counts": {
                    "image_count": 2,
                    "segmentation_image_count": 1,
                    "image_recapture_count": 1,
                    "ground_truth_count": 5,
                    "prediction_count": 2,
                    "matched_count": 2,
                    "false_negative_count": 1,
                    "false_positive_count": 0,
                    "false_negative_image_count": 1,
                    "false_positive_image_count": 0,
                    "approved_count": 1,
                    "approved_misrecognition_count": 0,
                    "unknown_count": 1,
                    "segment_recapture_count": 0,
                    "unknown_candidate_out_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_difficulty_breakdown(
        development_report_path=development,
        trace_path=trace,
        manifest_path=manifest,
        output=output,
    )

    assert report["overall"]["metrics"]["approved_over_segmentation_rate"] == 0.5
    assert report["overall"]["metrics"]["unknown_top3_over_segmentation_rate"] == 0.5
    assert report["overall"]["metrics"]["unknown_top3_candidate_out_rate_over_all_gt"] == 0.2
    assert report["overall"]["performance"]["mean_ms"] == 15.0
    assert report["by_difficulty"]["EASY"]["metrics"]["segmentation_rate"] == 1.0
    assert report["by_difficulty"]["HARD"]["metrics"]["image_recapture_rate"] == 1.0


def test_breakdown_rejects_trace_not_locked_by_report(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    trace = tmp_path / "trace.jsonl"
    development = tmp_path / "development.json"
    _write_jsonl(manifest, [{"image_id": 1, "difficulty": "EASY"}])
    _write_jsonl(trace, [{"image_id": 1}])
    development.write_text(
        json.dumps(
            {
                "evaluation": "scanner_2_0_development_300",
                "promotion_evidence": False,
                "dataset": {"manifest_sha256": sha256_file(manifest)},
                "trace": {"sha256": "0" * 64},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="trace differs"):
        build_difficulty_breakdown(
            development_report_path=development,
            trace_path=trace,
            manifest_path=manifest,
            output=tmp_path / "output.json",
        )
