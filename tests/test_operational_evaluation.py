from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from bixolon_scanner.training import evaluate_operational


def test_single_session_fit_report_can_pass_fit_but_never_production(monkeypatch, tmp_path):
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    records = []
    for index in range(42):
        records.append(
            {
                "record_type": "detection",
                "source": "operational_scan_log_v2",
                "scan_id": f"{index:032x}",
                "sequence_index": index,
                "capture_session_id": "one-session",
                "physical_target_group_id": "one-group",
                "expected_detector_action": "CONTINUE" if index < 34 else "RECAPTURE",
            }
        )
    manifest = manifest_dir / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    (manifest_dir / "metadata.json").write_text(
        json.dumps({"dataset_version": "bread-test"}), encoding="utf-8"
    )

    candidate = {
        "normal_count": 34,
        "normal_continued_count": 34,
        "normal_continue_rate": 1.0,
        "false_recapture_count": 0,
        "recapture_count": 8,
        "recapture_correct_count": 8,
        "recapture_recall": 1.0,
    }
    monkeypatch.setattr(evaluate_operational, "_run_package", lambda *args, **kwargs: candidate)
    output = tmp_path / "report.json"
    report = evaluate_operational.evaluate(
        Namespace(
            package_dir=Path("candidate"),
            baseline_package_dir=None,
            manifest=manifest,
            dataset_root=tmp_path,
            output=output,
            evidence_role="fit",
            provider="cpu",
            cuda_dll_dir=None,
        )
    )

    assert report["gates"]["decision_gate_satisfied"] is True
    assert report["gates"]["independent_data_gate_satisfied"] is False
    assert report["gates"]["production_promotion_eligible"] is False
    assert report["promotion_evidence"] is False
