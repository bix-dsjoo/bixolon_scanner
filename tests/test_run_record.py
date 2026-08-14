from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from bixolon_scanner.training.run_record import write_run_record


def test_run_record_omits_local_roots_and_signed_sources(tmp_path):
    args = Namespace(
        dataset_root=Path("C:/private/dataset"),
        manifest=Path("C:/private/manifest.jsonl"),
        output_dir=tmp_path,
        weights=Path("C:/private/model/checkpoint.pth"),
        seed=7,
        fold=1,
    )
    write_run_record(
        tmp_path,
        task="classifier_training",
        args=args,
        device="cpu",
        dataset_sizes={"train": 10, "validation": 3},
    )
    raw = (tmp_path / "run.json").read_text(encoding="utf-8")
    record = json.loads(raw)
    assert "C:/private" not in raw
    assert record["arguments"]["weights"] == "checkpoint.pth"
    assert record["arguments"]["seed"] == 7


def test_run_record_includes_component_training_pipeline_lock(tmp_path):
    contract_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "training"
        / "bread_classifier_pipeline_1.0.0.json"
    )
    args = Namespace(
        manifest=None,
        output_dir=tmp_path,
        pipeline_contract=contract_path,
        seed=20260813,
    )

    write_run_record(
        tmp_path,
        task="classifier_training",
        args=args,
        device="cpu",
        dataset_sizes={"train": 240, "validation": 80},
    )

    record = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert record["training_pipeline"]["component"] == "classifier"
    assert record["training_pipeline"]["version"] == "1.0.0"
    assert len(record["training_pipeline"]["contract_sha256"]) == 64
