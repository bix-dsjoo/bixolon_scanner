"""Confirm the frozen EXEs on the fixed set and on untrained log images."""

from __future__ import annotations

import runpy
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.evaluation.cpu_sleep_confirmation import external_workers
from bixolon_scanner.evaluation.three_bakery_http import measure, provider_parity
from bixolon_scanner.training.three_bakery_data import read_jsonl, write_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    version = root / "artifacts/versions/0.1.18"
    study = root / "artifacts/retraining/three-bakery-improvement-0.1.18"
    before = external_workers()
    runpy.run_path(str(root / "scripts/evaluate_release_018.py"), run_name="__main__")
    output = version / "packaged-log-cpu"
    report = measure(
        version / "staging",
        read_jsonl(study / "log-inputs.jsonl"),
        output,
        python=root / "artifacts/build-envs/worker-cpu-0.1.17-py311/Scripts/python.exe",
        provider="cpu",
        warmup=10,
        repetitions=3,
        match_iou=0.5,
        input_identity={
            "release_freeze_sha256": sha256_file(version / "final-evaluation/final-candidate.json"),
            "input_sha256": sha256_file(study / "log-inputs.jsonl"),
        },
        cpu_profile=(8, 12),
        worker_executable=root
        / "artifacts/installers/0.1.18/windows-payload/worker/bixolon-worker.exe",
        maximum_p95_ms=300,
    )
    parity = provider_parity(study / "baseline/log-cpu/responses.jsonl", output / "responses.jsonl")
    after = external_workers()
    interference = before.keys() != after.keys() or any(
        after[pid]["created"] != old["created"]
        or after[pid]["cpu_seconds"] - old["cpu_seconds"] > 0.5
        for pid, old in before.items()
        if pid in after
    )
    confirmation = {
        "parity": parity,
        "external_workers": {
            "before": before,
            "after": after,
            "possible_interference": interference,
            "scope": "before/after process snapshots; not continuous system monitoring",
        },
        "packaged_log_p95_ms": [r["latency"]["all"]["p95_ms"] for r in report["repetitions"]],
        "final_regression_passed": load_json_config(version / "final-evaluation/report.json")[
            "regression_passed"
        ],
        "passed": parity["status_rank_parity"]
        and not interference
        and all(r["latency"]["full_path"]["p95_ms"] <= 300 for r in report["repetitions"]),
    }
    write_json(version / "packaged-confirmation.json", confirmation)
    print(confirmation, flush=True)


if __name__ == "__main__":
    main()
