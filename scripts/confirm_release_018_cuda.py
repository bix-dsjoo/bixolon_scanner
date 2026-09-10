"""Diagnose the observed CUDA timing variation with a fixed AB/BA/AB order."""

from __future__ import annotations

from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.evaluation.cpu_sleep_confirmation import external_workers
from bixolon_scanner.evaluation.three_bakery_http import measure, provider_parity
from bixolon_scanner.training.three_bakery_data import read_jsonl, write_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    version = root / "artifacts/versions/0.1.18"
    output = version / "cuda-interleaved"
    final = version / "final-evaluation"
    config = {
        "reason": "Block comparison CUDA p95 increased 10.6% in one repeat; CPU-only change retained",
        "orders": [["0.1.17", "0.1.18"], ["0.1.18", "0.1.17"], ["0.1.17", "0.1.18"]],
        "warmup": 10,
        "repetitions_per_worker": 1,
        "profile": [8, 12],
        "freeze_sha256": sha256_file(final / "final-candidate.json"),
        "inputs_sha256": sha256_file(final / "final-inputs.jsonl"),
        "runner_sha256": sha256_file(Path(__file__)),
        "scope": "additional regression diagnosis only; no selection, training or policy changes",
        "reporting": "Retain every fixed repetition; report all p95 ratios and counts",
    }
    path = output / "config.json"
    if path.exists() and load_json_config(path) != config:
        raise ValueError("CUDA confirmation inputs or protocol changed")
    write_json(path, config)
    records = read_jsonl(final / "final-inputs.jsonl")
    before = external_workers()
    rows = []
    for index, order in enumerate(config["orders"], 1):
        reports = {}
        for product_version in order:
            candidate = root / f"artifacts/versions/{product_version}"
            reports[product_version] = measure(
                candidate / "staging",
                records,
                output / str(index) / product_version,
                python=root / "artifacts/build-envs/worker-cuda-py311/Scripts/python.exe",
                provider="cuda",
                cuda_dll_dir=candidate / "staging/cuda-runtime",
                warmup=10,
                repetitions=1,
                match_iou=0.5,
                input_identity={"diagnostic_config_sha256": sha256_file(path)},
                cpu_profile=(8, 12),
                worker_executable=candidate
                / f"bixolon-bakery-ai-scanner-{product_version}/worker/bixolon-worker.exe",
            )
        old, new = (reports[v]["summary"] for v in ("0.1.17", "0.1.18"))
        parity = provider_parity(
            output / str(index) / "0.1.17/responses.jsonl",
            output / str(index) / "0.1.18/responses.jsonl",
        )
        rows.append(
            {
                "repeat": index,
                "old": old,
                "new": new,
                "parity": parity,
                "p95_ratio": new["latency"]["all"]["p95_ms"] / old["latency"]["all"]["p95_ms"],
            }
        )
    after = external_workers()
    result = {
        "repetitions": rows,
        "config_sha256": sha256_file(path),
        "status_rank_parity": all(r["parity"]["status_rank_parity"] for r in rows),
        "all_p95_ratios_at_most_1_05": all(r["p95_ratio"] <= 1.05 for r in rows),
        "external_workers_before": before,
        "external_workers_after": after,
        "claim_scope": config["scope"],
    }
    write_json(output / "report.json", result)
    print({"p95_ratios": [r["p95_ratio"] for r in rows]}, flush=True)


if __name__ == "__main__":
    main()
