"""Evaluate the frozen 0.2.1 policy on the historical final300 set; never retune."""

from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.evaluation.recapture_improvement import retained_prediction_parity
from bixolon_scanner.evaluation.three_bakery_http import final_records, measure, provider_parity
from bixolon_scanner.training.three_bakery_data import read_jsonl, write_json


def main():
    work = Path("artifacts/retraining/recapture-0.2.1")
    freeze = work / "selection-context.json"
    config = load_json_config(Path("configs/experiments/bread/three_bakery_revised300.json"))
    output = work / "final300-context"
    records = final_records(config, freeze, output)
    reports = {}
    for provider in ("cpu", "cuda"):
        reports[provider] = {}
        for version in ("0.2.0", "0.2.1"):
            report = measure(
                Path(f"artifacts/versions/{version}/staging"),
                records,
                output / provider / version,
                python=Path("artifacts/build-envs")
                / ("worker-cuda-py311" if provider == "cuda" else "worker-cpu-0.1.17-py311")
                / "Scripts/python.exe",
                provider=provider,
                cpu_profile=(8, 12),
                warmup=10,
                repetitions=1,
                input_identity={
                    "selection_sha256": sha256_file(freeze),
                    "role": "historical final300 regression after candidate freeze; no retuning",
                },
                cuda_dll_dir=Path("artifacts/runtime/cuda-13.0-cudnn-9.13.1-win-x64")
                if provider == "cuda"
                else None,
            )
            s = report["summary"]
            reports[provider][version] = s
            print(
                provider,
                version,
                {
                    k: s[k]
                    for k in (
                        "correct_approved_count",
                        "wrong_approved_count",
                        "missed_count",
                        "extra_count",
                        "item_status_counts",
                    )
                },
                s["latency"]["all"],
                flush=True,
            )
    checks = {}
    for provider, versions in reports.items():
        b, c = versions["0.2.0"], versions["0.2.1"]
        checks[provider] = {
            "correct_approvals_not_lower": c["correct_approved_count"]
            >= b["correct_approved_count"],
            "wrong_approvals_not_higher": c["wrong_approved_count"] <= b["wrong_approved_count"],
            "misses_not_higher": c["missed_count"] <= b["missed_count"],
            "p95_not_slower": c["latency"]["all"]["p95_ms"] <= b["latency"]["all"]["p95_ms"],
        }
    retained = {
        p: retained_prediction_parity(
            read_jsonl(output / p / "0.2.0/responses.jsonl"),
            read_jsonl(output / p / "0.2.1/responses.jsonl"),
        )
        for p in reports
    }
    parity = provider_parity(
        output / "cpu/0.2.1/responses.jsonl", output / "cuda/0.2.1/responses.jsonl"
    )
    baseline_parity = provider_parity(
        output / "cpu/0.2.0/responses.jsonl", output / "cuda/0.2.0/responses.jsonl"
    )
    no_new_provider_mismatch = set(parity["status_rank_mismatch_image_ids"]) <= set(
        baseline_parity["status_rank_mismatch_image_ids"]
    )
    write_json(
        output / "comparison.json",
        {
            "reports": reports,
            "checks": checks,
            "retained_decisions": retained,
            "cpu_cuda_parity": parity,
            "baseline_cpu_cuda_parity": baseline_parity,
            "no_new_provider_mismatch": no_new_provider_mismatch,
            "no_regression": all(all(c.values()) for c in checks.values())
            and no_new_provider_mismatch
            and all(p["unchanged_retained_decisions"] for p in retained.values()),
            "passed": all(all(c.values()) for c in checks.values())
            and parity["status_rank_parity"]
            and all(p["unchanged_retained_decisions"] for p in retained.values()),
        },
    )


if __name__ == "__main__":
    main()
