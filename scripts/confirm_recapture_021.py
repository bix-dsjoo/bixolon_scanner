"""Serial paired accuracy and HTTP latency confirmation before packaging."""

import argparse
from pathlib import Path

from bixolon_scanner.contracts.artifact import directory_content_manifest
from bixolon_scanner.evaluation.cpu_sleep_confirmation import external_workers
from bixolon_scanner.evaluation.recapture_improvement import (
    compare_repetitions,
    retained_prediction_parity,
)
from bixolon_scanner.evaluation.three_bakery_http import measure, provider_parity
from bixolon_scanner.training.three_bakery_data import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="score040")
    parser.add_argument("--provider", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--dataset", choices=["log", "original"], default="log")
    args = parser.parse_args()
    work = Path("artifacts/retraining/recapture-0.2.1")
    candidates = {
        "baseline": Path("artifacts/versions/0.2.0/staging"),
        "candidate": work / "candidates" / args.candidate,
    }
    manifest = (
        Path("artifacts/retraining/three-bakery-improvement-0.1.18/log-inputs.jsonl")
        if args.dataset == "log"
        else Path("artifacts/retraining/three-bakery-revised300/prepared/original_detection.jsonl")
    )
    output = (
        work
        / ("confirmation" if args.candidate == "score040" else "confirmation-context")
        / args.dataset
        / args.provider
    )
    protocol = {
        "provider": args.provider,
        "cpu_profile": [8, 12],
        "warmup": 10,
        "order": [["baseline", "candidate"], ["candidate", "baseline"], ["baseline", "candidate"]]
        if args.dataset == "log"
        else [["baseline", "candidate"]],
        "runtime_identity": {
            k: directory_content_manifest(v / "runtime")["manifest_sha256"]
            for k, v in candidates.items()
        },
        "rule": "No wrong approvals or misses on log132, correct approvals >=1078, recapture reduction, and no paired HTTP p95 regression. Original set must not lose correct approvals or gain wrong approvals/misses.",
    }
    write_json(output / "protocol.json", protocol)
    records = read_jsonl(manifest)
    reports = {k: [] for k in candidates}
    interference = []
    for i, order in enumerate(protocol["order"]):
        for name in order:
            before = external_workers()
            result = measure(
                candidates[name],
                records,
                output / f"pair-{i + 1}" / name,
                python=Path("artifacts/build-envs")
                / ("worker-cuda-py311" if args.provider == "cuda" else "worker-cpu-0.1.17-py311")
                / "Scripts/python.exe",
                provider=args.provider,
                cpu_profile=(8, 12),
                warmup=10,
                repetitions=1,
                cuda_dll_dir=Path("artifacts/runtime/cuda-13.0-cudnn-9.13.1-win-x64")
                if args.provider == "cuda"
                else None,
            )
            after = external_workers()
            busy = before.keys() != after.keys() or any(
                after[k]["created"] != v["created"]
                or after[k]["cpu_seconds"] - v["cpu_seconds"] > 0.5
                for k, v in before.items()
                if k in after
            )
            interference.append(
                {
                    "pair": i + 1,
                    "name": name,
                    "before": before,
                    "after": after,
                    "possible_interference": busy,
                }
            )
            reports[name].append(result["summary"])
            s = result["summary"]
            print(
                args.dataset,
                args.provider,
                i + 1,
                name,
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
    if args.dataset == "log":
        comparison = compare_repetitions(reports["baseline"], reports["candidate"])
    else:
        b, c = reports["baseline"][0], reports["candidate"][0]
        comparison = {
            "accepted": c["correct_approved_count"] >= b["correct_approved_count"]
            and c["wrong_approved_count"] <= b["wrong_approved_count"]
            and c["missed_count"] <= b["missed_count"]
        }
    comparison["retained_decisions"] = [
        retained_prediction_parity(
            read_jsonl(output / f"pair-{i + 1}/baseline/responses.jsonl"),
            read_jsonl(output / f"pair-{i + 1}/candidate/responses.jsonl"),
        )
        for i in range(len(protocol["order"]))
    ]
    if args.candidate == "context040":
        comparison["accepted"] &= all(
            r["unchanged_retained_decisions"] for r in comparison["retained_decisions"]
        )
    comparison["interference"] = interference
    comparison["accepted"] &= not any(r["possible_interference"] for r in interference)
    comparison["reports"] = reports
    write_json(output / "comparison.json", comparison)
    print("COMPARISON", args.dataset, args.provider, comparison["accepted"], flush=True)
    if args.provider == "cuda":
        cpu = (
            work
            / ("confirmation" if args.candidate == "score040" else "confirmation-context")
            / args.dataset
            / "cpu"
            / "pair-1/candidate/responses.jsonl"
        )
        if cpu.exists():
            write_json(
                output / "cpu-cuda-parity.json",
                provider_parity(cpu, output / "pair-1/candidate/responses.jsonl"),
            )


if __name__ == "__main__":
    main()
